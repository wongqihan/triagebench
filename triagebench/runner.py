"""Concurrent trial runner with on-disk cache, retries, and JSONL logging.

The cache is the thing that makes iteration free: a re-run with the same
(model, messages, temperature) reads from disk instead of re-paying the API.
Every result is written as one JSON line to results/<experiment>/<timestamp>.jsonl.
"""

from __future__ import annotations

import hashlib
import json
import os
import random
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import List, Optional

from . import models
from .trial import Trial, TrialResult


def _cache_key(
    model_id: str, messages, system, temperature: float, run_index: int
) -> str:
    """Key includes run_index: repeats within a cell are independent samples,
    not replays of one response. Without it, n_runs=30 at temperature>0 would
    silently produce 1 API sample photocopied 30 times."""
    blob = json.dumps(
        {
            "m": model_id,
            "msgs": messages,
            "sys": system,
            "t": temperature,
            "r": run_index,
        },
        sort_keys=True,
    )
    return hashlib.sha256(blob.encode()).hexdigest()


class Cache:
    """Content-addressed JSON cache, one file per key under cache_dir."""

    def __init__(self, cache_dir: Path):
        self.dir = Path(cache_dir)
        self.dir.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()

    def get(self, key: str) -> Optional[dict]:
        path = self.dir / f"{key}.json"
        if path.exists():
            with path.open() as f:
                return json.load(f)
        return None

    def put(self, key: str, value: dict) -> None:
        path = self.dir / f"{key}.json"
        with self._lock:
            tmp = path.with_suffix(".tmp")
            with tmp.open("w") as f:
                json.dump(value, f)
            tmp.replace(path)


class Runner:
    def __init__(
        self,
        experiment: str,
        results_root: str = "results",
        cache_dir: str = ".cache",
        concurrency: int = 4,
        max_retries: int = 5,
        use_cache: bool = True,
        label: Optional[str] = None,
    ):
        self.experiment = experiment
        self.concurrency = concurrency
        self.max_retries = max_retries
        self.use_cache = use_cache
        self.cache = Cache(Path(cache_dir))
        # Each probe config gets its own results subdirectory (label), so the
        # leaderboard can resolve a config to exactly its own runs rather than
        # whatever shared-experiment file happened to run last.
        self.results_dir = Path(results_root) / experiment / (label or "_default")
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self._write_lock = threading.Lock()

    def _call_with_retry(self, trial: Trial) -> models.Completion:
        delay = 1.0
        last_exc: Optional[Exception] = None
        for attempt in range(self.max_retries):
            try:
                return models.complete(
                    messages=trial.messages,
                    system=trial.system,
                    model_id=trial.model_id,
                    temperature=trial.temperature,
                )
            except Exception as exc:  # noqa: BLE001 - want broad retry on API errors
                last_exc = exc
                if not _is_retryable(exc) or attempt == self.max_retries - 1:
                    raise
                sleep = delay + random.uniform(0, delay)  # full jitter
                time.sleep(sleep)
                delay = min(delay * 2, 60)
        raise last_exc  # pragma: no cover

    def _run_one(self, trial: Trial) -> TrialResult:
        key = _cache_key(
            trial.model_id,
            trial.messages,
            trial.system,
            trial.temperature,
            trial.run_index,
        )
        cached = self.cache.get(key) if self.use_cache else None
        if cached is not None:
            text = cached["text"]
            reasoning = cached.get("reasoning")
            latency = cached.get("latency", 0.0)
            usage = cached.get("token_usage", {})
            was_cached = True
        else:
            comp = self._call_with_retry(trial)
            text, reasoning = comp.text, comp.reasoning
            latency, usage = comp.latency, comp.token_usage
            if self.use_cache:
                self.cache.put(
                    key,
                    {
                        "text": text,
                        "reasoning": reasoning,
                        "latency": latency,
                        "token_usage": usage,
                    },
                )
            was_cached = False

        try:
            parsed = trial.parse(text)
            error = None
        except Exception as exc:  # noqa: BLE001
            parsed, error = {}, f"parse_error: {exc}"

        return TrialResult(
            trial_id=trial.id,
            cell=trial.cell,
            parsed=parsed,
            raw_text=text,
            reasoning=reasoning,
            model_id=trial.model_id,
            latency=latency,
            token_usage=usage,
            cached=was_cached,
            error=error,
            run_index=trial.run_index,
            meta=trial.meta,
        )

    def run(self, trials: List[Trial], verbose: bool = True) -> List[TrialResult]:
        """Run all trials concurrently, streaming each to JSONL as it finishes."""
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_path = self.results_dir / f"{ts}.jsonl"
        results: List[TrialResult] = []
        done = 0
        total = len(trials)

        with out_path.open("w") as fh, ThreadPoolExecutor(
            max_workers=self.concurrency
        ) as pool:
            futures = {pool.submit(self._run_one, t): t for t in trials}
            for fut in as_completed(futures):
                trial = futures[fut]
                try:
                    res = fut.result()
                except Exception as exc:  # noqa: BLE001 - record, don't crash the batch
                    res = TrialResult(
                        trial_id=trial.id,
                        cell=trial.cell,
                        parsed={},
                        raw_text="",
                        model_id=trial.model_id,
                        error=f"call_error: {exc}",
                        run_index=trial.run_index,
                        meta=trial.meta,
                    )
                results.append(res)
                with self._write_lock:
                    fh.write(json.dumps(res.to_json()) + "\n")
                    fh.flush()
                done += 1
                if verbose:
                    tag = "cache" if res.cached else "api"
                    err = f" ERROR {res.error}" if res.error else ""
                    print(f"  [{done:>4}/{total}] {res.model_id:<14} ({tag}){err}")

        if verbose:
            print(f"\nWrote {len(results)} results -> {out_path}")
        return results, out_path


def _is_retryable(exc: Exception) -> bool:
    """Retry on rate limits / transient server errors, not on auth/validation."""
    status = getattr(exc, "status_code", None) or getattr(exc, "code", None)
    if status in (429, 500, 502, 503, 504, 529):
        return True
    name = type(exc).__name__.lower()
    if any(k in name for k in ("ratelimit", "timeout", "connection", "apistatus")):
        return True
    msg = str(exc).lower()
    return any(k in msg for k in ("rate limit", "overloaded", "timeout", "503", "429"))


def load_results(jsonl_path: str) -> List[TrialResult]:
    """Re-load a JSONL run for replot/report without re-calling APIs."""
    results = []
    with open(jsonl_path) as f:
        for line in f:
            line = line.strip()
            if line:
                results.append(TrialResult.from_json(json.loads(line)))
    return results


def latest_jsonl(
    results_root: str, experiment: str, label: Optional[str] = None
) -> Optional[Path]:
    """Most recent JSONL for a probe. With `label`, resolves to that config's
    own subdirectory; without, searches the whole experiment tree (legacy)."""
    base = Path(results_root) / experiment
    d = base / label if label else base
    files = sorted(d.glob("*.jsonl")) if d.exists() else []
    if not files and label is None:
        files = sorted(base.glob("**/*.jsonl"))
    return files[-1] if files else None
