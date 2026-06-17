"""Provider-agnostic model adapter.

One `complete()` interface dispatches to Anthropic, OpenAI, or Google based on a
model registry. Returns a structured `Completion` with text, raw response,
latency, token usage, and any reasoning/thinking content the provider exposes.

API keys are read from environment variables. Official SDKs are imported lazily
so you only need the SDK for the providers you actually use.
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class Completion:
    text: str
    raw_response: Any
    latency: float
    token_usage: Dict[str, int] = field(default_factory=dict)
    reasoning: Optional[str] = None
    model_id: str = ""


@dataclass
class ModelSpec:
    """An entry in the model registry.

    provider: one of "anthropic", "openai", "google".
    api_model: the provider's actual model string (e.g. "claude-opus-4-8").
    max_tokens: default output cap.
    extra: provider-specific knobs (e.g. {"thinking": {"budget_tokens": 4000}}).
    """

    provider: str
    api_model: str
    max_tokens: int = 2048
    extra: Dict[str, Any] = field(default_factory=dict)
    # Newest-generation models (e.g. Opus 4.8/4.7, Fable 5) removed the sampling
    # parameters: sending `temperature` returns a 400. Set False for those so the
    # adapter omits it. Defaults True for every model that still accepts it.
    supports_temperature: bool = True
    # base_url + api_key_env let the openai adapter target any OpenAI-compatible
    # endpoint (DeepSeek, Together, Fireworks, local vLLM). Leave unset for the
    # vendor's default endpoint and standard key env var.
    base_url: Optional[str] = None
    api_key_env: Optional[str] = None


def _load_registry() -> Dict[str, ModelSpec]:
    """Registry comes from models.yaml (repo root, cwd, or $CETERISBENCH_MODELS).
    Adding a model on release day is one YAML entry — no code changes."""
    import pathlib

    import yaml

    candidates = []
    if os.environ.get("CETERISBENCH_MODELS"):
        candidates.append(pathlib.Path(os.environ["CETERISBENCH_MODELS"]))
    candidates.append(pathlib.Path(__file__).resolve().parent.parent / "models.yaml")
    candidates.append(pathlib.Path.cwd() / "models.yaml")
    for path in candidates:
        if path.exists():
            raw = yaml.safe_load(path.read_text()) or {}
            return {
                alias: ModelSpec(
                    provider=spec["provider"],
                    api_model=spec["api_model"],
                    max_tokens=spec.get("max_tokens", 2048),
                    extra=spec.get("extra", {}),
                    supports_temperature=spec.get("supports_temperature", True),
                    base_url=spec.get("base_url"),
                    api_key_env=spec.get("api_key_env"),
                )
                for alias, spec in raw.items()
            }
    # Fallback so the library works without a models.yaml.
    return {
        "claude": ModelSpec(
            "anthropic", "claude-opus-4-8", max_tokens=2048, supports_temperature=False
        ),
        "gpt": ModelSpec("openai", "gpt-5.4-mini", max_tokens=2048),
        "gemini": ModelSpec("google", "gemini-3.5-flash", max_tokens=2048),
    }


REGISTRY: Dict[str, ModelSpec] = _load_registry()


def resolve(model_id: str) -> ModelSpec:
    if model_id in REGISTRY:
        return REGISTRY[model_id]
    raise KeyError(
        f"Unknown model_id {model_id!r}. Known: {sorted(REGISTRY)}. "
        "Add it to models.yaml."
    )


def complete(
    messages: List[Dict[str, str]],
    system: Optional[str],
    model_id: str,
    temperature: float = 0.7,
    max_tokens: Optional[int] = None,
) -> Completion:
    """Dispatch a chat completion to the right provider and normalize the result."""
    spec = resolve(model_id)
    cap = max_tokens or spec.max_tokens
    start = time.time()

    if spec.provider == "anthropic":
        comp = _anthropic(messages, system, spec, temperature, cap)
    elif spec.provider == "openai":
        comp = _openai(messages, system, spec, temperature, cap)
    elif spec.provider == "google":
        comp = _google(messages, system, spec, temperature, cap)
    else:
        raise ValueError(f"Unknown provider {spec.provider!r}")

    comp.latency = time.time() - start
    comp.model_id = model_id
    return comp


# --- provider implementations -------------------------------------------------


# Hard per-request timeout. Without it, a dropped network connection leaves the
# SDK blocked on a dead socket and every runner thread hangs indefinitely.
REQUEST_TIMEOUT_S = 120.0


def _anthropic(messages, system, spec: ModelSpec, temperature, cap) -> Completion:
    import anthropic  # official SDK

    client = anthropic.Anthropic(
        api_key=os.environ["ANTHROPIC_API_KEY"], timeout=REQUEST_TIMEOUT_S
    )
    kwargs: Dict[str, Any] = dict(
        model=spec.api_model,
        max_tokens=cap,
        messages=messages,
    )
    if spec.supports_temperature:
        kwargs["temperature"] = temperature
    if system:
        kwargs["system"] = system
    kwargs.update(spec.extra)  # e.g. extended thinking config

    resp = client.messages.create(**kwargs)

    text_parts, reasoning_parts = [], []
    for block in resp.content:
        btype = getattr(block, "type", None)
        if btype == "text":
            text_parts.append(block.text)
        elif btype == "thinking":
            reasoning_parts.append(getattr(block, "thinking", ""))
    usage = {
        "input_tokens": resp.usage.input_tokens,
        "output_tokens": resp.usage.output_tokens,
    }
    return Completion(
        text="".join(text_parts),
        raw_response=resp.model_dump() if hasattr(resp, "model_dump") else str(resp),
        latency=0.0,
        token_usage=usage,
        reasoning="\n".join(reasoning_parts) or None,
    )


def _openai(messages, system, spec: ModelSpec, temperature, cap) -> Completion:
    from openai import OpenAI  # official SDK

    client = OpenAI(
        api_key=os.environ[spec.api_key_env or "OPENAI_API_KEY"],
        base_url=spec.base_url,  # None -> default OpenAI endpoint
        timeout=REQUEST_TIMEOUT_S,
    )
    full_messages = ([{"role": "system", "content": system}] if system else []) + messages
    oai_kwargs: Dict[str, Any] = dict(
        model=spec.api_model,
        messages=full_messages,
        max_completion_tokens=cap,
        **spec.extra,
    )
    if spec.supports_temperature:
        oai_kwargs["temperature"] = temperature
    resp = client.chat.completions.create(**oai_kwargs)
    choice = resp.choices[0].message
    usage = {}
    if resp.usage:
        usage = {
            "input_tokens": resp.usage.prompt_tokens,
            "output_tokens": resp.usage.completion_tokens,
        }
    # Some OpenAI reasoning models expose a separate reasoning field.
    reasoning = getattr(choice, "reasoning", None)
    return Completion(
        text=choice.content or "",
        raw_response=resp.model_dump() if hasattr(resp, "model_dump") else str(resp),
        latency=0.0,
        token_usage=usage,
        reasoning=reasoning,
    )


def _google(messages, system, spec: ModelSpec, temperature, cap) -> Completion:
    from google import genai  # official SDK (google-genai)
    from google.genai import types

    client = genai.Client(
        api_key=os.environ["GEMINI_API_KEY"],
        http_options=types.HttpOptions(timeout=int(REQUEST_TIMEOUT_S * 1000)),
    )
    # Flatten chat messages into Gemini's contents format.
    contents = []
    for m in messages:
        role = "model" if m["role"] == "assistant" else "user"
        contents.append({"role": role, "parts": [{"text": m["content"]}]})

    config = types.GenerateContentConfig(
        temperature=temperature if spec.supports_temperature else None,
        max_output_tokens=cap,
        system_instruction=system or None,
    )
    resp = client.models.generate_content(
        model=spec.api_model, contents=contents, config=config
    )
    usage = {}
    if getattr(resp, "usage_metadata", None):
        usage = {
            "input_tokens": resp.usage_metadata.prompt_token_count,
            "output_tokens": resp.usage_metadata.candidates_token_count or 0,
        }
    return Completion(
        text=resp.text or "",
        raw_response=resp.model_dump() if hasattr(resp, "model_dump") else str(resp),
        latency=0.0,
        token_usage=usage,
        reasoning=None,
    )
