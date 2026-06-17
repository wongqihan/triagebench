#!/usr/bin/env bash
# Reproduce the public TriageBench leaderboard in one shot: run all three probes
# (gender, language, SES), then reduce them to the TriageGap matrix. Every API
# call is cached, so a repeat run costs nothing and a crashed run resumes for free.
#
# Set your keys first:
#   export ANTHROPIC_API_KEY=...  OPENAI_API_KEY=...  GEMINI_API_KEY=...  DEEPSEEK_API_KEY=...
set -euo pipefail
cd "$(dirname "$0")/.."

for c in configs/leaderboard/*.yaml; do
  echo ">> running $c"
  triagebench run "$c"
done

triagebench leaderboard configs/leaderboard/*.yaml -o leaderboard/leaderboard.json
echo ">> board written to leaderboard/leaderboard.json and leaderboard/leaderboard.md"
