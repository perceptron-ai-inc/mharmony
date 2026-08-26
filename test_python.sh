#!/usr/bin/env bash
# Modified by Perceptron AI, Inc.; derived from OpenAI Harmony.
set -euo pipefail

maturin develop --release --locked --extras qwen35,test
pytest "$@"
