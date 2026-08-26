#!/usr/bin/env bash
# Modified by Perceptron AI, Inc.; derived from OpenAI Harmony.
set -euo pipefail

cargo fmt --all -- --check
cargo clippy --locked --all-targets --all-features -- -D warnings
cargo test --locked --all-targets --all-features
cmp SOURCE_MANIFEST.json python/mharmony/SOURCE_MANIFEST.json
cmp MODIFICATIONS.md python/mharmony/MODIFICATIONS.md
cmp THIRD_PARTY_LICENSES.md python/mharmony/THIRD_PARTY_LICENSES.md
python3 scripts/generate_third_party_licenses.py --check
./test_python.sh
