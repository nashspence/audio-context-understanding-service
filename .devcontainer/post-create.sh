#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$(cd "${SCRIPT_DIR}/.." && pwd)"

if [[ ! -f .env && -f .env.example ]]; then
  cp .env.example .env
fi

if [[ -f requirements-dev.txt ]]; then
  uv pip install --system -r requirements-dev.txt
fi
