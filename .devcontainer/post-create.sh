#!/usr/bin/env bash
set -euo pipefail

cd /workspaces/audio-understanding-service

if [[ ! -f .env && -f .env.example ]]; then
  cp .env.example .env
fi

if [[ -f requirements-dev.txt ]]; then
  uv pip install --system -r requirements-dev.txt
fi
