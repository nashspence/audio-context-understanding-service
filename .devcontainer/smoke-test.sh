#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT=/workspaces/atst-sed-service

cd "${PROJECT_ROOT}"

for cmd in docker curl python3 git nvcc nvidia-smi; do
  command -v "${cmd}" >/dev/null
done

python3 --version
nvcc --version | tail -n 1
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
docker compose version
test -S /var/run/docker.sock
docker ps >/dev/null
docker run --rm alpine:3.22 true
docker run --rm --gpus all nvidia/cuda:12.8.1-base-ubuntu24.04 \
  nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader

if [[ ! -f .env ]]; then
  cp .env.example .env
fi

set -a
# shellcheck disable=SC1091
source .env
set +a
API_BASE_URL="${API_BASE_URL:-http://host.docker.internal:${SERVICE_PORT}}"
export API_BASE_URL

cleanup() {
  docker compose down -v --remove-orphans
}

trap cleanup EXIT

docker compose down -v --remove-orphans >/dev/null 2>&1 || true
docker compose up --build -d

container_id="$(docker compose ps -q api)"
if [[ -z "${container_id}" ]]; then
  echo "API container did not start." >&2
  exit 1
fi

for _ in $(seq 1 240); do
  health="$(docker inspect --format '{{if .State.Health}}{{.State.Health.Status}}{{else}}none{{end}}' "${container_id}")"
  if [[ "${health}" == "healthy" ]]; then
    break
  fi
  sleep 5
done

health="$(docker inspect --format '{{if .State.Health}}{{.State.Health.Status}}{{else}}none{{end}}' "${container_id}")"
if [[ "${health}" != "healthy" ]]; then
  docker compose logs api >&2
  echo "API container did not become healthy." >&2
  exit 1
fi

python3 - <<'PY'
import json
import os
import urllib.request

with urllib.request.urlopen(f"{os.environ['API_BASE_URL']}/healthz") as response:
    payload = json.load(response)

assert payload["ready"] is True
assert payload["device"].startswith("cuda")
assert payload["checkpoint"]["base"]["present"] is True
assert payload["checkpoint"]["stage2"]["present"] is True
PY

curl -fsS \
  -F file=@test.opus \
  "${API_BASE_URL}/v1/detect" \
  > /tmp/atst-sed-detect.json

python3 - <<'PY'
import json

with open("/tmp/atst-sed-detect.json", "r", encoding="utf-8") as handle:
    payload = json.load(handle)

assert payload["model"] == "ATST-SED"
assert payload["audio"]["filename"] == "test.opus"
assert payload["audio"]["duration_seconds"] > 0
assert payload["frame_resolution_seconds"] > 0
assert isinstance(payload["segments"], list)
assert isinstance(payload["detected_labels"], list)
for segment in payload["segments"]:
    assert segment["label"] != "Speech"
    assert segment["start_seconds"] >= 0
    assert segment["end_seconds"] >= segment["start_seconds"]
PY

docker compose down -v --remove-orphans
trap - EXIT
