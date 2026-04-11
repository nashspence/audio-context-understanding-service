#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT=/workspaces/audio-understanding-service
cd "${PROJECT_ROOT}"

for cmd in docker curl python3; do
  command -v "${cmd}" >/dev/null
done

set -a
# shellcheck disable=SC1091
source .env
set +a

API_BASE_URL="${API_BASE_URL:-http://host.docker.internal:${SERVICE_PORT:-8000}}"
export API_BASE_URL

cleanup() {
  docker compose down --remove-orphans >/dev/null 2>&1 || true
}

trap cleanup EXIT

docker compose down --remove-orphans >/dev/null 2>&1 || true
docker compose up --build -d

container_id="$(docker compose ps -q api)"
if [[ -z "${container_id}" ]]; then
  echo "API container did not start." >&2
  exit 1
fi

for _ in $(seq 1 180); do
  health="$(docker inspect --format '{{if .State.Health}}{{.State.Health.Status}}{{else}}none{{end}}' "${container_id}")"
  if [[ "${health}" == "healthy" ]]; then
    break
  fi
  sleep 10
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
assert payload["model_id"] == "Qwen/Qwen2.5-Omni-7B"
PY

curl -fsS \
  -F file=@test.opus \
  -F 'metadata_json={"speaker_ids":["speaker_a"],"context_hints":["meeting","indoor"],"diarization_spans":[{"span_id":"intro","start_seconds":0.0,"end_seconds":1.0,"speaker_id":"speaker_a"}]}' \
  "${API_BASE_URL}/v1/analyze" \
  > /tmp/qwen-omni-audio-response.json

python3 - <<'PY'
import json

with open("/tmp/qwen-omni-audio-response.json", "r", encoding="utf-8") as handle:
    payload = json.load(handle)

assert payload["model"]["id"] == "Qwen/Qwen2.5-Omni-7B"
assert payload["audio"]["filename"] == "test.opus"
assert payload["audio"]["duration_seconds"] > 0
assert isinstance(payload["short_caption"], str)
assert isinstance(payload["detailed_summary"], str)
assert isinstance(payload["scene_context_tags"], dict)
assert isinstance(payload["salient_events"], list)
assert isinstance(payload["uncertainty_notes"], list)
assert isinstance(payload["per_span_outputs"], list)
assert len(payload["per_span_outputs"]) >= 1
PY

docker compose down --remove-orphans
trap - EXIT
