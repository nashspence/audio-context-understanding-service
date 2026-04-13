# audio-context-understanding-service

Minimal GPU-first Docker Compose stack exposing an HTTP API around `Qwen/Qwen2.5-Omni-7B` for audio understanding and captioning.

## What it does

- Accepts an uploaded audio file plus optional `metadata_json`
- Returns JSON with a short caption, detailed summary, structured scene/context tags, salient events, uncertainty notes, and optional per-span outputs
- Downloads model artifacts from the official Hugging Face repo on startup when needed
- Caches model artifacts in the `hf-cache` Docker volume
- Reports `503` on `/healthz` until the model is fully loaded and the API is ready

## Run

```bash
cp .env.example .env
docker compose up --build
```

## Request example

```bash
curl -fsS \
  -F file=@test.opus \
  -F 'metadata_json={
    "speaker_ids":["speaker_a","speaker_b"],
    "context_hints":["podcast","indoor"],
    "transcript_segments":[
      {"start_seconds":0.0,"end_seconds":3.1,"speaker_id":"speaker_a","text":"Hello and welcome back."}
    ],
    "sound_events":[
      {"label":"laughter","start_seconds":4.0,"end_seconds":4.8,"confidence":"medium"}
    ],
    "diarization_spans":[
      {"span_id":"intro","start_seconds":0.0,"end_seconds":5.0,"speaker_id":"speaker_a"}
    ]
  }' \
  http://localhost:8000/v1/analyze
```

## Smoke test

```bash
bash .devcontainer/smoke-test.sh
```
