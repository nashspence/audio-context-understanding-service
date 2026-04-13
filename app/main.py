from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import JSONResponse
from huggingface_hub import snapshot_download
from pydantic import BaseModel, Field, ValidationError
from transformers import Qwen2_5OmniProcessor, Qwen2_5OmniThinkerForConditionalGeneration


LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL, logging.INFO),
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)
logger = logging.getLogger("audio-context-understanding-service")


def env_int(name: str, default: int) -> int:
    return int(os.getenv(name, str(default)))


@dataclass(slots=True)
class Settings:
    model_id: str = os.getenv("MODEL_ID", "Qwen/Qwen2.5-Omni-7B")
    model_revision: str = os.getenv("MODEL_REVISION", "main")
    hf_home: str = os.getenv("HF_HOME", "/models/cache/huggingface")
    hf_token: str | None = os.getenv("HF_TOKEN") or None
    torch_dtype_name: str = os.getenv("TORCH_DTYPE", "bfloat16")
    device_map: str = os.getenv("DEVICE_MAP", "single_gpu")
    attn_implementation: str = os.getenv("ATTN_IMPLEMENTATION", "sdpa")
    gpu_memory_utilization: float = float(os.getenv("GPU_MEMORY_UTILIZATION", "0.98"))
    max_new_tokens: int = env_int("MAX_NEW_TOKENS", 384)
    max_new_tokens_per_span: int = env_int("MAX_NEW_TOKENS_PER_SPAN", 224)
    max_spans: int = env_int("MAX_SPANS", 8)
    normalize_sample_rate: int = env_int("NORMALIZE_SAMPLE_RATE", 16000)
    upload_dir: str = os.getenv("UPLOAD_DIR", "/tmp/audio-understanding")

    @property
    def torch_dtype(self) -> torch.dtype:
        mapping = {
            "float16": torch.float16,
            "fp16": torch.float16,
            "bfloat16": torch.bfloat16,
            "bf16": torch.bfloat16,
            "float32": torch.float32,
            "fp32": torch.float32,
            "auto": torch.bfloat16,
        }
        return mapping.get(self.torch_dtype_name.lower(), torch.bfloat16)


settings = Settings()


class SpanInput(BaseModel):
    span_id: str | None = None
    start_seconds: float = Field(ge=0)
    end_seconds: float = Field(gt=0)
    speaker_id: str | None = None
    transcript: str | None = None
    sound_events: list[str] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)


class TranscriptSegment(BaseModel):
    start_seconds: float | None = Field(default=None, ge=0)
    end_seconds: float | None = Field(default=None, ge=0)
    speaker_id: str | None = None
    text: str


class SoundEvent(BaseModel):
    label: str
    start_seconds: float | None = Field(default=None, ge=0)
    end_seconds: float | None = Field(default=None, ge=0)
    confidence: str | None = None


class RequestMetadata(BaseModel):
    language_hint: str | None = None
    context_hints: list[str] = Field(default_factory=list)
    speaker_ids: list[str] = Field(default_factory=list)
    transcript_text: str | None = None
    transcript_segments: list[TranscriptSegment] = Field(default_factory=list)
    sound_events: list[SoundEvent] = Field(default_factory=list)
    diarization_spans: list[SpanInput] = Field(default_factory=list)
    spans: list[SpanInput] = Field(default_factory=list)


class RuntimeState:
    def __init__(self) -> None:
        self.ready = False
        self.loading = False
        self.error: str | None = None
        self.model: Qwen2_5OmniThinkerForConditionalGeneration | None = None
        self.processor: Qwen2_5OmniProcessor | None = None
        self.model_path: str | None = None
        self.device: str = "uninitialized"
        self.gpu_name: str | None = None
        self.attn_implementation: str | None = None
        self.generation_lock = asyncio.Lock()
        self.load_task: asyncio.Task[None] | None = None


state = RuntimeState()


def compact_json(data: Any, max_chars: int = 5000) -> str:
    if not data:
        return "null"
    payload = json.dumps(data, ensure_ascii=False, separators=(",", ":"))
    if len(payload) <= max_chars:
        return payload
    return payload[: max_chars - 3] + "..."


def parse_metadata(metadata_json: str | None) -> RequestMetadata:
    if not metadata_json:
        return RequestMetadata()
    try:
        return RequestMetadata.model_validate_json(metadata_json)
    except ValidationError as exc:
        raise HTTPException(status_code=422, detail=json.loads(exc.json())) from exc


def ensure_gpu() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this service but no GPU is available.")


def run_command(command: list[str]) -> str:
    result = subprocess.run(command, check=True, capture_output=True, text=True)
    return result.stdout.strip()


def save_upload(upload: UploadFile, target_path: Path) -> None:
    with target_path.open("wb") as handle:
        shutil.copyfileobj(upload.file, handle)


def probe_duration_seconds(audio_path: Path) -> float:
    output = run_command(
        [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            str(audio_path),
        ]
    )
    return round(float(output), 3)


def normalize_audio(input_path: Path, output_path: Path) -> None:
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-i",
            str(input_path),
            "-vn",
            "-ac",
            "1",
            "-ar",
            str(settings.normalize_sample_rate),
            "-sample_fmt",
            "s16",
            str(output_path),
        ],
        check=True,
        capture_output=True,
    )


def extract_audio_span(input_path: Path, output_path: Path, start_seconds: float, end_seconds: float) -> None:
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-ss",
            f"{start_seconds:.3f}",
            "-to",
            f"{end_seconds:.3f}",
            "-i",
            str(input_path),
            "-ac",
            "1",
            "-ar",
            str(settings.normalize_sample_rate),
            "-sample_fmt",
            "s16",
            str(output_path),
        ],
        check=True,
        capture_output=True,
    )


def build_prompt(
    *,
    duration_seconds: float,
    metadata: RequestMetadata,
    span: SpanInput | None = None,
) -> str:
    prompt_parts = [
        "Analyze the audio carefully and return only valid JSON.",
        "Use cautious language for anything uncertain or weakly evidenced.",
        "Treat supplied metadata as helpful context, not unquestionable ground truth.",
        "If metadata conflicts with the audio, mention that in uncertainty_notes.",
        f"Audio duration seconds: {duration_seconds:.3f}.",
    ]
    if metadata.language_hint:
        prompt_parts.append(f"Language hint: {metadata.language_hint}.")
    if metadata.context_hints:
        prompt_parts.append(f"Context hints: {', '.join(metadata.context_hints)}.")
    if metadata.speaker_ids:
        prompt_parts.append(f"Known speaker ids: {', '.join(metadata.speaker_ids)}.")
    if metadata.transcript_text:
        prompt_parts.append(f"Transcript text: {metadata.transcript_text[:2000]}")
    if metadata.transcript_segments:
        prompt_parts.append(
            f"Transcript segments JSON: {compact_json([segment.model_dump(exclude_none=True) for segment in metadata.transcript_segments], 3000)}"
        )
    if metadata.sound_events:
        prompt_parts.append(
            f"Detected sound events JSON: {compact_json([event.model_dump(exclude_none=True) for event in metadata.sound_events], 2000)}"
        )
    if span is not None:
        prompt_parts.append(
            f"Focus on this span only: {compact_json(span.model_dump(exclude_none=True), 800)}"
        )

    schema = {
        "short_caption": "One sentence, under 25 words.",
        "detailed_summary": "Two to five sentences.",
        "scene_context_tags": {
            "environment": ["lowercase tags"],
            "activities": ["lowercase tags"],
            "sound_types": ["lowercase tags"],
            "speaker_ids": ["speaker ids if any"],
            "qualities": ["tone or recording traits"],
        },
        "salient_events": [
            {
                "label": "short event label",
                "start_seconds": 0.0,
                "end_seconds": 0.0,
                "speaker_id": "optional speaker id",
                "details": "brief explanation",
                "confidence": "high|medium|low",
            }
        ],
        "uncertainty_notes": ["brief notes"],
    }
    prompt_parts.append(f"Return JSON matching this shape: {json.dumps(schema, ensure_ascii=False)}")
    return "\n".join(prompt_parts)


def extract_json_payload(text: str) -> dict[str, Any]:
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
        cleaned = re.sub(r"\s*```$", "", cleaned)
    start = cleaned.find("{")
    if start == -1:
        raise ValueError("Model output did not contain a JSON object.")
    cleaned = cleaned[start:]

    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass

    decoder = json.JSONDecoder()
    try:
        parsed, _ = decoder.raw_decode(cleaned)
        return parsed
    except json.JSONDecodeError:
        pass

    repaired = repair_json_fragment(cleaned)
    return json.loads(repaired)


def repair_json_fragment(text: str) -> str:
    stack: list[str] = []
    in_string = False
    escaped = False
    buffer: list[str] = []

    for char in text:
        buffer.append(char)
        if escaped:
            escaped = False
            continue
        if char == "\\":
            escaped = True
            continue
        if char == '"':
            in_string = not in_string
            continue
        if in_string:
            continue
        if char in "{[":
            stack.append("}" if char == "{" else "]")
        elif char in "}]":
            if stack and stack[-1] == char:
                stack.pop()
            else:
                break

    repaired = "".join(buffer).rstrip()
    if in_string:
        repaired += '"'
    if stack:
        repaired += "".join(reversed(stack))
    repaired = re.sub(r",(\s*[}\]])", r"\1", repaired)
    return repaired


def normalize_analysis_payload(
    payload: dict[str, Any],
    *,
    duration_seconds: float,
    speaker_ids: list[str],
) -> dict[str, Any]:
    scene_context = payload.get("scene_context_tags") or {}
    if not isinstance(scene_context, dict):
        scene_context = {}
    normalized = {
        "short_caption": str(payload.get("short_caption") or "").strip(),
        "detailed_summary": str(payload.get("detailed_summary") or "").strip(),
        "scene_context_tags": {
            "environment": list(scene_context.get("environment") or []),
            "activities": list(scene_context.get("activities") or []),
            "sound_types": list(scene_context.get("sound_types") or []),
            "speaker_ids": list(scene_context.get("speaker_ids") or speaker_ids),
            "qualities": list(scene_context.get("qualities") or []),
        },
        "salient_events": [],
        "uncertainty_notes": [str(note).strip() for note in payload.get("uncertainty_notes") or [] if str(note).strip()],
    }
    for event in payload.get("salient_events") or []:
        if not isinstance(event, dict):
            continue
        start = event.get("start_seconds")
        end = event.get("end_seconds")
        try:
            start_value = max(0.0, float(start)) if start is not None else None
            end_value = min(duration_seconds, float(end)) if end is not None else None
        except (TypeError, ValueError):
            start_value = None
            end_value = None
        normalized["salient_events"].append(
            {
                "label": str(event.get("label") or "").strip(),
                "start_seconds": start_value,
                "end_seconds": end_value,
                "speaker_id": (str(event.get("speaker_id")).strip() or None) if event.get("speaker_id") else None,
                "details": str(event.get("details") or "").strip(),
                "confidence": str(event.get("confidence") or "low").strip().lower(),
            }
        )
    return normalized


def generate_json_analysis(
    *,
    audio_path: Path,
    duration_seconds: float,
    metadata: RequestMetadata,
    max_new_tokens: int,
    span: SpanInput | None = None,
) -> dict[str, Any]:
    if state.model is None or state.processor is None:
        raise RuntimeError("Model runtime is not initialized.")

    conversation = [
        {
            "role": "user",
            "content": [
                {"type": "audio", "path": str(audio_path)},
                {"type": "text", "text": build_prompt(duration_seconds=duration_seconds, metadata=metadata, span=span)},
            ],
        },
    ]

    inputs = state.processor.apply_chat_template(
        conversation,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
        padding=True,
    ).to(state.model.device)

    with torch.inference_mode():
        generated = state.model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            use_cache=True,
            pad_token_id=state.processor.tokenizer.eos_token_id,
        )
    generated_ids = generated[:, inputs.input_ids.shape[1] :]
    raw_text = state.processor.batch_decode(
        generated_ids,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )[0]
    logger.info("Model response preview: %s", raw_text[:400].replace("\n", " "))
    try:
        parsed = extract_json_payload(raw_text)
    except Exception:
        logger.exception("Failed to parse model JSON output.")
        parsed = {
            "short_caption": "",
            "detailed_summary": raw_text.strip(),
            "scene_context_tags": {},
            "salient_events": [],
            "uncertainty_notes": [
                "The model returned non-JSON output, so this response includes a raw-text fallback."
            ],
        }
    speaker_ids = metadata.speaker_ids
    if span and span.speaker_id and span.speaker_id not in speaker_ids:
        speaker_ids = [*speaker_ids, span.speaker_id]
    return normalize_analysis_payload(parsed, duration_seconds=duration_seconds, speaker_ids=speaker_ids)


def load_runtime() -> None:
    logger.info("Starting runtime load for %s", settings.model_id)
    ensure_gpu()
    Path(settings.hf_home).mkdir(parents=True, exist_ok=True)
    model_path = snapshot_download(
        repo_id=settings.model_id,
        revision=settings.model_revision,
        cache_dir=settings.hf_home,
        token=settings.hf_token,
    )
    logger.info("Model snapshot available at %s", model_path)
    state.model_path = model_path

    load_kwargs: dict[str, Any] = {
        "dtype": settings.torch_dtype,
        "low_cpu_mem_usage": True,
    }
    if settings.device_map.lower() in {"single_gpu", "cuda", "cuda:0"}:
        load_kwargs["device_map"] = {"": 0}
    else:
        load_kwargs["device_map"] = settings.device_map
        if str(load_kwargs["device_map"]).lower() == "auto":
            total_memory = torch.cuda.get_device_properties(0).total_memory
            usable_memory = max(int(total_memory * settings.gpu_memory_utilization), total_memory - (512 * 1024 * 1024))
            load_kwargs["max_memory"] = {0: f"{usable_memory // (1024 ** 3)}GiB"}
    requested_attn = settings.attn_implementation.strip()
    try:
        state.model = Qwen2_5OmniThinkerForConditionalGeneration.from_pretrained(
            model_path,
            attn_implementation=requested_attn,
            **load_kwargs,
        )
        state.attn_implementation = requested_attn
    except Exception:
        logger.exception("Falling back from attn_implementation=%s to sdpa", requested_attn)
        state.model = Qwen2_5OmniThinkerForConditionalGeneration.from_pretrained(
            model_path,
            attn_implementation="sdpa",
            **load_kwargs,
        )
        state.attn_implementation = "sdpa"

    state.processor = Qwen2_5OmniProcessor.from_pretrained(model_path)
    state.device = str(state.model.device)
    state.gpu_name = torch.cuda.get_device_name(0)
    state.ready = True
    state.error = None
    logger.info(
        "Runtime ready model=%s device=%s gpu=%s attn=%s",
        settings.model_id,
        state.device,
        state.gpu_name,
        state.attn_implementation,
    )


async def ensure_runtime_started() -> None:
    if state.load_task is None:
        state.loading = True

        async def runner() -> None:
            try:
                await asyncio.to_thread(load_runtime)
            except Exception as exc:
                logger.exception("Runtime failed to initialize.")
                state.error = str(exc)
                state.ready = False
            finally:
                state.loading = False

        state.load_task = asyncio.create_task(runner())


app = FastAPI(
    title="Audio Context Understanding Service",
    version="0.1.0",
    docs_url="/docs",
)


@app.on_event("startup")
async def startup_event() -> None:
    await ensure_runtime_started()


@app.get("/")
async def root() -> dict[str, str]:
    return {"service": "audio-context-understanding-service", "model": settings.model_id}


@app.get("/healthz")
async def healthcheck() -> JSONResponse:
    status_code = 200 if state.ready else 503
    payload = {
        "ready": state.ready,
        "loading": state.loading,
        "model_id": settings.model_id,
        "model_revision": settings.model_revision,
        "model_path": state.model_path,
        "device": state.device,
        "gpu_name": state.gpu_name,
        "attn_implementation": state.attn_implementation,
        "error": state.error,
    }
    return JSONResponse(status_code=status_code, content=payload)


@app.post("/v1/analyze")
async def analyze_audio(
    file: UploadFile = File(...),
    metadata_json: str | None = Form(default=None),
) -> JSONResponse:
    await ensure_runtime_started()
    if not state.ready:
        raise HTTPException(status_code=503, detail="Model is still loading.")

    metadata = parse_metadata(metadata_json)
    async with state.generation_lock:
        Path(settings.upload_dir).mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix="audio-understanding-", dir=settings.upload_dir) as tmpdir:
            tmpdir_path = Path(tmpdir)
            upload_name = Path(file.filename or "audio.bin").name
            raw_path = tmpdir_path / upload_name
            normalized_path = tmpdir_path / "normalized.wav"

            await asyncio.to_thread(save_upload, file, raw_path)
            await asyncio.to_thread(normalize_audio, raw_path, normalized_path)
            duration_seconds = await asyncio.to_thread(probe_duration_seconds, normalized_path)

            analysis = await asyncio.to_thread(
                generate_json_analysis,
                audio_path=normalized_path,
                duration_seconds=duration_seconds,
                metadata=metadata,
                max_new_tokens=settings.max_new_tokens,
            )

            span_requests = metadata.spans or metadata.diarization_spans
            span_outputs: list[dict[str, Any]] = []
            for span in span_requests[: settings.max_spans]:
                start_seconds = max(0.0, span.start_seconds)
                end_seconds = min(duration_seconds, span.end_seconds)
                if end_seconds <= start_seconds:
                    continue
                span_path = tmpdir_path / f"span-{len(span_outputs)}.wav"
                await asyncio.to_thread(
                    extract_audio_span,
                    normalized_path,
                    span_path,
                    start_seconds,
                    end_seconds,
                )
                span_duration = max(0.001, end_seconds - start_seconds)
                span_analysis = await asyncio.to_thread(
                    generate_json_analysis,
                    audio_path=span_path,
                    duration_seconds=span_duration,
                    metadata=metadata,
                    max_new_tokens=settings.max_new_tokens_per_span,
                    span=span,
                )
                span_outputs.append(
                    {
                        "span_id": span.span_id or f"span-{len(span_outputs) + 1}",
                        "start_seconds": start_seconds,
                        "end_seconds": end_seconds,
                        "speaker_id": span.speaker_id,
                        "short_caption": span_analysis["short_caption"],
                        "summary": span_analysis["detailed_summary"],
                        "salient_events": span_analysis["salient_events"],
                        "uncertainty_notes": span_analysis["uncertainty_notes"],
                    }
                )

            response = {
                "model": {
                    "id": settings.model_id,
                    "revision": settings.model_revision,
                    "mode": "text-only thinker",
                    "device": state.device,
                    "gpu_name": state.gpu_name,
                    "dtype": settings.torch_dtype_name,
                    "attn_implementation": state.attn_implementation,
                },
                "audio": {
                    "filename": upload_name,
                    "duration_seconds": duration_seconds,
                },
                "short_caption": analysis["short_caption"],
                "detailed_summary": analysis["detailed_summary"],
                "scene_context_tags": analysis["scene_context_tags"],
                "salient_events": analysis["salient_events"],
                "uncertainty_notes": analysis["uncertainty_notes"],
                "per_span_outputs": span_outputs,
            }
            return JSONResponse(content=response)
