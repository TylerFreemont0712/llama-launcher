#!/usr/bin/env python3
"""
whisper_server.py
FastAPI server wrapping faster-whisper for high-accuracy Japanese
transcription. Receives audio files via POST /transcribe and writes
the resulting transcript to disk.

Run directly:
    python whisper_server.py --host 127.0.0.1 --port 8000

All accuracy-relevant knobs are CLI flags so the GUI can drive them.
"""

from __future__ import annotations

import argparse
import os
import sys
import tempfile
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path

import uvicorn
from fastapi import FastAPI, File, HTTPException, UploadFile
from faster_whisper import WhisperModel


# ─── App factory ──────────────────────────────────────────────────────────────

def create_app(args: argparse.Namespace) -> FastAPI:
    state: dict = {"model": None}
    out_dir = Path(os.path.expanduser(args.output_dir)).resolve()

    @asynccontextmanager
    async def lifespan(_app: FastAPI):
        out_dir.mkdir(parents=True, exist_ok=True)
        print(
            f"[whisper] Loading model={args.model} device={args.device} "
            f"compute_type={args.compute_type} ...",
            flush=True,
        )
        state["model"] = WhisperModel(
            args.model,
            device=args.device,
            compute_type=args.compute_type,
        )
        print(f"[whisper] Ready. Output dir: {out_dir}", flush=True)
        yield
        print("[whisper] Shutting down.", flush=True)

    app = FastAPI(
        title="Whisper Transcription Server",
        version="1.0",
        lifespan=lifespan,
    )

    @app.get("/health")
    async def health():
        return {
            "status": "ok",
            "model": args.model,
            "device": args.device,
            "compute_type": args.compute_type,
            "language": args.language,
            "loaded": state["model"] is not None,
            "output_dir": str(out_dir),
        }

    @app.post("/transcribe")
    async def transcribe(file: UploadFile = File(...)):
        model = state["model"]
        if model is None:
            raise HTTPException(status_code=503, detail="Model not loaded.")

        # Persist upload to a temp file because faster-whisper wants a path.
        original_name = file.filename or "audio"
        suffix = Path(original_name).suffix or ".bin"
        fd, temp_path = tempfile.mkstemp(suffix=suffix, prefix="whisper_")
        try:
            with os.fdopen(fd, "wb") as f:
                f.write(await file.read())

            language = None if args.language.lower() in ("auto", "") else args.language

            # Accuracy-tuned settings — patience/beam_size dominate cost.
            # Temperature is a fallback ladder: greedy first, only escalates
            # when the result fails compression_ratio/log_prob sanity checks,
            # so this gives us hallucination resilience without paying the
            # full cost on clean audio.
            segments, info = model.transcribe(
                temp_path,
                language=language,
                beam_size=args.beam_size,
                best_of=args.best_of,
                patience=args.patience,
                temperature=[0.0, 0.2, 0.4, 0.6, 0.8, 1.0],
                vad_filter=args.vad_filter,
                condition_on_previous_text=args.condition_on_previous,
                word_timestamps=args.word_timestamps,
                compression_ratio_threshold=2.4,
                log_prob_threshold=-1.0,
                no_speech_threshold=0.6,
            )

            lines: list[str] = []
            for seg in segments:  # generator — iteration triggers the actual decode
                lines.append(f"[{seg.start:7.2f} - {seg.end:7.2f}] {seg.text.strip()}")
            transcript = "\n".join(lines)

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            stem = Path(original_name).stem or "audio"
            # Sanitize filename — strip path separators in case stem contains them.
            stem = stem.replace("/", "_").replace("\\", "_")
            output_path = out_dir / f"{timestamp}_{stem}.txt"
            output_path.write_text(transcript, encoding="utf-8")

            return {
                "status": "success",
                "file": str(output_path),
                "source_filename": original_name,
                "language": info.language,
                "language_probability": round(info.language_probability, 4),
                "duration": round(info.duration, 2),
                "segment_count": len(lines),
                "transcript": transcript,
            }
        finally:
            try:
                os.unlink(temp_path)
            except OSError:
                pass

    return app


# ─── Argparse ─────────────────────────────────────────────────────────────────

def _add_bool_flag(parser, name: str, default: bool, help_text: str):
    """--foo / --no-foo style flag with explicit default."""
    dest = name.replace("-", "_")
    grp = parser.add_mutually_exclusive_group()
    grp.add_argument(f"--{name}", dest=dest, action="store_true", help=help_text)
    grp.add_argument(f"--no-{name}", dest=dest, action="store_false")
    parser.set_defaults(**{dest: default})


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Whisper FastAPI transcription server")

    # Network
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8000)

    # Model
    p.add_argument("--model", default="large-v3",
                   help="Whisper model name or local path.")
    p.add_argument("--device", default="cuda", choices=["cuda", "cpu", "auto"])
    p.add_argument("--compute-type", dest="compute_type", default="float16",
                   choices=["float16", "int8_float16", "int8", "float32"])

    # Transcription
    p.add_argument("--language", default="ja",
                   help="Source language ISO code (e.g. ja, en) or 'auto'.")
    p.add_argument("--beam-size", dest="beam_size", type=int, default=10)
    p.add_argument("--best-of", dest="best_of", type=int, default=10)
    p.add_argument("--patience", type=float, default=2.0)
    _add_bool_flag(p, "vad-filter", True, "Filter non-speech via VAD before decoding.")
    _add_bool_flag(p, "condition-on-previous", True,
                   "Use prior segment as context — improves consistency.")
    _add_bool_flag(p, "word-timestamps", False,
                   "Generate per-word timestamps (slower).")

    # IO
    p.add_argument("--output-dir", dest="output_dir",
                   default="~/Program/Whisper-Audio/Results")

    return p.parse_args(argv)


def main():
    args = parse_args()
    app = create_app(args)
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()