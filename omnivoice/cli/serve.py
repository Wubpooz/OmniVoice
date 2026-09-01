#!/usr/bin/env python3
# Copyright    2026  Xiaomi Corp.        (authors:  Han Zhu)
#
# See ../../LICENSE for clarification regarding multiple authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""FastAPI serving entrypoint for OmniVoice."""

from __future__ import annotations

import argparse
import base64
import io
import json
import logging
from typing import AsyncIterator, Optional

import numpy as np
import soundfile as sf
import torch
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import Response, StreamingResponse
from pydantic import BaseModel, Field

from omnivoice.models.omnivoice import OmniVoice
from omnivoice.utils.common import get_best_device
from omnivoice.utils.serving import apply_speed_step_skipping, inject_inline_emotion


class TTSRequest(BaseModel):
    text: str = Field(..., min_length=1)
    language: Optional[str] = None
    instruct: Optional[str] = None
    emotion: Optional[str] = Field(
        default=None,
        description="Maps to OmniVoice inline tags such as 'laughter' or 'sigh'.",
    )
    speed: float = Field(default=1.0, gt=0.0, le=2.0)
    duration: Optional[float] = Field(default=None, gt=0.0)
    num_step: int = Field(default=32, ge=1, le=128)
    guidance_scale: float = Field(default=2.0, ge=0.0, le=20.0)
    t_shift: float = Field(default=0.1, ge=0.0, le=1.0)
    denoise: bool = True
    postprocess_output: bool = True
    layer_penalty_factor: float = Field(default=5.0, ge=0.0)
    position_temperature: float = Field(default=5.0, ge=0.0)
    class_temperature: float = Field(default=0.0, ge=0.0)


def _wav_bytes(audio: np.ndarray, sample_rate: int) -> bytes:
    buffer = io.BytesIO()
    sf.write(buffer, audio, sample_rate, format="WAV", subtype="PCM_16")
    return buffer.getvalue()


def _model_kwargs(req: TTSRequest) -> dict:
    text = inject_inline_emotion(req.text, req.emotion)
    effective_num_step = apply_speed_step_skipping(req.num_step, req.speed)
    return dict(
        text=text,
        language=req.language,
        instruct=req.instruct,
        duration=req.duration,
        speed=req.speed,
        num_step=effective_num_step,
        guidance_scale=req.guidance_scale,
        t_shift=req.t_shift,
        denoise=req.denoise,
        postprocess_output=req.postprocess_output,
        layer_penalty_factor=req.layer_penalty_factor,
        position_temperature=req.position_temperature,
        class_temperature=req.class_temperature,
    )


def create_app(model: OmniVoice) -> FastAPI:
    app = FastAPI(
        title="OmniVoice FastAPI",
        version="1.0.0",
        description=(
            "OpenAPI serving layer for OmniVoice with emotion-to-inline-tag mapping "
            "and chunked low-latency streaming output."
        ),
    )

    @app.get("/healthz")
    def healthz():
        return {"status": "ok"}

    @app.post("/v1/tts")
    def tts(req: TTSRequest):
        kwargs = _model_kwargs(req)
        try:
            audio = model.generate(**kwargs)[0]
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"{type(e).__name__}: {e}") from e
        payload = _wav_bytes(audio, model.sampling_rate)
        return Response(content=payload, media_type="audio/wav")

    @app.post("/v1/tts/stream")
    async def tts_stream(req: TTSRequest):
        kwargs = _model_kwargs(req)

        async def _iter() -> AsyncIterator[bytes]:
            try:
                for idx, chunk_audio in enumerate(model.generate_stream(**kwargs)):
                    chunk_wav = _wav_bytes(chunk_audio, model.sampling_rate)
                    packet = {
                        "index": idx,
                        "sample_rate": model.sampling_rate,
                        "audio_wav_base64": base64.b64encode(chunk_wav).decode("ascii"),
                    }
                    yield (json.dumps(packet) + "\n").encode("utf-8")
            except Exception as e:
                packet = {"error": f"{type(e).__name__}: {e}"}
                yield (json.dumps(packet) + "\n").encode("utf-8")

        return StreamingResponse(_iter(), media_type="application/x-ndjson")

    return app


def get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Serve OmniVoice with FastAPI.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--model", type=str, default="k2-fsa/OmniVoice")
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    return parser


def main():
    logging.basicConfig(
        format="%(asctime)s %(levelname)s [%(filename)s:%(lineno)d] %(message)s",
        level=logging.INFO,
        force=True,
    )
    args = get_parser().parse_args()

    device = args.device or get_best_device()
    logging.info("Loading model from %s on %s ...", args.model, device)
    model = OmniVoice.from_pretrained(args.model, device_map=device, dtype=torch.float16)

    app = create_app(model)
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
