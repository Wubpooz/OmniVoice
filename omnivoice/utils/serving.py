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

"""Serving-layer helpers for request mutation and low-latency controls."""

from __future__ import annotations

from typing import Optional


_KNOWN_EMOTION_TAGS = {
    "laughter",
    "sigh",
    "confirmation-en",
    "question-en",
    "question-ah",
    "question-oh",
    "question-ei",
    "question-yi",
    "surprise-ah",
    "surprise-oh",
    "surprise-wa",
    "surprise-yo",
    "dissatisfaction-hnn",
}

_EMOTION_ALIAS = {
    "laugh": "laughter",
    "laughing": "laughter",
}


def map_emotion_to_tag(emotion: Optional[str]) -> Optional[str]:
    """Map external emotion controls to OmniVoice inline tag names."""
    if emotion is None:
        return None
    value = emotion.strip().lower()
    if not value:
        return None
    value = _EMOTION_ALIAS.get(value, value)
    if value in _KNOWN_EMOTION_TAGS:
        return value
    return None


def inject_inline_emotion(text: str, emotion: Optional[str]) -> str:
    """Inject an inline OmniVoice emotion tag prefix into text."""
    tag = map_emotion_to_tag(emotion)
    clean_text = text.strip()
    if not tag or not clean_text:
        return clean_text
    inline = f"[{tag}]"
    if clean_text.startswith("["):
        return clean_text
    return f"{inline} {clean_text}"


def apply_speed_step_skipping(num_step: int, speed: float) -> int:
    """Translate speed controls into reverse-sampling step skipping."""
    if speed <= 1.0:
        return max(1, int(num_step))
    scaled = int(round(num_step / speed))
    return max(4, scaled)
