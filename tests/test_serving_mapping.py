#!/usr/bin/env python3

from omnivoice.utils.serving import (
    apply_speed_step_skipping,
    inject_inline_emotion,
    map_emotion_to_tag,
)


def test_map_emotion_alias_and_known_tags():
    assert map_emotion_to_tag("laugh") == "laughter"
    assert map_emotion_to_tag("laughter") == "laughter"
    assert map_emotion_to_tag("unknown") is None


def test_inject_inline_emotion_prefixes_clean_text():
    out = inject_inline_emotion("Kif int?", "laughter")
    assert out.startswith("[laughter] ")
    assert out.endswith("Kif int?")


def test_inject_inline_emotion_skips_when_already_tagged():
    out = inject_inline_emotion("[sigh] Kif int?", "laughter")
    assert out == "[sigh] Kif int?"


def test_apply_speed_step_skipping_behavior():
    assert apply_speed_step_skipping(32, 1.0) == 32
    assert apply_speed_step_skipping(32, 2.0) == 16
    assert apply_speed_step_skipping(6, 2.0) == 4
