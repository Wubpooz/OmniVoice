#!/usr/bin/env python3

from omnivoice.utils.text import chunk_text_rolling_tokens


def test_chunk_text_rolling_tokens_prefers_punctuation_boundaries():
    text = (
        "Kif int illum? Jiena tajjeb ħafna, grazzi lilek! "
        "Illum se mmur il-laboratorju aktar tard."
    )
    chunks = chunk_text_rolling_tokens(text, min_chunk_tokens=4, max_chunk_tokens=8)
    assert len(chunks) >= 2
    assert "?" in chunks[0]
    assert "grazzi lilek!" in " ".join(chunks)
    assert "il-laboratorju" in " ".join(chunks)


def test_chunk_text_rolling_tokens_respects_max_without_punctuation():
    text = " ".join(["kelma"] * 17)
    chunks = chunk_text_rolling_tokens(text, min_chunk_tokens=5, max_chunk_tokens=7)
    lengths = [len(c.split()) for c in chunks]
    assert all(1 <= n <= 7 for n in lengths)
    assert sum(lengths) == 17


def test_chunk_text_rolling_tokens_with_overlap():
    text = " ".join([f"word{i}" for i in range(1, 13)])
    chunks = chunk_text_rolling_tokens(
        text, min_chunk_tokens=4, max_chunk_tokens=5, overlap_tokens=2
    )
    assert chunks[0].split()[-2:] == chunks[1].split()[:2]
