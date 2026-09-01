#!/usr/bin/env python3

from omnivoice.utils.maltese import (
    g2p_cw_rules,
    preprocess_maltese_text,
    should_apply_maltese_g2p,
)


def test_should_apply_maltese_g2p_variants():
    assert should_apply_maltese_g2p("mt")
    assert should_apply_maltese_g2p("mt-MT")
    assert should_apply_maltese_g2p("maltese")
    assert not should_apply_maltese_g2p("en")


def test_g2p_distinguishes_z_and_z_with_dot():
    z_plain = g2p_cw_rules("zarbun")
    z_dotted = g2p_cw_rules("żarbun")
    assert "ts" in z_plain
    assert "z" in z_dotted


def test_g2p_handles_gh_silence_lengthening():
    out = g2p_cw_rules("għada")
    assert "ɐː" in out


def test_preprocess_maps_loanword_to_english_ipa():
    out = preprocess_maltese_text("Se mmur il-computer lab.")
    assert "kəmpjuːtə" in out
    assert "." in out
