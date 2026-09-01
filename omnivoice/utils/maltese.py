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

"""Maltese preprocessing and deterministic G2P for inference.

Adapted from University of Malta MASRI resources:
- g2p rules: https://github.com/UMSpeech/MASRI/blob/main/masri/transcribe/g2p/maltese_g2p.py
- tokenization patterns:
  https://github.com/UMSpeech/MASRI/blob/main/masri/tokenise/tokenise.py

This module keeps a lightweight, dependency-free inference path.
"""

from __future__ import annotations

import re
import string
from typing import List

ALL_VOWELS = [
    "ί",
    "a",
    "e",
    "i",
    "o",
    "u",
    "ϊ",
    "ä",
    "ë",
    "ï",
    "ö",
    "ü",
    "ɐ",
    "ɛ",
    "ɪ",
    "ɔ",
    "ʊ",
    "à",
    "è",
    "ì",
    "ò",
    "ù",
]

ALL_CONSONANTS = [
    "b",
    "ċ",
    "d",
    "f",
    "g",
    "ġ",
    "h",
    "ħ",
    "j",
    "k",
    "l",
    "m",
    "n",
    "p",
    "q",
    "r",
    "s",
    "t",
    "v",
    "w",
    "x",
    "z",
    "ż",
    "c",
    "y",
    "ʔ",
    "ʦ",
    "ʧ",
    "ǳ",
    "ʤ",
    "ʃ",
    "ʒ",
]

_MT_TOKEN_RE = re.compile(
    r"\d{1,2}[-/]\d{1,2}[-/]\d{2,4}"
    r"|\d{2,4}[-/]\d{1,2}[-/]\d{1,2}"
    r"|\d+[\.,/]\d+"
    r"|\d+"
    r"|\w+[`'’]?"
    r"|[^\w\s]",
    re.UNICODE | re.MULTILINE | re.DOTALL | re.IGNORECASE,
)
_WORD_RE = re.compile(r"^\w+(?:[-'’]\w+)*$", re.UNICODE)
_ENG_WORD_RE = re.compile(r"^[A-Za-z][A-Za-z'’-]*$")
_MALTESE_UNIQUE_CHARS = set("ħċġż")

_LOANWORD_IPA = {
    "computer": "kəmpjuːtə",
    "internet": "ɪntənet",
    "manager": "mænɪdʒə",
    "email": "iːmeɪl",
    "mobile": "məʊbaɪl",
    "laptop": "læptɒp",
    "server": "sɜːvə",
    "software": "sɒftweə",
    "hardware": "hɑːdweə",
    "meeting": "miːtɪŋ",
}


def _apply_rule_c_graph_c(word: str, graph_in: str, graph_out: str) -> str:
    for c_left in ALL_CONSONANTS:
        for c_right in ALL_CONSONANTS:
            word = word.replace(
                c_left + graph_in + c_right,
                c_left + graph_out + c_right,
            )
    return word


def _apply_rule_v_graph_v(word: str, graph_in: str, graph_out: str) -> str:
    for v_left in ALL_VOWELS:
        for v_right in ALL_VOWELS:
            word = word.replace(
                v_left + graph_in + v_right,
                v_left + graph_out + v_right,
            )
    return word


def _apply_rule_c_graph_v(word: str, graph_in: str, graph_out: str) -> str:
    for c_left in ALL_CONSONANTS:
        for v_right in ALL_VOWELS:
            word = word.replace(
                c_left + graph_in + v_right,
                c_left + graph_out + v_right,
            )
    return word


def _apply_rule_v_graph_c(word: str, graph_in: str, graph_out: str) -> str:
    for v_left in ALL_VOWELS:
        for c_right in ALL_CONSONANTS:
            word = word.replace(
                v_left + graph_in + c_right,
                v_left + graph_out + c_right,
            )
    return word


def _apply_rule_graph_c(word: str, graph_in: str, graph_out: str) -> str:
    for c_right in ALL_CONSONANTS:
        word = word.replace(graph_in + c_right, graph_out + c_right)
    return word


def _apply_rule_graph_v(word: str, graph_in: str, graph_out: str) -> str:
    for v_right in ALL_VOWELS:
        word = word.replace(graph_in + v_right, graph_out + v_right)
    return word


def _apply_rule_v_graph(word: str, graph_in: str, graph_out: str) -> str:
    for v_left in ALL_VOWELS:
        word = word.replace(v_left + graph_in, v_left + graph_out)
    return word


def g2p_cw_rules(text: str) -> str:
    """Apply CrimsonWing Maltese deterministic grapheme-to-phoneme rules."""
    text = str(text).lower()
    text = text.replace("-", "ŵ")
    text = text.replace("'", "")
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = re.sub(r"\s+", " ", text).strip()
    text = text.replace("ŵ", "-")

    text = text.replace("y", "ɪ")
    text = text.replace("ca", "ka")
    text = text.replace("ce", "se")
    text = text.replace("ci", "si")
    text = text.replace("co", "ko")
    text = text.replace("cu", "su")
    text = text.replace("ie", "ί")

    text = "#" + text + "#"
    text = text.replace(" ", "#")

    text = text.replace("għu", "ɔʊ")
    text = text.replace("għi", "ɛɪ")
    text = text.replace("aj", "ɐɪ")
    text = text.replace("aw", "ɐʊ")
    text = text.replace("ej", "ɛɪ")
    text = text.replace("ew", "ɛʊ")
    text = text.replace("iw", "ɪʊ")
    text = text.replace("oj", "ɔɪ")
    text = text.replace("ow", "ɔʊ")

    text = text.replace("ίgħe", "ίjë")
    text = text.replace("agħa", "ä")
    text = text.replace("egħe", "ë")
    text = text.replace("ogħo", "ö")
    text = _apply_rule_c_graph_c(text, "ehi", "ëhi")
    text = _apply_rule_c_graph_c(text, "egħi", "ëgħi")
    text = text.replace("aha", "ä")
    text = text.replace("aho", "ö")
    text = text.replace("ehe", "ë")
    text = text.replace("għa", "għä")
    text = text.replace("agħ", "ägħ")
    text = text.replace("għe", "għë")
    text = text.replace("egħ", "ëgħ")
    text = text.replace("għo", "għö")
    text = text.replace("ogħ", "ögħ")
    text = text.replace("ha", "hä")
    text = text.replace("ah", "äh")
    text = text.replace("he", "ëh")
    text = text.replace("eh", "ëh")
    text = _apply_rule_graph_c(text, "ίgħ", "ëgħ")
    text = text.replace("ί", "ϊ")
    text = text.replace("iħ", "ïħ")
    text = text.replace("igħ", "ïgħ")
    text = text.replace("ih", "ïh")
    text = text.replace("iq", "ïq")
    text = text.replace("a", "ɐ")
    text = text.replace("e", "ɛ")
    text = text.replace("i", "ɪ")
    text = text.replace("o", "ɔ")
    text = text.replace("u", "ʊ")

    text = text.replace("bċ", "pċ")
    text = text.replace("bf", "pf")
    text = text.replace("bħ", "pħ")
    text = text.replace("bk", "pk")
    text = text.replace("bp", "pp")
    text = text.replace("bq", "pq")
    text = text.replace("bs", "ps")
    text = text.replace("bt", "pt")
    text = text.replace("bx", "px")
    text = text.replace("bz", "pz")
    text = text.replace("b#", "p#")

    text = text.replace("ċb", "ʤb")
    text = text.replace("ċd", "ʤd")
    text = text.replace("ċġ", "ʤġ")
    text = text.replace("ċg", "ʤg")
    text = text.replace("ċv", "ʤv")
    text = text.replace("ċż", "ʤż")
    text = text.replace("ċ", "ʧ")
    text = _apply_rule_v_graph(text, "dx", "ʧx")
    text = _apply_rule_v_graph(text, "ddx", "ʧdx")
    text = _apply_rule_v_graph(text, "dtx", "ʧtx")
    text = _apply_rule_v_graph(text, "ds", "ʦs")
    text = _apply_rule_v_graph(text, "dds", "ʦds")
    text = text.replace("ndn", "nnn")
    text = text.replace("dċ", "tċ")
    text = text.replace("df", "tf")
    text = text.replace("dħ", "tħ")
    text = text.replace("dk", "tk")
    text = text.replace("dp", "tp")
    text = text.replace("dq", "tq")
    text = text.replace("ds", "ts")
    text = text.replace("dt", "t")
    text = text.replace("dx", "tx")
    text = text.replace("d#", "t#")
    text = text.replace("fb", "vb")
    text = text.replace("fd", "vd")
    text = text.replace("fġ", "vġ")
    text = text.replace("fg", "vg")
    text = text.replace("fv", "v")
    text = text.replace("fż", "vż")

    text = _apply_rule_v_graph_v(text, "għh", "Hh")
    text = _apply_rule_v_graph_v(text, "għ", "")
    text = _apply_rule_c_graph_v(text, "għ", "")
    text = _apply_rule_v_graph_c(text, "għ", "")
    text = text.replace("#għ", "#")
    text = text.replace("għ#", "h#")
    text = text.replace("ġċ", "ʧċ")
    text = text.replace("ġf", "ʧf")
    text = text.replace("ġħ", "ʧħ")
    text = text.replace("ġk", "ʧk")
    text = text.replace("ġp", "ʧp")
    text = text.replace("ġq", "ʧq")
    text = text.replace("ġs", "ʧs")
    text = text.replace("ġt", "ʧt")
    text = text.replace("ġx", "ʧx")
    text = text.replace("ġ#", "ʧ#")
    text = text.replace("ġ", "ʤ")
    text = text.replace("gċ", "kċ")
    text = text.replace("gf", "kf")
    text = text.replace("għ", "kħ")
    text = text.replace("gk", "kk")
    text = text.replace("gp", "kp")
    text = text.replace("gq", "kq")
    text = text.replace("gs", "ks")
    text = text.replace("gt", "kt")
    text = text.replace("gx", "kx")
    text = text.replace("g#", "k#")
    text = text.replace("#h", "#")
    text = text.replace("h#", "H#")
    text = _apply_rule_c_graph_v(text, "h", "")
    text = _apply_rule_graph_v(text, "ih", "ij")
    text = _apply_rule_graph_v(text, "ïh", "ïj")
    text = _apply_rule_graph_v(text, "ίh", "ίj")
    text = _apply_rule_graph_v(text, "ϊh", "ϊj")
    text = text.replace("ʊhί", "ʊwί")
    text = text.replace("ʊhɐ", "ʊwɐ")
    text = text.replace("ʊhɪ", "ʊwɪ")
    text = text.replace("ʊhɔ", "ʊwɔ")
    text = text.replace("ʊhʊ", "ʊwʊ")
    text = text.replace("ʊhϊ", "ʊwϊ")
    text = text.replace("ʊhä", "ʊwä")
    text = text.replace("ʊhï", "ʊwï")
    text = text.replace("ʊhö", "ʊwö")
    text = text.replace("ʊhü", "ʊwü")
    text = text.replace("ühί", "ʊwί")
    text = text.replace("ühɐ", "ʊwɐ")
    text = text.replace("ühɪ", "ʊwɪ")
    text = text.replace("ühɔ", "ʊwɔ")
    text = text.replace("ühʊ", "ʊwʊ")
    text = text.replace("ühϊ", "ʊwϊ")
    text = text.replace("ühä", "ʊwä")
    text = text.replace("ühï", "ʊwï")
    text = text.replace("ühö", "ʊwö")
    text = text.replace("ühü", "ʊwü")
    text = _apply_rule_v_graph_v(text, "h", "")
    text = text.replace("h", "")
    text = text.replace("H", "h")
    text = text.replace("ħ", "h")

    text = text.replace("kb", "gb")
    text = text.replace("kd", "gd")
    text = text.replace("kġ", "gġ")
    text = text.replace("kg", "gg")
    text = text.replace("kv", "gv")
    text = text.replace("kż", "gż")
    text = text.replace("nb", "mb")
    text = text.replace("np", "mp")
    text = text.replace("ɪnl", "ɪll")
    text = text.replace("ϊnl", "ϊll")
    text = text.replace("ɪnm", "ɪmm")
    text = text.replace("ϊnm", "ϊmm")
    text = text.replace("ɪnr", "ɪrr")
    text = text.replace("ϊnr", "ϊrr")
    text = text.replace("pb", "bb")
    text = text.replace("pd", "bd")
    text = text.replace("pġ", "bġ")
    text = text.replace("pg", "bg")
    text = text.replace("pv", "bv")
    text = text.replace("pż", "bż")
    text = text.replace("q", "ʔ")
    text = _apply_rule_v_graph(text, "ss-x", "ssʃ")
    text = _apply_rule_v_graph(text, "sx#", "sʃ#")
    text = text.replace("ssx#", "ssʃ#")
    text = text.replace("sb", "Zb")
    text = text.replace("sd", "Zd")
    text = text.replace("sġ", "Zġ")
    text = text.replace("sʤ", "Zʤ")
    text = text.replace("sg", "Zg")
    text = text.replace("sv", "Zv")
    text = text.replace("sż", "Zż")
    text = text.replace("tb", "db")
    text = text.replace("td", "dd")
    text = text.replace("tġ", "dġ")
    text = text.replace("tʤ", "dʤ")
    text = text.replace("tg", "dg")
    text = text.replace("tv", "dv")
    text = text.replace("tż", "dż")
    text = _apply_rule_v_graph(text, "tx", "ʧx")
    text = _apply_rule_v_graph(text, "ts", "ʦs")
    text = text.replace("vċ", "fċ")
    text = text.replace("vf", "ff")
    text = text.replace("vħ", "fħ")
    text = text.replace("vk", "fk")
    text = text.replace("vp", "fp")
    text = text.replace("vq", "fq")
    text = text.replace("vs", "fs")
    text = text.replace("vt", "ft")
    text = text.replace("vx", "fx")
    text = text.replace("vz", "fz")
    text = text.replace("v#", "f#")
    text = text.replace("xb", "ʒb")
    text = text.replace("xd", "ʒd")
    text = text.replace("xġ", "ʒġ")
    text = text.replace("xg", "ʒg")
    text = _apply_rule_v_graph_v(text, "x", "ʒ")
    text = text.replace("x", "X")
    text = _apply_rule_v_graph_v(text, "zz", "ǳ ")
    text = text.replace("z", "ʦ")
    text = _apply_rule_v_graph(text, "żż-X", "zsʃ")
    text = text.replace("żżx#", "żzx#")
    text = text.replace("żżX#", "żzX#")
    text = text.replace("żż-", "żz-")
    text = text.replace("żċ", "sċ")
    text = text.replace("żf", "sf")
    text = text.replace("żħ", "sħ")
    text = text.replace("żk", "sk")
    text = text.replace("żp", "sp")
    text = text.replace("żq", "sq")
    text = text.replace("żs", "ss")
    text = text.replace("żt", "st")
    text = text.replace("żx", "sx")
    text = text.replace("żX", "sX")
    text = text.replace("ż#", "s#")
    text = text.replace("ż", "z")

    text = text.replace("#", " ").strip()
    text = text.replace("-", "")
    text = text.replace("'", "")
    text = text.replace("Z", "z")
    text = text.replace("X", "ʃ")
    text = text.replace("ä", "ɐː")
    text = text.replace("ë", "ɛː")
    text = text.replace("ï", "iː")
    text = text.replace("ί", "ɪː")
    text = text.replace("ϊ", "ɪː")
    text = text.replace("ö", "ɔː")
    text = text.replace("ü", "ʊː")
    text = text.replace("ʦ", "ts")
    text = text.replace("ǳ ", "dz")
    return text.strip()


def _tokenize_maltese(text: str) -> List[str]:
    return _MT_TOKEN_RE.findall(text)


def _is_word(token: str) -> bool:
    return bool(_WORD_RE.match(token))


def _is_english_loan_candidate(token: str) -> bool:
    if not _ENG_WORD_RE.match(token):
        return False
    lowered = token.lower()
    return "għ" not in lowered and not any(ch in _MALTESE_UNIQUE_CHARS for ch in lowered)


def preprocess_maltese_text(text: str) -> str:
    """Convert Maltese text to IPA-like phonemic sequence with punctuation."""
    tokens = _tokenize_maltese(text)
    out: List[str] = []
    for tok in tokens:
        if not _is_word(tok):
            out.append(tok)
            continue
        lowered = tok.lower()
        if _is_english_loan_candidate(tok) and lowered in _LOANWORD_IPA:
            out.append(_LOANWORD_IPA[lowered])
        elif _is_english_loan_candidate(tok) and lowered not in _LOANWORD_IPA:
            out.append(tok)
        else:
            out.append(g2p_cw_rules(tok))

    result = " ".join(out)
    result = re.sub(r"\s*-\s*", "-", result)
    result = re.sub(r"\s+([,.;:!?])", r"\1", result)
    result = re.sub(r"\(\s+", "(", result)
    result = re.sub(r"\s+\)", ")", result)
    return re.sub(r"\s+", " ", result).strip()


def should_apply_maltese_g2p(language: str | None) -> bool:
    if language is None:
        return False
    normalized = language.strip().lower()
    return normalized in {"mt", "mt-mt", "maltese"}
