"""Explicitly supported native-language prompt packs."""

from __future__ import annotations

from .base import NativeLanguagePromptPack
from .c_v3 import PACK as C_PACK
from .cpp_v3 import PACK as CPP_PACK
from .java_v3 import PACK as JAVA_PACK
from .javascript_v3 import PACK as JAVASCRIPT_PACK


PACKS: dict[str, NativeLanguagePromptPack] = {
    pack.language: pack for pack in (JAVA_PACK, C_PACK, CPP_PACK, JAVASCRIPT_PACK)
}


def prompt_pack_for(language: str) -> NativeLanguagePromptPack:
    try:
        return PACKS[language]
    except KeyError as exc:
        raise ValueError(f"unsupported native prompt language: {language}") from exc


__all__ = ["NativeLanguagePromptPack", "prompt_pack_for"]
