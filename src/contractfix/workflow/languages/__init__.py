"""Language-backend registry for qualification and frozen-artifact replay."""

from __future__ import annotations

from collections.abc import Iterable

from .base import LanguageBackend, UnsupportedLanguageError


_BACKENDS: dict[str, LanguageBackend] = {}
_FROZEN_BACKENDS: dict[str, LanguageBackend] = {}
_BUILTINS_LOADED = False


def register_backend(backend: LanguageBackend) -> None:
    """Register one backend and reject ambiguous language/artifact ownership."""
    names = {backend.language, *backend.aliases}
    for name in names:
        key = normalize_language(name)
        owner = _BACKENDS.get(key)
        if owner is not None and owner is not backend:
            raise ValueError(f"language backend already registered: {key}")
        _BACKENDS[key] = backend
    for version in backend.frozen_versions:
        owner = _FROZEN_BACKENDS.get(version)
        if owner is not None and owner is not backend:
            raise ValueError(f"frozen artifact backend already registered: {version}")
        _FROZEN_BACKENDS[version] = backend


def normalize_language(language: str) -> str:
    return language.strip().lower().replace("_", "-")


def backend_for_language(language: str) -> LanguageBackend:
    _load_builtin_backends()
    key = normalize_language(language)
    try:
        return _BACKENDS[key]
    except KeyError as exc:
        supported = ", ".join(supported_languages())
        raise UnsupportedLanguageError(
            f"unsupported repository language {language!r}; available backends: {supported}"
        ) from exc


def backend_for_frozen_version(version: str) -> LanguageBackend:
    _load_builtin_backends()
    try:
        return _FROZEN_BACKENDS[version]
    except KeyError as exc:
        raise UnsupportedLanguageError(
            f"unsupported frozen contract artifact version: {version!r}"
        ) from exc


def supported_languages() -> tuple[str, ...]:
    _load_builtin_backends()
    return tuple(sorted({backend.language for backend in _BACKENDS.values()}))


def registered_backends() -> Iterable[LanguageBackend]:
    _load_builtin_backends()
    unique = {backend.language: backend for backend in _BACKENDS.values()}
    return tuple(unique.values())


def _load_builtin_backends() -> None:
    global _BUILTINS_LOADED
    if _BUILTINS_LOADED:
        return
    from .python import PythonLanguageBackend

    register_backend(PythonLanguageBackend())
    _BUILTINS_LOADED = True
