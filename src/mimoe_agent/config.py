"""Settings: CLI flags > process environment > ``.env`` in the current directory > defaults.

Nothing here touches the network or prints; problems surface as :class:`ConfigError` with a hint
the CLI and server can show verbatim.
"""

from __future__ import annotations

import io
import os
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

from dotenv import dotenv_values

ENV_PREFIX = "MIMOE_"
STRING_FIELDS: tuple[str, ...] = ("base_url", "api_key", "model", "workspace")
BOOL_FIELDS: tuple[str, ...] = ("think", "auto_approve", "allow_network", "force_tools", "trace")
FIELDS: tuple[str, ...] = STRING_FIELDS + BOOL_FIELDS
DOTENV_FIELDS: tuple[str, ...] = STRING_FIELDS
"""Only these four keys are read from ``.env``; booleans come from flags or the real environment."""
DEFAULT_API_KEY = "1234"
TRACING_ENV_VARS: tuple[str, ...] = (
    "LANGSMITH_TRACING",
    "LANGCHAIN_TRACING_V2",
    "LANGSMITH_TRACING_V2",
)

_TRUE = frozenset({"1", "true", "yes", "on"})
_FALSE = frozenset({"0", "false", "no", "off"})


class ConfigError(Exception):
    """A settings problem the user can fix; ``hint`` says how."""

    def __init__(self, message: str, hint: str = "") -> None:
        super().__init__(message)
        self.message = message
        self.hint = hint


@dataclass(frozen=True)
class Settings:
    """Effective configuration plus the origin of every value."""

    base_url: str | None
    """mimOE OpenAI-compatible base URL; ``None`` means auto-discover."""
    api_key: str
    model: str | None
    """Model id to use; ``None`` means the first loaded chat model."""
    workspace: Path
    """Resolved absolute path of an existing directory."""
    think: bool
    auto_approve: bool
    allow_network: bool
    force_tools: bool
    trace: bool
    sources: Mapping[str, str]
    """Field name -> ``"flag"`` | ``"env"`` | ``".env"`` | ``"default"``."""


def env_name(field: str) -> str:
    """Return the environment variable that sets ``field`` (``"model"`` -> ``"MIMOE_MODEL"``)."""
    return ENV_PREFIX + field.upper()


def parse_bool(value: object, *, name: str = "value") -> bool:
    """Parse ``1/true/yes/on`` and ``0/false/no/off`` (case-insensitive); bools pass through.

    Raises:
        ConfigError: for any other value, so a typo never silently means ``False``.
    """
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in _TRUE:
        return True
    if text in _FALSE:
        return False
    raise ConfigError(
        f"{name} must be a boolean, got {value!r}",
        hint="use 1/true/yes/on or 0/false/no/off",
    )


def load_settings(
    overrides: Mapping[str, object] | None = None, *, cwd: Path | None = None
) -> Settings:
    """Build :class:`Settings` from overrides, ``MIMOE_*`` variables, ``cwd/.env`` and defaults.

    Args:
        overrides: CLI flag values keyed by field name; ``None`` values mean "not given".
        cwd: Directory whose ``.env`` and ``workspace/`` are consulted (default: ``Path.cwd()``).

    Raises:
        ConfigError: for an unknown override key, an unparsable boolean, or a missing workspace.
    """
    base = (cwd if cwd is not None else Path.cwd()).resolve()
    flags = {k: v for k, v in (overrides or {}).items() if v is not None}
    unknown = sorted(set(flags) - set(FIELDS))
    if unknown:
        raise ConfigError(
            f"unknown setting(s): {', '.join(unknown)}",
            hint=f"known settings: {', '.join(FIELDS)}",
        )
    dotenv = _read_dotenv(base / ".env")

    raw: dict[str, object] = {}
    sources: dict[str, str] = {}
    for name in FIELDS:
        env_value = os.environ.get(env_name(name), "").strip()
        if name in flags:
            raw[name], sources[name] = flags[name], "flag"
        elif env_value:
            raw[name], sources[name] = env_value, "env"
        elif name in dotenv:
            raw[name], sources[name] = dotenv[name], ".env"
        else:
            raw[name], sources[name] = None, "default"

    bools = {
        name: parse_bool(raw[name], name=f"{env_name(name)} ({sources[name]})")
        if raw[name] is not None
        else False
        for name in BOOL_FIELDS
    }
    base_url = _optional_str(raw["base_url"])
    return Settings(
        base_url=base_url.rstrip("/") if base_url else None,
        api_key=_optional_str(raw["api_key"]) or DEFAULT_API_KEY,
        model=_optional_str(raw["model"]),
        workspace=_resolve_workspace(raw["workspace"], base, sources["workspace"]),
        think=bools["think"],
        auto_approve=bools["auto_approve"],
        allow_network=bools["allow_network"],
        force_tools=bools["force_tools"],
        trace=bools["trace"],
        sources=sources,
    )


def apply_tracing_env(settings: Settings) -> None:
    """Force LangSmith tracing variables to ``false`` unless ``settings.trace`` is set.

    A reviewer with ``LANGSMITH_TRACING=true`` in their shell would otherwise upload every prompt;
    entry points call this once before importing LangChain.
    """
    if settings.trace:
        return
    for var in TRACING_ENV_VARS:
        os.environ[var] = "false"


def _read_text_file(path: Path) -> str:
    """A small text file in UTF-8 (a BOM would hide the first key) or, by its BOM, UTF-16:
    what ``echo MIMOE_MODEL=... > .env`` writes in Windows PowerShell 5.1."""
    data = path.read_bytes()
    if data.startswith((b"\xff\xfe", b"\xfe\xff")):
        return data.decode("utf-16", errors="replace")
    return data.decode("utf-8-sig", errors="replace")


def _read_dotenv(path: Path) -> dict[str, str]:
    """Return the ``MIMOE_*`` string settings found in ``path`` (never booleans, never parents)."""
    if not path.is_file():
        return {}
    values = dotenv_values(stream=io.StringIO(_read_text_file(path)))
    found: dict[str, str] = {}
    for name in DOTENV_FIELDS:
        value = values.get(env_name(name))
        if value is not None and value.strip():
            found[name] = value.strip()
    return found


def _optional_str(value: object) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _resolve_workspace(value: object, base: Path, source: str) -> Path:
    if value is None:
        default = base / "workspace"
        if default.is_dir():
            return default.resolve()
        raise ConfigError("no workspace", hint="pass --workspace PATH")
    path = Path(str(value)).expanduser()
    if not path.is_absolute():
        path = base / path
    path = path.resolve()
    if not path.is_dir():
        raise ConfigError(
            f"workspace is not a directory: {path} (from {source})",
            hint="pass --workspace PATH pointing at an existing folder",
        )
    return path
