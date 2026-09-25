"""Settings precedence, ``.env`` scoping, boolean parsing and workspace resolution."""

from __future__ import annotations

import dataclasses
import os
from pathlib import Path

import pytest

from mimoe_agent.config import (
    BOOL_FIELDS,
    FIELDS,
    ConfigError,
    apply_tracing_env,
    env_name,
    load_settings,
    parse_bool,
)


def test_defaults(workspace_tmp: Path) -> None:
    settings = load_settings(cwd=workspace_tmp.parent)
    assert settings.base_url is None
    assert settings.api_key == "1234"
    assert settings.model is None
    assert settings.workspace == workspace_tmp.resolve()
    assert settings.workspace.is_absolute()
    assert not any(getattr(settings, name) for name in BOOL_FIELDS)
    assert set(settings.sources) == set(FIELDS)
    assert set(settings.sources.values()) == {"default"}


def test_precedence_flag_over_env_over_dotenv(
    workspace_tmp: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cwd = workspace_tmp.parent
    (cwd / ".env").write_text(
        "MIMOE_BASE_URL=http://dotenv:8083/mimik-ai/openai/v1/\n"
        "MIMOE_API_KEY=dotenv-key\n"
        "MIMOE_MODEL=dotenv-model\n"
        "MIMOE_THINK=1\n"
        "MIMOE_AUTO_APPROVE=true\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("MIMOE_API_KEY", "env-key")
    monkeypatch.setenv("MIMOE_MODEL", "env-model")
    settings = load_settings({"model": "flag-model", "think": None}, cwd=cwd)
    assert settings.base_url == "http://dotenv:8083/mimik-ai/openai/v1"  # trailing slash dropped
    assert settings.sources["base_url"] == ".env"
    assert settings.api_key == "env-key"
    assert settings.sources["api_key"] == "env"
    assert settings.model == "flag-model"
    assert settings.sources["model"] == "flag"
    # booleans are never read from .env, and a None override means "not given"
    assert settings.think is False
    assert settings.auto_approve is False
    assert settings.sources["think"] == "default"


def test_dotenv_is_read_only_from_cwd(tmp_path: Path) -> None:
    (tmp_path / ".env").write_text("MIMOE_MODEL=from-parent\n", encoding="utf-8")
    project = tmp_path / "project"
    (project / "workspace").mkdir(parents=True)
    settings = load_settings(cwd=project)
    assert settings.model is None
    assert settings.sources["model"] == "default"
    (project / ".env").write_text("MIMOE_MODEL=from-cwd\n", encoding="utf-8")
    assert load_settings(cwd=project).model == "from-cwd"


def test_dotenv_with_utf8_bom(workspace_tmp: Path) -> None:
    """Windows editors may save .env with a BOM; the first key must still be read."""
    (workspace_tmp.parent / ".env").write_bytes(b"\xef\xbb\xbfMIMOE_MODEL=bom-model\n")
    assert load_settings(cwd=workspace_tmp.parent).model == "bom-model"


def test_dotenv_does_not_touch_process_env(workspace_tmp: Path) -> None:
    (workspace_tmp.parent / ".env").write_text("MIMOE_MODEL=quiet\nLANGSMITH_TRACING=true\n")
    settings = load_settings(cwd=workspace_tmp.parent)
    assert settings.model == "quiet"
    assert "MIMOE_MODEL" not in os.environ
    assert "LANGSMITH_TRACING" not in os.environ


@pytest.mark.parametrize("text", ["1", "true", "YES", "On", " yes "])
def test_bool_env_true(text: str, workspace_tmp: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MIMOE_THINK", text)
    settings = load_settings(cwd=workspace_tmp.parent)
    assert settings.think is True
    assert settings.sources["think"] == "env"


@pytest.mark.parametrize("text", ["0", "false", "No", "OFF"])
def test_bool_env_false(text: str, workspace_tmp: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MIMOE_AUTO_APPROVE", text)
    assert load_settings(cwd=workspace_tmp.parent).auto_approve is False


def test_bool_env_garbage_raises(workspace_tmp: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MIMOE_FORCE_TOOLS", "maybe")
    with pytest.raises(ConfigError) as info:
        load_settings(cwd=workspace_tmp.parent)
    assert "MIMOE_FORCE_TOOLS" in info.value.message
    assert "1/true/yes/on" in info.value.hint


def test_empty_env_value_means_unset(workspace_tmp: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MIMOE_BASE_URL", "")
    monkeypatch.setenv("MIMOE_TRACE", "   ")
    settings = load_settings(cwd=workspace_tmp.parent)
    assert settings.base_url is None
    assert settings.trace is False
    assert settings.sources["trace"] == "default"


def test_parse_bool() -> None:
    assert parse_bool(True) is True
    assert parse_bool(False) is False
    assert parse_bool("on") is True
    assert parse_bool("off") is False
    with pytest.raises(ConfigError):
        parse_bool("nope", name="x")


def test_flag_values_may_be_bools_or_strings(workspace_tmp: Path) -> None:
    settings = load_settings(
        {"auto_approve": True, "allow_network": "yes", "force_tools": "0", "trace": True},
        cwd=workspace_tmp.parent,
    )
    assert settings.auto_approve is True
    assert settings.allow_network is True
    assert settings.force_tools is False
    assert settings.trace is True
    assert settings.sources["trace"] == "flag"


def test_workspace_default_missing_raises(tmp_path: Path) -> None:
    with pytest.raises(ConfigError) as info:
        load_settings(cwd=tmp_path)
    assert info.value.message == "no workspace"
    assert info.value.hint == "pass --workspace PATH"


def test_workspace_explicit_relative_path(tmp_path: Path) -> None:
    (tmp_path / "ws").mkdir()
    settings = load_settings({"workspace": "ws"}, cwd=tmp_path)
    assert settings.workspace == (tmp_path / "ws").resolve()
    assert settings.sources["workspace"] == "flag"


def test_workspace_explicit_path_object_and_env(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (tmp_path / "a").mkdir()
    (tmp_path / "b").mkdir()
    monkeypatch.setenv(env_name("workspace"), str(tmp_path / "b"))
    assert load_settings(cwd=tmp_path).workspace == (tmp_path / "b").resolve()
    assert (
        load_settings({"workspace": tmp_path / "a"}, cwd=tmp_path).workspace
        == (tmp_path / "a").resolve()
    )


def test_workspace_not_a_directory_raises(tmp_path: Path) -> None:
    (tmp_path / "file.txt").write_text("x")
    with pytest.raises(ConfigError) as info:
        load_settings({"workspace": "file.txt"}, cwd=tmp_path)
    assert "not a directory" in info.value.message
    assert "--workspace" in info.value.hint
    with pytest.raises(ConfigError):
        load_settings({"workspace": "missing"}, cwd=tmp_path)


def test_unknown_override_key_raises(workspace_tmp: Path) -> None:
    with pytest.raises(ConfigError) as info:
        load_settings({"modle": "typo"}, cwd=workspace_tmp.parent)
    assert "modle" in info.value.message


def test_settings_is_frozen(workspace_tmp: Path) -> None:
    settings = load_settings(cwd=workspace_tmp.parent)
    with pytest.raises(dataclasses.FrozenInstanceError):
        settings.model = "x"  # type: ignore[misc]


def test_apply_tracing_env(workspace_tmp: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LANGSMITH_TRACING", "true")
    apply_tracing_env(load_settings(cwd=workspace_tmp.parent))
    assert os.environ["LANGSMITH_TRACING"] == "false"
    assert os.environ["LANGCHAIN_TRACING_V2"] == "false"
    monkeypatch.setenv("LANGSMITH_TRACING", "true")
    apply_tracing_env(load_settings({"trace": True}, cwd=workspace_tmp.parent))
    assert os.environ["LANGSMITH_TRACING"] == "true"
