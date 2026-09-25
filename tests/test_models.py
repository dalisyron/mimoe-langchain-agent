"""Registry helpers: presets, spec parsing, pulls against both fake stores, model switching."""

from __future__ import annotations

import json

import httpx
import pytest
from conftest import HF_REPOS, NODE_ID, FakeMimoe

from mimoe_agent.mimoe import MimoeClient, MimoeError
from mimoe_agent.models import (
    DEFAULT_PRESET,
    PRESETS,
    PULL_TIMEOUT,
    V10_ONLY_PRESETS,
    PullSpec,
    _iter_sse_objects,
    download_url,
    memory_note,
    parse_spec,
    preset_by_id,
    pull_model,
    quant_of,
    switch_model,
    valid_model_id,
)

STORE = "/mimik-ai/store/v1"


def _client(fake: FakeMimoe, api_key: str = "1234") -> MimoeClient:
    return MimoeClient(fake.base_url, api_key, client=fake.client())


class Progress:
    """Collects ``(text, fraction)`` callbacks."""

    def __init__(self) -> None:
        self.events: list[tuple[str, float | None]] = []

    def __call__(self, text: str, fraction: float | None) -> None:
        self.events.append((text, fraction))

    @property
    def texts(self) -> list[str]:
        return [text for text, _ in self.events]

    @property
    def fractions(self) -> list[float]:
        return [fraction for _, fraction in self.events if fraction is not None]


def _store_posts(fake: FakeMimoe) -> list[httpx.Request]:
    """POSTs to the model store (the JSON-RPC ``getMe`` of discovery is a POST too)."""
    return [r for r in fake.requests if r.method == "POST" and r.url.path.startswith(STORE)]


def _posts(fake: FakeMimoe, suffix: str) -> list[dict]:
    return [json.loads(r.content) for r in _store_posts(fake) if r.url.path.endswith(suffix)]


# -- presets -------------------------------------------------------------------------------------


def test_presets_integrity() -> None:
    ids = [p.id for p in PRESETS]
    assert len(ids) == len(set(ids)) == 6
    assert ids[0] == DEFAULT_PRESET == "qwen3-4b-instruct-2507"
    assert set(ids) == {
        "qwen3-4b-instruct-2507",
        "qwen3-8b",
        "smollm3-3b",
        "smollm2-360m",
        "qwen3.5-4b",
        "qwen3.5-9b",
    }
    for preset in PRESETS:
        assert preset.size_gb > 0
        assert preset.repo.count("/") == 1
        assert preset.file.endswith(".gguf")
        assert quant_of(preset.file) in ("Q4_K_M", "Q8_0")
        assert preset.note and len(preset.note) < 200
        assert preset.file == HF_REPOS[preset.repo][0]["file"]
    assert {"qwen3.5-4b", "qwen3.5-9b"} == V10_ONLY_PRESETS


def test_preset_notes_follow_the_matrix() -> None:
    by_id = {p.id: p for p in PRESETS}
    assert "recommended" in by_id["qwen3-4b-instruct-2507"].note
    assert by_id["qwen3-4b-instruct-2507"].size_gb == 2.5
    assert by_id["qwen3-8b"].size_gb == pytest.approx(5.03)
    assert "chat only" in by_id["smollm2-360m"].note
    assert "1.0" in by_id["qwen3.5-4b"].note and "1.0" in by_id["qwen3.5-9b"].note
    assert "0.6" in by_id["smollm3-3b"].note
    assert preset_by_id("QWEN3-8B") is by_id["qwen3-8b"]
    assert preset_by_id("nope") is None


def test_quant_of_and_download_url() -> None:
    assert quant_of("Qwen3-8B-Q4_K_M.gguf") == "Q4_K_M"
    assert quant_of("SmolLM2-360M-Instruct-q8_0.gguf") == "Q8_0"
    assert quant_of("Model-UD-Q4_K_XL.gguf") == "UD-Q4_K_XL"
    assert quant_of("Model-IQ4_XS.gguf") == "IQ4_XS"
    assert quant_of("Model-BF16.gguf") == "BF16"
    assert quant_of("Model.gguf") is None
    assert (
        download_url("Qwen/Qwen3-8B-GGUF", "Qwen3-8B-Q4_K_M.gguf")
        == "https://huggingface.co/Qwen/Qwen3-8B-GGUF/resolve/main/Qwen3-8B-Q4_K_M.gguf"
    )


# -- spec parsing --------------------------------------------------------------------------------


def test_parse_spec_preset() -> None:
    spec = parse_spec(" qwen3-8b ")
    assert spec == PullSpec(
        model_id="qwen3-8b",
        repo="Qwen/Qwen3-8B-GGUF",
        quant="Q4_K_M",
        file="Qwen3-8B-Q4_K_M.gguf",
        preset=preset_by_id("qwen3-8b"),
    )


def test_parse_spec_custom_quant_derives_id_and_file() -> None:
    spec = parse_spec("someone/My-Model-GGUF:q5_k_m")
    assert spec.model_id == "my-model-q5_k_m"
    assert spec.repo == "someone/My-Model-GGUF"
    assert spec.quant == "Q5_K_M"
    assert spec.file == "My-Model-Q5_K_M.gguf"
    assert spec.preset is None


def test_parse_spec_custom_matching_a_preset_maps_to_it() -> None:
    assert parse_spec("qwen/qwen3-8b-gguf:q4-k-m").model_id == "qwen3-8b"
    assert parse_spec("Qwen/Qwen3-8B-GGUF:Q4_K_M").preset is preset_by_id("qwen3-8b")
    assert parse_spec("Qwen/Qwen3-8B-GGUF:Q8_0").model_id == "qwen3-8b-q8_0"


def test_parse_spec_file_form_and_repo_only() -> None:
    spec = parse_spec("someone/My-Model-GGUF:My-Model-IQ4_XS.gguf")
    assert (spec.quant, spec.file, spec.model_id) == (
        "IQ4_XS",
        "My-Model-IQ4_XS.gguf",
        "my-model-iq4_xs",
    )
    plain = parse_spec("someone/Solo-Model-GGUF")
    assert (plain.quant, plain.file, plain.model_id) == (None, None, "solo-model")
    odd = parse_spec("someone/Model:weights.gguf")
    assert (odd.quant, odd.file, odd.model_id) == (None, "weights.gguf", "model")


@pytest.mark.parametrize(
    "spec",
    ["", "   ", "qwen3", "a/b/c", "a/b:", ":Q4", "a b/c:Q4", "/b", "a/", "a/b:Q4:Q5", "a/b:Q 4"],
)
def test_parse_spec_invalid(spec: str) -> None:
    with pytest.raises(MimoeError) as info:
        parse_spec(spec)
    assert "qwen3-4b-instruct-2507" in info.value.hint
    assert "owner/repo:QUANT" in info.value.hint


# -- memory note ---------------------------------------------------------------------------------


def test_memory_note_thresholds() -> None:
    assert memory_note(0) is None
    assert memory_note(-5) is None
    assert memory_note(2_497_281_120) is None
    assert memory_note(4_500_000_000) is None
    assert memory_note(4_600_000_000) == "about 6 GB at 12k context; tight on a 16 GB machine"
    assert memory_note(5_027_783_488) == "about 7 GB at 12k context; tight on a 16 GB machine"
    assert memory_note(5_680_522_464) == "about 7 GB at 12k context; tight on a 16 GB machine"
    assert memory_note(5_027_783_488, max_context=32000) == (
        "about 9 GB at 32k context; tight on a 16 GB machine"
    )
    assert memory_note(12_000_000_000) == (
        "about 14 GB at 12k context; too much for a 16 GB machine"
    )


# -- pull: 1.0 store -----------------------------------------------------------------------------


def test_pull_v10_preset_streams_progress(fake_mimoe_v10: FakeMimoe) -> None:
    progress = Progress()
    model = pull_model(_client(fake_mimoe_v10), "qwen3-8b", on_progress=progress)
    assert model.id == "qwen3-8b" and model.ready and model.size_bytes == 5027783488
    assert fake_mimoe_v10.registry_ready["qwen3-8b"] is True
    request = next(r for r in fake_mimoe_v10.requests if r.url.path.endswith("/hf/models/pull"))
    assert request.method == "POST"
    assert request.url.path == f"{STORE}/hf/models/pull"
    assert request.headers["accept"] == "text/event-stream"
    assert request.headers["authorization"] == "Bearer 1234"
    assert json.loads(request.content) == {
        "repo": "Qwen/Qwen3-8B-GGUF",
        "quant": "Q4_K_M",
        "name": "qwen3-8b",
    }
    texts = progress.texts
    assert texts[0] == "resolving Qwen/Qwen3-8B-GGUF (Q4_K_M)"
    assert texts[1] == "resolved Qwen3-8B-Q4_K_M.gguf as qwen3-8b"
    assert [t for t in texts if t.startswith("downloading")] == [
        "downloading 0.00 / 5.03 GB",
        "downloading 2.51 / 5.03 GB",
        "downloading 5.03 / 5.03 GB",
    ]
    assert texts[-1] == "qwen3-8b is ready"
    assert progress.fractions == [0.0, pytest.approx(0.5, abs=1e-6), 1.0, 1.0]
    # the 0.6 two-step route was not used
    assert _posts(fake_mimoe_v10, "/download") == []


def test_pull_v10_custom_spec_names_the_model(fake_mimoe_v10: FakeMimoe) -> None:
    model = pull_model(_client(fake_mimoe_v10), "Qwen/Qwen3-8B-GGUF:Q8_0", on_progress=Progress())
    assert model.id == "qwen3-8b-q8_0" and model.size_bytes == 8709519808
    body = _posts(fake_mimoe_v10, "/hf/models/pull")[0]
    assert body == {"repo": "Qwen/Qwen3-8B-GGUF", "quant": "Q8_0", "name": "qwen3-8b-q8_0"}


def test_pull_v10_repo_only_lets_the_store_pick(fake_mimoe_v10: FakeMimoe) -> None:
    progress = Progress()
    model = pull_model(_client(fake_mimoe_v10), "someone/Solo-Model-GGUF", on_progress=progress)
    assert model.id == "solo-model" and model.ready
    assert _posts(fake_mimoe_v10, "/hf/models/pull")[0] == {
        "repo": "someone/Solo-Model-GGUF",
        "name": "solo-model",
    }
    assert progress.texts[0] == "resolving someone/Solo-Model-GGUF (auto)"


def test_pull_v10_quant_required_lists_the_quants(fake_mimoe_v10: FakeMimoe) -> None:
    with pytest.raises(MimoeError) as info:
        pull_model(_client(fake_mimoe_v10), "Qwen/Qwen3-8B-GGUF", on_progress=Progress())
    assert info.value.message == (
        "Qwen/Qwen3-8B-GGUF needs a quantization (available: Q4_K_M, Q8_0)"
    )
    assert info.value.hint == "run again with Qwen/Qwen3-8B-GGUF:Q4_K_M"
    assert "qwen3-8b" not in fake_mimoe_v10.registry


def test_pull_v10_unknown_quant_and_repo(fake_mimoe_v10: FakeMimoe) -> None:
    client = _client(fake_mimoe_v10)
    with pytest.raises(MimoeError) as info:
        pull_model(client, "Qwen/Qwen3-8B-GGUF:Q2_K", on_progress=Progress())
    assert 'Quantization "Q2_K" not found in Qwen/Qwen3-8B-GGUF' in info.value.message
    assert "Available: Q4_K_M, Q8_0" in info.value.message
    assert info.value.hint == "run again with Qwen/Qwen3-8B-GGUF:Q4_K_M"
    with pytest.raises(MimoeError) as info:
        pull_model(client, "nobody/Nothing-GGUF:Q4_K_M", on_progress=Progress())
    assert "pull of nobody/Nothing-GGUF failed: HTTP 404" in info.value.message
    assert "spelling" in info.value.hint


def test_pull_v10_already_registered_skips_the_store(fake_mimoe_v10: FakeMimoe) -> None:
    progress = Progress()
    model = pull_model(_client(fake_mimoe_v10), "qwen3-4b-instruct-2507", on_progress=progress)
    assert model.id == "qwen3-4b-instruct-2507" and model.ready
    assert progress.events == [("qwen3-4b-instruct-2507 is already in the registry", 1.0)]
    assert _posts(fake_mimoe_v10, "/hf/models/pull") == []


def test_pull_v10_download_error(fake_mimoe_v10: FakeMimoe) -> None:
    fake_mimoe_v10.pull_fail = "SHA-256 mismatch"
    with pytest.raises(MimoeError) as info:
        pull_model(_client(fake_mimoe_v10), "smollm3-3b", on_progress=Progress())
    assert (
        info.value.message
        == "pull of bartowski/HuggingFaceTB_SmolLM3-3B-GGUF failed: SHA-256 mismatch"
    )
    assert fake_mimoe_v10.registry_ready["smollm3-3b"] is False


def test_pull_v10_stream_ending_early_is_an_error(fake_mimoe_v10: FakeMimoe) -> None:
    """A registered-but-not-ready entry after the stream is reported, not returned."""
    fake_mimoe_v10.registry.append("qwen3-8b")
    fake_mimoe_v10.registry_ready["qwen3-8b"] = False  # the store thinks it exists...
    fake_mimoe_v10.hf_repos.clear()  # ...but cannot resolve anything: error stage
    with pytest.raises(MimoeError, match="failed"):
        pull_model(_client(fake_mimoe_v10), "qwen3-8b", on_progress=Progress())


def test_pull_v10_bad_key(fake_mimoe_v10: FakeMimoe) -> None:
    with pytest.raises(MimoeError, match="rejected the API key"):
        pull_model(_client(fake_mimoe_v10, api_key="nope"), "qwen3-8b", on_progress=Progress())


def test_pull_v10_file_form_without_quant_is_refused(fake_mimoe_v10: FakeMimoe) -> None:
    with pytest.raises(MimoeError) as info:
        pull_model(_client(fake_mimoe_v10), "someone/Model:weights.gguf", on_progress=Progress())
    assert "cannot read a quantization from 'weights.gguf'" in info.value.message
    assert "someone/Model:Q4_K_M" in info.value.hint


# -- pull: 0.6 store -----------------------------------------------------------------------------


def test_pull_v06_two_step_registration_and_download(fake_mimoe: FakeMimoe) -> None:
    progress = Progress()
    model = pull_model(_client(fake_mimoe), "qwen3-8b", on_progress=progress)
    assert model.id == "qwen3-8b" and model.ready and model.size_bytes == 5027783488
    posts = _store_posts(fake_mimoe)
    assert [r.url.path for r in posts] == [
        f"{STORE}/models",
        f"{STORE}/models/qwen3-8b/download",
    ]
    assert json.loads(posts[0].content) == {
        "id": "qwen3-8b",
        "version": "1.0.0",
        "kind": "llm",
        "gguf": {"initContextSize": 12000, "initGpuLayerSize": 99},
    }
    assert json.loads(posts[1].content) == {
        "url": "https://huggingface.co/Qwen/Qwen3-8B-GGUF/resolve/main/Qwen3-8B-Q4_K_M.gguf"
    }
    assert posts[1].headers["accept"] == "text/event-stream"
    assert progress.texts[:2] == [
        "registered qwen3-8b",
        "downloading https://huggingface.co/Qwen/Qwen3-8B-GGUF/resolve/main/Qwen3-8B-Q4_K_M.gguf",
    ]
    assert progress.fractions == [0.0, pytest.approx(0.5, abs=1e-6), 1.0, 1.0]
    assert progress.texts[-1] == "qwen3-8b is ready"
    assert not any(r.url.path.endswith("/hf/models/pull") for r in fake_mimoe.requests)


def test_pull_v06_custom_spec_derives_the_file_name(fake_mimoe: FakeMimoe) -> None:
    progress = Progress()
    model = pull_model(
        _client(fake_mimoe), "bartowski/HuggingFaceTB_SmolLM3-3B-GGUF:Q4_K_M", on_progress=progress
    )
    assert model.id == "smollm3-3b"  # matches the preset's repo and quant
    assert _posts(fake_mimoe, "/download")[0]["url"].endswith(
        "/bartowski/HuggingFaceTB_SmolLM3-3B-GGUF/resolve/main/HuggingFaceTB_SmolLM3-3B-Q4_K_M.gguf"
    )
    model = pull_model(_client(fake_mimoe), "someone/Other-GGUF:Q6_K", on_progress=progress)
    assert model.id == "other-q6_k"
    assert _posts(fake_mimoe, "/download")[1]["url"].endswith(
        "/someone/Other-GGUF/resolve/main/Other-Q6_K.gguf"
    )


def test_pull_v06_needs_a_quant(fake_mimoe: FakeMimoe) -> None:
    with pytest.raises(MimoeError) as info:
        pull_model(_client(fake_mimoe), "someone/Solo-Model-GGUF", on_progress=Progress())
    assert "0.6-generation engines need the quantization" in info.value.message
    assert "someone/Solo-Model-GGUF:Q4_K_M" in info.value.hint
    assert _store_posts(fake_mimoe) == []


def test_pull_v06_download_error(fake_mimoe: FakeMimoe) -> None:
    fake_mimoe.pull_fail = "HTTP 404 from huggingface.co"
    with pytest.raises(MimoeError) as info:
        pull_model(_client(fake_mimoe), "qwen3-8b", on_progress=Progress())
    assert info.value.message == "download of qwen3-8b failed: HTTP 404 from huggingface.co"
    assert fake_mimoe.registry_ready["qwen3-8b"] is False
    # a second attempt re-registers (200, upsert) and downloads again
    model = pull_model(_client(fake_mimoe), "qwen3-8b", on_progress=Progress())
    assert model.ready


def test_pull_v06_warns_about_v10_only_presets(fake_mimoe: FakeMimoe) -> None:
    progress = Progress()
    pull_model(_client(fake_mimoe), "qwen3.5-4b", on_progress=progress)
    assert progress.texts[0].startswith("warning: qwen3.5-4b does not load on 0.6-generation")


def test_pull_v06_already_registered_skips_the_store(fake_mimoe: FakeMimoe) -> None:
    progress = Progress()
    model = pull_model(_client(fake_mimoe), "smollm2-360m", on_progress=progress)
    assert model.ready
    assert progress.events == [("smollm2-360m is already in the registry", 1.0)]
    assert _store_posts(fake_mimoe) == []


def test_pull_store_unreachable(fake_mimoe: FakeMimoe) -> None:
    client = _client(fake_mimoe)
    client.discover()
    fake_mimoe.down = True
    with pytest.raises(MimoeError, match="not reachable"):
        pull_model(client, "qwen3-8b", on_progress=Progress())


# -- raw shapes of the fake's new routes ---------------------------------------------------------


def test_fake_store_pull_shapes(fake_mimoe: FakeMimoe, fake_mimoe_v10: FakeMimoe) -> None:
    headers = {"Authorization": "Bearer 1234"}
    store = "http://fake" + STORE
    with fake_mimoe.client() as http:
        created = http.post(
            f"{store}/models",
            json={"id": "x-1", "version": "1.0.0", "kind": "llm"},
            headers=headers,
        )
        again = http.post(
            f"{store}/models",
            json={"id": "x-1", "version": "1.0.0", "kind": "llm"},
            headers=headers,
        )
        bad = http.post(f"{store}/models", json={"id": "a/b", "version": "1", "kind": "llm"})
        download = http.post(
            f"{store}/models/x-1/download", json={"url": "https://h/x/resolve/main/x.gguf"}
        )
        missing = http.post(f"{store}/models/nope/download", json={"url": "https://h/x.gguf"})
        no_hf = http.post(f"{store}/hf/models/pull", json={"repo": "a/b"}, headers=headers)
    assert created.status_code == 201 and created.json()["readyToUse"] is False
    assert again.status_code == 200
    assert bad.status_code == 400 and bad.json()["statusCode"] == 400
    lines = [line for line in download.text.splitlines() if line]
    assert download.headers["content-type"] == "text/event-stream"
    assert lines[0] == 'data: {"size": 0, "totalSize": 1000000000}'
    assert lines[-1] == 'data: {"size": 1000000000, "totalSize": 1000000000}'
    assert missing.status_code == 404 and "not found" in missing.json()["message"]
    assert no_hf.status_code == 404
    with fake_mimoe_v10.client() as http:
        no_repo = http.post(f"{store}/hf/models/pull", json={}, headers=headers)
        bad_repo = http.post(f"{store}/hf/models/pull", json={"repo": "junk"}, headers=headers)
        pulled = http.post(
            f"{store}/hf/models/pull",
            json={"repo": "lmstudio-community/SmolLM2-360M-Instruct-GGUF", "quant": "Q8_0"},
            headers=headers,
        )
    assert no_repo.status_code == 400 and no_repo.json()["error"]["message"] == "repo is required"
    assert bad_repo.status_code == 500 and "owner/name" in bad_repo.json()["error"]["message"]
    events = [
        json.loads(line[6:]) for line in pulled.text.splitlines() if line.startswith("data: ")
    ]
    assert events[0] == {
        "stage": "resolving",
        "repo": "lmstudio-community/SmolLM2-360M-Instruct-GGUF",
        "quant": "Q8_0",
    }
    assert events[1]["stage"] == "resolved"
    assert events[1]["modelId"] == "SmolLM2-360M-Instruct-Q8_0"  # no name: file stem, as the store
    assert events[-1] == {"size": 386404992, "totalSize": 386404992}
    assert "SmolLM2-360M-Instruct-Q8_0" in fake_mimoe_v10.registry


# -- switch_model --------------------------------------------------------------------------------


@pytest.fixture(params=["fake_mimoe", "fake_mimoe_v10"])
def any_fake(request: pytest.FixtureRequest) -> FakeMimoe:
    return request.getfixturevalue(request.param)


def test_switch_model_unloads_the_previous_model(any_fake: FakeMimoe) -> None:
    fake = any_fake
    statuses: list[str] = []
    model = switch_model(
        _client(fake), "smollm2-360m", unload_previous=True, on_status=statuses.append
    )
    assert model.id == "smollm2-360m"
    assert fake.loaded == ["smollm2-360m"]
    assert statuses[:2] == ["unloading qwen3-4b", "loading smollm2-360m"]
    assert statuses[2:] == ["loading model 0%", "loading model 50%", "loading model 100%"]
    unloads = [
        r
        for r in fake.requests
        if r.method == "DELETE" or (r.method == "PUT" and b'"unload"' in r.content)
    ]
    assert len(unloads) == 1
    if fake.generation == "0.6":
        assert unloads[0].method == "DELETE" and unloads[0].url.params["modelId"] == "qwen3-4b"
    else:
        assert json.loads(unloads[0].content) == {"id": "qwen3-4b", "action": "unload"}


def test_switch_model_keeps_the_previous_model(any_fake: FakeMimoe) -> None:
    statuses: list[str] = []
    model = switch_model(
        _client(any_fake),
        f"{NODE_ID}/smollm2-360m",
        unload_previous=False,
        on_status=statuses.append,
    )
    assert model.id == "smollm2-360m"
    assert any_fake.loaded == ["qwen3-4b", "smollm2-360m"]
    assert "unloading qwen3-4b" not in statuses


def test_switch_model_already_loaded(any_fake: FakeMimoe) -> None:
    fake = any_fake
    fake.loaded.append("smollm2-360m")
    statuses: list[str] = []
    model = switch_model(
        _client(fake), "smollm2-360m", unload_previous=True, on_status=statuses.append
    )
    assert model.id == "smollm2-360m"
    assert fake.loaded == ["smollm2-360m"]
    assert statuses == ["unloading qwen3-4b", "smollm2-360m is already loaded"]
    assert not any(r.url.path.endswith("/v1/models") for r in fake.requests if r.method == "POST")
    assert not any(r.method == "PUT" and b'"load"' in r.content for r in fake.requests)


def test_switch_model_leaves_non_llm_models_alone(fake_mimoe_v10: FakeMimoe) -> None:
    fake = fake_mimoe_v10
    fake.loaded.append("embedder")
    fake.kinds["embedder"] = "embed"
    switch_model(_client(fake), "smollm2-360m", unload_previous=True, on_status=lambda _t: None)
    assert fake.loaded == ["embedder", "smollm2-360m"]


def test_switch_model_unregistered_id_hints_at_pull(any_fake: FakeMimoe) -> None:
    with pytest.raises(MimoeError) as info:
        switch_model(_client(any_fake), "qwen3-8b", unload_previous=True, on_status=lambda _t: None)
    assert info.value.message == (
        "'qwen3-8b' is not in the model registry "
        "(registered: qwen3-4b, qwen3-4b-instruct-2507, smollm2-360m)"
    )
    assert "mimoe-agent models pull qwen3-8b" in info.value.hint
    assert "owner/repo:QUANT" in info.value.hint
    assert any_fake.loaded == ["qwen3-4b"]  # nothing was unloaded before the check failed


def test_switch_model_not_downloaded(fake_mimoe_v10: FakeMimoe) -> None:
    fake_mimoe_v10.registry.append("qwen3-8b")
    fake_mimoe_v10.registry_ready["qwen3-8b"] = False
    with pytest.raises(MimoeError) as info:
        switch_model(
            _client(fake_mimoe_v10), "qwen3-8b", unload_previous=True, on_status=lambda _t: None
        )
    assert info.value.message == (
        "'qwen3-8b' is registered but its file is not downloaded (status downloading)"
    )
    assert "models pull qwen3-8b" in info.value.hint


def test_switch_model_tolerates_unload_failure(any_fake: FakeMimoe) -> None:
    fake = any_fake
    fake.unload_error = (500, "sessions still active")
    statuses: list[str] = []
    model = switch_model(
        _client(fake), "smollm2-360m", unload_previous=True, on_status=statuses.append
    )
    assert model.id == "smollm2-360m"
    assert fake.loaded == ["qwen3-4b", "smollm2-360m"]
    assert statuses[0] == "unloading qwen3-4b"
    assert statuses[1].startswith("could not unload qwen3-4b: mimOE returned HTTP 500")
    assert "sessions still active" in statuses[1]
    assert statuses[2] == "loading smollm2-360m"


def test_switch_model_warns_about_big_models(fake_mimoe: FakeMimoe) -> None:
    fake_mimoe.registry.append("qwen3-8b")
    fake_mimoe.registry_sizes["qwen3-8b"] = 5027783488
    statuses: list[str] = []
    switch_model(_client(fake_mimoe), "qwen3-8b", unload_previous=False, on_status=statuses.append)
    assert statuses[0] == (
        "warning: qwen3-8b is about 7 GB at 12k context; tight on a 16 GB machine"
    )
    fake_mimoe.registry.append("qwen3.5-9b")
    fake_mimoe.registry_sizes["qwen3.5-9b"] = None  # linked by local path: preset size instead
    statuses.clear()
    switch_model(
        _client(fake_mimoe), "qwen3.5-9b", unload_previous=False, on_status=statuses.append
    )
    assert statuses[0].startswith("warning: qwen3.5-9b is about 7 GB")


def test_switch_model_load_failure_names_the_unloaded_model(fake_mimoe: FakeMimoe) -> None:
    fake = fake_mimoe

    def sabotage(text: str) -> None:
        if text == "loading smollm2-360m":
            fake.registry.remove("smollm2-360m")  # the load will now answer 404

    with pytest.raises(MimoeError) as info:
        switch_model(_client(fake), "smollm2-360m", unload_previous=True, on_status=sabotage)
    assert "could not find" in info.value.message
    assert "qwen3-4b was unloaded before the failure" in info.value.hint
    assert "mimoe-agent models use qwen3-4b" in info.value.hint
    assert fake.loaded == []


# -- ids, stream parsing, timeouts and callbacks (review regressions) ---------------------------


class Boom(RuntimeError):
    """Raised by a misbehaving progress callback."""


@pytest.mark.parametrize(
    "spec",
    [
        "a/..",  # id ".." (a path segment the stores refuse)
        "a/.hidden",
        "a/-flag",
        "a/b..c",
        "a/b--c",
        ".a/b",
        "a-/b",
        "a/b.",
        "a/gguf",  # empty stem -> empty id
        "a/GGUF:Q4_K_M",  # empty stem -> id "-q4_k_m"
        "a/b:-Q4",
        "a/b:..gguf",
    ],
)
def test_parse_spec_refuses_ids_the_stores_reject(spec: str) -> None:
    with pytest.raises(MimoeError) as info:
        parse_spec(spec)
    assert "owner/repo:QUANT" in info.value.hint


def test_valid_model_id_mirrors_the_stores() -> None:
    assert parse_spec("a_b/c_d").model_id == "c_d"
    for good in ("qwen3.5-4b", "_x", "a.b-c_d", "x" * 255):
        assert valid_model_id(good), good
    for bad in ("", "..", ".x", "-x", "a..b", "a/b", "a b", "x" * 256):
        assert not valid_model_id(bad), bad


def test_pull_never_sends_a_rejected_id(any_fake: FakeMimoe) -> None:
    with pytest.raises(MimoeError):
        pull_model(_client(any_fake), "someone/..:Q4_K_M", on_progress=Progress())
    assert any_fake.requests == []


def test_sse_reader_joins_multiline_data_and_accepts_bare_json() -> None:
    lines = [
        "event: progress",
        "id: 7",
        "retry: 1000",
        ": keep-alive",
        'data: {"size": 1,',
        'data:  "totalSize": 2}',
        "",
        "data: [DONE]",
        "",
        "not json at all",
        '{"message": "boom", "statusCode": 500}',
        'data:{"tail": true}',
    ]
    assert list(_iter_sse_objects(lines)) == [
        {"size": 1, "totalSize": 2},
        {"message": "boom", "statusCode": 500},
        {"tail": True},
    ]
    # CRLF / CR separators (httpx splits them) and a final bare line without a newline
    body = b'data: {"a": 1}\r\n\r\ndata: {"b": 2}\r\r\n{"statusCode": 500, "message": "x"}'
    assert list(_iter_sse_objects(httpx.Response(200, content=body).iter_lines())) == [
        {"a": 1},
        {"b": 2},
        {"statusCode": 500, "message": "x"},
    ]


def test_pull_reads_crlf_streams(any_fake: FakeMimoe) -> None:
    any_fake.sse_newline = "\r\n"
    progress = Progress()
    model = pull_model(_client(any_fake), "qwen3-8b", on_progress=progress)
    assert model.ready and progress.fractions[-1] == 1.0
    assert "downloading 5.03 / 5.03 GB" in progress.texts
    any_fake.registry_ready["qwen3-8b"] = False
    any_fake.pull_fail = "boom"
    with pytest.raises(MimoeError, match="boom"):
        pull_model(_client(any_fake), "qwen3-8b", on_progress=Progress())


def test_pull_stream_has_a_connect_timeout_only(any_fake: FakeMimoe) -> None:
    pull_model(_client(any_fake), "qwen3-8b", on_progress=Progress())
    request = next(
        r for r in any_fake.requests if r.url.path.endswith(("/hf/models/pull", "/download"))
    )
    assert request.extensions["timeout"] == {
        "connect": 5.0,
        "read": None,
        "write": None,
        "pool": None,
    }
    assert PULL_TIMEOUT.read is None and PULL_TIMEOUT.connect == 5.0


def test_pull_reports_a_stream_cut_by_the_server(any_fake: FakeMimoe) -> None:
    any_fake.pull_drop = True
    with pytest.raises(MimoeError) as info:
        pull_model(_client(any_fake), "qwen3-8b", on_progress=Progress())
    assert info.value.message.startswith("the model store closed the stream before the pull")
    assert any_fake.registry_ready["qwen3-8b"] is False


def test_pull_lets_a_failing_progress_callback_propagate(any_fake: FakeMimoe) -> None:
    def explode(text: str, fraction: float | None) -> None:
        if text.startswith("downloading 2."):
            raise Boom(text)

    with pytest.raises(Boom):
        pull_model(_client(any_fake), "qwen3-8b", on_progress=explode)
    last = any_fake.requests[-1]  # the pull stopped where the callback failed
    assert last.method == "POST" and last.url.path.endswith(("/hf/models/pull", "/download"))


def test_switch_model_lets_a_failing_status_callback_propagate(any_fake: FakeMimoe) -> None:
    def explode(text: str) -> None:
        if text.startswith("loading model"):
            raise Boom(text)

    with pytest.raises(Boom):
        switch_model(_client(any_fake), "smollm2-360m", unload_previous=True, on_status=explode)


def test_switch_model_unloads_every_other_llm(any_fake: FakeMimoe) -> None:
    fake = any_fake
    fake.loaded.append("qwen3-4b-instruct-2507")
    statuses: list[str] = []
    model = switch_model(
        _client(fake), "smollm2-360m", unload_previous=True, on_status=statuses.append
    )
    assert model.id == "smollm2-360m"
    assert fake.loaded == ["smollm2-360m"]
    assert statuses[:3] == [
        "unloading qwen3-4b",
        "unloading qwen3-4b-instruct-2507",
        "loading smollm2-360m",
    ]


def test_pull_v10_resumes_a_registered_but_unfinished_entry(fake_mimoe_v10: FakeMimoe) -> None:
    fake = fake_mimoe_v10
    fake.registry.append("qwen3-8b")
    fake.registry_ready["qwen3-8b"] = False  # a pull that was interrupted
    progress = Progress()
    model = pull_model(_client(fake), "qwen3-8b", on_progress=progress)
    assert model.ready and fake.registry_ready["qwen3-8b"] is True
    assert "resolved Qwen3-8B-Q4_K_M.gguf as qwen3-8b" in progress.texts
    assert not any("already in the registry" in text for text in progress.texts)


def test_switch_model_memory_note_uses_the_entry_context(fake_mimoe_v10: FakeMimoe) -> None:
    fake = fake_mimoe_v10
    fake.registry.append("qwen3-8b")
    fake.registry_sizes["qwen3-8b"] = 5027783488
    fake.registry_context["qwen3-8b"] = 32768  # what a 1.0 HuggingFace pull registers
    statuses: list[str] = []
    switch_model(_client(fake), "qwen3-8b", unload_previous=False, on_status=statuses.append)
    assert statuses[0] == "warning: qwen3-8b is about 9 GB at 33k context; tight on a 16 GB machine"
