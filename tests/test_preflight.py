from pathlib import Path

from openphonic.core.settings import Settings
from openphonic.pipeline.config import PipelineConfig
from openphonic.pipeline.preflight import pipeline_preflight_issues


def test_default_pipeline_has_no_preflight_issues(tmp_path, monkeypatch) -> None:
    settings = make_settings(tmp_path)

    issues = pipeline_preflight_issues(
        PipelineConfig(name="default", stages={"loudness": {"enabled": True}}),
        settings,
    )

    assert issues == []


def test_preflight_reports_missing_deepfilternet_cli(tmp_path, monkeypatch) -> None:
    settings = make_settings(tmp_path, deepfilternet_bin="deepFilter")
    monkeypatch.setattr("openphonic.pipeline.preflight._binary_available", lambda binary: False)

    issues = pipeline_preflight_issues(
        PipelineConfig(name="cleanup", stages={"noise_reduction": {"enabled": True}}),
        settings,
    )

    assert [(issue.stage, issue.message) for issue in issues] == [
        (
            "noise_reduction",
            (
                "DeepFilterNet noise reduction is enabled, but the deepFilter CLI was "
                "not found on PATH."
            ),
        )
    ]


def test_preflight_reports_missing_ml_python_dependencies(tmp_path, monkeypatch) -> None:
    settings = make_settings(tmp_path, hf_token=None)
    monkeypatch.setattr("openphonic.pipeline.preflight._module_available", lambda module: False)

    issues = pipeline_preflight_issues(
        PipelineConfig(
            name="ml",
            stages={
                "music_separation": {
                    "enabled": True,
                    "model": "htdemucs",
                    "stem": "vocals",
                },
                "transcription": {"enabled": True},
                "diarization": {"enabled": True},
            },
        ),
        settings,
    )

    messages = [issue.message for issue in issues]
    assert any("demucs is not installed" in message for message in messages)
    assert any("faster-whisper is not installed" in message for message in messages)
    assert any("Diarization requires HF_TOKEN" in message for message in messages)
    assert any("pyannote.audio is not installed" in message for message in messages)


def test_preflight_validates_intro_outro_assets(tmp_path) -> None:
    settings = make_settings(tmp_path)
    config_path = tmp_path / "presets" / "show.yml"
    config_path.parent.mkdir()

    issues = pipeline_preflight_issues(
        PipelineConfig(
            name="show",
            source_path=config_path,
            stages={"intro_outro": {"enabled": True, "intro_path": "missing.wav"}},
        ),
        settings,
    )

    assert len(issues) == 1
    assert issues[0].stage == "intro_outro"
    assert "intro_path does not exist" in issues[0].message


def test_preflight_rejects_filler_suggestions_without_transcription(tmp_path) -> None:
    settings = make_settings(tmp_path)

    issues = pipeline_preflight_issues(
        PipelineConfig(name="cuts", stages={"filler_removal": {"enabled": True}}),
        settings,
    )

    assert len(issues) == 1
    assert issues[0].stage == "filler_removal"
    assert "require stages.transcription" in issues[0].message


def make_settings(
    tmp_path: Path,
    *,
    hf_token: str | None = "hf_test",
    deepfilternet_bin: str = "deepFilter",
) -> Settings:
    return Settings(
        data_dir=tmp_path / "data",
        database_path=tmp_path / "data" / "openphonic.sqlite3",
        pipeline_config=tmp_path / "config.yml",
        preset_dir=tmp_path / "presets",
        max_upload_mb=1024,
        retention_days=0,
        public_base_url="http://127.0.0.1:8000",
        hf_token=hf_token,
        whisper_model="small",
        whisper_device="auto",
        pyannote_model="pyannote/speaker-diarization-3.1",
        deepfilternet_bin=deepfilternet_bin,
    )
