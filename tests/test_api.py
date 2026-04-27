from pathlib import Path

from fastapi.testclient import TestClient

from openphonic.core.database import create_job, init_db
from openphonic.core.settings import get_settings
from openphonic.main import create_app
from openphonic.services.storage import job_dir


def configure_tmp_settings(tmp_path, monkeypatch) -> Path:
    data_dir = tmp_path / "data"
    db_path = data_dir / "openphonic.sqlite3"
    monkeypatch.setenv("OPENPHONIC_DATA_DIR", str(data_dir))
    monkeypatch.setenv("OPENPHONIC_DATABASE_PATH", str(db_path))
    monkeypatch.setenv("OPENPHONIC_PIPELINE_CONFIG", "config/default.yml")
    get_settings.cache_clear()
    init_db(db_path)
    return db_path


def test_index_route_renders(tmp_path, monkeypatch) -> None:
    configure_tmp_settings(tmp_path, monkeypatch)

    with TestClient(create_app()) as client:
        response = client.get("/")

    assert response.status_code == 200
    assert "Openphonic" in response.text


def test_job_artifact_routes_list_and_serve_files(tmp_path, monkeypatch) -> None:
    db_path = configure_tmp_settings(tmp_path, monkeypatch)
    input_path = tmp_path / "input.wav"
    input_path.write_bytes(b"audio")
    create_job(db_path, job_id="job-1", original_filename="input.wav", input_path=input_path)
    work_dir = job_dir(get_settings(), "job-1")
    (work_dir / "job-events.jsonl").write_text('{"event":"job.started"}\n', encoding="utf-8")
    (work_dir / "commands.jsonl").write_text('{"event":"process.started"}\n', encoding="utf-8")
    (work_dir / "00_media_metadata.json").write_text('{"format_name":"wav"}', encoding="utf-8")
    (work_dir / "pipeline_manifest.json").write_text('{"status":"failed"}', encoding="utf-8")

    with TestClient(create_app()) as client:
        list_response = client.get("/api/jobs/job-1/artifacts")
        manifest_response = client.get("/api/jobs/job-1/manifest")
        events_response = client.get("/api/jobs/job-1/events")
        page_response = client.get("/jobs/job-1")

    assert list_response.status_code == 200
    artifacts = list_response.json()
    assert [artifact["name"] for artifact in artifacts] == [
        "00_media_metadata.json",
        "commands.jsonl",
        "job-events.jsonl",
        "pipeline_manifest.json",
    ]
    assert artifacts[0]["url"] == "/api/jobs/job-1/artifacts/00_media_metadata.json"
    assert manifest_response.status_code == 200
    assert manifest_response.json() == {"status": "failed"}
    assert events_response.status_code == 200
    assert "job.started" in events_response.text
    assert page_response.status_code == 200
    assert "Pipeline manifest" in page_response.text
    assert "job-events.jsonl" in page_response.text


def test_artifact_download_rejects_missing_and_traversal_paths(tmp_path, monkeypatch) -> None:
    db_path = configure_tmp_settings(tmp_path, monkeypatch)
    input_path = tmp_path / "input.wav"
    input_path.write_bytes(b"audio")
    create_job(db_path, job_id="job-1", original_filename="input.wav", input_path=input_path)
    job_dir(get_settings(), "job-1")

    with TestClient(create_app()) as client:
        missing_response = client.get("/api/jobs/job-1/artifacts/missing.json")
        traversal_response = client.get("/api/jobs/job-1/artifacts/%2E%2E/openphonic.sqlite3")

    assert missing_response.status_code == 404
    assert traversal_response.status_code == 400
