from fastapi.testclient import TestClient

from openphonic.core.settings import get_settings
from openphonic.main import create_app


def test_index_route_renders(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("OPENPHONIC_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("OPENPHONIC_DATABASE_PATH", str(tmp_path / "data" / "openphonic.sqlite3"))
    monkeypatch.setenv("OPENPHONIC_PIPELINE_CONFIG", "config/default.yml")
    get_settings.cache_clear()

    with TestClient(create_app()) as client:
        response = client.get("/")

    assert response.status_code == 200
    assert "Openphonic" in response.text
