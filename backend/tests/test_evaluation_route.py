import json
from fastapi.testclient import TestClient

from backend.app import create_app
from backend.web import create_web_app


def test_evaluation_unavailable_without_build(tmp_path, monkeypatch):
    monkeypatch.setenv("EVALUATION_REPORT_PATH", str(tmp_path / "missing.json"))
    app = create_web_app(tmp_path / "public", api=create_app(db_path=str(tmp_path / "docs.db")))
    result = TestClient(app).get("/api/evaluation")
    assert result.status_code == 503


def test_evaluation_route_serves_static_prebuilt_json(tmp_path, monkeypatch):
    report = tmp_path / "evaluation.json"
    report.write_text(json.dumps({"schema_version": 1, "summary": {"cases": 2}, "cases": []}))
    monkeypatch.setenv("EVALUATION_REPORT_PATH", str(report))
    app = create_web_app(tmp_path / "public", api=create_app(db_path=str(tmp_path / "docs.db")))
    client = TestClient(app)
    result = client.get("/api/evaluation")
    assert result.status_code == 200
    assert result.json()["summary"]["cases"] == 2
    assert result.headers["content-type"].startswith("application/json")
    assert "max-age=300" in result.headers["cache-control"]
