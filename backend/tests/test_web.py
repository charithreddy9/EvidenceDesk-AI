from fastapi.testclient import TestClient
from backend.app import create_app
from backend.web import create_web_app

def test_single_origin_frontend_and_api(tmp_path):
    root = tmp_path / "dist"
    root.mkdir()
    (root / "index.html").write_text("<html>SPA works</html>")
    (root / "assets").mkdir()
    (root / "assets" / "index.js").write_text("console.log('ok')")
    api = create_app(db_path=str(tmp_path / "demo.sqlite3"))
    client = TestClient(create_web_app(root, api=api))
    assert client.get("/").status_code == 200
    assert "SPA works" in client.get("/").text
    assert client.get("/assets/index.js").status_code == 200
    assert client.get("/api/health").json()["status"] == "ok"
    assert client.get("/project").status_code == 200
    assert client.get("/api/nonexistent").status_code == 404
    assert client.get("/missing.js").status_code == 404

def test_api_mode_without_frontend(tmp_path):
    api = create_app(db_path=str(tmp_path / "demo.sqlite3"))
    client = TestClient(create_web_app(tmp_path / "missing", api=api))
    assert client.get("/api/health").status_code == 200
    assert client.get("/").status_code == 404
