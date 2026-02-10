from fastapi.testclient import TestClient

from api.service import app


def test_health_endpoint():
    client = TestClient(app)
    resp = client.get("/health")
    assert resp.status_code == 200
    assert "status" in resp.json()
