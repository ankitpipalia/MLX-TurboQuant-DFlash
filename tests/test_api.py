from fastapi.testclient import TestClient

from local_llm_control.api import app


def test_dashboard_is_served_without_external_assets() -> None:
    response = TestClient(app).get("/")
    assert response.status_code == 200
    assert "Local LLM Control" in response.text
    assert "http://" not in response.text
    assert "https://" not in response.text


def test_dashboard_exposes_normal_macos_release_button() -> None:
    client = TestClient(app)
    dashboard = client.get("/")
    script = client.get("/static/app.js")

    assert dashboard.status_code == 200
    assert 'id="normalMacosButton"' in dashboard.text
    assert "Normal macOS — Release GPU" in dashboard.text
    assert "data-normal-macos-button" in dashboard.text
    assert script.status_code == 200
    assert "/api/system/llm-mode/normal" in script.text


def test_monitor_exposes_unified_memory_views_and_profile_baselines() -> None:
    client = TestClient(app)
    snapshot = client.get("/api/dashboard/snapshot")
    assert snapshot.status_code == 200
    assert {
        "mem_free_mib",
        "mem_available_mib",
        "vram_free_mib",
        "metal_headroom_mib",
    } <= snapshot.json().keys()

    baselines = client.get("/api/dashboard/profile-memory-baselines")
    assert baselines.status_code == 200
    names = {row["profile"] for row in baselines.json()["profiles"]}
    assert names == {"35B Q4_K_P default", "27B Q6_K_P default"}
