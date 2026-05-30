"""
Tests for the FastAPI inference service.
"""

from fastapi.testclient import TestClient

from api.main import app
from data.generate import generate_synthetic_data


client = TestClient(app)


def sample_record() -> dict:
    """Return one valid synthetic employee payload."""
    row = generate_synthetic_data(2).iloc[0].to_dict()
    return {
        "employee_id": int(row["employee_id"]),
        "department": row["department"],
        "job_title": row["job_title"],
        "salary": float(row["salary"]),
        "engagement_score": float(row["engagement_score"]),
        "years_at_company": float(row["years_at_company"]),
        "manager_id": int(row["manager_id"]),
        "level": row["level"],
        "comp_vs_band_pct": float(row["comp_vs_band_pct"]),
        "months_since_promo": int(row["months_since_promo"]),
        "perf_trend": float(row["perf_trend"]),
        "pto_usage": float(row["pto_usage"]),
        "one_on_one_freq": float(row["one_on_one_freq"]),
        "internal_moves": int(row["internal_moves"]),
    }


def test_health_endpoint():
    response = client.get("/health")

    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "ok"
    assert payload["model_loaded"] is True


def test_metadata_and_schema_endpoints():
    metadata_response = client.get("/metadata")
    schema_response = client.get("/schema")

    assert metadata_response.status_code == 200
    assert schema_response.status_code == 200
    assert metadata_response.json()["metadata"]["model_version"] == "v1"
    assert "feature_columns" in schema_response.json()["schema"]
    assert "department" in schema_response.json()["vocabularies"]


def test_predict_endpoint():
    response = client.post("/predict", json={"records": [sample_record()]})

    assert response.status_code == 200
    prediction = response.json()["predictions"][0]
    assert prediction["model_version"] == "v1"
    assert 0 <= prediction["risk_score"] <= 1
    assert prediction["risk_level"] in {"Low", "Medium", "High"}


def test_explain_endpoint():
    response = client.post("/explain", json={"record": sample_record(), "top_n": 3})

    assert response.status_code == 200
    payload = response.json()
    assert 0 <= payload["risk_score"] <= 1
    assert len(payload["top_drivers"]) == 3
