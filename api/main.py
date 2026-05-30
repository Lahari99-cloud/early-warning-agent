"""
FastAPI service for model inference and artifact inspection.
"""

from __future__ import annotations

import logging
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException

from api.schemas import (
    DriverContribution,
    ExplainRequest,
    ExplainResponse,
    HealthResponse,
    MetadataResponse,
    PredictionRequest,
    PredictionResponse,
    PredictionResult,
    SchemaResponse,
    VersionsResponse,
)
try:
    from config import RISK_HIGH_THRESHOLD, RISK_MEDIUM_THRESHOLD
except ImportError:
    RISK_MEDIUM_THRESHOLD = 0.40
    RISK_HIGH_THRESHOLD = 0.70
from model.artifacts import ARTIFACT_ROOT, artifact_path, read_active_version, read_json
from model.explain import explain_prediction
from model.features import load_category_vocabularies, load_feature_columns, prepare_features
from model.train import load_artifact_metadata, load_model


logger = logging.getLogger(__name__)

app = FastAPI(
    title="Attrition Early-Warning Agent API",
    version="1.0.0",
    description="Registry-backed inference API for synthetic employee attrition risk.",
)


def risk_level(score: float) -> str:
    """Map a risk score to a stable display band."""
    if score >= RISK_HIGH_THRESHOLD:
        return "High"
    if score >= RISK_MEDIUM_THRESHOLD:
        return "Medium"
    return "Low"


@lru_cache(maxsize=1)
def get_runtime() -> dict:
    """Load model runtime artifacts once per process."""
    metadata = load_artifact_metadata()
    model = load_model()
    return {
        "model": model,
        "metadata": metadata,
        "feature_columns": load_feature_columns(),
        "vocabularies": load_category_vocabularies(),
        "active_version": read_active_version(),
    }


def records_to_frame(records) -> pd.DataFrame:
    """Convert Pydantic employee records into a DataFrame."""
    return pd.DataFrame([record.model_dump(exclude_none=True) for record in records])


def predict_frame(df: pd.DataFrame) -> np.ndarray:
    """Run registry-backed feature prep and prediction."""
    runtime = get_runtime()
    try:
        X, _ = prepare_features(
            df,
            label_encoders=runtime["vocabularies"],
            training_columns=runtime["feature_columns"],
        )
        return runtime["model"].predict_proba(X)[:, 1]
    except Exception as exc:
        logger.exception("Prediction failed")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {exc}") from exc


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    """Return service and active model health."""
    try:
        runtime = get_runtime()
        return HealthResponse(
            status="ok",
            active_version=runtime["active_version"],
            model_loaded=runtime["model"] is not None,
        )
    except Exception as exc:
        logger.exception("Health check failed")
        raise HTTPException(status_code=503, detail=f"Service unhealthy: {exc}") from exc


@app.get("/metadata", response_model=MetadataResponse)
def metadata() -> MetadataResponse:
    """Return active model artifact metadata."""
    return MetadataResponse(metadata=get_runtime()["metadata"])


@app.get("/versions", response_model=VersionsResponse)
def versions() -> VersionsResponse:
    """Return available artifact versions and the active version."""
    root = Path(ARTIFACT_ROOT)
    available = sorted(path.name for path in root.iterdir() if path.is_dir()) if root.exists() else []
    return VersionsResponse(active_version=read_active_version(), versions=available)


@app.get("/schema", response_model=SchemaResponse)
def schema() -> SchemaResponse:
    """Return active feature schema and category vocabularies."""
    schema_path = artifact_path("schema.json", read_active_version())
    schema_payload = read_json(schema_path) if schema_path.exists() else {
        "feature_columns": get_runtime()["feature_columns"],
    }
    return SchemaResponse(
        feature_schema=schema_payload,
        vocabularies=get_runtime()["vocabularies"],
    )


@app.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest) -> PredictionResponse:
    """Predict attrition risk for one or more employees."""
    df = records_to_frame(request.records)
    scores = predict_frame(df)
    version = get_runtime()["active_version"]

    predictions = []
    for idx, (record, score) in enumerate(zip(request.records, scores)):
        score_value = float(score)
        predictions.append(
            PredictionResult(
                index=idx,
                employee_id=record.employee_id,
                risk_score=score_value,
                risk_level=risk_level(score_value),
                model_version=version,
            )
        )
    return PredictionResponse(predictions=predictions)


@app.post("/explain", response_model=ExplainResponse)
def explain(request: ExplainRequest) -> ExplainResponse:
    """Return SHAP top drivers for a single employee."""
    df = records_to_frame([request.record])
    score = float(predict_frame(df)[0])

    try:
        explanation, _, _ = explain_prediction(get_runtime()["model"], df)
    except Exception as exc:
        logger.exception("Explanation failed")
        raise HTTPException(status_code=500, detail=f"Explanation failed: {exc}") from exc

    shap_values = explanation["shap_values"]
    if len(shap_values.shape) == 2:
        shap_values = shap_values[0]

    feature_names = explanation["feature_names"]
    top_idx = np.argsort(np.abs(shap_values))[-request.top_n:][::-1]
    drivers = [
        DriverContribution(feature=feature_names[i], shap_value=float(shap_values[i]))
        for i in top_idx
    ]

    return ExplainResponse(
        employee_id=request.record.employee_id,
        risk_score=score,
        risk_level=risk_level(score),
        top_drivers=drivers,
        model_version=get_runtime()["active_version"],
    )
