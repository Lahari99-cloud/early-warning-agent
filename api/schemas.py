"""
Typed request and response schemas for the inference API.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field


class EmployeeRecord(BaseModel):
    """Raw employee features accepted by the model service."""

    department: str
    job_title: str
    salary: float = Field(gt=0)
    engagement_score: float = Field(ge=0, le=100)
    years_at_company: float = Field(ge=0)
    manager_id: int = Field(ge=0)
    level: str
    comp_vs_band_pct: float = Field(ge=0)
    months_since_promo: int = Field(ge=0)
    perf_trend: float = Field(ge=-1, le=1)
    pto_usage: float = Field(ge=0, le=100)
    one_on_one_freq: float = Field(ge=0)
    internal_moves: int = Field(ge=0)
    employee_id: Optional[int] = None
    first_name: Optional[str] = None
    last_name: Optional[str] = None


class PredictionRequest(BaseModel):
    """Batch prediction request."""

    records: List[EmployeeRecord] = Field(min_length=1, max_length=500)


class PredictionResult(BaseModel):
    """Single employee prediction output."""

    index: int
    employee_id: Optional[int] = None
    risk_score: float
    risk_level: str
    model_version: str


class PredictionResponse(BaseModel):
    """Batch prediction response."""

    predictions: List[PredictionResult]


class ExplainRequest(BaseModel):
    """Single-row explanation request."""

    record: EmployeeRecord
    top_n: int = Field(default=5, ge=1, le=20)


class DriverContribution(BaseModel):
    """Feature contribution in a local explanation."""

    feature: str
    shap_value: float


class ExplainResponse(BaseModel):
    """SHAP explanation response."""

    employee_id: Optional[int] = None
    risk_score: float
    risk_level: str
    top_drivers: List[DriverContribution]
    model_version: str


class HealthResponse(BaseModel):
    """Service health response."""

    status: str
    active_version: str
    model_loaded: bool


class MetadataResponse(BaseModel):
    """Artifact metadata response."""

    metadata: Dict[str, Any]


class VersionsResponse(BaseModel):
    """Available model versions response."""

    active_version: str
    versions: List[str]


class SchemaResponse(BaseModel):
    """Feature schema and category vocabulary response."""

    model_config = ConfigDict(populate_by_name=True)

    feature_schema: Dict[str, Any] = Field(alias="schema")
    vocabularies: Dict[str, List[str]]
