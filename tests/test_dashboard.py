"""
Tests for dashboard display helpers.
"""

import pandas as pd

from app.dashboard import (
    build_display_table,
    build_fallback_prep_note,
    build_person_explanation,
    finalize_prep_note,
    get_risk_badge,
    risk_band_description,
)
from data.generate import load_data_from_parquet
from model.explain import explain_prediction
from model.features import load_category_vocabularies, load_feature_columns, prepare_features
from model.train import load_model


def test_get_risk_badge():
    assert get_risk_badge(0.7) == "🔴 High"
    assert get_risk_badge(0.4) == "🟠 Medium"
    assert get_risk_badge(0.39) == "🟢 Low"
    assert risk_band_description() == "Low: < 0.40 | Medium: 0.40-0.69 | High: >= 0.70"


def test_build_display_table_formats_decimal_score_and_badge():
    data = pd.DataFrame({
        "employee_id": [1],
        "name": ["Test User"],
        "risk_score": [0.756],
    })

    display = build_display_table(data)

    assert list(display.columns) == ["ID", "Name", "Model Score", "Review Band"]
    assert display.iloc[0]["Model Score"] == "0.76"
    assert display.iloc[0]["Review Band"] == "🔴 High"


def test_build_person_explanation_mentions_employee_and_drivers():
    employee = pd.Series({
        "first_name": "Sarah",
        "risk_score": 0.89,
    })
    shap_df = pd.DataFrame({
        "Feature": ["Comp Vs Band Pct", "Perf Trend", "Months Since Promo"],
        "SHAP Value": ["-0.420", "+0.280", "+0.180"],
    })

    explanation = build_person_explanation(employee, shap_df)

    assert "Sarah" in explanation
    assert "89.0%" in explanation
    assert "Comp Vs Band Pct" in explanation
    assert "conversation prompts" in explanation


def test_fallback_note_uses_review_drivers_and_complete_talking_points():
    note = build_fallback_prep_note(
        {
            "first_name": "Matthew",
            "last_name": "Jackson",
            "name": "Matthew Jackson",
            "risk_score": 0.84,
        },
        {
            "feature_contributions": {
                "engagement_score": 0.92,
                "pto_usage": 0.79,
                "comp_vs_band_pct": 0.59,
            }
        },
    )

    assert "largest score-raising review drivers" in note
    assert "largestframework" not in note
    assert "3. Ask whether compensation" in note
    assert "SUGGESTED ACTION:" in note


def test_finalize_prep_note_replaces_damaged_or_incomplete_note():
    fallback_note = build_fallback_prep_note(
        {
            "first_name": "Matthew",
            "last_name": "Jackson",
            "name": "Matthew Jackson",
            "risk_score": 0.84,
        },
        {"feature_contributions": {"engagement_score": 0.92}},
    )
    damaged_note = """DRAFT - FOR HUMAN REVIEW ONLY

LIKELY ISSUE:
Matthew Jackson's largestframework drivers are Engagement Score (+0.92).

TALKING POINTS:
1. Explore what parts of the role feel energizing.
2. Discuss workload balance.
3."""

    finalized = finalize_prep_note(damaged_note, fallback_note)

    assert finalized == fallback_note


def test_low_band_fallback_note_stays_neutral():
    note = build_fallback_prep_note(
        {"first_name": "Taylor", "last_name": "Lee", "risk_score": 0.21},
        {"feature_contributions": {"perf_trend": -0.8, "engagement_score": -0.4}},
    )

    assert "Low review band" in note
    assert "No elevated review signal" in note
    assert "largest score-raising" not in note
    assert "Ask what is going well" in note
    assert "Confirm goals and priorities" in note


def test_selected_employee_row_can_be_explained():
    df = load_data_from_parquet()
    model = load_model()
    X, _ = prepare_features(
        df,
        load_category_vocabularies(),
        training_columns=load_feature_columns(),
    )
    df = df.copy()
    df["name"] = df["first_name"] + " " + df["last_name"]
    df["risk_score"] = model.predict_proba(X)[:, 1]
    selected_employee = df.sort_values("risk_score", ascending=False).iloc[0]

    _, shap_df, _ = explain_prediction(model, selected_employee.to_frame().T)

    assert shap_df.shape[1] == len(load_feature_columns())
    assert shap_df.abs().sum(axis=1).iloc[0] > 0
