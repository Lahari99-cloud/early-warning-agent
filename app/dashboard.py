""" Streamlit dashboard for the attrition early-warning agent. """
import sys
import os
import logging

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import streamlit as st
import pandas as pd
import numpy as np
from data.generate import load_data_from_parquet
from model.train import load_model
from model.features import load_category_vocabularies, load_feature_columns, prepare_features
from model.explain import explain_prediction
from agent.prep_note import draft_prep_note
from agent.guardrails import validate_prep_note, sanitize_output
try:
    from config import RISK_HIGH_THRESHOLD, RISK_MEDIUM_THRESHOLD
except ImportError:
    RISK_MEDIUM_THRESHOLD = 0.40
    RISK_HIGH_THRESHOLD = 0.70

logger = logging.getLogger(__name__)

def get_risk_badge(score: float) -> str:
    """Return the visual risk badge for a score."""
    if score >= RISK_HIGH_THRESHOLD:
        return "🔴 High"
    if score >= RISK_MEDIUM_THRESHOLD:
        return "🟠 Medium"
    return "🟢 Low"

def risk_band_description() -> str:
    """Return the human-readable score bands shown in the dashboard."""
    return (
        f"Low: < {RISK_MEDIUM_THRESHOLD:.2f} | "
        f"Medium: {RISK_MEDIUM_THRESHOLD:.2f}-{RISK_HIGH_THRESHOLD - 0.01:.2f} | "
        f"High: >= {RISK_HIGH_THRESHOLD:.2f}"
    )

def format_feature_name(feature: str) -> str:
    """Convert feature ids into readable dashboard labels."""
    return feature.replace("_", " ").title()

def build_display_table(data: pd.DataFrame) -> pd.DataFrame:
    """Build the dashboard table with user-facing labels and formats."""
    return pd.DataFrame({
        "ID": data["employee_id"].astype(int),
        "Name": data["name"],
        "Model Score": data["risk_score"].map(lambda score: f"{score:.2f}"),
        "Review Band": data["risk_score"].map(get_risk_badge),
    }).reset_index(drop=True)  # FIXED: Align indices perfectly for UI click matching

def top_driver_table(model, employee: pd.Series) -> tuple[pd.DataFrame, dict[str, float]]:
    """Return top 5 SHAP drivers and the full contribution mapping."""
    explanation_dict, _, _ = explain_prediction(model, employee.to_frame().T)
    shap_values = explanation_dict["shap_values"]
    if len(shap_values.shape) == 2:
        shap_values = shap_values[0]
    feature_names = explanation_dict["feature_names"]
    feature_contributions = dict(zip(feature_names, shap_values))
    top_5_idx = np.argsort(np.abs(shap_values))[-5:][::-1]
    driver_rows = []
    for idx in top_5_idx:
        value = float(shap_values[idx])
        driver_rows.append({
            "Feature": format_feature_name(feature_names[idx]),
            "SHAP Value": f"{value:+.3f}",
            "Impact": "Raises score" if value > 0 else "Lowers score",
        })
    return pd.DataFrame(driver_rows), feature_contributions

def build_person_explanation(employee: pd.Series, shap_df: pd.DataFrame) -> str:
    """Build a short readable explanation for an employee's top risk drivers."""
    first_name = employee.get("first_name", "This employee")
    risk_score = float(employee.get("risk_score", 0.0))
    drivers = ", ".join(shap_df["Feature"].head(3).astype(str).tolist())
    if not drivers:
        drivers = "the available risk drivers"
    return (
        f"{first_name} has a {risk_score:.1%} model score for 90-day attrition. "
        f"Use {drivers} as conversation prompts for the manager's next 1:1."
    )

DRIVER_GUIDANCE = {
    "perf_trend": {
        "topic": "recent performance trend",
        "question": "Review recent performance changes and ask what support would help stabilize momentum.",
        "action": "Agree on one near-term priority, success measure, and support checkpoint.",
    },
    "one_on_one_freq": {
        "topic": "manager check-in cadence",
        "question": "Ask whether the current 1:1 cadence gives enough coaching, context, and unblock time.",
        "action": "Set a recurring 1:1 cadence with clear agenda ownership for the next month.",
    },
    "engagement_score": {
        "topic": "engagement and motivation",
        "question": "Explore what parts of the role feel energizing, draining, or disconnected from goals.",
        "action": "Pick one engagement blocker to address before the next check-in.",
    },
    "pto_usage": {
        "topic": "time off and workload balance",
        "question": "Discuss workload balance, recovery time, and whether PTO patterns signal burnout or scheduling friction.",
        "action": "Review workload coverage and encourage a sustainable PTO plan.",
    },
    "comp_vs_band_pct": {
        "topic": "compensation positioning",
        "question": "Ask whether compensation, recognition, or role scope feels aligned with expectations.",
        "action": "Review compensation band context and clarify any available recognition or growth path.",
    },
    "months_since_promo": {
        "topic": "career progression",
        "question": "Discuss promotion readiness, next-level expectations, and what evidence is needed for growth.",
        "action": "Create a development plan with milestones tied to the next role level.",
    },
    "internal_moves": {
        "topic": "internal mobility",
        "question": "Explore whether a role change, stretch assignment, or cross-functional project would improve fit.",
        "action": "Identify one internal opportunity or stretch project to investigate.",
    },
}

DEFAULT_DRIVER_GUIDANCE = {
    "topic": "work experience",
    "question": "Ask what is helping or blocking their current work experience.",
    "action": "Document one concrete support commitment and follow up on it.",
}

LOW_BAND_GUIDANCE = [
    "Ask what is going well in the current role.",
    "Check whether any workload or support needs have changed recently.",
    "Confirm goals and priorities for the next regular check-in.",
]

def driver_guidance(feature: str) -> dict[str, str]:
    """Return manager guidance for a model feature."""
    return DRIVER_GUIDANCE.get(feature, DEFAULT_DRIVER_GUIDANCE)

def is_incomplete_prep_note(note: str) -> bool:
    """Return True when a note visibly stops mid-section or mid-list."""
    if not note:
        return True
    stripped = note.strip()
    required_sections = ["LIKELY ISSUE:", "TALKING POINTS:", "SUGGESTED ACTION:"]
    if any(section not in stripped for section in required_sections):
        return True
    return stripped.endswith(("1.", "2.", "3.", "1", "2", "3", ":"))

def finalize_prep_note(note: str, fallback_note: str) -> str:
    """Validate/sanitize a note, then fall back if it was damaged or incomplete."""
    candidate = note if validate_prep_note(note) else sanitize_output(note)
    if "largestframework" in candidate.lower() or is_incomplete_prep_note(candidate):
        return fallback_note
    return candidate

def build_fallback_prep_note(
    employee_data: dict,
    shap_explanation: dict | None = None,
) -> str:
    """Build a deterministic note when the local LLM is unavailable."""
    first_name = employee_data.get("first_name", "")
    last_name = employee_data.get("last_name", "")
    full_name = employee_data.get("name") or f"{first_name} {last_name}".strip() or "Employee"

    driver_text = ""
    sorted_drivers = []
    contributions = (shap_explanation or {}).get("feature_contributions") or {}
    if contributions:
        numeric_drivers = []
        for feature, value in contributions.items():
            try:
                numeric_drivers.append((feature, float(value)))
            except (TypeError, ValueError):
                continue
        sorted_drivers = sorted(
            (item for item in numeric_drivers if item[1] > 0),
            key=lambda item: item[1],
            reverse=True,
        )[:3]
        driver_text = ", ".join(
            f"{feature.replace('_', ' ').title()} ({value:+.2f})"
            for feature, value in sorted_drivers
        )

    try:
        model_score = float(employee_data.get("risk_score", 0.0))
    except (TypeError, ValueError):
        model_score = 0.0

    if model_score < RISK_MEDIUM_THRESHOLD:
        likely_issue = (
            f"{full_name}'s model score is in the Low review band. "
            "No elevated review signal is present; continue regular manager check-ins."
        )
        sorted_drivers = []
    elif driver_text:
        band = "High" if model_score >= RISK_HIGH_THRESHOLD else "Medium"
        likely_issue = (
            f"{full_name}'s model score is in the {band} review band. "
            f"The largest score-raising review drivers are {driver_text}. "
            "Use these as conversation prompts to understand workload, support needs, and engagement."
        )
    else:
        likely_issue = (
            f"{full_name}'s model score warrants manager review, but no score-raising "
            "driver explanation is available. Review the employee context manually."
        )

    guidance_items = [driver_guidance(feature) for feature, _ in sorted_drivers]
    if model_score < RISK_MEDIUM_THRESHOLD:
        talking_points = LOW_BAND_GUIDANCE
    else:
        while len(guidance_items) < 3:
            guidance_items.append(DEFAULT_DRIVER_GUIDANCE)
        talking_points = [item["question"] for item in guidance_items[:3]]

    suggested_action = (
        "Continue the regular 1:1 cadence and monitor for changes."
        if model_score < RISK_MEDIUM_THRESHOLD
        else guidance_items[0]["action"]
    )

    return f"""DRAFT - FOR HUMAN REVIEW ONLY

LIKELY ISSUE:
{likely_issue}

TALKING POINTS:
1. {talking_points[0]}
2. {talking_points[1]}
3. {talking_points[2]}

SUGGESTED ACTION:
Schedule a 1:1 with {full_name} this week. {suggested_action}"""

def render_employee_details(model, selected_employee: pd.Series, use_ollama: bool = False) -> None:
    """Render the selected employee detail panel."""
    st.divider()
    st.markdown(
        "### " f"EMPLOYEE DETAILS: {selected_employee['name']} " f"(ID: {selected_employee['employee_id']}) "
        "<span style='font-size:0.78em; color:#6b7280;'>[EXPANDED]</span>",
        unsafe_allow_html=True,
    )
    col1, col2 = st.columns([1, 1])
    with col1:
        st.write("**Model Score:**", f"{selected_employee['risk_score']:.1%}")
        st.write("**Review Band:**", get_risk_badge(float(selected_employee["risk_score"])))
        st.caption(risk_band_description())
        st.write("**Top Model Drivers:**")
        try:
            shap_df, feature_contributions = top_driver_table(model, selected_employee)
            st.dataframe(shap_df, hide_index=True)
        except Exception as e:
            logger.exception("SHAP failure for employee %s", selected_employee.get("employee_id"))
            st.error(f"Could not generate explanation: {e}")
            feature_contributions = {}

    with col2:
        st.write("**Generated 1:1 Prep Note (Draft):**")
        shap_explanation_for_prep = {"feature_contributions": feature_contributions}

        if use_ollama:
            try:
                with st.spinner("Generating draft note with local Ollama..."):
                    prep_note = draft_prep_note(selected_employee.to_dict(), shap_explanation_for_prep)
                fallback_note = build_fallback_prep_note(
                    selected_employee.to_dict(),
                    shap_explanation_for_prep,
                )
                sanitized_note = finalize_prep_note(prep_note, fallback_note)

                st.text_area(
                    "Draft Output Box", value=sanitized_note, height=320, disabled=False, label_visibility="collapsed", key=f"note_{selected_employee['employee_id']}"
                )
            except Exception as e:
                logger.exception("Prep note failure for employee %s", selected_employee.get("employee_id"))
                st.warning(f"Local Ollama generation failed: {e}")
                fallback_note = build_fallback_prep_note(
                    selected_employee.to_dict(),
                    shap_explanation_for_prep,
                )
                sanitized_note = finalize_prep_note(fallback_note, fallback_note)
                st.text_area(
                    "Draft Output Box Fallback", value=sanitized_note, height=320, disabled=False, label_visibility="collapsed", key=f"note_err_{selected_employee['employee_id']}"
                )
        else:
            fallback_note = build_fallback_prep_note(
                selected_employee.to_dict(),
                shap_explanation_for_prep,
            )
            sanitized_note = finalize_prep_note(fallback_note, fallback_note)
            st.text_area(
                "Draft Output Box Fast", value=sanitized_note, height=320, disabled=False, label_visibility="collapsed", key=f"note_fast_{selected_employee['employee_id']}"
            )

            if st.button("Generate with Ollama", key=f"ollama_{selected_employee['employee_id']}"):
                try:
                    with st.spinner("Generating draft note with local Ollama..."):
                        prep_note = draft_prep_note(selected_employee.to_dict(), shap_explanation_for_prep)
                    fallback_note = build_fallback_prep_note(
                        selected_employee.to_dict(),
                        shap_explanation_for_prep,
                    )
                    sanitized_note = finalize_prep_note(prep_note, fallback_note)
                    st.text_area(
                        "Ollama Draft Output Box",
                        value=sanitized_note,
                        height=320,
                        disabled=False,
                        label_visibility="collapsed",
                        key=f"note_ollama_{selected_employee['employee_id']}",
                    )
                except Exception as e:
                    logger.exception("Manual Ollama prep note failure for employee %s", selected_employee.get("employee_id"))
                    st.warning(f"Local Ollama generation failed: {e}")

def main():
    """Main function to run the Streamlit dashboard."""
    st.title("ATTRITION EARLY-WARNING AGENT")
    st.write("Estimates 90-day regretted-attrition probability from synthetic employee data, explains model drivers, and drafts manager 1:1 prep notes.")
    st.caption("Review bands are model triage labels, not performance ratings or employment decisions.")

    @st.cache_data(ttl=300)
    def get_data():
        return load_data_from_parquet()

    @st.cache_resource(ttl=300)
    def get_model():
        return load_model()

    data = get_data()
    model = get_model()
    category_vocabularies = load_category_vocabularies()
    training_columns = load_feature_columns()
    data = data.copy()
    data["name"] = data["first_name"] + " " + data["last_name"]

    try:
        model_features, _ = prepare_features(
            data, label_encoders=category_vocabularies, training_columns=training_columns,
        )
        probabilities = model.predict_proba(model_features)
        attrition_prob = probabilities[:, 1]
    except Exception as e:
        logger.exception("Prediction failed for dashboard data")
        st.error(f"Could not calculate risk scores: {e}")
        st.stop()

    data["risk_score"] = attrition_prob
    sorted_data = data.sort_values("risk_score", ascending=False).reset_index(drop=True) # FIXED index reset

    with st.sidebar:
        st.header("Controls")
        risk_threshold = st.slider("Review threshold for flags", 0.0, 1.0, RISK_HIGH_THRESHOLD, 0.05)
        use_ollama = st.toggle("Auto-generate with Ollama", value=False)
        st.caption(risk_band_description())
        st.write(f"Showing employees with model score >= {risk_threshold:.2f}")

    tab1, tab2 = st.tabs(["Team Roster", "This week's new flags"])

    with tab1:
        st.subheader("All Employees Sorted by Risk Score")
        st.caption("Click any row to see full details below.")
        display_data = build_display_table(sorted_data)
        roster_event = st.dataframe(
            display_data, on_select="rerun", selection_mode="single-row", hide_index=True, key="team_roster_table",
        )
        if roster_event.selection.rows:
            selected_employee = sorted_data.iloc[roster_event.selection.rows[0]]
            render_employee_details(model, selected_employee, use_ollama)

    with tab2:
        st.subheader("This Week's New Flags")
        st.write(f"Employees Above Review Threshold (>= {risk_threshold:.2f})")
        flags_data = sorted_data[sorted_data["risk_score"] >= risk_threshold].reset_index(drop=True) # FIXED index reset
        if flags_data.empty:
            st.write("No employees above the risk threshold.")
        else:
            st.caption("Click any row to see full details below.")
            flags_display = build_display_table(flags_data)
            flags_event = st.dataframe(
                flags_display, on_select="rerun", selection_mode="single-row", hide_index=True, key="new_flags_table",
            )
            if flags_event.selection.rows:
                selected_employee = flags_data.iloc[flags_event.selection.rows[0]]
                render_employee_details(model, selected_employee, use_ollama)

if __name__ == "__main__":
    main()
