"""
Streamlit dashboard for the attrition early-warning agent.
"""

import streamlit as st
import pandas as pd
import numpy as np
from data.generate import load_data_from_parquet
from model.train import load_model
from model.explain import explain_prediction
from agent.prep_note import draft_prep_note
from agent.guardrails import validate_prep_note, sanitize_output


def main():
    """Main function to run the Streamlit dashboard."""
    st.title("Attrition Early-Warning Agent")
    st.write(
        """
        This demo predicts 90-day regretted-attrition risk per employee,
        explains the drivers, and drafts a manager 1:1 prep note.
        """
    )

    # Load data and model with caching
    @st.cache_data
    def get_data():
        return load_data_from_parquet()

    @st.cache_resource
    def get_model():
        return load_model()

    data = get_data()
    model = get_model()

    # Create full name column for display
    data_with_name = data.copy()
    data_with_name["name"] = data_with_name["first_name"] + " " + data_with_name["last_name"]

    # Calculate risk scores (probability of attrition) for all employees
    # Assuming model.predict_proba returns [prob_no_attrition, prob_attrition]
    probabilities = model.predict_proba(data)
    attrition_prob = probabilities[:, 1]  # probability of attrition (class 1)
    data_with_name["risk_score"] = attrition_prob

    # Sort by risk score descending
    sorted_data = data_with_name.sort_values("risk_score", ascending=False)

    # Define risk thresholds for badges
    def get_risk_badge(score):
        if score >= 0.7:
            return "🔴 High"
        elif score >= 0.4:
            return "🟠 Medium"
        else:
            return "🟢 Low"

    sorted_data["risk_badge"] = sorted_data["risk_score"].apply(get_risk_badge)

    # Sidebar for controls
    with st.sidebar:
        st.header("Controls")
        risk_threshold = st.slider(
            "Risk threshold for flags", 0.0, 1.0, 0.7, 0.05
        )
        st.write(f"Showing employees with risk score >= {risk_threshold}")

    # Main content: two tabs
    tab1, tab2 = st.tabs(["Team Roster", "This week's new flags"])

    with tab1:
        st.subheader("All Employees Sorted by Risk Score")
        # Display table with selection
        display_cols = ["employee_id", "name", "risk_score", "risk_badge"]
        display_data = sorted_data[display_cols].rename(
            columns={
                "employee_id": "ID",
                "name": "Name",
                "risk_score": "Risk Score",
                "risk_badge": "Risk Level",
            }
        )

        event = st.dataframe(
            display_data,
            on_select="rerun",
            selection_mode="single-row",
        )

    with tab2:
        st.subheader("This Week's New Flags (High Risk Employees)")
        # Filter for high-risk employees based on sidebar threshold
        flags_data = sorted_data[sorted_data["risk_score"] >= risk_threshold]
        if flags_data.empty:
            st.write("No employees above the risk threshold.")
        else:
            flags_display = flags_data[display_cols].rename(
                columns={
                    "employee_id": "ID",
                    "name": "Name",
                    "risk_score": "Risk Score",
                    "risk_badge": "Risk Level",
                }
            )

            event = st.dataframe(
                flags_display,
                on_select="rerun",
                selection_mode="single-row",
            )

    # Show details for selected employee
    selected_employee = None
    if event.selection.rows:
        selected_index = event.selection.rows[0]
        # Determine which tab we're in to get the correct dataframe
        # Streamlit doesn't give us direct tab info, so we check both
        if len(flags_data) > 0 and selected_index < len(flags_data):
            selected_employee = flags_data.iloc[selected_index]
        else:
            selected_employee = sorted_data.iloc[selected_index]

    if selected_employee is not None:
        st.divider()
        st.subheader(f"Employee Details: {selected_employee['name']} (ID: {selected_employee['employee_id']})")

        col1, col2 = st.columns([1, 1])

        with col1:
            st.write("**Risk Score:**", f"{selected_employee['risk_score']:.1%}")
            st.write("**Risk Level:**", selected_employee["risk_badge"])

            # Show SHAP drivers (explanation)
            st.write("**Top Drivers of Attrition Risk:**")
            try:
                explanation_dict, shap_values_df = explain_prediction(model, selected_employee.to_frame().T)
                # Plot SHAP values for the selected employee
                # explanation_dict contains expected_value, feature_names, shap_values (array)
                # We'll create a waterfall plot or bar chart for the top features
                # For simplicity, we'll show the top 5 features by absolute SHAP value
                shap_values = explanation_dict['shap_values']
                # If shap_values is 2D (samples, features), take the first row
                if len(shap_values.shape) == 2:
                    shap_values = shap_values[0]
                feature_names = explanation_dict['feature_names']
                # Get top 5 features by absolute SHAP value
                top_5_idx = np.argsort(np.abs(shap_values))[-5:][::-1]
                top_features = [feature_names[i] for i in top_5_idx]
                top_shap = [shap_values[i] for i in top_5_idx]
                # Create a DataFrame for display
                shap_df = pd.DataFrame({
                    'Feature': top_features,
                    'SHAP Value': top_shap
                })
                st.dataframe(shap_df)
            except Exception as e:
                st.error(f"Could not generate explanation: {e}")

        with col2:
            st.write("**Generated 1:1 Prep Note (Draft):**")
            try:
                explanation_dict, _ = explain_prediction(model, selected_employee.to_frame().T)
                prep_note = draft_prep_note(selected_employee.to_frame().T, explanation_dict)
                validated_note = validate_prep_note(prep_note)
                sanitized_note = sanitize_output(validated_note)
                st.write(sanitized_note)
            except Exception as e:
                st.error(f"Could not generate prep note: {e}")


if __name__ == "__main__":
    main()