"""
Streamlit dashboard for the attrition early-warning agent.
"""

import streamlit as st
import pandas as pd
from data.generate import load_data_from_parquet, generate_synthetic_data
from model.train import train_model, evaluate_model
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

    # TODO: Implement dashboard functionality
    # For now, show a placeholder
    st.info("Dashboard not implemented yet.")


if __name__ == "__main__":
    main()