"""
LLM agent for drafting manager 1:1 preparation notes.
"""

import anthropic
from typing import Dict, Any
from ..config import SEED


def build_prompt(employee_data: Dict[str, Any], shap_explanation: Dict[str, Any]) -> str:
    """
    Build a prompt for the LLM to generate a manager 1:1 prep note.

    Args:
        employee_data: Dictionary containing employee features and attrition risk.
        shap_explanation: Dictionary containing SHAP explanation (drivers).

    Returns:
        Prompt string for the LLM.
    """
    # TODO: Implement prompt building
    raise NotImplementedError("Prompt building not implemented yet")


def call_llm(prompt: str) -> str:
    """
    Call the LLM (Anthropic API) to generate a prep note.

    Args:
        prompt: The prompt string for the LLM.

    Returns:
        Generated text from the LLM.
    """
    # TODO: Implement LLM call
    raise NotImplementedError("LLM call not implemented yet")


def draft_prep_note(employee_data: Dict[str, Any], shap_explanation: Dict[str, Any]) -> str:
    """
    Draft a manager 1:1 prep note using the LLM.

    Args:
        employee_data: Dictionary containing employee features and attrition risk.
        shap_explanation: Dictionary containing SHAP explanation (drivers).

    Returns:
        Draft prep note string.
    """
    # TODO: Implement drafting process
    raise NotImplementedError("Prep note drafting not implemented yet")