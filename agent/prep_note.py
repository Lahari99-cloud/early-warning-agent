"""
LLM agent for drafting manager 1:1 preparation notes.
"""

import anthropic
import os
import sys
from typing import Dict, Any

# Add parent directory to path to import config
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import SEED


def build_prompt(employee_data: Dict[str, Any], shap_explanation: Dict[str, Any]) -> str:
    """
    Build a prompt for the LLM to generate a manager 1:1 prep note.

    Args:
        employee_data: Dictionary containing employee features and attrition risk.
        shap_explanation: Dictionary containing SHAP explanation (drivers).

    Returns:
        Prompt string for the LLM.
    """
    # Extract employee data
    attrition_risk = employee_data.get('attrition_risk', 0.0)
    first_name = employee_data.get('first_name', 'Employee')
    last_name = employee_data.get('last_name', '')
    full_name = f"{first_name} {last_name}".strip()
    exit_note = employee_data.get('exit_note', '')
    survey_blurb = employee_data.get('survey_blurb', '')

    # Format SHAP explanation
    shap_text = ""
    if shap_explanation and 'feature_contributions' in shap_explanation:
        contributions = shap_explanation['feature_contributions']
        # Sort by absolute contribution value (descending) to get top drivers
        sorted_contributions = sorted(
            contributions.items(),
            key=lambda x: abs(x[1]),
            reverse=True
        )

        # Take top 3 drivers
        top_drivers = sorted_contributions[:3]
        shap_lines = []
        for feature, contribution in top_drivers:
            # Format feature name to be more readable
            readable_name = feature.replace('_', ' ').title()
            shap_lines.append(f"  * {readable_name}: {contribution:.3f}")

        if shap_lines:
            shap_text = "Key Risk Drivers (SHAP values):\n" + "\n".join(shap_lines)

    # Build the prompt
    prompt = f"""You are an HR analytics assistant helping managers prepare for 1:1 meetings with employees at risk of attrition.

Employee Data:
- Name: {full_name if full_name else "Employee"}
- Attrition Risk Score: {attrition_risk:.2f} (0-1 scale, higher = higher risk)
{shap_text if shap_text else "- Key Risk Drivers: Not available"}

Additional Context:
- Exit Note: {exit_note if exit_note else "Not available"}
- Survey Feedback: {survey_blurb if survey_blurb else "Not available"}

Task: Generate a manager 1:1 preparation note that includes:
1. LIKELY ISSUE: A brief statement of the primary concern based on the risk drivers
2. TALKING POINTS: Three specific, constructive topics for the manager to discuss
3. SUGGESTED ACTION: One concrete, supportive action the manager can take

Important Guidelines:
- Output must be labeled as "DRAFT - FOR HUMAN REVIEW ONLY"
- Do not suggest any automated actions or decisions
- Keep tone supportive and developmental
- Focus on employee growth and retention
- Do not mention attrition risk directly; frame as development opportunities
- Maximum length: 300 words

Draft Note:
"""
    return prompt


def call_llm(prompt: str) -> str:
    """
    Call the LLM (Anthropic API) to generate a prep note.

    Args:
        prompt: The prompt string for the LLM.

    Returns:
        Generated text from the LLM.
    """
    # Get API key from environment
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        raise ValueError("ANTHROPIC_API_KEY environment variable is required")

    # Initialize Anthropic client
    client = anthropic.Anthropic(api_key=api_key)

    # Call the API
    try:
        response = client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=500,
            temperature=0.3,
            system="You are an HR analytics assistant helping managers prepare for 1:1 meetings with employees at risk of attrition. Your output must be supportive, developmental, and focused on employee growth. Never suggest automated decisions or actions.",
            messages=[
                {
                    "role": "user",
                    "content": prompt
                }
            ]
        )

        # Extract text from response
        if response.content and len(response.content) > 0:
            return response.content[0].text
        else:
            raise ValueError("Empty response from LLM")

    except Exception as e:
        raise RuntimeError(f"Failed to call Anthropic API: {str(e)}")


def draft_prep_note(employee_data: Dict[str, Any], shap_explanation: Dict[str, Any]) -> str:
    """
    Draft a manager 1:1 prep note using the LLM.

    Args:
        employee_data: Dictionary containing employee features and attrition risk.
        shap_explanation: Dictionary containing SHAP explanation (drivers).

    Returns:
        Draft prep note string.
    """
    # Build prompt
    prompt = build_prompt(employee_data, shap_explanation)

    # Call LLM with fallback for API failure
    try:
        note = call_llm(prompt)
    except Exception as e:
        # Fallback note when LLM API fails
        first_name = employee_data.get('first_name', 'Employee')
        last_name = employee_data.get('last_name', '')
        full_name = f"{first_name} {last_name}".strip()
        if not full_name:
            full_name = "Employee"
        note = f"""DRAFT - FOR HUMAN REVIEW ONLY

LIKELY ISSUE: Unable to generate detailed note due to technical difficulties. Please review {full_name}'s recent performance and engagement.

TALKING POINTS:
1. Discuss recent accomplishments and challenges
2. Explore current role satisfaction and future goals
3. Review development needs and support required

SUGGESTED ACTION: Schedule a follow-up 1:1 to continue the conversation."""

    return note