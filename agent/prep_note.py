""" LLM agent for drafting manager 1:1 preparation notes. """
import os
import sys
from typing import Any, Dict
import requests
from dotenv import load_dotenv

# Add parent directory to path to import config
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
try:
    from config import RISK_HIGH_THRESHOLD, RISK_MEDIUM_THRESHOLD, SEED  # noqa: F401
except ImportError:
    SEED = 42
    RISK_MEDIUM_THRESHOLD = 0.40
    RISK_HIGH_THRESHOLD = 0.70

load_dotenv()

OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434/api/generate")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.2:3b")
OLLAMA_TIMEOUT_SECONDS = int(os.getenv("OLLAMA_TIMEOUT_SECONDS", "25"))
OLLAMA_NUM_PREDICT = int(os.getenv("OLLAMA_NUM_PREDICT", "450"))

DRIVER_GUIDANCE = {
    "perf_trend": {
        "question": "Review recent performance changes and ask what support would help stabilize momentum.",
        "action": "Agree on one near-term priority, success measure, and support checkpoint.",
    },
    "one_on_one_freq": {
        "question": "Ask whether the current 1:1 cadence gives enough coaching, context, and unblock time.",
        "action": "Set a recurring 1:1 cadence with clear agenda ownership for the next month.",
    },
    "engagement_score": {
        "question": "Explore what parts of the role feel energizing, draining, or disconnected from goals.",
        "action": "Pick one engagement blocker to address before the next check-in.",
    },
    "pto_usage": {
        "question": "Discuss workload balance, recovery time, and whether PTO patterns signal burnout or scheduling friction.",
        "action": "Review workload coverage and encourage a sustainable PTO plan.",
    },
    "comp_vs_band_pct": {
        "question": "Ask whether compensation, recognition, or role scope feels aligned with expectations.",
        "action": "Review compensation band context and clarify any available recognition or growth path.",
    },
    "months_since_promo": {
        "question": "Discuss promotion readiness, next-level expectations, and what evidence is needed for growth.",
        "action": "Create a development plan with milestones tied to the next role level.",
    },
    "internal_moves": {
        "question": "Explore whether a role change, stretch assignment, or cross-functional project would improve fit.",
        "action": "Identify one internal opportunity or stretch project to investigate.",
    },
}

DEFAULT_DRIVER_GUIDANCE = {
    "question": "Ask what is helping or blocking their current work experience.",
    "action": "Document one concrete support commitment and follow up on it.",
}

LOW_BAND_GUIDANCE = [
    "Ask what is going well in the current role.",
    "Check whether any workload or support needs have changed recently.",
    "Confirm goals and priorities for the next regular check-in.",
]

class PrepNoteGenerationError(RuntimeError):
    """Raised when the local LLM cannot generate a prep note."""

def _employee_name(employee_data: Dict[str, Any]) -> str:
    first_name = employee_data.get("first_name", "")
    last_name = employee_data.get("last_name", "")
    if not first_name and "Name" in employee_data:
        full_name = employee_data["Name"]
    elif not first_name and "name" in employee_data:
        full_name = employee_data["name"]
    else:
        full_name = f"{first_name} {last_name}".strip()
    return full_name or "Employee"

def _top_driver_lines(shap_explanation: Dict[str, Any] | None, limit: int = 3) -> list[str]:
    if not shap_explanation or "feature_contributions" not in shap_explanation:
        return []
    contributions = shap_explanation.get("feature_contributions") or {}
    numeric_contributions = []
    for feature, contribution in contributions.items():
        try:
            numeric_contributions.append((feature, float(contribution)))
        except (TypeError, ValueError):
            continue
    sorted_contributions = sorted(
        (item for item in numeric_contributions if item[1] > 0),
        key=lambda item: item[1],
        reverse=True,
    )
    return [
        f"{feature.replace('_', ' ').title()}: {contribution:+.3f}"
        for feature, contribution in sorted_contributions[:limit]
    ]

def _top_driver_items(shap_explanation: Dict[str, Any] | None, limit: int = 3) -> list[tuple[str, float]]:
    if not shap_explanation or "feature_contributions" not in shap_explanation:
        return []
    contributions = shap_explanation.get("feature_contributions") or {}
    numeric_contributions = []
    for feature, contribution in contributions.items():
        try:
            numeric_contributions.append((feature, float(contribution)))
        except (TypeError, ValueError):
            continue
    return sorted(
        (item for item in numeric_contributions if item[1] > 0),
        key=lambda item: item[1],
        reverse=True,
    )[:limit]

def _driver_guidance(feature: str) -> dict[str, str]:
    return DRIVER_GUIDANCE.get(feature, DEFAULT_DRIVER_GUIDANCE)

def build_fallback_prep_note(
    employee_data: Dict[str, Any], shap_explanation: Dict[str, Any] | None = None,
) -> str:
    """Build a deterministic note when the local LLM is unavailable."""
    full_name = _employee_name(employee_data)
    try:
        model_score = float(employee_data.get("risk_score", employee_data.get("attrition_risk", 0.0)))
    except (TypeError, ValueError):
        model_score = 0.0

    driver_items = _top_driver_items(shap_explanation)
    if model_score < RISK_MEDIUM_THRESHOLD:
        likely_issue = (
            f"{full_name}'s model score is in the Low review band. "
            "No elevated review signal is present; continue regular manager check-ins."
        )
        driver_items = []
    elif driver_items:
        band = "High" if model_score >= RISK_HIGH_THRESHOLD else "Medium"
        driver_text = ", ".join(
            f"{feature.replace('_', ' ').title()}: {contribution:+.3f}"
            for feature, contribution in driver_items
        )
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

    # FIXED: Safely build and extract exact indexes out of the list to prevent crash truncations
    guidance_items = []
    for feature, _ in driver_items:
        guidance_items.append(_driver_guidance(feature))

    while len(guidance_items) < 3:
        guidance_items.append(DEFAULT_DRIVER_GUIDANCE)

    # FIXED: Extracting index position safely out of list item maps
    g1 = guidance_items[0]
    g2 = guidance_items[1]
    g3 = guidance_items[2]

    talking_points = (
        LOW_BAND_GUIDANCE
        if model_score < RISK_MEDIUM_THRESHOLD
        else [g1["question"], g2["question"], g3["question"]]
    )

    suggested_action = (
        "Continue the regular 1:1 cadence and monitor for changes."
        if model_score < RISK_MEDIUM_THRESHOLD
        else g1.get("action", DEFAULT_DRIVER_GUIDANCE["action"])
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

def build_prompt(employee_data: Dict[str, Any], shap_explanation: Dict[str, Any]) -> str:
    """Build a prompt for the LLM to generate a manager 1:1 prep note."""
    attrition_risk = employee_data.get("attrition_risk", 0.0)
    if not attrition_risk and "risk_score" in employee_data:
        attrition_risk = employee_data.get("risk_score", 0.0)
    if not attrition_risk and "Risk Score" in employee_data:
        attrition_risk = employee_data.get("Risk Score", 0.0)
    try:
        attrition_risk = float(attrition_risk)
    except (TypeError, ValueError):
        attrition_risk = 0.0

    full_name = _employee_name(employee_data)

    if attrition_risk >= RISK_HIGH_THRESHOLD:
        risk_level = "High"
    elif attrition_risk >= RISK_MEDIUM_THRESHOLD:
        risk_level = "Medium"
    else:
        risk_level = "Low"

    driver_lines = _top_driver_lines(shap_explanation)
    top_drivers_str = "; ".join(driver_lines) if driver_lines else "Not available"

    feedback_signals = [
        employee_data.get("exit_note"),
        employee_data.get("survey_blurb"),
        employee_data.get("Feedback"),
        employee_data.get("feedback"),
    ]
    feedback_str = "; ".join(str(signal) for signal in feedback_signals if signal) or "Not available"

    prompt = f"""You are a human resources assistant. Generate a manager 1:1 preparation note for the following employee:
Name: {full_name}
Risk Score: {attrition_risk:.2f}
Risk Level: {risk_level}
Top Risk Drivers: {top_drivers_str}
Feedback Signals: {feedback_str}

Output your response using this exact structure. Do not include introductory text, conversational filler, or wrap the text in markdown code blocks. Start directly with the text block:

DRAFT - FOR HUMAN REVIEW ONLY

LIKELY ISSUE:
[Provide a specific 1-2 sentence assessment based on their name, feedback signals, and risk drivers]

TALKING POINTS:
1. [Custom talking point 1 tailored to their profile]
2. [Custom talking point 2 tailored to their profile]
3. [Custom talking point 3 tailored to their profile]

SUGGESTED ACTION:
[Provide a clear action step for the manager]

Keep the full note complete and concise. Do not stop mid-sentence or leave numbered items blank.
"""
    return prompt

def call_llm(prompt: str) -> str:
    """Call local Ollama API to generate a prep note."""
    try:
        response = requests.post(
            OLLAMA_URL,
            json={
                "model": OLLAMA_MODEL,
                "prompt": prompt,
                "temperature": 0.3,
                "stream": False,
                "num_predict": OLLAMA_NUM_PREDICT,
            },
            timeout=OLLAMA_TIMEOUT_SECONDS,
        )
        if response.status_code != 200:
            raise PrepNoteGenerationError(f"Ollama API returned {response.status_code}")
        result = response.json()
        return result.get("response", "").strip()
    except requests.exceptions.ConnectionError:
        raise PrepNoteGenerationError(
            "Failed to connect to Ollama at http://localhost:11434. "
            f"Make sure: 1) 'ollama serve' is running, 2) {OLLAMA_MODEL} model is pulled"
        )
    except requests.exceptions.Timeout:
        raise PrepNoteGenerationError(
            f"Ollama did not respond within {OLLAMA_TIMEOUT_SECONDS} seconds. "
            "Start Ollama, pull the configured model, or increase OLLAMA_TIMEOUT_SECONDS."
        )
    except PrepNoteGenerationError:
        raise
    except Exception as e:
        raise PrepNoteGenerationError(f"Failed to call Ollama: {str(e)}")

def draft_prep_note(employee_data: Dict[str, Any], shap_explanation: Dict[str, Any]) -> str:
    """Draft a manager 1:1 prep note using the local LLM."""
    prompt = build_prompt(employee_data, shap_explanation)
    note = call_llm(prompt)
    if not note:
        raise PrepNoteGenerationError("Ollama returned an empty response string.")

    required_sections = ["LIKELY ISSUE:", "TALKING POINTS:", "SUGGESTED ACTION:"]
    if any(section not in note for section in required_sections):
        raise PrepNoteGenerationError("Ollama returned an incomplete prep note.")
    if note.rstrip().endswith(("1.", "2.", "3.", "1", "2", "3", ":")):
        raise PrepNoteGenerationError("Ollama stopped before finishing the prep note.")

    return note
