"""
Guardrails for the LLM agent to ensure safe and appropriate output.
"""

def validate_prep_note(note: str) -> bool:
    """
    Validate the generated prep note for safety and appropriateness.

    Args:
        note: The generated prep note string.

    Returns:
        True if the note passes validation, False otherwise.
    """
    # TODO: Implement validation logic (e.g., length, profanity, sensitivity)
    raise NotImplementedError("Validation not implemented yet")


def sanitize_output(note: str) -> str:
    """
    Sanitize the generated prep note to remove any undesirable content.

    Args:
        note: The generated prep note string.

    Returns:
        Sanitized string.
    """
    # TODO: Implement sanitization
    raise NotImplementedError("Sanitization not implemented yet")