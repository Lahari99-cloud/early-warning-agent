"""
Guardrails for the LLM agent to ensure safe and appropriate output.
"""

import re


def validate_prep_note(note: str) -> bool:
    """
    Validate the generated prep note for safety and appropriateness.

    Args:
        note: The generated prep note string.

    Returns:
        True if the note passes validation, False otherwise.
    """
    if not note or not isinstance(note, str):
        return False

    # Check length. Keep the lower bound modest so concise valid notes do not
    # fall into sanitizer fallback paths.
    word_count = len(note.strip().split())
    if word_count < 25 or word_count > 600:
        return False

    # Check for required sections
    required_sections = ["LIKELY ISSUE:", "TALKING POINTS:", "SUGGESTED ACTION:"]
    for section in required_sections:
        if section not in note:
            return False

    # Check for DRAFT label
    if "DRAFT -" not in note.upper() and "DRAFT:" not in note.upper():
        return False

    # Check for action-oriented language that implies automation.
    # Terms like "model drivers" are allowed when describing explainability.
    action_phrases = [
        r'\bwill be (terminated|fired|laid off|dismissed)\b',
        r'\bshould be (terminated|fired|laid off|dismissed)\b',
        r'\b(must|shall|will) (terminate|fire|lay off|dismiss)\b',
        r'\brecommend (termination|firing|layoff|dismissal)\b',
        r'\bautomatic(ally)? (terminate|fire|lay off|dismiss|send|decide)\b',
        r'\b(system|algorithm|ai|model) (will|must|shall|should) (terminate|fire|lay off|dismiss|decide|send)\b',
        r'\b(decided|sent|triggered) by (the )?(system|algorithm|ai|model)\b',
    ]

    note_lower = note.lower()
    for pattern in action_phrases:
        if re.search(pattern, note_lower):
            return False

    # Check for profanity (basic list)
    profanity_list = [
        'damn', 'hell', 'shit', 'fuck', 'bitch', 'ass', 'crap'
    ]
    for word in profanity_list:
        if re.search(rf'\b{word}\b', note_lower):
            return False

    # Check that tone is not overly negative or accusatory
    negative_phrases = [
        r'\b(you are|you\'re) (bad|terrible|awful|horrible)\b',
        r'\byou (always|never) (fail|mess up|screw up)\b',
        r'\byour performance is (unacceptable|pathetic|dismal)\b'
    ]

    for pattern in negative_phrases:
        if re.search(pattern, note_lower):
            return False

    return True


def sanitize_output(note: str) -> str:
    """
    Sanitize the generated prep note to remove any undesirable content.

    Args:
        note: The generated prep note string.

    Returns:
        Sanitized string.
    """
    if not note:
        return (
            "DRAFT - FOR HUMAN REVIEW ONLY\n\n"
            "LIKELY ISSUE:\nAssessment note unavailable. Please review the employee's model score and top drivers manually.\n\n"
            "TALKING POINTS:\n"
            "1. Review current role satisfaction and recent workload.\n"
            "2. Discuss recent accomplishments, challenges, and support needs.\n"
            "3. Ask about future goals and any barriers to engagement.\n\n"
            "SUGGESTED ACTION:\nSchedule a manager-reviewed 1:1 and document any agreed follow-up support."
        )

    # Ensure it's a string
    note = str(note)

    # Trim whitespace
    note = note.strip()

    if "TECHNICAL DIFFICULTIES" in note.upper() or "UNABLE TO GENERATE" in note.upper():
        note = re.sub(
            r'unable to generate(?: detailed)? note due to technical difficulties\.?',
            "Assessment note unavailable. Please review the employee's model score and top drivers manually.",
            note,
            flags=re.IGNORECASE,
        )

    # Ensure DRAFT label is present
    if "DRAFT -" not in note.upper() and "DRAFT:" not in note.upper():
        note = "DRAFT - FOR HUMAN REVIEW ONLY\n\n" + note

    # Remove any content that looks like automated action language.
    # Keep neutral explainability phrases such as "model drivers" intact.
    action_patterns = [
        (r'\bwill be (terminated|fired|laid off|dismissed)\b', 'will be discussed for development'),
        (r'\bshould be (terminated|fired|laid off|dismissed)\b', 'should be engaged in development conversation'),
        (r'\b(must|shall|will) (terminate|fire|lay off|dismiss)\b', r'\1 consider development options'),
        (r'\brecommend (termination|firing|layoff|dismissal)\b', 'recommend developmental discussion'),
        (r'\bautomatic(ally)? (terminate|fire|lay off|dismiss|send|decide)\b', 'review with a human before taking action'),
        (r'\b(system|algorithm|ai|model) (will|must|shall|should) (terminate|fire|lay off|dismiss|decide|send)\b', 'a manager should review before taking action'),
        (r'\b(decided|sent|triggered) by (the )?(system|algorithm|ai|model)\b', 'reviewed by a manager'),
    ]

    for pattern, replacement in action_patterns:
        note = re.sub(pattern, replacement, note, flags=re.IGNORECASE)

    # Remove profanity (replace with specific characters)
    profanity_replacements = [
        ('damn', '****'),   # 4 asterisks
        ('hell', '////'),   # 4 forward slashes
        ('shit', '****'),   # 4 asterisks
        ('fuck', '****'),   # 4 asterisks
        ('bitch', '******'), # 6 asterisks
        ('ass', '***'),     # 3 asterisks
        ('crap', '****')    # 4 asterisks
    ]
    for word, replacement in profanity_replacements:
        note = re.sub(rf'\b{word}\b', replacement, note, flags=re.IGNORECASE)

    # Ensure required sections exist (add if missing)
    # We'll build the note section by section to ensure proper ordering
    sections = {}

    # Extract existing sections if they exist
    if "LIKELY ISSUE:" in note:
        # Find the LIKELY ISSUE section and everything until next section or end
        likely_issue_start = note.find("LIKELY ISSUE:")
        talking_points_start = note.find("TALKING POINTS:")
        suggested_action_start = note.find("SUGGESTED ACTION:")

        # Determine end of LIKELY ISSUE section
        section_ends = [pos for pos in [talking_points_start, suggested_action_start] if pos != -1]
        likely_issue_end = min(section_ends) if section_ends else len(note)

        sections["LIKELY ISSUE:"] = note[likely_issue_start:likely_issue_end].strip()

    if "TALKING POINTS:" in note:
        talking_points_start = note.find("TALKING POINTS:")
        suggested_action_start = note.find("SUGGESTED ACTION:")

        # Determine end of TALKING POINTS section
        section_ends = [pos for pos in [suggested_action_start] if pos != -1]
        talking_points_end = min(section_ends) if section_ends else len(note)

        sections["TALKING POINTS:"] = note[talking_points_start:talking_points_end].strip()

    if "SUGGESTED ACTION:" in note:
        suggested_action_start = note.find("SUGGESTED ACTION:")
        sections["SUGGESTED ACTION:"] = note[suggested_action_start:].strip()

    # Build final note with all sections in correct order
    final_sections = []

    # Always include DRAFT line if not already in a section
    has_draft_in_content = any("DRAFT -" in content.upper() or "DRAFT:" in content.upper()
                              for content in sections.values())
    if not has_draft_in_content:
        final_sections.append("DRAFT - FOR HUMAN REVIEW ONLY")

    # Add sections in order
    for section in ["LIKELY ISSUE:", "TALKING POINTS:", "SUGGESTED ACTION:"]:
        if section in sections:
            final_sections.append(sections[section])
        else:
            # Add default content for missing sections
            if section == "LIKELY ISSUE:":
                final_sections.append("LIKELY ISSUE: Discussion needed regarding role fit and engagement.")
            elif section == "TALKING POINTS:":
                final_sections.append("TALKING POINTS: 1. Current role satisfaction\n2. Recent accomplishments and challenges\n3. Future goals and aspirations")
            elif section == "SUGGESTED ACTION:":
                final_sections.append("SUGGESTED ACTION: Schedule regular check-ins to discuss progress and development.")

    # Join sections with double newlines
    note = "\n\n".join(final_sections)

    # Keep complete structured notes. Validation handles word count bounds; do not
    # truncate here because partial numbered lists are worse than a longer draft.

    return note.strip()
