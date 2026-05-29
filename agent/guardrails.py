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

    # Check length (reasonable bounds for a prep note) - 50-500 words
    word_count = len(note.strip().split())
    if word_count < 50 or word_count > 500:
        return False

    # Check for required sections
    required_sections = ["LIKELY ISSUE:", "TALKING POINTS:", "SUGGESTED ACTION:"]
    for section in required_sections:
        if section not in note:
            return False

    # Check for DRAFT label
    if "DRAFT -" not in note.upper() and "DRAFT:" not in note.upper():
        return False

    # Check for action-oriented language that implies automation
    action_phrases = [
        r'\bwill be (terminated|fired|laid off|dismissed)\b',
        r'\bshould be (terminated|fired|laid off|dismissed)\b',
        r'\b(must|shall|will) (terminate|fire|lay off|dismiss)\b',
        r'\brecommend (termination|firing|layoff|dismissal)\b',
        r'\b automatic(ly)?\b',
        r'\b system(atic)?ally\b',
        r'\b algorithm\b',
        r'\b AI\b',
        r'\b model\b'
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
        if word in note_lower:
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
        return "DRAFT - FOR HUMAN REVIEW ONLY\n\nLIKELY ISSUE: Unable to generate note due to insufficient data.\nTALKING POINTS: 1. Discuss employee goals and concerns\n2. Review recent performance and development needs\n3. Explore training and growth opportunities\nSUGGESTED ACTION: Schedule follow-up meeting to continue discussion."

    # Ensure it's a string
    note = str(note)

    # Trim whitespace
    note = note.strip()

    # Ensure DRAFT label is present
    if "DRAFT -" not in note.upper() and "DRAFT:" not in note.upper():
        note = "DRAFT - FOR HUMAN REVIEW ONLY\n\n" + note

    # Remove any content that looks like automated action language
    action_patterns = [
        (r'\bwill be (terminated|fired|laid off|dismissed)\b', 'will be discussed for development'),
        (r'\bshould be (terminated|fired|laid off|dismissed)\b', 'should be engaged in development conversation'),
        (r'\b(must|shall|will) (terminate|fire|lay off|dismiss)\b', r'\1 consider development options'),
        (r'\brecommend (termination|firing|layoff|dismissal)\b', 'recommend developmental discussion'),
        (r'\b automatic(ly)?\b', ''),
        (r'\b system(atic)?ally\b', 'regularly'),
        (r'\b algorithm\b', 'approach'),
        (r'\b AI\b', 'HR guidance'),
        (r'\b model\b', 'framework')
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
        note = re.sub(rf'{word}', replacement, note, flags=re.IGNORECASE)

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

    # Limit length to reasonable bounds
    if len(note) > 500:
        # Truncate to last complete sentence within limit
        truncated = note[:500]
        last_period = truncated.rfind('.')
        if last_period > 400:  # Only truncate if we have a reasonable sentence
            note = truncated[:last_period + 1]
        else:
            note = truncated + "..."

    # Ensure minimum length
    if len(note.strip()) < 50:
        note = "DRAFT - FOR HUMAN REVIEW ONLY\n\nLIKELY ISSUE: Discussion needed regarding role fit and engagement.\nTALKING POINTS: 1. Discuss employee goals and concerns\n2. Review recent performance and development needs\n3. Explore training and growth opportunities\nSUGGESTED ACTION: Schedule follow-up meeting to continue discussion."

    return note.strip()