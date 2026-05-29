"""
Tests for the guardrails module.
"""

import pytest
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from agent.guardrails import validate_prep_note, sanitize_output


def test_validate_prep_note_valid():
    """Test validation of a valid prep note."""
    valid_note = """
    DRAFT - FOR HUMAN REVIEW ONLY

    LIKELY ISSUE: Employee shows signs of disengagement and may benefit from new challenges.

    TALKING POINTS:
    1. Discuss current project satisfaction and interests
    2. Explore skill development opportunities aligned with career goals
    3. Review recent feedback and areas for growth

    SUGGESTED ACTION: Schedule a career development discussion to explore internal opportunities.
    """

    assert validate_prep_note(valid_note) == True


def test_validate_prep_note_missing_sections():
    """Test validation fails when sections are missing."""
    incomplete_note = """
    DRAFT - FOR HUMAN REVIEW ONLY

    LIKELY ISSUE: Employee shows signs of disengagement.

    TALKING POINTS:
    1. Discuss current project satisfaction
    """

    assert validate_prep_note(incomplete_note) == False


def test_validate_prep_note_no_draft_label():
    """Test validation fails without DRAFT label."""
    no_draft_note = """
    LIKELY ISSUE: Employee shows signs of disengagement.

    TALKING POINTS:
    1. Discuss current project satisfaction and interests
    2. Explore skill development opportunities aligned with career goals
    3. Review recent feedback and areas for growth

    SUGGESTED ACTION: Schedule a career development discussion to explore internal opportunities.
    """

    assert validate_prep_note(no_draft_note) == False


def test_validate_prep_note_too_short():
    """Test validation fails when note is too short."""
    short_note = """
    DRAFT - FOR HUMAN REVIEW ONLY

    LIKELY ISSUE: Issue.

    TALKING POINTS:
    1. Point
    2. Point
    3. Point

    SUGGESTED ACTION: Action.
    """

    assert validate_prep_note(short_note) == False


def test_validate_prep_note_action_language():
    """Test validation fails with action-oriented language."""
    action_note = """
    DRAFT - FOR HUMAN REVIEW ONLY

    LIKELY ISSUE: Employee will be terminated if performance doesn't improve.

    TALKING POINTS:
    1. Discuss current project satisfaction and interests
    2. Explore skill development opportunities aligned with career goals
    3. Review recent feedback and areas for growth

    SUGGESTED ACTION: Schedule a performance improvement plan.
    """

    assert validate_prep_note(action_note) == False


def test_validate_prep_note_profanity():
    """Test validation fails with profanity."""
    profanity_note = """
    DRAFT - FOR HUMAN REVIEW ONLY

    LIKELY ISSUE: This damn situation needs fixing.

    TALKING POINTS:
    1. Discuss current project satisfaction and interests
    2. Explore skill development opportunities aligned with career goals
    3. Review recent feedback and areas for growth

    SUGGESTED ACTION: Schedule a career development discussion.
    """

    assert validate_prep_note(profanity_note) == False


def test_sanitize_output_basic():
    """Test sanitization of a basic note."""
    raw_note = """
    This is a test note about employee engagement.

    LIKELY ISSUE: Employee shows signs of disengagement.

    TALKING POINTS:
    1. Discuss current project satisfaction
    2. Explore development opportunities
    3. Review recent feedback

    SUGGESTED ACTION: Schedule a discussion.
    """

    sanitized = sanitize_output(raw_note)
    assert "DRAFT -" in sanitized
    assert "LIKELY ISSUE:" in sanitized
    assert "TALKING POINTS:" in sanitized
    assert "SUGGESTED ACTION:" in sanitized


def test_sanitize_output_remove_action_language():
    """Test sanitization removes action-oriented language."""
    action_note = """
    DRAFT - FOR HUMAN REVIEW ONLY

    LIKELY ISSUE: Employee will be terminated if they don't improve.

    TALKING POINTS:
    1. Discuss current project satisfaction and interests
    2. Explore skill development opportunities
    3. Review recent feedback

    SUGGESTED ACTION: Recommend termination if no improvement.
    """

    sanitized = sanitize_output(action_note)
    assert "will be terminated" not in sanitized.lower()
    assert "recommend termination" not in sanitized.lower()


def test_sanitize_output_add_missing_sections():
    """Test sanitization adds missing sections."""
    minimal_note = "Some random text without proper structure."

    sanitized = sanitize_output(minimal_note)
    assert "DRAFT -" in sanitized
    assert "LIKELY ISSUE:" in sanitized
    assert "TALKING POINTS:" in sanitized
    assert "SUGGESTED ACTION:" in sanitized


def test_sanitize_output_length_limiting():
    """Test sanitization limits excessive length."""
    long_note = "DRAFT - FOR HUMAN REVIEW ONLY\n\n"
    long_note += "LIKELY ISSUE: " + "A" * 600 + "\n\n"
    long_note += "TALKING POINTS:\n1. Point\n2. Point\n3. Point\n\n"
    long_note += "SUGGESTED ACTION: Action."

    sanitized = sanitize_output(long_note)
    assert len(sanitized) <= 520  # Allow some buffer over 500 for formatting


def test_sanitize_output_empty_input():
    """Test sanitization handles empty input."""
    assert sanitize_output("") != ""
    assert "DRAFT -" in sanitize_output("")
    assert "LIKELY ISSUE:" in sanitize_output("")
    assert "TALKING POINTS:" in sanitize_output("")
    assert "SUGGESTED ACTION:" in sanitize_output("")


def test_sanitize_output_profanity_replacement():
    """Test sanitization replaces profanity with asterisks."""
    profane_note = """
    DRAFT - FOR HUMAN REVIEW ONLY

    LIKELY ISSUE: This damnsituation is hellish.

    TALKING POINTS:
    1. Discuss current project satisfaction
    2. Explore development opportunities
    3. Review recent feedback

    SUGGESTED ACTION: Schedule a discussion.
    """

    sanitized = sanitize_output(profane_note)
    assert "****" in sanitized  # damn -> ****
    assert "////" in sanitized  # hell -> ////