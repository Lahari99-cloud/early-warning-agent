"""
Tests for the LLM agent prep note functionality.
"""

import pytest
from unittest.mock import Mock, patch
import sys
import os
import requests
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from agent.prep_note import (
    PrepNoteGenerationError,
    build_fallback_prep_note,
    build_prompt,
    call_llm,
    draft_prep_note,
)


def test_build_prompt_basic():
    """Test prompt building with basic employee data."""
    employee_data = {
        'first_name': 'John',
        'last_name': 'Doe',
        'attrition_risk': 0.75,
        'exit_note': 'Seeking better opportunities',
        'survey_blurb': 'Feeling undervalued in current role'
    }

    shap_explanation = {
        'feature_contributions': {
            'comp_vs_band_pct': -0.3,
            'perf_trend': -0.2,
            'months_since_promo': 0.4
        }
    }

    prompt = build_prompt(employee_data, shap_explanation)

    # Check that key components are present
    assert 'John Doe' in prompt
    assert '0.75' in prompt
    assert 'Seeking better opportunities' in prompt
    assert 'Feeling undervalued in current role' in prompt
    assert 'LIKELY ISSUE:' in prompt
    assert 'TALKING POINTS:' in prompt
    assert 'SUGGESTED ACTION:' in prompt
    assert 'DRAFT - FOR HUMAN REVIEW ONLY' in prompt
    assert 'Months Since Promo' in prompt
    assert 'Comp Vs Band Pct' not in prompt
    assert 'Perf Trend' not in prompt


def test_build_prompt_minimal():
    """Test prompt building with minimal employee data."""
    employee_data = {
        'first_name': 'Jane',
        'last_name': 'Smith',
        'attrition_risk': 0.3
    }

    shap_explanation = {}

    prompt = build_prompt(employee_data, shap_explanation)

    # Check that key components are present
    assert 'Jane Smith' in prompt
    assert '0.30' in prompt
    assert 'Feedback Signals: Not available' in prompt
    assert 'Top Risk Drivers: Not available' in prompt


def test_build_prompt_empty_shap():
    """Test prompt building with empty SHAP explanation."""
    employee_data = {
        'first_name': 'Bob',
        'last_name': 'Wilson',
        'attrition_risk': 0.6,
        'exit_note': '',
        'survey_blurb': ''
    }

    shap_explanation = None

    prompt = build_prompt(employee_data, shap_explanation)

    # Should handle None shap_explanation gracefully
    assert 'Bob Wilson' in prompt
    assert 'Top Risk Drivers: Not available' in prompt


def test_call_llm_success():
    """Test successful LLM call."""
    with patch('agent.prep_note.requests.post') as mock_post:
        # Setup mock
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"response": "Generated prep note content"}
        mock_post.return_value = mock_response

        # Test
        result = call_llm("Test prompt")

        # Verify
        assert result == "Generated prep note content"
        mock_post.assert_called_once()


def test_call_llm_api_error():
    """Test LLM call handling API errors."""
    with patch('agent.prep_note.requests.post') as mock_post:
        # Setup mock to raise exception
        mock_response = Mock()
        mock_response.status_code = 500
        mock_post.return_value = mock_response

        # Test and verify exception
        with pytest.raises(PrepNoteGenerationError, match="Ollama API returned 500"):
            call_llm("Test prompt")


def test_call_llm_connection_error():
    """Test LLM call handling connection errors."""
    with patch('agent.prep_note.requests.post') as mock_post:
        # Setup mock to raise connection error
        mock_post.side_effect = requests.exceptions.ConnectionError("Connection failed")

        # Test and verify exception
        with pytest.raises(PrepNoteGenerationError, match="Failed to connect to Ollama"):
            call_llm("Test prompt")


def test_call_llm_timeout():
    """Test LLM call handling timeout errors."""
    with patch('agent.prep_note.requests.post') as mock_post:
        mock_post.side_effect = requests.exceptions.Timeout("Timed out")

        with pytest.raises(PrepNoteGenerationError, match="did not respond"):
            call_llm("Test prompt")


@patch('agent.prep_note.build_prompt')
@patch('agent.prep_note.call_llm')
def test_draft_prep_note(mock_call_llm, mock_build_prompt):
    """Test the full draft_prep_note function."""
    # Setup mocks
    mock_build_prompt.return_value = "Test prompt"
    mock_call_llm.return_value = """DRAFT - FOR HUMAN REVIEW ONLY

LIKELY ISSUE:
Generated note.

TALKING POINTS:
1. First point.
2. Second point.
3. Third point.

SUGGESTED ACTION:
Follow up with the employee."""

    # Test data
    employee_data = {
        'first_name': 'Test',
        'last_name': 'User',
        'attrition_risk': 0.5
    }
    shap_explanation = {
        'feature_contributions': {
            'comp_vs_band_pct': -0.1
        }
    }

    # Test
    result = draft_prep_note(employee_data, shap_explanation)

    # Verify
    assert "Generated note" in result
    mock_build_prompt.assert_called_once_with(employee_data, shap_explanation)
    mock_call_llm.assert_called_once_with("Test prompt")


@patch('agent.prep_note.build_prompt')
@patch('agent.prep_note.call_llm')
def test_draft_prep_note_raises_when_llm_fails(mock_call_llm, mock_build_prompt):
    """draft_prep_note should not silently return fallback text."""
    mock_build_prompt.return_value = "Test prompt"
    mock_call_llm.side_effect = PrepNoteGenerationError("Ollama unavailable")

    with pytest.raises(PrepNoteGenerationError, match="Ollama unavailable"):
        draft_prep_note({"first_name": "Test"}, {"feature_contributions": {}})


@patch('agent.prep_note.build_prompt')
@patch('agent.prep_note.call_llm')
def test_draft_prep_note_raises_when_llm_returns_incomplete_note(mock_call_llm, mock_build_prompt):
    """draft_prep_note should reject truncated Ollama output."""
    mock_build_prompt.return_value = "Test prompt"
    mock_call_llm.return_value = """DRAFT - FOR HUMAN REVIEW ONLY

LIKELY ISSUE:
Test issue.

TALKING POINTS:
1. First point
2."""

    with pytest.raises(PrepNoteGenerationError, match="incomplete|stopped"):
        draft_prep_note({"first_name": "Test"}, {"feature_contributions": {}})


def test_build_fallback_prep_note_uses_driver_data():
    """Fallback generation remains available for the dashboard error path."""
    employee_data = {
        'first_name': 'Test',
        'last_name': 'User',
        'risk_score': 0.8,
    }
    shap_explanation = {
        'feature_contributions': {
            'comp_vs_band_pct': -0.4,
            'perf_trend': 0.2,
        }
    }

    result = build_fallback_prep_note(employee_data, shap_explanation)

    assert "DRAFT - FOR HUMAN REVIEW ONLY" in result
    assert "Perf Trend" in result
    assert "Comp Vs Band Pct" not in result
    assert "Test User" in result


def test_build_fallback_prep_note_varies_guidance_by_top_driver():
    employee_data = {
        'first_name': 'Test',
        'last_name': 'User',
        'risk_score': 0.8,
    }

    perf_note = build_fallback_prep_note(
        employee_data,
        {'feature_contributions': {'perf_trend': 1.4, 'engagement_score': 0.2}},
    )
    pto_note = build_fallback_prep_note(
        employee_data,
        {'feature_contributions': {'pto_usage': 1.4, 'engagement_score': 0.2}},
    )

    assert "largest score-raising review drivers" in perf_note
    assert "recent performance changes" in perf_note
    assert "workload balance" in pto_note
    assert perf_note != pto_note
