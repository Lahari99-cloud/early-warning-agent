"""
Tests for the LLM agent prep note functionality.
"""

import pytest
from unittest.mock import Mock, patch
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from agent.prep_note import build_prompt, call_llm, draft_prep_note


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
    assert 'Comp Vs Band Pct' in prompt
    assert 'Perf Trend' in prompt
    assert 'Months Since Promo' in prompt


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
    assert 'Not available' in prompt  # For exit_note and survey_blurb
    assert 'Key Risk Drivers: Not available' in prompt


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
    assert 'Key Risk Drivers: Not available' in prompt


@patch('agent.prep_note.anthropic.Anthropic')
@patch.dict('os.environ', {'ANTHROPIC_API_KEY': 'test-key'})
def test_call_llm_success(mock_anthropic):
    """Test successful LLM call."""
    # Setup mock
    mock_client = Mock()
    mock_response = Mock()
    mock_content = [Mock()]
    mock_content[0].text = "Generated prep note content"
    mock_response.content = mock_content
    mock_client.messages.create.return_value = mock_response
    mock_anthropic.return_value = mock_client

    # Test
    result = call_llm("Test prompt")

    # Verify
    assert result == "Generated prep note content"
    mock_anthropic.assert_called_once()
    mock_client.messages.create.assert_called_once()


@patch('agent.prep_note.anthropic.Anthropic')
@patch.dict('os.environ', {'ANTHROPIC_API_KEY': 'test-key'})
def test_call_llm_api_error(mock_anthropic):
    """Test LLM call handling API errors."""
    # Setup mock to raise exception
    mock_client = Mock()
    mock_client.messages.create.side_effect = Exception("API Error")
    mock_anthropic.return_value = mock_client

    # Test and verify exception
    with pytest.raises(RuntimeError, match="Failed to call Anthropic API"):
        call_llm("Test prompt")


def test_call_llm_missing_api_key():
    """Test LLM call with missing API key."""
    with patch.dict('os.environ', {}, clear=True):
        with pytest.raises(ValueError, match="ANTHROPIC_API_KEY environment variable is required"):
            call_llm("Test prompt")


@patch('agent.prep_note.build_prompt')
@patch('agent.prep_note.call_llm')
def test_draft_prep_note(mock_call_llm, mock_build_prompt):
    """Test the full draft_prep_note function."""
    # Setup mocks
    mock_build_prompt.return_value = "Test prompt"
    mock_call_llm.return_value = "Generated note"

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
    assert result == "Generated note"
    mock_build_prompt.assert_called_once_with(employee_data, shap_explanation)
    mock_call_llm.assert_called_once_with("Test prompt")