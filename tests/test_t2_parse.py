# tests/test_t2_parse.py
"""Tests for T2 response parsing — no VLM calls needed."""
import json
import pytest
from unittest.mock import MagicMock
from finchartaudit.agents.t2_visual import T2VisualAgent
from finchartaudit.memory.filing_memory import FilingMemory


@pytest.fixture
def agent():
    vlm = MagicMock()
    memory = FilingMemory()
    return T2VisualAgent(vlm=vlm, memory=memory)


class TestExtractJson:
    def test_direct_json(self, agent):
        text = '{"misleaders": {}, "completeness": {}}'
        result = agent._extract_json(text)
        assert result is not None
        assert "misleaders" in result

    def test_json_in_markdown(self, agent):
        text = 'Here is my analysis:\n```json\n{"misleaders": {}, "completeness": {}}\n```'
        result = agent._extract_json(text)
        assert result is not None

    def test_json_with_surrounding_text(self, agent):
        text = 'I found issues. {"misleaders": {"truncated axis": {"present": true, "confidence": 0.9, "evidence": "Y starts at 80"}}, "completeness": {}} That is all.'
        result = agent._extract_json(text)
        assert result is not None
        assert result["misleaders"]["truncated axis"]["present"] is True

    def test_no_json(self, agent):
        result = agent._extract_json("No JSON here at all.")
        assert result is None


class TestParseResponse:
    def test_misleader_finding(self, agent):
        text = json.dumps({
            "chart_type": "bar",
            "metric_name": "Revenue",
            "is_gaap": True,
            "misleaders": {
                "truncated axis": {"present": True, "confidence": 0.9, "evidence": "Y-axis starts at 80"},
                "3d": {"present": False, "confidence": 0.1, "evidence": ""},
            },
            "completeness": {},
        })
        findings = agent._parse_response(text, "c1", 1, [])
        assert len(findings) == 1
        assert findings[0].subcategory == "truncated axis"
        assert findings[0].confidence == 0.9

    def test_completeness_finding(self, agent):
        text = json.dumps({
            "misleaders": {},
            "completeness": {
                "missing_chart_title": {"present": True, "confidence": 0.8, "evidence": "No title"},
                "missing_legend": {"present": False, "confidence": 0.2, "evidence": ""},
            },
        })
        findings = agent._parse_response(text, "c1", 1, [])
        assert len(findings) == 1
        assert findings[0].category == "completeness"
        assert findings[0].subcategory == "missing_chart_title"

    def test_low_confidence_filtered(self, agent):
        text = json.dumps({
            "misleaders": {
                "truncated axis": {"present": True, "confidence": 0.2, "evidence": "Maybe"},
            },
            "completeness": {},
        })
        findings = agent._parse_response(text, "c1", 1, [])
        assert len(findings) == 0

    def test_risk_level_assignment(self, agent):
        text = json.dumps({
            "misleaders": {
                "truncated axis": {"present": True, "confidence": 0.95, "evidence": "Y starts at 80"},
            },
            "completeness": {},
        })
        findings = agent._parse_response(text, "c1", 1, [])
        assert findings[0].risk_level == "HIGH"
