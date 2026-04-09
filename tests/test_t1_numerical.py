"""Tests for T1 Numerical Consistency Agent — parsing only."""
import json
import pytest
from unittest.mock import MagicMock
from finchartaudit.agents.t1_numerical import T1NumericalAgent
from finchartaudit.memory.filing_memory import FilingMemory


@pytest.fixture
def agent():
    vlm = MagicMock()
    memory = FilingMemory()
    return T1NumericalAgent(vlm=vlm, memory=memory)


class TestT1ParseResponse:
    def test_mismatch_detected(self, agent):
        text = json.dumps({
            "claims": [{
                "text_claim": "revenue of $1.2B",
                "text_value": 1200,
                "chart_value": 900,
                "metric": "revenue",
                "match": False,
                "evidence": "Text says $1.2B, chart shows $900M",
            }],
            "mismatches_found": 1,
        })
        findings = agent._parse_response(text, "c1", 1)
        assert len(findings) == 1
        assert findings[0].tier == "T1"
        assert findings[0].subcategory == "value_mismatch"
        assert findings[0].risk_level == "HIGH"

    def test_no_mismatch(self, agent):
        text = json.dumps({
            "claims": [{
                "text_claim": "revenue of $1.2B",
                "text_value": 1200,
                "chart_value": 1210,
                "metric": "revenue",
                "match": True,
                "evidence": "Values match within tolerance",
            }],
            "mismatches_found": 0,
        })
        findings = agent._parse_response(text, "c1", 1)
        assert len(findings) == 0

    def test_no_text_context_skips(self, agent):
        findings = agent.execute({
            "image_path": "fake.png",
            "text_context": "",
            "page": 1,
        })
        assert len(findings) == 0
