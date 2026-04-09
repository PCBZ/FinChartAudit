"""Tests for T3 Pairing Agent — parsing only."""
import json
import pytest
from unittest.mock import MagicMock
from finchartaudit.agents.t3_pairing import T3PairingAgent, NONGAAP_TO_GAAP
from finchartaudit.memory.filing_memory import FilingMemory


@pytest.fixture
def agent():
    vlm = MagicMock()
    memory = FilingMemory()
    return T3PairingAgent(vlm=vlm, memory=memory)


class TestT3ParseResponse:
    def test_missing_gaap_counterpart(self, agent):
        text = json.dumps({
            "metrics": [{
                "name": "Adjusted EBITDA Margin",
                "type": "non_gaap",
                "expected_gaap": "Net Income Margin",
                "gaap_found": False,
                "prominence_issue": False,
                "evidence": "Only Adjusted EBITDA Margin shown",
            }],
            "pairing_matrix": {"total_nongaap": 1, "paired": 0, "missing": 1, "violations": []},
        })
        findings = agent._build_findings(json.loads(text),"MYE", 1)
        assert len(findings) == 1
        assert findings[0].subcategory == "missing_gaap_counterpart"
        assert findings[0].risk_level == "HIGH"

    def test_undue_prominence(self, agent):
        text = json.dumps({
            "metrics": [{
                "name": "Adjusted EBITDA",
                "type": "non_gaap",
                "expected_gaap": "Net Income",
                "gaap_found": True,
                "prominence_issue": True,
                "evidence": "Adjusted EBITDA in 18pt bold, Net Income in 12pt",
            }],
            "pairing_matrix": {"total_nongaap": 1, "paired": 1, "missing": 0, "violations": []},
        })
        findings = agent._build_findings(json.loads(text),"MYE", 1)
        assert len(findings) == 1
        assert findings[0].subcategory == "undue_prominence"

    def test_all_paired_no_issues(self, agent):
        text = json.dumps({
            "metrics": [{
                "name": "Adjusted Operating Income",
                "type": "non_gaap",
                "expected_gaap": "Operating Income",
                "gaap_found": True,
                "prominence_issue": False,
                "evidence": "Both shown equally",
            }],
            "pairing_matrix": {"total_nongaap": 1, "paired": 1, "missing": 0, "violations": []},
        })
        findings = agent._build_findings(json.loads(text),"CTAS", 1)
        assert len(findings) == 0


class TestNongaapMapping:
    def test_known_mappings(self):
        assert NONGAAP_TO_GAAP["adjusted ebitda"] == "net income"
        assert NONGAAP_TO_GAAP["free cash flow"] == "net cash from operations"
