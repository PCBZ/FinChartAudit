# tests/test_rule_check.py
"""Tests for the deterministic rule engine."""
import pytest
from finchartaudit.tools.rule_check import RuleEngine


@pytest.fixture
def engine():
    return RuleEngine()


class TestTruncatedAxis:
    def test_truncated_bar_chart(self, engine):
        result = engine.run_check("truncated_axis", {
            "axis_values": [80, 85, 90, 95, 100],
            "chart_type": "bar",
        })
        assert result["is_truncated"] is True
        assert result["origin"] == 80
        assert result["exaggeration_factor"] > 1

    def test_non_truncated_bar(self, engine):
        result = engine.run_check("truncated_axis", {
            "axis_values": [0, 25, 50, 75, 100],
            "chart_type": "bar",
        })
        assert result["is_truncated"] is False

    def test_line_chart_allowed(self, engine):
        result = engine.run_check("truncated_axis", {
            "axis_values": [80, 85, 90, 95, 100],
            "chart_type": "line",
        })
        assert result["is_truncated"] is False

    def test_empty_values(self, engine):
        result = engine.run_check("truncated_axis", {
            "axis_values": [],
            "chart_type": "bar",
        })
        assert result["is_truncated"] is False


class TestBrokenScale:
    def test_consistent_intervals(self, engine):
        result = engine.run_check("broken_scale", {
            "axis_values": [0, 10, 20, 30, 40],
        })
        assert result["is_broken"] is False

    def test_broken_intervals(self, engine):
        result = engine.run_check("broken_scale", {
            "axis_values": [0, 10, 20, 50, 60],
        })
        assert result["is_broken"] is True

    def test_too_few_values(self, engine):
        result = engine.run_check("broken_scale", {
            "axis_values": [0, 10],
        })
        assert result["is_broken"] is False


class TestValueMismatch:
    def test_matching_values(self, engine):
        result = engine.run_check("value_mismatch", {
            "text_value": 100,
            "chart_value": 102,
            "tolerance": 0.05,
        })
        assert result["is_mismatch"] is False

    def test_mismatched_values(self, engine):
        result = engine.run_check("value_mismatch", {
            "text_value": 100,
            "chart_value": 120,
            "tolerance": 0.05,
        })
        assert result["is_mismatch"] is True
        assert result["difference_pct"] == 20.0


class TestPairingCompleteness:
    def test_all_paired(self, engine):
        result = engine.run_check("pairing_completeness", {
            "nongaap_charts": [
                {"metric_name": "Adjusted EBITDA", "expected_gaap_metric": "net income"},
            ],
            "gaap_charts": [
                {"metric_name": "Net Income"},
            ],
        })
        assert result["all_paired"] is True

    def test_missing_pair(self, engine):
        result = engine.run_check("pairing_completeness", {
            "nongaap_charts": [
                {"metric_name": "Adjusted EBITDA", "expected_gaap_metric": "net income"},
            ],
            "gaap_charts": [],
        })
        assert result["all_paired"] is False
        assert result["missing_pairs"] == 1


class TestProminence:
    def test_balanced(self, engine):
        result = engine.run_check("prominence_score", {
            "nongaap": {"size": 14, "position": 0.5},
            "gaap": {"size": 14, "position": 0.5},
        })
        assert result["is_undue"] is False

    def test_undue_prominence(self, engine):
        result = engine.run_check("prominence_score", {
            "nongaap": {"size": 24, "position": 0.2},
            "gaap": {"size": 12, "position": 0.8},
        })
        assert result["is_undue"] is True


class TestInvertedAxis:
    def test_normal_axis(self, engine):
        # Normal Y-axis: top=100, bottom=0 → decreasing top-to-bottom → NOT inverted
        result = engine.run_check("inverted_axis", {
            "axis_values": [100, 75, 50, 25, 0],
        })
        assert result["is_inverted"] is False

    def test_inverted_axis(self, engine):
        # Inverted: top=0, bottom=100 → increasing top-to-bottom → inverted
        result = engine.run_check("inverted_axis", {
            "axis_values": [0, 25, 50, 75, 100],
        })
        assert result["is_inverted"] is True

    def test_negative_values_not_inverted(self, engine):
        # Axis with negatives (e.g., -1000000 to 0) is normal, not inverted
        result = engine.run_check("inverted_axis", {
            "axis_values": [0, -200000, -400000, -600000, -800000, -1000000],
        })
        assert result["is_inverted"] is False

    def test_too_few_values(self, engine):
        result = engine.run_check("inverted_axis", {
            "axis_values": [10],
        })
        assert result["is_inverted"] is False


class TestDualAxis:
    def test_dual_axis_detected(self, engine):
        result = engine.run_check("dual_axis", {
            "left_axis_values": [0, 50, 100, 150, 200],
            "right_axis_values": [0, 5, 10, 15, 20],
        })
        assert result["has_dual_axis"] is True
        assert result["scale_ratio"] == 10.0

    def test_no_right_axis(self, engine):
        result = engine.run_check("dual_axis", {
            "left_axis_values": [0, 50, 100],
            "right_axis_values": [],
        })
        assert result["has_dual_axis"] is False

    def test_similar_scales(self, engine):
        result = engine.run_check("dual_axis", {
            "left_axis_values": [0, 50, 100],
            "right_axis_values": [0, 40, 80],
        })
        assert result["has_dual_axis"] is True
        assert result["scale_ratio"] < 2.0


class TestAxisRange:
    def test_narrow_range_bar(self, engine):
        # Very narrow range on bar chart: 95-100, ratio=5%
        result = engine.run_check("inappropriate_axis_range", {
            "axis_values": [95, 96, 97, 98, 99, 100],
            "chart_type": "bar",
        })
        assert result["is_inappropriate"] is True

    def test_narrow_range_line_ok(self, engine):
        # Same narrow range but line chart → acceptable
        result = engine.run_check("inappropriate_axis_range", {
            "axis_values": [95, 96, 97, 98, 99, 100],
            "chart_type": "line",
        })
        assert result["is_inappropriate"] is False

    def test_full_range(self, engine):
        result = engine.run_check("inappropriate_axis_range", {
            "axis_values": [0, 25, 50, 75, 100],
            "chart_type": "bar",
        })
        assert result["is_inappropriate"] is False

    def test_moderate_range(self, engine):
        result = engine.run_check("inappropriate_axis_range", {
            "axis_values": [50, 60, 70, 80, 90, 100],
            "chart_type": "bar",
        })
        # range=50, max=100, ratio=0.5, min=50 < 70% of max → not inappropriate
        assert result["is_inappropriate"] is False


class TestBinning:
    def test_consistent_bins(self, engine):
        result = engine.run_check("inconsistent_binning", {
            "bin_edges": [0, 10, 20, 30, 40],
        })
        assert result["is_inconsistent"] is False

    def test_inconsistent_bins(self, engine):
        result = engine.run_check("inconsistent_binning", {
            "bin_edges": [0, 10, 20, 50, 60],
        })
        assert result["is_inconsistent"] is True

    def test_too_few_edges(self, engine):
        result = engine.run_check("inconsistent_binning", {
            "bin_edges": [0, 10],
        })
        assert result["is_inconsistent"] is False


class TestUnknownCheck:
    def test_raises_on_unknown(self, engine):
        with pytest.raises(ValueError, match="Unknown check type"):
            engine.run_check("nonexistent_check", {})
