"""Deterministic rule engine for quantitative validation."""
from __future__ import annotations


class RuleEngine:
    """Runs deterministic checks on extracted data. No LLM, no hallucination."""

    def run_check(self, check_type: str, data: dict) -> dict:
        dispatch = {
            "truncated_axis": self._check_truncated_axis,
            "broken_scale": self._check_broken_scale,
            "inverted_axis": self._check_inverted_axis,
            "dual_axis": self._check_dual_axis,
            "inappropriate_axis_range": self._check_axis_range,
            "inconsistent_binning": self._check_binning,
            "font_size_comparison": self._check_font_size,
            "value_mismatch": self._check_value_mismatch,
            "pairing_completeness": self._check_pairing,
            "prominence_score": self._check_prominence,
        }
        handler = dispatch.get(check_type)
        if handler is None:
            raise ValueError(f"Unknown check type: {check_type}")
        return handler(data)

    def _check_truncated_axis(self, data: dict) -> dict:
        axis_values = [float(v) for v in data.get("axis_values", [])]
        chart_type = data.get("chart_type", "bar").lower()
        if not axis_values:
            return {"is_truncated": False, "explanation": "No axis values provided"}

        origin = min(axis_values)
        value_range = max(axis_values) - origin

        # Only bar/area charts should start at 0. Line/scatter charts are allowed
        # to have non-zero origins. Also skip if any negative values (legitimate).
        truncated_types = {"bar", "area", "bar chart", "3d_bar", "histogram"}
        has_negative = any(v < 0 for v in axis_values)

        is_truncated = (
            origin > 0
            and chart_type in truncated_types
            and not has_negative
            and value_range > 0
            # Only flag if exaggeration is significant (>2x)
            and (max(axis_values) / value_range) > 2.0
        )

        return {
            "is_truncated": is_truncated,
            "origin": origin,
            "expected_origin": 0,
            "max_value": max(axis_values),
            "value_range": value_range,
            "exaggeration_factor": round(max(axis_values) / value_range, 2) if value_range > 0 else 1.0,
            "explanation": (
                f"Y-axis starts at {origin} instead of 0. "
                f"Visual differences are exaggerated by ~{max(axis_values) / value_range:.1f}x."
                if is_truncated else "Axis is not truncated."
            ),
        }

    def _check_broken_scale(self, data: dict) -> dict:
        axis_values = [float(v) for v in data.get("axis_values", [])]
        if len(axis_values) < 3:
            return {"is_broken": False, "explanation": "Need at least 3 values"}

        intervals = [axis_values[i + 1] - axis_values[i] for i in range(len(axis_values) - 1)]
        # Use absolute intervals (axis might be read top-to-bottom = negative intervals)
        abs_intervals = [abs(iv) for iv in intervals]
        avg_interval = sum(abs_intervals) / len(abs_intervals)

        if avg_interval == 0:
            return {"is_broken": False, "explanation": "Zero average interval"}

        max_deviation = max(abs(iv - avg_interval) for iv in abs_intervals)
        # Relaxed threshold: 30% deviation (was 15%, too strict for OCR noise)
        threshold = avg_interval * 0.30

        return {
            "is_broken": max_deviation > threshold,
            "intervals": [round(iv, 4) for iv in intervals],
            "avg_interval": round(avg_interval, 4),
            "max_deviation": round(max_deviation, 4),
            "explanation": (
                f"Tick intervals are inconsistent (max deviation {max_deviation:.3f} vs avg {avg_interval:.3f})."
                if max_deviation > threshold else "Tick intervals are consistent."
            ),
        }

    def _check_inverted_axis(self, data: dict) -> dict:
        """Check if Y-axis values increase from top to bottom (inverted).

        Normal Y-axis: top=large, bottom=small → OCR reads top-to-bottom as DECREASING.
        Inverted Y-axis: top=small, bottom=large → OCR reads top-to-bottom as INCREASING.

        So if values read top-to-bottom are strictly INCREASING, the axis is inverted.
        """
        axis_values = [float(v) for v in data.get("axis_values", [])]
        if len(axis_values) < 3:
            return {"is_inverted": False, "explanation": "Need at least 3 values"}

        increasing = sum(1 for i in range(len(axis_values) - 1)
                        if axis_values[i + 1] > axis_values[i])
        decreasing = sum(1 for i in range(len(axis_values) - 1)
                        if axis_values[i + 1] < axis_values[i])
        total_pairs = len(axis_values) - 1

        # Inverted = values INCREASE top-to-bottom (strictly: all or nearly all increasing)
        # Normal = values DECREASE top-to-bottom (or mixed with negative values)
        # Also exclude axes with negative values (e.g., -400000 to 0 is normal)
        has_negative = any(v < 0 for v in axis_values)

        is_inverted = (
            increasing >= total_pairs * 0.8  # >=80% of pairs are increasing
            and increasing >= 3              # at least 3 increasing pairs
            and not has_negative             # no negative values
            and all(v >= 0 for v in axis_values)  # all non-negative
        )

        return {
            "is_inverted": is_inverted,
            "values_top_to_bottom": axis_values,
            "increasing_pairs": increasing,
            "decreasing_pairs": decreasing,
            "explanation": (
                f"Y-axis appears inverted: values increase from top ({axis_values[0]}) to "
                f"bottom ({axis_values[-1]}), suggesting the axis direction is reversed."
                if is_inverted else "Y-axis direction is normal."
            ),
        }

    def _check_dual_axis(self, data: dict) -> dict:
        """Check if chart has dual Y-axes with different scales."""
        left_values = [float(v) for v in data.get("left_axis_values", [])]
        right_values = [float(v) for v in data.get("right_axis_values", [])]

        has_dual = len(left_values) >= 2 and len(right_values) >= 2

        if not has_dual:
            return {"has_dual_axis": False, "explanation": "No dual axis detected."}

        left_range = max(left_values) - min(left_values) if left_values else 0
        right_range = max(right_values) - min(right_values) if right_values else 0

        if left_range == 0 or right_range == 0:
            scale_ratio = 0
        else:
            scale_ratio = max(left_range, right_range) / min(left_range, right_range)

        return {
            "has_dual_axis": True,
            "left_range": [min(left_values), max(left_values)],
            "right_range": [min(right_values), max(right_values)],
            "scale_ratio": round(scale_ratio, 2),
            "explanation": (
                f"Dual Y-axis detected. Left range: {min(left_values)}-{max(left_values)}, "
                f"Right range: {min(right_values)}-{max(right_values)}, "
                f"scale ratio: {scale_ratio:.1f}x."
            ),
        }

    def _check_axis_range(self, data: dict) -> dict:
        """Check if axis range is inappropriately narrow, exaggerating differences."""
        axis_values = [float(v) for v in data.get("axis_values", [])]
        chart_type = data.get("chart_type", "bar").lower()
        if len(axis_values) < 2:
            return {"is_inappropriate": False, "explanation": "Need at least 2 values"}

        min_val = min(axis_values)
        max_val = max(axis_values)
        value_range = max_val - min_val

        if max_val == 0 or value_range == 0:
            return {"is_inappropriate": False, "explanation": "Zero range or max value"}

        range_ratio = value_range / abs(max_val)

        # Only flag for bar/area charts (where truncation matters most).
        # Must be very narrow (<10% of max) AND min well above 0 (>70% of max).
        # Line/scatter charts legitimately zoom into narrow ranges.
        bar_types = {"bar", "area", "bar chart", "3d_bar", "histogram"}
        is_inappropriate = (
            chart_type in bar_types
            and range_ratio < 0.10
            and min_val > 0
            and min_val > max_val * 0.7
        )

        return {
            "is_inappropriate": is_inappropriate,
            "min_value": min_val,
            "max_value": max_val,
            "value_range": value_range,
            "range_ratio": round(range_ratio, 3),
            "explanation": (
                f"Axis range is very narrow: {min_val}-{max_val} "
                f"(covers only {range_ratio * 100:.0f}% of max value). "
                f"This significantly exaggerates visual differences."
                if is_inappropriate
                else f"Axis range {min_val}-{max_val} is acceptable."
            ),
        }

    def _check_binning(self, data: dict) -> dict:
        """Check if histogram/bar bins have inconsistent widths."""
        bin_edges = [float(v) for v in data.get("bin_edges", [])]
        if len(bin_edges) < 3:
            return {"is_inconsistent": False, "explanation": "Need at least 3 bin edges"}

        widths = [bin_edges[i + 1] - bin_edges[i] for i in range(len(bin_edges) - 1)]
        avg_width = sum(widths) / len(widths)

        if avg_width == 0:
            return {"is_inconsistent": False, "explanation": "Zero average bin width"}

        max_deviation = max(abs(w - avg_width) / avg_width for w in widths)
        is_inconsistent = max_deviation > 0.2  # >20% deviation from average

        return {
            "is_inconsistent": is_inconsistent,
            "bin_widths": [round(w, 4) for w in widths],
            "avg_width": round(avg_width, 4),
            "max_deviation_pct": round(max_deviation * 100, 1),
            "explanation": (
                f"Bin widths are inconsistent (max deviation {max_deviation * 100:.0f}% from average). "
                f"Widths: {[round(w, 2) for w in widths]}."
                if is_inconsistent
                else f"Bin widths are consistent (max deviation {max_deviation * 100:.0f}%)."
            ),
        }

    def _check_font_size(self, data: dict) -> dict:
        a = data.get("element_a", {})
        b = data.get("element_b", {})
        size_a = float(a.get("size", 0))
        size_b = float(b.get("size", 0))

        if size_b == 0:
            return {"ratio": 0, "more_prominent": a.get("name", "A"), "explanation": "Element B has zero size"}

        ratio = size_a / size_b
        more_prominent = a.get("name", "A") if ratio > 1 else b.get("name", "B")

        return {
            "ratio": round(ratio, 2),
            "more_prominent": more_prominent,
            "size_a": size_a,
            "size_b": size_b,
            "explanation": (
                f"{a.get('name', 'A')} ({size_a}px) is {ratio:.1f}x the size of {b.get('name', 'B')} ({size_b}px)."
            ),
        }

    def _check_value_mismatch(self, data: dict) -> dict:
        try:
            text_val = float(data.get("text_value", 0))
            chart_val = float(data.get("chart_value", 0))
        except (ValueError, TypeError):
            return {"is_mismatch": True, "explanation": "Cannot parse values as numbers"}

        tolerance = float(data.get("tolerance", 0.05))
        if text_val == 0:
            diff = abs(chart_val)
            is_mismatch = diff > tolerance
        else:
            diff = abs(text_val - chart_val) / abs(text_val)
            is_mismatch = diff > tolerance

        return {
            "is_mismatch": is_mismatch,
            "text_value": text_val,
            "chart_value": chart_val,
            "difference_pct": round(diff * 100, 2),
            "tolerance_pct": round(tolerance * 100, 2),
            "explanation": (
                f"Values differ by {diff * 100:.1f}% (threshold: {tolerance * 100:.0f}%)."
                if is_mismatch else "Values match within tolerance."
            ),
        }

    def _check_pairing(self, data: dict) -> dict:
        nongaap = data.get("nongaap_charts", [])
        gaap = data.get("gaap_charts", [])
        gaap_metrics = {c.get("metric_name", "").lower() for c in gaap}

        pairings = []
        for ng in nongaap:
            expected = ng.get("expected_gaap_metric", "").lower()
            found = expected in gaap_metrics or any(expected in g for g in gaap_metrics)
            pairings.append({
                "nongaap_metric": ng.get("metric_name", ""),
                "expected_gaap": ng.get("expected_gaap_metric", ""),
                "gaap_found": found,
                "status": "paired" if found else "missing",
            })

        missing = sum(1 for p in pairings if p["status"] == "missing")
        return {
            "pairings": pairings,
            "total_nongaap": len(nongaap),
            "missing_pairs": missing,
            "all_paired": missing == 0,
            "explanation": (
                f"{missing}/{len(nongaap)} Non-GAAP charts lack GAAP counterparts."
                if missing > 0 else "All Non-GAAP charts have GAAP counterparts."
            ),
        }

    def _check_prominence(self, data: dict) -> dict:
        ng = data.get("nongaap", {})
        gaap = data.get("gaap", {})
        ng_size = float(ng.get("size", 0))
        gaap_size = float(gaap.get("size", 0))

        if gaap_size == 0:
            return {"score": 1.0, "is_undue": True, "explanation": "GAAP element has zero size"}

        ratio = ng_size / gaap_size
        # Position score: higher on page = more prominent
        ng_pos = float(ng.get("position", 0.5))
        gaap_pos = float(gaap.get("position", 0.5))
        position_advantage = 1.0 if ng_pos < gaap_pos else 0.0  # Lower y = higher on page

        score = (ratio * 0.7) + (position_advantage * 0.3)

        return {
            "score": round(score, 2),
            "size_ratio": round(ratio, 2),
            "position_advantage": position_advantage,
            "is_undue": score > 1.2,
            "explanation": (
                f"Non-GAAP prominence score: {score:.2f} (size ratio {ratio:.1f}x, "
                f"{'higher' if position_advantage else 'lower'} on page). "
                f"{'Undue prominence detected.' if score > 1.2 else 'Prominence is balanced.'}"
            ),
        }
