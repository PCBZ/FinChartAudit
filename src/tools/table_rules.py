"""Rule checks based on DePlot-extracted data tables.

These rules analyze the STRUCTURED output from DePlot (headers, rows, first column)
rather than raw mixed values, to avoid false positives from mixing axis ticks with data.
"""
from __future__ import annotations

import re


def _extract_column_numbers(rows: list[list[str]], col: int = 0) -> list[float]:
    """Extract numeric values from a specific column of DePlot rows."""
    nums = []
    for row in rows:
        if col < len(row):
            for m in re.findall(r'-?\d+\.?\d*', row[col].replace(",", "")):
                try:
                    nums.append(float(m))
                except ValueError:
                    pass
    return nums


def _is_sequential_axis(values: list[float], min_count: int = 4) -> bool:
    """Check if values look like axis ticks (sequential, mostly monotonic)."""
    if len(values) < min_count:
        return False
    # Check monotonicity (allow 1 violation for noise)
    increasing = sum(1 for i in range(len(values) - 1) if values[i + 1] > values[i])
    decreasing = sum(1 for i in range(len(values) - 1) if values[i + 1] < values[i])
    total = increasing + decreasing
    if total == 0:
        return False
    return max(increasing, decreasing) / total >= 0.7


def check_inconsistent_tick_intervals(axis_values: list[float], tolerance: float = 0.2) -> dict:
    """Check if axis tick values have inconsistent intervals.

    Only operates on values identified as sequential axis ticks.
    """
    if not _is_sequential_axis(axis_values):
        return {"flagged": False, "reason": "not a sequential axis"}

    sorted_vals = sorted(set(axis_values))
    if len(sorted_vals) < 3:
        return {"flagged": False, "reason": "too few unique ticks"}

    intervals = [sorted_vals[i + 1] - sorted_vals[i] for i in range(len(sorted_vals) - 1)]
    intervals = [abs(iv) for iv in intervals if abs(iv) > 1e-9]

    if not intervals:
        return {"flagged": False, "reason": "no intervals"}

    median_iv = sorted(intervals)[len(intervals) // 2]
    if median_iv < 1e-9:
        return {"flagged": False, "reason": "zero median interval"}

    deviations = [abs(iv - median_iv) / median_iv for iv in intervals]
    max_dev = max(deviations)

    return {
        "flagged": max_dev > tolerance,
        "max_deviation": round(max_dev, 3),
        "n_ticks": len(sorted_vals),
        "reason": f"tick deviation {max_dev:.1%}" if max_dev > tolerance else "consistent",
    }


def check_inconsistent_binning(rows: list[list[str]], tolerance: float = 0.3) -> dict:
    """Check if first column contains bin edges with inconsistent widths.

    Only flags if first column looks like numeric bin edges (e.g., "0-10", "10-25", "25-50").
    """
    # Look for range patterns in first column: "10-20", "20-30" etc.
    edges = []
    for row in rows:
        if not row:
            continue
        cell = row[0]
        # Match range patterns: "10-20", "10 - 20", "10 to 20"
        range_match = re.match(r'(-?\d+\.?\d*)\s*[-–to]+\s*(-?\d+\.?\d*)', cell)
        if range_match:
            edges.append(float(range_match.group(1)))
            edges.append(float(range_match.group(2)))

    if not edges:
        # Try plain numeric first column as bin edges
        col0 = _extract_column_numbers(rows, 0)
        if _is_sequential_axis(col0) and len(col0) >= 4:
            edges = sorted(set(col0))
        else:
            return {"flagged": False, "reason": "no bin edges found"}

    edges = sorted(set(edges))
    if len(edges) < 3:
        return {"flagged": False, "reason": "too few bin edges"}

    widths = [edges[i + 1] - edges[i] for i in range(len(edges) - 1)]
    widths = [abs(w) for w in widths if abs(w) > 1e-9]

    if not widths:
        return {"flagged": False, "reason": "no widths"}

    median_w = sorted(widths)[len(widths) // 2]
    if median_w < 1e-9:
        return {"flagged": False, "reason": "zero median width"}

    deviations = [abs(w - median_w) / median_w for w in widths]
    max_dev = max(deviations)

    return {
        "flagged": max_dev > tolerance,
        "max_deviation": round(max_dev, 3),
        "n_edges": len(edges),
        "reason": f"bin width deviation {max_dev:.1%}" if max_dev > tolerance else "consistent",
    }


def check_inappropriate_axis_range(rows: list[list[str]]) -> dict:
    """Check if data values cluster in a narrow range suggesting axis manipulation.

    Extracts all data values (non-first-column), checks if the spread is very small
    relative to the values themselves (e.g., values 95-100 on a 0-100 scale).
    """
    data_vals = []
    for row in rows:
        for cell in row[1:]:  # Skip first column (labels)
            for m in re.findall(r'-?\d+\.?\d*', cell.replace(",", "").replace("%", "")):
                try:
                    v = float(m)
                    if abs(v) < 1e6:  # Filter out year-like values
                        data_vals.append(v)
                except ValueError:
                    pass

    if len(data_vals) < 3:
        return {"flagged": False, "reason": "too few data values"}

    data_min = min(data_vals)
    data_max = max(data_vals)
    data_range = data_max - data_min

    if data_max <= 0:
        return {"flagged": False, "reason": "no positive data"}

    # Flag if: range is small relative to max, AND min is far from zero
    range_ratio = data_range / data_max if data_max > 0 else 0
    min_ratio = data_min / data_max if data_max > 0 else 0

    flagged = range_ratio < 0.15 and min_ratio > 0.5

    return {
        "flagged": flagged,
        "range_ratio": round(range_ratio, 3),
        "min_ratio": round(min_ratio, 3),
        "data_range": f"{data_min:.1f}-{data_max:.1f}",
        "reason": f"narrow range ({range_ratio:.1%} of max, min={min_ratio:.1%})" if flagged else "reasonable range",
    }


def check_inverted_axis(rows: list[list[str]]) -> dict:
    """Check if first column values are in inverted order.

    DePlot reads top-to-bottom. For Y-axis data, normal = decreasing.
    If first column is strongly increasing, axis may be inverted.
    """
    col0 = _extract_column_numbers(rows, 0)

    if len(col0) < 4:
        return {"flagged": False, "reason": "too few values"}

    if not _is_sequential_axis(col0):
        return {"flagged": False, "reason": "not sequential"}

    increasing = sum(1 for i in range(len(col0) - 1) if col0[i + 1] > col0[i])
    decreasing = sum(1 for i in range(len(col0) - 1) if col0[i + 1] < col0[i])

    # For Y-axis read top-to-bottom: normal = decreasing, inverted = increasing
    # But X-axis is normally increasing (left-to-right) — so only flag strong cases
    total = increasing + decreasing
    if total == 0:
        return {"flagged": False, "reason": "no ordering"}

    # Only flag if STRONGLY increasing AND values look like Y-axis (not years, not categories)
    is_year = all(1900 <= v <= 2100 for v in col0)
    flagged = (increasing / total > 0.85) and (not is_year) and len(col0) >= 5

    return {
        "flagged": flagged,
        "increasing": increasing,
        "decreasing": decreasing,
        "is_year": is_year,
        "reason": f"strongly increasing ({increasing}/{total})" if flagged else "normal order",
    }


def analyze_deplot_table(deplot_result: dict) -> dict:
    """Run all table-based rule checks on DePlot output.

    Uses structured rows/headers instead of raw values to avoid mixing
    axis ticks with data values.
    """
    rows = deplot_result.get("rows", [])
    if not rows:
        return {}

    results = {}

    # Tick intervals: check first column as potential axis
    col0 = _extract_column_numbers(rows, 0)
    if _is_sequential_axis(col0):
        results["tick_intervals"] = check_inconsistent_tick_intervals(col0)

    # Binning: check for bin edge patterns
    results["binning"] = check_inconsistent_binning(rows)

    # Axis range: analyze data value spread
    results["axis_range"] = check_inappropriate_axis_range(rows)

    # Inverted axis: check first column order
    results["inverted_axis"] = check_inverted_axis(rows)

    return results
