"""Tool definitions for VLM function calling."""
from __future__ import annotations

TOOL_DEFINITIONS: list[dict] = [
    {
        "name": "extract_pdf_text",
        "description": (
            "Extract embedded text from a PDF page. Zero OCR error. "
            "Returns None if scanned. Use FIRST before trying OCR."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "page": {"type": "integer", "description": "Page number (1-indexed)"},
                "extract_tables": {"type": "boolean", "description": "Also extract tables", "default": True},
            },
            "required": ["page"],
        },
    },
    {
        "name": "traditional_ocr",
        "description": (
            "Extract text with precise bounding boxes from an image using OCR. "
            "Returns text + bbox + font size. ONLY tool that provides bounding boxes. "
            "Use for axis labels, data labels, font size comparison."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "image_id": {"type": "string", "description": "Image reference"},
                "region": {
                    "type": "string",
                    "description": "Region to OCR: full, y_axis, x_axis, legend, title, bottom",
                    "default": "full",
                },
                "mode": {
                    "type": "string",
                    "enum": ["text", "bbox", "table", "chart_to_table"],
                    "description": "Output mode",
                    "default": "bbox",
                },
            },
            "required": ["image_id"],
        },
    },
    {
        "name": "doc_ocr",
        "description": (
            "LLM-based OCR for complex/degraded documents. More accurate on messy "
            "layouts but slower, no bounding boxes. Use as fallback."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "image_id": {"type": "string"},
                "mode": {"type": "string", "enum": ["text", "table", "markdown"], "default": "markdown"},
                "focus": {"type": "string", "description": "Specific content to focus on"},
            },
            "required": ["image_id"],
        },
    },
    {
        "name": "query_memory",
        "description": (
            "Query filing memory for previously extracted data, findings, or OCR cache. "
            "Check BEFORE calling OCR to avoid redundant extraction."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "What to look for"},
                "scope": {
                    "type": "string",
                    "enum": ["current_filing", "historical", "all"],
                    "default": "current_filing",
                },
            },
            "required": ["query"],
        },
    },
    {
        "name": "rule_check",
        "description": (
            "Run deterministic rule-based validation on extracted data. "
            "More reliable than VLM for quantitative checks. No hallucination."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "check_type": {
                    "type": "string",
                    "enum": [
                        "truncated_axis", "broken_scale", "font_size_comparison",
                        "value_mismatch", "pairing_completeness", "prominence_score",
                    ],
                },
                "data": {"type": "object", "description": "Data to validate"},
            },
            "required": ["check_type", "data"],
        },
    },
    {
        "name": "query_dspm",
        "description": (
            "Query DSPM Filing Edition for cross-section information. "
            "Zero LLM calls. Use for T4 checks."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "question": {"type": "string"},
                "mode": {
                    "type": "string",
                    "enum": ["context", "changelog", "events", "synthesis"],
                    "default": "context",
                },
            },
            "required": ["question"],
        },
    },
    {
        "name": "html_extract",
        "description": (
            "Extract text, tables, and Non-GAAP/GAAP mentions from an HTML SEC filing. "
            "Returns plain text, parsed tables, and term positions."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "file_path": {"type": "string", "description": "Path to HTML filing"},
            },
            "required": ["file_path"],
        },
    },
]


def get_tool_schemas(tool_names: list[str] | None = None) -> list[dict]:
    """Get tool schemas, optionally filtered by name. Formatted for Claude API."""
    if tool_names is None:
        return TOOL_DEFINITIONS
    return [t for t in TOOL_DEFINITIONS if t["name"] in tool_names]


def get_tool_by_name(name: str) -> dict | None:
    """Get a single tool definition by name."""
    for t in TOOL_DEFINITIONS:
        if t["name"] == name:
            return t
    return None
