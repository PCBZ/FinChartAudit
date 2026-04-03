"""FinChartAudit — Streamlit Demo Application.

Two pages:
  1. Single Chart Audit — upload a chart image, run T2 visual encoding detection
  2. Filing Scanner — upload/select an SEC filing, run T3 Non-GAAP pairing analysis
"""
import json
import os
import sys
import time
from pathlib import Path

_project_root = str(Path(__file__).resolve().parent.parent)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

import streamlit as st
from PIL import Image

os.environ.setdefault("PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK", "True")

# ── Constants ──

RISK_COLORS = {
    "HIGH": "#e74c3c", "CRITICAL": "#c0392b",
    "MEDIUM": "#f39c12", "LOW": "#3498db",
    "RiskLevel.HIGH": "#e74c3c", "RiskLevel.CRITICAL": "#c0392b",
    "RiskLevel.MEDIUM": "#f39c12", "RiskLevel.LOW": "#3498db",
}

TIER_DESCRIPTIONS = {
    "T1": "Numerical Consistency",
    "T2": "Visual Encoding",
    "T3": "Non-GAAP Pairing",
}


# ── Initialization ──

def init_components():
    """Initialize VLM, OCR, agents (cached in session state)."""
    if "initialized" in st.session_state:
        return

    from finchartaudit.config import get_config
    from finchartaudit.vlm.claude_client import OpenRouterVLMClient
    from finchartaudit.memory.filing_memory import FilingMemory
    from finchartaudit.tools.traditional_ocr import TraditionalOCRTool

    config = get_config()

    if not config.openrouter_api_key:
        st.error("Set FCA_OPENROUTER_API_KEY in .env or environment")
        st.stop()

    vlm = OpenRouterVLMClient(
        api_key=config.openrouter_api_key,
        model=config.vlm_model,
        base_url=config.openrouter_base_url,
    )

    ocr = TraditionalOCRTool()
    memory = FilingMemory()

    st.session_state.vlm = vlm
    st.session_state.ocr = ocr
    st.session_state.memory = memory
    st.session_state.config = config
    st.session_state.initialized = True


def get_t2_agent(memory):
    """Create T2 agent based on selected mode."""
    mode = st.session_state.get("t2_mode", "Pipeline")
    if mode == "Pipeline":
        from finchartaudit.agents.t2_pipeline import T2PipelineAgent
        agent = T2PipelineAgent(vlm=st.session_state.vlm, memory=memory)
    else:
        from finchartaudit.agents.t2_visual import T2VisualAgent
        agent = T2VisualAgent(vlm=st.session_state.vlm, memory=memory)
    agent.set_ocr_tool(st.session_state.ocr)
    return agent


def reset_memory():
    """Reset memory for a new analysis."""
    from finchartaudit.memory.filing_memory import FilingMemory
    memory = FilingMemory()
    st.session_state.memory = memory
    return memory


# ── Shared UI Components ──

def render_finding(f, show_tier=False):
    """Render a single AuditFinding as styled HTML."""
    risk_str = str(f.risk_level).replace("RiskLevel.", "")
    color = RISK_COLORS.get(risk_str, RISK_COLORS.get(str(f.risk_level), "#95a5a6"))

    tier_badge = ""
    if show_tier:
        tier_label = TIER_DESCRIPTIONS.get(str(f.tier).replace("Tier.", ""), f.tier)
        tier_badge = (f'<span style="background: #e8e8e8; padding: 2px 8px; '
                      f'border-radius: 3px; font-size: 0.78em; margin-right: 6px;">'
                      f'{f.tier} {tier_label}</span>')

    category_badge = (f'<span style="background: #f0f0f0; padding: 2px 6px; '
                      f'border-radius: 3px; font-size: 0.78em; margin-left: 8px;">'
                      f'{f.category}</span>')

    sec_ref = ""
    for ev in f.evidence:
        if isinstance(ev, str) and "SEC basis:" in ev:
            sec_ref = ev
            break
    sec_html = (f'<br/><span style="color: #8e44ad; font-size: 0.85em;">'
                f'{sec_ref}</span>' if sec_ref else "")

    st.markdown(
        f'<div style="border-left: 4px solid {color}; padding: 12px; '
        f'margin: 8px 0; background: #f8f9fa; border-radius: 4px;">'
        f'{tier_badge}'
        f'<span style="color: {color}; font-weight: bold;">{risk_str}</span> '
        f'&mdash; <strong>{f.subcategory}</strong>{category_badge} '
        f'<span style="color: #666;">(confidence: {f.confidence:.0%})</span><br/>'
        f'<span>{f.description}</span><br/>'
        f'<span style="color: #27ae60; font-size: 0.9em;">'
        f'Suggestion: {f.correction}</span>'
        f'{sec_html}</div>',
        unsafe_allow_html=True,
    )


def render_trace(memory):
    """Render the audit trace."""
    trace = memory.audit_trace.get_trace()
    if not trace:
        st.info("No trace entries.")
        return

    for i, entry in enumerate(trace, 1):
        icon = {
            "tool_call": "🔧", "tool_result": "📋", "vlm_reasoning": "🧠",
            "finding": "🎯", "decision": "⚡",
        }.get(entry.action, "•")

        tool_info = f" **[{entry.tool_name}]**" if entry.tool_name else ""
        with st.expander(f"{icon} Step {i}: {entry.action}{tool_info}", expanded=(i <= 3)):
            if entry.input_summary:
                st.markdown(f"**Input:** `{entry.input_summary[:300]}`")
            if entry.output_summary:
                st.markdown(f"**Output:** `{entry.output_summary[:500]}`")
            if entry.decision:
                st.markdown(f"**Decision:** {entry.decision}")


def render_json_export(memory, filename="finchartaudit_results.json"):
    """Render JSON export tab."""
    export = memory.export_json()
    st.json(export)
    st.download_button(
        "Download JSON",
        data=json.dumps(export, indent=2, default=str),
        file_name=filename,
        mime="application/json",
    )


def sort_findings(findings):
    """Sort findings by risk level (CRITICAL first)."""
    order = {"CRITICAL": 0, "HIGH": 1, "MEDIUM": 2, "LOW": 3}
    return sorted(findings, key=lambda f: order.get(
        str(f.risk_level).replace("RiskLevel.", ""), 3))


# ── Main ──

def main():
    st.set_page_config(page_title="FinChartAudit", page_icon="📊", layout="wide")

    with st.sidebar:
        st.title("📊 FinChartAudit")
        st.caption("Cross-modal consistency verification for SEC financial filings")
        st.divider()

        page = st.radio("Navigation", ["Single Chart Audit", "Filing Scanner"], index=0)

        st.divider()
        st.subheader("Settings")
        init_components()

        config = st.session_state.config
        model_display = config.vlm_model.split("/")[-1]
        st.text(f"Model: {model_display}")

        ocr_status = "✅" if st.session_state.ocr.is_available else "❌"
        st.text(f"OCR: {st.session_state.ocr._backend} {ocr_status}")

        st.selectbox(
            "T2 Detection Mode",
            ["Pipeline", "Agentic (multi-turn)"],
            index=0,
            key="t2_mode",
            help="Pipeline: OCR+Rules pre-computed, single VLM call (faster, more accurate)\n"
                 "Agentic: VLM autonomously calls tools in multi-turn loop",
        )

        st.divider()
        st.markdown(
            '<span style="font-size: 0.8em; color: #888;">'
            'CS 6180 Generative AI | Cornell University<br/>'
            'FinChartAudit: Detecting Misleading Financial Charts</span>',
            unsafe_allow_html=True,
        )

    if page == "Single Chart Audit":
        page_single_chart()
    else:
        page_filing_scanner()


# ── Page 1: Single Chart Audit ──

def page_single_chart():
    st.header("Single Chart Audit")
    st.markdown(
        "Upload a chart image to detect **misleading visualization techniques** "
        "and **completeness issues**. The system uses VLM + PaddleOCR + rule engine."
    )

    col_upload, col_options = st.columns([3, 1])

    with col_upload:
        uploaded = st.file_uploader(
            "Upload chart image",
            type=["png", "jpg", "jpeg", "webp"],
            help="Drag and drop a chart image or click to browse",
        )

    with col_options:
        text_context = ""
        with st.expander("Text Context (optional)"):
            text_context = st.text_area(
                "Ground-truth data values",
                placeholder="e.g., Revenue: Q1=120M, Q2=125M",
                height=80,
                label_visibility="collapsed",
            )

    if not uploaded:
        st.info("Upload a chart image to begin, or try a sample chart below.")
        show_sample_selector()
        return

    import tempfile
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
        f.write(uploaded.read())
        image_path = f.name

    run_chart_analysis(image_path, text_context)


def show_sample_selector():
    """Show sample Misviz charts and SEC filing charts for quick testing."""
    # SEC filing charts
    sec_charts_dir = Path("data/charts")
    sec_samples = []
    if sec_charts_dir.exists():
        for company_dir in sorted(sec_charts_dir.iterdir()):
            if not company_dir.is_dir():
                continue
            for img in company_dir.glob("*"):
                if img.suffix.lower() in (".jpg", ".png", ".gif") and img.stat().st_size > 10000:
                    sec_samples.append({
                        "path": str(img),
                        "label": f"{company_dir.name}/{img.name}",
                    })

    if sec_samples:
        st.subheader("SEC Filing Charts")
        cols = st.columns(min(len(sec_samples), 4))
        for i, sample in enumerate(sec_samples[:4]):
            with cols[i % 4]:
                try:
                    img = Image.open(sample["path"])
                    st.image(img, caption=sample["label"], use_container_width=True)
                    if st.button("Analyze", key=f"sec_sample_{i}"):
                        run_chart_analysis(sample["path"], "")
                except Exception:
                    pass

    # Misviz samples
    misviz_json = Path("data/misviz/misviz.json")
    if not misviz_json.exists():
        return

    data = json.loads(misviz_json.read_text(encoding="utf-8"))
    samples = {}
    for d in data:
        for m in d.get("misleader", []):
            if m not in samples:
                img_name = d["image_path"].split("/")[-1]
                img_path = Path("data/misviz/img") / img_name
                if img_path.exists():
                    samples[m] = str(img_path)
        if len(samples) >= 6:
            break

    if not samples:
        return

    st.subheader("Misviz Benchmark Samples")
    cols = st.columns(min(len(samples), 3))
    for i, (misleader, path) in enumerate(list(samples.items())[:6]):
        with cols[i % 3]:
            try:
                img = Image.open(path)
                st.image(img, caption=misleader, use_container_width=True)
                if st.button("Analyze", key=f"misviz_sample_{i}"):
                    run_chart_analysis(path, "")
            except Exception:
                pass


def run_chart_analysis(image_path: str, text_context: str = ""):
    """Run T2 analysis on a chart image."""
    memory = reset_memory()
    agent = get_t2_agent(memory)
    mode = st.session_state.get("t2_mode", "Pipeline")

    col_img, col_results = st.columns([1, 1])

    with col_img:
        st.subheader("Chart Image")
        try:
            img = Image.open(image_path)
            st.image(img, use_container_width=True)
        except Exception as e:
            st.error(f"Cannot load image: {e}")
            return

    with col_results:
        st.subheader("Analysis")

        mode_label = "Pipeline (OCR → Rules → VLM)" if mode == "Pipeline" else "Agentic (multi-turn tool-use)"
        status = st.status(f"Running T2 audit [{mode_label}]...", expanded=True)
        start = time.time()

        try:
            with status:
                st.write(f"Mode: **{mode_label}**")
                findings = agent.execute({
                    "image_path": image_path,
                    "page": 1,
                    "chart_id": "uploaded_chart",
                })
                elapsed = time.time() - start

                trace = memory.audit_trace.get_trace()
                tool_calls = [e for e in trace if e.action == "tool_call"]
                st.write(f"Done: {len(tool_calls)} tool calls, {elapsed:.1f}s")

            status.update(label=f"Analysis complete ({elapsed:.1f}s)", state="complete")
        except Exception as e:
            status.update(label="Analysis failed", state="error")
            st.error(f"Error: {e}")
            import traceback
            st.code(traceback.format_exc())
            return

    # Summary metrics
    summary = memory.get_summary()
    mcol1, mcol2, mcol3, mcol4 = st.columns(4)
    mcol1.metric("Findings", summary["total_findings"])

    misleader_count = len([f for f in findings if f.category == "misleader"])
    completeness_count = len([f for f in findings if f.category == "completeness"])
    mcol2.metric("Misleaders", misleader_count)
    mcol3.metric("Completeness Issues", completeness_count)

    risk_dist = summary.get("findings_by_risk", {})
    high_count = sum(v for k, v in risk_dist.items() if "HIGH" in str(k) or "CRITICAL" in str(k))
    mcol4.metric("High/Critical", high_count)

    # Tabs
    st.divider()
    tab_findings, tab_trace, tab_json = st.tabs(["Findings", "Audit Trace", "Raw JSON"])

    with tab_findings:
        if not findings:
            st.success("No misleading techniques detected. Chart appears compliant.")
        else:
            # Separate misleaders and completeness
            misleaders = [f for f in findings if f.category == "misleader"]
            completeness = [f for f in findings if f.category == "completeness"]

            if misleaders:
                st.markdown("#### Misleading Visual Encoding")
                for f in sort_findings(misleaders):
                    render_finding(f)

            if completeness:
                st.markdown("#### Completeness Issues")
                for f in sort_findings(completeness):
                    render_finding(f)

    with tab_trace:
        render_trace(memory)

    with tab_json:
        render_json_export(memory)


# ── Page 2: Filing Scanner ──

def page_filing_scanner():
    st.header("Filing Scanner")
    st.markdown(
        "Upload or select an SEC filing to run **T3 Non-GAAP pairing analysis**. "
        "Detects prominence violations, missing GAAP counterparts, and reconciliation gaps."
    )

    col1, col2 = st.columns([2, 1])
    with col1:
        uploaded = st.file_uploader(
            "Upload SEC filing",
            type=["htm", "html", "pdf"],
            help="Upload an 8-K, 10-K, or DEF 14A filing",
        )
    with col2:
        ticker = st.text_input("Ticker (optional)", placeholder="e.g., MYE")
        filing_type = st.selectbox("Filing Type", ["8-K", "10-K", "DEF 14A", "20-F", "6-K"])

    # Filing selector from downloaded data
    with st.expander("Or select from downloaded filings"):
        filings_dir = Path("data/filings")
        if filings_dir.exists():
            companies = sorted([d.name for d in filings_dir.iterdir() if d.is_dir()])
            selected_company = st.selectbox("Company", [""] + companies)
            if selected_company:
                company_dir = filings_dir / selected_company
                filing_subdir = company_dir / "filing"
                files = sorted([
                    f.name for f in filing_subdir.glob("*")
                    if f.suffix in (".htm", ".html", ".pdf") and "_meta" not in f.name
                ]) if filing_subdir.exists() else []
                selected_file = st.selectbox("Filing", [""] + files)
                if selected_file and st.button("Load selected filing"):
                    st.session_state.scanner_file = str(filing_subdir / selected_file)
                    st.session_state.scanner_ticker = selected_company
                    st.rerun()

    # Resolve file path
    file_path = None
    if uploaded:
        import tempfile
        suffix = "." + uploaded.name.split(".")[-1]
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as f:
            f.write(uploaded.read())
            file_path = f.name
    elif "scanner_file" in st.session_state:
        file_path = st.session_state.scanner_file
        ticker = ticker or st.session_state.get("scanner_ticker", "")

    if not file_path:
        st.info("Upload a filing or select from downloaded filings to begin.")
        return

    # Show file info
    fp = Path(file_path)
    st.markdown(f"**Selected:** `{fp.name}` ({fp.stat().st_size / 1024:.0f} KB)")

    if st.button("Run Filing Audit", type="primary"):
        run_filing_analysis(file_path, ticker, filing_type)


def run_filing_analysis(file_path: str, ticker: str, filing_type: str):
    """Run T3 pairing analysis on a filing."""
    from finchartaudit.agents.orchestrator import Orchestrator

    memory = reset_memory()
    orchestrator = Orchestrator(vlm=st.session_state.vlm, memory=memory)
    orchestrator.set_ocr_tool(st.session_state.ocr)

    status = st.status("Running Filing Audit (T3 Non-GAAP Pairing)...", expanded=True)
    start = time.time()

    try:
        with status:
            st.write(f"Analyzing **{Path(file_path).name}**...")
            st.write("Extracting text, tables, and Non-GAAP mentions...")
            findings = orchestrator.audit_filing(
                file_path=file_path, ticker=ticker, filing_type=filing_type)
            elapsed = time.time() - start
            st.write(f"Done in {elapsed:.1f}s")
        status.update(label=f"Filing audit complete ({elapsed:.1f}s)", state="complete")
    except Exception as e:
        status.update(label="Audit failed", state="error")
        st.error(f"Error: {e}")
        import traceback
        st.code(traceback.format_exc())
        return

    # Summary
    summary = memory.get_summary()
    mcol1, mcol2, mcol3, mcol4 = st.columns(4)
    mcol1.metric("Total Findings", summary["total_findings"])

    t3_count = len([f for f in findings if str(f.tier).replace("Tier.", "") == "T3"])
    mcol2.metric("T3 Pairing Issues", t3_count)

    risk_dist = summary.get("findings_by_risk", {})
    high_count = sum(v for k, v in risk_dist.items() if "HIGH" in str(k) or "CRITICAL" in str(k))
    mcol3.metric("High/Critical", high_count)
    mcol4.metric("Pairings Tracked", len(memory.pairing_matrix))

    # Results
    st.divider()
    tab_findings, tab_trace, tab_json = st.tabs(["Findings", "Audit Trace", "Raw JSON"])

    with tab_findings:
        if not findings:
            st.success("No Non-GAAP compliance violations detected.")
        else:
            # Group by subcategory
            missing = [f for f in findings if "missing" in f.subcategory]
            prominence = [f for f in findings if "prominence" in f.subcategory]
            other = [f for f in findings if f not in missing and f not in prominence]

            if missing:
                st.markdown("#### Missing GAAP Counterparts")
                for f in sort_findings(missing):
                    render_finding(f, show_tier=True)

            if prominence:
                st.markdown("#### Undue Prominence")
                for f in sort_findings(prominence):
                    render_finding(f, show_tier=True)

            if other:
                st.markdown("#### Other Violations")
                for f in sort_findings(other):
                    render_finding(f, show_tier=True)

    with tab_trace:
        render_trace(memory)

    with tab_json:
        render_json_export(memory, f"finchartaudit_{ticker or 'filing'}_results.json")


if __name__ == "__main__":
    main()
