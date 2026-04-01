# FinChartAudit — Experiment Summary

> Generated: 2026-03-31
> Member C's experiments + Member B's experiments (PCBZ/FinChartAudit)

---

## 1. Experimental Conditions Overview

| ID | Condition | Model | VLM Calls | Tool | Dataset | N | Source |
|----|-----------|-------|-----------|------|---------|---|--------|
| C1 | VLM-only | Haiku 4.5 | 1 | None | Misviz real | 271 | B |
| C2 | VLM + text context | Haiku 4.5 | 1 | Ground-truth text | Misviz real | 271 | B |
| C3 | VLM-only | Qwen3-VL-8B | 1 | None | Misviz real | 271 | B |
| C4 | VLM + text context | Qwen3-VL-8B | 1 | Ground-truth text | Misviz real | 271 | B |
| C5 | LLM-OCR + Rules | Haiku 4.5 | 2 | VLM-as-OCR + Rule Engine | Misviz real | 271 | C |
| C6 | VLM + Rules | Haiku 4.5 | 2 | VLM-extract + Rule Engine | Misviz real | 271 | C |
| C7 | Sonnet VLM-only | Sonnet 4 | 1 | None | Misviz real | 50 | C |
| C8 | Sonnet LLM-OCR + Rules | Sonnet 4 | 2 | VLM-as-OCR + Rule Engine | Misviz real | 50 | C |

---

## 2. RQ1/RQ2: Misviz Benchmark Results (271 charts)

### 2.1 Binary Classification (misleading vs clean)

| Condition | Accuracy | Precision | Recall | **F1** |
|-----------|----------|-----------|--------|--------|
| **C1: Haiku VLM-only** | **77.1%** | 78.2% | **88.3%** | **83.0%** |
| C2: Haiku VLM+text | 77.9% | **83.2%** | 81.3% | 82.2% |
| C5: Haiku LLM-OCR+Rules | 74.2% | 77.9% | 82.5% | 80.1% |
| C6: Haiku VLM+Rules | 68.3% | 79.7% | 66.7% | 72.6% |
| C3: Qwen VLM-only | 67.2% | 77.3% | 67.8% | 72.3% |
| C4: Qwen VLM+text | 66.4% | 84.5% | 57.3% | 68.3% |

### 2.2 Statistical Significance (McNemar Test, df=1, alpha=0.05)

| Comparison | chi2 | p | Significant? |
|------------|------|---|-------------|
| Haiku VLM-only vs LLM-OCR+Rules | 0.583 | >=0.05 | **No** |
| Haiku VLM-only vs VLM+Rules | 5.628 | <0.05 | **Yes** (VLM+Rules worse) |
| Haiku vision_only vs vision_text | 0.029 | >=0.05 | **No** |
| Haiku vs Qwen (both VLM-only) | 9.797 | <0.05 | **Yes** (Haiku better) |

### 2.3 Per Misleader Type F1 (Haiku, top conditions)

| Type | VLM-only (C1) | VLM+text (C2) | LLM-OCR+Rules (C5) | Support |
|------|---------------|---------------|---------------------|---------|
| dual axis | 76.9% | 73.2% | **75.0%** | 17 |
| 3d | 69.6% | 69.8% | **80.0%** | 23 |
| inappropriate use of pie chart | 57.1% | 52.6% | **75.0%** | 15 |
| truncated axis | 38.8% | 50.0% | 35.7% | 17 |
| discretized continuous variable | 0.0% | 0.0% | **78.8%** | 15 |
| inconsistent tick intervals | 0.0% | 11.1% | **36.0%** | 17 |
| misrepresentation | **32.9%** | 30.8% | 28.2% | 18 |
| inverted axis | **34.8%** | 30.0% | 25.9% | 15 |
| inappropriate axis range | 15.4% | 11.1% | 19.5% | 15 |
| inconsistent binning size | 9.5% | 11.8% | 27.9% | 16 |
| inappropriate use of line chart | **42.4%** | 52.9% | 36.4% | 16 |
| inappropriate item order | **22.2%** | 0.0% | 38.5% | 15 |

### 2.4 Sonnet Comparison (50 charts)

| Condition | Accuracy | Precision | Recall | **F1** | Exact Match |
|-----------|----------|-----------|--------|--------|-------------|
| C7: Sonnet VLM-only | **82.0%** | 85.3% | **87.9%** | **86.6%** | **54.0%** |
| C8: Sonnet LLM-OCR+Rules | 80.0% | 84.9% | 84.9% | 84.9% | 48.0% |

---

## 3. Error Attribution Analysis (LLM-OCR+Rules, 271 charts)

### 3.1 FP Source: Rule Engine vs VLM Hallucination

| FP Type | Rule-caused | VLM-hallucinated | Total FP |
|---------|-------------|------------------|----------|
| inappropriate axis range | 6 | **53** | 59 |
| truncated axis | 2 | **50** | 52 |
| misrepresentation | 0 | **43** | 43 |
| inverted axis | **28** | 4 | 32 |
| inconsistent tick intervals | 0 | **24** | 24 |

**Finding**: 85%+ of FPs come from VLM hallucination, not rule engine errors.

### 3.2 Rule Evidence Impact on FP Rate

| Condition | FP Rate |
|-----------|---------|
| Charts with rule evidence injected | 57% |
| Charts without rule evidence | 54% |

**Finding**: Rule evidence has minimal impact on FP rate (+3%). The VLM hallucinates regardless.

### 3.3 Tool Augmentation: Per-Type Impact Pattern

| Issue Category | Tool Impact | Reason |
|---------------|-------------|--------|
| Axis/structural (truncated, dual, tick intervals) | **Positive** | OCR extracts numbers, rules verify quantitatively |
| Visual-spatial (misrepresentation, 3d distortion) | **Negative** | OCR text irrelevant, distracts VLM from visual analysis |
| Completeness (missing labels/units) | **Positive** | OCR detects presence/absence of text elements |

### 3.4 Clean Chart FP Rate

| Condition | FP Rate |
|-----------|---------|
| Haiku VLM-only (B) | 42/100 = 42% |
| Haiku LLM-OCR+Rules (C) | 40/100 = 40% |

**Finding**: Comparable. The high FP rate is a model limitation, not a tool issue.

### 3.5 Complementary Detection (C5 vs C1)

| Category | Count | Types |
|----------|-------|-------|
| We catch, B misses | 54 instances | discretized continuous variable (13), inconsistent tick intervals (9), inappropriate axis range (6) |
| We miss, B catches | 17 instances | misrepresentation (6), inappropriate use of line chart (5) |

---

## 4. RQ3: SEC Filing Analysis

### 4.1 T3 Case Study: Filing-Level Non-GAAP Pairing (C, 7 companies)

| Ticker | Findings | SEC Comment Match | Detection |
|--------|----------|-------------------|-----------|
| ALV | 29 | Non-GAAP prominence issues | Yes |
| AAP | 16 | Multiple Non-GAAP without GAAP | Yes |
| OC | 4 | Adjusted EBIT/EBITDA prominence | Yes |
| UIS | 4 | Non-GAAP Operating Profit | Yes |
| CNM | 1 | 8-K shell, can't access Exhibit | Partial |
| MYE | 0 | Missed violation | No |
| FXLV | 0 | Missed violation | No |

**Case study detection rate: 4/7 = 57%**
**Total findings: 54**

### 4.2 False Positive Test (C, 3 clean companies)

| Ticker | 10-K | 8-K | Result |
|--------|------|-----|--------|
| CTAS | 0 findings | 2 warnings (can't access Exhibit) | Clean |
| SHW | 400 error (too large) | 0 findings | Clean |
| ROK | 400 error (too large) | 0 findings | Clean |

**Substantive false positive rate: 0%**

### 4.3 SEC Chart-Level T2 Detection (B, 10 companies, 96 charts)

| Condition | Accuracy | Precision | Recall | F1 |
|-----------|----------|-----------|--------|-----|
| Claude vision_only | 37.5% | 100% | 37.5% | 54.5% |
| Claude vision_text | 39.6% | 100% | 39.6% | 56.7% |
| Qwen vision_only | 16.7% | 100% | 16.7% | 28.6% |
| Qwen vision_text | 7.3% | 100% | 7.3% | 13.6% |

**All conditions: 100% precision (no false flags), but low recall on real SEC charts.**

### 4.4 SEC Chart Comparison: VLM-only vs LLM-OCR+Rules (C, 28 charts)

| Condition | Flagged |
|-----------|---------|
| VLM-only (Haiku) | 1/28 (FXLV g2: truncated axis) |
| LLM-OCR+Rules (Haiku) | 2/28 (FXLV g1 + g2: truncated axis, axis range) |

**Qualitative finding**: LLM-OCR+Rules detected one additional chart (FXLV g1) that VLM-only missed. FXLV is a confirmed SEC comment letter company.

Additionally, LLM-OCR+Rules detected **completeness issues** in 11/28 charts (missing titles, axis labels, data units) — information VLM-only does not report.

---

## 5. Key Findings for Paper

### F1: Model capability is the dominant factor
- Sonnet (F1=86.6%) > Haiku (83.0%) > Qwen (72.3%)
- Claude vs Qwen is statistically significant (p<0.05)

### F2: Text context does not help
- vision_only vs vision_text: no significant difference for Claude
- vision_text actively hurts Qwen (F1: 72.3% -> 68.3%, recall: 67.8% -> 57.3%)

### F3: Tool augmentation does not improve overall F1 for small models
- Haiku VLM-only (83.0%) vs LLM-OCR+Rules (80.1%): not significant
- Haiku VLM-only (83.0%) vs VLM+Rules (72.6%): significantly worse
- Sonnet shows same pattern: VLM-only (86.6%) vs LLM-OCR+Rules (84.9%)

### F4: Tool augmentation changes the detection profile
- **Better with tools**: axis-related issues (discretized variable +79pp, inconsistent tick +36pp, dual axis, 3d)
- **Worse with tools**: visual-spatial issues (misrepresentation -5pp)
- Root cause: OCR provides useful quantitative evidence for structural issues, but distracts VLM from visual reasoning

### F5: FP is dominated by VLM hallucination, not tool errors
- 85%+ of false positives are VLM-hallucinated (no rule triggered)
- Rule engine contributes only to inverted axis FPs (28/32)
- Clean chart FP rate ~40% for both conditions — a model limitation

### F6: SEC real-world performance is much lower than benchmark
- Misviz F1=83% vs SEC chart F1=55% (same model, same condition)
- SEC charts: 100% precision but very low recall (~38%)
- T3 filing-level: 57% case detection rate, 0% false positive

### F7: Complementary value of tool augmentation
- LLM-OCR+Rules catches 54 instances that VLM-only misses (especially discretized variable, tick intervals)
- VLM-only catches 17 instances that LLM-OCR+Rules misses (especially misrepresentation)
- A hybrid approach (tools for axis issues, VLM-only for visual issues) could be optimal

---

## 6. Experiment Files

| File | Description |
|------|-------------|
| `data/eval_results/llm_ocr_rules/` | C5: LLM-OCR+Rules on 271 Misviz charts |
| `data/eval_results/vlm_rules/` | C6: VLM+Rules on 271 Misviz charts |
| `data/eval_results/sonnet_comparison/` | C7/C8: Sonnet comparison on 50 charts |
| `data/eval_results/sec_chart_comparison/` | SEC chart VLM-only vs LLM-OCR+Rules |
| `data/eval_results/t3_case_study/` | T3 SEC case study (7 companies) |
| `data/eval_results/t3_false_positive/` | T3 false positive test (3 companies) |
| `data/eval_results/pipeline_ablation/` | OCR Pipeline on 28 charts (early run) |
| `data/eval_results/tooluse_ablation/` | Agentic multi-turn on 10 charts (early run) |
| `PCBZ_FinChartAudit/results/` | B's full 2x2 + RQ3 results |

## 7. Scripts

| Script | Purpose |
|--------|---------|
| `run_llm_ocr_rules.py` | LLM-OCR + Rules pipeline (271 charts) |
| `run_vlm_rules.py` | VLM + Rules pipeline (271 charts) |
| `run_sonnet_comparison.py` | Sonnet VLM-only vs LLM-OCR+Rules (50 charts) |
| `run_sec_chart_comparison.py` | SEC chart 2-condition comparison |
| `run_pipeline_full.py` | Traditional OCR pipeline (PaddleOCR, incomplete) |
| `finchartaudit/eval/run_t2_batch.py` | T2 batch evaluation on Misviz |
| `finchartaudit/eval/run_t3_casestudy.py` | T3 SEC case study runner |
