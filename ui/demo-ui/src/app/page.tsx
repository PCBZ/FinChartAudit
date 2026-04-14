"use client";

import { useState, useCallback, useRef, useEffect } from "react";
import { createPortal } from "react-dom";
import {
  BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip,
  ResponsiveContainer, Cell,
} from "recharts";
import {
  Upload, BarChart2, FileText, ChevronRight,
  AlertTriangle, CheckCircle, Shield, Eye,
  Download, BookOpen, TrendingUp, AlertCircle,
  Building2, Scale, X, Play, Loader2, FolderOpen, HelpCircle,
} from "lucide-react";

// ── Static compliance data built from real SEC results ────────────────────────

// Real experiment data from sec_claude_vision_text.json
const UPS_FINDINGS = [
  { id: "F-001", file: "table_019.png", severity: "HIGH",   rule: "Reg G / Item 10(e)",  type: "Non-GAAP Prominence",    dim: "nongaap", summary: "Non-GAAP Adjusted Operating Expenses ($9,444M) and Adjusted Operating Profit ($1,122M) presented more prominently than GAAP measures." },
  { id: "F-002", file: "table_021.png", severity: "HIGH",   rule: "Reg G / Item 10(e)",  type: "Non-GAAP Prominence",    dim: "nongaap", summary: "Non-GAAP Adjusted Compensation and Benefits ($48,185M) and Adjusted Total Operating Expenses presented with equal or greater visual prominence alongside GAAP figures." },
  { id: "F-003", file: "table_016.png", severity: "HIGH",   rule: "Reg G / Item 10(e)",  type: "Non-GAAP Prominence",    dim: "nongaap", summary: "Non-GAAP Adjusted Operating Expenses, Adjusted Operating Profit, and Adjusted Operating Margin presented with identical visual prominence to GAAP counterparts." },
  { id: "F-004", file: "table_014.png", severity: "HIGH",   rule: "Reg G / Item 10(e)",  type: "Non-GAAP Prominence",    dim: "nongaap", summary: "Non-GAAP Adjusted Operating Profit ($4,516M) and Adjusted Operating Margin (7.5%) presented immediately adjacent to GAAP equivalents with equal prominence." },
  { id: "F-005", file: "table_010.png", severity: "MEDIUM", rule: "Regulation G",         type: "Disclosure Violation",   dim: "nongaap", summary: "Non-GAAP adjustments table presents adjusted measures without sufficient clarity distinguishing them from GAAP measures." },
  { id: "F-006", file: "table_009.png", severity: "MEDIUM", rule: "Regulation G",         type: "Disclosure Violation",   dim: "nongaap", summary: "Non-GAAP adjustments table presents reconciling items without sufficient context showing the comparable GAAP measures." },
  { id: "F-007", file: "table_020.png", severity: "MEDIUM", rule: "Regulation G",         type: "Disclosure Violation",   dim: "nongaap", summary: "Non-GAAP adjustments to operating expenses presented without clearly differentiating adjustments from comparable GAAP figures." },
  { id: "F-008", file: "table_011.png", severity: "MEDIUM", rule: "Regulation G",         type: "Disclosure Violation",   dim: "nongaap", summary: "Non-GAAP adjustments to income tax presented with multiple transformation strategy cost line items without clear GAAP baseline context." },
];

const STZ_FINDINGS = [
  { id: "F-001", file: "stz-20240229_g22.jpg", severity: "HIGH",   rule: "Visualization Standards", type: "Truncated Axis",         dim: "visual", summary: "Y-axis begins at $11,500M instead of $0, exaggerating visual differences between bars. The $12,461M bar appears dramatically taller than the $11,500M bar despite only ~8% actual difference." },
  { id: "F-002", file: "stz-20250228_g20.jpg", severity: "HIGH",   rule: "Visualization Standards", type: "Truncated Axis",         dim: "visual", summary: "Y-axis severely truncated starting at $10,500M. Revenue bars ranging $11,497M–$11,879M appear to show dramatic growth when actual variance is less than 4%." },
  { id: "F-003", file: "stz-20250228_g21.jpg", severity: "MEDIUM", rule: "Visualization Standards", type: "Inconsistent Intervals", dim: "visual", summary: "Debt maturity chart displays years in non-sequential, irregular pattern (2033, 2047, 2048, 2050), obscuring actual maturity distribution." },
  { id: "F-004", file: "stz-20250228_g6.jpg",  severity: "MEDIUM", rule: "Visualization Standards", type: "Pie Chart Misuse",       dim: "visual", summary: "Pie chart used for Consolidated Net Sales composition where Beer dominates at 83.7%, making proportional comparison of other segments visually meaningless." },
  { id: "F-005", file: "stz-20240229_g8.jpg",  severity: "MEDIUM", rule: "Visualization Standards", type: "Pie Chart Misuse",       dim: "visual", summary: "Pie chart presents packaging material composition (Glass 58%, Aluminum 39%, Steel 3%) — data that does not represent parts of a financially meaningful whole." },
  { id: "F-006", file: "stz-20240229_g5.jpg",  severity: "MEDIUM", rule: "Visualization Standards", type: "Pie Chart Misuse",       dim: "visual", summary: "Donut chart shows Fiscal 2023 Net Sales across three beverage categories where Beer at 79% extreme dominance makes the pie format misleading." },
  { id: "F-007", file: "stz-20250228_g5.jpg",  severity: "MEDIUM", rule: "Visualization Standards", type: "Pie Chart Misuse",       dim: "visual", summary: "Pie chart presents product category breakdowns (Beer 83.7%, Wine 14.2%, Spirits 2.1%) — appropriate as part-to-whole but extreme dominance distorts visual comparison." },
  { id: "F-008", file: "stz-20250228_g9.jpg",  severity: "LOW",    rule: "Visualization Standards", type: "Pie Chart Misuse",       dim: "visual", summary: "Donut chart presents geographic distribution data (Mexico 57%, U.S. 38%, Other 5%) — pie format is suboptimal for regional comparison." },
  { id: "F-009", file: "stz-20240229_g9.jpg",  severity: "LOW",    rule: "Visualization Standards", type: "Pie Chart Misuse",       dim: "visual", summary: "Donut chart for geographic distribution (Mexico 57%, U.S. 38%, Other Non-U.S. 5%) — bar chart would better support comparison." },
  { id: "F-010", file: "stz-20240229_g22.jpg", severity: "HIGH",   rule: "Reg G / Item 10(e)",      type: "Non-GAAP Prominence",    dim: "nongaap", summary: "Chart displays multiple financial line items with inconsistent bar heights that do not clearly correspond to their labeled values, combined with Non-GAAP prominence concerns." },
];

// Real experiment data — flag rates from sec_claude_vision_text evaluation
const COMPANY_RISK = [
  { ticker: "STZ",  name: "Constellation Brands",  flagged: 10, total: 10, flagRate: 1.00, type: "Mixed"     },
  { ticker: "UPS",  name: "United Parcel Service",  flagged: 8,  total: 10, flagRate: 0.80, type: "Non-GAAP"  },
  { ticker: "BCO",  name: "Brink's Company",        flagged: 7,  total: 10, flagRate: 0.70, type: "Non-GAAP"  },
  { ticker: "VERI", name: "Veritiv Corp",            flagged: 2,  total: 7,  flagRate: 0.29, type: "Non-GAAP"  },
  { ticker: "KMI",  name: "Kinder Morgan",           flagged: 2,  total: 9,  flagRate: 0.22, type: "Non-GAAP"  },
  { ticker: "LYFT", name: "Lyft Inc.",               flagged: 2,  total: 10, flagRate: 0.20, type: "Non-GAAP"  },
  { ticker: "IRBT", name: "iRobot Corp",             flagged: 2,  total: 10, flagRate: 0.20, type: "Non-GAAP"  },
  { ticker: "LITE", name: "Lumentum Holdings",       flagged: 1,  total: 10, flagRate: 0.10, type: "Non-GAAP"  },
  { ticker: "CHPT", name: "ChargePoint Holdings",    flagged: 0,  total: 10, flagRate: 0.00, type: "—"         },
  { ticker: "OGN",  name: "Organon & Co.",           flagged: 0,  total: 10, flagRate: 0.00, type: "—"         },
];

const MISLEADER_LABELS: Record<string, string> = {
  "misrepresentation": "Misrepresentation", "3d": "3D Distortion",
  "truncated axis": "Truncated Axis", "inappropriate use of pie chart": "Pie Chart Misuse",
  "inconsistent tick intervals": "Inconsistent Ticks", "dual axis": "Dual Axis Abuse",
  "inconsistent binning size": "Inconsistent Bins", "discretized continuous variable": "Discretized Variable",
  "inappropriate use of line chart": "Line Chart Misuse", "inappropriate item order": "Inappropriate Order",
  "inverted axis": "Inverted Axis", "inappropriate axis range": "Axis Range Abuse",
};

const MISLEADER_DETAILS: Array<{ key: string; label: string; short: string; desc: string; example: string; color: string }> = [
  {
    key: "truncated axis", label: "Truncated Axis", short: "Y-axis doesn't start at zero",
    desc: "The y-axis begins at a non-zero value, compressing the visible range so that small differences appear dramatically larger than they actually are.",
    example: "A revenue bar chart starting at $10.5B makes a 3% year-over-year gain look like a 200% jump visually.",
    color: "#f87171",
  },
  {
    key: "misrepresentation", label: "Misrepresentation", short: "Bar/area sizes don't match labeled values",
    desc: "The visual size of chart elements (bar height, area, bubble) is inconsistent with the numeric values they represent, creating a false impression of magnitude.",
    example: "A bar labeled '$12M' is drawn twice as tall as a bar labeled '$10M', implying a 100% difference instead of 20%.",
    color: "#f87171",
  },
  {
    key: "3d", label: "3D Distortion", short: "3D effects distort visual comparison",
    desc: "Three-dimensional rendering adds perspective depth that makes elements closer to the viewer appear larger, breaking the accurate proportional relationship between data and visual area.",
    example: "A 3D pie chart's front slice looks nearly 30% larger than the back slice even when both represent 25% of the total.",
    color: "#fb923c",
  },
  {
    key: "dual axis", label: "Dual Axis Abuse", short: "Two y-axes with incompatible scales",
    desc: "Two separate y-axes with different scales are overlaid on the same chart, allowing the presenter to choose scales that make unrelated series appear correlated or divergent.",
    example: "Net income ($M) and stock price ($) plotted on a dual-axis chart where adjusting the right scale makes earnings appear to perfectly track share price.",
    color: "#fb923c",
  },
  {
    key: "inappropriate use of pie chart", label: "Pie Chart Misuse", short: "Pie used for non-part-to-whole data",
    desc: "Pie charts are only meaningful when all segments sum to a relevant 100% whole. Using them for independent metrics, comparisons, or data that doesn't represent a compositional whole creates false part-to-whole impressions.",
    example: "A pie chart showing five regional office revenues where the regions are not mutually exclusive segments of a total addressable market.",
    color: "#fbbf24",
  },
  {
    key: "inappropriate axis range", label: "Axis Range Abuse", short: "Range exaggerates or hides variation",
    desc: "The axis range is manually set far narrower or wider than the data's natural range, either amplifying minor fluctuations into apparent crises or suppressing real differences into apparent flatness.",
    example: "Market share plotted on a 98%–102% axis makes a 1% loss look like a catastrophic collapse across the chart.",
    color: "#fbbf24",
  },
  {
    key: "inverted axis", label: "Inverted Axis", short: "Axis direction reversed",
    desc: "The axis runs in the opposite direction from convention (e.g., high values at the bottom), so a visually upward trend actually represents declining data.",
    example: "A cost reduction chart with the y-axis running 100→0 top-to-bottom makes rising costs appear as a downward (good) trend.",
    color: "#a78bfa",
  },
  {
    key: "inconsistent tick intervals", label: "Inconsistent Ticks", short: "Unevenly spaced axis ticks",
    desc: "Axis tick marks are placed at irregular numeric intervals while appearing evenly spaced visually. This distorts the perceived rate of change between any two points on the chart.",
    example: "A debt maturity chart labeled 2025, 2027, 2030, 2033, 2040 with equal horizontal spacing makes 3-year gaps look identical to 7-year gaps.",
    color: "#a78bfa",
  },
  {
    key: "inappropriate use of line chart", label: "Line Chart Misuse", short: "Line chart used for categorical data",
    desc: "Line charts imply continuity and trend between connected points. Using them for discrete, categorical, or non-sequential data creates a false impression of interpolation or progression.",
    example: "Connecting revenue figures for five unrelated business units with a line chart implies a sequential relationship (Unit A → B → C) that doesn't exist.",
    color: "#38bdf8",
  },
  {
    key: "inappropriate item order", label: "Inappropriate Order", short: "Ordering creates false narrative",
    desc: "Items in a chart are sorted in a non-natural order (e.g., value-sorted instead of chronological, or reverse-alphabetical) to create a desired visual impression that wouldn't exist with neutral ordering.",
    example: "Sorting quarterly EPS bars from highest to lowest instead of Q1→Q4 makes an overall declining year look like a performance peak.",
    color: "#38bdf8",
  },
  {
    key: "inconsistent binning size", label: "Inconsistent Bins", short: "Histogram bins have unequal widths",
    desc: "A histogram uses bins of different widths without normalizing bar heights to frequency density. Wider bins accumulate more observations, making those ranges appear disproportionately prominent.",
    example: "An income distribution histogram with a '$0–$50K' bin (width 50) next to '$50K–$60K' (width 10) makes the lower-income band look only 2× as common when it's actually 10× as common.",
    color: "#34d399",
  },
  {
    key: "discretized continuous variable", label: "Discretized Variable", short: "Continuous data binned to hide distribution",
    desc: "A naturally continuous variable is grouped into coarse, manually chosen bins that mask the underlying distribution — hiding skew, outliers, or bimodal patterns that would be visible in a smooth histogram or box plot.",
    example: "Showing executive tenure as 'Under 5 yrs / 5–10 yrs / Over 10 yrs' bins hides that 90% of the 'Over 10 yrs' bucket is actually over 25 years.",
    color: "#34d399",
  },
];

const NONGAAP_DETAILS: Array<{ key: string; label: string; short: string; regulation: string; desc: string; example: string; howToFix: string }> = [
  {
    key: "Non-GAAP Prominence",
    label: "Non-GAAP Prominence",
    short: "Non-GAAP metric displayed more prominently than GAAP",
    regulation: "Item 10(e) of Regulation S-K",
    desc: "SEC rules prohibit presenting a Non-GAAP financial measure with greater visual prominence than the most directly comparable GAAP measure. 'Prominence' covers font size, placement order, column positioning, and bolding.",
    example: "Adjusted Operating Profit ($1.1B) appears in the first column in bold, while GAAP Operating Profit ($0.9B) appears in a smaller secondary column — a direct Item 10(e) violation.",
    howToFix: "Present the comparable GAAP figure first, in equal or larger font and visual weight. Non-GAAP figures must follow with an explicit 'Non-GAAP' or 'Adjusted' label.",
  },
  {
    key: "Missing Reconciliation",
    label: "Missing Reconciliation",
    short: "No GAAP-to-Non-GAAP reconciliation table provided",
    regulation: "SEC Regulation G",
    desc: "Whenever a Non-GAAP financial measure is disclosed publicly, Regulation G requires a quantitative reconciliation to the most directly comparable GAAP measure, presented with equal prominence. Omitting this table is a direct violation.",
    example: "A press release reports Adjusted EBITDA of $2.3B without a table tracing the adjustments from GAAP Net Income — no depreciation add-back, no stock-comp add-back line items shown.",
    howToFix: "Add a reconciliation table immediately adjacent to every Non-GAAP figure. Each adjustment line (e.g., depreciation, restructuring charges) must be separately quantified.",
  },
  {
    key: "Labeling Issue",
    label: "Labeling Issue",
    short: "Non-GAAP measure not explicitly identified as such",
    regulation: "Item 10(e) of Regulation S-K",
    desc: "Every Non-GAAP measure must be explicitly identified as 'Non-GAAP', 'Adjusted', or an equivalent label at the point of presentation — not only in footnotes or the glossary. Ambiguous or missing labels mislead investors about whether a figure is audited.",
    example: "'Operating Expenses' reported as $9.4B with no indication it excludes restructuring charges — only discoverable by reading footnote 14 on page 47.",
    howToFix: "Label Non-GAAP measures at first use and in every table or chart where they appear. Use consistent naming (e.g., 'Adjusted Operating Expenses (Non-GAAP)') throughout the filing.",
  },
  {
    key: "Non-GAAP Per-Share Liquidity",
    label: "Per-Share Liquidity Measure",
    short: "Non-GAAP liquidity metric shown on per-share basis",
    regulation: "Item 10(e)(1)(ii) of Regulation S-K",
    desc: "SEC rules explicitly prohibit presenting Non-GAAP liquidity measures (e.g., Free Cash Flow, Adjusted Cash from Operations) on a per-share basis. Per-share liquidity figures imply a precision and comparability that is misleading for non-income metrics.",
    example: "Reporting 'Free Cash Flow per Share of $4.21' alongside EPS in a summary table — Free Cash Flow is a liquidity measure and cannot be legitimately expressed per share under SEC rules.",
    howToFix: "Remove per-share presentation from any Non-GAAP liquidity measure. Present absolute dollar amounts only, with a full reconciliation to the nearest GAAP cash flow line.",
  },
];

// ── PDF export helpers ────────────────────────────────────────────────────────

// Strip characters the jsPDF standard font cannot render (em-dash, arrows, curly quotes, ~×, etc.)
function sanitizeForPDF(s: string) {
  return s
    .replace(/[—–]/g, "-")
    .replace(/[→←↑↓]/g, "->")
    .replace(/[""]/g, '"')
    .replace(/['']/g, "'")
    .replace(/~/g, "~")
    .replace(/×/g, "x")
    .replace(/[^\x00-\x7F]/g, "");   // drop any remaining non-ASCII
}

async function generateReportPDF(
  company: string,
  findings: Array<{ id: string; file: string; severity: string; rule: string; type: string; dim: string; summary: string }>,
  info: { name: string; filing: string; cik: string },
) {
  const { default: jsPDF } = await import("jspdf");
  const { default: autoTable } = await import("jspdf-autotable");

  const doc = new jsPDF();
  const high   = findings.filter(f => f.severity === "HIGH").length;
  const medium = findings.filter(f => f.severity === "MEDIUM").length;
  const low    = findings.filter(f => f.severity === "LOW").length;

  // Title
  doc.setFontSize(18);
  doc.setTextColor(15, 23, 42);
  doc.text("FinChartAudit - Compliance Audit Report", 14, 22);
  doc.setFontSize(10);
  doc.setTextColor(100, 116, 139);
  doc.text(sanitizeForPDF(info.name), 14, 30);
  doc.text(`${info.filing}  |  CIK: ${info.cik}  |  Generated: ${new Date().toLocaleDateString()}`, 14, 36);

  // Summary table
  doc.setFontSize(12);
  doc.setTextColor(15, 23, 42);
  doc.text("Executive Summary", 14, 48);
  autoTable(doc, {
    startY: 52,
    head: [["Severity", "Count"]],
    body: [["High Risk", String(high)], ["Medium Risk", String(medium)], ["Low Risk", String(low)], ["Total Findings", String(findings.length)]],
    theme: "grid",
    headStyles: { fillColor: [5, 150, 105] },
    styles: { fontSize: 10 },
    margin: { left: 14 },
    tableWidth: 70,
  });

  // Findings table — no Conf. column, sanitized text
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  const afterSummary = (doc as any).lastAutoTable.finalY + 10;
  doc.setFontSize(12);
  doc.setTextColor(15, 23, 42);
  doc.text("Detailed Findings", 14, afterSummary);
  autoTable(doc, {
    startY: afterSummary + 4,
    head: [["ID", "Severity", "Dimension", "Type", "Rule", "Summary"]],
    body: findings.map(f => [
      f.id,
      f.severity,
      f.dim === "visual" ? "Visual" : "Non-GAAP",
      sanitizeForPDF(f.type),
      sanitizeForPDF(f.rule),
      sanitizeForPDF(f.summary),
    ]),
    theme: "striped",
    headStyles: { fillColor: [5, 150, 105] },
    styles: { fontSize: 8, cellPadding: 3, overflow: "linebreak" },
    columnStyles: {
      0: { cellWidth: 16 },
      1: { cellWidth: 18 },
      2: { cellWidth: 24 },
      3: { cellWidth: 28 },
      4: { cellWidth: 30 },
      5: { cellWidth: "auto" },
    },
    margin: { left: 14, right: 14 },
  });

  doc.setFontSize(7);
  doc.setTextColor(148, 163, 184);
  doc.text("Generated by FinChartAudit | Claude Haiku 4.5 | For informational purposes only.", 14, doc.internal.pageSize.height - 8);
  doc.save(`FinChartAudit_${company}_${new Date().toISOString().slice(0, 10)}.pdf`);
}

async function generateCheckPDF(files: FileItem[]) {
  const { default: jsPDF } = await import("jspdf");
  const { default: autoTable } = await import("jspdf-autotable");

  const doc = new jsPDF();
  const done    = files.filter(f => f.status === "done");
  const flagged = done.filter(f => f.result?.misleading).length;
  const high    = done.filter(f => f.result?.severity === "HIGH").length;
  const medium  = done.filter(f => f.result?.severity === "MEDIUM").length;

  doc.setFontSize(18);
  doc.setTextColor(15, 23, 42);
  doc.text("FinChartAudit — Pre-Filing Check Report", 14, 22);
  doc.setFontSize(10);
  doc.setTextColor(100, 116, 139);
  doc.text(`Generated: ${new Date().toLocaleDateString()}  ·  ${done.length} files analyzed`, 14, 30);

  doc.setFontSize(12);
  doc.setTextColor(15, 23, 42);
  doc.text("Summary", 14, 42);
  autoTable(doc, {
    startY: 46,
    head: [["Metric", "Value"]],
    body: [["Files Analyzed", String(done.length)], ["Issues Found", String(flagged)], ["High Risk", String(high)], ["Medium Risk", String(medium)], ["Clean", String(done.length - flagged)]],
    theme: "grid",
    headStyles: { fillColor: [5, 150, 105] },
    styles: { fontSize: 10 },
    margin: { left: 14 },
    tableWidth: 70,
  });

  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  const afterSummary = (doc as any).lastAutoTable.finalY + 10;
  doc.setFontSize(12);
  doc.setTextColor(15, 23, 42);
  doc.text("File Results", 14, afterSummary);
  autoTable(doc, {
    startY: afterSummary + 4,
    head: [["File", "Status", "Severity", "Issue Types", "Violation", "Analysis"]],
    body: done.map(f => {
      const r = f.result;
      return [
        sanitizeForPDF(f.name),
        r?.misleading ? "Flagged" : "Clean",
        r?.severity ?? "-",
        sanitizeForPDF(r?.misleader_types?.map(t => MISLEADER_LABELS[t] ?? t).join(", ") || "-"),
        sanitizeForPDF(r?.violation ?? "-"),
        sanitizeForPDF(r?.explanation ?? "-"),
      ];
    }),
    theme: "striped",
    headStyles: { fillColor: [5, 150, 105] },
    styles: { fontSize: 7.5, cellPadding: 2 },
    columnStyles: { 0: { cellWidth: 30 }, 1: { cellWidth: 14 }, 2: { cellWidth: 14 }, 3: { cellWidth: 24 }, 4: { cellWidth: 28 } },
    margin: { left: 14, right: 14 },
  });

  doc.setFontSize(7);
  doc.setTextColor(148, 163, 184);
  doc.text("Generated by FinChartAudit · Claude Haiku 4.5 · For informational purposes only.", 14, doc.internal.pageSize.height - 8);
  doc.save(`FinChartAudit_PreFiling_${new Date().toISOString().slice(0, 10)}.pdf`);
}

const NAV_ITEMS = [
  { id: "check",     icon: Shield,    label: "Pre-Filing Check" },
  { id: "report",    icon: FileText,  label: "Audit Report" },
  { id: "dashboard", icon: BarChart2, label: "Risk Dashboard" },
  { id: "how",       icon: BookOpen,  label: "How It Works" },
];

// ── Demo results (used when no API key) ───────────────────────────────────────

const DEMO_RESULTS: Array<{ misleading: boolean; misleader_types: string[]; violation: string | null; severity: "HIGH" | "MEDIUM" | "LOW" | null; rule: string | null; explanation: string }> = [
  {
    misleading: true,
    misleader_types: ["truncated axis"],
    violation: null,
    severity: "HIGH",
    rule: "Visualization Standards",
    explanation: "The y-axis begins at $10,500M instead of $0, making a 3.3% revenue difference ($11,497M vs. $11,879M) appear as an ~80% visual increase. This exaggeration misleads readers about the magnitude of year-over-year growth.",
  },
  {
    misleading: true,
    misleader_types: [],
    violation: "Non-GAAP Prominence Violation — Reg G / Item 10(e) of Reg S-K",
    severity: "HIGH",
    rule: "Reg G / Item 10(e) S-K",
    explanation: "Non-GAAP Adjusted Operating Profit ($1,122M) is presented before and with identical visual weight to the comparable GAAP Operating Profit. Per Item 10(e) of Regulation S-K, Non-GAAP measures must not be displayed more prominently than the most directly comparable GAAP measure.",
  },
  {
    misleading: false,
    misleader_types: [],
    violation: null,
    severity: null,
    rule: null,
    explanation: "Axis starts at zero, consistent tick intervals, appropriate chart type for the data shown. No Non-GAAP measures detected. No compliance concerns identified.",
  },
  {
    misleading: true,
    misleader_types: ["inappropriate use of pie chart"],
    violation: null,
    severity: "MEDIUM",
    rule: "Visualization Standards",
    explanation: "A donut chart is used to display three product segment revenues that are not parts of a meaningful compositional whole. Pie charts are appropriate only for part-to-whole relationships; using them for segment revenues creates a misleading impression of proportional significance.",
  },
  {
    misleading: true,
    misleader_types: [],
    violation: "Missing GAAP Reconciliation — Regulation G",
    severity: "MEDIUM",
    rule: "SEC Regulation G",
    explanation: "Non-GAAP Adjusted EBITDA ($977M) is presented as the primary profitability metric without a reconciliation to the nearest comparable GAAP measure (Net Income). Regulation G requires that whenever a Non-GAAP measure is disclosed, a reconciliation to the comparable GAAP measure must be presented with equal prominence.",
  },
];

// ── Batch file item type ──────────────────────────────────────────────────────

type FileStatus = "queued" | "processing" | "done" | "error";
type ItemKind = "chart" | "table";
type FileItem = {
  id: string;
  name: string;
  imageUrl: string;
  imageB64: string;
  status: FileStatus;
  result?: typeof DEMO_RESULTS[0] | null;
  error?: string;
  kind?: ItemKind;          // "chart" (image) or "table" (text-only)
  tableText?: string;       // plain text for table items (used for VLM analysis)
  tableHtml?: string;       // original HTML for table preview
};

// ── Shared analysis store ────────────────────────────────────────────────────

type AnalysisRecord = {
  id?: number;            // auto-increment IndexedDB key
  ticker: string;
  companyName: string;
  filingDate: string;
  items: FileItem[];
  analyzedAt: string;
  commentLetters: number;
};

// ── UI Primitives ─────────────────────────────────────────────────────────────

function Badge({ children, color = "blue" }: { children: React.ReactNode; color?: "blue"|"green"|"red"|"amber"|"purple"|"slate" }) {
  const cls = {
    blue:   "bg-blue-50   text-blue-700   border-blue-200",
    green:  "bg-emerald-50 text-emerald-700 border-emerald-200",
    red:    "bg-red-50    text-red-700    border-red-200",
    amber:  "bg-amber-50  text-amber-700  border-amber-200",
    purple: "bg-purple-50 text-purple-700 border-purple-200",
    slate:  "bg-slate-100 text-slate-600  border-slate-200",
  }[color];
  return <span className={`inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium border ${cls}`}>{children}</span>;
}

function SeverityBadge({ s }: { s: string }) {
  if (s === "HIGH")   return <Badge color="red">● High Risk</Badge>;
  if (s === "MEDIUM") return <Badge color="amber">● Medium Risk</Badge>;
  return <Badge color="slate">● Low Risk</Badge>;
}

function Card({ children, className = "", onClick }: { children: React.ReactNode; className?: string; onClick?: () => void }) {
  return <div className={`rounded-xl border border-slate-200 bg-white p-5 ${className}`} onClick={onClick}>{children}</div>;
}

function MetricCard({ label, value, sub, icon: Icon, color = "#059669" }: { label: string; value: string; sub?: string; icon: React.ElementType; color?: string }) {
  return (
    <Card className="flex items-center gap-4">
      <div className="rounded-lg p-3 flex-shrink-0" style={{ background: `${color}18` }}>
        <Icon size={20} style={{ color }} />
      </div>
      <div>
        <p className="text-sm text-slate-500">{label}</p>
        <p className="text-2xl font-bold text-slate-900">{value}</p>
        {sub && <p className="text-xs text-slate-400 mt-0.5">{sub}</p>}
      </div>
    </Card>
  );
}

function RiskScore({ score }: { score: number }) {
  const color = score >= 70 ? "#f87171" : score >= 40 ? "#fbbf24" : "#34d399";
  const label = score >= 70 ? "HIGH" : score >= 40 ? "MEDIUM" : "LOW";
  return (
    <div className="flex items-center gap-2">
      <div className="relative w-10 h-10 flex-shrink-0">
        <svg viewBox="0 0 36 36" className="w-10 h-10 -rotate-90">
          <circle cx="18" cy="18" r="15" fill="none" stroke="rgba(0,0,0,0.08)" strokeWidth="3" />
          <circle cx="18" cy="18" r="15" fill="none" stroke={color} strokeWidth="3"
            strokeDasharray={`${(score / 100) * 94.2} 94.2`} strokeLinecap="round" />
        </svg>
        <span className="absolute inset-0 flex items-center justify-center text-xs font-bold" style={{ color }}>{score}</span>
      </div>
      <span className="text-xs font-semibold" style={{ color }}>{label}</span>
    </div>
  );
}

// ── Pre-Filing Check (Batch) ──────────────────────────────────────────────────

const API_BASE = process.env.NEXT_PUBLIC_API_URL || "";

function CheckPage({ onAnalysisComplete }: { onAnalysisComplete?: (record: AnalysisRecord) => void }) {
  const useDemo = !API_BASE;  // demo mode only when no backend configured
  const [files, setFiles]       = useState<FileItem[]>([]);
  const [running, setRunning]   = useState(false);
  const [done, setDone]         = useState(false);
  const [dragging, setDragging] = useState(false);
  const demoIndexRef             = useRef(0);
  const filesRef                 = useRef<FileItem[]>([]);

  const [extracting, setExtracting] = useState<string | null>(null);
  const [ticker, setTicker]     = useState("");
  const [filingTypes, setFilingTypes] = useState<string[]>(["10-K"]);
  const [fetching, setFetching] = useState(false);
  const currentYear = new Date().getFullYear();
  const [selectedYears, setSelectedYears] = useState<number[]>([currentYear]);
  const [fetchedTicker, setFetchedTicker] = useState("");
  const [fetchedCompanyName, setFetchedCompanyName] = useState("");
  const [fetchedDate, setFetchedDate] = useState("");
  const [docContext, setDocContext] = useState("");
  const [commentLetters, setCommentLetters] = useState(0);
  const [skippedTables, setSkippedTables] = useState(0);
  const [exporting, setExporting] = useState(false);

  const readImageFile = (file: File): Promise<FileItem> =>
    new Promise(resolve => {
      const reader = new FileReader();
      reader.onload = e => {
        const url = e.target?.result as string;
        resolve({ id: `${Date.now()}-${Math.random()}`, name: file.name, imageUrl: url, imageB64: url.split(",")[1], status: "queued" });
      };
      reader.readAsDataURL(file);
    });

  const extractPdfPages = async (file: File): Promise<FileItem[]> => {
    const pdfjsLib = await import("pdfjs-dist");
    pdfjsLib.GlobalWorkerOptions.workerSrc = `https://cdnjs.cloudflare.com/ajax/libs/pdf.js/${pdfjsLib.version}/pdf.worker.min.mjs`;

    const arrayBuffer = await file.arrayBuffer();
    const pdf = await pdfjsLib.getDocument({ data: arrayBuffer }).promise;
    const MAX_PAGES = 20;
    const pageCount = Math.min(pdf.numPages, MAX_PAGES);
    const items: FileItem[] = [];

    for (let pageNum = 1; pageNum <= pageCount; pageNum++) {
      setExtracting(`${file.name} — extracting page ${pageNum}/${pageCount}…`);
      const page = await pdf.getPage(pageNum);
      const viewport = page.getViewport({ scale: 2.0 });
      const canvas = document.createElement("canvas");
      canvas.width  = viewport.width;
      canvas.height = viewport.height;
      const ctx = canvas.getContext("2d")!;
      // eslint-disable-next-line @typescript-eslint/no-explicit-any
      await (page.render as any)({ canvasContext: ctx, viewport }).promise;
      const imageUrl = canvas.toDataURL("image/jpeg", 0.85);
      items.push({
        id: `${Date.now()}-${pageNum}-${Math.random()}`,
        name: `${file.name} — Page ${pageNum}`,
        imageUrl,
        imageB64: imageUrl.split(",")[1],
        status: "queued",
      });
    }
    return items;
  };

  const addFiles = useCallback(async (incoming: FileList | File[]) => {
    const all = Array.from(incoming);
    const imageFiles = all.filter(f => f.type.startsWith("image/"));
    const pdfFiles   = all.filter(f => f.type === "application/pdf");

    const imageItems = await Promise.all(imageFiles.map(readImageFile));
    setFiles(prev => [...prev, ...imageItems]);

    for (const pdf of pdfFiles) {
      setExtracting(`Opening ${pdf.name}…`);
      try {
        const pages = await extractPdfPages(pdf);
        setFiles(prev => [...prev, ...pages]);
      } catch (err) {
        console.error("PDF extraction failed:", err);
      }
    }

    setExtracting(null);
    setDone(false);
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  const fetchSEC = async () => {
    if (!ticker.trim() || fetching || filingTypes.length === 0) return;
    setFetching(true);
    setExtracting(`Downloading ${ticker.toUpperCase()} ${filingTypes.join(", ")} from SEC EDGAR…`);
    try {
      const res = await fetch(`${API_BASE}/api/sec-fetch`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ ticker: ticker.trim(), filing_type: filingTypes, years: selectedYears }),
      });
      if (!res.ok) {
        const text = await res.text().catch(() => "Unknown error");
        throw new Error(`Server error ${res.status}: ${text.slice(0, 200)}`);
      }
      const data = await res.json();
      if (data.error || data.detail) throw new Error(data.error || data.detail);

      const chartItems: FileItem[] = (data.images || []).map((img: { name: string; base64: string; alt: string }) => ({
        id: `sec-${Date.now()}-${Math.random()}`,
        name: `${data.ticker} — ${img.name}`,
        imageUrl: `data:image/jpeg;base64,${img.base64}`,
        imageB64: img.base64,
        status: "queued" as FileStatus,
        kind: "chart" as ItemKind,
      }));

      const tableItems: FileItem[] = (data.tables || [])
        .map((t: { name: string; text: string; html?: string; has_nongaap: boolean }) => ({
          id: `sec-tbl-${Date.now()}-${Math.random()}`,
          name: `${data.ticker} — ${t.name}${t.has_nongaap ? " (Non-GAAP)" : ""} (table)`,
          imageUrl: "",
          imageB64: "",
          status: "queued" as FileStatus,
          kind: "table" as ItemKind,
          tableText: t.text,
          tableHtml: t.html,
        }));

      setFiles(prev => [...prev, ...chartItems, ...tableItems]);
      setFetchedTicker(data.ticker || ticker.toUpperCase());
      setFetchedCompanyName(data.company_name || data.ticker || ticker.toUpperCase());
      setFetchedDate(data.filings?.[0]?.date || data.filing_date || "");
      setDocContext(data.doc_context || "");
      setCommentLetters(data.comment_letters || 0);
      setSkippedTables(data.skipped_tables || 0);
      setExtracting(null);
      setDone(false);
    } catch (err) {
      setExtracting(null);
      alert(`SEC fetch failed: ${err}`);
    }
    setFetching(false);
  };

  const onDrop = useCallback((e: React.DragEvent) => {
    e.preventDefault(); setDragging(false);
    if (e.dataTransfer.files.length) addFiles(e.dataTransfer.files);
  }, [addFiles]);

  const removeFile = (id: string) => setFiles(prev => prev.filter(f => f.id !== id));

  const reset = () => { setFiles([]); setDone(false); setRunning(false); };

  const runAll = async () => {
    if (!files.length || running) return;
    setRunning(true); setDone(false);
    // Demo mode when no backend URL configured
    let di = demoIndexRef.current;

    for (let i = 0; i < files.length; i++) {
      // Mark as processing
      setFiles(prev => prev.map((f, idx) => idx === i ? { ...f, status: "processing" } : f));
      await new Promise(r => setTimeout(r, 300));

      let result: typeof DEMO_RESULTS[0] | null = null;
      let error: string | undefined;

      if (useDemo) {
        // Simulate analysis delay (1.2–2s per file)
        await new Promise(r => setTimeout(r, 1200 + Math.random() * 800));
        result = DEMO_RESULTS[di % DEMO_RESULTS.length];
        di++;
      } else {
        try {
          const item = files[i];
          let res: Response;
          if (item.kind === "table" && item.tableText) {
            // Table text analysis — no image needed
            res = await fetch(`${API_BASE}/api/analyze-table`, {
              method: "POST",
              headers: { "Content-Type": "application/json" },
              body: JSON.stringify({ tableText: item.tableText, tableName: item.name, secContext: docContext || undefined }),
            });
          } else {
            // Chart image analysis
            res = await fetch(`${API_BASE}/api/analyze`, {
              method: "POST",
              headers: { "Content-Type": "application/json" },
              body: JSON.stringify({ imageBase64: item.imageB64, secContext: docContext || undefined }),
            });
          }
          if (!res.ok) {
            const text = await res.text().catch(() => "Unknown error");
            throw new Error(`Server error ${res.status}: ${text.slice(0, 200)}`);
          }
          const data = await res.json();
          if (data.error || data.detail) throw new Error(data.error || data.detail);
          result = {
            ...data,
            violation: data.sec_violation ?? null,
            severity: data.severity ?? (data.misleading ? "MEDIUM" : null),
            rule: data.rule ?? null,
          };
        } catch (err) {
          error = String(err);
        }
      }

      setFiles(prev => {
        const updated = prev.map((f, idx) =>
          idx === i ? { ...f, status: (error ? "error" : "done") as FileStatus, result, error } : f
        );
        filesRef.current = updated;
        return updated;
      });
    }

    demoIndexRef.current = di;
    setRunning(false); setDone(true);

    // Save to analysis history for Report/Dashboard (deferred to avoid setState-during-render)
    if (onAnalysisComplete) {
      setTimeout(() => {
        onAnalysisComplete({
          ticker: fetchedTicker || "Upload",
          companyName: fetchedCompanyName || fetchedTicker || "Uploaded Files",
          filingDate: fetchedDate || new Date().toISOString().slice(0, 10),
          items: filesRef.current,
          analyzedAt: new Date().toISOString(),
          commentLetters,
        });
      }, 0);
    }
  };

  const queued     = files.filter(f => f.status === "queued").length;
  const processing = files.filter(f => f.status === "processing").length;
  const completed  = files.filter(f => f.status === "done" || f.status === "error").length;
  const flagged    = files.filter(f => f.status === "done" && f.result?.misleading).length;
  const highRisk   = files.filter(f => f.result?.severity === "HIGH").length;

  // ── EMPTY STATE ───────────────────────────────────────────────────────────
  if (!files.length) {
    return (
      <div className="space-y-6 fade-in">
        <div className="flex items-start justify-between">
          <div>
            <h2 className="text-2xl font-bold text-slate-900">Pre-Filing Compliance Check</h2>
            <p className="text-slate-500 mt-1">Enter a ticker to auto-fetch from SEC EDGAR, or upload your own filing. FinChartAudit analyzes charts for visual misleaders and tables for Non-GAAP prominence violations.</p>
          </div>
          <div className="flex-shrink-0 rounded-xl border border-blue-200 bg-blue-50 px-4 py-2 text-xs text-blue-700 text-right">
            <p className="font-semibold">Checks Against</p>
            <p className="text-blue-500 mt-0.5">SEC Reg G · Item 10(e) · 6 Visual Misleader Types · ViT Classifier Veto</p>
          </div>
        </div>

        {/* SEC Ticker Fetch */}
        <Card className="border-emerald-200 bg-emerald-50/30">
          <div className="flex items-center gap-4">
            <div className="flex-shrink-0 rounded-lg p-2.5 bg-emerald-100 border border-emerald-200">
              <Building2 size={18} className="text-emerald-600" />
            </div>
            <div className="flex-1">
              <p className="text-sm font-semibold text-slate-700">Fetch SEC Filing by Ticker</p>
              <p className="text-xs text-slate-500 mt-0.5">Enter any US public company ticker — auto-download from EDGAR and extract charts + financial tables</p>
            </div>
            <div className="flex items-center gap-2">
              <input
                type="text" value={ticker} onChange={e => setTicker(e.target.value.toUpperCase())}
                onKeyDown={e => e.key === "Enter" && fetchSEC()}
                placeholder="e.g. AAPL, MSFT"
                className="w-32 bg-white border border-slate-200 rounded-lg px-3 py-2 text-sm text-slate-900 placeholder-slate-400 focus:outline-none focus:border-emerald-400 transition-colors font-mono"
              />
              <div className="flex items-center gap-1.5 flex-wrap">
                {[
                  { v: "10-K", label: "10-K" },
                  { v: "10-Q", label: "10-Q" },
                  { v: "8-K",  label: "8-K" },
                  { v: "DEF 14A", label: "DEF 14A" },
                  { v: "S-1",  label: "S-1" },
                ].map(ft => {
                  const checked = filingTypes.includes(ft.v);
                  return (
                    <button key={ft.v}
                      onClick={() => setFilingTypes(prev => checked ? prev.filter(x => x !== ft.v) : [...prev, ft.v])}
                      className={`px-2.5 py-1 rounded-md text-xs font-medium border transition-colors ${checked ? "bg-emerald-50 border-emerald-300 text-emerald-700" : "bg-white border-slate-200 text-slate-400 hover:text-slate-600 hover:border-slate-300"}`}
                    >{ft.label}</button>
                  );
                })}
              </div>
              <div className="flex items-center gap-1 border-l border-slate-200 pl-2">
                <span className="text-xs text-slate-400 whitespace-nowrap">FY:</span>
                {Array.from({ length: 6 }, (_, i) => currentYear - i).map(yr => {
                  const checked = selectedYears.includes(yr);
                  return (
                    <button key={yr} onClick={() => setSelectedYears(prev => checked ? prev.filter(y => y !== yr) : [...prev, yr].sort((a, b) => b - a))}
                      className={`px-2 py-1 rounded-md text-xs font-medium border transition-colors ${checked ? "bg-emerald-50 border-emerald-300 text-emerald-700" : "bg-white border-slate-200 text-slate-400 hover:text-slate-600 hover:border-slate-300"}`}
                    >{yr}</button>
                  );
                })}
              </div>
              <button
                onClick={fetchSEC}
                disabled={fetching || !ticker.trim() || filingTypes.length === 0}
                className="flex items-center gap-2 px-4 py-2 rounded-lg bg-emerald-600 hover:bg-emerald-500 disabled:bg-slate-300 text-white text-sm font-medium transition-colors"
              >
                {fetching ? <Loader2 size={14} className="animate-spin" /> : <Download size={14} />}
                {fetching ? "Fetching…" : "Fetch"}
              </button>
            </div>
          </div>
          {extracting && (
            <div className="flex items-center gap-2 mt-3 px-3 py-2 rounded-lg bg-blue-50 border border-blue-200 text-blue-700 text-xs">
              <Loader2 size={12} className="animate-spin flex-shrink-0" />{extracting}
            </div>
          )}
        </Card>

        {/* Divider */}
        <div className="flex items-center gap-4 text-xs text-slate-400">
          <div className="flex-1 h-px bg-slate-200" />
          <span>or upload manually</span>
          <div className="flex-1 h-px bg-slate-200" />
        </div>

        {/* Big drop zone */}
        <div
          onDrop={onDrop}
          onDragOver={e => { e.preventDefault(); setDragging(true); }}
          onDragLeave={() => setDragging(false)}
          onClick={() => document.getElementById("batch-input")?.click()}
          className={`relative border-2 border-dashed rounded-2xl cursor-pointer transition-all duration-300 flex flex-col items-center justify-center gap-5 ${dragging ? "border-emerald-400 bg-emerald-50 scale-[1.01]" : "border-slate-200 hover:border-slate-300 bg-white hover:bg-slate-50"}`}
          style={{ minHeight: 260 }}
        >
          <input id="batch-input" type="file" accept="image/*,application/pdf" multiple className="hidden" onChange={e => e.target.files && addFiles(e.target.files)} />
          <div className={`w-16 h-16 rounded-2xl border flex items-center justify-center transition-all duration-300 ${dragging ? "border-emerald-400 bg-emerald-100" : "border-slate-200 bg-slate-100"}`}>
            <FolderOpen size={28} className={dragging ? "text-emerald-600" : "text-slate-400"} />
          </div>
          <div className="text-center">
            <p className={`text-base font-semibold transition-colors ${dragging ? "text-emerald-700" : "text-slate-700"}`}>
              {dragging ? "Drop to add files" : "Drag & drop images or PDF"}
            </p>
            <p className="text-slate-500 mt-1 text-sm">
              <span className="text-slate-700 font-medium">PDF</span> (auto-extracted page by page) or individual images — PNG, JPG
            </p>
          </div>
        </div>

        {/* Regulatory rules */}
        <div className="grid grid-cols-3 gap-3">
          {[
            { rule: "SEC Regulation G",  desc: "Non-GAAP measures require equal-prominence GAAP reconciliation" },
            { rule: "Item 10(e) S-K",    desc: "Non-GAAP metrics cannot appear more prominently than GAAP" },
            { rule: "Visual Integrity",  desc: "6 SEC-relevant chart types: truncated axes, 3D effects, dual-axis, pie chart misuse + ViT veto" },
          ].map(r => (
            <div key={r.rule} className="rounded-lg border border-slate-200 bg-white px-3 py-2.5">
              <p className="text-xs font-semibold text-slate-700 flex items-center gap-1.5"><Scale size={11} className="text-emerald-600" />{r.rule}</p>
              <p className="text-xs text-slate-500 mt-1">{r.desc}</p>
            </div>
          ))}
        </div>
      </div>
    );
  }

  // ── FILES QUEUED / RUNNING / DONE STATE ──────────────────────────────────
  return (
    <div className="space-y-5 fade-in">
      {/* Header */}
      {extracting && (
        <div className="flex items-center gap-2 px-4 py-2.5 rounded-xl bg-blue-50 border border-blue-200 text-blue-700 text-sm">
          <Loader2 size={14} className="animate-spin flex-shrink-0" />
          <span>{extracting}</span>
        </div>
      )}

      <div className="flex items-center justify-between">
        <div>
          <h2 className="text-2xl font-bold text-slate-900">Pre-Filing Compliance Check</h2>
          <p className="text-slate-500 mt-1">
            {files.filter(f => f.kind !== "table").length} charts + {files.filter(f => f.kind === "table").length} financial tables queued
            {skippedTables > 0 && <span className="text-slate-400"> ({skippedTables} non-financial tables skipped)</span>}
            {" · "}{useDemo ? "Demo Mode" : "Live Mode"}
          </p>
        </div>
        <div className="flex items-center gap-2">
          {!running && !done && (
            <button onClick={() => document.getElementById("batch-input")?.click()} className="flex items-center gap-2 px-3 py-2 rounded-lg border border-slate-200 text-xs text-slate-500 hover:text-slate-900 hover:bg-slate-50 transition-colors">
              <Upload size={13} />Add more
            </button>
          )}
          {done && (
            <button onClick={reset} className="flex items-center gap-2 px-3 py-2 rounded-lg border border-slate-200 text-xs text-slate-500 hover:text-slate-900 hover:bg-slate-50 transition-colors">
              New batch
            </button>
          )}
          <input id="batch-input" type="file" accept="image/*,application/pdf" multiple className="hidden" onChange={e => e.target.files && addFiles(e.target.files)} />
        </div>
      </div>

      {/* Progress bar (while running) */}
      {(running || done) && (
        <div className="space-y-1.5">
          <div className="flex items-center justify-between text-xs text-slate-500">
            <span>{running ? `Analyzing file ${completed + 1} of ${files.length}…` : "Analysis complete"}</span>
            <span>{completed}/{files.length}</span>
          </div>
          <div className="w-full bg-slate-200 rounded-full h-1.5 overflow-hidden">
            <div className="h-full rounded-full bg-emerald-500 transition-all duration-500" style={{ width: `${(completed / files.length) * 100}%` }} />
          </div>
        </div>
      )}

      {/* Summary (when done) */}
      {done && (
        <div className="grid grid-cols-4 gap-3 fade-in">
          <div className="rounded-xl border border-slate-200 bg-white p-4 text-center">
            <p className="text-2xl font-bold text-slate-900">{files.length}</p>
            <p className="text-xs text-slate-500 mt-0.5">Files Scanned</p>
          </div>
          <div className="rounded-xl border border-red-200 bg-red-50 p-4 text-center">
            <p className="text-2xl font-bold text-red-600">{flagged}</p>
            <p className="text-xs text-slate-500 mt-0.5">Issues Found</p>
          </div>
          <div className="rounded-xl border border-amber-200 bg-amber-50 p-4 text-center">
            <p className="text-2xl font-bold text-amber-600">{highRisk}</p>
            <p className="text-xs text-slate-500 mt-0.5">High Risk</p>
          </div>
          <div className="rounded-xl border border-emerald-200 bg-emerald-50 p-4 text-center">
            <p className="text-2xl font-bold text-emerald-600">{files.length - flagged}</p>
            <p className="text-xs text-slate-500 mt-0.5">Clean</p>
          </div>
        </div>
      )}

      {/* Run button */}
      {!running && !done && (
        <button onClick={runAll}
          className="w-full py-3.5 rounded-xl font-semibold flex items-center justify-center gap-2 bg-emerald-600 hover:bg-emerald-500 text-white transition-all duration-200 text-sm shadow-lg shadow-emerald-600/20">
          <Play size={15} />Run Compliance Check on {files.length} File{files.length > 1 ? "s" : ""}
        </button>
      )}

      {/* File cards grid */}
      <div className="grid grid-cols-1 gap-3">
        {files.map((f) => (
          <FileCard key={f.id} item={f} onRemove={() => removeFile(f.id)} canRemove={!running} />
        ))}
      </div>

      {/* Export when done */}
      {done && (
        <button
          disabled={exporting}
          onClick={async () => {
            setExporting(true);
            try { await generateCheckPDF(files); }
            catch (err) { alert(`PDF export failed: ${err}`); }
            finally { setExporting(false); }
          }}
          className="w-full py-2.5 rounded-xl font-medium flex items-center justify-center gap-2 border border-slate-200 text-slate-500 hover:text-slate-900 hover:bg-slate-50 disabled:opacity-50 transition-colors text-sm"
        >
          {exporting ? <Loader2 size={14} className="animate-spin" /> : <Download size={14} />}
          {exporting ? "Generating PDF…" : "Export Full Audit Report"}
        </button>
      )}
    </div>
  );
}

function FileCard({ item, onRemove, canRemove }: { item: FileItem; onRemove: () => void; canRemove: boolean }) {
  const [modalOpen, setModalOpen] = useState(false);
  const r = item.result;
  const isIssue = r?.misleading;

  return (
    <div
      className={`rounded-xl border transition-all duration-300 overflow-hidden ${item.status === "done" && r ? "cursor-pointer hover:shadow-sm" : ""} ${
        item.status === "queued"     ? "border-slate-200 bg-white" :
        item.status === "processing" ? "border-blue-200 bg-blue-50" :
        isIssue                      ? "border-red-200 bg-red-50" :
        item.status === "error"      ? "border-red-200 bg-red-50" :
                                       "border-emerald-200 bg-emerald-50"
      }`}
      onClick={() => { if (item.status === "done" && r) setModalOpen(true); }}
    >
      <div className="flex items-center gap-4 p-4">
        {/* Thumbnail */}
        {item.kind === "table" ? (
          <div className="w-16 h-12 rounded-lg flex-shrink-0 bg-purple-50 border border-purple-200 flex items-center justify-center">
            <Scale size={18} className="text-purple-400" />
          </div>
        ) : (
          <div className="w-16 h-12 rounded-lg overflow-hidden flex-shrink-0 bg-slate-100 border border-slate-200">
            {item.imageUrl ? <img src={item.imageUrl} alt={item.name} className="w-full h-full object-cover" /> : <FileText size={18} className="text-slate-400 m-auto" />}
          </div>
        )}

        {/* File info */}
        <div className="flex-1 min-w-0">
          <p className="text-sm font-medium text-slate-900 truncate">{item.name}</p>
          {/* Table content preview */}
          {item.kind === "table" && item.tableHtml && (
            <div className="mt-1 max-h-16 overflow-hidden rounded border border-slate-200 bg-white px-2 py-1 text-xs"
              dangerouslySetInnerHTML={{ __html: item.tableHtml }}
              style={{ fontSize: "9px", lineHeight: "1.3" }}
            />
          )}
          {item.kind === "table" && !item.tableHtml && item.tableText && (
            <p className="text-xs text-slate-400 mt-0.5 line-clamp-2 font-mono leading-relaxed">{item.tableText.slice(0, 150)}…</p>
          )}
          {item.status === "queued"     && <p className="text-xs text-slate-500 mt-0.5">Queued — waiting to run</p>}
          {item.status === "processing" && <p className="text-xs text-blue-400 mt-0.5 flex items-center gap-1"><Loader2 size={10} className="animate-spin" />Analyzing…</p>}
          {item.status === "done" && r && (
            <div className="flex items-center gap-2 mt-1 flex-wrap">
              {isIssue ? (
                <>
                  {r.severity === "HIGH"   && <Badge color="red">● High Risk</Badge>}
                  {r.severity === "MEDIUM" && <Badge color="amber">● Medium Risk</Badge>}
                  {r.severity === "LOW"    && <Badge color="slate">● Low Risk</Badge>}
                  {r.misleader_types.map(t => <Badge key={t} color="red">{MISLEADER_LABELS[t] ?? t}</Badge>)}
                  {r.violation && <Badge color="amber">Non-GAAP Violation</Badge>}
                </>
              ) : (
                <Badge color="green">✓ Clear to File</Badge>
              )}
            </div>
          )}
          {item.status === "error" && <p className="text-xs text-red-400 mt-0.5">{item.error}</p>}
        </div>

        {/* Status icon + actions */}
        <div className="flex items-center gap-2 flex-shrink-0">
          {item.status === "queued"     && <div className="w-6 h-6 rounded-full border border-slate-200 bg-slate-100" />}
          {item.status === "processing" && <Loader2 size={18} className="animate-spin text-blue-400" />}
          {item.status === "done" && isIssue  && <AlertTriangle size={18} className="text-red-400" />}
          {item.status === "done" && !isIssue && <CheckCircle   size={18} className="text-emerald-400" />}
          {item.status === "error"      && <AlertCircle size={18} className="text-red-400" />}

          {item.status === "done" && r && (
            <button onClick={() => setModalOpen(true)} className="text-xs text-blue-400 hover:text-blue-300 px-2 py-1 rounded-lg hover:bg-blue-500/[0.1] transition-colors flex items-center gap-1">
              View <ChevronRight size={12} />
            </button>
          )}
          {canRemove && item.status === "queued" && (
            <button onClick={onRemove} className="p-1 rounded-lg hover:bg-slate-100 text-slate-400 hover:text-slate-700 transition-colors">
              <X size={14} />
            </button>
          )}
        </div>
      </div>

      {/* Finding modal */}
      {modalOpen && r && (
        <FindingModal
          finding={{
            imageUrl: item.imageUrl,
            tableText: item.tableText,
            tableHtml: item.tableHtml,
            filename: item.name,
            severity: r.severity ?? null,
            rule: r.rule ?? null,
            type: r.violation
              ? "Non-GAAP / SEC Violation"
              : r.misleader_types?.length
                ? (MISLEADER_LABELS[r.misleader_types[0]] ?? r.misleader_types[0])
                : "Compliance Issue",
            summary: r.explanation,
            misleader_types: r.misleader_types,
            violation: r.violation,
          }}
          onClose={() => setModalOpen(false)}
        />
      )}
    </div>
  );
}

// ── Finding Modal ─────────────────────────────────────────────────────────────

type ModalFinding = {
  imageUrl?: string;       // source image (from batch upload)
  tableText?: string;      // plain text (fallback)
  tableHtml?: string;      // original HTML for table display
  filename: string;
  severity: string | null;
  rule: string | null;
  type: string;
  summary: string;
  misleader_types?: string[];
  violation?: string | null;
  howToFix?: string;
  // confidence removed — model does not output confidence scores
};

function FindingModal({ finding, onClose }: { finding: ModalFinding; onClose: () => void }) {
  const HOW_TO_FIX: Record<string, string> = {
    "Truncated Axis":       "Start the y-axis at zero. If a zoomed view is needed, provide a dual panel: one full-scale and one zoomed, clearly labeled.",
    "Non-GAAP Prominence":  "Per Item 10(e) of Reg S-K, present the most directly comparable GAAP measure first, in equal or greater font size and visual prominence. Non-GAAP labels must be explicit.",
    "Missing Reconciliation": "Add a reconciliation table mapping every Non-GAAP adjustment to its GAAP counterpart, per Regulation G. Place it immediately adjacent to or below the Non-GAAP figure.",
    "Pie Chart Misuse":     "Replace with a bar chart for comparisons or a table for precise values. Pie charts should only be used when all segments sum to a meaningful 100% whole.",
    "Dual Axis Abuse":      "Use separate charts for metrics with different scales, or normalize both to a common baseline. Clearly label both y-axes and add a note on scale differences.",
    "3D Distortion":        "Use flat 2D charts. 3D perspective distorts the perceived size of data points relative to actual values.",
    "Inconsistent Intervals": "Ensure all axis tick marks are evenly spaced. If using non-linear scale, add a clear label (e.g. 'logarithmic scale').",
    "Labeling Issue":       "Explicitly label all Non-GAAP measures as 'Non-GAAP' or 'Adjusted' at the point of presentation, not just in footnotes.",
  };

  const fix = finding.howToFix
    || HOW_TO_FIX[finding.type]
    || "Review the identified element against SEC Regulation G and Item 10(e) of Regulation S-K before filing.";

  const severityColor =
    finding.severity === "HIGH"   ? { bg: "bg-red-50",    border: "border-red-200",    text: "text-red-700" } :
    finding.severity === "MEDIUM" ? { bg: "bg-amber-50",  border: "border-amber-200",  text: "text-amber-700" } :
    finding.severity === "LOW"    ? { bg: "bg-slate-100", border: "border-slate-200",  text: "text-slate-600" } :
                                    { bg: "bg-blue-50",   border: "border-blue-200",   text: "text-blue-700" };

  return createPortal(
    <div className="fixed inset-0 z-50 flex items-center justify-center p-4" onClick={onClose}>
      {/* Backdrop */}
      <div className="absolute inset-0 bg-slate-900/40 backdrop-blur-sm" />

      {/* Modal */}
      <div
        className="relative w-full max-w-3xl max-h-[90vh] overflow-y-auto rounded-2xl border border-slate-200 shadow-2xl fade-in bg-white"
        onClick={e => e.stopPropagation()}
      >
        {/* Header */}
        <div className="flex items-start justify-between p-6 border-b border-slate-100">
          <div className="flex items-start gap-3">
            {finding.severity === "HIGH"   && <AlertTriangle size={20} className="text-red-500 mt-0.5 flex-shrink-0" />}
            {finding.severity === "MEDIUM" && <AlertTriangle size={20} className="text-amber-500 mt-0.5 flex-shrink-0" />}
            {!finding.severity             && <AlertCircle   size={20} className="text-blue-500 mt-0.5 flex-shrink-0" />}
            <div>
              <p className="text-slate-900 font-bold text-lg leading-tight">{finding.type}</p>
              <div className="flex items-center gap-2 mt-1 flex-wrap">
                {finding.severity && <SeverityBadge s={finding.severity} />}
                {finding.rule && <Badge color="slate">{finding.rule}</Badge>}
<span className="text-xs font-mono text-slate-400">{finding.filename}</span>
              </div>
            </div>
          </div>
          <button onClick={onClose} className="p-2 rounded-lg hover:bg-slate-100 text-slate-400 hover:text-slate-700 transition-colors flex-shrink-0">
            <X size={18} />
          </button>
        </div>

        {/* Source image (if available) */}
        {finding.imageUrl && (
          <div className="px-6 pt-5">
            <p className="text-xs font-semibold text-slate-500 uppercase tracking-wider mb-2 flex items-center gap-1.5">
              <Eye size={11} />Source Document
            </p>
            <div className="rounded-xl overflow-hidden border border-slate-200 bg-slate-50 relative">
              <img src={finding.imageUrl} alt="source" className="w-full object-contain max-h-72" />
              <div className={`absolute bottom-0 left-0 right-0 px-3 py-2 ${severityColor.bg} border-t ${severityColor.border} backdrop-blur-sm`}>
                <p className={`text-xs font-semibold ${severityColor.text} flex items-center gap-1.5`}>
                  <AlertTriangle size={11} />
                  {finding.violation || finding.misleader_types?.map(t => MISLEADER_LABELS[t] ?? t).join(", ") || finding.type} detected in this image
                </p>
              </div>
            </div>
          </div>
        )}

        {/* No image — show filename reference */}
        {!finding.imageUrl && !finding.tableText && (
          <div className="px-6 pt-5">
            <p className="text-xs font-semibold text-slate-500 uppercase tracking-wider mb-2">Source File</p>
            <div className="flex items-center gap-3 px-4 py-3 rounded-xl border border-slate-200 bg-slate-50">
              <FileText size={18} className="text-slate-400 flex-shrink-0" />
              <div>
                <p className="text-slate-900 text-sm font-mono">{finding.filename}</p>
                <p className="text-slate-400 text-xs mt-0.5">Extracted from SEC 10-K filing · Research dataset</p>
              </div>
            </div>
          </div>
        )}

        {/* Table content */}
        {(finding.tableHtml || finding.tableText) && (
          <div className="px-6 pt-5">
            <p className="text-xs font-semibold text-slate-500 uppercase tracking-wider mb-2 flex items-center gap-1.5">
              <Scale size={11} />Source Table — {finding.filename}
            </p>
            {finding.tableHtml ? (
              <div className="rounded-xl border border-purple-200 bg-white p-3 max-h-64 overflow-auto text-xs"
                dangerouslySetInnerHTML={{ __html: finding.tableHtml }}
                style={{ fontSize: "11px", lineHeight: "1.4" }}
              />
            ) : (
              <div className="rounded-xl border border-purple-200 bg-purple-50/30 p-4 max-h-48 overflow-y-auto">
                <pre className="text-xs text-slate-600 font-mono whitespace-pre-wrap leading-relaxed">{finding.tableText}</pre>
              </div>
            )}
          </div>
        )}

        <div className="p-6 space-y-5">
          {/* Violation detail */}
          {finding.violation && (
            <div className={`rounded-xl border px-4 py-3 ${severityColor.border} ${severityColor.bg}`}>
              <p className="text-xs font-semibold text-slate-500 uppercase tracking-wider mb-1">SEC Violation</p>
              <p className="text-sm text-slate-700">{finding.violation}</p>
            </div>
          )}

          {/* Misleader types */}
          {finding.misleader_types && finding.misleader_types.length > 0 && (
            <div>
              <p className="text-xs font-semibold text-slate-500 uppercase tracking-wider mb-2">Issue Types</p>
              <div className="flex flex-wrap gap-2">
                {finding.misleader_types.map(t => <Badge key={t} color="red">{MISLEADER_LABELS[t] ?? t}</Badge>)}
              </div>
            </div>
          )}

          {/* Analysis */}
          <div>
            <p className="text-xs font-semibold text-slate-500 uppercase tracking-wider mb-2">Analysis</p>
            <p className="text-sm text-slate-700 leading-relaxed">{finding.summary}</p>
          </div>

          {/* Regulatory footer */}
          <div className="border-t border-slate-100 pt-3 text-xs text-slate-400 flex items-center gap-2">
            <Scale size={11} />
            Checked against: SEC Regulation G · Item 10(e) of Regulation S-K · 12 Visual Misleader Types
          </div>
        </div>
      </div>
    </div>,
    document.body
  );
}

// ── Detection Reference Modal (tabbed) ───────────────────────────────────────

function DetectionReferenceModal({ onClose }: { onClose: () => void }) {
  const [tab, setTab] = useState<"visual" | "nongaap">("visual");

  return createPortal(
    <div className="fixed inset-0 z-50 flex items-center justify-center p-4" onClick={onClose}>
      <div className="absolute inset-0 bg-slate-900/40 backdrop-blur-sm" />
      <div
        className="relative w-full max-w-4xl max-h-[90vh] overflow-y-auto rounded-2xl border border-slate-200 shadow-2xl fade-in bg-white"
        onClick={e => e.stopPropagation()}
      >
        {/* Header */}
        <div className="sticky top-0 z-10 px-6 pt-5 pb-0 border-b border-slate-200 bg-white">
          <div className="flex items-start justify-between mb-4">
            <div className="flex items-start gap-3">
              <div className="w-9 h-9 rounded-lg flex items-center justify-center bg-indigo-50 border border-indigo-200 flex-shrink-0 mt-0.5">
                <HelpCircle size={17} className="text-indigo-600" />
              </div>
              <div>
                <p className="text-slate-900 font-bold text-lg leading-tight">Detection Reference</p>
                <p className="text-slate-500 text-sm mt-0.5">What FinChartAudit looks for in every SEC filing</p>
              </div>
            </div>
            <button onClick={onClose} className="p-2 rounded-lg hover:bg-slate-100 text-slate-400 hover:text-slate-700 transition-colors flex-shrink-0">
              <X size={18} />
            </button>
          </div>
          {/* Tabs */}
          <div className="flex gap-1">
            <button
              onClick={() => setTab("visual")}
              className={`px-4 py-2 text-sm font-medium rounded-t-lg border-b-2 transition-all ${tab === "visual" ? "text-blue-700 border-blue-500 bg-blue-50" : "text-slate-500 border-transparent hover:text-slate-700"}`}
            >
              <span className="flex items-center gap-2"><Eye size={13} />Visual Misleaders <span className="text-xs opacity-60">12 types</span></span>
            </button>
            <button
              onClick={() => setTab("nongaap")}
              className={`px-4 py-2 text-sm font-medium rounded-t-lg border-b-2 transition-all ${tab === "nongaap" ? "text-purple-700 border-purple-500 bg-purple-50" : "text-slate-500 border-transparent hover:text-slate-700"}`}
            >
              <span className="flex items-center gap-2"><Scale size={13} />Non-GAAP Violations <span className="text-xs opacity-60">4 types</span></span>
            </button>
          </div>
        </div>

        {/* Visual tab */}
        {tab === "visual" && (
          <div className="p-6 grid grid-cols-1 sm:grid-cols-2 gap-4">
            {MISLEADER_DETAILS.map((m) => (
              <div key={m.key} className="rounded-xl border border-slate-200 bg-slate-50 p-5 hover:border-slate-300 hover:bg-white transition-all duration-150">
                <div className="flex items-center gap-2 mb-1.5">
                  <span className="w-2 h-2 rounded-full flex-shrink-0" style={{ background: m.color }} />
                  <p className="text-slate-900 font-semibold">{m.label}</p>
                </div>
                <p className="text-sm text-slate-500 italic mb-2">{m.short}</p>
                <p className="text-sm text-slate-600 leading-relaxed mb-3">{m.desc}</p>
                <div className="rounded-lg border border-amber-200 bg-amber-50 px-3 py-2.5">
                  <p className="text-xs font-semibold text-amber-700 mb-1">Financial Example</p>
                  <p className="text-sm text-amber-900/70 leading-relaxed">{m.example}</p>
                </div>
              </div>
            ))}
            <div className="sm:col-span-2 text-xs text-slate-400 flex items-center gap-2 pt-2 border-t border-slate-100">
              <Eye size={11} />Visual misleader taxonomy from academic research on deceptive data visualization · checked on every uploaded chart
            </div>
          </div>
        )}

        {/* Non-GAAP tab */}
        {tab === "nongaap" && (
          <div className="p-6 space-y-4">
            <div className="rounded-xl border border-purple-200 bg-purple-50 px-4 py-3 text-sm text-slate-600 leading-relaxed">
              <span className="font-semibold text-purple-700">About Non-GAAP compliance: </span>
              SEC rules require that whenever a company discloses a Non-GAAP financial measure (e.g., Adjusted EPS, Adjusted EBITDA), it must follow strict rules on labeling, prominence, and reconciliation. FinChartAudit detects violations in financial tables and disclosure text — not just charts.
            </div>
            {NONGAAP_DETAILS.map((n) => (
              <div key={n.key} className="rounded-xl border border-slate-200 bg-slate-50 p-5 hover:border-slate-300 hover:bg-white transition-all duration-150">
                <div className="flex items-start justify-between gap-4 mb-2">
                  <div>
                    <p className="text-slate-900 font-semibold">{n.label}</p>
                    <p className="text-sm text-slate-500 italic mt-0.5">{n.short}</p>
                  </div>
                  <span className="flex-shrink-0 inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium border bg-purple-50 text-purple-700 border-purple-200 whitespace-nowrap">{n.regulation}</span>
                </div>
                <p className="text-sm text-slate-600 leading-relaxed mb-3">{n.desc}</p>
                <div className="rounded-lg border border-amber-200 bg-amber-50 px-3 py-2.5">
                  <p className="text-xs font-semibold text-amber-700 mb-1">Example Violation</p>
                  <p className="text-sm text-amber-900/70 leading-relaxed">{n.example}</p>
                </div>
              </div>
            ))}
            <div className="text-xs text-slate-400 flex items-center gap-2 pt-2 border-t border-slate-100">
              <Scale size={11} />SEC Regulation G (17 CFR 244.100) · Item 10(e) of Regulation S-K (17 CFR 229.10(e))
            </div>
          </div>
        )}
      </div>
    </div>,
    document.body
  );
}

// ── Audit Report ──────────────────────────────────────────────────────────────

function ReportPage({ analysisHistory = [], onDelete }: { analysisHistory?: AnalysisRecord[]; onDelete?: (record: AnalysisRecord) => void }) {
  // Available companies: hardcoded examples + user-analyzed (deduplicated)
  const userTickers = [...new Set(analysisHistory.map(r => r.ticker))];
  const allCompanies = [...new Set([...userTickers, "UPS", "STZ"])]; // user-analyzed first
  const [company, setCompany]         = useState(allCompanies[0] || "UPS");
  const [selectedVersion, setSelectedVersion] = useState(0); // 0 = latest

  // Auto-select the most recently analyzed company whenever a new analysis arrives
  const latestUser = analysisHistory[0]?.ticker || "";
  const latestAt = analysisHistory[0]?.analyzedAt || "";
  useEffect(() => {
    if (latestUser) {
      setCompany(latestUser);
      setSelectedVersion(0);
    }
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [latestAt]);
  const [modalFinding, setModalFinding] = useState<ModalFinding | null>(null);
  const [exporting, setExporting] = useState(false);

  // All records for this ticker, sorted newest first
  const companyRecords = analysisHistory
    .filter(r => r.ticker === company)
    .sort((a, b) => b.analyzedAt.localeCompare(a.analyzedAt));
  const userRecord = companyRecords[selectedVersion] ?? companyRecords[0] ?? null;

  // Reset version when switching company
  useEffect(() => { setSelectedVersion(0); }, [company]);

  // Build findings from user record or hardcoded data
  const allFindings = userRecord
    ? userRecord.items
        .filter(f => f.status === "done")
        .map((f, i) => ({
          id: `F-${String(i + 1).padStart(3, "0")}`,
          file: f.name,
          severity: f.result?.misleading ? (f.result?.severity ?? "MEDIUM") : "CLEAN",
          rule: f.result?.rule ?? (f.result?.misleading ? "SEC Compliance" : null),
          type: f.result?.violation ? "Non-GAAP Violation"
            : f.result?.misleader_types?.length ? (f.result.misleader_types[0] ?? "Compliance Issue")
            : f.result?.misleading ? "Compliance Issue" : "No Issue",
          dim: (f.kind === "table" || f.result?.violation) ? "nongaap" as const : "visual" as const,
          summary: f.result?.explanation ?? "",
          imageUrl: f.imageUrl || undefined,
          tableText: f.tableText || undefined,
          misleading: f.result?.misleading ?? false,
        }))
    : null;
  // Hardcoded findings for demo companies (flagged only)
  const findings = allFindings ?? (company === "UPS" ? UPS_FINDINGS : STZ_FINDINGS);
  const flaggedFindings = allFindings ? allFindings.filter(f => f.misleading) : findings;
  const cleanFindings = allFindings ? allFindings.filter(f => !f.misleading) : [];
  const high   = flaggedFindings.filter(f => f.severity === "HIGH").length;
  const medium = flaggedFindings.filter(f => f.severity === "MEDIUM").length;
  const low    = flaggedFindings.filter(f => f.severity === "LOW").length;
  const totalItems = allFindings ? allFindings.length : (company === "UPS" ? 10 : 10);
  const flagRate = totalItems > 0 ? Math.round((flaggedFindings.length / totalItems) * 100) : 0;
  const info = userRecord
    ? { name: userRecord.companyName || company, filing: userRecord.filingDate || "SEC Filing", cik: `${userRecord.commentLetters} comment letter(s)` }
    : company === "UPS"
      ? { name: "United Parcel Service, Inc.", filing: "10-K FY2024", cik: "0001090727" }
      : { name: "Constellation Brands, Inc.", filing: "10-K FY2024", cik: "0000016918" };

  return (
    <div className="space-y-6 fade-in">
      {/* Research banner */}
      <div className="rounded-xl border border-purple-200 bg-purple-50 px-4 py-3 flex items-start gap-3">
        <BookOpen size={15} className="text-purple-600 flex-shrink-0 mt-0.5" />
        <div>
          <p className="text-sm font-semibold text-purple-700">Research Study Results</p>
          <p className="text-xs text-slate-600 mt-0.5">
            These reports were generated by running FinChartAudit on real SEC 10-K filings as part of our capstone research.
            Ground truth violations were validated against actual SEC comment letters issued to these companies.
            This demonstrates the tool operating at scale — not live user submissions.
          </p>
        </div>
      </div>

      {/* Header */}
      <div className="flex items-start justify-between">
        <div>
          <h2 className="text-2xl font-bold text-slate-900">Compliance Audit Report</h2>
          <p className="text-slate-500 mt-1">Automated SEC disclosure review — {totalItems} items analyzed, {flaggedFindings.length} findings flagged.</p>
        </div>
        <button
          disabled={exporting}
          onClick={async () => {
            setExporting(true);
            try { await generateReportPDF(company, flaggedFindings as typeof UPS_FINDINGS, info); }
            catch (err) { alert(`PDF export failed: ${err}`); }
            finally { setExporting(false); }
          }}
          className="flex items-center gap-2 px-4 py-2 rounded-lg border border-slate-200 bg-white text-sm text-slate-600 hover:bg-slate-50 disabled:opacity-50 transition-colors"
        >
          {exporting ? <Loader2 size={14} className="animate-spin" /> : <Download size={14} />}
          {exporting ? "Generating…" : "Export PDF"}
        </button>
      </div>

      {/* Company selector */}
      <div className="flex gap-2 flex-wrap">
        {allCompanies.map(t => (
          <button key={t} onClick={() => setCompany(t)}
            className={`px-4 py-2 rounded-lg text-sm font-medium transition-all ${company === t ? "bg-emerald-50 border border-emerald-200 text-emerald-700" : "border border-slate-200 text-slate-500 hover:text-slate-900 hover:bg-slate-50"}`}>
            {t}{userTickers.includes(t) && !["UPS", "STZ"].includes(t) ? " (live)" : ""}
          </button>
        ))}
      </div>

      {/* Version selector + delete */}
      {companyRecords.length > 0 && (
        <div className="flex items-center gap-2 text-xs flex-wrap">
          <span className="text-slate-500">Analysis history ({companyRecords.length} run{companyRecords.length > 1 ? "s" : ""}):</span>
          <div className="flex gap-1.5 flex-wrap">
            {companyRecords.map((r, i) => (
              <div key={r.analyzedAt} className="flex items-center gap-0.5">
                <button onClick={() => setSelectedVersion(i)}
                  className={`px-2.5 py-1 rounded-l-md border border-r-0 text-xs font-medium transition-colors ${selectedVersion === i ? "bg-emerald-50 border-emerald-300 text-emerald-700" : "bg-white border-slate-200 text-slate-400 hover:text-slate-600 hover:border-slate-300"}`}>
                  {new Date(r.analyzedAt).toLocaleString()}{i === 0 ? " (latest)" : ""}
                </button>
                {onDelete && (
                  <button
                    onClick={(e) => { e.stopPropagation(); onDelete(r); }}
                    className="px-1.5 py-1 rounded-r-md border border-slate-200 text-slate-300 hover:text-red-500 hover:border-red-300 hover:bg-red-50 transition-colors"
                    title="Delete this analysis"
                  >
                    <X size={11} />
                  </button>
                )}
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Report header card */}
      <Card>
        <div className="flex items-start justify-between gap-6">
          <div className="flex-1">
            <div className="flex items-center gap-2 mb-1">
              <Building2 size={14} className="text-slate-400" />
              <span className="text-xs text-slate-400 font-medium uppercase tracking-wider">Issuer</span>
            </div>
            <p className="text-xl font-bold text-slate-900">{info.name}</p>
            <div className="flex items-center gap-4 mt-2 text-xs text-slate-500">
              <span>Filing: <span className="text-slate-700">{info.filing}</span></span>
              <span>CIK: <span className="text-slate-700 font-mono">{info.cik}</span></span>
            </div>
          </div>
          <div className="flex items-center gap-6 flex-shrink-0">
            <RiskScore score={flagRate} />
            <div className="space-y-1.5 text-right">
              <div className="flex items-center justify-end gap-2 text-xs"><span className="text-slate-500">High Risk</span><span className="font-bold text-red-400">{high}</span></div>
              <div className="flex items-center justify-end gap-2 text-xs"><span className="text-slate-500">Medium Risk</span><span className="font-bold text-amber-400">{medium}</span></div>
              <div className="flex items-center justify-end gap-2 text-xs"><span className="text-slate-500">Low Risk</span><span className="font-bold text-slate-400">{low}</span></div>
            </div>
          </div>
        </div>
        <div className="mt-4 pt-4 border-t border-slate-100">
          <div className="w-full bg-slate-100 rounded-full h-1.5 overflow-hidden flex gap-0.5">
            <div className="h-full rounded-full bg-red-400 transition-all" style={{ width: `${findings.length ? (high / findings.length) * 100 : 0}%` }} />
            <div className="h-full rounded-full bg-amber-400 transition-all" style={{ width: `${findings.length ? (medium / findings.length) * 100 : 0}%` }} />
            <div className="h-full rounded-full bg-slate-400 transition-all" style={{ width: `${findings.length ? (low / findings.length) * 100 : 0}%` }} />
          </div>
          <div className="flex gap-4 mt-1.5 text-xs text-slate-500">
            <span><span className="text-red-400">■</span> High ({high})</span>
            <span><span className="text-amber-400">■</span> Medium ({medium})</span>
            <span><span className="text-slate-400">■</span> Low ({low})</span>
          </div>
        </div>
      </Card>

      {/* Risk level legend */}
      <div className="rounded-xl border border-slate-200 bg-white divide-y divide-slate-100">
        <div className="px-4 py-2.5 flex items-center gap-2">
          <AlertCircle size={13} className="text-slate-400 flex-shrink-0" />
          <span className="text-xs font-semibold text-slate-500 uppercase tracking-wider">Risk Level Definitions</span>
        </div>
        <div className="grid grid-cols-3 divide-x divide-slate-100">
          <div className="px-4 py-3">
            <div className="flex items-center gap-2 mb-1">
              <span className="w-2 h-2 rounded-full bg-red-400 flex-shrink-0" />
              <span className="text-sm font-semibold text-red-600">High Risk</span>
            </div>
            <p className="text-xs text-slate-500 leading-relaxed">Direct violation of an SEC mandatory rule — e.g., Non-GAAP measure presented more prominently than GAAP, or missing required reconciliation table. Must be remediated before filing.</p>
          </div>
          <div className="px-4 py-3">
            <div className="flex items-center gap-2 mb-1">
              <span className="w-2 h-2 rounded-full bg-amber-400 flex-shrink-0" />
              <span className="text-sm font-semibold text-amber-600">Medium Risk</span>
            </div>
            <p className="text-xs text-slate-500 leading-relaxed">Potentially misleading visual or disclosure practice that may draw SEC comment — e.g., truncated axis, pie chart misuse, inconsistent ordering. Review and consider revising.</p>
          </div>
          <div className="px-4 py-3">
            <div className="flex items-center gap-2 mb-1">
              <span className="w-2 h-2 rounded-full bg-slate-400 flex-shrink-0" />
              <span className="text-sm font-semibold text-slate-500">Low Risk</span>
            </div>
            <p className="text-xs text-slate-500 leading-relaxed">Minor presentation issue that does not constitute a regulatory violation — e.g., ambiguous labeling, suboptimal chart choice. Best practice to address but not blocking.</p>
          </div>
        </div>
      </div>

      {/* Flagged findings */}
      <div>
        <div className="flex items-center justify-between mb-3">
          <h3 className="text-sm font-semibold text-slate-700">Issues Found ({flaggedFindings.length})</h3>
          <p className="text-xs text-slate-500">Sorted by severity</p>
        </div>
        {flaggedFindings.length === 0 && (
          <Card className="border-emerald-200 bg-emerald-50/50">
            <div className="flex items-center gap-3 py-2">
              <CheckCircle size={18} className="text-emerald-500" />
              <div>
                <p className="text-sm font-medium text-emerald-700">No compliance issues detected</p>
                <p className="text-xs text-slate-500 mt-0.5">All {totalItems} items passed the compliance check.</p>
              </div>
            </div>
          </Card>
        )}
        <div className="space-y-2.5">
          {flaggedFindings.map(f => {
            const imgUrl = (f as Record<string, unknown>).imageUrl as string | undefined;
            return (
            <Card
              key={f.id}
              className={`cursor-pointer transition-all hover:shadow-sm group ${f.severity === "HIGH" ? "border-red-200 bg-red-50/50" : f.severity === "MEDIUM" ? "border-amber-200 bg-amber-50/50" : ""}`}
              onClick={() => setModalFinding({ filename: f.file, severity: f.severity, rule: f.rule, type: f.type, summary: f.summary, imageUrl: imgUrl })}
            >
              <div className="flex items-start gap-4">
                {/* Thumbnail */}
                {imgUrl ? (
                  <div className="w-20 h-14 rounded-lg overflow-hidden flex-shrink-0 bg-slate-100 border border-slate-200">
                    <img src={imgUrl} alt={f.file} className="w-full h-full object-cover" />
                  </div>
                ) : f.dim === "nongaap" ? (
                  <div className="w-20 h-14 rounded-lg flex-shrink-0 bg-purple-50 border border-purple-200 flex items-center justify-center">
                    <Scale size={18} className="text-purple-400" />
                  </div>
                ) : (
                  <div className="w-20 h-14 rounded-lg flex-shrink-0 bg-slate-100 border border-slate-200 flex items-center justify-center">
                    <FileText size={18} className="text-slate-400" />
                  </div>
                )}
                <div className="flex-shrink-0 pt-0.5 w-28">
                  <SeverityBadge s={f.severity} />
                </div>
                <div className="flex-1 min-w-0">
                  <div className="flex items-center gap-2 flex-wrap mb-1.5">
                    <span className="font-mono text-xs text-slate-500">{f.id}</span>
                    {f.dim === "visual"
                      ? <span className="inline-flex items-center gap-1 px-2 py-0.5 rounded-full text-xs font-medium border bg-blue-50 text-blue-600 border-blue-200"><Eye size={9} />Visual</span>
                      : <span className="inline-flex items-center gap-1 px-2 py-0.5 rounded-full text-xs font-medium border bg-purple-50 text-purple-600 border-purple-200"><Scale size={9} />Non-GAAP</span>
                    }
                    <Badge color="slate">{f.type}</Badge>
                    <span className="text-xs text-slate-500 font-medium">{f.rule}</span>
                    <span className="text-xs font-mono text-slate-400">{f.file}</span>
                  </div>
                  <p className="text-sm text-slate-600 leading-relaxed">{f.summary}</p>
                </div>
                <div className="flex-shrink-0 opacity-0 group-hover:opacity-100 transition-opacity">
                  <span className="text-xs text-slate-500 flex items-center gap-1">view <ChevronRight size={12} /></span>
                </div>
              </div>
            </Card>
            );
          })}
        </div>
      </div>

      {/* Clean items */}
      {cleanFindings.length > 0 && (
        <div>
          <div className="flex items-center justify-between mb-3">
            <h3 className="text-sm font-semibold text-slate-700 flex items-center gap-2">
              <CheckCircle size={14} className="text-emerald-500" />
              Passed ({cleanFindings.length})
            </h3>
          </div>
          <div className="space-y-1.5">
            {cleanFindings.map(f => (
              <div key={f.id} className="flex items-center gap-3 px-4 py-2.5 rounded-lg border border-emerald-100 bg-emerald-50/30">
                <CheckCircle size={14} className="text-emerald-400 flex-shrink-0" />
                <span className="text-sm text-slate-600 truncate flex-1">{f.file}</span>
                <Badge color="green">Clear</Badge>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Footer */}
      <div className="text-xs text-slate-400 border-t border-slate-100 pt-4 flex items-center justify-between">
        <span>Generated by FinChartAudit · Claude Haiku 4.5 · Powered by OpenRouter</span>
        <span>This report is for informational purposes. Consult legal counsel before making filing decisions.</span>
      </div>

      {/* Finding modal */}
      {modalFinding && <FindingModal finding={modalFinding} onClose={() => setModalFinding(null)} />}
    </div>
  );
}

// ── Risk Dashboard ────────────────────────────────────────────────────────────

function DashboardPage({ analysisHistory = [] }: { analysisHistory?: AnalysisRecord[] }) {
  // Use only the latest record per ticker for dashboard metrics
  const latestByTicker = new Map<string, AnalysisRecord>();
  for (const r of analysisHistory) {
    const existing = latestByTicker.get(r.ticker);
    if (!existing || r.analyzedAt > existing.analyzedAt) {
      latestByTicker.set(r.ticker, r);
    }
  }

  const userRows = [...latestByTicker.values()].map(r => {
    const done = r.items.filter(f => f.status === "done");
    const flagged = done.filter(f => f.result?.misleading).length;
    const hasNongaap = done.some(f => f.result?.violation);
    const hasChart = done.some(f => f.kind !== "table" && f.result?.misleading && f.result?.misleader_types?.length);
    return {
      ticker: r.ticker,
      name: r.companyName || r.ticker,
      flagged,
      total: done.length,
      flagRate: done.length > 0 ? flagged / done.length : 0,
      type: hasChart && hasNongaap ? "Mixed" : hasNongaap ? "Non-GAAP" : hasChart ? "Chart" : "—",
      isLive: true,
    };
  });

  // Combine: user data overrides hardcoded for same ticker
  const userTickers = new Set(userRows.map(r => r.ticker));
  const combinedRisk = [
    ...userRows,
    ...COMPANY_RISK.filter(c => !userTickers.has(c.ticker)).map(c => ({ ...c, name: c.name, isLive: false })),
  ].sort((a, b) => b.flagRate - a.flagRate);

  const barData = combinedRisk.map(c => ({
    ticker: c.ticker, score: Math.round(c.flagRate * 100), isLive: (c as { isLive?: boolean }).isLive,
  }));

  const typeCount = combinedRisk.reduce((acc, c) => {
    acc[c.type] = (acc[c.type] || 0) + 1; return acc;
  }, {} as Record<string, number>);
  const pieData = Object.entries(typeCount).filter(([k]) => k !== "—").map(([name, val]) => ({ name, val }));

  const totalFlagged = combinedRisk.reduce((a, c) => a + c.flagged, 0);
  const totalItems = combinedRisk.reduce((a, c) => a + c.total, 0);
  const highRiskCos  = combinedRisk.filter(c => c.flagRate >= 0.5).length;

  return (
    <div className="space-y-6 fade-in">
      {/* Research banner */}
      <div className="rounded-xl border border-purple-200 bg-purple-50 px-4 py-3 flex items-start gap-3">
        <BookOpen size={15} className="text-purple-600 flex-shrink-0 mt-0.5" />
        <div>
          <p className="text-sm font-semibold text-purple-700">Research Study Results — 10 Companies</p>
          <p className="text-xs text-slate-600 mt-0.5">
            FinChartAudit was run across 10 SEC 10-K filers (96 items total).
            Flag rates represent the proportion of filing visuals flagged by Claude Haiku 4.5.
            Use <span className="text-slate-800 font-medium">Pre-Filing Check</span> to run the tool on your own files.
          </p>
        </div>
      </div>

      <div>
        <h2 className="text-2xl font-bold text-slate-900">Risk Dashboard</h2>
        <p className="text-slate-500 mt-1">Compliance flag rates across {combinedRisk.length} SEC filers. {totalItems} items analyzed (charts + financial tables).</p>
      </div>

      <div className="grid grid-cols-2 lg:grid-cols-4 gap-4">
        <MetricCard label="Companies Screened" value={String(combinedRisk.length)} sub={analysisHistory.length ? `${analysisHistory.length} live + ${combinedRisk.length - analysisHistory.length} example` : "SEC 10-K filings"} icon={Building2} color="#059669" />
        <MetricCard label="Items Flagged"      value={String(totalFlagged)}   sub={`of ${totalItems} total items`} icon={AlertCircle}  color="#dc2626" />
        <MetricCard label="High Flag Rate"     value={String(highRiskCos)}    sub="Flag rate ≥ 50%"        icon={AlertTriangle} color="#dc2626" />
        <MetricCard label="Items Analyzed"     value={String(totalItems)}     sub="Charts + tables"        icon={Eye}          color="#059669" />
      </div>

      {/* Company risk table */}
      <Card>
        <h3 className="text-base font-semibold text-slate-900 mb-4">Company Flag Rates</h3>
        <div className="overflow-x-auto">
          <table className="w-full text-sm">
            <thead>
              <tr className="border-b border-slate-200">
                <th className="text-left py-2 pr-4 text-slate-400 font-medium text-xs">Company</th>
                <th className="text-left py-2 px-2 text-slate-400 font-medium text-xs">Flag Rate</th>
                <th className="text-left py-2 px-2 text-slate-400 font-medium text-xs">Primary Issue</th>
                <th className="text-right py-2 px-2 text-slate-400 font-medium text-xs">Flagged / Total</th>
              </tr>
            </thead>
            <tbody>
              {combinedRisk.map((c) => (
                <tr key={c.ticker} className="border-b border-slate-100 hover:bg-slate-50 transition-colors">
                  <td className="py-3 pr-4">
                    <div className="flex items-center gap-2">
                      <div>
                        <p className="text-slate-900 font-medium text-xs">{c.ticker}</p>
                        <p className="text-slate-500 text-xs">{c.name}</p>
                      </div>
                      {(c as { isLive?: boolean }).isLive && <span className="text-xs text-emerald-600 bg-emerald-50 border border-emerald-200 px-1.5 py-0.5 rounded">live</span>}
                    </div>
                  </td>
                  <td className="py-3 px-2"><RiskScore score={Math.round(c.flagRate * 100)} /></td>
                  <td className="py-3 px-2">
                    <Badge color={c.type === "Non-GAAP" ? "purple" : c.type === "Chart" ? "blue" : c.type === "Mixed" ? "amber" : "slate"}>
                      {c.type}
                    </Badge>
                  </td>
                  <td className="text-right py-3 px-2 text-slate-600 text-xs font-mono">{c.flagged} / {c.total}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </Card>

      {/* Charts row */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <Card>
          <h3 className="text-base font-semibold text-slate-900 mb-1">Flag Rate Distribution</h3>
          <p className="text-xs text-slate-500 mb-4">Percentage of filing items flagged for compliance concerns</p>
          <ResponsiveContainer width="100%" height={220}>
            <BarChart data={barData} margin={{ top: 5, right: 10, left: -15, bottom: 5 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="rgba(0,0,0,0.06)" />
              <XAxis dataKey="ticker" tick={{ fill: "#64748b", fontSize: 10 }} />
              <YAxis tick={{ fill: "#64748b", fontSize: 10 }} domain={[0, 100]} />
              <Tooltip contentStyle={{ background: "#ffffff", border: "1px solid #e2e8f0", borderRadius: 8, color: "#0f172a" }} />
              <Bar dataKey="score" radius={[3, 3, 0, 0]}>
                {barData.map((e, i) => (
                  <Cell key={i} fill={e.score >= 70 ? "#f87171" : e.score >= 40 ? "#fbbf24" : "#34d399"} />
                ))}
              </Bar>
            </BarChart>
          </ResponsiveContainer>
        </Card>

        <Card>
          <h3 className="text-base font-semibold text-slate-900 mb-1">Issue Type Breakdown</h3>
          <p className="text-xs text-slate-500 mb-4">Non-GAAP prominence vs. chart visualization issues</p>
          <ResponsiveContainer width="100%" height={220}>
            <BarChart data={[
              { type: "Non-GAAP", count: combinedRisk.filter(c => c.type === "Non-GAAP").reduce((a, c) => a + c.flagged, 0) },
              { type: "Chart",   count: combinedRisk.filter(c => c.type === "Chart").reduce((a, c) => a + c.flagged, 0) },
              { type: "Mixed",   count: combinedRisk.filter(c => c.type === "Mixed").reduce((a, c) => a + c.flagged, 0) },
            ]} margin={{ top: 5, right: 10, left: -15, bottom: 5 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="rgba(0,0,0,0.06)" />
              <XAxis dataKey="type" tick={{ fill: "#64748b", fontSize: 10 }} />
              <YAxis tick={{ fill: "#64748b", fontSize: 10 }} />
              <Tooltip contentStyle={{ background: "#ffffff", border: "1px solid #e2e8f0", borderRadius: 8, color: "#0f172a" }} />
              <Bar dataKey="count" fill="#8b5cf6" radius={[3, 3, 0, 0]} name="Items Flagged" />
            </BarChart>
          </ResponsiveContainer>
        </Card>
      </div>
    </div>
  );
}

// ── How It Works ──────────────────────────────────────────────────────────────

function HowPage() {
  const steps = [
    { n: "01", icon: Upload,      title: "Upload Your Filing",     desc: "Upload charts, financial tables, or an entire 10-K PDF. FinChartAudit automatically extracts every visual element for review.", color: "#38bdf8" },
    { n: "02", icon: Eye,         title: "AI Compliance Review",   desc: "Claude Haiku analyzes each item against SEC Regulation G, Item 10(e), and 12 visual misleader categories simultaneously.", color: "#818cf8" },
    { n: "03", icon: FileText,    title: "Actionable Audit Report", desc: "Receive a structured report with severity-rated findings, specific regulatory citations, and recommended remediation steps.", color: "#34d399" },
  ];
  const stats = [
    { value: "< 2 min", label: "Per filing reviewed",    sub: "vs. 2–3 hours manual" },
    { value: "83%",     label: "Detection F1 score",    sub: "on Misviz benchmark" },
    { value: "12",      label: "Misleader categories",  sub: "SEC + visual standards" },
    { value: "13",      label: "Companies validated",   sub: "against real SEC letters" },
  ];
  const audiences = [
    { icon: Building2, title: "IR & Compliance Teams",    desc: "Check every filing before submission. Catch Non-GAAP prominence issues before SEC review." },
    { icon: Scale,     title: "Audit Firms",              desc: "Extend financial statement audits to cover visual disclosures. Defensible, evidence-based findings." },
    { icon: TrendingUp, title: "Investment Research",     desc: "Quickly flag misleading competitor disclosures. Screen portfolios for visual presentation risk." },
  ];
  return (
    <div className="space-y-10 fade-in">
      <div>
        <h2 className="text-2xl font-bold text-slate-900">How FinChartAudit Works</h2>
        <p className="text-slate-500 mt-1">Automated SEC disclosure compliance in three steps — from upload to audit report.</p>
      </div>

      {/* Steps */}
      <div className="relative">
        <div className="absolute top-8 left-0 right-0 h-px bg-gradient-to-r from-transparent via-white/10 to-transparent hidden lg:block" />
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
          {steps.map((s) => (
            <Card key={s.n} className="relative text-center">
              <div className="absolute -top-3 left-1/2 -translate-x-1/2 text-xs font-mono px-2 py-0.5 rounded border border-slate-200 bg-white text-slate-500">{s.n}</div>
              <div className="w-12 h-12 rounded-xl mx-auto mb-4 flex items-center justify-center" style={{ background: `${s.color}20`, border: `1px solid ${s.color}30` }}>
                <s.icon size={22} style={{ color: s.color }} />
              </div>
              <h3 className="text-slate-900 font-semibold mb-2">{s.title}</h3>
              <p className="text-slate-400 text-sm leading-relaxed">{s.desc}</p>
            </Card>
          ))}
        </div>
      </div>

      {/* Stats */}
      <div className="grid grid-cols-2 lg:grid-cols-4 gap-4">
        {stats.map(s => (
          <Card key={s.label} className="text-center">
            <p className="text-3xl font-bold text-slate-900">{s.value}</p>
            <p className="text-sm text-slate-600 mt-1">{s.label}</p>
            <p className="text-xs text-slate-500 mt-0.5">{s.sub}</p>
          </Card>
        ))}
      </div>

      {/* Who it's for */}
      <div>
        <h3 className="text-sm font-semibold text-slate-400 uppercase tracking-wider mb-4">Who uses FinChartAudit</h3>
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-4">
          {audiences.map(a => (
            <Card key={a.title} className="flex gap-4">
              <div className="w-10 h-10 rounded-lg bg-blue-500/15 flex items-center justify-center flex-shrink-0">
                <a.icon size={18} className="text-blue-400" />
              </div>
              <div>
                <p className="text-slate-900 font-semibold text-sm">{a.title}</p>
                <p className="text-slate-400 text-xs mt-1 leading-relaxed">{a.desc}</p>
              </div>
            </Card>
          ))}
        </div>
      </div>

      {/* Regulatory framework */}
      <Card className="border-emerald-200 bg-emerald-50">
        <div className="flex items-start gap-4">
          <Scale size={20} className="text-emerald-600 flex-shrink-0 mt-0.5" />
          <div>
            <h3 className="text-slate-900 font-semibold">Regulatory Framework</h3>
            <p className="text-slate-600 text-sm mt-1 leading-relaxed">FinChartAudit checks against <span className="text-emerald-700 font-medium">SEC Regulation G</span> (Non-GAAP reconciliation requirements) and <span className="text-emerald-700 font-medium">Item 10(e) of Regulation S-K</span> (prohibition on presenting Non-GAAP measures more prominently than comparable GAAP measures). Visual checks cover 12 chart misleader types identified in academic literature on deceptive data visualization.</p>
            <div className="flex gap-2 mt-3 flex-wrap">
              <Badge color="green">SEC Regulation G</Badge>
              <Badge color="green">Item 10(e) Reg S-K</Badge>
              <Badge color="green">12 Visual Misleader Types</Badge>
              <Badge color="green">Misviz-synth Benchmark</Badge>
            </div>
          </div>
        </div>
      </Card>

      <div className="text-center text-slate-400 text-xs pt-4 border-t border-slate-100">
        CS 6180 Generative AI Capstone · Northeastern University · 2026
      </div>
    </div>
  );
}

// ── IndexedDB persistence ────────────────────────────────────────────────────

const IDB_NAME = "finchartaudit";
const IDB_VERSION = 2;
const IDB_STORE = "analyses";

function openDB(): Promise<IDBDatabase> {
  return new Promise((resolve, reject) => {
    const req = indexedDB.open(IDB_NAME, IDB_VERSION);
    req.onupgradeneeded = (event) => {
      const db = req.result;
      const oldVersion = event.oldVersion;
      // v1 → v2 migration: drop old ticker-keyed store, create auto-increment store with ticker index
      if (oldVersion < 2) {
        if (db.objectStoreNames.contains(IDB_STORE)) {
          db.deleteObjectStore(IDB_STORE);
        }
        const store = db.createObjectStore(IDB_STORE, { keyPath: "id", autoIncrement: true });
        store.createIndex("ticker", "ticker", { unique: false });
        store.createIndex("analyzedAt", "analyzedAt", { unique: false });
      }
    };
    req.onsuccess = () => resolve(req.result);
    req.onerror = () => reject(req.error);
  });
}

async function idbGetAll(): Promise<AnalysisRecord[]> {
  const db = await openDB();
  return new Promise((resolve, reject) => {
    const tx = db.transaction(IDB_STORE, "readonly");
    const store = tx.objectStore(IDB_STORE);
    const req = store.getAll();
    req.onsuccess = () => resolve(req.result as AnalysisRecord[]);
    req.onerror = () => reject(req.error);
  });
}

async function idbPut(record: AnalysisRecord): Promise<void> {
  const db = await openDB();
  return new Promise((resolve, reject) => {
    const tx = db.transaction(IDB_STORE, "readwrite");
    const store = tx.objectStore(IDB_STORE);
    store.add(record);   // always insert new record, never overwrite
    tx.oncomplete = () => resolve();
    tx.onerror = () => reject(tx.error);
  });
}

async function idbDelete(id: number): Promise<void> {
  const db = await openDB();
  return new Promise((resolve, reject) => {
    const tx = db.transaction(IDB_STORE, "readwrite");
    const store = tx.objectStore(IDB_STORE);
    store.delete(id);
    tx.oncomplete = () => resolve();
    tx.onerror = () => reject(tx.error);
  });
}

// ── Root ──────────────────────────────────────────────────────────────────────

export default function Home() {
  const [page, setPage] = useState("check");
  const [showMisleaders, setShowMisleaders] = useState(false);
  const [analysisHistory, setAnalysisHistory] = useState<AnalysisRecord[]>([]);

  // Load persisted analyses from IndexedDB on mount
  useEffect(() => {
    idbGetAll()
      .then(records => {
        if (records.length > 0) {
          // Sort by analyzedAt descending (newest first)
          records.sort((a, b) => b.analyzedAt.localeCompare(a.analyzedAt));
          setAnalysisHistory(records);
        }
      })
      .catch(err => console.warn("IndexedDB load failed:", err));
  }, []);

  const saveAnalysis = (record: AnalysisRecord) => {
    setAnalysisHistory(prev => [record, ...prev]);
    idbPut(record).catch(err => console.warn("IndexedDB save failed:", err));
  };

  const deleteAnalysis = (record: AnalysisRecord) => {
    if (record.id != null) {
      setAnalysisHistory(prev => prev.filter(r => r.id !== record.id));
      idbDelete(record.id).catch(err => console.warn("IndexedDB delete failed:", err));
    } else {
      // Fallback for records without id (e.g. not yet persisted)
      setAnalysisHistory(prev => prev.filter(r => r.analyzedAt !== record.analyzedAt || r.ticker !== record.ticker));
    }
  };

  // Pages rendered once, hidden via CSS to preserve state across tab switches
  const pageComponents = {
    check:     <CheckPage onAnalysisComplete={saveAnalysis} />,
    report:    <ReportPage analysisHistory={analysisHistory} onDelete={deleteAnalysis} />,
    dashboard: <DashboardPage analysisHistory={analysisHistory} />,
    how:       <HowPage />,
  };
  return (
    <div className="flex h-screen overflow-hidden bg-slate-50">
      {/* Sidebar */}
      <aside className="w-56 flex-shrink-0 flex flex-col border-r border-slate-200 bg-white">
        <div className="px-5 py-6 border-b border-slate-100">
          <div className="flex items-center gap-2.5">
            <div className="w-8 h-8 rounded-lg flex items-center justify-center" style={{ background: "linear-gradient(135deg, #059669, #0d9488)" }}>
              <Shield size={15} className="text-white" />
            </div>
            <div>
              <p className="text-slate-900 font-bold text-sm leading-tight">FinChartAudit</p>
              <p className="text-slate-400 text-xs">SEC Compliance AI</p>
            </div>
          </div>
        </div>
        <nav className="flex-1 px-3 py-4 space-y-1 flex flex-col">
          {NAV_ITEMS.map(({ id, icon: Icon, label }) => (
            <button key={id} onClick={() => setPage(id)}
              className={`w-full flex items-center gap-3 px-3 py-2.5 rounded-lg text-sm transition-all duration-150 text-left ${page === id ? "bg-emerald-50 text-emerald-700 border border-emerald-200 font-medium" : "text-slate-600 hover:text-slate-900 hover:bg-slate-50"}`}>
              <Icon size={15} />{label}
            </button>
          ))}

          {/* Divider */}
          <div className="pt-3 mt-1 border-t border-slate-100">
            <p className="text-xs text-slate-400 uppercase tracking-wider px-3 mb-2">Reference</p>
            <button
              onClick={() => setShowMisleaders(true)}
              className="w-full flex items-center gap-3 px-3 py-2.5 rounded-lg text-sm transition-all duration-150 text-left text-slate-600 hover:text-slate-900 hover:bg-slate-50 group"
            >
              <HelpCircle size={15} className="text-indigo-500 group-hover:text-indigo-600" />
              <span>Detection Reference</span>
            </button>
          </div>
        </nav>
        <div className="px-4 py-4 border-t border-slate-100 space-y-1.5">
          <div className="flex items-center gap-2"><div className="w-1.5 h-1.5 rounded-full bg-emerald-500 pulse-dot" /><span className="text-xs text-slate-500">Claude Haiku 4.5</span></div>
          <p className="text-xs text-slate-400">NEU CS 6180 · 2026</p>
        </div>
      </aside>

      {/* Main */}
      <main className="flex-1 overflow-y-auto bg-slate-50">
        <div className="sticky top-0 z-10 px-8 py-3.5 border-b border-slate-200 flex items-center justify-between bg-white/90 backdrop-blur-sm">
          <div className="flex items-center gap-2 text-sm text-slate-500">
            <span>FinChartAudit</span><ChevronRight size={13} className="text-slate-300" /><span className="text-slate-900 font-medium">{NAV_ITEMS.find(n => n.id === page)?.label}</span>
          </div>
          <div className="flex items-center gap-2">
            <Badge color="blue">SEC Reg G</Badge>
            <Badge color="purple">Item 10(e)</Badge>
            <button onClick={() => setShowMisleaders(true)} className="inline-flex items-center gap-1.5 px-2.5 py-0.5 rounded-full text-xs font-medium border bg-indigo-50 text-indigo-700 border-indigo-200 hover:bg-indigo-100 transition-colors">
              <HelpCircle size={10} />Detection Reference
            </button>
          </div>
        </div>
        <div className="px-8 py-6 max-w-full">
          {Object.entries(pageComponents).map(([id, component]) => (
            <div key={id} style={{ display: page === id ? "block" : "none" }}>{component}</div>
          ))}
        </div>
      </main>

      {/* Detection reference modal */}
      {showMisleaders && <DetectionReferenceModal onClose={() => setShowMisleaders(false)} />}
    </div>
  );
}
