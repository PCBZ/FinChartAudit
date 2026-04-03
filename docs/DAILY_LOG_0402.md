# Daily Log — 2026-04-02

## Overview

T2 pipeline 大规模架构搜索：从 V3 到 V8 共 10+ 架构变体，最终通过 ViT classifier selective veto 实现 PT F1=45.2%，首次超过所有 baseline。核心发现：工具应做验证而非注入，VLM 盲区是注意力问题而非能力问题。

---

## 1. 基础设施改进

### OCR: PaddleOCR → RapidOCR + 缓存
- 默认 backend 切为 RapidOCR（2.5x 快，质量一致）
- 加了 content-hash 磁盘缓存：首次 50min，后续 ~3 秒（694x 加速）
- 文件：`finchartaudit/tools/traditional_ocr.py`

### DePlot: chart-to-table 提取
- Google DePlot (Pix2Struct 282M) GPU 推理 ~8s/chart
- 带 content-hash 缓存
- 文件：`finchartaudit/tools/deplot.py`

### GPU PyTorch
- 安装 PyTorch 2.11.0+cu128（RTX 5060 8GB）
- DePlot GPU 加速：CPU 95s → GPU 8s/chart

---

## 2. 架构搜索全记录

### 实验结果总表（Per-Type 指标）

| 架构 | PT F1 | PT Prec | PT Rec | FP | FN | EM | VLM calls |
|------|-------|---------|--------|-----|-----|------|-----------|
| B vision-only (baseline) | 39.3% | 35.7% | 43.7% | 157 | 112 | 41.0% | 1 |
| V3 (OCR+Rules 注入 prompt) | 38.9% | 31.3% | 51.3% | 224 | 97 | 30.3% | 1 |
| V4 (B prompt + CLEAN veto) | 43.5% | 44.3% | 42.7% | 107 | 114 | 45.0% | 1 |
| V5 (V4 + DePlot rules ADD) | 37.6% | 30.0% | 50.3% | 233 | 99 | 34.3% | 1 |
| V6a (targeted 4-at-once re-ask) | 43.0% | 36.5% | 52.3% | 181 | 95 | 36.5% | 2 |
| V6b (收紧 Call 2 触发) | 43.0% | 36.4% | 52.3% | — | — | — | 2 |
| V6c (要求证据) | 34.0% | 22.8% | 66.8% | — | — | — | 2 |
| V7 (逐类型 sequential re-ask) | 40.1% | 30.1% | 59.8% | 276 | 80 | 25.1% | ~7 |
| V7 + expanded veto | 40.9% | 31.6% | 57.8% | 249 | 84 | 26.9% | ~7 |
| V8 (V7 + self-consistency 3-vote) | 38.9% | 29.1% | 58.8% | 285 | 82 | 22.9% | ~19 |
| V7 + classifier veto @0.40 | 44.3% | 39.9% | 49.7% | 149 | 100 | — | ~7 |
| **V4 + classifier veto @0.20** | **45.2%** | **50.0%** | 41.2% | **82** | 117 | — | **1** |

### 架构演化路线

```
B baseline (39.3%) 
  → V3 注入 OCR 到 prompt (38.9%, 更差) 
  → V4 只做 CLEAN veto (43.5%, 首次超 B)
  → V5 DePlot ADD (37.6%, FP 爆炸)
  → V6a 4-at-once re-ask (43.0%)
  → V6b/c 调 Call 2 (无效)
  → V7 逐类型 re-ask (40.1%, recall 最高但 FP 爆炸)
  → V8 self-consistency (38.9%, 投票无效)
  → V7 + classifier veto (44.3%, 突破)
  → V4 + classifier veto (45.2%, 最优)
```

---

## 3. 核心发现

### 发现 1：工具注入 VLM prompt 有害
OCR/DePlot 数据注入 prompt → PT F1 下降。OCR [FLAGGED] 准确率 0-4%，[CLEAN] 准确率 99-100%。工具只适合做后处理验证。

### 发现 2：VLM 盲区是注意力问题，非能力问题
单独问 "这张图有 inconsistent tick intervals 吗？" → VLM 能正确回答 YES。但在 12 类型同时检测的通用 prompt 里被淹没。

| 类型 | 通用 prompt recall | 单独问 | 差距原因 |
|------|-------------------|--------|---------|
| tick intervals | 0-6% | YES | 注意力稀释 |
| binning | 0% | YES | 注意力稀释 |
| discretized | 0% | YES | 注意力稀释 |

### 发现 3：Self-consistency 投票对校准错误无效
V8 的 3-vote majority 没有减少 FP（276→285）。VLM 的错误是系统性的（"我认为这张图有 item order 问题"），不是随机幻觉。3 次问 3 次都说 YES。

### 发现 4：Binary F1 不够评估 chart misleader detection
B 的 83% binary F1 伴随 157 per-type FP、42% clean chart 误报率、20.9% misrepresentation precision。Per-type F1 才能反映真实检测质量。

### 发现 5：Synth 训练的 classifier 可以做 selective veto
ViT-B 在 57K synth 上训练（val F1=0.522），尽管有 domain gap，对特定类型的 veto 准确率很高：

| 类型 | Veto 准确率 |
|------|-----------|
| tick intervals | 100% |
| binning | 100% |
| truncated axis | 91% |
| axis range | 94% |
| item order | 90% |

### 发现 6：理论天花板分析

| 架构 | 当前 PT F1 | TP | 天花板 (FP→0) |
|------|----------|-----|-------------|
| V4 | 43.5% | 85 | 59.9% |
| V6a | 43.0% | 104 | 68.6% |
| V7 | 40.1% | 119 | 74.8% |

V7 天花板最高（找到最多 TP），但需要强验证层。

---

## 4. Claim Decomposition 可行性分析

分析了 V7 的 276 FP 是否能被现有工具验证：

| 可验证度 | FP 数 | 占比 | 类型 |
|---------|-------|------|------|
| 能验证 | 27 | 10% | truncated (OCR), pie (DePlot sum) |
| 部分能验证 | 68 | 25% | axis range, line chart, binning |
| 无法验证 | 181 | 66% | misrep, discretized, item order, tick |

结论：现有工具不足以支撑 claim decomposition 架构。

---

## 5. ViT Classifier 训练

### 配置
- 模型：ViT-B/16 (86M params), pretrained ImageNet
- 数据：57,665 Misviz-synth (90/10 train/val split)
- 训练：3 epochs (epoch 0 head only, epoch 1-2 backbone unfrozen)
- 硬件：RTX 5060 8GB, ~30 min total

### 结果
```
Epoch 1: val_F1=0.409 (head only)
Epoch 2: val_F1=0.481 (backbone unfrozen)  
Epoch 3: val_F1=0.522 (best)
```

### Synth per-type performance (threshold=0.5)
- 3d: F1=1.00, dual axis: F1=0.99, line chart: F1=0.95 (excellent)
- misrepresentation: F1=0.62, discretized: F1=0.62 (good)
- tick intervals: F1=0.48, item order: F1=0.49, axis range: F1=0.52 (moderate)
- truncated axis: F1=0.14 (poor — few training samples)

### Real-world 应用：Selective veto
- 全类型 veto：PT F1 下降（domain gap 过大）
- **Selective veto（5 种高准确率类型）**：PT F1 提升

---

## 6. 最优架构详解：V4 + Classifier Selective Veto

```
图片
  │
  ▼
VLM Call 1 (B's prompt + 6 few-shots)
  │
  ▼
OCR CLEAN Veto
  ├─ truncated axis: 轴含 0 → 否决 (99% 准确)
  └─ dual axis: 无右轴 → 否决 (100% 准确)
  │
  ▼
ViT Classifier Selective Veto (threshold=0.20)
  ├─ tick intervals: prob < 0.2 → 否决
  ├─ binning: prob < 0.2 → 否决
  ├─ truncated axis: prob < 0.2 → 否决
  ├─ axis range: prob < 0.2 → 否决
  └─ item order: prob < 0.2 → 否决
  │
  ▼
最终结果: PT F1=45.2%, Prec=50.0%
```

### vs B baseline per-type

| 类型 | B F1 | V4+clf F1 | 改善 |
|------|------|-----------|------|
| 3d | 69.6% | **80.8%** | +11pp |
| truncated axis | 38.8% | **53.1%+** | +14pp |
| misrepresentation | 32.9% | **41.7%** | +9pp |
| binning | 9.5% | **34.5%** | +25pp |
| tick intervals | 0% | **20.5%** | +21pp |
| inverted axis | 34.8% | **43.5%** | +9pp |

---

## 7. 剩余瓶颈

| 瓶颈 | FP | 需要的能力 |
|------|-----|-----------|
| misrepresentation 幻觉 | 39 | 像素级 bar 高度测量 |
| item order | 56 (V7) | 语义理解（类别自然顺序） |
| discretized continuous | 38 (V7) | 语义理解（变量类型） |
| Synth → Real domain gap | classifier 受限 | Domain adaptation 或 real-world 标注 |

### 未探索的方向
1. **Sonnet 模型** — 更强的 VLM 可能直接改善 calibration
2. **Grounding VLM (Qwen2-VL)** — 能定位 tick marks 像素位置
3. **Domain adaptation** — 在少量 real-world 数据上微调 classifier
4. **HuggingFace 精确对齐** — 消除图片匹配误差

---

## 8. 文件清单

### 新建
- `finchartaudit/tools/deplot.py` — DePlot chart-to-table
- `finchartaudit/tools/table_rules.py` — DePlot 数据规则检查
- `train_classifier.py` — ViT classifier 训练
- `run_pipeline_v3_veto.py` ~ `v8_selfconsist.py` — 8 个实验脚本
- `apply_expanded_veto.py` — 扩展 veto 分析
- `simulate_routing.py` — 路由策略模拟
- `EXPERIMENT_REPORT_0402.md` — 完整实验报告

### 修改
- `finchartaudit/tools/traditional_ocr.py` — RapidOCR + 缓存
- `finchartaudit/agents/t2_pipeline.py` — V3 tiered verdicts
- `run_pipeline_full.py` — V3 prompt 集成

### 数据产物
- `data/ocr_cache/` — RapidOCR 缓存 (271 charts)
- `data/deplot_cache/` — DePlot 缓存 (271 charts)
- `data/models/chart_misleader_vit.pt` — ViT classifier (328MB)
- `data/eval_results/v4_combo/` ~ `v8_selfconsist/` — 所有实验结果

### 发布
- PR: https://github.com/logisticPM/genai/pull/1
- Wiki: https://github.com/PCBZ/FinChartAudit/wiki/T2-Pipeline-Experiment-Report-2026-04-02

## Next Steps

1. **HuggingFace 精确对齐** — 消除 146 张图片匹配误差
2. **Sonnet 评估** — 在 V4+clf 架构上换 Sonnet，验证 upper bound
3. **论文撰写** — 用今天的数据写 RQ1/RQ2 实验部分
4. **T3 SEC demo** — 把 T2 改进反馈到完整系统
