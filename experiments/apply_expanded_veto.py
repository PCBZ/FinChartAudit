"""Apply expanded CLEAN veto on V7 results (no VLM calls needed)."""
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

# Load V7 raw results
results = json.loads(open('data/eval_results/v7_sequential/raw_results.json').read())

# Load OCR cache
ocr_cache = {}
for f in sorted(Path('data/eval_results/pipeline_full/ocr_cache').glob('*.json')):
    try:
        ocr_cache.update(json.loads(f.read_text(encoding='utf-8')))
    except Exception:
        pass

# Load DePlot cache
from finchartaudit.tools.deplot import DePlotTool
from data_tools.misviz.loader import MisvizLoader

deplot = DePlotTool(device="cpu")
loader = MisvizLoader()
real_data = loader.load_real()

b_data = json.loads(open('C:/Users/chntw/Documents/7180/PCBZ_FinChartAudit/results/claude_vision_only.json').read())
local_by_bbox = {}
for i, item in enumerate(real_data):
    bbox = item.get("bbox", [])
    if bbox:
        bk = json.dumps(bbox, sort_keys=True)
        if bk not in local_by_bbox:
            local_by_bbox[bk] = i

matched = {}
used = set()
for b in b_data['results']:
    bbox = b.get('bbox', [])
    if bbox:
        bk = json.dumps(bbox, sort_keys=True)
        if bk in local_by_bbox:
            matched[str(b['id'])] = local_by_bbox[bk]
            used.add(local_by_bbox[bk])

content_to_local = defaultdict(list)
for i, d in enumerate(real_data):
    if i not in used:
        key = (frozenset(d.get('misleader', [])), tuple(sorted(d.get('chart_type', []))))
        content_to_local[key].append(i)

for b in b_data['results']:
    if str(b['id']) in matched:
        continue
    key = (frozenset(b['gt_misleaders']), tuple(sorted(b.get('chart_type', []))))
    cands = [i for i in content_to_local.get(key, []) if i not in used]
    if cands:
        matched[str(b['id'])] = cands[0]
        used.add(cands[0])

dp_cache = {}
for r in results:
    iid = r['instance_id']
    if iid in matched:
        idx = matched[iid]
        inst = loader.get_real_instance(idx)
        dp = deplot._cache_load(inst.image_path)
        if dp:
            dp_cache[iid] = dp


def has_temporal_labels(dp):
    if not dp or not dp.get('rows'):
        return False
    headers = dp.get('headers', [])
    temporal_kw = ['year', 'date', 'month', 'quarter', 'q1', 'q2', 'q3', 'q4',
                   'jan', 'feb', 'mar', 'apr', 'may', 'jun']
    if headers and any(kw in headers[0].lower() for kw in temporal_kw):
        return True
    years = 0
    for row in dp['rows'][:10]:
        if row:
            for m in re.findall(r'\b(19|20)\d{2}\b', row[0]):
                years += 1
    return years >= 2


def get_data_spread(dp):
    if not dp or not dp.get('rows'):
        return None
    data_vals = []
    for row in dp['rows']:
        for cell in row[1:]:
            for m in re.findall(r'-?\d+\.?\d*', cell.replace(',', '').replace('%', '')):
                try:
                    v = float(m)
                    if abs(v) < 1e6:
                        data_vals.append(v)
                except ValueError:
                    pass
    if len(data_vals) < 3:
        return None
    dmax = max(data_vals)
    if dmax <= 0:
        return None
    return (dmax - min(data_vals)) / dmax


# Apply expanded veto
new_results = []
veto_stats = Counter()

for r in results:
    iid = r['instance_id']
    predicted = list(r['predicted'])
    pp_log = list(r.get('pp_log', []))
    ocr = ocr_cache.get(iid, {})
    dp = dp_cache.get(iid, {})

    new_pred = []
    for name in predicted:
        vetoed = False

        # 1. OCR: truncated axis
        if name == 'truncated axis' and ocr and 'error' not in ocr:
            axis_values = ocr.get('axis_values', [])
            rule_results = ocr.get('rule_results', [])
            trunc_flagged = any('instead of 0' in r.lower() or 'exaggerated' in r.lower()
                                for r in rule_results if r.startswith('truncated_axis:'))
            if axis_values and not trunc_flagged and min(axis_values) <= 0:
                pp_log.append('VETO truncated_axis: OCR axis includes 0')
                veto_stats['truncated_axis_ocr'] += 1
                vetoed = True

        # 2. OCR: dual axis
        if name == 'dual axis' and ocr and 'error' not in ocr:
            right_axis_values = ocr.get('right_axis_values', [])
            rule_results = ocr.get('rule_results', [])
            dual_flagged = any(r.startswith('dual_axis:') for r in rule_results)
            if not right_axis_values and not dual_flagged:
                pp_log.append('VETO dual_axis: no right Y-axis in OCR')
                veto_stats['dual_axis_ocr'] += 1
                vetoed = True

        # 3. NEW: DePlot axis_range CLEAN (spread > 0.5)
        if name == 'inappropriate axis range' and dp:
            spread = get_data_spread(dp)
            if spread is not None and spread > 0.5:
                pp_log.append(f'VETO axis_range: DePlot spread={spread:.2f}>0.5')
                veto_stats['axis_range_deplot'] += 1
                vetoed = True

        # 4. NEW: DePlot item_order temporal labels
        if name == 'inappropriate item order' and dp:
            if has_temporal_labels(dp):
                pp_log.append('VETO item_order: temporal labels detected')
                veto_stats['item_order_temporal'] += 1
                vetoed = True

        # 5. NEW: Cross-type misrep+pie co-occurrence
        if name == 'misrepresentation' and 'inappropriate use of pie chart' in predicted:
            pp_log.append('VETO misrep: co-occurs with pie chart issue')
            veto_stats['misrep_pie_cross'] += 1
            vetoed = True

        if not vetoed:
            new_pred.append(name)

    new_results.append({
        'instance_id': iid,
        'ground_truth': r['ground_truth'],
        'predicted': new_pred,
        'pp_log': pp_log,
    })

# Calculate metrics
tp = fp = fn = 0
for r in new_results:
    g = set(r['ground_truth'])
    p = set(r['predicted'])
    for x in p:
        if x in g: tp += 1
        else: fp += 1
    for x in g:
        if x not in p: fn += 1

prec = tp / (tp + fp) if (tp + fp) else 0
rec = tp / (tp + fn) if (tp + fn) else 0
f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0
em = sum(1 for r in new_results if set(r['ground_truth']) == set(r['predicted'])) / len(new_results)

# Binary
btp = bfp = bfn = btn = 0
for r in new_results:
    g, p = bool(r['ground_truth']), bool(r['predicted'])
    if g and p: btp += 1
    elif not g and p: bfp += 1
    elif g and not p: bfn += 1
    else: btn += 1
bp = btp / (btp + bfp) if (btp + bfp) else 0
br = btp / (btp + bfn) if (btp + bfn) else 0
bf1 = 2 * bp * br / (bp + br) if (bp + br) else 0

print('=== V7 + Expanded CLEAN Veto ===')
print(f'Binary:   F1={bf1*100:.1f}%  Prec={bp*100:.1f}%  Rec={br*100:.1f}%')
print(f'Per-type: F1={f1*100:.1f}%  Prec={prec*100:.1f}%  Rec={rec*100:.1f}%')
print(f'TP={tp}  FP={fp}  FN={fn}  EM={em*100:.1f}%')
print()
print('Veto stats:')
for k, v in veto_stats.most_common():
    print(f'  {k:30s} {v:4d}')
print(f'  Total vetoed: {sum(veto_stats.values())}')

print()
print('Comparison:')
print(f'  {"":35s} {"PT F1":>7s} {"PT Prec":>8s} {"PT Rec":>7s} {"FP":>5s} {"FN":>5s} {"EM":>7s}')
print(f'  {"B vision-only":35s} {"39.3%":>7s} {"35.7%":>8s} {"43.7%":>7s} {"157":>5s} {"112":>5s} {"41.0%":>7s}')
print(f'  {"V4 (best per-type)":35s} {"43.5%":>7s} {"44.3%":>8s} {"42.7%":>7s} {"107":>5s} {"114":>5s} {"45.0%":>7s}')
print(f'  {"V7 original":35s} {"40.1%":>7s} {"30.1%":>8s} {"59.8%":>7s} {"276":>5s} {"80":>5s} {"25.1%":>7s}')
print(f'  {"V7 + expanded veto":35s} {f1*100:6.1f}% {prec*100:7.1f}% {rec*100:6.1f}% {fp:5d} {fn:5d} {em*100:6.1f}%')

# Save
out = Path('data/eval_results/v7_expanded_veto')
out.mkdir(parents=True, exist_ok=True)
(out / 'raw_results.json').write_text(json.dumps(new_results, indent=2, default=str), encoding='utf-8')
print(f'\nSaved to {out}')
