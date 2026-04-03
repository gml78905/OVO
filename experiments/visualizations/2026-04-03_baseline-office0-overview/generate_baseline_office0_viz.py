from __future__ import annotations

import json
import re
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

ROOT = Path('/ws/external')
OUT = ROOT / 'experiments/visualizations/2026-04-03_baseline-office0-overview'
OUT.mkdir(parents=True, exist_ok=True)

sns.set_theme(style='whitegrid')
plt.rcParams['figure.dpi'] = 160
plt.rcParams['savefig.dpi'] = 200


def load_structured_metrics(path: Path) -> dict:
    data = json.loads(path.read_text())
    if 'run' in data:
        return {
            'experiment_id': data.get('experiment_id'),
            'name': data.get('name'),
            'date': data.get('date'),
            'branch': data.get('branch'),
            'source': str(path),
            'runtime_sec': data['run']['scene_time_sec'],
            'avg_fps': data['run']['avg_fps'],
            'avg_spf': data['run']['avg_spf'],
            'avg_t_sam': data['run']['avg_t_sam'],
            'avg_t_obj': data['run']['avg_t_obj'],
            'avg_t_clip': data['run']['avg_t_clip'],
            'avg_t_up': data['run']['avg_t_up'],
            'avg_vram_gb': data['run']['avg_vram_gb'],
            'max_vram_gb': data['run']['max_vram_gb'],
            'max_reserved_vram_gb': data['run']['max_vram_reserved_gb'],
            'miou': data['eval']['miou'] * 100.0,
            'macc': data['eval']['macc'] * 100.0,
            'f_miou': data['eval']['f_miou'] * 100.0,
            'f_macc': data['eval']['f_macc'] * 100.0,
            'head_iou': data['eval'].get('head_iou', 0.0) * 100.0,
            'comm_iou': data['eval'].get('comm_iou', 0.0) * 100.0,
            'tail_iou': data['eval'].get('tail_iou', 0.0) * 100.0,
            'proto_objects': None,
            'semantic_groups': None,
            'multi_member_groups': None,
            'active_pair_edges': None,
            'grouping_variant': False,
            'runtime_caveat': False,
        }
    return {
        'experiment_id': None,
        'name': path.parent.name,
        'date': path.parent.name.split('_')[0],
        'branch': None,
        'source': str(path),
        'runtime_sec': data['runtime_sec'],
        'avg_fps': data['avg_fps'],
        'avg_spf': data['avg_spf'],
        'avg_t_sam': data['avg_t_sam'],
        'avg_t_obj': data['avg_t_obj'],
        'avg_t_clip': data['avg_t_clip'],
        'avg_t_up': data['avg_t_up'],
        'avg_vram_gb': data['avg_vram_gb'],
        'max_vram_gb': data['max_vram_gb'],
        'max_reserved_vram_gb': data['max_reserved_vram_gb'],
        'miou': data['miou'],
        'macc': data['macc'],
        'f_miou': data['f_miou'],
        'f_macc': data['f_macc'],
        'head_iou': None,
        'comm_iou': None,
        'tail_iou': None,
        'proto_objects': data.get('proto_objects'),
        'semantic_groups': data.get('semantic_groups'),
        'multi_member_groups': data.get('multi_member_groups'),
        'active_pair_edges': data.get('active_pair_edges'),
        'grouping_variant': data.get('semantic_groups') is not None,
        'runtime_caveat': 'v2_soft_fusion' in path.parent.name,
    }


def load_mobile_stdout(path: Path) -> dict:
    text = path.read_text()
    def grab(pattern: str) -> float:
        return float(re.search(pattern, text).group(1))
    miou, macc, f_miou, f_macc = map(float, re.findall(r'mIoU:\s+([0-9.]+)%; mAcc:\s+([0-9.]+)%.*?f-mIoU:\s+([0-9.]+)%; f-mAcc:\s+([0-9.]+)%', text, re.S)[-1])
    head_iou, head_acc, comm_iou, comm_acc, tail_iou, tail_acc = map(float, re.findall(r'head:\s+([0-9.]+)%\s*\nhead:\s+([0-9.]+)%\s*\n---\s*\ncomm:\s+([0-9.]+)%\s*\ncomm:\s+([0-9.]+)%\s*\n---\s*\ntail:\s+([0-9.]+)%\s*\ntail:\s+([0-9.]+)%', text, re.S)[-1])
    return {
        'experiment_id': 'mobile-sam',
        'name': 'mobile_sam_office0',
        'date': '2026-04-02',
        'branch': 'exp/mobile-sam-baseline-office0',
        'source': str(path),
        'runtime_sec': grab(r'Scene office0 took: ([0-9.]+)'),
        'avg_fps': grab(r"'Avg avg_fps': np.float64\(([0-9.]+)\)"),
        'avg_spf': grab(r"'Avg spf': np.float64\(([0-9.]+)\)"),
        'avg_t_sam': grab(r"'Avg t_sam': np.float64\(([0-9.]+)\)"),
        'avg_t_obj': grab(r"'Avg t_obj': np.float64\(([0-9.]+)\)"),
        'avg_t_clip': grab(r"'Avg t_clip': np.float64\(([0-9.]+)\)"),
        'avg_t_up': grab(r"'Avg t_up': np.float64\(([0-9.]+)\)"),
        'avg_vram_gb': grab(r"'Avg vram': np.float64\(([0-9.]+)\)"),
        'max_vram_gb': grab(r"'Max vRAM': ([0-9.]+)"),
        'max_reserved_vram_gb': grab(r"'Avg vram_reserved': np.float64\(([0-9.]+)\)"),
        'miou': miou,
        'macc': macc,
        'f_miou': f_miou,
        'f_macc': f_macc,
        'head_iou': head_iou,
        'comm_iou': comm_iou,
        'tail_iou': tail_iou,
        'proto_objects': None,
        'semantic_groups': None,
        'multi_member_groups': None,
        'active_pair_edges': None,
        'grouping_variant': False,
        'runtime_caveat': False,
    }


records = [
    load_structured_metrics(ROOT / 'experiments/2026-04-02_baseline-office0/metrics.json'),
    load_mobile_stdout(ROOT / 'experiments/2026-04-02_mobile-sam-office0/stdout.log'),
    load_structured_metrics(ROOT / 'experiments/2026-04-02_fast-sam-office0/metrics.json'),
    load_structured_metrics(ROOT / 'experiments/2026-04-02_sam2-hiera-s-office0/metrics.json'),
    load_structured_metrics(ROOT / 'experiments/2026-04-02_sam2-hiera-t-office0/metrics.json'),
    load_structured_metrics(ROOT / 'experiments/2026-04-02_prob-tracklet-graph-v1-office0/metrics.json'),
    load_structured_metrics(ROOT / 'experiments/2026-04-03_prob-tracklet-graph-v1-hiera-l-office0/metrics.json'),
    load_structured_metrics(ROOT / 'experiments/2026-04-03_prob-tracklet-graph-v2-soft-fusion-hiera-l-office0/metrics.json'),
]

df = pd.DataFrame(records)
label_map = {
    'baseline_office0': 'baseline\nhiera_l',
    'mobile_sam_office0': 'mobile\nSAM',
    'fast_sam_office0': 'FastSAM',
    'sam2_hiera_s_office0': 'SAM2\nhiera_s',
    'sam2_hiera_t_office0': 'SAM2\nhiera_t',
    '2026-04-02_prob-tracklet-graph-v1-office0': 'prob-v1\nhiera_t',
    '2026-04-03_prob-tracklet-graph-v1-hiera-l-office0': 'prob-v1\nhiera_l',
    '2026-04-03_prob-tracklet-graph-v2-soft-fusion-hiera-l-office0': 'prob-v2\nsoft+hiera_l',
}
df['label'] = df['name'].map(label_map).fillna(df['name'])
order = [
    'baseline_office0',
    'mobile_sam_office0',
    'fast_sam_office0',
    'sam2_hiera_s_office0',
    'sam2_hiera_t_office0',
    '2026-04-02_prob-tracklet-graph-v1-office0',
    '2026-04-03_prob-tracklet-graph-v1-hiera-l-office0',
    '2026-04-03_prob-tracklet-graph-v2-soft-fusion-hiera-l-office0',
]
df['order'] = df['name'].apply(order.index)
df = df.sort_values('order').reset_index(drop=True)

baseline = df[df['name'] == 'baseline_office0'].iloc[0]
df['delta_miou_pp_vs_baseline'] = df['miou'] - baseline['miou']
df['delta_macc_pp_vs_baseline'] = df['macc'] - baseline['macc']
df['delta_runtime_sec_vs_baseline'] = df['runtime_sec'] - baseline['runtime_sec']
df['delta_vram_gb_vs_baseline'] = df['avg_vram_gb'] - baseline['avg_vram_gb']

df.to_csv(OUT / 'baseline_office0_summary.csv', index=False)

palette = ['#1f2937', '#b45309', '#0f766e', '#1d4ed8', '#2563eb', '#7c3aed', '#be185d', '#059669']

# Quality bars
fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
for ax, metric, title in zip(
    axes.flat,
    ['miou', 'macc', 'f_miou', 'f_macc'],
    ['mIoU', 'mAcc', 'f-mIoU', 'f-mAcc'],
):
    sns.barplot(data=df, x='label', y=metric, palette=palette, ax=ax)
    ax.axhline(baseline[metric], color='#111827', linestyle='--', linewidth=1)
    ax.set_title(title)
    ax.set_xlabel('')
    ax.set_ylabel('%')
    ax.tick_params(axis='x', rotation=0)
fig.suptitle('Baseline Office0 Quality Comparison', fontsize=16)
fig.savefig(OUT / 'quality_bars.png')
plt.close(fig)

# Runtime/memory
fig, axes = plt.subplots(1, 3, figsize=(14, 4.8), constrained_layout=True)
for ax, metric, title, ylabel in [
    (axes[0], 'runtime_sec', 'Scene Runtime', 'seconds'),
    (axes[1], 'avg_t_sam', 'Mean SAM Time', 'seconds'),
    (axes[2], 'avg_vram_gb', 'Mean VRAM', 'GB'),
]:
    sns.barplot(data=df, x='label', y=metric, palette=palette, ax=ax)
    ax.axhline(baseline[metric], color='#111827', linestyle='--', linewidth=1)
    ax.set_title(title)
    ax.set_xlabel('')
    ax.set_ylabel(ylabel)
    ax.tick_params(axis='x', rotation=0)
fig.suptitle('Runtime and Memory Comparison', fontsize=16)
fig.savefig(OUT / 'runtime_memory.png')
plt.close(fig)

# Long-tail breakdown where available
breakdown = df.dropna(subset=['head_iou', 'comm_iou', 'tail_iou']).copy()
long_df = breakdown.melt(id_vars=['label'], value_vars=['head_iou', 'comm_iou', 'tail_iou'], var_name='slice', value_name='iou')
fig, ax = plt.subplots(figsize=(12, 5), constrained_layout=True)
sns.barplot(data=long_df, x='label', y='iou', hue='slice', ax=ax)
ax.set_title('Head / Common / Tail IoU')
ax.set_xlabel('')
ax.set_ylabel('%')
ax.legend(title='slice', labels=['head', 'common', 'tail'])
fig.savefig(OUT / 'head_common_tail.png')
plt.close(fig)

# Probabilistic grouping stats
prob_df = df[df['grouping_variant']].copy()
if not prob_df.empty:
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.5), constrained_layout=True)
    sns.barplot(data=prob_df, x='label', y='proto_objects', ax=axes[0], color='#64748b')
    sns.barplot(data=prob_df, x='label', y='semantic_groups', ax=axes[1], color='#0f766e')
    sns.barplot(data=prob_df, x='label', y='active_pair_edges', ax=axes[2], color='#7c3aed')
    axes[0].set_title('Proto-objects')
    axes[1].set_title('Semantic Groups')
    axes[2].set_title('Active Pair Edges')
    for ax in axes:
        ax.set_xlabel('')
        ax.tick_params(axis='x', rotation=0)
    fig.suptitle('Probabilistic Grouping Structure', fontsize=16)
    fig.savefig(OUT / 'grouping_structure.png')
    plt.close(fig)

# Dashboard
fig = plt.figure(figsize=(15, 11), constrained_layout=True)
gs = fig.add_gridspec(3, 2)
ax1 = fig.add_subplot(gs[0, 0])
ax2 = fig.add_subplot(gs[0, 1])
ax3 = fig.add_subplot(gs[1, 0])
ax4 = fig.add_subplot(gs[1, 1])
ax5 = fig.add_subplot(gs[2, :])

sns.barplot(data=df, x='label', y='miou', palette=palette, ax=ax1)
ax1.axhline(baseline['miou'], color='#111827', linestyle='--', linewidth=1)
ax1.set_title('mIoU')
ax1.set_xlabel('')
ax1.set_ylabel('%')

sns.barplot(data=df, x='label', y='f_miou', palette=palette, ax=ax2)
ax2.axhline(baseline['f_miou'], color='#111827', linestyle='--', linewidth=1)
ax2.set_title('f-mIoU')
ax2.set_xlabel('')
ax2.set_ylabel('%')

sns.barplot(data=df, x='label', y='runtime_sec', palette=palette, ax=ax3)
ax3.axhline(baseline['runtime_sec'], color='#111827', linestyle='--', linewidth=1)
ax3.set_title('Runtime')
ax3.set_xlabel('')
ax3.set_ylabel('sec')

sns.barplot(data=df, x='label', y='avg_vram_gb', palette=palette, ax=ax4)
ax4.axhline(baseline['avg_vram_gb'], color='#111827', linestyle='--', linewidth=1)
ax4.set_title('Mean VRAM')
ax4.set_xlabel('')
ax4.set_ylabel('GB')

scatter_df = df.copy()
for _, row in scatter_df.iterrows():
    ax5.scatter(row['runtime_sec'], row['miou'], s=90, color=palette[int(row['order'])])
    ax5.text(row['runtime_sec'] + 8, row['miou'] + 0.08, row['label'].replace('\n', ' '), fontsize=8)
ax5.set_title('Quality / Runtime Frontier')
ax5.set_xlabel('scene runtime (sec)')
ax5.set_ylabel('mIoU (%)')
if df['runtime_caveat'].any():
    caveat_rows = df[df['runtime_caveat']]['label'].str.replace('\n', ' ', regex=False).tolist()
    fig.text(0.01, 0.01, 'Runtime caveat: ' + ', '.join(caveat_rows) + ' likely reused cached SAM artifacts, so quality is more trustworthy than speed for those points.', fontsize=9)
fig.suptitle('Baseline Office0 Experiment Dashboard', fontsize=18)
fig.savefig(OUT / 'dashboard.png')
plt.close(fig)

# Markdown summary
best_miou = df.loc[df['miou'].idxmax()]
fastest = df.loc[df['runtime_sec'].idxmin()]
summary = f'''# Baseline Office0 Visualization Summary

Artifacts in this folder summarize the baseline-derived `Replica/office0` experiments.

## Included Experiments

{chr(10).join(f'- `{row.label.replace(chr(10), " ")}`: mIoU `{row.miou:.1f}%`, runtime `{row.runtime_sec:.2f}s`' for _, row in df.iterrows())}

## Quick Takeaways

- Best `mIoU`: `{best_miou['label'].replace(chr(10), ' ')}` at `{best_miou['miou']:.1f}%`
- Fastest runtime: `{fastest['label'].replace(chr(10), ' ')}` at `{fastest['runtime_sec']:.2f}s`
- Best plain SAM-family tradeoff without probabilistic grouping: `SAM2 hiera_t`
- Best probabilistic variant by quality: `prob-v2 soft+hiera_l`

## Caveat

- `prob-v2 soft+hiera_l` reused cached SAM artifacts on rerun, so its runtime is not directly comparable to the cold-cache runs. Its quality metrics are still useful.

## Files

- [dashboard.png](/ws/external/experiments/visualizations/2026-04-03_baseline-office0-overview/dashboard.png)
- [quality_bars.png](/ws/external/experiments/visualizations/2026-04-03_baseline-office0-overview/quality_bars.png)
- [runtime_memory.png](/ws/external/experiments/visualizations/2026-04-03_baseline-office0-overview/runtime_memory.png)
- [head_common_tail.png](/ws/external/experiments/visualizations/2026-04-03_baseline-office0-overview/head_common_tail.png)
- [grouping_structure.png](/ws/external/experiments/visualizations/2026-04-03_baseline-office0-overview/grouping_structure.png)
- [baseline_office0_summary.csv](/ws/external/experiments/visualizations/2026-04-03_baseline-office0-overview/baseline_office0_summary.csv)
'''
(OUT / 'README.md').write_text(summary)
print('wrote', OUT)
