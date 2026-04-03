# Baseline Office0 Visualization Summary

Artifacts in this folder summarize the baseline-derived `Replica/office0` experiments.

## Included Experiments

- `baseline hiera_l`: mIoU `35.6%`, runtime `710.54s`
- `mobile SAM`: mIoU `28.4%`, runtime `1718.14s`
- `FastSAM`: mIoU `27.7%`, runtime `640.45s`
- `SAM2 hiera_s`: mIoU `29.5%`, runtime `606.74s`
- `SAM2 hiera_t`: mIoU `30.6%`, runtime `568.27s`
- `prob-v1 hiera_t`: mIoU `31.5%`, runtime `559.10s`
- `prob-v1 hiera_l`: mIoU `34.1%`, runtime `648.45s`
- `prob-v2 soft+hiera_l`: mIoU `35.4%`, runtime `242.09s`

## Quick Takeaways

- Best `mIoU`: `baseline hiera_l` at `35.6%`
- Fastest runtime: `prob-v2 soft+hiera_l` at `242.09s`
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
