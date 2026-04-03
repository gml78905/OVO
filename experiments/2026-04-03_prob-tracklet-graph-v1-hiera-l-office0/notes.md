# Probabilistic Tracklet Graph V1 Hiera-L Office0

- Experiment id: `exp-006`
- Name: `prob_tracklet_graph_v1_hiera_l_office0`
- Date: `2026-04-03`
- Branch: `exp/prob-tracklet-graph-v1-hiera-l-office0`
- Code commit: `6baaf80`
- Scene: `Replica/office0`
- Baseline: `exp-001`
- Secondary comparison: `exp-005`

## Goal

Re-run the first reversible probabilistic grouping design on the original `SAM2.1 hiera_l` baseline setting so we can see whether the stronger SAM backbone makes the grouping graph more useful, or whether the probabilistic layer still gives away too much quality versus the true baseline.

## Code Changes

- No new code changes in this run. This experiment reuses the probabilistic grouping v1 implementation from [probabilistic_grouping.py](/ws/external/ovo/entities/probabilistic_grouping.py), [ovo.py](/ws/external/ovo/entities/ovo.py), [ovomapping.py](/ws/external/ovo/entities/ovomapping.py), and [run_eval.py](/ws/external/run_eval.py).
- The only run-time difference from the `exp-005` setup is the SAM backbone override: `OVO_SAM_ENCODER=hiera_l`.

## Command

```bash
OVO_DATA_ROOT=/ws/data/OVO DISABLE_WANDB=true \
OVO_SAM_ENCODER=hiera_l \
OVO_PROB_GROUPING=true \
OVO_PROB_COMMIT_CONFIDENCE=0.70 \
OVO_PROB_COMMIT_MARGIN=0.20 \
OVO_PROB_EDGE_POSTERIOR_TH=0.55 \
OVO_PROB_NEG_OBS_WEIGHT=0.20 \
python run_eval.py --dataset_name Replica --experiment_name prob_tracklet_graph_v1_hiera_l_office0 \
  --scenes office0 --run --segment --eval
```

## Result

- Scene runtime: `648.45s`
- Mean FPS: `0.319`
- Mean SPF: `2.860s`
- Mean SAM time: `2.705s`
- Mean object time: `0.057s`
- Mean CLIP time: `0.097s`
- Mean update time: `0.007s`
- Mean VRAM: `4.419 GB`
- Max VRAM: `5.85 GB`
- Max reserved VRAM: `8.33 GB`
- mIoU: `34.1%`
- mAcc: `45.9%`
- f-mIoU: `53.3%`
- f-mAcc: `66.5%`
- Final proto-objects: `141`
- Final semantic groups: `134`
- Multi-member groups: `6`
- Stored pairwise evidence edges: `3370`
- Active final grouping edges: `8`
- Ambiguous segments remembered: `228`
- Conservative hard-merge skips: `158`
- Hard committed segments: `2843`

## Delta vs Baseline (`exp-001`)

- Runtime: `-62.09s`
- Mean FPS: `+0.028`
- Mean SPF: `-0.281s`
- Mean SAM time: `-0.312s`
- Mean object time: `+0.034s`
- Mean VRAM: `-0.006 GB`
- Max VRAM: `-0.05 GB`
- mIoU: `-1.5pp`
- mAcc: `-1.5pp`
- f-mIoU: `-1.4pp`
- f-mAcc: `-2.1pp`

## Delta vs `exp-005`

- Runtime: `+89.35s`
- Mean FPS: `-0.051`
- Mean SPF: `+0.424s`
- Mean SAM time: `+0.440s`
- Mean object time: `-0.016s`
- Mean VRAM: `+0.758 GB`
- Max VRAM: `+0.802 GB`
- mIoU: `+2.6pp`
- mAcc: `+0.7pp`
- f-mIoU: `-2.2pp`
- f-mAcc: `-1.4pp`

## Analysis

- This run is a healthier comparison than `exp-005` because it starts from the real baseline backbone. The stronger `hiera_l` SAM clearly helps the probabilistic version recover standard semantic quality: `mIoU` jumped from `31.5%` to `34.1%` compared with the `hiera_t` probabilistic run.
- Even with the reversible grouping layer enabled, we still finish below the plain `hiera_l` baseline on every reported semantic metric. That means the current grouping rule is not yet a free win when the base segmentation is already strong.
- Structurally, the graph got richer. We ended with `141` proto-objects and `134` final semantic groups, which is a larger and more expressive grouping state than the `hiera_t` probabilistic run (`134 -> 130`). The `hiera_l` backbone seems to create better evidence for pair memory, but that extra grouping is not converting cleanly into filtered quality.
- The strongest memory edge in the stored graph was `[3, 5]` with posterior `0.8891`, but it did not survive the final activation rule. That is useful because it shows the memory is broader than the final semantic graph; the current centroid-and-CLIP gate is acting as a second conservative filter.
- Runtime still improved over the plain `hiera_l` baseline, which is nice, but the gain mostly comes from overall pipeline variance and lower observed `t_sam`. The real overhead of the probabilistic method is still concentrated in object association bookkeeping.
- The key takeaway is that `v1` is behaving like a conservative semantic regularizer, not yet like a fully better tracker. The next improvement should focus on using pair posterior more smoothly during descriptor fusion instead of relying on a hard final grouping threshold.

## Key Grouped Pairs

- `[4, 77, 82]`
- `[18, 53]`
- `[43, 57]`
- `[54, 89]`
- `[64, 118]`
- `[112, 113]`

## Artifacts

- Metrics: [metrics.json](/ws/external/experiments/2026-04-03_prob-tracklet-graph-v1-hiera-l-office0/metrics.json)
- Run summary: [stdout.log](/ws/external/experiments/2026-04-03_prob-tracklet-graph-v1-hiera-l-office0/stdout.log)
- Output root: `/ws/data/OVO/output/Replica/prob_tracklet_graph_v1_hiera_l_office0`
- Analysis report: [report.md](/ws/data/OVO/output/Replica/prob_tracklet_graph_v1_hiera_l_office0/analysis/report.md)
- Prediction file: `/ws/data/OVO/output/Replica/prob_tracklet_graph_v1_hiera_l_office0/instance_pred/office0.txt`

## Conclusion

The first probabilistic grouping idea still looks promising, but this rerun makes the current limitation clearer: with a stronger `hiera_l` baseline, `v1` preserves a richer reversible graph but still loses semantic accuracy versus the plain baseline. The next step should keep the reversible memory and make the final semantic fusion softer, so strong pair evidence can help without forcing a coarse all-or-nothing group decision.
