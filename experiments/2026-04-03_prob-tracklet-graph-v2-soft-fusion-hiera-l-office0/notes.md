# Probabilistic Tracklet Graph V2 Soft Fusion Hiera-L Office0

- Experiment id: `exp-007`
- Name: `prob_tracklet_graph_v2_soft_fusion_hiera_l_office0`
- Date: `2026-04-03`
- Branch: `exp/prob-tracklet-graph-v2-soft-fusion-hiera-l-office0`
- Code commit: `9674acb`
- Scene: `Replica/office0`
- Baseline: `exp-001`
- Primary comparison: `exp-006`

## Goal

Analyze why `exp-006` lost quality against the plain `hiera_l` baseline, then test the most likely fix: keep the reversible grouping memory but stop replacing each member object's CLIP descriptor with a shared group descriptor. Instead, use a self-preserving posterior-weighted soft fusion so every object keeps its own identity while still borrowing signal from high-confidence neighbors.

## Root-Cause Hypothesis

- `exp-006` preserved the same grouping structure as intended, but every grouped object consumed the same group-level CLIP descriptor at query time.
- With a strong `hiera_l` backbone, the original per-object descriptor is already fairly good. Replacing it with a shared group descriptor can over-smooth semantics and wash out member-specific class evidence.
- If that hypothesis is right, then keeping the same graph while softening only descriptor fusion should recover semantic quality without changing grouping statistics much.

## Code Changes

- Updated [probabilistic_grouping.py](/ws/external/ovo/entities/probabilistic_grouping.py) so grouped objects use personalized CLIP descriptors built from:
  - a strong self weight (`soft_clip_self_weight=3.0`)
  - posterior-weighted neighbor contributions inside the same semantic group
- The reversible grouping graph, pair evidence, and final group activation rule were left unchanged.

## Command

```bash
OVO_DATA_ROOT=/ws/data/OVO DISABLE_WANDB=true \
OVO_SAM_ENCODER=hiera_l \
OVO_PROB_GROUPING=true \
OVO_PROB_COMMIT_CONFIDENCE=0.70 \
OVO_PROB_COMMIT_MARGIN=0.20 \
OVO_PROB_EDGE_POSTERIOR_TH=0.55 \
OVO_PROB_NEG_OBS_WEIGHT=0.20 \
python run_eval.py --dataset_name Replica --experiment_name prob_tracklet_graph_v2_soft_fusion_hiera_l_office0 \
  --scenes office0 --run --segment --eval
```

## Result

- Scene runtime: `242.09s`
- Mean FPS: `0.889`
- Mean SPF: `0.891s`
- Mean SAM time: `0.745s`
- Mean object time: `0.051s`
- Mean CLIP time: `0.097s`
- Mean update time: `0.003s`
- Mean VRAM: `4.419 GB`
- Max VRAM: `5.85 GB`
- Max reserved VRAM: `8.33 GB`
- mIoU: `35.4%`
- mAcc: `47.6%`
- f-mIoU: `54.4%`
- f-mAcc: `68.0%`
- Final proto-objects: `141`
- Final semantic groups: `134`
- Multi-member groups: `6`
- Stored pairwise evidence edges: `3370`
- Active final grouping edges: `8`

## Delta vs `exp-006`

- mIoU: `+1.3pp`
- mAcc: `+1.7pp`
- f-mIoU: `+1.1pp`
- f-mAcc: `+1.5pp`
- Group statistics: unchanged (`141 -> 134`, `6` multi-member groups, `8` active pair edges)
- Runtime and SAM time improved sharply, but this rerun appears to have reused cached SAM artifacts, so that speedup should not be credited to the soft-fusion code change itself.

## Delta vs Baseline (`exp-001`)

- mIoU: `-0.2pp`
- mAcc: `+0.2pp`
- f-mIoU: `-0.3pp`
- f-mAcc: `-0.6pp`
- VRAM: effectively unchanged
- Runtime: not directly comparable because this rerun likely benefited from cached mask artifacts.

## Analysis

- The hypothesis was mostly right. We changed only how grouped descriptors are consumed, and semantic quality recovered substantially while the grouping graph stayed exactly the same.
- That is strong evidence that the main regression in `exp-006` came from descriptor over-smoothing, not from the reversible graph itself.
- This version now lands very close to the plain `hiera_l` baseline. `mIoU` is only `0.2pp` lower and `mAcc` is actually `0.2pp` higher.
- Filtered metrics still trail baseline slightly, so the next issue is probably not grouping itself but how much confidence we should give grouped neighbors on harder long-tail classes.
- The cleanest next move is to tune the soft-fusion strength rather than redesign the graph again. The graph memory is doing its job; now we need to calibrate how strongly it influences the final descriptor.

## Conclusion

This rerun supports a clear conclusion: the reversible probabilistic memory was not the main problem. The main problem was replacing each object's CLIP with a shared group CLIP. Switching to self-preserving posterior-weighted soft fusion almost closes the gap to the `hiera_l` baseline while keeping the same reversible grouping state.
