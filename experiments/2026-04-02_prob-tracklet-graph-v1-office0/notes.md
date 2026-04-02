# Probabilistic Tracklet Graph V1 Office0

- Experiment id: `exp-005`
- Name: `prob_tracklet_graph_v1_office0`
- Date: `2026-04-02`
- Branch: `exp/prob-tracklet-graph-v1-office0`
- Code commit: `6baaf80`
- Scene: `Replica/office0`
- Baseline: `exp-003`

## Goal

Add a first reversible probabilistic grouping layer on top of OVO's hard proto-object tracking so that ambiguous cross-frame relations can be remembered, grouped later, and split again by recomputing the grouping graph instead of irrevocably fusing instances.

## Code Changes

- Added [probabilistic_grouping.py](/ws/external/ovo/entities/probabilistic_grouping.py) to accumulate pairwise merge evidence, sparse ambiguous point beliefs, and dynamic semantic groups.
- Updated [ovo.py](/ws/external/ovo/entities/ovo.py) so ambiguous masks use conservative hard commits, emit probabilistic segment evidence, and classify/query with group-level CLIP descriptors.
- Updated [ovomapping.py](/ws/external/ovo/entities/ovomapping.py) to run a final probabilistic grouping pass before saving the representation.
- Updated [run_eval.py](/ws/external/run_eval.py) with environment overrides for probabilistic-grouping ablations and debug frame limits.

## Command

```bash
OVO_DATA_ROOT=/ws/data/OVO DISABLE_WANDB=true \
OVO_SAM_ENCODER=hiera_t \
OVO_PROB_GROUPING=true \
OVO_PROB_COMMIT_CONFIDENCE=0.70 \
OVO_PROB_COMMIT_MARGIN=0.20 \
OVO_PROB_EDGE_POSTERIOR_TH=0.55 \
OVO_PROB_NEG_OBS_WEIGHT=0.20 \
python run_eval.py --dataset_name Replica --experiment_name prob_tracklet_graph_v1_office0 \
  --scenes office0 --run --segment --eval
```

## Result

- Scene runtime: `559.10s`
- Mean FPS: `0.370`
- Mean SPF: `2.436s`
- Mean SAM time: `2.265s`
- Mean object time: `0.073s`
- Mean CLIP time: `0.096s`
- Mean update time: `0.005s`
- Mean VRAM: `3.661 GB`
- Max VRAM: `5.05 GB`
- Max reserved VRAM: `7.42 GB`
- mIoU: `31.5%`
- mAcc: `45.2%`
- f-mIoU: `55.5%`
- f-mAcc: `67.9%`
- Final proto-objects: `134`
- Final semantic groups: `130`
- Multi-member groups: `4`
- Ambiguous segments remembered: `252`
- Conservative hard-merge skips: `155`

## Delta vs Baseline (`exp-003`)

- Runtime: `-9.17s`
- Mean FPS: `+0.006`
- Mean SPF: `-0.036s`
- Mean SAM time: `-0.089s`
- Mean object time: `+0.052s`
- Mean VRAM: `-0.005 GB`
- Max VRAM: `-0.011 GB`
- mIoU: `+0.9pp`
- mAcc: `+2.7pp`
- f-mIoU: `-0.2pp`
- f-mAcc: `-0.8pp`

## Analysis

- This first version behaved the way we hoped structurally: it kept the low-level map conservative and remembered ambiguous relations instead of immediately collapsing them. We finished with slightly more proto-objects than the plain `hiera_t` baseline (`134` vs `131`), but the reversible graph still recovered `4` higher-level pair merges and reduced the final semantic group count to `130`.
- The quality signal is encouraging. Standard `mIoU` and `mAcc` both improved over `exp-003`, which suggests that group-level CLIP sharing can repair semantics even when we avoid irreversible map-id fusion.
- The cost landed almost entirely in object association. `t_obj` grew from about `0.021s` to `0.073s`, but total runtime stayed slightly better than the `hiera_t` baseline because SAM still dominates the full pipeline.
- The filtered metrics dipped a bit, so the current rule set seems to help broad semantic consistency more than it helps the filtered-tail slice.
- The stored evidence graph is much richer than the final grouping: `2948` pairwise evidence entries were accumulated, but only `4` edges survived the final clip+centroid gating. That is useful because it means the memory is actually preserving uncertainty, but it also tells us the final activation rule is still quite strict.

## Key Grouped Pairs

- `[5, 75]` posterior `0.9066`
- `[17, 19]` posterior `0.8943`
- `[39, 100]` posterior `0.5938`
- `[73, 74]` posterior `0.7344`

## Artifacts

- Metrics: [metrics.json](/ws/external/experiments/2026-04-02_prob-tracklet-graph-v1-office0/metrics.json)
- Run summary: [stdout.log](/ws/external/experiments/2026-04-02_prob-tracklet-graph-v1-office0/stdout.log)
- Output root: `/ws/data/OVO/output/Replica/prob_tracklet_graph_v1_office0`
- Analysis report: [report.md](/ws/data/OVO/output/Replica/prob_tracklet_graph_v1_office0/analysis/report.md)
- Prediction file: `/ws/data/OVO/output/Replica/prob_tracklet_graph_v1_office0/replica/office0.txt`

## Conclusion

This is a solid first probabilistic version: it is reversible, does not explode object count, and improves standard semantic metrics over the `hiera_t` baseline with almost no runtime or memory penalty. The next gain is likely to come from using the pair posterior more smoothly during CLIP fusion instead of only after a hard grouping threshold.
