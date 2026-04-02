# Mobile-SAM Office0

- Experiment id: `exp-002`
- Name: `mobile_sam_office0`
- Date: `2026-04-02`
- Branch: `exp/mobile-sam-baseline-office0`
- Code commit: `d8aec63`
- Baseline: `exp-001`
- Scene: `Replica/office0`

## Goal

Replace only the SAM backbone with Mobile-SAM while keeping the rest of the
pipeline and experiment settings aligned with the baseline.

## What Changed

- `sam_version` override: `mobile`
- `sam_encoder` override: `vit_t`
- All other semantic settings were kept aligned with the baseline run, including
  `points_per_side=16`, `multi_crop=True`, `segment_every=10`, CLIP settings,
  tracking thresholds, and evaluation flow.

## Command

```bash
OVO_DATA_ROOT=/ws/data/OVO DISABLE_WANDB=true \
OVO_SAM_VERSION=mobile OVO_SAM_ENCODER=vit_t \
python run_eval.py \
  --dataset_name Replica \
  --experiment_name mobile_sam_office0 \
  --scenes office0 \
  --run --segment --eval
```

## Result

- Scene runtime: `1718.14s`
- Mean FPS: `0.118`
- Mean SPF: `8.217s`
- Mean SAM time: `8.084s`
- Mean CLIP time: `0.104s`
- Mean object time: `0.028s`
- Mean update time: `0.013s`
- Mean VRAM: `3.472 GB`
- Max VRAM: `4.85 GB`
- Max reserved VRAM: `7.992 GB`
- mIoU: `28.4%`
- mAcc: `40.6%`
- f-mIoU: `54.0%`
- f-mAcc: `70.0%`

## Delta Vs Baseline

- Runtime: `+1007.60s`
- Mean FPS: `-0.173`
- Mean SPF: `+5.076s`
- Mean SAM time: `+5.067s`
- Mean VRAM: `-0.953 GB`
- Max VRAM: `-1.05 GB`
- Max reserved VRAM: `-0.338 GB`
- mIoU: `-7.2pp`
- mAcc: `-6.8pp`
- f-mIoU: `-0.7pp`
- f-mAcc: `+1.4pp`

## Analysis

- In this environment and with the current automatic-mask setup, Mobile-SAM was
  substantially slower than the SAM2.1 baseline.
- The main cost still sits in segmentation. Mean `t_sam` rose from about `3.0s`
  to `8.1s`, which largely explains the drop in FPS.
- Memory use improved meaningfully. Mean VRAM dropped by about `0.95 GB`, and
  max VRAM dropped by about `1.05 GB`.
- Accuracy also dropped on this scene. The largest visible regressions were in
  classes such as `table`, `tablet`, and several sparse categories that remained
  at or near zero IoU.
- This means the current Mobile-SAM swap is not a win on `office0` if the goal
  is wall-clock speed or scene mIoU. It may still be useful when GPU memory is
  the primary constraint.

## Artifacts

- Stdout log: `/ws/external/experiments/2026-04-02_mobile-sam-office0/stdout.log`
- Output root: `/ws/data/OVO/output/Replica/mobile_sam_office0`
- Analysis report: `/ws/data/OVO/output/Replica/mobile_sam_office0/analysis/report.md`
- Prediction file: `/ws/data/OVO/output/Replica/mobile_sam_office0/replica/office0.txt`

## Next Step

- If we want a faster low-memory variant, compare Mobile-SAM against a smaller
  SAM2.1 encoder instead of the large baseline encoder.
