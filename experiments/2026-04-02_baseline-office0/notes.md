# Baseline Office0

- Experiment id: `exp-001`
- Name: `baseline_office0`
- Date: `2026-04-02`
- Branch: `baseline/2026-04-02-current-worktree`
- Commit: `3b0474e`
- Scene: `Replica/office0`

## Goal

Capture the first reproducible baseline run for the current worktree on a single
reference scene.

## Command

```bash
OVO_DATA_ROOT=/ws/data/OVO DISABLE_WANDB=true python run_eval.py \
  --dataset_name Replica \
  --experiment_name baseline_office0 \
  --scenes office0 \
  --run --segment --eval
```

## Result

- Scene runtime: `710.54s`
- Mean FPS: `0.291`
- Mean SPF: `3.141s`
- Mean SAM time: `3.017s`
- Mean CLIP time: `0.097s`
- Mean object time: `0.023s`
- Mean update time: `0.008s`
- Mean VRAM: `4.425 GB`
- Max VRAM: `5.9 GB`
- Max reserved VRAM: `8.33 GB`
- mIoU: `35.6%`
- mAcc: `47.4%`
- f-mIoU: `54.7%`
- f-mAcc: `68.6%`

## Analysis

- The dominant runtime bottleneck is SAM. Based on the logger summary, SAM takes
  about `95.9%` of the tracked semantic-module time.
- CLIP is present on almost every keyframe but is not the primary cost in this
  baseline: `0.097s` mean versus `3.017s` mean for SAM.
- The run confirms the earlier code reading: delayed semantic processing is
  pipelined in time, but total CLIP work is still accumulated rather than truly
  skipped.
- Accuracy is reasonable for a single-scene baseline, with strong object classes
  such as `chair`, `sofa`, `rug`, `desk-organizer`, and weak classes concentrated
  in small or sparse categories.

## Artifacts

- Stdout log: `/ws/external/experiments/2026-04-02_baseline-office0/stdout.log`
- Output root: `/ws/data/OVO/output/Replica/baseline_office0`
- Analysis report: `/ws/data/OVO/output/Replica/baseline_office0/analysis/report.md`
- Prediction file: `/ws/data/OVO/output/Replica/baseline_office0/replica/office0.txt`

## Next Step

- Use this run as the comparison point for queue, CLIP, or top-view ablations.
