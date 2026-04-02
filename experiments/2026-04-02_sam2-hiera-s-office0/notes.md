# SAM2 Hiera-S Office0

- Experiment id: `exp-004`
- Name: `sam2_hiera_s_office0`
- Date: `2026-04-02`
- Branch: `exp/sam2-hiera-s-office0`
- Code commit: `bdfe6f1`
- Scene: `Replica/office0`
- Baseline: `exp-001`

## Goal

Keep the baseline pipeline intact and replace only the baseline SAM2.1 `hiera_l` encoder with the lighter SAM2.1 `hiera_s` encoder.

## Command

```bash
OVO_DATA_ROOT=/ws/data/OVO DISABLE_WANDB=true \
OVO_SAM_ENCODER=hiera_s \
python run_eval.py --dataset_name Replica --experiment_name sam2_hiera_s_office0 \
  --scenes office0 --run --segment --eval
```

## Result

- Scene runtime: `606.74s`
- Mean FPS: `0.340`
- Mean SPF: `2.664s`
- Mean SAM time: `2.545s`
- Mean CLIP time: `0.097s`
- Mean object time: `0.021s`
- Mean update time: `0.006s`
- Mean VRAM: `3.699 GB`
- Max VRAM: `5.10 GB`
- Max reserved VRAM: `7.575 GB`
- mIoU: `29.5%`
- mAcc: `40.3%`
- f-mIoU: `52.7%`
- f-mAcc: `65.0%`

## Delta vs Baseline

- Runtime: `-103.80s`
- Mean FPS: `+0.049`
- Mean SPF: `-0.477s`
- Mean SAM time: `-0.472s`
- Mean VRAM: `-0.726 GB`
- Max VRAM: `-0.80 GB`
- Max reserved VRAM: `-0.755 GB`
- mIoU: `-6.1pp`
- mAcc: `-7.1pp`
- f-mIoU: `-2.0pp`
- f-mAcc: `-3.6pp`

## Comparison

| Encoder | Runtime (s) | Mean FPS | Mean VRAM (GB) | mIoU | mAcc | f-mIoU | f-mAcc |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `hiera_l` | 710.54 | 0.291 | 4.425 | 35.6 | 47.4 | 54.7 | 68.6 |
| `hiera_s` | 606.74 | 0.340 | 3.699 | 29.5 | 40.3 | 52.7 | 65.0 |
| `hiera_t` | 568.27 | 0.364 | 3.666 | 30.6 | 42.5 | 55.7 | 68.7 |

## Analysis

- `hiera_s` is faster and lighter than baseline, but on `office0` it does not land between `large` and `tiny` in a favorable way.
- Compared with `hiera_t`, `hiera_s` was both slower and less accurate on this scene.
- The expected "middle ground" did not show up here: `hiera_t` dominated `hiera_s` on runtime, VRAM, `mIoU`, `mAcc`, `f-mIoU`, and `f-mAcc`.
- Within the SAM2.1 family on this scene, `hiera_t` currently looks like the strongest tradeoff option.

## Artifacts

- Stdout log: `/ws/external/experiments/2026-04-02_sam2-hiera-s-office0/stdout.log`
- Output root: `/ws/data/OVO/output/Replica/sam2_hiera_s_office0`
- Analysis report: `/ws/data/OVO/output/Replica/sam2_hiera_s_office0/analysis/report.md`
- Prediction file: `/ws/data/OVO/output/Replica/sam2_hiera_s_office0/replica/office0.txt`
