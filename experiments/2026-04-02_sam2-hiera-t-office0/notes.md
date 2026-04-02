# SAM2 Hiera-T Office0

- Experiment id: `exp-003`
- Name: `sam2_hiera_t_office0`
- Date: `2026-04-02`
- Branch: `exp/sam2-hiera-t-office0`
- Code commit: `79212bf`
- Scene: `Replica/office0`
- Baseline: `exp-001`

## Goal

Keep the baseline pipeline intact and replace only the baseline SAM2.1 `hiera_l` encoder with the lighter SAM2.1 `hiera_t` encoder.

## Command

```bash
OVO_DATA_ROOT=/ws/data/OVO DISABLE_WANDB=true \
OVO_SAM_ENCODER=hiera_t \
python run_eval.py --dataset_name Replica --experiment_name sam2_hiera_t_office0 \
  --scenes office0 --run --segment --eval
```

## Result

- Scene runtime: `568.27s`
- Mean FPS: `0.364`
- Mean SPF: `2.471s`
- Mean SAM time: `2.355s`
- Mean CLIP time: `0.096s`
- Mean object time: `0.021s`
- Mean update time: `0.005s`
- Mean VRAM: `3.666 GB`
- Max VRAM: `5.06 GB`
- Max reserved VRAM: `7.418 GB`
- mIoU: `30.6%`
- mAcc: `42.5%`
- f-mIoU: `55.7%`
- f-mAcc: `68.7%`

## Delta vs Baseline

- Runtime: `-142.27s`
- Mean FPS: `+0.073`
- Mean SPF: `-0.670s`
- Mean SAM time: `-0.662s`
- Mean VRAM: `-0.759 GB`
- Max VRAM: `-0.84 GB`
- Max reserved VRAM: `-0.912 GB`
- mIoU: `-5.0pp`
- mAcc: `-4.9pp`
- f-mIoU: `+1.0pp`
- f-mAcc: `+0.1pp`

## Analysis

- This is a cleaner light-weight SAM ablation than FastSAM because it stays within the same SAM2.1 family and changes only the encoder size.
- `hiera_t` delivered a clear efficiency win: about `20.0%` faster wall-clock time than baseline and lower GPU memory across the run.
- The tracked semantic bottleneck remained SAM at about `95.1%` of semantic-module time, but the absolute SAM cost dropped from `3.017s` to `2.355s` per step.
- Standard `mIoU` and `mAcc` dropped relative to baseline, but filtered metrics held up better: `f-mIoU` improved slightly from `54.7%` to `55.7%`, and `f-mAcc` was effectively unchanged.
- Compared with FastSAM, this looks like the safer light-model tradeoff so far: smaller speedup than the most aggressive replacement is still possible, but the quality drop is noticeably less severe.

## Artifacts

- Stdout log: `/ws/external/experiments/2026-04-02_sam2-hiera-t-office0/stdout.log`
- Output root: `/ws/data/OVO/output/Replica/sam2_hiera_t_office0`
- Analysis report: `/ws/data/OVO/output/Replica/sam2_hiera_t_office0/analysis/report.md`
- Prediction file: `/ws/data/OVO/output/Replica/sam2_hiera_t_office0/replica/office0.txt`

## Conclusion

If we want a lighter SAM while keeping the baseline architecture style intact, `SAM2.1 hiera_t` is a much stronger candidate than FastSAM on `office0`.
