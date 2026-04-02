# FastSAM Office0

- Experiment id: `exp-002`
- Name: `fast_sam_office0`
- Date: `2026-04-02`
- Branch: `exp/fast-sam-baseline-office0`
- Code commit: `88243fb`
- Scene: `Replica/office0`
- Baseline: `exp-001`

## Goal

Replace only the SAM backend with FastSAM and compare runtime, memory, and semantic accuracy against the office0 baseline.

## Command

```bash
OVO_DATA_ROOT=/ws/data/OVO DISABLE_WANDB=true \
OVO_SAM_VERSION=fast OVO_SAM_ENCODER=FastSAM \
OVO_SAM_CKPT_PATH=/ws/data/OVO/input/sam_ckpts \
python run_eval.py --dataset_name Replica --experiment_name fast_sam_office0 \
  --scenes office0 --run --segment --eval
```

## Result

- Scene runtime: `640.45s`
- Mean FPS: `0.323`
- Mean SPF: `2.811s`
- Mean SAM time: `2.693s`
- Mean CLIP time: `0.103s`
- Mean object time: `0.021s`
- Mean update time: `0.005s`
- Mean VRAM: `3.813 GB`
- Max VRAM: `5.08 GB`
- Max reserved VRAM: `8.20 GB`
- mIoU: `27.7%`
- mAcc: `43.5%`
- f-mIoU: `48.1%`
- f-mAcc: `61.4%`

## Delta vs Baseline

- Runtime: `-70.09s`
- Mean FPS: `+0.032`
- Mean SPF: `-0.330s`
- Mean SAM time: `-0.324s`
- Mean VRAM: `-0.612 GB`
- Max VRAM: `-0.82 GB`
- mIoU: `-7.9pp`
- mAcc: `-3.9pp`
- f-mIoU: `-6.6pp`
- f-mAcc: `-7.2pp`

## Analysis

- FastSAM improved efficiency on this scene: it finished about `9.9%` faster than baseline and used less GPU memory throughout the run.
- Accuracy dropped substantially. The largest quality hit was on tail classes, where tail IoU fell from `34.4%` to `20.9%`.
- The dominant runtime module was still SAM-like segmentation at about `95.4%` of tracked semantic time, even though raw FastSAM inference logs were only around `25ms`. That suggests the OVO-side mask handling and integration overhead still dominates much of the semantic step.
- CLIP cost stayed effectively unchanged, which matches the earlier code reading: this ablation changes the mask producer, not the downstream per-keyframe CLIP/update pattern.
- FastSAM required a runtime compatibility adjustment because its vendored ultralytics stack imports `pkg_resources`; the environment was pinned to `setuptools==80.9.0` before the run.

## Artifacts

- Stdout log: `/ws/external/experiments/2026-04-02_fast-sam-office0/stdout.log`
- Output root: `/ws/data/OVO/output/Replica/fast_sam_office0`
- Analysis report: `/ws/data/OVO/output/Replica/fast_sam_office0/analysis/report.md`
- Prediction file: `/ws/data/OVO/output/Replica/fast_sam_office0/replica/office0.txt`

## Conclusion

FastSAM is a plausible speed-and-memory tradeoff option for OVO on `office0`, but in its current integration it is not a drop-in replacement for baseline SAM if semantic quality is the priority.
