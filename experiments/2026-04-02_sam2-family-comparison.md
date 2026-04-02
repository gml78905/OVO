# SAM2 Family Comparison on Replica Office0

| Encoder | Runtime (s) | Mean FPS | Mean SAM Time (s) | Mean VRAM (GB) | Max Reserved VRAM (GB) | mIoU | mAcc | f-mIoU | f-mAcc |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `hiera_l` | 710.54 | 0.291 | 3.017 | 4.425 | 8.330 | 35.6 | 47.4 | 54.7 | 68.6 |
| `hiera_s` | 606.74 | 0.340 | 2.545 | 3.699 | 7.575 | 29.5 | 40.3 | 52.7 | 65.0 |
| `hiera_t` | 568.27 | 0.364 | 2.355 | 3.666 | 7.418 | 30.6 | 42.5 | 55.7 | 68.7 |

## Takeaways

- `hiera_t` is the best overall tradeoff on `office0` among the three tested SAM2.1 encoders.
- `hiera_t` is `20.0%` faster than `hiera_l`, uses `0.759 GB` less mean VRAM, and still preserves filtered metrics best.
- `hiera_s` improves efficiency over `hiera_l`, but on this scene it is dominated by `hiera_t` on both efficiency and accuracy.
- If the priority is pure quality on this scene, `hiera_l` still wins on raw `mIoU` and `mAcc`.
- If the priority is speed/memory with minimal downside, `hiera_t` is the current recommendation.
