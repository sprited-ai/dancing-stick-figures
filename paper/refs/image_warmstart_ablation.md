# Archived 50-frame DiT warm-start ablation

This experiment is preserved as supporting research evidence, but is not part of the
native 120-frame protocol in the dataset paper.

## Question

How does single-frame pretraining change subsequent video learning in a small
pixel-space transformer?

## Compared runs

- Random initialization:
  `pod_results/k1_canonical_s2_scratch_mix10k`
- Image initialization:
  `pod_results/t2v50_s2_text_p4_fg2_img10_i2v20_from_t2i30k_10k`
- Initial image checkpoint:
  `runs/t2i64_color_text_p4_fg2_30k/ckpt_030000.pt`

Both video checkpoints identify their architecture as `dit_fm_t2v` and contain the
same 302 EMA tensors with 39,959,104 parameters. The model has 12 blocks that
alternate within-frame and across-frame attention. Both runs use 64×64, 50-frame
windows at 10 fps, frozen T5-small features, a flow-matching objective, and a
10,000-step video-training budget with 70% T2V, 20% I2V, and 10% T2I batches.

## Result

On 64 held-out prompts with paired noise, image initialization changes TVR from
.406 to .107 (paired 95% CI for the absolute change: [-.336, -.261]). Foreground
occupancy moves from .102 to .076 against a .060 real reference, and angular jerk
changes from .335 to .285. Centroid speed instead moves from .571 to .674 against a
.416 real reference. The experiment therefore shows a trade-off rather than a
uniform improvement.

## Interpretation limits

- This is a 50-frame, 10-fps experiment, not the dataset paper's native 120-frame,
  20-fps protocol.
- It compares one seed per condition.
- The initialized arm includes an additional 30,000 image-training steps, so it
  does not establish total-compute efficiency.
- The initialized run resumed at step 6,000 with model, EMA, and optimizer state
  restored, but random-number and data-order state restarted.
- Prompt-swap sensitivity does not establish semantic prompt adherence.

## Evidence

- `pod_results/k1_final_eval_n64/correct_colored_cache/correct_paired_comparison_n64.json`
- `pod_results/k1_final_eval_n64/correct_colored_cache/correct_scratch_metrics_n64.json`
- `pod_results/k1_final_eval_n64/correct_colored_cache/correct_warm_metrics_n64.json`
- `paper/figs/k1_warmstart_tradeoff.pdf`
- `paper/figs/running_scratch_warm_strip.png`
- `paper/figs/sweat_scratch_warm_strip.png`

