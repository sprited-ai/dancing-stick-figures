# If v9.x enters the dataset paper

## Recommendation

Do not present v9.x as another beginner baseline. Present **v9.0 varied-caption, 100k** as one advanced-reference
case study showing what the released rig labels enable after the direct-pixel curriculum. Keep the architectural
ablation history (v8, v9.1, v9.3 and the coupling sweep) in the companion model paper.

This preserves the dataset paper's hierarchy:

1. dataset and deterministic renderer;
2. diagnostic measurements;
3. reproducible direct-pixel UNet/DiT baselines;
4. one advanced use of the labels: jointly generate pixels and a rig.

## Smallest defensible insertion

Add a half-page subsection titled **Advanced reference: rig co-generation** after the direct-pixel baselines.

- One compact diagram: prompt → Video VAE / full-ST block-AR DiT → RGBA video + 27-joint rig.
- One qualitative row: decoded video, emitted-rig overlay, and the corresponding released reference.
- One table with v8 and v9.0 at the same 100k budget, reporting both seeds for v9.0:
  - TVR and LIE;
  - centroid speed, motion fraction, and angular jerk;
  - teacher-forced and free-running next-block divergence;
  - SRE(pixel) ↔ emitted-rig self-consistency.
- One paragraph explaining that v9.0 is an advanced released reference, not the Colab route and not evidence that
  this architecture is generally best for video.

The central result would be narrowly stated: adding co-generated rig tokens improves continuation prediction over the
matched pixel-only v8 model at 100k, and the improvement replicates across two v9.0 seeds. The current evidence is:

- v8 100k: teacher-forced/free-running divergence `.3139/.4584`;
- v9.0 seed 0: `.2870/.4399`, TVR `.126`, centroid speed `.275`;
- v9.0 seed 1: `.2917/.4387`, TVR `.129`, centroid speed `.253`;
- real centroid speed `.314` and motion fraction `.371` under the recorded evaluation.

## What not to include

- Do not call v9.3 the flagship. It failed its pre-declared motion and divergence criteria despite excellent
  structure and rig-pixel binding.
- Do not narrate the complete v9.1/v9.3 sweep in this paper. That turns the dataset report into an architecture paper.
- Do not compare v9.x numerically against the 50-frame pixel DiT in one ranking table. They use different
  representations, training budgets, temporal factorisations, and evaluation contracts.
- Do not call the emitted rig an oracle. Its agreement with pixels is measured through the independently trained SRE;
  prompt correctness remains a separate question.

## Release gate

Include the subsection only if the v9.0 checkpoint, Video VAE, latent statistics, inference script, SRE checkpoint,
and exact evaluation manifests can be released together. Otherwise mention rig co-generation in future work and keep
the complete result for the companion architecture paper.
