# Dancing Stick Figures — release TODO

The paper's native video unit is **120 frames at 20 fps (six seconds)**. The older 50-frame DiT warm-start study is
preserved in `paper/refs/image_warmstart_ablation.md`, but is outside the dataset paper's evidence chain. Claims remain
TODO until the corresponding artifact and measurement exist.

All new runs follow `paper/TRAINING_PROTOCOL.md`: fixed manifests and noise, immutable milestone checkpoints,
resumable latest state, measured wall time/VRAM, and off-pod artifact verification.

## Paper narrative and native evaluation

- [x] Frame the contribution as a dataset, deterministic rendering pipeline, and evaluation suite—not a new SOTA model.
- [x] Open the Introduction with the inaccessible data/compute/feedback/evaluation problem.
- [x] Keep model history subordinate to the dataset contribution; explain only the chosen reference architectures.
- [x] Evaluate controlled temporal failures and FVD on complete 120-frame clips.
- [x] Quantify FVD's reversal blind spot with paired subset trials.
- [x] Explain pixel-space training and defer Video-VAE curriculum to a separate optional lesson.
- [x] Add the public-motion reconstruction route and instructor-specific renderer variations.
- [x] Verify reconstruction over the complete released corpus in both tiers. Both 128² and 64² checks passed over
      all 514,800 frames with no missing rows or metadata-label mismatches.

## v0.2 factorised-UNet Colab

- [x] Use one fixed factorised 3D UNet backbone for T=1 image training and T>1 video training.
- [x] Expose only 32² sanity and 64² reference resolutions; readers may inspect/edit the source without an option maze.
- [x] Make complete-prompt T5-small conditioning the default for both stages.
- [x] Add a typed-prompt, five-second rollout and label the older public checkpoint as unconditional.
- [x] Complete an uninterrupted 32² engineering run on RTX 4090 (2k image + 1.2k video; 500 s total).
- [x] Finish the lower-cost 64² reference run (30k image + 10k video), preserve checkpoints, samples, logs, and
      measured resources.
- [x] Finish the native-cadence 64² video run and its complete 120-frame, three-seed prompt/reference evaluation.
- [x] Run both exact 64² training stages and the 120-frame rollout on a Colab-class T4; record update speed,
      validation, and peak memory separately from the complete RTX execution contract.
- [ ] When Colab quota permits, repeat the final typed-prompt/scoring cells in the same T4 provider session. This is a
      provenance improvement, not a missing fit/training/rollout measurement.
- [x] Add a fixed-noise prompt suite. Do not call the model prompt-following until blinded human or motion-grounded
      adherence evidence exists; prompt swaps establish sensitivity only.

## Release QA

- [x] Keep rendered TODOs out of the reader PDF while preserving them here and as source comments.
- [x] Re-render and visually inspect every final PDF page after the 64²/T4 evidence is incorporated.
- [x] Execute the complete notebook contract from a clean RTX 4090 environment; keep outputs cleared in the released
      notebook. The same source separately completed both training stages and the 120-frame rollout on hosted T4.
- [x] Run the complete test suite and verify video, JSON, notebook, arXiv-source, and PDF artifacts off the training pod.
- [x] Produce and inspect the 12-scene Korean narrated MP4 walkthrough with page/paragraph highlights.

## Future work, outside this report

- Optional Video-VAE/latent-space lesson with codec reconstruction scored before generator training.
- Learned rig estimator with calibrated joint confidence and analysis-by-synthesis re-rendering.
- A multi-seed, compute-matched DiT warm-start study under the native 120-frame tier, if pursued as a companion study.
- Validated prompt-adherence evaluation, hidden evaluation container, and hosted submissions.
