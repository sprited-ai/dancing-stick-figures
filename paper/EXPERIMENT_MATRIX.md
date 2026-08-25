# Dancing Stick Figures — experiment matrix

This ledger separates experiments that support the **dataset report** from the later **model-research track**. A row is
one executed protocol or matched comparison; milestone checkpoints and sampler sweeps are listed inside the row instead
of being mistaken for independent experiments. “Paper” means evidence used in the current dataset report. “Released”
means an artifact already exists locally or in the public release; it does not mean that every historical checkpoint
belongs in the beginner Colab.

## A. Dataset, reconstruction, and evaluation

| ID | Experiment | Unit / setting | Question | Result or current state | Evidence | Paper role |
|---|---|---|---|---|---|---|
| D0 | Released-data audit | 4,290 clips; 514,800 frames; three views | Do rows, frames, labels, splits, and source motions obey the release contract? | Release manifests and dataset tables built; training-data forward-motion audit retained | data/; output/training_data_audit_runs_forward/manifest.json | Dataset specification |
| D1 | Public-motion reconstruction smoke | 1 source motion, 3 views, 360 frames | Can the public motion table regenerate the released observations without ARDY? | Semantic fields exact; colour/segmentation/visibility disagreement .0035/.0012/.0007 after the published downsampling path | output/runpod_v02/rebuild_verification.json | Reproducibility evidence |
| D2 | Public-motion full reconstruction | 1,430 motions, 4,290 clips, 514,800 frames in both released tiers | Does D1 hold over the complete public corpus? | **Passed in both tiers:** no missing rows or metadata-label mismatches. Colour/segmentation/visibility agreement is 99.88%/99.92%/99.50% at 128² and 99.75%/99.92%/99.50% at 64². | paper/results/rebuild_full_frames_v2.json; paper/results/rebuild_full_mini_v2.json; scripts/verify_rebuild.py | Required release gate passed |
| E0 | Real-reference split | two source-motion-disjoint sets, n=128, 120 frames, 20 fps | What variation appears between two valid samples from the dataset? | Real–real FVD 90.6; reference motion and structure statistics frozen | output/runpod_v02/native120_manifest.json; output/runpod_v02/native120_baselines.json | Native reference |
| E1 | Controlled temporal failures | freeze, shuffle, reverse, loop-8, train replay; n=128, 120 frames | Which visible failures do the structural/motion signals and FVD detect? | Freeze zeros motion; shuffle spikes acceleration/jerk; reverse preserves time-symmetric signals; FVD catches freeze/shuffle/loop but not reversal | paper/results/degenerate_120f_n128.json | Main diagnostic validation |
| E2 | FVD reversal uncertainty | 30 paired trials, n=64, complete 120-frame clips | Is the reversal result a one-sample accident? | Delta-FVD = −0.44 ± 2.16; reversal increases FVD in 11/30 trials | paper/results/fvd_120f_reverse_uncertainty.json | Main diagnostic validation |
| E3 | Legacy 50-frame corruption protocol | n=128, 50 frames, stride 2 | Earlier protocol used while developing the evaluator | Retained for traceability; superseded by E0–E2 for dataset claims | paper/results/degenerate_50f_n128.json; paper/results/fvd_50f_n128.json | Historical only |

## B. Released pixel-space baselines

| ID | Model / comparison | Training setting | Question | Result or current state | Evidence | Paper role |
|---|---|---|---|---|---|---|
| I0 | 64² image UNet (ia64) | 30k | Can a compact pixel UNet learn the rendered frame domain? | Completed and scored | paper/results/score_ia64.json | Released baseline |
| I1 | 64² image DiT (ib64) | 30k | Does a small pixel DiT provide a second image reference? | Completed and scored | paper/results/score_ib64.json | Released baseline |
| I2 | 64² long image UNet (ia64L) | 100k, min-SNR | Where does longer image training place the structural floor? | Completed; near real-reference topology | paper/results/score_ia64L.json | Learnability reference |
| I3 | 64² long image DiT (ib64L) | 50k, p2 | Same question for the DiT image reference | Completed and scored | paper/results/score_ib64L.json | Learnability reference |
| I4 | 128² image UNet / DiT (ia128, ib128) | 20k / 40k | Does the same frame task remain learnable at the full released resolution? | Both completed and scored | paper/results/score_ia128.json; paper/results/score_ib128.json | Released higher-resolution reference |
| I5 | Category-conditioned image UNet / DiT (ic64, id64) | 64², 30k targets | Is coarse category conditioning sufficient for a useful teaching interface? | Executed as an early conditioning route; complete prompts replace it in v0.2 | historical RunPod collection noted in STATUS.md | Historical only |
| V0 | 64² video UNet scratch (a64) | 8-frame windows, 85k | Can the released UNet learn short videos from scratch? | Completed; final evaluation available | paper/results/eval_a64_final.json | Released baseline |
| V1 | 64² video UNet image-init (a64i) | 8-frame windows, 61k | Does image initialization improve the short-video route? | Completed; final evaluation available | paper/results/eval_a64i_final.json | Released baseline |
| V2 | 64² video DiT scratch / image-init (b64, b64i) | matched interim short-video route | Does the initialization question reproduce in the earlier DiT recipe? | Executed during v0.1; not promoted to the student route | historical collector and checkpoints noted in STATUS.md | Historical only |
| V3 | Autoregressive UNet rollout | released unet_ar64.pt, 5-second rollout | Can the short-window UNet be rolled forward beyond one training window? | Released qualitative and evaluator route | scripts/rollout.py; baseline model repository | Released demonstration |

## C. Current factorised-UNet Colab v0.2

| ID | Experiment | Training setting | Question | Result or current state | Evidence | Release decision |
|---|---|---|---|---|---|---|
| C0 | 32² complete-prompt engineering run | image 2k → video 1.2k; factorised 3D UNet; frozen T5-small | Does the exact image→video code path run end to end and accept a typed prompt? | 500 s total on RTX 4090; 9.2/8.2 GB peaks; five-second rollout produced | output/video/v02_full_prompt_32px_2k_1p2k.mp4 | Code-path evidence, not T4 timing |
| C1 | 64² complete-prompt image reference | image 30k | Does the chosen reference backbone reproduce sharp coloured limbs before temporal training? | **Completed** in 4,289 s on RTX 4090; 30k sample preserves clean coloured limbs across the five prompt groups; final fixed validation loss .0146; 6.9 GB training peak | output/runpod_v02/full64_final/ | v0.2 reference checkpoint |
| C2 | 64² complete-prompt video reference | C1 warm-start → video 10k; 8-frame context + 8-frame continuation; stride 2 | Can the same backbone learn and roll out multi-second video? | **Completed**; the 5k→10k resumed segment took 2,457 s; final first/continuation validation .0108/.0093. At 60 frames, TVR .157 vs .127 real, but centroid speed is 1.53× and angular jerk 3.21× real; FVD 407.3 vs real–real 178.0. | output/runpod_v02/full64_text_video_10k/ | v0.2 lower-cost reference and diagnostic result |
| C3 | Fixed-noise prompt sensitivity suite | 5 training-distribution prompts + 3 held-out sport prompts; prompt/noise swaps | Does changing text change the output? | **Completed** at C2 step 10k. Fixed-noise varied-prompt pairwise L1 .0766 vs fixed-prompt varied-noise .0975, ratio .786. This establishes sensitivity, not semantic adherence; the strips show several prompts remain weak. | output/runpod_v02/full64_prompt_suite_60f/ | Required evidence boundary |
| C4 | Colab-class T4 resource run | released default 64² route | Do both stages and full-clip sampling fit and run on the intended T4 GPU? | **Complete through the 120-frame rollout.** Image 2k: .76 s/update, 6.9 GB, val .0266. Video 1.2k: 1.12 s/update, 11.3 GB, val .0222/.0217. The provider ended the accumulated backend lifetime before the final typed-prompt/scoring cells; a fresh GPU was quota-blocked. | `paper/results/colab_v02_t4.json`; `notebooks/dancing_stick_figures_colab_v0_2.ipynb` | Supports T4 fit, training speed, and rollout claims; does not claim a final single-session completion manifest |
| C4b | Release-contract completion | same 64² source on RTX 4090; image 2k → video 1.2k → typed prompt → scoring | Do all released cells and artifacts complete after the portability fixes? | **Complete in 944 s.** Image/video cells 296/364 s; peaks 6.9/11.3 GB; typed-prompt GIF and image/reference score JSONs saved. | `paper/results/notebook_v02_4090_completion.json`; `output/runpod_v02/colab_contract_4090/` | End-to-end code/artifact evidence, kept separate from T4 timing |
| C5 | Native 20-fps complete-prompt video reference | C1 warm-start → video 10k; stride 1; 120-frame rollout; n=64 × 3 sampling seeds | Does the released route produce six-second motion, and what values should a reproduction match? | **Complete.** UNet: TVR .203, LIE .145, CPE .042, centroid speed .479, acceleration .642, motion fraction .604, angular jerk .189, FVD 510.1 ± 31.4. Reference: .105, .090, .036, .296, .290, .336, .053, and real–real FVD 129.9. | `paper/results/unet_native120_v02.json`; full run at `output/runpod_v02/full64_text_video_native20fps_10k/eval/010000.json` | Main reference-model evidence |
| C6 | Native qualitative comparison | C5 checkpoint; walking, running, and sitting; fixed noise seed 1234; 120 frames | What does the reference model visibly learn, and where does prompt separation fail? | **Complete.** Sitting produces a visible descent, while walking and running remain similar. Prompt-swap L1 is .0447 versus .0950 for noise swaps (ratio .471), so this is a sensitivity diagnostic rather than prompt-adherence evidence. | `paper/results/unet_prompt_suite_legible_v02.json`; `output/video/unet_dataset_prompt_comparison.mp4` | Figure 4 and released qualitative evidence |

## D. Archived 50-frame DiT warm-start research

This section preserves an exploratory study for traceability. It does not provide a quantitative claim in the dataset
paper; see `paper/refs/image_warmstart_ablation.md` for the protocol and limitations.

| ID | Experiment | Training setting | Question | Result or current state | Evidence | Paper role |
|---|---|---|---|---|---|---|
| K1-M1 | Single-frame text-conditioned DiT | 64², 30k | Can the K1 backbone learn the spatial domain before video training? | Completed; checkpoint used only to initialize the treatment | K1 manifests and released comparison artifacts | Experimental prerequisite |
| K1-M2 | 50-frame K1 from scratch | 40.0M parameters, 10k, 50 frames at 10 fps | What does video training learn without image initialization? | Completed | output/comparisons/k1_matched_500_to_10000/manifest.json | Archived control |
| K1-M3 | 50-frame K1 image warm-start | same model and video protocol; M1 initialization | What changes under image initialization? | TVR .406→.107; validation .0159→.0118; topology improves while several translation statistics overshoot; the resumed run has an RNG/data-order caveat | K1 score JSON and paper/figs/k1_warmstart_tradeoff.pdf | Archived treatment |
| K1-P | Prompt/noise swap suite | fixed-noise varied-prompt and fixed-prompt varied-noise | Is the K1 output sensitive to text? | Ratio .360→.550 after warm-start; does not establish semantic correctness | output/inference/m3_diverse_prompts_10k/manifest.json | Supporting limitation-aware evidence |
| K1-S | Sampler/jitter ablation | steps 50/100, CFG 1/2/3 | Are visible jitter changes caused by sampler settings alone? | Executed and summarized | output/inference/m3_jitter_ablation/sampler_metric_summary.json | Supporting diagnostic |

## E. Historical latent-video and long-horizon model research

These experiments belong to the companion architecture investigation, not to the dataset paper or beginner Colab.

| ID | Experiment | Variants | Question | Result or current state | Evidence |
|---|---|---|---|---|---|
| L0 | Patch-space feasibility | patch reconstruction and metric audit | Can thin coloured limbs survive a compressed patch representation? | Executed | output/patch_space/patch-space-metrics.json |
| L1 | Video-VAE temporal receptive-field audit | candidate temporal blocks/windows | Does the codec use temporal context causally and consistently? | Executed | output/codec_temporal_receptive_field.json |
| L2 | Video-VAE compression selection | f8t2 vs f4t4 | Which spatial/temporal compression preserves thin edges and long clips? | f4t4d32 retained in the first gate | output/codec_selection_f8t2_vs_f4t4/comparison.json |
| L3 | Lower-channel codec selection | f8t4d16 vs corrected f8t2d32 | Can stronger compression reduce generator cost without failing reconstruction gates? | The declared reconstruction gate selected f8t2d32. The later M6 study nevertheless froze f8t4d16 as its explicitly cost-oriented modern-reference codec, so that choice must not be described as the fidelity winner. | output/codec_selection_f8t4d16_vs_f8t2d32/comparison.json; M6 run manifests |
| L4 | Codec training progression | f8t2 and f4t4, 1k–40k milestones; t1/t2/t4 windows | Where do short- and long-window reconstructions stabilize? | Completed milestone and long-clip audits | output/vae_f8t2_mirrored_progress/; output/vae_f4t4_f80_progress/; output/vae_long_audit_30k/; output/vae_long_audit_40k/ |
| M6-0 | Latent block-AR main run | 39.8M, full spatial-temporal attention, 10k | Can a compact latent autoregressive model generate the domain? | Completed; structure improves but motion collapses at longer training | output/m6_f8t4d16_fullst_10k_s0/run_manifest.json |
| M6-1 | Horizon ablation | h4, h8, h40, 2k each | Is motion collapse driven by the teacher-forced short-horizon factorization? | Executed; horizon materially changes motion/structure behavior | output/m6v3_start_aligned_h4_2k_s0/; output/m6v3_start_aligned_h8_2k_s0/; output/m6v3_start_aligned_h40_2k_s0/ |
| M6-2 | Decoded-RGBA auxiliary loss | h8, 2k and n=64 evaluation | Does a decoded-pixel auxiliary objective repair structural/motion trade-offs? | No-go under the declared gate | output/m6v4_start_aligned_h8_decoded_aux_2k_s0/; output/m6_h8_decoded_aux_n64/ |
| M6-3 | Main-run milestone evaluation | 5k/10k/15k/20k, n=64 | Does additional training monotonically improve both structure and motion? | Structure improves; motion collapses by 10k and remains flat | paper/results/m6_h8_milestones_n64.json |
| R0 | Full-clip control | 10k | Is M6 motion collapse merely a capacity or latent-loss failure? | No freeze; supports the short-horizon factorization diagnosis | experiment log/status history; companion-paper artifacts |
| SRE-0 | Skeleton recovery evaluator gate | 3.30M-param single-frame RGBA→27-joint regressor; 20k steps; corruption/offscreen tests | Can generated pixels be mapped back to a 2D rig for explanatory scoring? | **Coordinate instrument passed its declared gates:** test mean .657 px, PCK@2 .934, PCK@4 .976; chain-complete corruption localization passed. Far-offscreen gate is vacuous because the released mini tier contains no such frames. A 5k UNet rollout overlay confirms that the instrument can track generated pixels, but v1 has no calibrated joint confidence. | paper/results/sre_v1_validation_test.json; paper/results/sre_v1_validation_corruptions_v2.json; output/video/sre_overlay_unet_5k_running_man.mp4 | Optional learned-geometry baseline, not an oracle; confidence and analysis-by-synthesis remain a separate gate |

## Reading the matrix

- The current paper's completed quantitative claims come from **E0–E2** and the native reference-model evaluation **C5**.
- The beginner release is the factorised-UNet route **C0–C5**; it does not require reproducing the archived DiT or M6 studies.
- **L0–R0** explain why a Video VAE and autoregressive latent generator were not placed in the core Colab. They are
  useful research history, but including them in the dataset report would blur its contribution.
- A future learned rig estimator is not an “oracle” until SRE-style recovery has calibrated confidence and passes
  controlled malformation, occlusion, and renderer-shift gates.
