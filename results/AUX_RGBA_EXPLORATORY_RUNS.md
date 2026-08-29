# RGBA-auxiliary exploratory reference runs

Status: final matched-exposure queue in progress. These runs remain separate
from Table 3 until their canonical evaluations are verified.

## Scope and interpretation

- All four runs use the released 4,020-clip `mini_v02` cache, the first 64
  frames at 64x64, T5-small conditioning, velocity-prediction flow matching,
  foreground weight 2, and an RGBA clean-prediction auxiliary term of weight 1.
- The pixel-space auxiliary is not mathematically identical to Mini-Wan's
  decoded auxiliary. Pixel DiTs apply the RGBA term directly to predicted clean
  pixels; Mini-Wan decodes predicted and target latents through its frozen
  codec. These runs must not be described as a loss-controlled Mini-Wan
  architecture ablation.
- A resumable full-state `ckpt.pt`, an archived EMA-only numbered checkpoint,
  and a 50-step EMA GIF are written every 1,000 optimizer updates.
- The trainer deployed to both hosts now writes future checkpoints through a
  process-specific temporary file followed by atomic replacement. The two
  already-running processes retain their loaded code; the change applies to
  any recovery launch and to fin's queued random-initialised run.

## Runs

### Local mixer, image initialisation

- Host: `gin`
- Output: `/data/dancing-stick-figure-paper/results/paper1_v03c_t2v64_local3d_image30k_rgba_s0`
- Model: 41.8M pixel DiT, factorised attention plus local 3x3x3 mixer
- Optimisation: batch 8 x accumulation 2, 30,000 optimizer updates, BF16,
  `torch.compile`, no activation checkpointing
- Verified milestones: rolling full-state checkpoint plus numbered EMA-only
  snapshots at 11k through 30k, with 64-frame GIFs; checked manifests record
  50 sampling steps. Validation loss at 14k: 0.0083; at 16k: 0.0073; at 18k:
  0.0066; at 20k: 0.0062; at 22k: 0.0068; at 24k: 0.0062; at 26k: 0.0063;
  at 28k: 0.0061; at 30k: 0.0064. The run completed at 30k. A final audit
  verified all 20 numbered checkpoints, GIFs, and manifests from 11k through
  30k; every GIF has 64 frames and every manifest records 50 sampling steps.
- Recovery snapshot: `resume_013000.pt`, byte-for-byte SHA-256 match to the
  loaded and verified 13k full-state `ckpt.pt`:
  `f11d6908a207f1513a131c858340e54435bb937dea4ed483e615a7573402d45e`
- Recovery snapshot: `resume_014000.pt`, byte-for-byte SHA-256 match to
  the loaded and verified 14k full-state `ckpt.pt`:
  `df0fec405be12ec66f37ce0c3432ae11c0d25aabd6305e86e97b6957c2fe0559`
- Recovery snapshot: `resume_015000.pt`, byte-for-byte SHA-256 match to
  the loaded and verified 15k full-state `ckpt.pt`:
  `ca4fbf772167c56d298342ed1a8e9f8e24c6a618b5140596995f55671b86c045`
- Recovery snapshot: `resume_016000.pt`, byte-for-byte SHA-256 match to
  the loaded and verified 16k full-state `ckpt.pt`:
  `7b7c85ef114aac32ab237673f7e46339103f3d428900d02fd9d465a1494ce5ef`
- Recovery snapshot: `resume_017000.pt`, byte-for-byte SHA-256 match to
  the loaded and verified 17k full-state `ckpt.pt`:
  `3512fbd8fca7eef6d6187a701ec92aeaefd3a9ab88b1d5a9664bcf7f097a8aa7`
- Recovery snapshot: `resume_018000.pt`, byte-for-byte SHA-256 match to
  the loaded and verified 18k full-state `ckpt.pt`:
  `75eaeb8b82ecb50c7f8f6f3793f483e636aa511d75008413a5dccfeeae3fb43c`
- Recovery snapshot: `resume_019000.pt`, byte-for-byte SHA-256 match to
  the loaded and verified 19k full-state `ckpt.pt`:
  `5c2e56d83ef80f241829ea0ee1feae5394661b2c54e416e292b6bb083b742f1c`
- Recovery snapshot: `resume_020000.pt`, byte-for-byte SHA-256 match to
  the loaded and verified 20k full-state `ckpt.pt`:
  `931d56c8e01208460190aa98ce672b1e3bab9b2c593290fadc8f37e674972c82`
- Recovery snapshot: `resume_021000.pt`, byte-for-byte SHA-256 match to
  the loaded and verified 21k full-state `ckpt.pt`:
  `4770b1a6e75232045f8b6ea35dc86e557c31089abf522b39a2a90763443cab86`
- Recovery snapshot: `resume_022000.pt`, byte-for-byte SHA-256 match to
  the loaded and verified 22k full-state `ckpt.pt`:
  `737bb1a938703f5933a52537af6574ed37e35cbd26387b70b3d9116c46c63e02`
- Recovery snapshot: `resume_023000.pt`, byte-for-byte SHA-256 match to
  the loaded and verified 23k full-state `ckpt.pt`:
  `6cf98e95ce0fb86f20909933e5f86facd6d891bcdc2b0139d6eabdb39d3a3c50`
- Recovery snapshot: `resume_024000.pt`, byte-for-byte SHA-256 match to
  the loaded and verified 24k full-state `ckpt.pt`:
  `18b7027c8da0d3af014037c38a5c0c7d4b113a0122e960e30e0c549a49cf006f`
- Recovery snapshot: `resume_025000.pt`, byte-for-byte SHA-256 match to
  the loaded and verified 25k full-state `ckpt.pt`:
  `98a575e61aed795bfd4e4e0e9539a8e681a7f0c7dbc59f04a6f1f8d9ca24a65f`
- Recovery snapshot: `resume_026000.pt`, byte-for-byte SHA-256 match to
  the loaded and verified 26k full-state `ckpt.pt`:
  `b410cdf88635fd23f8df47e68745478581378f6ecf76930e8fb6acefae77a888`
- Recovery snapshot: `resume_027000.pt`, byte-for-byte SHA-256 match to
  the loaded and verified 27k full-state `ckpt.pt`:
  `6bb21ed14b28e9c1523cc66651d2765107f1cc1078cb5ecf41210c160fe28ecd`
- Recovery snapshot: `resume_028000.pt`, byte-for-byte SHA-256 match to
  the loaded and verified 28k full-state `ckpt.pt`:
  `35bad1935ed1346fa52cead514124cc271a7f654ca89f0c8b7f41df193783e2a`
- Recovery snapshot: `resume_029000.pt`, byte-for-byte SHA-256 match to
  the loaded and verified 29k full-state `ckpt.pt`:
  `8c242f905e76dca053ac757e8f36b54a0a4b0179cb7e39f4bef9b4d30e2eeec2`
- Final recovery snapshot: `resume_030000.pt`, byte-for-byte SHA-256 match to
  the loaded and verified 30k full-state `ckpt.pt`:
  `d3e59ee420f8133a959bc4ffa61de677e3ab328f7fd0f9f5d2bb7600bcec54e4`

### Joint spatio-temporal DiT, image initialisation (closed exploratory run)

- Host: `gin`
- Output: `/data/dancing-stick-figure-paper/results/paper1_v03c_t2v64_fullst_image30k_rgba_s0`
- Model: 39.9M pixel DiT with joint attention over all spatio-temporal tokens
- Optimisation: batch 8 x accumulation 2, 30,000 optimizer updates, BF16,
  `torch.compile`, and activation checkpointing; all loss, optimiser, data,
  cadence, and seed settings match the other RGBA-auxiliary pixel runs
- Initialisation: `results/paper1_v02c_t2i30k_s0/ckpt_030000.pt`, SHA-256
  `d3865d04e2f7f0660d13c56050af89f625f1d034d437b6a9c2c07b5ae4b084e9`
- Launch preflight: `args.json` records the image checkpoint, empty `resume`,
  `full_st=true`, `rgba_aux_loss=1`, batch 8 x accumulation 2, and 30,000
  optimizer updates. The first compiled log point is step 10 at 2.50 s/update,
  7.0 GB peak, and an initial ETA of 20.8 hours.
- Verified milestones: the 2k and 5k--7k rolling full-state checkpoints record
  `full_st=true`, the expected image checkpoint, and `rgba_aux_loss=1`; the
  corresponding numbered EMA-only checkpoints, 64-frame GIFs, and NFE-50
  manifests are present, with no temporary checkpoint remnants. Validation
  loss at 2k is 0.0176.
- Recovery snapshot: `resume_002000.pt`, byte-for-byte SHA-256 match to the
  loaded and verified 2k full-state `ckpt.pt`:
  `7f6531223489b3e395ad377e76b4c93364bc9bf64299bfd07ca70fecd95c904a`
- Recovery snapshot: `resume_005000.pt`, byte-for-byte SHA-256 match to
  the loaded and verified 5k full-state `ckpt.pt`:
  `425c83adf0780a8d8a9f503b08b13e1015de932d77fcd77c6a3735c6fe82de06`
- Recovery snapshot: `resume_006000.pt`, byte-for-byte SHA-256 match to
  the loaded and verified 6k full-state `ckpt.pt`:
  `49e857517403de074dda832acff538ff2c8d144c8ca51ee779f9fe75596848c1`
- Latest recovery snapshot: `resume_007000.pt`, byte-for-byte SHA-256 match to
  the loaded and verified 7k full-state `ckpt.pt`:
  `cea3189bfae37ee9b3b9073bbeb6e832660a10c75fc217c1591e81583a891d5d`
- Final disposition: intentionally stopped after the verified 7k recovery
  point when the paper comparison was narrowed to factorised and local-mixer
  references. This incomplete run will not be resumed, evaluated for Table 3,
  or used as paper evidence.
- Recovery note: the first launch omitted the `--init` trainer argument and was
  stopped after 10 random-initialised updates. Its artifacts were preserved as
  `results/aborted/paper1_v03c_t2v64_fullst_random_accidental_20260828T135911Z`;
  no values from that aborted launch are used here.

### Final matched-exposure factorised queue

- Host: `gin`
- Queue log: `/data/dancing-stick-figure-paper/results/gin_factorised_matched_epochs_queue.log`
- Image-initialised output:
  `/data/dancing-stick-figure-paper/results/paper1_v04_t2v64_factorised_image30k_rgba_b8a2_s0`
- Random-initialised output:
  `/data/dancing-stick-figure-paper/results/paper1_v04_t2v64_factorised_random30k_rgba_b8a2_s0`
- Queue order: image initialisation, then random initialisation; both start
  from step zero and run for 30,000 optimizer updates.
- Optimisation: batch 8 x accumulation 2, BF16, `torch.compile`, no activation
  checkpointing, and the same loss, optimiser, data, conditioning mixture,
  cadence, and seed settings as the completed local-mixer run.
- Exposure: 480,000 sampled windows per run, or 37.3 loader epochs where one
  epoch is 12,864 sampled windows (four samples from each of 3,216 training
  source motions).
- Scope: these are the final new training runs for the paper. No additional
  architecture or recovery experiment will be launched if their results are
  disappointing; the verified outcomes will be reported as observed.
- Image-initialised run verified through 13,000 optimizer updates. The loaded
  rolling full-state checkpoint records step 13,000 and the declared image
  checkpoint; its stored arguments confirm 64 frames, batch 8 x accumulation
  2, seed 0, factorised attention, RGBA auxiliary weight 1, `torch.compile`,
  and no activation checkpointing. `ckpt_013000.pt`, the 64-frame
  `sample_013000.gif`, and the NFE-50/CFG-3 manifest are present, with no
  temporary remnants. `resume_013000.pt` is a byte-for-byte recovery copy of
  the rolling checkpoint with SHA-256
  `3f38f8e16cdd6d8921a6ffbd1dd415c54edb5b32a09d83b1cb670e9c437e33b3`.
- Image-initialised run subsequently verified at 19,000 optimizer updates. The
  loaded rolling state records step 19,000 and the same declared image
  checkpoint and matched configuration. `ckpt_019000.pt`, the 64-frame
  `sample_019000.gif`, and its NFE-50/CFG-3 manifest are present, and no
  temporary remnants were found. `resume_019000.pt` is a byte-for-byte
  recovery copy of the rolling checkpoint with SHA-256
  `a94d2147f482fc92744e090a43def061231b9be7c96faaf0de81ef008dc382d6`.
- Image-initialised run completed and was verified at 30,000 optimizer
  updates; its final numbered EMA checkpoint, 64-frame GIF, and
  NFE-50/CFG-3 manifest are present.
- Random-initialised run subsequently started from step zero and was verified
  at 6,000 optimizer updates. The loaded rolling state records step 6,000,
  an empty `init` field, 64 frames, batch 8 x accumulation 2, seed 0,
  factorised attention, RGBA auxiliary weight 1, `torch.compile`, and no
  activation checkpointing. `ckpt_006000.pt`, the 64-frame
  `sample_006000.gif`, and its NFE-50/CFG-3 manifest are present; no temporary
  remnants were found. `resume_006000.pt` is a byte-for-byte recovery copy of
  the rolling checkpoint with SHA-256
  `4d5ad67b2364e199eec2070e69f17989069b652aa4c8698c624985c520c30619`.
- Random-initialised run subsequently verified at 13,000 optimizer updates.
  The loaded rolling state retains the matched random-init configuration;
  `ckpt_013000.pt`, the 64-frame `sample_013000.gif`, and its
  NFE-50/CFG-3 manifest are present with no temporary remnants.
  `resume_013000.pt` is a byte-for-byte recovery copy with SHA-256
  `bd6a90d3938d7db331fd70a7d8a39193e918b85be0b7161b4271930bd3c97a6d`.
- Random-initialised run subsequently verified at 18,000 optimizer updates;
  validation loss was 0.0086. The matched random-init configuration,
  numbered EMA checkpoint, 64-frame GIF, and NFE-50/CFG-3 manifest were
  verified with no temporary remnants. `resume_018000.pt` is a byte-for-byte
  recovery copy with SHA-256
  `121050f2222698a242b2d02d018ae0e65db64211f6ff7ad93a4e6df8e6b998a3`.
- Random-initialised run subsequently verified at 19,000 optimizer updates.
  Its matched configuration, numbered EMA checkpoint, 64-frame GIF, and
  NFE-50/CFG-3 manifest are present with no temporary remnants.
  `resume_019000.pt` matches the rolling state byte-for-byte with SHA-256
  `a3b0c9239a08b9aa2dda3f76052c17f68b8524c0636da0da9e2e105f55cce048`.
- Random-initialised run subsequently verified at 22,000 optimizer updates;
  validation loss was 0.0082. Its matched configuration, numbered checkpoint,
  64-frame GIF, and NFE-50/CFG-3 manifest are present with no temporary
  remnants. `resume_022000.pt` matches the rolling state byte-for-byte with
  SHA-256
  `c597c0d5255148db307f00dffc4684c44f78f15bc8667c4b4dd218648aabc099`.
- Random-initialised run subsequently verified at 24,000 optimizer updates;
  validation loss was 0.0087. Its matched configuration, numbered checkpoint,
  64-frame GIF, and NFE-50/CFG-3 manifest are present with no temporary
  remnants. `resume_024000.pt` matches the rolling state byte-for-byte with
  SHA-256
  `ac9ca87708425693911873a00f9c751537c03cfa22045821dd0b6563c9a06be4`.
- Both final factorised runs completed at 30,000 optimizer updates. Their
  rolling full-state checkpoints record the declared initialisation and
  matched configuration; final numbered EMA checkpoints, 64-frame GIFs, and
  NFE-50/CFG-3 manifests are present with no temporary remnants.
- Canonical evaluation used the frozen `results/v02c_eval/win64_manifest.json`,
  128 held-out source animations, sampling seeds 0/1/2, 64 frames at stride 1,
  50 Euler steps, and CFG 3. The final image-initialised result is TVR .1605,
  LIE .0330, CPE .0285, centroid speed .3496, motion fraction .4467, angular
  jerk .1506, and FVD 319.0 (seed standard deviation 3.7). Its result JSON has
  SHA-256 `c75ef54f26acac4154b23c1ce272204fb25c026229c04ceb55341f7b8c11fac7`.
- Under the same protocol, the final random-initialised result is TVR .2592,
  LIE .0417, CPE .0316, centroid speed .3657, motion fraction .4307, angular
  jerk .1723, and FVD 483.7 (seed standard deviation 12.0). Its result JSON has
  SHA-256 `55114d60152a347318787ddc892c53dc551fe52f851743f1538db0677084d292`.
- The gin evaluation queue exited after both verified result JSONs were
  written. No manuscript number was changed automatically.

### Factorised DiT, image initialisation

- Host: `fin`
- Output: `/home/fin/dancing-stick-figures/results/explore_factorised_rgba_aux_image30k_fin_s0`
- Model: 39.9M pixel DiT, factorised attention
- Optimisation: batch 4, no accumulation, 30,000 optimizer updates, BF16,
  `torch.compile`, activation checkpointing
- Initialisation: exact 30k Image DiT checkpoint copied from `gin`
- Verified milestones: rolling full-state checkpoint plus numbered EMA-only
  snapshots at 1k through 30k, with 64-frame GIFs; each checked manifest records
  50 sampling steps, and each GIF contains 64 frames. Validation loss at 4k:
  0.0151; at 6k: 0.0137; at 8k: 0.0114; at 10k: 0.0123; at 12k: 0.0106;
  at 14k: 0.0105; at 16k: 0.0114; at 18k: 0.0113; at 20k: 0.0083;
  at 22k: 0.0092; at 24k: 0.0108; at 26k: 0.0080; at 28k: 0.0066;
  at 30k: 0.0094. The run completed at 30k.
- Recovery snapshot: `resume_004000.pt`, byte-for-byte SHA-256 match to the
  loaded and verified 4k full-state `ckpt.pt`:
  `c65a93816f80bde16d946e3707ab5ba8e26f864e40269e2793d5af6143aa20fa`
- Recovery snapshot: `resume_005000.pt`, byte-for-byte SHA-256 match to
  the loaded and verified 5k full-state `ckpt.pt`:
  `4d3cda125bb4a436308c0096f06ef1a3a756114e9250499ce35b2cc2dbcefc12`
- Recovery snapshot: `resume_006000.pt`, byte-for-byte SHA-256 match to
  the loaded and verified 6k full-state `ckpt.pt`:
  `5b9b8dd8b2ce11202c8146800cc27a63eabcda57c783932ef8941874ebff1c03`
- Recovery snapshot: `resume_007000.pt`, byte-for-byte SHA-256 match to
  the loaded and verified 7k full-state `ckpt.pt`:
  `a732b85476b2ace468328ea81373643d9065085b9cdcea9ba900fc8f77f817b1`
- Recovery snapshot: `resume_008000.pt`, byte-for-byte SHA-256 match to
  the loaded and verified 8k full-state `ckpt.pt`:
  `c6c752924d8edec17de462f0f8677ba2c6ca03c840e0a3b8e848f0d191c36687`
- Recovery snapshot: `resume_009000.pt`, byte-for-byte SHA-256 match to
  the loaded and verified 9k full-state `ckpt.pt`:
  `4970fdfe95091b88bf108c7beafcfa0fb6c56444b9f90758a0285a24e3922e9e`
- Recovery snapshot: `resume_010000.pt`, byte-for-byte SHA-256 match to
  the loaded and verified 10k full-state `ckpt.pt`:
  `d3aa7d0938ad4e5abf2cc0c02346bed32085e47687b5031bd7a150a4d9670a24`
- Recovery snapshot: `resume_011000.pt`, byte-for-byte SHA-256 match to
  the loaded and verified 11k full-state `ckpt.pt`:
  `77b93b2774550488908aafd415a63582309ef9a138acb98bb345780692f51bb8`
- Recovery snapshot: `resume_012000.pt`, byte-for-byte SHA-256 match to
  the loaded and verified 12k full-state `ckpt.pt`:
  `63962433a489b82cd516d1ec765b10dda67e7547aa6b130105461947cbd92ad3`
- Recovery snapshot: `resume_013000.pt`, byte-for-byte SHA-256 match to
  the loaded and verified 13k full-state `ckpt.pt`:
  `f25d34ec1b13091db5b62a79fb805d5ee87f4fbd912153b9d2ec10f21ba572ae`
- Recovery snapshot: `resume_014000.pt`, byte-for-byte SHA-256 match to
  the loaded and verified 14k full-state `ckpt.pt`:
  `198655572dd942b685ff33cdb1cf5aa48099c44beed45ba527321bfcbc3e2a17`
- Recovery snapshot: `resume_015000.pt`, byte-for-byte SHA-256 match to
  the loaded and verified 15k full-state `ckpt.pt`:
  `37d4c5180bc03d98b7f87a5efd23902d6f6f13abc89ce541ba8b2fc8c80ec6b5`
- Recovery snapshot: `resume_016000.pt`, byte-for-byte SHA-256 match to
  the loaded and verified 16k full-state `ckpt.pt`:
  `75a5e7049968073e6257a1081735a98d98faa976ce6ebfdb59a7957ecd1cd4c3`
- Recovery snapshot: `resume_017000.pt`, byte-for-byte SHA-256 match to
  the loaded and verified 17k full-state `ckpt.pt`:
  `470c8595b0403a5cdc7dffb7c029ac37d9c15f44a1a443752f5c79386c2d7525`
- Recovery snapshot: `resume_018000.pt`, byte-for-byte SHA-256 match to
  the loaded and verified 18k full-state `ckpt.pt`:
  `3164a4d0aad9d4d29c26903b15f2a3bafe5a4f7f5b1b76dbe3e91511421fb3cc`
- Recovery snapshot: `resume_019000.pt`, byte-for-byte SHA-256 match to
  the loaded and verified 19k full-state `ckpt.pt`:
  `d19546966c48caba09ddff185aaa0cfac99ca42090df5203a1da66451d7e7d9e`
- Recovery snapshot: `resume_020000.pt`, byte-for-byte SHA-256 match to
  the loaded and verified 20k full-state `ckpt.pt`:
  `62e7cbc394042b912ba0e3e81286ae17289bfbb96acb5c5f2e5ae8621851ed`
- Recovery snapshot: `resume_021000.pt`, byte-for-byte SHA-256 match to
  the loaded and verified 21k full-state `ckpt.pt`:
  `9485961a2cb865d9b5ef9b746c7d2e4d19d0c6d6fdca20f9ec99738be93a98c3`
- Recovery snapshot: `resume_022000.pt`, byte-for-byte SHA-256 match to
  the loaded and verified 22k full-state `ckpt.pt`:
  `89cf33d5a695118cc983968bd11fef3f5a9bf1c701286a2e268a635b87e1c571`
- Recovery snapshot: `resume_023000.pt`, byte-for-byte SHA-256 match to
  the loaded and verified 23k full-state `ckpt.pt`:
  `5bc471bc8e4f0fdd13ebc364c3c344e13f702c8fc4e9bf6fc97aaa2c750e982a`
- Recovery snapshot: `resume_024000.pt`, byte-for-byte SHA-256 match to
  the loaded and verified 24k full-state `ckpt.pt`:
  `ba1b1febeb11df34762d7fa076b0c3392911993ac6e904f7fa2fae12e577f918`
- Recovery snapshot: `resume_025000.pt`, byte-for-byte SHA-256 match to
  the loaded and verified 25k full-state `ckpt.pt`:
  `95b196678b119670878beb588ec1e12b1fcb9fd856264c21ebdd8e27a35719aa`
- Recovery snapshot: `resume_026000.pt`, byte-for-byte SHA-256 match to
  the loaded and verified 26k full-state `ckpt.pt`:
  `4361f05621538d25721c9102610ebcf9b96d1d3406ca25736b35b05538211726`
- Recovery snapshot: `resume_027000.pt`, byte-for-byte SHA-256 match to
  the loaded and verified 27k full-state `ckpt.pt`:
  `a0f19d81832da1de4a10ba0e5799283cf1c248a345ed7f9842fe5cff85dc8179`
- Recovery snapshot: `resume_028000.pt`, byte-for-byte SHA-256 match to
  the loaded and verified 28k full-state `ckpt.pt`:
  `4a6bca74520b0717c340bf603929f32f61a73a20cd2a0e0e3de3928d1bf35871`
- Recovery snapshot: `resume_029000.pt`, byte-for-byte SHA-256 match to
  the loaded and verified 29k full-state `ckpt.pt`:
  `ea674a67745b9c6c5c70631b5d940e4decaf8a40b8d0a226c07436ad87790330`
- Final recovery snapshot: `resume_030000.pt`, byte-for-byte SHA-256 match to
  the loaded and verified 30k full-state `ckpt.pt`:
  `41af96777dd69d440bf841f66899da34be8ddc275b293a8a046277febf488615`

### Factorised DiT, random initialisation

- Host: `fin`
- Output: `/home/fin/dancing-stick-figures/results/explore_factorised_rgba_aux_random30k_fin_s0`
- Configuration: same as the factorised image-initialised run except for random
  denoiser initialisation
- Queue state: started automatically after the image-initialised run completed;
  the live command contains no `--init` argument and otherwise retains the
  matched factorised configuration above
- Verified milestones: rolling full-state checkpoint plus numbered EMA-only
  snapshots at 1k through 30k, each with a 64-frame GIF and a manifest recording 50
  sampling steps. The loaded checkpoints record an empty `init` field, and no
  temporary checkpoint file remained after either atomic save. Validation loss
  at 2k: 0.0412; at 4k: 0.0285; at 6k: 0.0234; at 8k: 0.0195;
  at 10k: 0.0196; at 12k: 0.0150; at 14k: 0.0142; at 16k: 0.0144;
  at 18k: 0.0147; at 20k: 0.0106; at 22k: 0.0114; at 24k: 0.0133;
  at 26k: 0.0100; at 30k: 0.0092. A host power interruption stopped the
  process after the verified 27k save; training resumed from that exact
  full-state checkpoint and completed at 30k.
- Recovery snapshot: `resume_001000.pt`, byte-for-byte SHA-256 match to
  the loaded and verified 1k full-state `ckpt.pt`:
  `60350511eef3225cfd829c2cc9493101c860f96ca6cca6179e363558b720ba91`
- Recovery snapshot: `resume_002000.pt`, byte-for-byte SHA-256 match to
  the loaded and verified 2k full-state `ckpt.pt`:
  `489de1db3182ab9a07b6c16c05b39723144e5b30eb6a28c8b01cfb36d958c686`
- Recovery snapshot: `resume_003000.pt`, byte-for-byte SHA-256 match to
  the loaded and verified 3k full-state `ckpt.pt`:
  `fcfbb79114c3642f23365ba2499859d332cebbb1c83bb2c4d46f648573f2205c`
- Recovery snapshot: `resume_004000.pt`, byte-for-byte SHA-256 match to
  the loaded and verified 4k full-state `ckpt.pt`:
  `0deb5aa66f591e1094257c576f9c6007e7d7b3d3cf11eecc7469bb68999a28d9`
- Recovery snapshot: `resume_005000.pt`, byte-for-byte SHA-256 match to
  the loaded and verified 5k full-state `ckpt.pt`:
  `d0ee8d866aca93034e851a5155cd87e07e8c877d41f81ef9583e24e3ec5ac4f6`
- Recovery snapshot: `resume_006000.pt`, byte-for-byte SHA-256 match to
  the loaded and verified 6k full-state `ckpt.pt`:
  `0cb53de8ef1900d24a027ae4e06fb9d291b19327d58c98be1b915f67823a5ef0`
- Recovery snapshot: `resume_007000.pt`, byte-for-byte SHA-256 match to
  the loaded and verified 7k full-state `ckpt.pt`:
  `f1ba4a50e89d048119f9133a6b9790922e0513d2fb68384a7d4a9b25374960cb`
- Recovery snapshot: `resume_008000.pt`, byte-for-byte SHA-256 match to
  the loaded and verified 8k full-state `ckpt.pt`:
  `79e775fcdebe976853686d1ef8de15134fcaee14d60ad6ca13e63361af86d7b1`
- Recovery snapshot: `resume_009000.pt`, byte-for-byte SHA-256 match to
  the loaded and verified 9k full-state `ckpt.pt`:
  `5347e096197159d53e92f000b28d45bfee7fe923bb8be788026e2dc58676b9d5`
- Recovery snapshot: `resume_010000.pt`, byte-for-byte SHA-256 match to
  the loaded and verified 10k full-state `ckpt.pt`:
  `08ec963d55f7cd5f11f4ce46557f86ccf61e4d614690e0bef03d56c118e0c3c7`
- Recovery snapshot: `resume_011000.pt`, byte-for-byte SHA-256 match to
  the loaded and verified 11k full-state `ckpt.pt`:
  `edc5b4967c1b3d9240d00e216c2ce6dcdb50eb396b403856ccf6e13048938fd7`
- Recovery snapshot: `resume_012000.pt`, byte-for-byte SHA-256 match to
  the loaded and verified 12k full-state `ckpt.pt`:
  `0f2a292f9055be1d89a62050e75b3d08289ea0fffe1ae5b344816034983d5a58`
- Recovery snapshot: `resume_013000.pt`, byte-for-byte SHA-256 match to
  the loaded and verified 13k full-state `ckpt.pt`:
  `f91481fe6048b7e532be5139f6276dc4cd69c5e3a45bc189cb040cd39e3cfff2`
- Recovery snapshot: `resume_014000.pt`, byte-for-byte SHA-256 match to
  the loaded and verified 14k full-state `ckpt.pt`:
  `b74af4c22ae4adf9ae0a2577d919a601270afc88c4dd4e25e75cd316d4de5c75`
- Recovery snapshot: `resume_015000.pt`, byte-for-byte SHA-256 match to
  the loaded and verified 15k full-state `ckpt.pt`:
  `94f5d558de829d9465279ae61b3bfa685fb709f94a1454321379c7fad545dee9`
- Recovery snapshot: `resume_016000.pt`, byte-for-byte SHA-256 match to
  the loaded and verified 16k full-state `ckpt.pt`:
  `bb3bfe86b0408308146c120602dd555edd429a4bcc8632f45bea8b21f47ebb6f`
- Recovery snapshot: `resume_017000.pt`, byte-for-byte SHA-256 match to
  the loaded and verified 17k full-state `ckpt.pt`:
  `251ed2eca10b517ccdaac7aa612d1bdc11f9a097d2b09e97a4719e020f9add1d`
- Recovery snapshot: `resume_018000.pt`, byte-for-byte SHA-256 match to
  the loaded and verified 18k full-state `ckpt.pt`:
  `c83b2b8da47b25feaf8d2166486b6b38b6f411084e0a6ff01f1ef41451ad5d98`
- Recovery snapshot: `resume_019000.pt`, byte-for-byte SHA-256 match to
  the loaded and verified 19k full-state `ckpt.pt`:
  `55ca966b999704776e75bc2b9b4d8fc2471a47574dfa927ce6485edf35b691be`
- Recovery snapshot: `resume_020000.pt`, byte-for-byte SHA-256 match to
  the loaded and verified 20k full-state `ckpt.pt`:
  `21582d95406f9bce3e4c40ab943c9f3b2263f22ca5e05641a0f6b15da7209387`
- Recovery snapshot: `resume_021000.pt`, byte-for-byte SHA-256 match to the
  loaded and verified 21k full-state `ckpt.pt`:
  `841a8011d85c4c654a51ecbe47fe5b8dce3907f303ce689bfc8db3dad958703c`
- Latest recovery snapshot: `resume_027000.pt`, byte-for-byte SHA-256 match to
  the loaded and verified 27k full-state `ckpt.pt`:
  `1d7fcd529add6f6902b89f7b7bee00f45731cd2935ee898960ee8b8ecab06c38`
- Final recovery snapshot: `resume_030000.pt`, byte-for-byte SHA-256 match to
  the loaded and verified 30k full-state `ckpt.pt`:
  `4b95b92a5852c0025651a2f24bc91c5b37a891af0efd7597d214189c7fd1cd78`

## Input provenance

- `cache/mini_v02/frames.npy`: `3d70537fdfb43f85db9dc1227c49ed1b5174a2e3dcceda3659e64b650f813a90`
- `cache/mini_v02/clips.json`: `461db6d0be28005e4d2821fa39502936db285f8538cdbb7d048d2e775b49508b`
- `cache/mini_v02/meta.json`: `af56fa12e81de2bceeb714f37ebe915a34791b98c5e6fa0a2fb6e76559ad22bc`
- Image checkpoint: `d3865d04e2f7f0660d13c56050af89f625f1d034d437b6a9c2c07b5ae4b084e9`

## Canonical evaluation queue

- `fin` will begin evaluation only after its factorised random-initialisation
  run reaches 30k, so evaluation never shares the 16 GB GPU with training.
- Queue order: factorised image initialisation, factorised random
  initialisation, then the completed local-mixer image-initialisation model.
- Each evaluation uses the frozen `win64_manifest.json`, 128 held-out source
  motions, three sampling seeds, 64 frames at stride 1, 50 Euler steps, and
  CFG 3, matching the Table 3 protocol.
- The local-mixer 30k EMA checkpoint was copied from `gin` to the isolated
  `fin` evaluation import directory and verified byte-identical with SHA-256
  `d9b53e25f98ab6aa06b051cc9fa20d7816eae611d90c08b4c79af9d8ee500de9`.
- Results remain exploratory and are not promoted into Table 3 automatically.
- Canonical 30k evaluation completed for factorised image initialisation:
  TVR .1609, LIE .0415, CPE .0325, centroid speed .3713, motion fraction
  .4532, angular jerk .1626, and FVD 339.5.
- Canonical 30k evaluation completed for factorised random initialisation:
  TVR .3918, LIE .0746, CPE .0427, centroid speed .3443, motion fraction
  .4222, angular jerk .1961, and FVD 716.0.
- Canonical 30k evaluation completed for local-mixer image initialisation:
  TVR .1492, LIE .0504, CPE .0352, centroid speed .3552, motion fraction
  .4695, angular jerk .1461, and FVD 282.9. Per-seed FVD values are 297.8,
  281.1, and 269.7.
- A separate Figure 5 preview was generated from this local-mixer 30k
  checkpoint with the original four prompts, noise seeds, 50-step sampler,
  CFG 3, and displayed frames 0/21/42/63. The verified manifest and source
  strip are in `paper/results/figure5_preview_local3d_30k`; the assembled PNG
  and labeled GIF are in `output/figure5-preview`. It does not replace the
  manuscript figure.
