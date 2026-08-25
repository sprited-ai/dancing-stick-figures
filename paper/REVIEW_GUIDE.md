# Dancing Stick Figures dataset paper — Jin review guide (v0.2)

Authoritative source: `paper/paper.tex`

Live PDF: `output/pdf/paper.pdf`

The older `paper_mixed_v01.tex` and `REPORT.md` are archived provenance, not review targets.

## The question to keep in mind

Would a professor adopt this release to let students train, modify, and diagnose a video generator from scratch on
ordinary hardware? The paper should not read as a small substitute for a frontier model. It should read as a complete
experiment whose data, renderer, reference models, and measurements can all be inspected and changed.

## 30-minute review route

1. **Abstract and Introduction (pages 1–2, 8 min).**
   - Is the access problem recognisable: unavailable training data, expensive pretraining, and evaluation that does
     not explain visible failures?
   - Is the answer concrete: a 0.85-GB route, one-GPU training, known rendering state, and runnable image-to-video
     curriculum?
   - Reject any sentence that sounds like a claim of photorealism, state of the art, or general real-video validity.

2. **Related Work (page 2, 5 min).**
   - SURREAL and BEDLAM are stronger synthetic-human datasets.
   - Synthetic Video Enhances Physical Fidelity and DynaVid give stronger evidence that synthetic supervision can
     improve large video generators.
   - VBench, GeneVA, and Ref4D provide broader or more human-aligned evaluation.
   - Our defensible distinction is the conjunction: reconstructible data, from-scratch text-conditioned video
     training, controlled corruptions, and a short classroom-scale feedback loop.

3. **Data and evaluation (pages 3–5, 7 min).**
   - Confirm 1,430 motions, three views, 4,290 videos, 514,800 frames, 20 fps, and 120-frame source clips.
   - Read the controlled-corruption figure as a metric unit test: freezing, shuffling, reversing, and looping expose
     different blind spots. Reversal is intentionally a negative result for FVD and the time-symmetric diagnostics.
   - Check that each structural measurement is tied to a visible palette or connectivity failure rather than being
     presented as a universal quality judge.

4. **Reference models and Colab route (pages 5–6, 7 min).**
   - The primary released reference is a factorised 3D UNet whose spatial and temporal blocks can be read directly
     and whose released lesson fits a Colab T4; the larger RTX 4090 batch is used for throughput.
   - Table 3 is the reproduction target: three-seed, 120-frame UNet values are compared with the matching real
     reference, and Figure 4 places walking, running, and sitting outputs directly below released clips.
   - The Colab should visibly train an image generator, transfer spatial weights, train a text-conditioned video
     generator, sample labelled prompts, and score results.
   - Check that 32×32 is presented only as a faster sanity-check configuration.

5. **Limits and release contract (final pages, 3 min).**
   - The paper must distinguish exact public reconstruction from instructor-private renderer variations.
   - Prompt conditioning must not be described as verified semantic prompt following.
   - Learned rig recovery is proposed as future work, not reported as a metric or result of this release.
   - Any unfinished measurement or experiment must remain marked `[TODO]` in prose or the experiment matrix rather
     than being implied complete.

## Release gates still being verified

- Native-cadence 64×64 factorised 3D UNet: complete at 10k steps with three sampling seeds, 120-frame evaluation,
  and the walking/running/sitting qualitative suite.
- T4 fit, update speed, validation, peak memory, and 120-frame rollout are measured. The exact release source also
  completes typed-prompt generation and scoring on RTX 4090; a later quota-available run may consolidate both records
  into one T4 provider session without changing the paper's current hardware-specific claims.
- Full 1,430-motion reconstruction is complete: both released tiers passed over all 514,800 frames.
- Final PDF visual QA and Korean narrated walkthrough MP4.

The native model's FVD is 510.1 ± 31.4 across three sampling seeds, compared with a 129.9 real–real reference under
the same n=64 manifest. Its motion is faster and less smooth than the reference, and fixed-noise walking/running
samples remain similar. These are useful baseline weaknesses, not hidden release defects or prompt-following claims.

Evidence and remaining work are tracked in `paper/EXPERIMENT_MATRIX.md`; M6 autoregressive research belongs to the
separate mechanism-paper track and is not part of this dataset-paper review.
