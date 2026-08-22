# Modern video-generation paper trail

Primary sources checked on 2026-08-20.  These references position the model
ladder; they are not evidence for results in this repository.

- **Wan** — *Wan: Open and Advanced Large-Scale Video Generative Models*,
  arXiv:2503.20314. Open diffusion-transformer family with a video VAE and
  large-scale pretraining. <https://arxiv.org/abs/2503.20314>
- **Wan 2.1 reference implementation** — the released `WanVAE` wrapper uses
  `z_dim=16`; its causal 3D encoder/decoder downsamples space by $8\times$ and
  time by $4\times$ (the first frame is handled separately, then video is
  processed in four-frame groups). These are architecture facts from the
  official code, not measurements on DSF.
  <https://github.com/Wan-Video/Wan2.1/blob/main/wan/modules/vae.py>
- **CogVideoX** — *Text-to-Video Diffusion Models with an Expert Transformer*,
  arXiv:2408.06072. The paper specifies a causal 3D VAE with $8\times8$
  spatial and $4\times$ temporal compression; the official configuration uses
  16 latent channels. <https://arxiv.org/abs/2408.06072>
- **LTX-Video** — *Realtime Video Latent Diffusion*, arXiv:2501.00103. Its
  substantially more aggressive $32\times32\times8$ pixels-per-token codec and
  128-channel latent illustrate a different design point: more representational
  work is moved into the codec so the transformer sees far fewer tokens.
  <https://arxiv.org/abs/2501.00103>
- **Seedance 1.0** — *Exploring the Boundaries of Video Generation Models*,
  arXiv:2506.09113. Unified T2I/T2V/I2V formulation and image-video training
  stages; the public report does not justify a claim that spatial layers were
  frozen under a particular schedule. <https://arxiv.org/abs/2506.09113>
- **MAGI-1** — *Autoregressive Video Generation at Scale*, arXiv:2505.13211.
  Autoregressive prediction of fixed-length video chunks with causal temporal
  modeling. <https://arxiv.org/abs/2505.13211>
- **Diffusion Forcing** — *Next-token Prediction Meets Full-Sequence
  Diffusion*, NeurIPS 2024, arXiv:2407.01392. Independent noise levels across
  sequence tokens; relevant only after the simpler causal baseline is
  measured. <https://arxiv.org/abs/2407.01392>
- **Self Forcing** — *Bridging the Train-Test Gap in Autoregressive Video
  Diffusion*, NeurIPS 2025, arXiv:2506.08009. Conditions training on prior
  self-generated outputs to address exposure bias.
  <https://arxiv.org/abs/2506.08009>
- **T2V-CompBench** — CVPR 2025, arXiv:2407.14505. Seven compositional prompt
  categories with MLLM-, detection-, and tracking-based evaluation validated
  against human judgements. <https://arxiv.org/abs/2407.14505>
- **VBench-2.0** — arXiv:2503.21755. Intrinsic-faithfulness dimensions include
  human fidelity, controllability, physics, creativity, and commonsense.
  <https://arxiv.org/abs/2503.21755>

## Positioning consequence

The current release is a small traced testbed, not a scaled competitor to Wan
or Seedance and not a substitute for broad semantic benchmarks.  K1 provides a
full-clip factorised baseline; K2 may test causal chunking.  Diffusion Forcing
and Self Forcing are follow-ups only if K2 exhibits measured horizon or
exposure-bias failures.  The palette oracle should be reported beside, not in
place of, prompt and human-aligned evaluation.

## Codec decision for M6

The predeclared comparison treats `f8t4d16` as a **modern-reference compression
geometry**, not as a reproduction of Wan or CogVideoX. DSF retains its small,
auditable causal ResNet codec and changes the interface to premultiplied RGBA:
four decoder output channels, foreground-supported RGB reconstruction, and
separate foreground/background alpha reconstruction. It does not copy the
reference models' full channel schedule, depth, normalization, discriminator,
training corpus, or weights.

At $64\times64$, `f8t4d16` produces an $8\times8$ latent grid once per four
video frames. Relative to a four-channel pixel video, its scalar compression is

\[
  (8\cdot8\cdot4\cdot4)/16 = 64\times.
\]

This is distinct from the stride shorthand: `f8t4` means $8\times$ per spatial
axis and $4\times$ in time, not “32 scalar values compressed into one.” The
domain-specific alternative `f8t2d32` has $16\times$ scalar compression and
twice as many temporal latent tokens, so it should preserve thin limbs more
easily but makes the downstream block-AR denoiser more expensive.

For M6's fixed four-video-frame target and one-second history at 20 fps:

| Codec | History latents | Target latents | Latent cells in max window |
|---|---:|---:|---:|
| `f8t2d32` | 10 | 2 | $12\cdot8\cdot8=768$ |
| `f8t4d16` | 5 | 1 | $6\cdot8\cdot8=384$ |

Selection is empirical. The reference geometry is accepted only if it passes
the same fixed-validation reconstruction, alpha-IoU, edge-fidelity, long
sliding-decode, strict causal-prefix, and visual thin-limb gates. If it fails,
retaining `f8t2d32` is a measured domain-specific deviation rather than an
unsupported custom choice. LTX-style compression is not promoted to the main
comparison because $32\times$ spatial compression would leave only a
$2\times2$ latent grid at 64px, a severe risk for one- to three-pixel-wide
limbs.

## Receptive field is not compression ratio

Static dependency tracing of the pinned public implementations gives the
following steady-state spans. These are code-derived upper bounds, not numbers
reported by the original authors and not measures of how strongly the trained
weights use the oldest dependency.

| Codec graph | Encoder raw-frame span | Decoder latent span | Approx. video-frame-equivalent decoder span | Per-layer causal cache |
|---|---:|---:|---:|---:|
| DSF `f8t4`, 2 ResBlocks/stage | 91 | 23--24 | 92--96 | 2 feature frames (not yet statefully retained) |
| Wan 2.1 public VAE | 113 inferred | 38--39 inferred | 152--156 inferred | 2 feature frames |
| CogVideoX Diffusers VAE | 178 inferred | 43--44 inferred | 172--176 inferred | 2 feature frames |

Thus a temporal stride of four does not imply a four-frame receptive horizon.
Wan and CogVideoX preserve the composed long horizon cheaply with small caches
at every causal convolution. DSF's current sliding decoder instead re-decodes a
five-latent/20-frame window, deliberately truncating its theoretical 23--24
latent dependency. Long-window reconstruction and seam metrics must therefore
be reported; stateful per-layer decoding is required before claiming exact
indefinite streaming equivalence.

The spatial dependency is not restricted to the same latent cell. Each DSF
codec layer uses a $3\times3\times3$ causal convolution, as do the reference
causal VAE families. On the tiny $8\times8$ DSF latent grid, reverse dependency
tracing for a central late output pixel reaches all 64 cells at each of its
23--24 visible latent lags. The reproducible trace and pinned source revisions
are written by `scripts/analyze_temporal_receptive_field.py` to
`output/codec_temporal_receptive_field.json`.

## M6 v2 denoiser decision

The first M6 draft inherited alternating spatial and temporal attention from
the raw-pixel teaching baseline. Before any M6 training result existed, this was
replaced by a reference-family design: flattened full spatiotemporal
self-attention, axis-split 3D RoPE, and a block-AR prefix mask. Wan's public
denoiser likewise flattens the video grid, applies global attention by default,
and splits RoPE over temporal, height, and width axes; CogVideoX describes 3D
full attention. At `f8t4d16`, M6 has only 384 tokens in its maximum six-latent
window, so the raw-pixel motivation for factorisation no longer outweighs its
inductive limitation. The prefix mask is the deliberate difference: history
queries cannot read target tokens, while target queries can read the entire
clean history and bidirectional target block.
