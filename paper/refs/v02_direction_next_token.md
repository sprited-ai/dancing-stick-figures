# v0.2 direction — "a next-token machine over rig and pixels" (Jin+Claudia, 2026-08-24)

The morning's conversation converged on the project's organizing frame. This
note is the canonical record; treat it as the north star for v0.2 decisions.

## The five ideas, in the order Jin built them

1. **Identity: we are an autoregressive model.** The differentiators are the
   things only AR can do — indefinite rollout, mid-stream prompt switching,
   constant per-block latency/memory. Every current metric is a 5-second
   clip metric and captures none of them. AR-native evaluation and AR-native
   demos (live prompt-switched infinite character) should lead.

2. **ARDY exists.** A proven AR motion engine with the semantic competence
   our pixel model lacks (every alignment lever we tried sits at chance).
   Division of labor: ARDY supplies semantically correct rig streams; our
   model renders pixels conditioned on rig. The verified projection chain
   (npz → figure frame → bone_scale → camera → 2D screen rig) makes the
   interface exact. **rig-drop training** (probabilistically freeze rig
   tokens as given conditions, cfg-drop pattern) gives ONE checkpoint both
   modes: self-generate rig (today's v9) or follow a supplied rig (M-cond).

3. **Big picture: one next-token machine.** v9 already predicts mixed-modal
   blocks (7 rig tokens + 448 pixel tokens jointly). Conditioning = prefix;
   what is given vs generated is a training-time masking policy, not
   architecture. Ladder: v9 mixed blocks → rig-drop prefix flexibility →
   ARDY as external prefix engine → absorb motion tokens into the same
   machine. Each rung is independently valuable.

4. **The missing metric = perplexity's analog.** Teacher-force GT history,
   generate the next block, measure divergence to the DATA's actual next
   block. Paired, prompt-conditioned, model-agnostic — the common judge the
   v8-vs-v9 comparison lacked, and it captures semantics for free (you must
   actually squat to predict a squat's next block). Design requirements:
   best-of-N (the future is multimodal), fg-weighted pixel distance,
   divergence-vs-horizon curve (also measures the exposure gap
   teacher-forced vs free-running), and a REAL-data floor (same-prompt real
   pairs) for scale. `val_continuation` is the latent-space version already
   logged.

5. **Three-layer metric stack** once SRE (pixels→cskel27 regressor,
   supervised on the free rig cache) exists:
   latent flow-NLL → pixel next-block divergence → rig-space divergence
   (per-joint, interpretable, appearance-robust; also validates v9's
   self-reported rig). This stack is a v0.2 selling point: "video-model
   perplexity in three spaces."

## Standing state feeding into this

- v9.3 render coupling: mechanism PROVEN (coupled arm binds rig to pixels,
  on-figure .68→.93 at 10k = what uncoupled needs 100k for) but weight 2.0
  taxes pixels (TVR .721). Needs weight sweep (0.25/0.5/1.0, + coupling
  warmup arm) before any flagship claim. Jin prefers v9.3 as the flagship
  if the sweep lands.
- Flagship release plan: sweep → winner recipe 100k × 2 seeds → HF model
  card ("reference flagship of the testbed", honest limits: alignment open).
- Semantic alignment: every solo-model lever rejected (t5-base, rig-loss
  strength, budget incl. 300k, syntactic captions). ARDY conditioning (idea
  2) is the only strong open path; semantically rich captions remain
  untested.

## Future study (Jin, deferred by design)

- **ARDY feedback cycle as an OOD probe**: feed our model's generated
  motion (rig, via the exact projection chain inverted or directly in
  joint space) back into ARDY as history and observe its continuation.
  If ARDY continues naturally, our motion is in-distribution for the
  motion prior; degenerate continuations flag OOD output. A free
  realism/OOD detector (no training), and potentially a training signal
  later (cycle-consistency). Explicitly future work — not in the current
  cycle.

## Claudia's assessment (recorded at Jin's request)

- **Best idea of the day: #4 (next-block divergence).** Highest
  value-to-cost in the whole backlog — no new model needed, one day of eval
  work, and it retroactively scores every checkpoint we have. Prior art
  exists (action-conditioned video prediction / world-model literature
  evaluates exactly this way), so frame it as "perplexity discipline brought
  to a traced testbed", not an invention claim.
- **#2 (ARDY division of labor) is the right product path**, and honest
  framing matters: it makes our model a conditioned renderer, which
  DISSOLVES the alignment problem rather than solving it. The solo-model
  alignment question stays scientifically open and is worth keeping as a
  tracked benchmark axis precisely because everything failed — that null
  result is publishable signal.
- **#1/#3 are the right north star with a scope hazard.** "Absorb ARDY into
  one machine" is a research program, not a milestone. The ladder framing is
  the protection: ship each rung; never let rung 4 block rung 1.
- **On v9.3 as flagship: sympathetic but unproven.** My caution: coupling's
  demonstrated benefit is SPEED of rig-pixel binding, but uncoupled v9.0
  reaches the same binding by 100k anyway — and the flagship IS a 100k run.
  The sweep must answer "does coupling still buy anything at the operating
  budget?" If not, flagship = v9.0 with rig-drop added, and that is not a
  loss. Also coupled costs 1.6-1.8x per step (decode in the loop).
- **Sequencing I recommend**: ① next-block divergence metric (unblocks all
  comparisons) → ② v9.3 weight sweep scored by it → ③ rig-drop + ARDY
  prefix pilot (the semantic unlock) → ④ flagship 100k×2 + release → ⑤ SRE
  + rig-space layer. arXiv decision for the v0.1 paper stays independent
  and should not wait on any of this.
