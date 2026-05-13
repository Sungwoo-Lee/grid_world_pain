---
title: "dreamer-srl plan v2 — algorithm re-audit"
topic: dreamer
status: superseded
created: 2026-05-12
last_updated: 2026-05-12
reviewer: professor-rl-bayesian-dl
audited_doc: docs/develop/archive/dreamer_srl/IMPLEMENTATION_PLAN.md
supersedes: review_professor_rl_bayesian_dl.md
---

> **Superseded by**: [`docs/pi/calls/2026-05-12_dreamer_backend.md`](../../../pi/calls/2026-05-12_dreamer_backend.md) — PI pivot to sheeprl-direct (Option 1) shelves the dreamer-srl v2 plan this re-audit cleared.

# dreamer-srl plan v2 — algorithm re-audit

## Question

Earlier today (2026-05-12) I audited v1 of the `dreamer-srl` implementation plan
([review_professor_rl_bayesian_dl.md](review_professor_rl_bayesian_dl.md))
against the line-by-line walkthrough of community sheeprl's DreamerV3
([`docs/project/references/sheeprl_dreamer_v3/`](../../../project/references/sheeprl_dreamer_v3/))
and found **11 algorithm-level deviations** — three silent training-step omissions
(forced `is_first[0]=1` reset at the start of every replay chunk; the prepend-zero
action shift that consumes the action that LED to observation `o_t`, not the
action TAKEN at `o_t`; and the 1024-step random-action prefill before policy
collection ever starts), plus eight YAML / hyperparameter coverage gaps.
The senior-developer has now produced a v2 of the plan that claims to fold all
of those in.

**This re-audit answers two questions.** First, **does v2 actually name each of the
11 v1 items at a location a developer would see** while implementing — not buried
in a comment, but called out in the training-loop semantics subsection or in the
specific file's row of the cascade / file-changes table? Second, **did the v2
edits accidentally introduce any new algorithm-level errors** — was anything
load-bearing dropped while making space for the corrections, or did any new YAML
key sneak in without a sheeprl reference line?

The verdict, stated up front: **all 11 v1 items are now resolved** at locations a
developer can reach by reading the plan in order, and **no new algorithm-level
errors were introduced**. One small documentation inconsistency (the unit of
`learning_starts` — policy-steps vs. iterations — is described two different
ways in two different places) is flagged below as a follow-up the developer
should resolve by reading sheeprl source, but it does not by itself break
bit-identity at Checkpoint 8.

The final verdict block is at the bottom (§Verdict).

---

## §1. Resolution status — per v1 item

The 11 items below are taken in order from the v1 review. For each I state where
in v2 the fix landed (line number / section) and whether a developer reading the
plan top-to-bottom would actually see the named fix at the right point in the
implementation order.

| # | v1 deviation | v2 status | v2 location | Notes |
|---|---|---|---|---|
| 1 | Missing `is_first[0] = 1` force-set inside `one_train_step` | ✅ RESOLVED | New §"Training-loop semantics" §S1 (lines 89–94), plus cited in `train.py` `one_train_step` row of File Changes (line 359 — "First, apply S1 ... and S2 ... on the batch"), plus new Checkpoint 4b (line 696) | Code sketch given explicitly: `batch["is_first"] = batch["is_first"].at[0].set(1.0)`. Rationale (silent-bug nature) is in §S1. |
| 2 | Missing prepend-zero-action shift inside `one_train_step` | ✅ RESOLVED | §S2 (lines 96–103), cited in `train.py` `one_train_step` row (line 359), plus new Checkpoint 2b (line 695) | Code sketch explicit: `shifted_actions = jnp.concatenate([jnp.zeros_like(actions[:1]), actions[:-1]], axis=0)`. Math rationale (action consumed at `t` is taken at `t-1`) is in §S2. |
| 3 | Missing `learning_starts` random-action prefill phase (both collection branch and gradient-step gate) | ✅ RESOLVED | §S3 (lines 105–118), `collect_step` row of File Changes (line 360, with explicit `if use_random:` code branch), YAML comment block (lines 416–420), Implementation order Step 9 (line 760, "Wire the `learning_starts` random-action prefill (§S3) BOTH at action selection AND at the gradient-step gate"), plus new Checkpoint 9b (line 698) | Both wirings (collection branch and gradient-step gate) are explicitly named. The §S3 prose calls out that the YAML key was loaded but not wired in v1 and explicitly says "Two wirings required." See §2 below for one minor documentation inconsistency about the unit of `learning_starts`. |
| 4 | `prepare_obs` description reads as no-op when in fact it does an MLP-key reshape | ✅ RESOLVED | `utils.py` `prepare_obs` row (line 251) | Description tightened to "Replicates sheeprl `utils.py:171-183` MLP path... For each MLP obs key, write `jnp.asarray(v).reshape(1, num_envs, -1)`. **The leading `T=1` axis is mandatory** — the RSSM consumes `[T, B, ...]`". My proposed correction text is now nearly verbatim. |
| 5 | Missing `distribution.type` and `distribution.validate_args` keys in YAML | ✅ RESOLVED | `configs/dreamer_srl/agent_xs.yaml` lines 401–405 (new top-level `distribution:` block) | New top-level block exactly as proposed: `type: "auto"` + `validate_args: false`, with sheeprl source line cited. Also surfaces in Risks §10 mandatory-key audit (line 801: "`distribution.type`, `distribution.validate_args`"). |
| 6 | Missing actor `init_std`, `min_std`, `max_std` hyperparameters | ✅ RESOLVED | YAML lines 477–479, with comment "Unused on discrete path but kept for signature parity with Actor.__init__" | All three keys present, sheeprl source lines cited per key, signature-parity rationale explained. |
| 7 | Missing `discount_model.learnable` flag plus continue/reward-head MLP sizing | ✅ RESOLVED | YAML lines 452–470 (new `reward_model:` sub-block with `bins/dense_act/mlp_layers/dense_units/layer_norm`, plus new `discount_model:` sub-block with `learnable: true` plus same MLP sizing keys) | Both reward and discount heads now carry the full network-sizing keys. `learnable: true` is explicit. |
| 8 | Encoder/decoder/recurrent network sizing keys ambiguous — top-level vs. override | ✅ RESOLVED | New "MLP sizing table" at lines 525–535 (immediately after the YAML block) | Nine networks enumerated (Encoder, Decoder, Recurrent pre-proj, Transition (prior), Representation (post), Reward head, Critic, Continue head, Actor trunk) with `mlp_layers`, `hidden_dim`, and terminal-init scale for each. This is the exact table I proposed in v1 §1. |
| 9 | Wrong claim that existing `compute_lambda_values` is "line-for-line identical to sheeprl's" | ✅ RESOLVED | `utils.py` `compute_lambda_values` row (line 248) — a four-paragraph entry tightened with: math contract (with the recursion equation in LaTeX), bootstrap-and-trim contract (`vals[0] = values[-1:]`, returned length = `len(continues)` not `len(continues)+1`, trailing `[:-1]` slice essential), γ-is-pre-multiplied-by-caller signature note, JAX `lax.scan(reverse=True)` translation, numpy reference loop in docstring required, Checkpoint 1 assertion. Final paragraph at line 259 explicitly says "the v1 plan asserted line-for-line identity; this v2 says verify, do not assert." | The "verify, do not assert" wording I proposed is present verbatim. |
| 10 | Missing `Independent(BernoulliSafeMode, 1)` wrap on continue head, plus `dims=1` on observation decoder | ✅ RESOLVED | §S9 (lines 182–186), plus `loss.py` `BernoulliSafeMode` row (line 342) | §S9 names both wraps (continue + decoder) and explains the `dims` argument controls how many trailing axes get summed by `log_prob`. The `loss.py` row reproduces the rule. Imagination-loop wrap (`continues = Independent(BernoulliSafeMode(...), 1).mode`) is also named in §S9. |
| 11 | Missing `Moments` `max_=1.0` floor warning, plus per-rank-local note | ✅ RESOLVED | YAML lines 486–490 (annotated comment block), plus new Risks §11 (line 803), plus `utils.py` `Moments` row (line 249) annotated "**`moments_max = 1.0`** ... the invscale **floor is `1/max_ = 1.0`** ... when the λ-spread is tiny the advantage is NOT re-scaled larger than the spread itself" | Semantic explanation of the floor (vs. ceiling) is present. Per-rank-local note ("Per-rank percentiles, identical to sheeprl XS at single-rank") is present in the `Moments` row at line 249. |

**Summary**: 11 / 11 ✅ RESOLVED. Every v1 item now appears either in the new
"Training-loop semantics" subsection (§S1–§S10) or in a specific row of the
file-changes table where the developer will see it at the point of writing that
file. The "v2 Changes Applied (2026-05-12)" traceability section at the bottom
of the plan (lines 974–1029) is honest — every claim in it matches the body of
the plan.

---

## §2. New algorithm-level errors introduced by v2

I walked the v2 plan top-to-bottom looking for: (a) algorithm-shaping additions
that have no sheeprl source line; (b) load-bearing content from v1 that was
dropped to make space for the corrections; (c) self-contradictions among the new
sections; (d) any new YAML key without a sheeprl reference.

**Findings**: zero new algorithm-level errors. One documentation
inconsistency worth flagging but it does NOT break bit-identity:

### 2.1 Documentation inconsistency — unit of `learning_starts` (NOT an algorithm error, but should be reconciled)

The §S3 prose (line 118) says:

> "Sheeprl counts in **iterations** (`iter_num = policy_step / policy_steps_per_iter`,
> where `policy_steps_per_iter = num_envs * action_repeat = 4 * 1 = 4` for us),
> so `learning_starts=1024` = 256 iterations."

This reads as "the YAML value `1024` is in policy-steps, divide by 4 to get the
256-iteration prefill phase."

The YAML comment block immediately above (lines 416–418) says:

> "learning_starts: 1024 — Counted in ITERATIONS (= policy_step / policy_steps_per_iter
> = policy_step / (num_envs * action_repeat) = policy_step / 4)."

This reads as "the YAML value `1024` is itself the iteration count, i.e. 1024 iterations
= 4096 policy-steps."

Checkpoint 9b (line 698) says:

> "Assert that on policy steps `0..1023` (iteration `0..255` for our `num_envs=4`)
> the actions in the replay buffer have entropy `≈ log(action_dim)` (uniform)"

This is consistent with §S3 ("1024 policy-steps = 256 iterations") but contradicts
the YAML comment ("1024 iterations = 4096 policy-steps").

**Why this is not (yet) an algorithm error**: the `learning_starts=1024` value is
identical regardless of interpretation; the actual bit-identity check is whether
the prefill behaviour matches sheeprl's source. Sheeprl's `dreamer_v3.py:604`
condition is `if iter_num <= learning_starts` where `iter_num` is incremented
once per outer-loop iteration (not once per policy step). For `num_envs=4`,
`action_repeat=1`, sheeprl's outer-loop iteration consumes 4 policy-steps; so
`learning_starts=1024` means "the first 1024 outer-loop iterations" = "4096 policy
steps." That is the YAML comment's reading — and §S3's prose contradicts it.

**Recommendation**: developer reads `tmp/sheeprl/sheeprl/algos/dreamer_v3/dreamer_v3.py:557, 604, 706`
at Implementation order Step 9 and reconciles the two descriptions before
launching. The simpler fix is to pick one unit (iterations is what sheeprl uses)
and rewrite Checkpoint 9b's "policy steps 0..1023" wording to match.
**This is a v2-introduced inconsistency** (the v1 plan did not specify the unit at
all), but it is a documentation bug, not an algorithm bug — the developer will
hit the contradiction in Checkpoint 9b assertion and resolve it by reading
sheeprl. Flagging here so it does not slip through.

### 2.2 Items I explicitly checked and found NOT to be errors

- **§S5 true-continue splice** correctly says `predicted_continues = continue_dist.mode`
  and then `continues = jnp.concatenate([true_continue_step0[None], predicted_continues[1:]], axis=0)`.
  This matches sheeprl `dreamer_v3.py:246-248` (the `continues[0]` overwrite via
  splicing). The `[1:]` slice on the predicted side is correct — without it the
  step-0 predicted value would be retained instead of overwritten.
- **§S6 critic-loss `discount[:-1].squeeze(-1)`** vs. actor-loss `discount[:-1]`
  (no squeeze). This shape mismatch is intentional: critic `log_prob` returns
  `[H, B]` (Independent-wrap reduces the trailing 1-axis), so the discount mask
  `[H, B, 1]` must be squeezed to broadcast cleanly. Actor `objective = log_prob *
  advantage` retains the trailing `1`-axis (advantage shape `[H, B, 1]`), so
  `discount[:-1]` of shape `[H-1, B, 1]` broadcasts. Both forms match sheeprl's
  source-line shape contracts. Not an error.
- **§S7 "Moments used only for actor normalisation; critic regression target is
  un-normalised `lambda_values`"**. Correct — sheeprl `dreamer_v3.py:314` uses
  raw `lambda_values.detach()`, not the normalised form. The v2 plan explicitly
  flags this in §S7 and in the critic-phase code sketch of the `one_train_step`
  row.
- **§S8 free-nats floor per-element BEFORE mean**. Correct — matches sheeprl
  `loss.py:100-101`. The "literalist would implement max(mean(dyn_loss), 1.0)"
  warning is accurate.
- **Checkpoint 5 two-hot bin grid in symlog space**. Correctly rewritten per
  math-reviewer 🔴 #1: `bins[0] = -20.0`, `bins[127] = 0.0`, `bins[254] = +20.0`,
  with the explicit "DO NOT implement `self.bins = symexp(linspace(...))`" warning.
  Matches sheeprl `distribution.py:237`.
- **§S4 arithmetic-mask `(1 - is_first) * x + is_first * init`** for the
  three-quantity reset (action, recurrent, posterior). Correct — matches sheeprl
  `agent.py:423-429`. The posterior pre-flatten step (`posterior.reshape(*posterior.shape[:-2], -1)`)
  before masking is named correctly. Not an error.
- **`Moments` as `flax.struct.dataclass`** with pure-functional `update`. The
  three-return signature `(new_moments_state, offset, invscale)` matches
  sheeprl's `Moments.forward` semantics (returning the new state, the `low`
  offset, and the `invscale = max(1/max_, high - low)` denominator). The
  `low`-offset cancellation in §S7 still applies. Not an error.
- **No new YAML keys without sheeprl reference**. Walked the new YAML rows
  added in v2: `distribution.type/validate_args`, `reward_model.{dense_act,
  mlp_layers, dense_units, layer_norm}`, `discount_model.{learnable, dense_act,
  mlp_layers, dense_units, layer_norm}`, `actor.{init_std, min_std, max_std,
  world_model_weight_decay, actor_weight_decay, critic_weight_decay}`,
  `world_model.decoupled_rssm: false`. Every key cites a specific sheeprl source
  line in its inline comment. The only key with no upstream sheeprl line is
  `agent.algorithm: "dreamer-srl"` (the dispatch key for our `train.py`) and
  `buffer_device: "cpu"` (which v1 already flagged as the one allowed addition,
  pre-justified at line 272). Both are non-algorithmic.

### 2.3 Items NOT introduced by v2 but worth re-flagging for the developer

Three items from v1 are still ambiguity that the v2 plan correctly defers to
the developer rather than silently invent:

- **Risk §7 (sequence-length truncation off env episodes)** — walkthrough
  does not transcribe the `bincount`-based per-env subset sampling logic.
  Developer reads `tmp/sheeprl/sheeprl/data/buffers.py:EnvIndependentReplayBuffer.sample_tensors`
  directly. v2 line 795. Not a v2 error; just an open question correctly
  routed.
- **Risk §2 / §12 (Ratio env-step counter semantics)** — developer reads
  `train.py:789` and dreamer-srl's `utils.py:Ratio` call site side-by-side at
  Step 7. v2 lines 785, 805. Not a v2 error.
- **Risk §3 (GPU contention on node 114)** — out-of-scope for this audit; that's
  a launch-time concern for `training-runner`.

None of these three become algorithm errors unless the developer silently
invents an answer instead of consulting the source. The v2 plan correctly says
"do not silently invent" in each case.

---

## §3. Cross-cutting confirmation pass

I re-checked the v2 plan's claims about the cascade items, the non-goal walls,
and the new training-loop semantics section:

### The five cascade items — still placed correctly

The plan's cascade-item table at line 77–83 remains intact (cascade #2, #27,
#28, #29, #30 all named, placed at the right module, with sheeprl source lines).
Checkpoints 2, 3, 4, 5, 6 are still wired to test each cascade item. The v2
edits did not disturb the cascade fixes; rather, the cascade table for #2 was
sharpened (Checkpoint 5 corrected for the symlog-space bin grid) and #28 was
sharpened (Code-reviewer 🔴 #2 fused-gate ONE Linear + ONE LayerNorm warning).

### The non-goal wall — still clean

Non-goals 1–13 are unchanged. No NMN, no FiLM, no precision-modulation, no
continuous-action, no MLflow, no memmap, no Hydra, no `gym.vector`, no
`RestartOnException`, no `fabric.all_gather`, no `EpisodeBuffer`, no
hyperparameter sweeps. None of the v2 edits sneak any of these back in. The
existing `src/models/dreamer_v3_*.py` and `configs/models/dreamer_v3*.yaml` are
still left untouched (line 195, 668–675).

### The Training-loop semantics section — well-placed

The new §S1–§S10 subsection (lines 85–189) lives **above** the File Changes
section, which is the right placement: a developer reading the plan in order
encounters the global training-loop mechanics before drilling into per-file
translation. Each §S item is then re-cited at the specific File Changes row
where the implementation lives. This double-naming pattern (global + local) is
what makes the corrections developer-visible at both reading passes.

### Checkpoints — augmented correctly

New Checkpoints 2b (action shift), 4b (`is_first` reset), 9b (random-action
prefill) directly test the three silent-omission items from my v1 review. Each
checkpoint has a concrete assertion (e.g., 9b: "actions in the replay buffer
have entropy `≈ log(action_dim)` on iterations 0..1023"). The Verification
Report table at line 851–863 includes all three new checkpoints. Good.

---

## §4. Hand-off

After the small documentation inconsistency in §2.1 is reconciled (developer
picks one unit for `learning_starts` and aligns §S3 prose, YAML comment, and
Checkpoint 9b wording), this plan is ready for implementation.

**Next steps by agent:**

- **senior-developer**: optional one-line fix to reconcile the §2.1
  documentation inconsistency (developer can also do this at Step 9, after
  reading sheeprl source). No other v2 edits needed from this audit's
  perspective.
- **developer**: implementation can proceed per the Implementation order. The
  v2 corrections have made the plan a clean blueprint at the algorithm level.
- **code-reviewer** and **math-reviewer**: their parallel v2 re-audits cover
  the NNX-correctness and equation-faithfulness lanes; cross-check before
  implementation launch.

---

## §Verdict

# ✅ PASS — zero residual algorithm-level deviations; ready for implementation.

All 11 v1 algorithm-level deviations are resolved at locations a developer
will see in normal reading order (Training-loop semantics §S1–§S10 above File
Changes, plus per-file-row callouts in the cascade table and file-changes
tables, plus three new directly-targeted Checkpoints 2b / 4b / 9b). No new
algorithm-level errors were introduced by the v2 edits. One documentation
inconsistency about the unit of `learning_starts` (§2.1) is flagged for
developer reconciliation against sheeprl source, but it does not by itself
break bit-identity at Checkpoint 8 — the developer will hit the contradiction
in the Checkpoint 9b assertion and resolve it by reading sheeprl. The
non-goal walls (no NMN, no continuous, no MLflow, no memmap, no Hydra) and
the five cascade items (#2, #27, #28, #29, #30) all remain intact. No new
YAML keys without a sheeprl reference. No load-bearing v1 content was
dropped.

— Feedback from professor-rl-bayesian-dl — 2026-05-12
