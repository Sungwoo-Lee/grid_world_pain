# Diagnosis findings — src/algorithms/dreamer_srl/agent.py (2026-07-23)

Reviewer scope: agent.py (world model, actor, critic modules, imagination machinery), ~2141 lines.
Method: full read of agent.py; end-to-end trace of every suspect through train.py (one_train_step),
dreamer_srl_main.py (acting/reset paths), utils.py (initializers), and a line-level diff against the
vendored upstream at `vendor/sheeprl/sheeprl/algos/dreamer_v3/agent.py` and `.../dreamer_v3.py`
(the exact sheeprl@33b6366 the port cites). No code was modified.

Verdict up front: **no P0 or P1 findings**. Three P2 findings (one confirmed spec divergence in
initialization, two latent/quality hazards). All previously-fixed bugs touching agent.py are intact.

---

## Findings

### F1 [P2] agent.py:636-640, 659-663 (init applied at 699-714) — transition/representation output linears use the wrong Hafner init distribution

**Claim.** The RSSM's prior ("transition") and posterior ("representation") output linear layers
(`transition_out`, `repr_out`) are initialized with the truncated-normal `init_weights`, but upstream
sheeprl — with `hafner_initialization: True`, the default and the config this port targets — *overrides*
exactly those two layers with `uniform_init_weights(1.0)`:

    vendor/sheeprl/sheeprl/algos/dreamer_v3/agent.py:1173-1174
        rssm.transition_model.model[-1].apply(uniform_init_weights(1.0))
        rssm.representation_model.model[-1].apply(uniform_init_weights(1.0))

The port's RSSM.__init__ applies `init_weights` (trunc-normal, agent.py:704-706 for `transition_out`,
712-714 for `repr_out`) and never applies the uniform override. Every OTHER Hafner override is
faithfully reproduced (actor head 1.0 at agent.py:1470-1473; decoder head 1.0 at 1271-1274; continue
head 1.0 via FullMLPHead zero_init_output=False; reward/critic 0.0 via zero_init_output=True), so
these two layers are the only omission.

**Failure scenario.** None catastrophic. Both distributions have the *same variance* — the Hafner
constant 0.87962566103423978 is precisely the std correction for +/-2-sigma truncation, so effective
trunc-normal std = sqrt(1/denom) = uniform std sqrt(3*scale)/sqrt(3). The divergence is
distribution-shape only (bounded uniform vs unbounded-tail-clipped normal). Consequences: (a) any
bit-parity test against sheeprl on RSSM init will fail; (b) the initial prior/posterior logit
distribution differs slightly from the DreamerV3 spec at step 0. Training-quality impact ~nil.

**Evidence.** Direct diff of agent.py:690-714 against vendor agent.py:1170-1180.

**Fix direction.** In RSSM.__init__, initialize `transition_out.kernel` and `repr_out.kernel` with
`uniform_init_weights(1.0, I, O, k)` instead of `init_weights(I, O, k)`.

---

### F2 [P2] agent.py:1777-1782, 1816-1820 — `WorldModel.imagine()` returns log_probs/entropies computed on non-detached latents; dead outputs today, gradient trap if ever consumed

**Claim.** Upstream sheeprl always calls the actor on **detached** imagined latents
(`actor(imagined_latent_state.detach())`, dreamer_v3.py:220, 241, and `actor(imagined_trajectories.detach())`
at L281 for the loss). In `imagine()`, the actor is called on the *live* `init_latent` / `new_latent`
(agent.py:1779, 1817), and the returned `imagined_log_probs` / `imagined_entropies` therefore sit in
a graph where gradients flow backward through the whole imagined trajectory (via the straight-through
soft path of earlier actions) and into the world model.

**Failure scenario.** Currently none: train.py calls `imagine()` outside any `value_and_grad`
(train.py:833), stop-gradients `imagined_latents` / `imagined_actions` (train.py:905-906), and
*recomputes* log_probs + entropy inside `actor_loss_fn` via `forward_logits(sg_latents)`
(train.py:922-930). So `imagined_log_probs` and `imagined_entropies` are dead outputs — no caller in
train.py, eval.py, or dreamer_srl_main.py consumes them. The trap: a future caller (e.g., an NMN hook
or a diagnostics probe reused inside a loss) that consumes these outputs inside a grad context gets
REINFORCE gradients that leak through the trajectory — a silent divergence from sheeprl S7 semantics.

**Evidence.** agent.py:1777-1782 and 1816-1820 (no stop_gradient on the actor's latent input);
train.py:833/905-936 (dead-output confirmation); vendor dreamer_v3.py:220/241/281 (detach in upstream).

**Fix direction.** Either apply `jax.lax.stop_gradient` to the latent fed to `actor(...)` inside
`imagine()` (matching sheeprl's `.detach()`), or drop the log_probs/entropies outputs entirely and
document that losses must recompute via `forward_logits`.

---

### F3 [P2] agent.py:220-363, 1302-1380, 1944 — dead classes `RewardHead`, `CriticHead`, `ContinueHead` and stale type annotations misdescribe the live architecture

**Claim.** `build_agent` constructs the reward model, continue model, critic, and target critic from
`FullMLPHead` (agent.py:2069-2126); the single-layer `RewardHead` / `CriticHead` (agent.py:220-363)
and the standalone `ContinueHead` (agent.py:1302-1380) are never instantiated anywhere in the live
stack (grepped src/ + tests/). Yet `build_agent`'s signature/annotation reads
`Tuple["WorldModel", "Actor", CriticHead, CriticHead]` (agent.py:1944) and `WorldModel.__init__`
annotates `reward_model: RewardHead`, `continue_model: "ContinueHead"` (agent.py:1616-1617).

**Failure scenario.** Not a runtime bug (duck typing). Hazard is reviewer/maintainer confusion: a
future change "fixing" RewardHead (e.g., its zero-init) silently changes nothing, while the live
FullMLPHead path is missed; checkpoint-shape debugging against the annotated classes misleads.

**Evidence.** grep for constructors across src/algorithms/dreamer_srl and tests — only FullMLPHead,
MLPEncoder, MLPDecoder, RSSM, Actor, WorldModel are built in the live path.

**Fix direction.** Delete (or clearly mark test-only) the three dead classes and correct the
annotations to FullMLPHead.

---

## Fixed-bug regression check (agent.py parts)

- **Recon loss half-weighting + missing symlog** — NO regression. `MLPDecoder.__call__`
  (agent.py:1276-1293) returns the raw symlog-space prediction (docstring updated per WP-SRL P3);
  `observe()` returns it untransformed (agent.py:1717); train.py wraps it in
  `SymlogDistribution(..., dims=1)` (train.py:720) with no extra symlog anywhere in agent.py.
- **REINFORCE action-resampling (v1)** — NO regression. `Actor.forward_logits` (agent.py:1487-1512)
  exists, applies unimix, and is the actor-loss path in train.py:922-927, which uses stop-gradient'd
  *rollout* actions (`sg_imagined_actions`), not a fresh resample. No PRNG consumed in the loss.
- **Persistent-compilation fixes** — NO regression in agent.py's parts. `get_initial_states(batch_size:int)`
  is constant-shape (agent.py:931-969); dreamer_srl_main.py's fixed-width masked reset calls it with
  `num_envs` only on the hot path (main:254); `dynamic()` signature is shape-stable.
- (Gradient clipping, replay write-head, episode-logging fixes live in train.py/buffers.py/main —
  outside this unit, not re-checked here.)

## Reviewed but clean (verified end-to-end against vendor/sheeprl)

- **LayerNormGRUCell** (agent.py:46-164): concat order `[hx, x]`, chunk order (reset, cand, update),
  reset-inside-tanh, `sigmoid(update - 1)`, bias=False + eps=1e-3 wiring — all match vendor
  models.py forward exactly.
- **S4 three-quantity reset in `dynamic`** (agent.py:1037-1071): action zeroing, recurrent-state mask,
  posterior reshape-before-mask, arithmetic-mask form — 1:1 with vendor agent.py:426-431.
- **RSSM prior/posterior wiring** (agent.py:720-813): `_transition` = one-hidden MLP ->
  unimix -> sample; `_representation` = cat(hx, embed) -> one-hidden MLP -> unimix -> sample; both
  return post-unimix logits (the correct KL-balancing inputs; the KL stop-gradient routing lives in
  loss.py's reconstruction_loss, another unit).
- **Straight-through discrete latents** (agent.py:863-925): Gumbel-max forward + softmax backward is
  distribution- and gradient-equivalent to torch `OneHotCategoricalStraightThrough.rsample()`
  (`sample + (probs - probs.detach())`); softmax of post-unimix logits reproduces the mixed probs
  exactly. Mode path (argmax one-hot, no PRNG) matches `dist.mode` and get_initial_states'
  deterministic contract.
- **`get_initial_states`** (agent.py:931-969): tanh(learnable) + mode posterior, no PRNG — matches
  vendor agent.py:391-394; learnable param gradient flows via the is_first mask as upstream.
- **`observe()` carried initial state** (agent.py:1675): uses `get_initial_states(B)` where sheeprl
  uses zeros (dreamer_v3.py:108) — equivalent because train.py force-sets is_first[0]=1 (S1,
  train.py:690), making the reset overwrite the carried value with zero gradient leak. Documented
  caller dependency; not a bug.
- **`action_shift`** (agent.py:171-213) + S2 application in train.py:696 — matches dreamer_v3.py:104.
- **Sequence-first axes**: obs/actions/is_first are [T, B, ...] throughout observe; `terminated`
  reshape to [BT, 1] in train.py:873 is T-major, consistent with the [T,B]->[BT] latent flatten
  (train.py:827-830). No axis confusion found.
- **Imagination machinery** (agent.py:1732-1832): prior-only step = recurrent MLP + GRU +
  `_transition(sample=True)`, identical to vendor `RSSM.imagination`; H+1 latents/actions layout and
  latent = cat(stoch, recurrent) ordering match; train.py's start-state flattening + stop_gradient
  (train.py:820-830) matches `posteriors.detach().reshape(1, -1, ...)`.
- **Actor** (agent.py:1389-1564): body/head structure, unimix on head logits, Hafner init (body
  trunc-normal, head uniform 1.0), ST sampling, log_prob on sg(action) — match vendor Actor discrete
  branch (L829-838) and init (L1171). Entropy uses `log(probs + 1e-8)` instead of log_softmax —
  bias is O(1e-6) relative given the unimix probability floor (unimix/A); negligible, noted only.
- **Continue/discount semantics**: ContinueHead(FullMLPHead) outputs raw Bernoulli logit; S10
  target `1 - terminated` (no gamma) at train.py:725; imagined continues via `.mode` + S5
  true-continue splice inputs (train.py:867-882) — all match dreamer_v3.py:167-168, 246-254.
- **Actor/critic target inputs**: lambda-return bootstrap and advantage baseline use the LIVE critic
  (train.py:862-865, matching dreamer_v3.py:244); target critic reserved for the two-term critic
  loss on detached latents[:-1] (train.py:977-996, matching L307-315); tau=1.0 hard copy on first
  gradient step confirmed in dreamer_srl_main.py:1154-1156 and 1867-1871.
- **Initializers** (utils.py:115-179): trunc-normal with full-precision Hafner constant and +/-2-sigma
  truncation; uniform with sqrt(3*scale) limit; fan-avg symmetric so Flax [in,out] kernel orientation
  is immaterial. Bias defaults (zeros) match sheeprl's explicit bias zeroing.
- **Zero-init discipline** (cascade fix #27): reward + critic output linears zero-init with runtime
  assertion at build (agent.py:2128-2139).
- **dtype boundaries**: whole stack is float32; no bfloat16 mixing anywhere in agent.py.
- **Acting path** (dreamer_srl_main.py:304-352): dynamic-based player step with is_first reset,
  prev-action one-hot carry, sampled (non-greedy) actions — consistent with PlayerDV3 semantics;
  eval path uses deterministic `argmax(forward_logits)` = distribution mode, matching sheeprl greedy.

Known-bugs context honored: no OPEN rows re-reported (checkpoint momentum drop, eval estimator mix,
offline WM smoke test, etc. are acknowledged as owned elsewhere).
