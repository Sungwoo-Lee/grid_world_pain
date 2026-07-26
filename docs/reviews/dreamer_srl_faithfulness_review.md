---
title: "Dreamer-SRL faithfulness re-review (H2a) — residual algorithmic mis-implementation vs sheeprl"
topic: dreamer_srl
status: active
created: 2026-07-27
last_updated: 2026-07-27
---

# Dreamer-SRL Faithfulness Re-Review (H2a): Does Any Residual Mis-Implementation Degrade Learning?

## Verdict (plain language)

This review re-examined our in-house JAX/Flax port of DreamerV3 (`src/algorithms/dreamer_srl/`) against the reference implementation it was ported from (the vendored `sheeprl` library, commit 33b6366), with the standing instruction to distrust prior "fixed" claims and re-read the live code on branch v3.0. The question: is there any remaining place where our implementation computes something *different* from the reference in a way that could plausibly make the agent learn worse?

The answer is **no residual blocker-level (🔴) findings**. Every previously-claimed fix that matters for learning — gradient clipping, the reconstruction-loss weighting, the replay-buffer write-head separation, the prefill semantics, the discount-factor value, the two-term critic loss, the REINFORCE no-resample rule — was independently re-verified as present in the current code, by reading the live source side-by-side with the reference, not by trusting commit messages. The world-model loss, imagination rollout, lambda-return targets, return normalization, actor/critic objectives, target-critic schedule, replay sampling, and random-key threading are all line-faithful ports at our operating point (replay ratio 0.0625, 128 parallel envs, M-size network).

What remains is a set of five moderate (🟡) findings — all either **transient** (a small one-time loss of gradient steps at run start), **dormant** (only fire under configs we don't run), or **out-of-envelope** (only fire at replay-ratio × env-count combinations we don't use) — plus housekeeping nits (🟢). The one open red test (the offline world-model smoke test) turns out to abort on a *data-sufficiency precondition* before any world-model metric is computed, so it is not evidence of a world-model defect. If our DreamerV3 underperforms, the cause is very unlikely to be residual porting error in this code.

## Scope and method

- **Ours (fully read, current v3.0):** `src/algorithms/dreamer_srl/{train.py, loss.py, agent.py, utils.py, buffers.py, dreamer_srl_main.py}` (checkpoint/eval consulted for the re-verified fixes).
- **Reference:** `vendor/sheeprl/sheeprl/algos/dreamer_v3/{dreamer_v3.py, loss.py, utils.py, agent.py}`, `vendor/sheeprl/sheeprl/configs/algo/dreamer_v3.yaml` (+ XS/M overlays), `vendor/sheeprl/sheeprl/data/buffers.py`.
- **Prior work triaged, not re-derived:** the 2026-07-23 diagnosis pass ([findings_dreamer_main.md](diagnosis_20260723/findings_dreamer_main.md), [findings_dreamer_train.md](diagnosis_20260723/findings_dreamer_train.md), [findings_dreamer_loss.md](diagnosis_20260723/findings_dreamer_loss.md), [findings_dreamer_agent.md](diagnosis_20260723/findings_dreamer_agent.md), [findings_dreamer_eval_ckpt.md](diagnosis_20260723/findings_dreamer_eval_ckpt.md)). Its learning-relevant rows are absorbed below with fresh line numbers; its telemetry/logging rows (eval-video WandB step rejection, dead config keys, curriculum config rebinds, episode-row batching) are learning-neutral and stay owned by those docs.
- **Operating point for impact estimates:** replay_ratio = 0.0625, num_envs = 128, seq_len 64 × batch 16 (BT = 1024 imagination starts), horizon 15, M or XS network.

## Re-verification of previously-"fixed" claims (live code, not commits)

| Claim | Verdict | Live evidence |
|---|---|---|
| P1 grad clipping (WM 1000 / actor 100 / critic 100, clip-then-Adam) | **present** | `utils.py:45-55` (`make_optim_tx` = `optax.chain(clip_by_global_norm, adam)`), wired `dreamer_srl_main.py:774-776`; matches sheeprl `dreamer_v3.py:193-197/300-302/320-327` order |
| P3 recon loss full-weight, decoder in symlog space, tol clamp | **present** | `loss.py:83-86` (`-(mode - symlog(x))²`, no 0.5, `tol=1e-8`); wired `train.py:720`; decoder returns raw symlog-space (`agent.py:1276-1293`) — matches sheeprl `distribution.py` SymlogDistribution + `loss.py:61` |
| P6 prefill counted in env steps | **present** | `utils.py:80-82` (`learning_starts_cfg // num_envs`, off-by-one `prefill_steps`), gates `dreamer_srl_main.py:1452` (`iter_num <= learning_starts` random) / `:1890` (`>= train_start_iter`), `ratio_steps = policy_step - prefill_steps*num_envs` `:1900` — line-for-line sheeprl `dreamer_v3.py:508-511, 660-661` at world_size=1 |
| P2 per-env replay buffers, no shared write head | **present** | `EnvIndependentSequentialReplayBuffer` (`buffers.py:718-904`), driver sizing `buffer.size // num_envs` + capacity fail-fast (`dreamer_srl_main.py:789-819`), done-boundary rows routed only to done envs' sub-buffers (`buffers.py:832-839`, driver `:1603`) |
| P4 gamma typo | **fixed** | all `configs/models/dreamer_srl/*.yaml`: `gamma: 0.996996996996997` (= 1 − 1/333, sheeprl `dreamer_v3.yaml`) |
| H5 / b1dd90a episode-boundary buffer leak | **fixed** | two-row done semantics: `reset_data` write (`dreamer_srl_main.py:1594-1603`) then `_reset_terminal_step_data` (`:359-377`, called `:1619`) — matches sheeprl `dreamer_v3.py:639-657` |
| v1 REINFORCE resample bug | **fixed** | actor loss = `sum(sg(a_rollout) · log_softmax(forward_logits(sg(latent))))`, rollout actions threaded from `imagine()` (`train.py:843, 905-948`); no PRNG in the loss — matches sheeprl L273-297 |
| 166f261 optimizer state in checkpoints | **present** | `wm_opt/actor_opt/critic_opt` passed to `_save_checkpoint` (`dreamer_srl_main.py:1785-1787`) |
| b228117 eval uses testing.seed | **present** | `eval_seed = get_mandatory('testing.seed')` (`:662`), passed to both eval passes (`:1810, :1853`) |
| Recompile-storm fixes | **present** | single `@jax.jit` `_scan_grad_steps` with closed-over graphdefs (`:1161-1247`), fixed-width masked player reset (`:251-271`), constant `_G` scan length (`:1264`), constant-shape autoreset key split (`:1642-1643`) |

## Core-algorithm parity audit (fresh line-vs-line)

All verified equivalent to sheeprl@33b6366; cited ours ↔ theirs:

1. **World-model losses.** Two-hot reward head, 255 bins in **symlog space** `linspace(-20,20,255)`, symexp only at mean/mode, symlog target before bin lookup, cross-weight interpolation (`loss.py:134-285` ↔ `sheeprl/utils/distribution.py:224-276`). KL balancing dyn 0.5 (`KL(sg(post)‖prior)`) / rep 0.1 (`KL(post‖sg(prior))`), free-nats 1.0 **per-[T,B]-element before the mean**, over post-unimix logits (`loss.py:556-598` ↔ `sheeprl loss.py:64-79`); continue head `Independent(BernoulliSafeMode,1)` with target `1 − terminated`, no gamma (`loss.py:292-457, train.py:722-725` ↔ `dreamer_v3.py:167-168`); total = `(kl_reg·kl + obs + rew + cont).mean()` (`loss.py:613` ↔ `L80`). Stop-gradient placement identical at every site.
2. **Imagination + actor/critic.** BT = T·B flattened posterior starts, T-major flatten consistent with the `terminated` reshape (`train.py:820-873`); H=15; action sampled inside imagination from detached latents, actions stored per-latent, `[:-1]` alignment (`agent.py:1732-1832, train.py:920-948` ↔ `dreamer_v3.py:202-297`); λ-returns with `continues[1:]·gamma`, γ=0.996997, λ=0.95 (`utils.py:186-228` ↔ `sheeprl utils.py:66-77`, recursion verified identical); §S5 true-continue splice + `cumprod(c·γ)/γ` discount (`train.py:172-225, 401-498` ↔ `L246-260`); moments 5/95 percentile EMA decay 0.99, **max 1.0 via config** → `invscale = max(1, S)` (`utils.py:279-325` + `configs/models/dreamer_srl/*.yaml` ↔ `Moments` + `dreamer_v3.yaml:132-137`); per-term advantage normalization, `sg(action)`+`sg(advantage)` REINFORCE, ent_coef 3e-4 inside the discounted mean (`train.py:505-609` ↔ `L274-297`); two-term critic NLL vs **raw** λ-targets + EMA-target mean, discount-weighted (`train.py:232-329` ↔ `L307-316`); polyak τ=0.02, freq 1, hard copy at grad-step 0, **before** the train step on both legacy and scan paths (`dreamer_srl_main.py:1191-1212, 1966-1975` ↔ `L673-680`); live critic for bootstrap/baseline, target critic only in the loss (`train.py:862-865, 979-982` ↔ `L244/L307-310`).
3. **Replay.** Uniform sampling over per-env ring buffers with the write-head-adjacent window excluded exactly as upstream (`buffers.py:413-441` ↔ `sheeprl buffers.py:419-465`); newest written row **is** reachable (start ≤ `_pos − seq_len`, window ends at `_pos − 1`) — same recency as sheeprl, including the newest partial episode; sequences may straddle episode boundaries by design, handled by in-window `is_first` + §S4 reset (`agent.py:1037-1071`); bincount env allocation + axis-2 concat ↔ `EnvIndependentReplayBuffer.sample`; buffer row semantics (o_t, a_t-from-o_t, r-on-arrival, term_t, is_first_t) + §S2 action shift + §S1 `is_first[0]:=1` all sheeprl-exact (`train.py:689-696` ↔ `L100-104`).
4. **PRNG.** One main loop key; per-iteration splits for player/autoreset/sample; `_scan_grad_steps` carries and returns the advanced key which **replaces** the outer key (`dreamer_srl_main.py:2015-2026`); inside a train step: k_wm → per-T step keys → per-step prior/post keys; k_imag → per-step (k_rssm, k_act). No reuse found anywhere on the training path. Straight-through Gumbel-argmax sampling is distribution- and gradient-equivalent to torch `OneHotCategoricalStraightThrough.rsample`.
5. **MLP-only adaptation.** With `cnn_keys` empty, sheeprl's MLP path uses SymlogDistribution for decoder keys, symlog inputs in the encoder, no /255 scaling — ours matches (`agent.py:1180-1195` ↔ `dreamer_v3.py:98-99, 156-161`). `MSEDistribution` is CNN-only upstream; correctly not ported.

## Findings

| Sev | Ours (file:line) | Reference (file:line) | Issue | Expected learning impact @ rr=0.0625, 128 envs, M/XS |
|---|---|---|---|---|
| 🟡 | `dreamer_srl_main.py:1900-1913`; `utils.py:393-395` | sheeprl `dreamer_v3.py:660-671` (no ready gate — sample raises loudly) | **Ratio debt silently dropped while buffers are not yet sequence-ready.** `ratio()` advances its accumulator before the `ready_to_sample(seq_len)` gate; owed grad steps in the gap are never repaid. At our point: learning_starts=1024 env steps → prefill ends at iter 8, per-env buffers reach seq_len 64 at ~iter 64 → ~56 iters × 8 = **~450 grad steps lost, once, at run start** (0.014% of a 50M-step run's ~3.1M grad steps); effective prefill silently lengthens to ~8k env steps. Sheeprl would have crashed on this config instead (its envelope assumes learning_starts_iters ≥ seq_len). | Negligible (transient); flag because it also *masks* a config inconsistency sheeprl surfaces loudly. Fix: call `ratio()` only when ready, or fail fast when `learning_starts_cfg // num_envs < seq_len`. |
| 🟡 | `dreamer_srl_main.py:1264, 1905-1908` (D-015) | sheeprl `utils.py:259-301` (exact fractional-debt Ratio) | **D-015 quantization re-verified — benign at declared operating points, not in general.** `_G = max(1, int(rr×num_envs))`: 0.0625×128 = 8 exactly and 0.0625×16 = 1 exactly, and the steady-state Ratio return equals `_G` at both, so the scheduler and the scan agree bit-for-bit — the declaration holds. However `int()` **floors** any non-integer product (e.g. 24 envs → 1.5 → 1 = −33% effective replay ratio) and the scan runs `_G` whenever Ratio returns > 0. | None at our operating points (verified, not just declared). Real under-training hazard for any future rr×num_envs ∉ ℕ; observable via `Params/effective_replay_ratio`; exact cadence via `--legacy-grad-loop`. |
| 🟡 | `agent.py:704-706, 712-714` | sheeprl `agent.py:1173-1174` (`uniform_init_weights(1.0)` on `transition_model.model[-1]`, `representation_model.model[-1]`) | **RSSM prior/posterior output linears initialized trunc-normal instead of uniform(1.0)** (the only Hafner-init override not reproduced; all other heads match). Same variance by construction (the 0.8796… constant is the ±2σ truncation correction); distribution shape differs at step 0 only. | ~nil (init-shape only; both zero-mean equal-variance). Worth fixing for bit-parity tests. From prior F1 (agent), re-confirmed live. |
| 🟡 | `dreamer_srl_main.py:1484-1486` | sheeprl env API returns native `terminated` | **`terminated := (termination_reason ≥ 2)` computed every step interacts with the known env latent bug** (`overeating_death=True` sets reason=3 without `done=True`, KNOWN_BUGS). If any config enabled it, mid-episode rows would be written `terminated=1` while the episode continues → continue-head targets and §S5 splice corrupted for those samples. **Dormant**: `configs/environment/default.yaml:183` has `overeating_death: false` and no live experiment config overrides it (only archived ones). | Zero today; would be a real WM-corruption path if overeating death is ever switched on without fixing the env bug first. Guard suggestion: derive `terminated` from `done AND reason ≥ 2`. |
| 🟡 | `scripts/dreamer/dreamer_srl_offline_wm_test.py:735-748` | — | **Open red offline-WM smoke test assessed: fixture/data-sufficiency artifact, not a WM-defect signal.** "Only 36 valid starting states (< 50)" fires in the precondition that requires ≥ max(M/4, 50) rollout timesteps with `h_max` real steps remaining *inside the same episode* — it aborts **before any WM metric is computed**. 36 valid starts means the probe rollout's episodes were short relative to `--horizon-max` (a performance symptom of the probed checkpoint, or too small `--num-real-steps`), not that imagination/decoding is broken. Nit at `:712`: `key, key = jax.random.split(key)` discards one half. | None on training. Re-run with larger `--num-real-steps` (script's own suggestion) or smaller `--horizon-max`; treat the *short episodes themselves* as the finding to explain. |
| 🟢 | `train.py:930`; `agent.py:1562` | sheeprl exact `-Σp·log p` via logits | Entropy uses `log(p + 1e-8)`; with the unimix floor (p ≥ 0.01/n) the deviation is ~1e-6 relative. | Nil. |
| 🟢 | `utils.py:286` (`max_=1e8` default) | sheeprl config `dreamer_v3.yaml:134` (`max: 1.0`) | `moments_update`'s *default* encodes the class default, not the recipe; a bare call gives `max(1e-8, S)` → advantage explosion. Live path safe: driver passes config `moments.max: 1.0` (verified in all dreamer_srl configs). | Nil today; latent trap (prior P2-2, re-confirmed). |
| 🟢 | `utils.py:315-325` | sheeprl `Moments.forward` (3 detaches) | `moments_update` has no internal `stop_gradient`; safe only because the call site sits outside `value_and_grad` and re-wraps offset/invscale (`train.py:941-944`). | Nil today; refactor trap (prior P2-3). |
| 🟢 | `agent.py:1779, 1817` | sheeprl `dreamer_v3.py:220/241` (`.detach()`) | `imagine()` feeds the actor non-detached latents; the returned log_probs/entropies are dead outputs (loss recomputes via `forward_logits(sg(latents))`). Gradient trap only if a future consumer uses them in a grad context. | Nil today (prior F2 agent, re-confirmed dead). |
| 🟢 | `agent.py:220-363, 1302-1380, 1944` | — | Dead classes `RewardHead`/`CriticHead`/`ContinueHead` + stale annotations; live heads are `FullMLPHead`. Maintainer hazard only. | Nil (prior F3 agent). |
| 🟢 | `buffers.py:224-225` | sheeprl `buffers.py:200` (same bug) | Oversize single-`add()` slice mismatch — faithful port of an upstream bug; unreachable (driver always adds length-1 rows). | Nil (prior F3 train). |
| 🟢 | `utils.py:196` | — | `compute_lambda_values` docstring claims `[T+1, B]` values; implementation (correctly, sheeprl-exact) needs length T. Doc-only. | Nil (prior P2-1). |

Learning-neutral driver/telemetry findings (eval-video WandB step rejection, `eval_stats_num_envs` dead key, agent-YAML `env.num_envs` dead key, curriculum stage-swap config rebinds, legacy-logging memory growth, stage-boundary eval attribution) remain catalogued in [findings_dreamer_main.md](diagnosis_20260723/findings_dreamer_main.md) and [findings_dreamer_eval_ckpt.md](diagnosis_20260723/findings_dreamer_eval_ckpt.md) — none affects the gradient path.

## Conventions audit

- Pytree & immutability: ✅ (all state through `nnx.split/merge/update`, `MomentsState` flax struct, functional `.at[].set`)
- JIT recompilation: ✅ (constant `_G` scan length, fixed-width masked resets, constant-shape key splits, graphdefs closed over one `@jax.jit`)
- vmap/batch conventions: ✅ (env axis 0; per-env buffers; no batched `EnvParams`; renderer untouched by trainer)
- PRNG threading: ✅ (main key always advanced, incl. through the scan carry; no sub-key reuse found)
- Sensor/obs sync: ✅ (no sensor changes in this unit; curriculum modality-fingerprint pre-flight intact, `dreamer_srl_main.py:700-752`)
- Config protocol: ✅ (all algo hyperparameters via `get_mandatory`; `moments.max=1.0`, `gamma=1−1/333` verified in every dreamer_srl config)

## Conclusion

**No residual 🔴 findings.** The dreamer_srl algorithm core is a faithful sheeprl@33b6366 port at the current operating point; the residuals are one transient start-of-run grad-step shortfall (~450 steps, once), one verified-benign-but-envelope-limited quantization (D-015), one nil-impact init-distribution divergence, one dormant env-bug interaction, and housekeeping nits. Residual porting error is an unlikely explanation for any Dreamer learning-quality gap; look instead at settings/regime and task-difficulty axes (sibling docs below).

## Follow-up: regime-sensitive semantics (from settings critique)

Two targeted checks handed off by the settings-regime sibling ([[dreamer_srl_settings_regime_critique]]): cases where a *faithful* port could still be semantically wrong for our 27-dim vector / survival-task regime. Both checked against ours AND sheeprl, severity rated in OUR regime.

### F/U-1 — kl_free_nats clamp position: applied to the SUMMED KL (≈1 nat total). Correct; clamp only partially active at measured magnitudes. 🟢

**Ours.** `loss.py:560-571` (`_categorical_kl`): KL is summed over the D=32 classes (`kl_per_cat`, `:570`), then over the S=32 categoricals (`:571`) → one scalar KL per [T, B] element covering the *entire* 32×32 discrete latent. The free-nats floor is applied **after** that double summation, per [T, B] element, before the mean: `dyn_loss = kl_dynamic * jnp.maximum(dyn_loss, kl_free_nats)` at `loss.py:588`, `repr_loss` likewise at `:596`. Reduction order: sum(D) → sum(S) → clamp(1.0) → weight (0.5/0.1) → mean(T·B).

**Sheeprl.** Identical: `vendor/sheeprl/sheeprl/algos/dreamer_v3/loss.py:64-74` — `Independent(OneHotCategoricalStraightThrough(logits=[T,B,32,32]), 1)` makes the 32-categorical axis an event dim, so `kl_divergence` returns [T, B] with the KL already summed across all 32 categoricals; `free_nats = torch.full_like(dyn_loss, 1.0)` then `torch.maximum` clamps that summed value (`:68-74`). The DreamerV3 paper (Hafner et al. 2023, Eq. 5: `max(1, KL)`) specifies the same: 1 free nat on the full dynamics/representation KL, not per-latent.

**At our magnitudes.** H2b measured `loss_dyn_kl ≈ 0.9` at convergence. That logged key includes the 0.5 weight and the floor (`train.py:763`: `(kl_dynamic * max(kl, 1)).mean()`), so mean `max(KL, 1) ≈ 1.8` nats → the raw summed KL sits ~1.8× above the 1-nat floor on average. The clamp is therefore **not** saturating the dynamics loss — the prior receives real gradient — while still flooring low-KL elements as intended. The failure mode the critique feared (per-categorical clamp → 32-nat effective floor → `loss_dyn_kl` pinned at 0.5·32 = 16, prior untrained) is ruled out both by the code and by the measured value (0.9 ≪ 16).

**Verdict:** clamp semantics are summed-KL (paper- and sheeprl-exact); no regime issue. 🟢

### F/U-2 — 500-step episode cap is TRUNCATED, not terminated: bootstrap preserved on the best trajectories. Correct. 🟢

**Env.** `src/environment/core.py:700` — `truncated = next_step >= params.max_steps`; reason code 1 for max-steps (`:705`), death codes ≥ 2 (`:707-710`); real death captured *before* the merge (`real_death = done`, `:696`); `done = logical_or(done, truncated)` (`:717`) is only the episode-end/reset flag. If death coincides with the cap, the death code (≥2) overwrites reason 1 (`:707-710` run after `:705`) — correctly terminated.

**Driver.** `dreamer_srl_main.py:1484-1486` (the CP7-P2 fix, re-verified live): `terminated = (termination_reason >= 2)`, `truncated = (termination_reason == 1)` — a max-steps episode is written to the buffer as `terminated=0, truncated=1`.

**Trainer.** Both consumers read only `terminated`: the continue-head target `1 − batch["terminated"]` (`train.py:725`) is **1** at a truncation row — the head learns "the world continues here"; and the §S5 true-continue splice (`train.py:873-882` → `compute_imagined_returns`, `:480-484`) sets `continues[0] = 1` for imagination started from a truncation row, so the λ-return bootstrap through the critic is **preserved**, and the `cumprod` discount is not zeroed.

**Sheeprl.** Same semantics natively: gym returns separate `terminated`/`truncated`, `step_data["terminated"] = terminated` (`dreamer_v3.py:636-638`), continue target `1 − data["terminated"]` (`:168`), splice `1 − data["terminated"]` (`:247`) — time-limit ends bootstrap in the reference too.

**Regime note.** At γ = 0.996997 (effective horizon ≈ 333 steps > the 500-step cap), marking step-500 as terminal would have set V ≈ 0 exactly on the longest-surviving (best) trajectories — a large, systematic value bias against the behaviour we want to reinforce. That failure mode is **absent**: surviving to the cap bootstraps. (The one residual interaction remains F/U-independent: the dormant `overeating_death` reason-without-done quirk in the main findings table.)

**Verdict:** max-steps cap → truncated → bootstrap preserved, sheeprl-equivalent end-to-end. 🟢

## Cross-links

- [[dreamer_srl_h1_speed_investigation]] — docs/develop/active/diagnosis/dreamer_srl_h1_speed_investigation.md (H1 sibling: throughput axis)
- [[DREAMER_SRL_INVESTIGATION]] — docs/experiments/active/dreamer_srl_investigation/DREAMER_SRL_INVESTIGATION.md (parent investigation)
- [[dreamer_srl_settings_regime_critique]] — docs/project/critiques/dreamer_srl_settings_regime_critique.md (H3 sibling: hyperparameter regime)
- [[gridworld_vs_dreamerv3_benchmarks_difficulty]] — docs/project/critiques/gridworld_vs_dreamerv3_benchmarks_difficulty.md (H4 sibling: task difficulty)
- Prior line-level diagnosis: [review_full_diagnosis_20260723.md](diagnosis_20260723/review_full_diagnosis_20260723.md) + the five `findings_dreamer_*.md` files (absorbed above)

Reviewed by: code-reviewer (H2a parallel investigator), 2026-07-27
