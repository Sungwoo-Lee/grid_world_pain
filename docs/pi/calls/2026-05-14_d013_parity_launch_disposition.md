---
title: "PI Call — D-013 / parity-launch disposition: fix the dreamer-srl XS config (not the substrate)"
date: 2026-05-14
trigger: Pre-launch — D-013 (the XS-on-single-RTX-4090 OOM) was scheduled to be disposed at the parity-launch PI consultation per [pi.md](../../../.claude/agents/pi.md) pre-launch trigger; the consultation surfaced four substrate-change options, and the user's investigative push-back exposed that the options were framed against a mis-identified config target.
status: decided
---

# PI Call — D-013 / parity-launch disposition: fix the dreamer-srl XS config (not the substrate)

## Question

**What do we run as the dreamer-srl parity launch — a multi-GPU job, a single-GPU job with gradient checkpointing, a reduced-dimension job, or an intermediate compromise — given that the "XS" config we measured at 14.38 GB of compile memory will not fit on a single 24 GB GPU?**

## Headline

**The user picked a fifth option that was not in the original menu — "Go with XS".** The original four options all rested on a mis-identification of what "XS" meant. The dreamer-srl plan had been treating the config at `configs/dreamer_srl/01_food_only.yaml` as "the XS default", but that config was actually a port of sheeprl's *base* config without the XS-size overlay applied — so the values it carries (a 1024-wide dense layer, a 4096-wide recurrent state, 5 stacked MLP layers, a 96× CNN channel multiplier) are sheeprl's *XL* preset, not its XS. The parity baseline we are matching (a sheeprl run that survives the full episode in ~12.5 hours) was at the real XS preset (256-wide dense, 256-wide recurrent state, 1 MLP layer, 24× CNN multiplier). The user's decision is therefore **fix the config — correct `01_food_only.yaml` to actually mirror the real sheeprl XS (256/256/mlp_layers=1/cnn_multiplier=24)** — at which point the OOM disappears and single-GPU is the natural substrate. No multi-GPU plumbing, no gradient checkpointing, no dimensionality compromise.

This is a **config-correction disposition, not a substrate-change disposition.** D-013 closes via the config fix; CP10b's wall-clock measurement now runs single-GPU at the real XS scale and verifies the ≤25 hour budget gate.

## Context for a fresh reader

The project's last six weeks have been spent rebuilding a published RL agent called DreamerV3 (a world-model agent that learns by predicting future observations in a latent space, then training a policy via imagined rollouts inside that learned world-model) inside our codebase — a JAX port of a community PyTorch implementation called `sheeprl`. The point of the rebuild is so the project can attach its **neuromodulation hooks** (adjustable-gain conditioning blocks that simulate ascending neuromodulator effects on confidence, precision, and prediction-error weighting) into the agent's internals, in a single-stack JAX codebase, without having to maintain a Python-to-PyTorch bridge.

For the rebuild to count as a faithful port, it has to pass a **parity gate** — the JAX version has to reach the same survival performance as the upstream PyTorch version on the same simple environment (a 5×5 grid where the agent must find food before starving) within a **wall-clock budget** of about 25 hours per seed (twice the 12.5-hour upstream baseline). The parity gate is the bridge between "the code ports correctly checkpoint-by-checkpoint" (which 11 prior checkpoints have already verified) and "the project can actually use this thing for the neuromodulation experiments".

The wall-clock-budget question opened earlier this week when CP9 (the integration-smoke checkpoint) hit an **out-of-memory error** during JIT compilation on a single RTX 4090 GPU: the model's forward+backward graph needed ~14.4 GB of GPU memory just to compile, and on a single 24 GB GPU there wasn't enough headroom after JAX's default 90% pre-allocation. The CP9 smoke was forced to run on a *reduced-dimension* config to fit; the question of what config the **actual parity launch** runs on was tagged as **D-013** in the deviation log and routed to this PI consultation under the "pre-launch of a multi-run experiment" trigger.

The four options surfaced for this consultation were all variants of "the OOM is real, accept it, and change the substrate to fit": (A) multi-GPU launch across 4× RTX 4090s, (B) single-GPU with gradient checkpointing, (C) keep running at reduced dimensions, (D) intermediate-dim compromise. The user's push-back — *"Tell me why the OOM issue has appeared. Explain in brief and intuitive high level answer. Also check the original JAX dreamer's config and OOM setting. How they different?"* — triggered an investigation that exposed the framing error described in the Headline. The corrected understanding made the answer obvious: fix the config, drop the option set.

## Plain-language sheeprl size presets

Sheeprl ships **five named DreamerV3 size presets** as Hydra overrides on top of a single `dreamer_v3.yaml` base config. The base config itself carries XL-equivalent numbers; the overrides shrink the model down to the named size when launched with `algo=dreamer_v3_<size>`.

| Preset | dense layer width | stacked MLP layers | CNN channel multiplier | recurrent-state width | hidden-state width |
|---|---|---|---|---|---|
| **XS** (← the actual parity target) | **256** | **1** | **24** | **256** | **256** |
| S | 512 | 2 | 32 | 512 | 512 |
| M | 640 | 3 | 48 | 1024 | 640 |
| L | 768 | 4 | 64 | 2048 | 768 |
| **XL** (← what dreamer-srl was mis-ported to) | **1024** | **5** | **96** | **4096** | **1024** |

On the dominant axis (the recurrent state, which carries the agent's running summary of what's happened in the episode), **XS is 16× smaller than XL** (256 vs. 4096). The 14.38 GB compile cost we measured was an XL-config compile cost being asked to fit on a single-GPU XS budget. At the actual XS dimensions the compile cost drops well into comfortable single-GPU territory.

The matched-config measurement at [`docs/experiments/active/sheeprl_bridge/JAX_SHEEPRL_MATCHED_SPS_DESIGN.md`](../../experiments/active/sheeprl_bridge/JAX_SHEEPRL_MATCHED_SPS_DESIGN.md) had already confirmed (independently of this PI call) that the sheeprl 12.5h baseline was at 256/256, not at 1024/4096; the launcher at `scripts/launch_sheeprl.sh` defaults to `SIZE=XS → algo=dreamer_v3_XS`. So the parity-launch target on the sheeprl side is unambiguous; the mis-identification was entirely on the JAX-port side.

## Options considered

1. **Multi-GPU launch (4× RTX 4090, data-parallel).** Run the parity launch across 4 GPUs to give the JIT-compiled graph room. Projects 10–14 h per seed at the full XL config — inside the 25 h budget. *Cost:* multi-GPU plumbing under sheeprl's vendored Fabric path is non-trivial; the project would carry a multi-GPU dependency for every parity run going forward. *Buys:* preserves the full XL config "as-is".
2. **Single-GPU with gradient checkpointing.** Add gradient checkpointing to the JAX train step so the backward-pass activations don't all live in memory simultaneously. *Cost:* ~30–40% throughput hit; new code path under `train.py`; may still not fit the 25 h budget. *Buys:* keeps single-GPU.
3. **Reduced-dimension parity.** Permanently shrink the dreamer-srl config below "XS" to fit single-GPU. *Cost:* breaks the like-for-like parity claim — we're no longer matching what sheeprl's `dreamer_v3_XS` actually runs. *Buys:* trivially fits, fastest path.
4. **Intermediate-dim compromise.** Pick a between-XS-and-reduced point that fits single-GPU and is "close enough". *Cost:* same parity-claim erosion as (3), just smaller. *Buys:* better optics than (3).
5. **(Surfaced by user push-back; not in original menu) Config-correction.** The "XS" config at `configs/dreamer_srl/01_food_only.yaml` is not actually sheeprl XS — it's sheeprl base (≈ XL). Fix the config to mirror the real XS (256/256/mlp_layers=1/cnn_multiplier=24); the OOM disappears; single-GPU is the natural substrate. *Cost:* an audit + correction-note sweep across the v3 plan docs that mis-named the config as "XS-default" (CP9_PLAN.md, CP9B_PLAN.md, CP10B_SPEC.md, IMPLEMENTATION_PLAN.md frontmatter, DEVIATION_LOG.md). *Buys:* the cheapest path that preserves the like-for-like parity claim — we run what sheeprl runs.

## User decision

**Option 5 — Config-correction. Go with XS.**

Verbatim:

> "Go with XS"

The user surfaced the reframing themselves by demanding a root-cause explanation rather than accepting the menu as given. The decision is the cheapest path on every axis the project cares about: no substrate plumbing, no algorithm-faithfulness erosion, and no parity-claim weakening. D-013 closes as a config-correction, not a substrate change.

## Rationale captured

- **Like-for-like parity is preserved by running what the baseline runs.** Sheeprl's 12.5-hour baseline is at XS (256/256). The JAX port's parity claim is "the JAX rebuild matches sheeprl on the same env at the same config within 2× wall-clock". The only way to keep that claim is to actually run XS on the JAX side too — which option 5 makes possible.
- **The simpler diagnosis wins over the more complex one.** Three of the four original options (A, B, D) added engineering surface (multi-GPU plumbing, gradient checkpointing, a new compromise dimensionality) to work around a config bug. The fourth (C) eroded the parity claim. Fixing the config is strictly simpler than all four.
- **The user's investigative push-back is the load-bearing move.** The PI did not surface this option; the user did, by refusing the menu and asking for the root cause. Future PI consultations should test whether the option set rests on a hidden mis-identification before surfacing it.
- **The 14.38 GB measurement is not invalidated; it's recontextualised.** The OOM was real on the config it was measured on. That config was just labeled wrong. The measurement is now repurposed as a useful negative result — single-GPU XL does not fit a 24 GB card — which is itself a fact the eventual paper / methods section can cite.

## Process lesson — sheeprl base-config-without-overlay trap

The dreamer-srl developers ported their `configs/dreamer_srl/01_food_only.yaml` from `vendor/sheeprl/sheeprl/configs/algo/dreamer_v3.yaml` — i.e., they took the **base config** and treated it as a complete config. But sheeprl's config system uses Hydra-style **size overlays**: the base config carries XL-equivalent values, and the smaller presets are applied as overrides at launch time (`algo=dreamer_v3_XS` selects the XS overlay). The dreamer-srl port copied the base values into a flat YAML file without applying any overlay, then labeled it "XS default" throughout the v3 plan docs (CP9_PLAN.md, CP9B_PLAN.md, CP10B_SPEC.md, IMPLEMENTATION_PLAN.md, DEVIATION_LOG.md, D-013's cell itself). The mis-naming was internally consistent — every doc referred to the same flat YAML — but every doc was wrong about what the YAML represented.

**This is a generalisable trap, not a one-off mistake.** Any future port from a Hydra-style config tree should explicitly **resolve the overlay** before treating the resulting YAML as canonical (e.g., by running Hydra and dumping the merged config). The check-and-replace rule for the next time: *if the port doc says "we used the XS default from sheeprl", confirm that the numbers in the port match the overlay-applied XS, not the base config.*

**The historical references in the v3 plan docs need a correction sweep — NOT a silent rewrite.** The mis-named "XS-default" references in CP9_PLAN.md, CP9B_PLAN.md, CP10B_SPEC.md, IMPLEMENTATION_PLAN.md, and DEVIATION_LOG.md should each get an **inline correction note** flagging the mis-name and pointing at this PI call. The original wording stays so a post-hoc reader can see what happened (which is the same transparency discipline as the deviation log itself). The senior-developer owns this sweep as part of the config-fix plan.

## What this implies operationally

1. **D-013 closes as `✅ RESOLVED — config-correction (not substrate-change)`.** The deviation-log row at line 79 of [`docs/develop/active/dreamer_srl_v3/DEVIATION_LOG.md`](../../develop/active/dreamer_srl_v3/DEVIATION_LOG.md) gets its verdict cell flipped from `☐ pending` to `✅ RESOLVED`, with the resolution-cell text noting that the 14.38 GB compile cost was measured at an XL-equivalent config that had been mis-labeled "XS default", and that the corrected XS config is expected to fit single-GPU trivially.
2. **CP10b's scope is now well-defined.** [`CP10B_SPEC.md`](../../develop/active/dreamer_srl_v3/CP10B_SPEC.md) was originally scoped as "the like-for-like XS-vs-XS wall-clock measurement that runs after the PI consultation disposes D-013". With this disposition, CP10b is now: *run the corrected XS config on a single GPU, measure wall-clock per 200k env-steps, verify ≤ 25 h budget gate.*
3. **The parity-launch sweep is now scoped.** Once CP10b verifies the budget, the parity gate is 3 seeds × the corrected XS config × different env seeds, single-GPU per seed. Sheeprl-side baseline already exists at WandB run `i4ulpn95` (12.5 h, the 256/256 baseline).
4. **No multi-GPU plumbing.** The Fabric multi-GPU disposition (option A) is **rejected** because the config was the actual error — multi-GPU would have been a workaround, not a fix.
5. **No gradient checkpointing.** Same logic — option B was a workaround for a config error, not the OOM on real XS.
6. **Correction-note sweep across the v3 plan docs.** The mis-named "XS-default" references in CP9_PLAN.md, CP9B_PLAN.md, CP10B_SPEC.md, IMPLEMENTATION_PLAN.md, and DEVIATION_LOG.md each get an **inline correction note** flagging the mis-name and pointing at this PI call. **Original wording stays;** the corrections are additive.
7. **The portfolio is still unratified.** [`PORTFOLIO.md`](../PORTFOLIO.md)'s active publication tracks are still TBD. The recent-calls table gets a new row for this call; the first-track-ratification follow-up remains pending.

## Hand-off

- **Next agent (Step 1):** `senior-developer` — scope the config-fix plan:
  - Edit `configs/dreamer_srl/01_food_only.yaml` to mirror real sheeprl XS (`dense_units=256`, `mlp_layers=1`, `cnn_channels_multiplier=24`, `recurrent_state_size=256`, hidden_size=256). The other relevant XS-overlay knobs (stochastic_size, discrete, horizon — the world-model rollout horizon, etc.) need to be cross-checked against `vendor/sheeprl/sheeprl/configs/algo/dreamer_v3_XS.yaml` so the corrected YAML is overlay-equivalent end to end, not just on the dominant axes.
  - Author the **CP10b spec update** so it reflects the corrected config target and the substantially lower expected wall-clock (now well inside the 25 h budget gate; the 41–58 h projection in DEVIATION_LOG.md line 79 was an XL-config projection and is now obsolete).
  - Audit the v3 plan docs for "XS-default" mis-name references (CP9_PLAN.md, CP9B_PLAN.md, CP10B_SPEC.md, IMPLEMENTATION_PLAN.md frontmatter + body, DEVIATION_LOG.md frontmatter + D-013 cell) and **add inline correction notes** flagging each mis-name and pointing at this PI call. Do not rewrite history; the original wording stays and the correction is additive.
- **Step 2:** `developer` — implement the config edit per the senior-developer's plan.
- **Step 3:** `training-runner` — launch CP10b on a single GPU (likely node 114 or another free node; user confirms node + GPU at launch time per the `training-runner` agent's input contract).
- **Step 4 (gated on CP10b verification):** `experiment-designer` — author the parity-launch sweep configs (3 seeds × corrected XS × different env seeds).
- **Step 5:** `training-runner` — launch the parity-gate sweep.
- **Stop rule:** If the corrected XS config OOMs on single-GPU (i.e., the recontextualised diagnosis is wrong and the OOM was not config-driven), escalate back to PI before deferring to multi-GPU. The current call rests on "the 14.38 GB was an XL-config cost"; if CP10b shows otherwise, the option set in this call re-opens and option A (multi-GPU) becomes the natural fallback.

## Links

- Deviation log: [`DEVIATION_LOG.md`](../../develop/active/dreamer_srl_v3/DEVIATION_LOG.md) (D-013 row, line 79)
- CP10b spec (the like-for-like wall-clock measurement that closes D-013 numerically): [`CP10B_SPEC.md`](../../develop/active/dreamer_srl_v3/CP10B_SPEC.md)
- Matched-config measurement (independent confirmation that sheeprl's 12.5 h baseline was at 256/256): [`JAX_SHEEPRL_MATCHED_SPS_DESIGN.md`](../../experiments/active/sheeprl_bridge/JAX_SHEEPRL_MATCHED_SPS_DESIGN.md)
- Sheeprl launcher (defaults to XS): `scripts/launch_sheeprl.sh`
- Prior PI call (backbone decision to use sheeprl-direct): [`2026-05-12_dreamer_backend.md`](2026-05-12_dreamer_backend.md)
- Portfolio (unratified — first-track ratification still pending): [`PORTFOLIO.md`](../PORTFOLIO.md)
