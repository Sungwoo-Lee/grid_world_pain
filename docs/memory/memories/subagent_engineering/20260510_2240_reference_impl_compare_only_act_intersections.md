---
id: 20260510_2240_reference_impl_compare_only_act_intersections
date: 2026-05-10
time: "22:40"
folder: subagent_engineering
tags: [subagent, learned_lesson, decision, meta]
summary: "User-stated rule for using a third-party reference implementation (sheeprl) as a comparator: default action is to document differences in the concept doc; act on a difference (i.e., plan code changes to match) only when it intersects the live failure investigation. External reference impls always surface N differences, most of which are deliberate or framework-only divergences; reflexively matching them is a category error."
related: ["20260509_1537_professor_analysis_resets_exotic_investigation", "20260510_2241_residual_error_pattern_directs_next_fix"]
session_origin: claude_code
session_label: "dreamer_sheeprl_compare_and_zero_init_2026-05-10"
importance: medium
status: settled
supersedes: []
raw_source: claude_data/.claude/projects/-media-nas01-projects-Interoceptive-AI-grid-world-pain/4ae5401e-443a-406e-93e1-5431c67b5f59.jsonl
raw_completeness: full
---

# Reference-implementation comparison: document by default, act on intersections only

## Key conclusion
When using a third-party reference implementation as a comparator (concretely: sheeprl's DreamerV3 vs ours), the right default action on every surfaced difference is to **document it in the concept doc** — not to plan a code change to match. The user's rule, stated verbatim during this session: *"You don't have to update to match with sheeprl as there are intentionally diverged part. So, it will be enough to be documented. But if you think the differences are critical to the current training issue, then tell me it is."* The conditional that triggers a code-change plan is **direct intersection with the live failure investigation**, not "this is wrong relative to the paper" or "this differs from the reference impl." External reference impls surface many differences per comparison; some are deliberate research choices, some are framework-only idiom, some are genuinely-bugged-in-our-code, and some are bugged-in-theirs. Reflexively matching the third-party impl is a category error that conflates these four cases.

## Evidence, measurements, facts
- **Concrete arc this session**: senior-developer ran a sheeprl-vs-our-DreamerV3 comparison (commit `7186606`, §9 added to the concept doc, ~554 lines). The comparison surfaced 4 candidate deviations beyond the existing §6 list:
  1. GRU reset gate computed but never applied (`dreamer_v3_nnx.py:32–36` does `tanh(cand)` instead of paper-canonical `tanh(reset * cand)`).
  2. Prior + posterior heads have no hidden layer (single Linear vs paper + sheeprl one-hidden-layer MLPs).
  3. Critic self-EMA regularization term missing (sheeprl: `value_loss = -CE(λ-returns) - CE(target_critic_value)`; ours: just the first term).
  4. Reward + critic output layers not zero-initialized (`hafner_init` everywhere vs sheeprl's `uniform_init_weights(0.0)` on those specific layers).
- **Top-level Claude initially asked the user to fold all 4 into §6 + plan code fixes for the highest-priority ones.** The user corrected with the rule above, and asked: "if you think the differences are critical to the current training issue, then tell me it is."
- **Top-level Claude's analysis under the new rule**:
  - **#4 (zero-init)**: directly relevant to the just-localized reward-head failure (offline diagnostic showed reward MAE 2.6× threshold even on the simplest task). Output-layer init dominates early-training reward-prediction dynamics. → Triggered code change + Cell Z1 launch.
  - **#1 (GRU reset gate)**: real bug, broader impact, but offline diagnostic showed most observation channels predict fine — only reward (and prev-action one-hot) fail. So #1 is probably contributing to general WM weakness but isn't the specific reward-head cause. → Documented only.
  - **#2, #3**: less directly relevant to the localized reward-head failure. → Documented only.
- **Outcome**: only #4 was acted on. The other 3 were folded into §6 (items 28-30) and stayed as `MAJOR DEVIATION` flags without queued code changes. Cell Z1 (zero-init) achieved a 28% MAE reduction (H2 partial fix) — confirming #4 was a real and high-leverage intersection. The deferred #1/#2/#3 are not blocking; they remain on the §6 list for future re-evaluation if the cascade of acts-on-intersections runs out of candidates.
- **Generalizability**: the rule is not sheeprl-specific. It applies to any external comparator — paper canonical reference, second project's implementation, library port, etc. The pattern: scan → document → ask "does this intersect the live failure?" → only the answer "yes, mechanistically" triggers a code-change plan.

## Decisions and actions
- **Codified rule** for future investigations using external reference impls as comparators:
  1. Run the comparison. Document every difference in the concept doc (the "ours vs theirs" map).
  2. For each difference, classify into one of: framework-idiom (no-op), deliberate-divergence (no-op, document why), candidate-intersection-with-current-issue (plan a code change), candidate-intersection-with-other-known-issue (queue for later). Bias the threshold for "candidate-intersection" toward being conservative; reflexive matching is the failure mode.
  3. Surface the candidate-intersection list to the user with mechanistic reasoning per item; the user makes the act-or-defer call.
- **Hand-off**: top-level Claude (or the agent surfacing the comparison) should NOT auto-fold all differences into a fix plan — the surfacing should always come with a per-item mechanistic-relevance verdict and ask which to act on.

## Open questions and follow-ups
- Does this rule generalize beyond reference-impl comparisons? Probably yes — the same logic applies to any "audit surfaces N issues" output (env-config-auditor, code-reviewer reverse-pass completeness, math-reviewer reverse-pass). The general form: "audits surface candidates; only the subset that intersects the live failure gets acted on now; the rest get documented and queued."
- Should there be an upper bound on how many "deferred candidates" accumulate in §6 before a sweep-to-fix is warranted? Currently 30 deviations in the concept doc; some are tech-debt-class (dead code), some are research choices, some are likely-real-bugs-not-yet-acted-on. Worth a periodic audit.
- The same user might give a contradicting rule in a different context (e.g., "we should match the paper exactly going forward"); the rule above is investigation-driven, not policy. Future-Claude should re-confirm before applying broadly.

## References
- Concept doc with the §9 sheeprl comparison: `docs/project/concepts/dreamer_v3_implementation.md` (commit `7186606`). §9.11 has the four-candidate-deviation flagging table.
- §6 fold + zero-init plan: commits `cff1faf` (§6 update with items 27-30), `a88002f` (zero-init code), `dad09e1` (diary), `c4b7633` (Z1 verification report).
- Sibling insights from this session: `20260510_2239_z1_zero_init_h2_partial_pos_neg_asymmetry` (the scientific verdict that proved #4 was a real intersection), `20260510_2241_residual_error_pattern_directs_next_fix` (the closely-related methodology rule).
- Sheeprl reference checkout: `tmp/sheeprl/`, sheeprl rev `33b6366`.
- Raw conversation: synced via `./sync-agent-data.sh claude push`. To read on another node: `./sync-agent-data.sh claude pull`, then either `claude --resume 4ae5401e-443a-406e-93e1-5431c67b5f59` or `python scripts/claude_jsonl_to_md.py <jsonl> /tmp/<id>.md`.

<!-- BACKLINKS — auto-generated by scripts/regen_memory_graph.py; do not edit -->
## Backlinks
- [[20260513_2308_strong_strategy_validates_on_cp1]] (dreamer_diagnosis, 2026-05-13) — The Strong (A+B+C+D) deviation-prevention strategy paid off on the first checkpo
- [[20260723_1910_sheeprl_parity_drift_lives_in_glue]] (dreamer_diagnosis, 2026-07-23) — A from-scratch Fable-5 re-audit of both Dreamers vs sheeprl found the math cores
- [[20260723_1912_fresh_empirical_reaudit_finds_new_bug_class]] (dreamer_diagnosis, 2026-07-23) — A second independent audit pass with a fresh model generation (Fable 5) that EMP
<!-- END BACKLINKS -->
