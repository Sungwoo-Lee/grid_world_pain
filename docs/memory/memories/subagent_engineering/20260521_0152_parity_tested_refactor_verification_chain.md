---
id: 20260521_0152_parity_tested_refactor_verification_chain
date: 2026-05-21
time: "01:52"
folder: subagent_engineering
tags: [meta, learned_lesson, subagent, decision, design]
summary: "A 4-agent verification chain (senior-developer plan → code-reviewer + math-reviewer in parallel → developer commit-by-commit with parity gates → senior-developer Verification Protocol) successfully shipped a high-stakes architectural refactor on a parity-tested codebase without introducing math drift. The pre-implementation reviewer round caught 3 plan-doc bugs that would have produced silent failures during implementation. Reusable template for any future refactor of bit-identity-tested code."
related: ["20260513_2308_strong_strategy_validates_on_cp1", "20260518_1511_dreamer_srl_v2_parity_pass_outperform", "20260521_0151_xla_scan_body_compile_dominates_module_count"]
session_origin: claude_code
session_label: "dreamer-srl v2 perf fix — JointTrainer refactor session"
importance: high
status: settled
valid_until: null
confidence: high
supersedes: []
raw_source: claude_data/.claude/projects/-media-nas01-projects-Interoceptive-AI-grid-world-pain/7962c4de-7ac9-4c9a-9958-c22a36fd45c7.jsonl
raw_completeness: full
---

# The 4-agent verification chain for parity-tested refactors

## Key conclusion
When the user explicitly requested "careful plan with proper agents to avoid any potential mistake" before a high-stakes architectural refactor on a bit-identity-grad-parity-tested codebase (dreamer-srl v2's JointTrainer composition), the routing that worked was a **4-agent chain with a pre-implementation reviewer round**: `senior-developer` drafts the plan → `code-reviewer` and `math-reviewer` audit the plan in parallel BEFORE any code is written → `developer` implements commit-by-commit with per-commit parity gates → `senior-developer` runs the Verification Protocol on the dirty tree. The pre-implementation reviewer round was decisive: code-reviewer's verdict `accept-with-changes` surfaced 3 plan-doc bugs (non-runnable diagnostic pseudocode using wrong NNX-state key syntax, missing L2-coverage-gap callout, incomplete C3.a 5-constraint grep checklist) that would have produced silent failures or false-green gates during implementation. The chain shipped C1-C5 commits with zero out-of-scope edits, zero L1 bit-identity regressions, zero L2 math-equivalence regressions, and a single non-blocking R2 documentation gap. This is the reusable template for any future refactor of parity-tested code.

## Evidence, measurements, facts

**The chain that worked (4 agents, ~7h end-to-end)**:

1. **`senior-developer` — plan drafting** (~30 min): output `docs/develop/active/dreamer_srl_v2/joint_trainer_refactor_plan.md`. Contained: plain-language Context, Goals + Non-goals, Design with exact attribute list, full call-site enumeration table (19 rows with file:line), scan-body BEFORE/AFTER, commit topology C1-C6 with per-commit gate, Risks (R1 stale-reference, R2 optimizer-state pytree shape), Rollback, File Changes, Testing Strategy, Hand-off.

2. **`code-reviewer` + `math-reviewer` — parallel pre-impl audit** (~10 min each, concurrent): output `docs/reviews/joint_trainer_plan_review.md` and `docs/reviews/joint_trainer_math_review.md`. Orthogonal scopes:
   - code-reviewer covered JAX/Flax/NNX structural correctness (split-outside / merge-inside / update-outside placement, no stale references, no nested merges, optimizer-state pytree layout, PRNG threading, static-argnames discipline).
   - math-reviewer covered construction-preserves-equivalence (jax.grad boundaries, optimizer update order, target-critic EMA, symlog/twohot transforms, λ-returns, REINFORCE pairing, stop_gradient boundaries, vmap axes, PRNG semantics, bit-identity under the new container).
   - Verdicts: code-reviewer `accept-with-3-revisions`; math-reviewer `construction-preserves-equivalence`. The 3 revisions were applied by `senior-developer` in a focused ~10-min revision pass.

3. **`developer` — commit-by-commit implementation** (~4h dev time + ~6h of bench wall-clock the developer waited through): output commits C1-C5 (`3e1ec2d`, `f109b4a`, `2e8a433`, `c204b77`, plus C4-legacy-bench result-only at no separate commit). Per-commit gates:
   - C1: full pytest stays green (no behaviour change yet).
   - C2: L1 grad-parity 11/11 bit-identity vs sheeprl.
   - C3: L2 math-equivalence 5/5 at atol=1e-5 + L1 regression check.
   - C4 legacy bench: 41.14 inst SPS (matched Option-M reference within noise).
   - C5: L1 + L2 both still green.
   - C4 scan bench + C6 food-only smoke: deferred (XLA-compile pathology — separate finding, see [[20260521_0151_xla_scan_body_compile_dominates_module_count]]).

4. **`senior-developer` — Verification Protocol** (~25 min): output `docs/reviews/joint_trainer_verification.md`. Verdict: `verified-structural-complete`. Audited 11 questions covering every plan-named call site, legacy-path integrity, out-of-scope-edit count, commit topology, C3.a grep checklist, R2 diagnostic, L1+L2 re-run sanity samples, `--legacy-grad-loop` default state, orphan-file check, plan-doc ground-truth alignment. One non-blocking flag (R2 diagnostic full list went to tmp/ rather than the Implementation Report) plus a KEEP-on-v1.4 recommendation.

**What the chain caught that a less-careful flow would have missed**:

- **Plan-doc bug #1 (caught by code-reviewer pre-impl)**: R2 diagnostic pseudocode used `joint_state['wm_opt']` dict-subscript that does not match how `nnx.State` is keyed in flax 0.12.4. A naive developer would have followed the pseudocode, gotten a confusing KeyError or wrong layout, and either patched around it incorrectly or skipped the diagnostic entirely. Code-reviewer's fix: replace with `jax.tree_util.tree_paths(joint_state)` (or `tree_flatten_with_path` in JAX 0.9.x as the implementing developer actually used).
- **Plan-doc bug #2 (caught by code-reviewer pre-impl)**: the L2 math-equivalence tests intentionally stay on the seven-everything carry shape — meaning they do NOT cover the JointTrainer pytree-shape claim. The R2 diagnostic is therefore the ONLY check that catches optimizer-state pytree-shape drift. The plan needed an explicit "load-bearing role of the R2 diagnostic" callout instructing the verifier to block C2-advance until the diagnostic is captured. Without this, a future R2-shape regression would silently pass all unit tests.
- **Plan-doc bug #3 (caught by code-reviewer pre-impl)**: C3.a gate originally checked only `nnx.merge` count inside the body. Code-reviewer expanded to a 5-constraint checklist (split outside == 1, split inside == 0, merge inside == 1, update outside == 1, update inside == 1 for Polyak). A naive developer with only the original 1-constraint gate could have left, say, a stale `nnx.split` inside the scan body (the original 70×-regression cause) and the gate would have called it green.

**What the chain spent on the user's "no mistake" budget**:
- ~7h end-to-end (plan 30m + reviewers 10m × 2 in parallel + revision 10m + developer 4h + verification 25m + waiting/queuing buffer ~2h)
- Compared to a "just implement it" flow: probably 1-2h. So the careful-plan-with-agents premium was ~4-5h.
- The premium bought: 3 silent-failure modes caught at the plan stage (cheap), 0 out-of-scope edits, 0 math regressions, a documented audit chain that future-Claude can rerun if questions arise.

**The chain DID NOT prevent**: the XLA-compile-time pathology surfacing at C4 bench (this was a *finding*, not a *mistake* — the chain correctly identified it as a deferred perf-gate, not a regression). See [[20260521_0151_xla_scan_body_compile_dominates_module_count]] for that separate insight.

## Decisions and actions

- **Reusable template**: any future refactor of a parity-tested codebase (bit-identity tests, math-equivalence tests, end-to-end smoke launches) should follow this exact 4-agent chain. Don't skip the pre-impl reviewer round — it's where the cheapest catches live. Specifically:
  - Plan author: `senior-developer`
  - Pre-impl reviewers: `code-reviewer` for structural correctness + `math-reviewer` for construction-preserves-equivalence (parallel; orthogonal scopes)
  - Implementer: `developer` (commit-by-commit, parity-gated)
  - Verifier: `senior-developer` Verification Protocol on the dirty tree
- **The pre-impl reviewer round is the load-bearing piece**: 3-of-3 catches in this session happened at that stage. A senior-developer plan that has not been independently reviewed before implementation is the single highest-risk pattern in this project's agent ecosystem.
- **Commit-by-commit with parity gates is the implementer's discipline**: not optional. C1 was a no-behaviour-change skeleton that proved the test infrastructure still worked; C2 was the first behavioural change with L1 bit-identity as the gate; C3 was the perf-critical change with L2 math-equivalence as the gate. Skipping any of these to "save time" loses the ability to bisect a regression to a specific commit.
- **The Verification Protocol catches what the developer missed**: in this session, the verifier caught one non-blocking R2-documentation gap (the developer ran the diagnostic but its full output went to tmp/ rather than the Implementation Report). Without the verifier, this gap would have shipped silently and bitten a future C4 attempt.

## Open questions and follow-ups

- **When NOT to use this chain**: trivial bug fixes (1-line typo, obvious off-by-one) don't justify the 7h overhead. The cutoff for invoking the full chain is roughly: "would a regression here cost more than 7h to debug after-the-fact?" If yes, use the chain. If no, skip it.
- **Does the chain scale to >5-commit refactors?** This session was C1-C5 (6h dev + bench). For larger refactors (e.g., 20+ commits), the developer Verification Protocol step probably needs to be re-invoked at intermediate milestones, not just at the end. Untested.
- **The architectural insight separate from the chain**: see [[20260521_0151_xla_scan_body_compile_dominates_module_count]] for the perf finding this refactor surfaced — the chain shipped the refactor cleanly but did not (and could not) prevent the underlying XLA-compile-pathology blocker.

## References

- Working artifacts this session produced:
  - Plan: [`joint_trainer_refactor_plan.md`](../../../develop/active/dreamer_srl_v2/joint_trainer_refactor_plan.md)
  - Code-review: [`joint_trainer_plan_review.md`](../../../reviews/joint_trainer_plan_review.md)
  - Math-review: [`joint_trainer_math_review.md`](../../../reviews/joint_trainer_math_review.md)
  - Verification: [`joint_trainer_verification.md`](../../../reviews/joint_trainer_verification.md)
- Companion architectural insight: [[20260521_0151_xla_scan_body_compile_dominates_module_count]] (the perf finding; this insight is about the *process* that surfaced it).
- Prior settled insights on the Strong A+B+C+D pattern that this chain is the next-generation of:
  - [[20260513_2308_strong_strategy_validates_on_cp1]] (Strong A+B+C+D discipline first-validated on dreamer-srl v3 CP1; this 4-agent chain is the agentic version of that discipline).
  - [[20260518_1511_dreamer_srl_v2_parity_pass_outperform]] (v2 audit-chain that built the parity infrastructure this refactor leveraged).
- Reviewer profiles: `.claude/agents/code-reviewer.md`, `.claude/agents/math-reviewer.md`, `.claude/agents/senior-developer.md`, `.claude/agents/developer.md`.
- Raw conversation: synced via `./sync-agent-data.sh claude push`. To read on another node: `./sync-agent-data.sh claude pull`, then either `claude --resume 7962c4de-7ac9-4c9a-9958-c22a36fd45c7` or `python scripts/claude_jsonl_to_md.py <jsonl> /tmp/20260521_0152_parity_tested_refactor_verification_chain.md`.

<!-- BACKLINKS — auto-generated by scripts/regen_memory_graph.py; do not edit -->
## Backlinks
- _no inbound links yet_
<!-- END BACKLINKS -->
