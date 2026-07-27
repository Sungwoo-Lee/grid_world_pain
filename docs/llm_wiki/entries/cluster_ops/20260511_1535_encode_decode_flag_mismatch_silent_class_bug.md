---
id: 20260511_1535_encode_decode_flag_mismatch_silent_class_bug
date: 2026-05-11
time: "15:35"
folder: cluster_ops
tags: [dreamer, learned_lesson, meta, decision]
summary: "When a knob-gated encode/decode change lands in production training code, every auxiliary script that touches the same code path must be updated in lockstep — otherwise the trained model speaks the new layout, the auxiliary script listens in the old layout, and the output is silent garbage (no error, just wrong numbers). Concrete instance: `scripts/dreamer_offline_wm_test.py` missed the new `paper_canonical_bins` flag after the Z2 trainer change; first Z2 diagnostic run reported MAE = 1.04 (5.7× the true value 0.177) looking like a catastrophic regression. Generalizable heuristic that caught it: any diagnostic that decodes a trained model's outputs should cross-check at least one metric against the training-time logger on the same checkpoint, as a silent-failure-mode trip-wire."
related: ["20260509_1536_train_py_checkpoint_restore_nnx_skew", "20260510_2241_residual_error_pattern_directs_next_fix", "20260511_1534_z2_paper_bins_h2_partial_cascade_closure_attempt"]
session_origin: claude_code
session_label: "dreamer_sheeprl_compare_and_zero_init_2026-05-10"
importance: medium
status: settled
supersedes: []
raw_source: claude_data/.claude/projects/-media-nas01-projects-Interoceptive-AI-grid-world-pain/4ae5401e-443a-406e-93e1-5431c67b5f59.jsonl
raw_completeness: full
---

# Encode/decode flag mismatch is a silent-failure-mode class — cross-check against training-time logger as trip-wire

## Key conclusion
When a knob-gated encode/decode change ships in production training code, the failure mode in adjacent auxiliary tools is **silent**: no exception, no crash, just numerically-wrong outputs. The bug class: a `to_X` / `from_X` function pair gains an optional flag controlling the encoding layout; production code is updated to set the flag based on a new mandatory config key; the auxiliary tool (a diagnostic script, an eval harness, a visualizer) is *not* updated. The auxiliary tool's call still type-checks and runs, but it decodes the trained model's outputs using the *old* layout — round-tripping garbage. **The discriminating signal that catches this class of bug**: cross-check at least one output metric against the training-time logger metric on the same checkpoint. If the trainer logs an internal-decode metric and the auxiliary tool computes the same metric from a manual decode, an inconsistency forces triangulation. Without that cross-check, the auxiliary tool's wrong numbers ship as authoritative.

## Evidence, measurements, facts
- **Trigger event**: Cell Z2 added a `paper_canonical_bins: bool` flag to `to_twohot` / `from_twohot` in `src/models/dreamer_v3_util.py` (commits `f5df600` / `8089ee2`). The production trainer was updated — mandatory config key `agent.paper_canonical_twohot_bins` read at `trainer.py` agent-config-construction site, routed to all 9 trainer call sites + 1 nnx call site. The auxiliary diagnostic script `scripts/dreamer_offline_wm_test.py` was **missed**: its single `from_twohot(...)` call at line ~254 still passed no flag, defaulting to `False` (legacy ±20 bins).
- **Symptom**: first Z2 diagnostic run reported reward MAE @ h=5 = **1.04**. The expected ballpark from the cascade trajectory was around 0.18 — A1 was 0.39, Z1 was 0.28, the mechanistic prediction said Z2 should be in [0.15, 0.30). 1.04 is 5.7× the true value and would have shipped as a fake catastrophic regression — looking like the bin-range fix made things dramatically worse.
- **Cross-check that caught it**: the training-time WandB summary metric `model_reward_mae` was 0.372 — sane and consistent with Z2 being on a partial-improvement trajectory. The diagnostic's 1.04 was incompatible with both (a) the training-time metric on the same checkpoint and (b) the cascade's expected slope. The 5.7× incompatibility forced manual triangulation: monkey-patch the diagnostic with `paper_canonical_bins=True` → MAE = 0.177, matching expectations. The training-time metric is decoded via the in-trainer call path which uses the production flag automatically, so it's the natural anti-silent-bug oracle.
- **Patch**: developer fix in commit `1703a4c`. Two-part: (a) read `agent.paper_canonical_twohot_bins` from the checkpoint's *saved* config (not the trunk YAML — this matters for backward-compatibility with pre-fix checkpoints); (b) pass the flag through to all `from_twohot` calls in the script. Plus a backward-compatibility wrinkle: pre-fix checkpoints (A1, Z1) saved configs that *lack* the key entirely; the script falls back to `False` and injects the key into the merged config before trainer construction so the trainer's `get_mandatory` doesn't crash on historical checkpoints.
- **Backward-compat smoke tests** (load-bearing for the patch's correctness): re-running the patched script against Z1's checkpoint reproduced the analyzer's prior authoritative MAE = 0.277 with flag resolved to `False` (key absent → injected); re-running against Z2 reproduced 0.177 with flag resolved to `True` (key present from the post-fix training).
- **Generalizable pattern**: this is the **second instance** of the same bug class. The first was `train.py:981–1000` checkpoint-restore code that fell out of sync with NNX's evolving format expectations (insight `20260509_1536_train_py_checkpoint_restore_nnx_skew`). Both bugs are *silent* (no crash, wrong-or-no-output); both were caught by an external sanity check (in that case, the offline diagnostic script worked around it explicitly; in this case, the WandB cross-check). The bug class is "auxiliary code falls out of sync with production code's evolving data layout."

## Decisions and actions
- **Codified heuristic** for future fix-cascades:
  1. When a knob-gated encode/decode change ships in production training, sweep all auxiliary scripts and tools that touch the same `to_X` / `from_X` API. Update each in lockstep, with the patch landed in the *same* commit if possible. Use grep on the function names plus the new flag name to ensure coverage.
  2. When running a diagnostic on a trained checkpoint, **always cross-check at least one metric against the training-time logger output for that checkpoint**. If both numbers agree, the decode path is correct; if they disagree, halt and triangulate before reporting any verdict.
  3. For backward compatibility with historical checkpoints (which may have been trained before a new mandatory key was introduced), the auxiliary script should fall back to the legacy default for the absent key and inject it into the loaded config before passing to the trainer constructor — so the trainer's `get_mandatory` doesn't crash on the saved config.
- **Diagnostic script now self-prints the resolved flag value at startup** for auditability (the patch added this). If a future cascade introduces another flag, the same pattern applies — print resolved values at startup.
- **Hand-off**: the bug-class lesson belongs in `cluster_ops` (auxiliary-tooling infrastructure); the cross-check heuristic is the actionable defensive practice. Both should be considered before any future addition of a knob-gated encode/decode pair.

## Open questions and follow-ups
- Are there OTHER auxiliary scripts that touch the `to_twohot` / `from_twohot` path and are also out of sync? Worth a one-time grep sweep: `grep -rln "to_twohot\|from_twohot" scripts/ src/` and check each call site for proper flag routing.
- A more systemic fix: factor `to_twohot` / `from_twohot` to read the flag from a thread-local or a `Config` object passed in, removing the need for every caller to remember the flag. Tradeoff: makes the functions harder to use outside a Config context (e.g., in unit tests). Probably not worth the abstraction cost given how rare encode/decode flag changes are.
- Could static analysis catch this class of bug? In principle yes — a "function has a default argument; this caller doesn't pass it; is the default the correct value for this context?" check. In practice, the default IS the legacy correct value for some contexts (pre-fix checkpoints, unit tests), so a static check would have false positives. The cross-check heuristic is cheaper than a custom lint.

## References
- Cell Z2 verification report (where the bug was surfaced + the cross-check that caught it): `docs/develop/active/diagnosis/dreamer_twohot_bin_range_fix.md` §Verification Report + §V.6 (Metrics Requested, now resolved).
- Patch commits: `1703a4c` (script + plan §V.6 update), `b82dad8` (developer diary row).
- Production change that introduced the flag: commits `f5df600` (`to_twohot` / `from_twohot` signature change in `src/models/dreamer_v3_util.py`) + `8089ee2` (trainer routing).
- Sibling insight from this session (the scientific finding the bug obscured): `20260511_1534_z2_paper_bins_h2_partial_cascade_closure_attempt`.
- Predecessor instance of the same bug class (different mechanism, same shape): `20260509_1536_train_py_checkpoint_restore_nnx_skew` — trainer's orbax restore path fell out of sync with NNX's checkpoint-leaf serialization format; also silent-failure-mode.
- Related methodology rule that picked this fix in the first place: `20260510_2241_residual_error_pattern_directs_next_fix` — the Z2 fix selection that made the bug surface in the first place.
- Raw conversation: synced via `./sync-agent-data.sh claude push`. To read on another node: `./sync-agent-data.sh claude pull`, then either `claude --resume 4ae5401e-443a-406e-93e1-5431c67b5f59` or `python scripts/claude_jsonl_to_md.py <jsonl> /tmp/<id>.md`.

<!-- BACKLINKS — auto-generated by scripts/regen_wiki_graph.py; do not edit -->
## Backlinks
- _no inbound links yet_
<!-- END BACKLINKS -->
