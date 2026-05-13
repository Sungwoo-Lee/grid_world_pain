---
id: 20260509_1536_train_py_checkpoint_restore_nnx_skew
date: 2026-05-09
time: "15:36"
folder: cluster_ops
tags: [dreamer, learned_lesson, meta, training_runner]
summary: "Latent infrastructure bug surfaced during the offline WM-test build: `train.py:981–1000` orbax restore would fail with current NNX on a fresh checkpoint restore due to a string-key + `{'value': array}` leaf skew that `nnx.update` does not accept. The offline-test script worked around it via `_normalize_checkpoint`; the trainer itself has not been fixed. Any future `train.py --resume` workflow would currently fail."
related: []
session_origin: claude_code
session_label: "dreamer_conventional_fixes_battery_2026-05-09"
importance: medium
status: active
supersedes: []
raw_source: claude_data/.claude/projects/-media-nas01-projects-Interoceptive-AI-grid-world-pain/4ae5401e-443a-406e-93e1-5431c67b5f59.jsonl
raw_completeness: full
---

# train.py orbax-vs-NNX checkpoint-restore skew (latent bug)

## Key conclusion
Building `scripts/dreamer_offline_wm_test.py` against a real DreamerV3 checkpoint surfaced a latent bug in the trainer's own restore path: `train.py:981–1000` runs an orbax restore and feeds the result directly to `nnx.update`, but current NNX rejects two skews in the orbax-serialized payload — top-level dict keys are strings (e.g. `"0"`, `"1"`) where NNX expects ints, and individual leaves are wrapped as `{'value': array}` rather than bare arrays. The offline-test script needed a `_normalize_checkpoint` helper to pre-process the payload before `nnx.update` would accept it. The trainer itself has no such helper, so any future `train.py --resume` against a checkpoint produced by the current orbax+NNX combination would fail at restore time. The bug is silent until someone tries to resume.

## Evidence, measurements, facts
- Bug location: `train.py:981–1000` (orbax restore block).
- Workaround used by the offline-test script: `_normalize_checkpoint` (in `scripts/dreamer_offline_wm_test.py`, commit `3d93b0a`) — string-to-int key normalization on the top-level dict + recursive `{'value': array}` leaf unwrapping.
- Symptom seen during dev: `nnx.update(model, state)` raises a key-type or value-shape error before the loaded weights touch any forward pass.
- Verified by reproduction: the offline-test script now successfully loads Cell A1's checkpoint (`results/JAX_DreamerV3/20260509-050606_dreamer_conv_NoPred_rr06_s0_n113/`, step 700009) and the loaded weights match the trainer's weights bit-for-bit (verification step 7 in `dreamer_offline_wm_imagination_test.md`).
- Verification report flag: `senior-developer` flagged this as out-of-scope-for-this-verification but in-scope-for-a-separate-bug-fix-plan (commit `e6d2bbe`).
- Not currently triggered: training runs in this project are launched fresh (no `--resume` in the standard flow); this is why the bug has been latent. Any future flow that needs to resume training (e.g., extending A1 to 1.5M, restarting after a crash) will hit it.

## Decisions and actions
- Bug filed as a future plan target (no plan written yet — flagged for `senior-developer` at the next bug-fix routing pass).
- Workaround copy-path is documented (the `_normalize_checkpoint` helper in the offline-test script). When the trainer fix lands, it should either share that helper or include the same normalization inline.
- Not blocking the current investigation chain — this session uses fresh launches, not resumes.

## Open questions and follow-ups
- When did the orbax/NNX skew get introduced? Worth a `git log --diff-filter=M -- train.py` walk around the orbax restore block to find the last change. Could be a transitive dep bump too.
- Is the workaround the right fix, or should the trainer write checkpoints differently so they round-trip through `nnx.update` without normalization? The "right" fix probably belongs to the save side, not the load side.
- Are existing checkpoints (e.g. all `results/JAX_DreamerV3/*/` from prior cells) all affected, or only those produced after some specific code change? Worth a quick sample test.

## References
- Workaround source: `scripts/dreamer_offline_wm_test.py` (commit `3d93b0a`); `_normalize_checkpoint` helper.
- Plan with verification flag: `docs/develop/active/diagnosis/dreamer_offline_wm_imagination_test.md` (Verification Report section, commit `e6d2bbe`).
- Trainer bug location: `src/algorithms/JAX_DreamerV3/train.py:981–1000` (orbax restore block).
- Sibling insight (same session, broader investigation context): `20260509_1534_wm_reward_head_localized_failure_a1`.
- Raw conversation: synced via `./sync-agent-data.sh claude push`. To read on another node: `./sync-agent-data.sh claude pull`, then either `claude --resume 4ae5401e-443a-406e-93e1-5431c67b5f59` or `python scripts/claude_jsonl_to_md.py <jsonl> /tmp/<id>.md`.
