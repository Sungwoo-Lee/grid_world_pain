---
id: 20260629_1723_ghost_predator_inactive_slots_render
date: 2026-06-29
time: "17:23"
folder: env_entities
tags: [design, learned_lesson, decision, config]
summary: "Per-episode count masking gated damage/sensing/obs by animal_active but NOT per-step movement, so inactive 'ghost' predators un-parked from off-grid and chased/stuck to the agent — rendered but dealing 0 damage and invisible to the agent. Fix: re-park inactive animal slots off-grid every step in update_animals. Verified purely cosmetic (survival byte-identical)."
related: ["20260623_0143_per_episode_count_variance_masking", "20260623_0144_inactive_resource_slots_revive_respawn"]
session_origin: claude_code
session_label: "basic-05 random-init launch + ghost-predator bug diagnosis/fix + before-after eval"
importance: high
status: settled
valid_until: null
confidence: high
supersedes: []
raw_source: claude_data/.claude/projects/-media-nas01-projects-Interoceptive-AI-grid-world-pain/428cfe9b-4b3e-4d99-8065-98b46ee41016.jsonl
raw_completeness: full
---

# Ghost predators: inactive count-slots roam and render because only the agent-facing channels were masked

## Key conclusion
Under per-episode entity-count variance, each episode draws K active animals and parks the remaining `count_high - K` slots off-grid at `(height, width)` AT RESET. But `update_animals`/`_hunt_step`/`_wander_step` move EVERY slot each step with no `animal_active` gate, so an inactive predator un-parks (the hard-grid clip pulls it back in-bounds) and chases the agent. Damage (core.py:539), nociception (sensor.py:78), olfaction (sensor.py:33), distance (core.py:670), and the visual channel (sensor.py:229) ARE all gated by `animal_active`, so the inactive predator deals 0 damage and is completely invisible to the agent — but it is still MOVED and RENDERED. Net: a "ghost" predator that visually sticks to the agent dealing no damage (the user's bug report). Fix: after the movement updates in `update_animals`, re-park inactive slots with `new_pos = jnp.where(state.animal_active[:, None], new_pos, off_grid)` (off_grid = `[height, width]`). The lesson: in a JAX static-shape "compute-then-mask" design, masking must be applied at EVERY place an entity can leak into the world — movement/render was the one place it was missed.

## Evidence, measurements, facts
- Symptom (user, from eval video of run `ga5fkr1q` / rppo_basic05_randinit_n112 checkpoint `8900007`): predator sits on the agent, no further damage.
- Raw recording `episode_000002`: predator slot 1 on the agent's cell for steps 14-16 while injury RECOVERS 34.9->15.5->0.0 (0 damage). Synthetic isolation (no active animals, no obstacles): an inactive predator walked onto the agent's cell and sat there 12 steps at dmg=0.
- Severity = PURELY COSMETIC. Clean isolation (identical config, only code differing): survival byte-identical `[119,501,501,26,2,174,501,85,501,273,15,501]` before vs after, ghost-contact steps `1514 -> 0` across 12 episodes. Overlap resolution runs only at reset (core.py:945), not per step, so ghosts never displace active animals. So the learned policy / training signal is UNAFFECTED; only rendered eval videos showed ghosts.
- Fix commit `3634887` (core.py re-park + new regression test `tests/env/test_inactive_animal_offgrid.py`). Parity suite green; one fixture (`configs__environment__default.npz`) re-captured because its diff was confined to the inactive slot's distance (10.0 buggy -> 11.40 correct off-grid).
- Methodology note: the first before/after was confounded because I co-committed a predator damage-range change [15,120]->[15,45] (later REVERTED, `440382c` — the [15,120] was intended) AND a too-broad sed in the control. A clean isolation holding config identical settled it. Change ONE variable per comparison.

## Decisions and actions
- Shipped the re-park fix (`3634887`); reverted the unintended damage change (`440382c`).
- The running training `ga5fkr1q` (trained with the buggy code) does NOT need restarting — the bug is cosmetic, its learned policy is unaffected.
- Extends [[20260623_0143_per_episode_count_variance_masking]] (the feature that introduced the masks). Same "mask must cover every leak path" lesson as [[20260623_0144_inactive_resource_slots_revive_respawn]] (where the missed path was the resource respawn, here it is animal movement/render).

## Open questions and follow-ups
- None — fixed, tested, verified before/after with video.

## References
- Fix: commit `3634887`; revert of damage change: `440382c`. Regression test: `tests/env/test_inactive_animal_offgrid.py`.
- Code: `src/environment/core.py` (update_animals ~330-418 re-park; damage gate :539; reset park :1164-1171), `src/environment/sensor.py` (:33 olfaction, :78 noci, :229 visual gating).
- Before/after eval + videos: `results/eval/basic05_ghostfix/` (before/, after/, after_exactbefore/, SIDEBYSIDE/compare_ep1.mp4, original preserved).
- Raw conversation: synced via `./sync-agent-data.sh claude push`. To read on another node: `./sync-agent-data.sh claude pull`, then `claude --resume 428cfe9b-4b3e-4d99-8065-98b46ee41016` or `python scripts/claude_jsonl_to_md.py <jsonl> /tmp/<id>.md`.

<!-- BACKLINKS — auto-generated by scripts/regen_memory_graph.py; do not edit -->
## Backlinks
- [[20260630_1630_predator_params_per_episode_ranges]] (env_entities, 2026-06-30) — Predator behavioural params are per-episode randomizable via a [lo,hi] range. 5 
<!-- END BACKLINKS -->
