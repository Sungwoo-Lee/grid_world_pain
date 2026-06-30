---
name: trajectory-story
description: "Step-level, story-level QUALITATIVE analysis of an RL agent's behaviour from eval-rollout recordings (the .rec.gz files written by scripts/eval_rollout.py --record). Use when the question is 'what does the agent ACTUALLY do' rather than 'what do the summary metrics say' — e.g. 'tell the story of this run', 'read episode X step-by-step', 'does the agent avoid/approach/flee Y before contact', 'trajectory-level / qualitative behaviour read', 'why does the agent do Z', or when aggregate measures (mean distance, M1/M2/M5, flee-rate) seem to contradict what the video shows. Also use to inspect the RAW observation vector the agent receives (e.g. visual-channel counts). Distinct from wandb-analysis (metric curves over training) and experiment-analyzer (verdict-level results write-up): this is the per-step, per-episode microscope. Calls scripts/trajectory_story.py."
---

# Trajectory-story — step-by-step, story-level behaviour analysis of eval recordings

Read an RL agent's behaviour at the **individual-trajectory and raw-observation level**, the way a careful human watches the video — not through episode-averaged statistics. This skill is the microscope that catches what aggregates hide.

## When to use

Trigger when the user (or your own analysis) wants:
- "Tell me the story of episode X / this run" — a narrative walk-through.
- "What does the agent actually do when <event>?" — step-by-step behaviour around an event.
- "Does the agent avoid / flee / approach <entity> **before contact**?" — pre-contact behaviour.
- A check on whether a **summary metric** (mean distance, M1/M2/M5, eat-rate, flee-rate) matches the actual trajectories — especially when the video seems to contradict the numbers.
- To **decode the raw observation** the agent receives (e.g. "what does the agent actually see?" — visual-channel counts, olfaction, nociception).

Do NOT use for: metric curves over training (`wandb-analysis`), or the verdict-level results write-up into a design doc (`experiment-analyzer`). This skill feeds those — it produces the microscopic evidence; the analyzer writes the verdict.

## The methodology (why this skill exists — read first)

Hard-won lessons from the chasing-rabbit study (memory: `20260609_1721_aggregate_stats_hide_conditional_behavior`, `20260609_1747_avoidance_is_post_contact_not_preemptive`). Internalise these BEFORE trusting any aggregate:

1. **Inspect individual trajectories + the RAW observation vector EARLY.** Reason about what the agent *actually receives and does*, not what the obs/code "should" contain. The decisive evidence (a visual-channel count) sat in the recorded obs the whole time while several aggregate analyses pointed the wrong way.
2. **Aggregates hide conditional behaviour.** Episode-mean distance, fresh-vs-fresh flee rates, eat-rate, M1=0 — each can average away an effect that only fires in a sub-regime. When the hypothesis is "the agent does X", **bin by the state X might be gated on**, and read trajectories where that state holds.
3. **Design a clean CONTROL before trusting a clever mechanism.** A mechanism that is sensorily plausible and behaviourally suggestive can still be wrong — the only thing that settles it is removing/swapping the relevant entity and re-evaluating (e.g. predator-only eval). The control beats the theory.
4. **A careful observer's repeated, specific, contradicting observation is evidence the MODEL is wrong**, not noise to explain away. If the user keeps reporting the same thing your model denies, look harder at the trajectories, don't keep reaching for confounds.

## The script

`scripts/eval/trajectory_story.py` — run from project root with the conda interpreter
(`/home/vncuser/miniconda3/envs/grid_world_pain/bin/python`). It reads the recordings dir
`results/eval/<run>/models/<ckpt>/recordings/<pct>/` (must contain `run_meta.pkl` + `episode_*.rec.gz`,
which `scripts/eval_rollout.py --record` writes). It is animal-layout-agnostic — it reads classes/tags from `run_meta`.

| Subcommand | What it gives |
|---|---|
| `summary <recdir>` | Per-episode survival, first-predator-contact step, and nearest-animal distance distribution (on-cell / adjacent / near / far) per class. Start here. |
| `dump <recdir> [--episodes 0,1,2] [--longest N] [--every K] [--max-steps]` | **The story view.** Step-by-step table: `t, agent_pos, action, per-animal pos+dist, injury, nutrition, in-bush, contact markers`. Default picks the N longest pre-contact episodes. `--max-steps` stops just past first contact. |
| `flee <recdir> [--fresh] [--dist-max 6]` | **Flee decomposition** — measures the agent's OWN move's effect on distance to each animal (animal position held), distance-binned, predator vs neutral. `--fresh` conditions on each animal's own pre-contact window (fresh-vs-fresh, removes the "rabbit already vetted" confound). |
| `obs <recdir> --episode 0 [--steps 0-45] [--every 2]` | **Decode the observation vector** by sensor block; prints the visual block per step (ch5=predator, ch7=neutral COUNT of on-cell animals — `visual_sensor_range: 0` means contact-only). This is how the rabbit-counting signal was found. |

## Procedure

1. **`summary`** — get the lay of the land: survival, when/how often the predator contacts, how close each class gets. Note any class asymmetry.
2. **`dump`** the longest-pre-contact episodes (and a couple of typical / early-contact ones) — read them step-by-step and **write the narrative**: where does the agent go, what does it do as a threat approaches, when does it eat/rest/flee/hide, when is it hit (injury jumps), how does it end. Quote specific steps.
3. **`flee`** (and `flee --fresh`) — quantify pre-contact avoidance the right way (agent's own move, not raw distance which conflates the chase). Remember: a `Rest` is flee=0, so high %away over tiny n can still mean "mostly sitting".
4. **`obs`** — decode what the agent actually sees at the decisive moments (e.g. is a class channel active? a count? a smell gradient?). Inference the agent *could* do (e.g. elimination by counting) lives here.
5. **Design a control** if a mechanism is suspected: re-eval with the relevant entity removed/swapped (new config + `eval_rollout.py --record`), then re-run steps 1–4. The control is what settles it.
6. **Make a short watchable clip** if the user wants to see it (see below). Hand the verdict to `experiment-analyzer` to write into the design doc; capture surprises via `/memorize`.

## Short watchable video (clips that actually open)

```bash
# 1) record episodes (small N for qualitative work)
python scripts/eval/eval_rollout.py --config <cfg> --agent_config configs/models/recurrent_ppo/recurrent_ppo.yaml \
  --checkpoint <ckpt-dir> --output-root results/eval/<name> --eval-n-episodes 20 --record --record-n-episodes 20 --device gpu
# 2) render
python scripts/eval/render_recordings.py results/eval/<name>/models/<ckpt>/recordings/<pct>/ --workers 8 --fps 5 --concat
# 3) the consolidated eval_*.mp4 can be ~17 min for 20 episodes and may not open — make a short first-5 clip (stream-copy, instant):
VD=results/eval/<name>/models/<ckpt>/videos
printf "file '%s'\n" "$PWD/$VD/<pct>/episode_00000"{0,1,2,3,4}".mp4" > /tmp/c.txt
ffmpeg -y -f concat -safe 0 -i /tmp/c.txt -c copy "$VD/eval_first5.mp4"
```

## References

- Script: `scripts/eval/trajectory_story.py` (`--help` on each subcommand).
- Recording format: `src/utils/eval_recording.py` (per-step snapshots: agent_pos, animal_pos, obs, actions, injury, nutrition, obs_pos); `--record` hook in `scripts/eval/eval_rollout.py`.
- Worked example + the lessons: chasing-rabbit study `docs/experiments/active/hypervigilance/sameprop_chasing_rabbit.md`; obs audit `docs/reviews/chasingRabbit_obs_classLeak_audit.md`.
- Memory: `docs/memory/memories/hypervigilance/20260609_1721_aggregate_stats_hide_conditional_behavior.md`, `…1747_avoidance_is_post_contact_not_preemptive.md`, `…1720_chasing_rabbit_avoidance_damage_driven.md`.
- Siblings: `wandb-analysis` (training-metric curves), `summarize-study` (study-level report). Owner agent: `experiment-analyzer`.
