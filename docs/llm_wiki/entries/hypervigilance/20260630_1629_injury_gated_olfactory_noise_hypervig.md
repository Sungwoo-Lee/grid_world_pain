---
id: 20260630_1629_injury_gated_olfactory_noise_hypervig
date: 2026-06-30
time: "16:29"
folder: hypervigilance
tags: [hypervigilance, design, decision, noise]
summary: "Basic level 06 design: induce hypervigilance via INJURY-GATED OLFACTORY perceptual noise (precision-weighting). When injured the threat-carrying smell channel degrades (sigma_eff = sigma_base*(1+alpha*injury/max_injury)), so under a danger prior the agent should over-react defensively. Key design rules: couple injury to the THREAT channel (olfaction) not to interoception (keep the pain-gate reliable), and maximise the healthy<->injured contrast (low sigma_base + high injury_noise_scale). Untested hypothesis."
related: ["20260609_1747_avoidance_is_post_contact_not_preemptive", "20260622_1745_discrimination_weak_lethality_masks_gating", "20260624_0517_indist_random_init_reverses_hypervig"]
session_origin: claude_code
session_label: "basic level 05/06 randomization + sensory-noise hypervigilance probe + predator-param ranges"
importance: high
status: active
valid_until: null
confidence: medium
supersedes: []
raw_source: claude_data/.claude/projects/-media-nas01-projects-Interoceptive-AI-grid-world-pain/428cfe9b-4b3e-4d99-8065-98b46ee41016.jsonl
raw_completeness: full
---

# Hypervigilance via injury-gated olfactory perceptual noise (precision-weighting)

## Key conclusion
Basic level 06 (`06-sensory_noise_10x10.yaml`, extends level 05) probes whether hypervigilance can be MANUFACTURED with sensory noise. Mechanism = precision-weighting (predictive-coding pain model): the env's `perceptual_noise` makes a channel's effective noise grow with injury, `sigma_eff = sigma_base * (1 + alpha * injury/max_injury)` (alpha = injury_noise_scale; state_dependent mode, sensor.py:277). Put that coupling on the THREAT-carrying distal channel (OLFACTION — visual_sensor_range=0 is contact-only), so when injured the agent can no longer localise the predator's smell and, under a danger prior, falls back on defensive over-reaction (avoid even ambiguous cues = hypervigilance). Two non-obvious design rules: (1) keep INTEROCEPTION clean (alpha=0) so the "I am injured" gating trigger stays reliable — the archived configs wrongly made everything state_dependent; (2) maximise the healthy<->injured CONTRAST: low sigma_base (clear when healthy) + high alpha (degraded when injured). This is a HYPOTHESIS, not a result — no training run yet.

## Evidence, measurements, facts
- Noise system: 10 modalities (injury, nutrition, satiation, interoceptive_nociception, extero_nociception, olfaction, collision, proprioception, visual, location); each has mode (none/constant/state_dependent), sigma, injury_noise_scale, clip_min/max. Order is the index source-of-truth (config_loader reads it). default.yaml carries the block with enabled:false.
- Level-06 chosen params (verified resolve): olfaction state_dependent sigma 0.15 alpha 4.0 (sigma_eff 0.15->0.75 at max injury); visual state_dependent 0.10 alpha 2.0; satiation/interoceptive_noci/extero_noci constant alpha 0; injury/nutrition sigma 0 (perfectly clean gate). Inherits the 05 random-init scene (predator/rabbit 0-2, random start nutrition/injury) — random start injury makes the agent VISIT the high-injury regime so the gated olfaction is exercised across its range.
- Archived noise configs (basic-03/04 *-noise.yaml): the `-noise` suffix just flips perceptual_noise.enabled; values were olfaction sigma 0.2 / visual 0.2 / interoception 0.1, all alpha 1.5 (uniformly state_dependent — the design flaw this insight corrects).
- The single most important knob is `olfaction.injury_noise_scale`; the rigorous version is a sweep over olfaction.{sigma, injury_noise_scale}.

## Decisions and actions
- Shipped level 06 config (commit d703309); NOT yet trained.
- Builds on the project's repeated failure to get pre-emptive hypervigilance ([[20260609_1747_avoidance_is_post_contact_not_preemptive]], [[20260622_1745_discrimination_weak_lethality_masks_gating]], [[20260624_0517_indist_random_init_reverses_hypervig]]) — sensory noise is the new candidate mechanism.
- Two pre-run TODOs flagged: predator count_low 0 means some episodes have no threat (bump to 1 for the study); a sweep on the olfaction knob is the rigorous test.

## Open questions and follow-ups
- Does injury-gated olfactory noise actually produce HYPERVIGILANCE vs. just degraded foraging? (the construct-validity question for professor-pain-modeling) — untested.
- Need a training run + behavioural read (injured-state avoidance up, healthy foraging intact).

## References
- Config: `configs/environment/experiment/basic/06-sensory_noise_10x10.yaml` (commit d703309). Noise impl: `src/environment/sensor.py` apply_perceptual_noise (:277). default block: `configs/environment/default.yaml` perceptual_noise.
- Raw conversation: synced via `./sync-agent-data.sh claude push`. To read on another node: `./sync-agent-data.sh claude pull`, then `claude --resume 428cfe9b-4b3e-4d99-8065-98b46ee41016` or `python scripts/claude_jsonl_to_md.py <jsonl> /tmp/<id>.md`.

<!-- BACKLINKS — auto-generated by scripts/regen_wiki_graph.py; do not edit -->
## Backlinks
- _no inbound links yet_
<!-- END BACKLINKS -->
