# Active Publication-Track Portfolio

> **Status: not yet ratified.** This file is the source of truth for *what papers the project is currently building toward*. The PI fills it in during the first PI call where the user names the active tracks; until then, every other PI call is structurally illegible because there is no portfolio to anchor against.

## Purpose

Most PI decisions are some variant of "does this fit one of the active tracks, open a new track, or distract from a track?" That question requires the active tracks to be named. This file names them in one place so the PI, the user, and any future-Claude session has a single anchor.

## Healthy default

- **1–2 active publication tracks.** Each track is one paper-shaped question with a target venue and a stop-rule.
- **Explore-buffer ≤ 20% of effort.** Threads that don't fit either track but are worth a low-cost probe live in the explore buffer, not promoted to a track until they earn the slot.
- **Three or more tracks** is a flag the PI raises explicitly — it is sometimes correct (e.g., a near-finished paper plus the next paper plus an exploration), but more often a sign that the project is leaking effort.

## Tracks (to be filled by first PI call)

| Track | Paper-shaped question | Target venue | Stop rule | Owner | Last reviewed |
|---|---|---|---|---|---|
| Track A | _TBD_ | _TBD_ | _TBD_ | — | — |
| Track B | _TBD_ | _TBD_ | _TBD_ | — | — |

## Explore buffer (≤ 20%)

| Thread | One-line question | Why it's not (yet) a track | Promote-or-drop date |
|---|---|---|---|
| _TBD_ | — | — | — |

## Recent calls

(Newest first. Calls that touch the portfolio but pre-date track ratification anchor here until tracks are named.)

| Date | Call | One-line outcome | Call log |
|---|---|---|---|
| 2026-05-14 | D-013 / parity-launch disposition (single-GPU OOM on the dreamer-srl "XS" config) | User picked "Go with XS" — config-correction (not substrate-change): the 14.38 GB OOM was an XL-equivalent config mis-labeled "XS default"; fix `01_food_only.yaml` to mirror real sheeprl XS (256/256/mlp_layers=1/cnn_multiplier=24); single-GPU is the natural substrate, multi-GPU + gradient-checkpointing both rejected; senior-developer scopes the config-fix + correction-note sweep next | [2026-05-14_d013_parity_launch_disposition](calls/2026-05-14_d013_parity_launch_disposition.md) |
| 2026-05-12 | Dreamer backend: in-house JAX rebuild vs. sheeprl-direct | User picked sheeprl-direct (minimal bridge); `dreamer-srl` v2 plan and 6 reviewer files shelved; `senior-developer` drafts sheeprl-bridge integration plan next | [2026-05-12_dreamer_backend](calls/2026-05-12_dreamer_backend.md) |

## Recent portfolio changes

(Newest first. Each row links to the call log that ratified the change.)

| Date | Change | Call log |
|---|---|---|
| _TBD_ | _Initial portfolio draft — active tracks still pending PI call_ | _TBD_ |

## PI follow-up queue

- **First-track ratification.** The active publication tracks have never been formally named. Until they are, every other PI call (including the 2026-05-12 dreamer-backend call above) is structurally illegible — there is no portfolio for the call to anchor against. The neuromodulation / pain-modeling paper is the obvious candidate for Track A. A separate PI call to ratify Track A (and decide whether a Track B exists) should land soon.
