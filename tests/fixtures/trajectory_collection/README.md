# `tests/fixtures/trajectory_collection/`

`dual_format_config.yaml` is a **byte-identical copy** of

```
results/JAX_RecurrentPPO/20260609-191226_recurrent_ppo_08-singlePredRabbit_disengage_s42/models/config.yaml
```

one of the 12 real saved training configs (out of 334 scanned) that carry **both** a
modern `environment.entities:` block and a legacy `environment.predators:` /
`environment.neutral_animals:` block. Such a config cannot be faithfully reloaded: the
trainer's scene precedence changed at commit `828b77e` (2026-07-23), so reloading may
rebuild the scene the trainer *discarded*, and nothing in the run directory records which
branch was taken.

It lives here — rather than being read from `results/` — because `results/` is gitignored
NAS data, so a test that depends on it cannot run on a fresh clone. The corpus-wide
variant of the same check (all 12 runs) is kept as a `skip-if-absent` test.

Consumer: `tests/test_trajectory_collection.py` (V10, the scene-ambiguity guard).
Contract: `docs/environment/TRAJECTORY_STORE_SCHEMA.md` §6 "Applicability boundary".
