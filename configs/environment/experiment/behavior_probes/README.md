# Behavior-probe configs — two tiers

Probe environments for the interoceptive behavior-measure study (read the anchor doc:
`docs/experiments/active/behavior_measures/interoceptive_behavior_measure_study.md`).

Configs are split by how stable/maintained they are:

- **`core/`** — *canonical*. Probe sets that have produced a validated behavioral measure and
  are maintained for the whole project. Stable, but still updatable. Treat as load-bearing:
  changing one means re-checking the measure it underpins.
  - `forage_nutrition/` — Phase 1a. Controlled foraging vs. starting nutrition (validated the
    *departure-delay* satiety measure). Subject model: `basic05_randinit_n112` ckpt 8900007.

- **`explore/`** — *exploratory*. Iterative / one-off probes that may be dropped. Not maintained;
  do not build new work on these without first promoting them to `core/`. Several encode
  directions or findings that were later superseded (see the linked memory insights).
  - `conflict/` — eat-vs-flee spatial probe (earlier direction).
  - `hypervigilance/` — predator-vs-rabbit probes; the "hypervigilance" read was reversed as an
    off-distribution artifact (`docs/memory/memories/hypervigilance/20260624_0517_*`).
  - `nutrition_sweep/` — decay-1.0 sweep for the static-init model (off-distribution).
  - `nutrition_sweep_d2/` — decay-2.0 random-init sweep; superseded by `core/forage_nutrition/`.

**Promotion rule:** a probe set graduates `explore/` → `core/` only once it has produced a
measure we trust and intend to keep. When promoting, `git mv` the folder and fix any `extends:`
paths + doc references (they are config-root-relative and will silently fall back to defaults if
stale).
