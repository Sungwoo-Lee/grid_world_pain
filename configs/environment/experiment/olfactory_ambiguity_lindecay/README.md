# olfactory_ambiguity_lindecay — linear-decay re-run (decay_power = 1.0)

10 sparse configs (each nested-`extends:` its `olfactory_ambiguity/` twin) that vary **only the predator/rabbit smell**, to
map when the agent can discriminate the dangerous predator from the harmless rabbit. Everything else
is the cell-08 contrast, identical across all 10: **1 predator + 1 rabbit**, byte-identical matched
chase + disengage, **lethal predator `damage [5,120]`**, harmless rabbit, contact-only vision,
randomized starts (`nutrition [10,100]`, `injury [0,80]`), abundant food (8 sources), 10×10 grid,
seed 42, ~10M episodes. Olfactory distance decay = **1.0** (linear). Each config nested-`extends:` its `olfactory_ambiguity/` twin and overrides ONLY `sensory.decay_power: 1.0`, so smell carries farther (advance warning for pre-emptive avoidance).

**Smell scheme (symmetric, mirrored around 0.5):**
predator `properties = [0, 0.5+s, 0.5-s, 0, 0]`, rabbit `= [0, 0.5-s, 0.5+s, 0, 0]`,
`properties_std = [0, σ, σ, 0, 0]` (per-episode jitter on ch2 & ch3 of both).
`s` = mean separation (0 = identical → 0.5 = orthogonal/maximally distinct);
`σ` = per-episode noise (0 = fixed/learnable → 0.4 = strongly ambiguous each episode).

| # | file | s | σ | predator `properties` | rabbit `properties` | `properties_std` |
|---|---|---|---|---|---|---|
| 01 | `01-s0_sig0.yaml` | 0 | 0 | `[0, 0.5, 0.5, 0, 0]` | `[0, 0.5, 0.5, 0, 0]` | `[0, 0, 0, 0, 0]` |
| 02 | `02-s0.05_sig0.yaml` | 0.05 | 0 | `[0, 0.55, 0.45, 0, 0]` | `[0, 0.45, 0.55, 0, 0]` | `[0, 0, 0, 0, 0]` |
| 03 | `03-s0.1_sig0.yaml` | 0.1 | 0 | `[0, 0.6, 0.4, 0, 0]` | `[0, 0.4, 0.6, 0, 0]` | `[0, 0, 0, 0, 0]` |
| 04 | `04-s0.25_sig0.yaml` | 0.25 | 0 | `[0, 0.75, 0.25, 0, 0]` | `[0, 0.25, 0.75, 0, 0]` | `[0, 0, 0, 0, 0]` |
| 05 | `05-s0.5_sig0.yaml` | 0.5 | 0 | `[0, 1.0, 0.0, 0, 0]` | `[0, 0.0, 1.0, 0, 0]` | `[0, 0, 0, 0, 0]` |
| 06 | `06-s0.1_sig0.2.yaml` | 0.1 | 0.2 | `[0, 0.6, 0.4, 0, 0]` | `[0, 0.4, 0.6, 0, 0]` | `[0, 0.2, 0.2, 0, 0]` |
| 07 | `07-s0.25_sig0.2.yaml` | 0.25 | 0.2 | `[0, 0.75, 0.25, 0, 0]` | `[0, 0.25, 0.75, 0, 0]` | `[0, 0.2, 0.2, 0, 0]` |
| 08 | `08-s0.5_sig0.2.yaml` | 0.5 | 0.2 | `[0, 1.0, 0.0, 0, 0]` | `[0, 0.0, 1.0, 0, 0]` | `[0, 0.2, 0.2, 0, 0]` |
| 09 | `09-s0.1_sig0.4.yaml` | 0.1 | 0.4 | `[0, 0.6, 0.4, 0, 0]` | `[0, 0.4, 0.6, 0, 0]` | `[0, 0.4, 0.4, 0, 0]` |
| 10 | `10-s0.5_sig0.4.yaml` | 0.5 | 0.4 | `[0, 1.0, 0.0, 0, 0]` | `[0, 0.0, 1.0, 0, 0]` | `[0, 0.4, 0.4, 0, 0]` |

Rows 02–05 sweep separation at zero noise (reducible/learnable); 06–10 add irreducible per-episode
noise at two levels. `s = 0.5` (05/08/10) is the orthogonal "fully separable" anchor (its `1.0`/`0.0`
is max separation, not danger). At `s = 0.5` with `σ > 0` (08/10) the means sit on the [0,1] clip
edge, so the jitter is one-sided — treat those as anchors, not onset rungs.

**Paired baseline:** [`../olfactory_ambiguity/`](../olfactory_ambiguity/) — identical smell, `decay_power = 2.0` (the steep-decay arm).
**Design + results:** `docs/experiments/active/hypervigilance/20260620_hunger_gated_step1_linear_olfactory_decay.md`.
