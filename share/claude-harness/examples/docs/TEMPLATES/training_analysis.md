# [Experiment Title]

> **Status**: [COLLECTING | ANALYZING | COMPLETE]
> **Date**: [date]
> **Author**: [who performed this analysis]
> **Related**: [links to prior analyses, issue docs, or design docs]

---

## 1. Research Question

State the specific question this experiment answers. Frame as a testable hypothesis.

This section is the doc's plain-language entry point per CLAUDE.md "Documentation framing". A reader without prior context should be able to read this section alone and understand what the experiment is asking, why it exists, and what would count as a positive vs. negative result. Open with one paragraph in plain prose, *then* state the formal hypothesis.

**Plain-language framing example.** "Does the modulated agent survive longer than the unmodulated baseline when the world's noise pattern shifts heterogeneously between sensory channels? The modulator's job is to dynamically reweight which channels the policy reads — so if it works anywhere, it should work here."

**Formal hypothesis (after the plain-language framing):**

> **H₀** (null — *the modulator does not help*): [e.g., "Modulation type has no effect on survival steps compared to the unmodulated baseline."]
> **H₁** (alternative — *the modulator helps*): [e.g., "Multiplicative modulation with h=16 produces higher survival steps than the unmodulated baseline."]

<!-- Plain-language framing rule per CLAUDE.md "Documentation framing":
     - Translate the formal hypothesis on first mention. The English description
       comes BEFORE the H₀ / H₁ symbols, not after.
     - No bare WandB run IDs in this section, no bare config paths, no
       project-internal jargon (Cand. A1, Phase 0, T/P split, etc.) without a
       one-clause translation. Path-shaped detail moves to §3 Launch Manifest.
     - Multiple hypotheses are fine — number them H₁a, H₁b, etc., and translate
       each on first mention. -->


## 2. Experimental Design

### 2.1 Independent Variables

What is deliberately varied across runs.

| Variable | Values | Rationale |
|----------|--------|-----------|
| | | |

### 2.2 Controlled Variables

What is held constant across all runs. **List explicitly** — implicit "same config" is insufficient.

```yaml
# Paste the shared config block or reference the exact file + commit hash
```

### 2.3 Confounds & Limitations

Known factors that differ unintentionally or cannot be controlled.

| Confound | Affected Runs | Severity | Mitigation |
|----------|---------------|----------|------------|
| | | Low/Med/High | |

<!-- Be honest. Unacknowledged confounds invalidate conclusions.
     Common confounds: activation mismatch, different random seeds,
     hardware differences, different training durations. -->

## 3. Launch Manifest

System-of-record for every training run in this experiment. **Owned jointly:**
- `experiment-designer` writes the planned columns when authoring the doc.
- `training-runner` fills the actual columns at launch time, in place.
- `experiment-analyzer` reads the table to find WandB folders.

When the manifest exists, the runner's default tag/wandb-name convention is **overridden** by the planned values below. The runner never invents tags for runs in a manifest — it uses what's written here.

| Run | Status | Cell | Tag (= wandb-name) | wandb-group | wandb-job-type | Seed | Node | GPU | Launched at | WandB run ID | Log path |
|-----|--------|------|--------------------|-------------|----------------|------|------|-----|-------------|--------------|----------|
| 1 | planned | | | | prod | 0 | — | — | — | — | — |
| 2 | planned | | | | prod | 1 | — | — | — | — | — |

**Column semantics:**
- **Run** — sequential integer, unique within this manifest.
- **Status** — `planned` → `running` → `completed` / `failed` / `cancelled`.
- **Cell** — short slug for the experimental condition (e.g., `NoPred`, `PredInterval3`). Multiple Runs can share a Cell when they differ only by Seed.
- **Tag (= wandb-name)** — identical values for both. Designer's responsibility. Format: `<algo>_<config_stem>_s<seed>` (or include `_n<node>` if the node identity is meaningful for the experiment, e.g., a per-node systems study).
- **wandb-group** — top dir under `configs/experiment/` typically (e.g., `basic`, `hypervigilance`). Same group for all rows in one experiment.
- **wandb-job-type** — `prod` by default; `debug`/`pilot`/`test`/`ablation` when the run isn't a production run.
- **Seed** — integer.
- **Node / GPU / Launched at / WandB run ID / Log path** — runner fills at launch. Launched at uses ISO format (e.g., `2026-05-07T15:30:10`). Log path is `logs/<ts>_<tag>.log`.

### 3.1 Configs to Produce (designer-only, pre-launch)

| Run | Config (env) | Config (agent) |
|-----|--------------|----------------|
| 1 | `configs/experiment/<topic>/<file>.yaml` | `configs/models/<file>.yaml` |
| 2 | `configs/experiment/<topic>/<file>.yaml` | `configs/models/<file>.yaml` |

<!-- This sub-table maps each manifest Run to the exact YAMLs that train.py will load.
     Often all rows share the same agent config and differ only by env config or seed. -->

## 4. Results

### 4.1 Primary Metrics (Hypothesis Test)

The metrics that directly address the hypothesis in §1.

| Run | Metric₁ (mean ± std) | Metric₂ (mean ± std) | ... |
|-----|----------------------|----------------------|-----|
| | | | |

> **Verdict on H₁**: [Supported / Not supported / Inconclusive] — [one-sentence justification with effect size or confidence if available]

<!-- Report steady-state (SS) values with the window used (e.g., "last 20% of training").
     If reporting trajectory trends, specify start and end windows.
     Always include variance — a mean without spread is meaningless. -->

### 4.2 Secondary Metrics

Metrics that provide context but don't directly test the hypothesis (e.g., training speed, loss curves, gradient norms).

| Run | Metric | Value (SS) | Note |
|-----|--------|------------|------|
| | | | |

### 4.3 Diagnostic Metrics

Architecture-specific internals (e.g., modulator gate values, attention weights, hidden state statistics). These diagnose *how* the agent achieves its performance, not just *what* it achieves.

<!-- Use subsections (####) for each diagnostic category.
     Always interpret raw values — e.g., "sigmoid(-4.13) ≈ 1.6%, meaning the gate is near-closed." -->

### 4.4 Learning Dynamics

How metrics evolve over training, not just final values.

<!-- Key patterns to look for:
     - Convergence speed (when does the metric plateau?)
     - Monotonic vs oscillating trajectories
     - Phase transitions (sudden jumps or collapses)
     - Early vs late training behavior differences -->

## 5. Analysis

### 5.1 Key Findings

Number each finding and tie it back to the hypothesis or a specific metric.

<!-- Structure each finding as:
     **Finding N — [title]**
     What: [what the data shows]
     Why: [proposed causal mechanism]
     Evidence: [which metrics/runs support this]
     Confidence: [High/Medium/Low] -->

### 5.2 Cross-Run Comparisons

Pairwise or grouped comparisons that isolate the effect of each independent variable.

<!-- Use controlled comparisons: change ONE variable at a time.
     E.g., "Comparing runs A vs B (only grouping_size differs): ..." -->

### 5.3 Failure Modes & Pathologies

Any degenerate, collapsed, or unexpected behaviors observed.

<!-- For each pathology:
     - Which runs are affected?
     - When in training does it appear?
     - What is the likely cause?
     - Does it correlate with performance? -->

## 6. Conclusions

### 6.1 Summary

<!-- 3-5 bullet points. Each should be a standalone statement a reader can act on.
     Separate what the data SHOWS from what you INFER. -->

### 6.2 Limitations & Open Questions

What this experiment cannot answer. What remains ambiguous.

### 6.3 Recommended Next Experiments

| Priority | Experiment | Rationale | Effort |
|----------|-----------|-----------|--------|
| | | | |

<!-- Prioritize by: (1) what blocks progress, (2) what resolves ambiguity, (3) what explores new directions.
     Each experiment should be actionable — specific enough to set up without further design work. -->

---

## Appendix

### A. Raw Data Tables

<!-- Full metric dumps, extended time series, or per-run breakdowns too verbose for the main text.
     Reference from main sections as "see Appendix A.1". -->

### B. Config Diffs

<!-- Full YAML diffs between runs if not captured in §3. -->

### C. Changelog

| Date | Change | Author |
|------|--------|--------|
| | Initial analysis | |

<!-- Track revisions as new data arrives or analysis is updated.
     This prevents confusion when a document is revisited weeks later. -->
