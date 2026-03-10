# [Experiment Title]

> **Status**: [COLLECTING | ANALYZING | COMPLETE]
> **Date**: [date]
> **Author**: [who performed this analysis]
> **Related**: [links to prior analyses, issue docs, or design docs]

---

## 1. Research Question

State the specific question this experiment answers. Frame as a testable hypothesis.

> **H₀** (null): [e.g., "Modulation type has no effect on episode reward compared to baseline."]
> **H₁** (alternative): [e.g., "Multiplicative modulation with h=16 achieves higher reward than baseline."]

<!-- Keep hypotheses falsifiable and tied to observable metrics.
     Multiple hypotheses are fine — number them H₁a, H₁b, etc. -->

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

## 3. Run Inventory

| Label | WandB ID | Config Diff | Timesteps | Wall-Clock | Status |
|-------|----------|-------------|-----------|------------|--------|
| | | | | | |

<!-- "Config Diff" = only what differs from the shared config in §2.2.
     Include WandB project/entity if not obvious from context. -->

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
