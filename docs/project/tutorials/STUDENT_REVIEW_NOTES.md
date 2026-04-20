# Student's Review: What's Still Unclear or Missing in the Interactive HTML

> **Purpose**: Working notes from reviewing `interactive_pain_hypervigilance.html` from the perspective of a graduate student encountering the material for the first time.
> **File reviewed**: `docs/project/tutorials/interactive_pain_hypervigilance.html`
> **Source tutorial**: `docs/project/tutorials/computational_pain_hypervigilance_tutorial.md`
> **Created**: 2026-04-17
> **Status**: Identified gaps — not yet implemented. Use this as the implementation checklist.

---

## 📌 Global / Cross-Cutting Issues

### 1. No "Big Picture" introduction at the top
When landing on the page, the user sees the title and four visualizations with no orientation. Missing:
- **Why hypervigilance matters** as a research topic (clinical burden, economic cost, personal suffering)
- **How the four sections connect** — Bayes → DDM → LQG → Active Inference is a nested hierarchy (active inference subsumes the others), but this is never stated upfront
- **What the student should walk away understanding**
- **Reading order** — do the sections depend on each other? Are they independent?

**Suggested fix**: Add an "Introduction / How to Use This Page" section before Viz 1 with a flowchart showing the relationship of the 4 frameworks.

### 2. No glossary / cheat sheet
Students encounter dozens of technical terms (nociception, interoception, vmPFC, PAG, NPS, S1, S2, DLPFC, efference copy, Kalman gain, innovation, precision, likelihood, posterior, generative model, variational, exteroceptive) — some defined, some not. A sticky sidebar or expandable glossary would help readers look up terms without losing context.

### 3. No "What is hypervigilance?" dedicated introduction
Hypervigilance is in the title but never gets its own explanation. Readers learn about it piecemeal in each Background box. The clinical definition (Crombez et al., 2005: attentional phenomenon, distinct from sensitization) is buried in text.

---

## 📌 Viz 1: Bayesian Pain Inference

### 4. The forest/rustling/snake-vs-wind analogy is missing
The tutorial uses a concrete everyday example to introduce Bayesian inference. The HTML jumps straight to precision-weighted Gaussians. For a student, the analogy is a critical scaffold.

### 5. Precision is introduced but not *motivated*
"High precision = tight distribution = high confidence" — OK, but *why* should I care that the brain uses precision rather than variance directly? The answer: **Gaussian products add precisions (not variances)**. This is flagged in the derivation but not in the conceptual text.

### 6. Units on the x-axis of the plot are unclear
Slider says "Observation o" with range 0–10, axis reads "Pain Intensity." Is this meant to be a 0–10 pain scale (like VAS)? Or arbitrary? A one-line clarification would help.

### 7. The bar chart's relationship to the main plot is not explained
The main plot shows three curves; the bar chart shows weights. Students don't immediately grasp that the bar chart is *telling them why* the posterior sits where it does. Suggested caption: "The bar chart shows how much the posterior depends on each source — when Observation Weight (K) is 0.9, the posterior is almost entirely driven by the observation."

### 8. No counterexample / contrast preset
5 presets exist, but there's no "healthy with sharp sensation" case (low prior, precise observation, observation dominates). Without the contrast, it's hard to appreciate what's abnormal about hypervigilance.

### 9. "Kalman gain K" name is confusing here
Students will wonder: "Wait, this looks like the posterior weight. Why 'Kalman' gain?" The name hints at a connection to dynamics (Section 3), but that connection isn't made explicit here. A forward-pointer would help: "we'll see in Section 3 that this scalar K generalizes to a matrix L that does the same job over time."

---

## 📌 Viz 2: Drift-Diffusion Model

### 10. SDE notation \(dZ = v\,dt + \sigma\,dW_t\) appears without warning
A student who hasn't seen Itō calculus will see "\(dW_t\)" and freeze. The prerequisites box does explain Brownian motion, but the *interpretation of \(dW_t\) as a random kick per timestep* could be clearer.

**Suggested fix**: Show the discrete-time form side by side:
$$Z_{t+\Delta t} - Z_t = v\Delta t + \sigma\sqrt{\Delta t}\cdot\varepsilon_t, \quad \varepsilon_t \sim \mathcal{N}(0,1)$$

### 11. RT histogram reading is not explained
Why are some RTs plotted in red and others in blue? What is the student supposed to learn by looking at the shape? The visual is pedagogically rich but not explained. Suggested callout: "Notice how shifting the starting point shifts the *relative height* of the two histograms but not their *shape* — this is the signature of decisional bias vs. sensory gain."

### 12. What happens when starting point is outside the boundaries?
Slider for \(z\) goes 0.05 to 2.95, but if \(a = 0.5\) and \(z = 2.95\), the starting point is already above the upper boundary. Does the code clamp? Does it instantly decide? This matters pedagogically.

### 13. Urgent Threat preset doesn't actually animate boundary collapse
Urgent Threat sets `urgency = 0.4`, but the student can't *see* the boundary shrinking over time in the plot — the boundary lines remain horizontal. This is a missed pedagogical opportunity — the whole point of urgency-gating is that the bound moves. **This is essentially a bug in the visualization.**

### 14. The connection between Viz 1 and Viz 2 is stated but not *demonstrated*
Text says "starting point \(z\) corresponds to log prior odds" — but students can't verify this by playing with both visualizations. A worked example: "In Viz 1, setting prior mean = 8 with tight variance gives log prior odds of X; in Viz 2, this corresponds to starting point z = ..." would cement the connection.

---

## 📌 Viz 3: LQG Control Loop

### 15. Jump from scalar Bayes (Section 1) to multivariate state-space is abrupt
Suddenly there are matrices \(A, B, C, D, W, V\). The HTML says "simplify to scalar (1D)" — but the equations shown are written as if \(x\) is a vector. Either commit to scalar everywhere (lowercase \(a, b, c, ...\)) or explain that the plot is a 1D special case of the general framework.

### 16. What does the "injury state" y-axis actually mean?
Y-axis reads "Injury Severity" with range 0–1.3. Is 1.0 "fully injured"? Normalized? Students wonder what *real-world quantity* this is. Suggested clarification: "Think of this as the inflammation level, normalized so 0 = fully healed and 1 = peak acute injury."

### 17. The innovation plot is critical but has no interpretation guide
The whole point of the LQG visualization is that **the innovation signal flatlines when guarding is on**. But without instruction, a student might think "the plot just shows a noisy line, so what?"

**Suggested callout**: "Notice: when you toggle Guarding ON, the innovation signal collapses toward zero. The brain is no longer receiving useful updates — this is the mechanism of the chronic pain trap."

### 18. Kalman gain L on the third plot is unexplained
Students see L going up or down but don't know what it means visually. "L near 1: the estimator trusts observations; L near 0: the estimator ignores observations." Also, does L ever actually change in the simulation, or is it constant (steady-state)?

### 19. B = -0.1 is hardcoded but not exposed
Text says "rest accelerates healing" but the user can't play with this. A student who wants to ask "what if rest *slows* healing?" (e.g., aged patients with muscle atrophy) has no slider for B.

### 20. "Guarding" in code vs. real-world rest action is unclear
The toggle says "Guarding (restricts information)" but the physical mechanism — the agent is physically immobilizing the injured body part — is not stated. Students may not realize what guarding *is*.

### 21. No preset demonstrates meta-controller / Q/R rebalancing
The background mentions "in a safe shelter, R goes up; under predators, R goes down" — but there's no preset for this. Adding a "Threat Context" preset where R drops and the agent stops guarding despite being injured would illustrate meta-control.

---

## 📌 Viz 4: Active Inference

### 22. EFE formula is scary, intuition is easily lost
\(G_\pi = D_{\text{KL}}[q(o|\pi) \| p(o|C)] + \mathbb{E}_{q(s|\pi)}[H[p(o|s)]]\) — the KL and entropy terms have five nested things. Students read this and feel lost even if the background text is clear.

**Suggested fix**: Side-by-side English translation with icons:
- **Risk** = How much the outcomes of this policy differ from what I want.
- **Ambiguity** = How uncertain I am about what I'll see if I do this policy.

### 23. Policies are preset and opaque
The code hardcodes three likelihood matrices for Explore, Guard, Mixed. But students see no visualization of what these matrices *are*. Suggested fix: show the A matrices as 2x2 tables for each policy so students see *why* Guard has higher ambiguity (its observations are uninformative) than Explore.

### 24. "Tissue heals at step 12" annotation is cryptic
Students see a dashed line at step 12 labeled "Tissue heals" and think "wait, the tissue decides to heal on its own halfway through? Where does 12 come from?" The true state schedule is hardcoded but not exposed. Should be explained or made adjustable: "This simulation models a patient whose tissue finishes healing at step 12. Watch whether each policy's beliefs about injury correctly decrease after that."

### 25. "Epistemic action" and "risk-dominated EFE" are named but not connected to the plot
The background text uses these terms. The student can't directly see which term dominated in a given preset. Add a text indicator: "Currently, risk / ambiguity is the dominant term for the selected policy."

### 26. No link to pharmacology or treatment
If the message is "chronic pain is an active-inference trap," the natural next question is: "how do we unstick it?" The tutorial mentions CBT, gradual exposure, placebo — none appears in the HTML.

**Suggested closing callout**: "Treatments like graded exposure therapy are effectively forcing the agent into 'Explore' policies, generating new prediction errors that can update the stuck prior."

---

## 📌 Pedagogical Structure Issues

### 27. All four sections start expanded (`<details open>`)
This overwhelms the student on first load. Only the first section should be open by default, or the student should be greeted by a summary card that introduces the journey.

### 28. No "Check your understanding" prompts
A serious student would benefit from prompts like: "*Before reading further, predict: if you double the prior precision, what happens to the Kalman gain K?*" Then they can test it with the slider. This transforms passive reading into active engagement.

### 29. No recap / synthesis at the end
After all four visualizations, the page just ends with a footer. A closing "Synthesis" section tying everything together — *hypervigilance is characterized by specific parameter configurations across all four models* — would give students a takeaway framework.

### 30. No references / further reading links
The text cites "Wiech, 2016" and "Seymour et al., 2023" repeatedly but doesn't provide full citations or links. For a student who wants to dive deeper, this is frustrating.

---

## 🎯 Top 5 Priorities

| # | Priority | Why |
|---|---|---|
| 1 | Add top-level "Introduction / Roadmap" with 4-framework hierarchy | Students need the big picture before diving in |
| 2 | Add dedicated "What is Hypervigilance?" section near the top | It's in the title but never gets its own treatment |
| 3 | Fix DDM Urgent Threat preset to actually show boundary collapse | Currently silent on the key animation |
| 4 | Add interpretation callouts to LQG innovation plot | The "aha" moment of Section 3 is unmarked |
| 5 | Add closing "Synthesis: Hypervigilance Across the Four Models" section | Gives the student a take-home unified parameter vector |

**Recommended implementation order**:
1. First pass: #1, #2, #5 (scaffolding that makes everything cohere)
2. Second pass: #3, #4 (specific pedagogical fixes within existing sections)
3. Third pass: Items 6–26 (per-section improvements)
4. Fourth pass: Items 27–30 (polish and meta-structure)

---

## 📋 Additional Proposed Features (Beyond Priority 5)

### A. Parameter Dashboard Linking All Four Visualizations
Show a single unified "Hypervigilance parameter vector" at the top of the page that updates as the user plays with different visualizations. E.g.:

```
Current Hypervigilance Fingerprint:
  Prior precision:     ████░░ (0.4)
  Sensory precision:   ██░░░░ (0.2)
  DDM boundary:        █░░░░░ (0.1)
  DDM start bias:      █████░ (0.8)
  Sensitization (C):   █████░ (0.8)
  Information flow:    █░░░░░ (0.1)
  Preference prec:     ████░░ (0.5)
```

### B. "Clinical Case Study" Mode
Toggle that loads preset configurations corresponding to real clinical populations: healthy adult, acute injury, fibromyalgia, chronic low back pain, CRPS, phantom limb pain.

### C. Export / Screenshot Buttons
Let students export their parameter configurations and plot snapshots for lab notebooks or reports.

### D. "Next Step" Navigation
At the end of each viz, a button "Continue to Section 2" with a brief preview of what's next.

---

## Implementation Notes

- All fixes should maintain the single-file HTML architecture (no new dependencies).
- New sections should use the existing CSS design system (color palette, collapsibles, concept/callout/example boxes).
- Cross-references between sections should use existing anchor IDs (#bayesian, #ddm, #lqg, #active-inf).
- New presets should follow the existing `applyPreset(name)` pattern.
