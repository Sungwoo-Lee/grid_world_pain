# Continual learning in deep RL — from catastrophic forgetting to loss of plasticity

**Author**: professor-rl
**Date**: 2026-06-24
**Type**: Reference-library primer + annotated bibliography (foundation for a literature review)
**Folder purpose**: This is the study guide and download manifest for the `continual_learning` reference topic. The PDFs of every paper cited below go into `./sources/`. Every reference has been web-verified (title, authors, venue, and a working downloadable link) on 2026-06-24; the four entries I could *not* fully verify are flagged explicitly in §6.
**Cross-links**: [[curriculum_underperformed_baseline_plasticity_vs_budget]] (the project's own critique that this folder underpins — its §6 "missing references" list is the seed of this bibliography); [[nmn_continual_lifelong_probe]] (the project's continual-probe research direction, whose stability-gap framing this primer extends with a plasticity-loss lens).

---

## 1. Plain-language entry point

A neural network that learns one task after another suffers two *different* failures, and most people only know about the first one.

**Failure one — catastrophic forgetting (a backward, stability failure).** When you train a network on task B, it overwrites the weights that encoded task A, so its skill on the *old* task collapses. This is a failure to *retain*. It was discovered in 1989 and dominated the field for three decades; the famous fixes (Elastic Weight Consolidation, replay buffers, progressive networks) are all answers to "how do I stop the network from forgetting what it already learned."

**Failure two — loss of plasticity (a forward, plasticity failure).** This one is sneakier and was only taken seriously around 2020. After enough sequential training, a network gradually loses the ability to learn *anything new at all* — not because it is full, but because its internal machinery has quietly degraded: units go silent ("dormant"), the usable dimensionality of its features collapses ("rank collapse"), and the loss-landscape curvature the optimizer needs flattens out. Crucially, this happens *even when forgetting is not involved* — the network simply stops being able to fit the next task, no matter how much data you give it. More gradient steps do not help, because the substrate doing the learning is damaged.

These two pull in opposite directions — keeping old knowledge stable fights against staying plastic enough to learn new things. That tension is the **stability–plasticity dilemma**, and it is the organizing problem of the whole field.

**Why the field's attention shifted.** Deep reinforcement learning made loss of plasticity impossible to ignore. RL trains for a very long time on *non-stationary, self-generated* data (the agent's own changing policy keeps shifting what it sees), so the slow erosion of learning ability shows up as agents that plateau or even *decline* despite endless fresh experience — with no "old task" being forgotten. This folder traces that shift across three phases and organizes the fixes by which mechanism they repair. For why this matters to *our* project specifically — a curriculum agent that carried weights across a difficulty ladder and lost to a from-scratch baseline — see [[curriculum_underperformed_baseline_plasticity_vs_budget]].

---

## 2. Chronological field evolution

Each entry: what it claimed + what mechanism or fix it introduced. Full citation + download link in §6.

### Phase 1 — Catastrophic forgetting as "the" problem (1989 → ~2018)

The era when "continual learning" *meant* "don't forget." Solutions cluster into three families: **regularization** (penalize moving important weights), **replay** (rehearse old data), **parameter isolation** (give each task its own capacity).

- **McCloskey & Cohen (1989)** — Coined "catastrophic interference." Showed a connectionist net trained on "ones" arithmetic then "twos" arithmetic loses the ones almost completely after a single twos trial. First demonstration that sequential training in distributed networks is qualitatively unlike human gradual forgetting. *No fix — the problem statement.*
- **French (1999)** — Survey that crystallized the diagnosis: catastrophic forgetting is the flip side of distributed, overlapping representations, and is the symptom of the stability–plasticity dilemma. Named the trade-off the field still organizes around. *Conceptual framing.*
- **Progressive Neural Networks — Rusu et al. (2016)** — *Parameter-isolation* family. Freeze the old network and grow a fresh column per new task, wired to the old columns by lateral connections so prior features can transfer forward. Forgetting is *structurally impossible* (old weights never change). Cost: parameters grow with the number of tasks. Demonstrated on Atari/3D-maze RL.
- **EWC — Kirkpatrick et al. (2017)** — *Regularization* family, the canonical one. Estimates each weight's importance to past tasks via the Fisher information, then adds a quadratic penalty anchoring important weights near their old values (a diagonal Laplace approximation, applied online). "Elastic" because unimportant weights stay free to learn.
- **Synaptic Intelligence (SI) — Zenke, Poole & Ganguli (2017)** — *Regularization* family, EWC's online cousin. Accumulates each synapse's contribution to loss reduction *along the whole trajectory* (not just at task end), giving a per-weight importance computed cheaply during training.
- **iCaRL — Rebuffi et al. (2017)** — *Replay* family, for class-incremental classification. Keeps a small set of exemplars per class, learns a representation + nearest-mean classifier, and uses distillation on stored exemplars to resist forgetting. Set the template for exemplar-replay methods.
- **GEM — Lopez-Paz & Ranzato (2017)** — *Replay* family with a twist: store a few old examples, and project the current gradient so it never *increases* loss on them (an inequality-constrained gradient step). Allows positive backward transfer, not just non-forgetting.

### Phase 2 — Cracks: networks lose the ability to LEARN (≈2020–2022, RL-driven)

The realization that even with forgetting handled, sequential / warm-started training quietly *damages the learner itself*. RL surfaced it first because RL's non-stationarity is unavoidable.

- **Ash & Adams (2020)** — Warm-starting *hurts*. A network initialized from prior training generalizes **worse** than a fresh random init, *even when the training loss matches*. Mechanism: gradient imbalance — gradients from new data dwarf those from already-fit data, biasing the optimizer's path. Fix: **shrink-and-perturb** ($\theta \leftarrow \lambda\theta + \epsilon$). The first clean demonstration that a "head start" can be a liability. (Supervised, but the RL analogue is the whole reset line below.)
- **Kumar et al. (2021), "Implicit Under-Parameterization"** — In value-based RL with bootstrapping, repeatedly regressing onto your own moving targets *collapses the rank* of the value network's features — the net behaves as if it had far fewer parameters than it does, and performance drops. First paper to tie a concrete representational pathology (rank collapse) to the bootstrapping-plus-SGD interaction.
- **Berariu et al. (2021)** — A systematic study isolating *plasticity* as its own quantity: how much can a network still learn after pretraining? Showed pretrained nets often *cannot* reach a fresh net's generalization, separating the plasticity question from the forgetting question.
- **Lyle, Rowland & Dabney (2022), "Capacity Loss"** — Named **capacity loss**: training on a sequence of non-stationary targets erodes the network's ability to *fit new targets quickly* — especially damaging in sparse-reward RL. Fix: **InFeR** (Initial Feature Regularization), which regularizes a few outputs toward their init to preserve capacity.
- **Nikishin et al. (2022), "Primacy Bias"** — RL agents *overfit to early experience*, and that early overfit poisons all later learning. Fix: periodically **reset the last few layers** while keeping the replay buffer — counterintuitively, throwing away learned weights *improves* final performance. Introduced "resets" as a first-class RL tool.

### Phase 3 — Loss of plasticity as a first-class problem (2023–2024)

The phenomenon gets named, measured, decomposed, and given a capstone. Fixes proliferate.

- **Abbas et al. (2023), "Loss of Plasticity in Continual Deep RL"** — *The closest published precedent to our exact setup*: one network trained sequentially across a series of RL tasks (cycling Atari games). Sequential training substantially impairs learning on later tasks vs. fresh init. Mechanisms: **proliferating dormant/dead ReLU units** + **effective-rank collapse**. Most effective fix in their study: an **activation change to CReLU**, which keeps units responsive; resets/regularization helped less but still helped.
- **Sokar et al. (2023), "Dormant Neuron Phenomenon" / ReDo** — Names the micro-cause: neurons drift into a near-permanently inactive ("dormant") state and stop contributing, and the dormant fraction *grows* over training. Fix: **ReDo** (Recycle Dormant neurons) — periodically detect τ-dormant units and reinitialize their incoming weights (zeroing outgoing), restoring plasticity without a full reset.
- **Nikishin et al. (2023), "Plasticity Injection"** — A minimal intervention that *adds* fresh plasticity without changing the parameter count or the current predictions (it adds a freshly-initialized network whose output starts at zero). Doubles as a **diagnostic**: if injecting plasticity raises performance, the agent *was* plasticity-limited. Pinpointed Atari environments where plateaus are plasticity-caused.
- **Lyle et al. (2023), "Understanding Plasticity in Neural Networks"** — Ties plasticity loss to **loss-landscape curvature** changes, and shows it often occurs *without* saturated units — so "dead units" is not the whole story. Identifies parameterization/optimization choices that preserve plasticity.
- **Lyle et al. (2024), "Disentangling the Causes of Plasticity Loss"** — Shows plasticity loss is **several independent mechanisms** (rank, curvature, parameter-norm growth, unit saturation), and that fixing any *one* is insufficient — you must intervene on several at once. Finds **LayerNorm + weight decay** jointly highly effective. The "it's not one thing" paper.
- **Dohare et al. (2024, Nature), "Loss of plasticity in deep continual learning"** — The capstone. Standard deep learning, trained continually, *progressively and silently loses the ability to learn* — eventually performing no better than a shallow linear network — across supervised vision and RL. The loss is invisible in the current task's loss. Fix: **continual backprop**, which perpetually re-initializes a small fraction of the least-used units to keep injecting trainable diversity. Nature-level visibility made loss of plasticity a mainstream problem.

---

## 3. Why RL surfaced loss of plasticity (and supervised learning mostly didn't)

Three properties of deep RL make it the natural habitat of plasticity loss — and explain why the phenomenon was discovered *here* even though §2 shows it also afflicts supervised vision:

1. **Non-stationarity is intrinsic, not optional.** The targets a value network regresses onto are generated by *previous versions of the same network* (bootstrapping), and the data distribution is generated by the agent's *own changing policy*. The network is *always* chasing a moving target, even on a "single" task. Supervised learning on a fixed dataset never imposes this.
2. **Self-generated data couples representation damage to behaviour.** Once features collapse (rank/dormancy), the policy explores worse, which narrows the data, which further collapses features — a feedback loop with no analogue in i.i.d. supervised training.
3. **Training horizons are enormous.** RL runs span tens of millions to billions of environment steps. Slow erosions that are negligible over an ImageNet epoch budget become dominant over a billion-step RL run.

The diagnostic punchline that makes the forward/backward distinction concrete: in the canonical loss-of-plasticity demonstrations, **no old task is being forgotten** — the network is failing on the *current* task it has ample data for. Forgetting (Phase 1) cannot explain that; only a degraded *learner* can. That is precisely the regime our curriculum agent landed in (see [[curriculum_underperformed_baseline_plasticity_vs_budget]] §1.1).

---

## 4. Solution families (organized by the mechanism they repair)

The Phase-3 work makes clear there is no single cure (Lyle 2024). The fixes group by *which* damage they target:

**(a) Reset / recycle capacity — target: dormant units, rank collapse, accumulated over-specialization.**
- *Periodic resets* (Nikishin 2022) — reset the last layers, keep the buffer. Targets primacy-bias over-specialization.
- *ReDo* (Sokar 2023) — recycle only the *dormant* units. Targets the dormant-neuron count directly, more surgical than a full reset.
- *Continual backprop* (Dohare 2024) — perpetually reinitialize the least-used units. A *continuous* version of recycling; targets the slow capacity drain.
- *Plasticity injection* (Nikishin 2023) — add fresh capacity rather than reset old. Targets plasticity directly, and diagnoses whether plasticity is the bottleneck.

**(b) Freshen the initialization — target: the bad warm start / gradient imbalance.**
- *Shrink-and-perturb* (Ash & Adams 2020) — $\theta \leftarrow \lambda\theta + \epsilon$. Partially resets while keeping coarse structure; targets the warm-start generalization gap.
- *Regularize-toward-init* — InFeR (Lyle 2022) pulls a few features back toward their initial values; targets capacity loss without full reset.

**(c) Architecture / normalization — target: unit saturation, curvature, norm growth (preventive, not corrective).**
- *CReLU* (Shang 2016, used for plasticity by Abbas 2023) — concatenated ReLU keeps a unit responsive even when its "natural" sign would silence it; targets dormant units at the activation level. Strongest single fix in Abbas's RL study.
- *LayerNorm (+ weight decay)* (Lyle 2024) — normalization controls curvature and saturation; weight decay controls norm growth. Jointly target several mechanisms at once — the "intervene on multiple causes" prescription.

A useful mental model: families (a) and (b) are *corrective* (apply after damage accumulates or at a task boundary); family (c) is *preventive* (build the network so the damage accumulates more slowly). Lyle 2024's lesson is that robust continual learning usually needs *both* a preventive base (LayerNorm/weight-decay) *and* a corrective top-up (reset/recycle).

---

## 5. Adjacent threads relevant to our project

Three lines that aren't "loss of plasticity" proper but bridge directly to our setup. Each is developed in [[curriculum_underperformed_baseline_plasticity_vs_budget]].

- **Policy-entropy collapse in PPO — Cui et al. (2025).** PPO can drive the action distribution onto a near-deterministic point from which it cannot recover, because high-advantage/high-probability actions are self-reinforcing (entropy change tracks the advantage–log-prob covariance; performance obeys $R \approx -a\,e^{H}+b$). This is a *behavioural* near-absorbing failure that compounds with plasticity loss — and is the direct cause of the "eat-once-then-starve" degeneration in our easy stages. Bridges to the project's modulator-temperature-head as a candidate adaptive entropy floor.
- **Curriculum learning — when it helps vs. hurts — Narvekar et al. (2020).** The reference frame for *why* a curriculum can yield zero benefit or *negative* transfer: a curriculum is a bet that the inter-task transfer mechanism is net-positive. Our curriculum transferred a *collapsed* policy and *discarded* the belief state — net-negative. Required reading for interpreting our curriculum result.
- **Recurrent state across task switches — Caccia et al. (2022).** In a POMDP the recurrent state *is* the belief; carrying it across task boundaries can beat task-aware agents. We hard-reset it at every boundary — the opposite default — plausibly a self-inflicted cost on top of the plasticity and entropy problems.

---

## 6. Reference list (web-verified 2026-06-24)

Grouped by phase, matching §2. Each entry: exact title, authors, venue + year, downloadable PDF link, DOI where one exists. **Prefer the arXiv `pdf` link for downloading; the `abs` page links the PDF + all versions.** Verification status noted; the four imperfect cases are called out in §6.5.

### 6.1 Foundations + Phase 1 (catastrophic forgetting)

| # | Title / Authors / Venue | Download |
|---|---|---|
| 1 | **Catastrophic Interference in Connectionist Networks: The Sequential Learning Problem.** Michael McCloskey, Neal J. Cohen. *Psychology of Learning and Motivation* 24:109–165, 1989. | No open PDF (book chapter). DOI: [10.1016/S0079-7421(08)60536-8](https://doi.org/10.1016/S0079-7421(08)60536-8). See §6.5 — **flagged**. |
| 2 | **Catastrophic Forgetting in Connectionist Networks.** Robert M. French. *Trends in Cognitive Sciences* 3(4):128–135, 1999. | Author PDF: <https://www.cs.swarthmore.edu/~meeden/DevelopmentalRobotics/cat_forget.pdf> · DOI: [10.1016/S1364-6613(99)01294-2](https://doi.org/10.1016/S1364-6613(99)01294-2) |
| 3 | **Progressive Neural Networks.** Andrei A. Rusu, Neil C. Rabinowitz, Guillaume Desjardins, Hubert Soyer, James Kirkpatrick, Koray Kavukcuoglu, Razvan Pascanu, Raia Hadsell. arXiv preprint, 2016. | arXiv: <https://arxiv.org/abs/1606.04671> · PDF: <https://arxiv.org/pdf/1606.04671> |
| 4 | **Overcoming Catastrophic Forgetting in Neural Networks (EWC).** James Kirkpatrick, Razvan Pascanu, Neil Rabinowitz, Joel Veness, Guillaume Desjardins, Andrei A. Rusu, Kieran Milan, John Quan, Tiago Ramalho, Agnieszka Grabska-Barwinska, Demis Hassabis, Claudia Clopath, Dharshan Kumaran, Raia Hadsell. *PNAS* 114(13):3521–3526, 2017. | Open PDF: <https://www.pnas.org/doi/pdf/10.1073/pnas.1611835114> · DOI: [10.1073/pnas.1611835114](https://doi.org/10.1073/pnas.1611835114) |
| 5 | **Continual Learning Through Synaptic Intelligence (SI).** Friedemann Zenke, Ben Poole, Surya Ganguli. *ICML* 2017, PMLR 70:3987–3995. | arXiv: <https://arxiv.org/abs/1703.04200> · PDF: <https://arxiv.org/pdf/1703.04200> · PMLR: <https://proceedings.mlr.press/v70/zenke17a.html> |
| 6 | **iCaRL: Incremental Classifier and Representation Learning.** Sylvestre-Alvise Rebuffi, Alexander Kolesnikov, Georg Sperl, Christoph H. Lampert. *CVPR* 2017, pp. 5533–5542. | arXiv: <https://arxiv.org/abs/1611.07725> · PDF: <https://arxiv.org/pdf/1611.07725> · DOI: [10.1109/CVPR.2017.587](https://doi.org/10.1109/CVPR.2017.587) |
| 7 | **Gradient Episodic Memory for Continual Learning (GEM).** David Lopez-Paz, Marc'Aurelio Ranzato. *NeurIPS* 2017. | arXiv: <https://arxiv.org/abs/1706.08840> · PDF: <https://arxiv.org/pdf/1706.08840> · NeurIPS PDF: <https://proceedings.neurips.cc/paper_files/paper/2017/file/f87522788a2be2d171666752f97ddebb-Paper.pdf> |

### 6.2 Phase 2 (cracks — the learner degrades)

| # | Title / Authors / Venue | Download |
|---|---|---|
| 8 | **On Warm-Starting Neural Network Training.** Jordan T. Ash, Ryan P. Adams. *NeurIPS* 2020. | arXiv: <https://arxiv.org/abs/1910.08475> · PDF: <https://arxiv.org/pdf/1910.08475> · NeurIPS PDF: <https://papers.neurips.cc/paper_files/paper/2020/file/288cd2567953f06e460a33951f55daaf-Paper.pdf> |
| 9 | **Implicit Under-Parameterization Inhibits Data-Efficient Deep Reinforcement Learning.** Aviral Kumar, Rishabh Agarwal, Dibya Ghosh, Sergey Levine. *ICLR* 2021. | arXiv: <https://arxiv.org/abs/2010.14498> · PDF: <https://arxiv.org/pdf/2010.14498> · See §6.5 — arXiv ID inferred, **flagged**. |
| 10 | **A Study on the Plasticity of Neural Networks.** Tudor Berariu, Wojciech Czarnecki, Soham De, Jörg Bornschein, Samuel Smith, Razvan Pascanu, Claudia Clopath. arXiv preprint, 2021. | arXiv: <https://arxiv.org/abs/2106.00042> · PDF: <https://arxiv.org/pdf/2106.00042> |
| 11 | **Understanding and Preventing Capacity Loss in Reinforcement Learning.** Clare Lyle, Mark Rowland, Will Dabney. *ICLR* 2022. | arXiv: <https://arxiv.org/abs/2204.09560> · PDF: <https://arxiv.org/pdf/2204.09560> |
| 12 | **The Primacy Bias in Deep Reinforcement Learning.** Evgenii Nikishin, Max Schwarzer, Pierluca D'Oro, Pierre-Luc Bacon, Aaron Courville. *ICML* 2022, PMLR 162. | arXiv: <https://arxiv.org/abs/2205.07802> · PDF: <https://arxiv.org/pdf/2205.07802> · PMLR: <https://proceedings.mlr.press/v162/nikishin22a/nikishin22a.pdf> |

### 6.3 Phase 3 (loss of plasticity as a first-class problem)

| # | Title / Authors / Venue | Download |
|---|---|---|
| 13 | **Loss of Plasticity in Continual Deep Reinforcement Learning.** Zaheer Abbas, Rosie Zhao, Joseph Modayil, Adam White, Marlos C. Machado. *CoLLAs* 2023, PMLR 232:620–636. | arXiv: <https://arxiv.org/abs/2303.07507> · PDF: <https://arxiv.org/pdf/2303.07507> · PMLR: <https://proceedings.mlr.press/v232/abbas23a.html> |
| 14 | **The Dormant Neuron Phenomenon in Deep Reinforcement Learning (ReDo).** Ghada Sokar, Rishabh Agarwal, Pablo Samuel Castro, Utku Evci. *ICML* 2023, PMLR 202. | arXiv: <https://arxiv.org/abs/2302.12902> · PDF: <https://arxiv.org/pdf/2302.12902> · PMLR: <https://proceedings.mlr.press/v202/sokar23a.html> |
| 15 | **Deep Reinforcement Learning with Plasticity Injection.** Evgenii Nikishin, Junhyuk Oh, Georg Ostrovski, Clare Lyle, Razvan Pascanu, Will Dabney, André Barreto. *NeurIPS* 2023. | arXiv: <https://arxiv.org/abs/2305.15555> · PDF: <https://arxiv.org/pdf/2305.15555> · NeurIPS PDF: <https://proceedings.neurips.cc/paper_files/paper/2023/file/75101364dc3aa7772d27528ea504472b-Paper-Conference.pdf> |
| 16 | **Understanding Plasticity in Neural Networks.** Clare Lyle, Zeyu Zheng, Evgenii Nikishin, Bernardo Avila Pires, Razvan Pascanu, Will Dabney. *ICML* 2023, PMLR 202. | arXiv: <https://arxiv.org/abs/2303.01486> · PDF: <https://arxiv.org/pdf/2303.01486> · PMLR: <https://proceedings.mlr.press/v202/lyle23b/lyle23b.pdf> |
| 17 | **Disentangling the Causes of Plasticity Loss in Neural Networks.** Clare Lyle, Zeyu Zheng, Khimya Khetarpal, Hado van Hasselt, Razvan Pascanu, James Martens, Will Dabney. arXiv preprint, 2024. | arXiv: <https://arxiv.org/abs/2402.18762> · PDF: <https://arxiv.org/pdf/2402.18762> |
| 18 | **Loss of Plasticity in Deep Continual Learning.** Shibhansh Dohare, J. Fernando Hernandez-Garcia, Qingfeng Lan, Parash Rahman, A. Rupam Mahmood, Richard S. Sutton. *Nature* 632:768–774, 2024. | Open at PMC: <https://pmc.ncbi.nlm.nih.gov/articles/PMC11338828/> (PDF link on page) · DOI: [10.1038/s41586-024-07711-7](https://doi.org/10.1038/s41586-024-07711-7) · Code: <https://github.com/shibhansh/loss-of-plasticity> |

### 6.4 Adjacent threads (project-relevant; §5)

| # | Title / Authors / Venue | Download |
|---|---|---|
| 19 | **Concatenated ReLU (CReLU): Understanding and Improving Convolutional Neural Networks via Concatenated Rectified Linear Units.** Wenling Shang, Kihyuk Sohn, Diogo Almeida, Honglak Lee. *ICML* 2016, PMLR 48:2217–2225. | arXiv: <https://arxiv.org/abs/1603.05201> · PDF: <https://arxiv.org/pdf/1603.05201> · PMLR: <https://proceedings.mlr.press/v48/shang16.html> |
| 20 | **The Entropy Mechanism of Reinforcement Learning for Reasoning Language Models.** Ganqu Cui, Yuchen Zhang, Jiacheng Chen, Lifan Yuan, Zhi Wang, et al. (18 authors). arXiv preprint, 2025. | arXiv: <https://arxiv.org/abs/2505.22617> · PDF: <https://arxiv.org/pdf/2505.22617> · See §6.5 — venue is a 2025 preprint, **flagged**. |
| 21 | **Curriculum Learning for Reinforcement Learning Domains: A Framework and Survey.** Sanmit Narvekar, Bei Peng, Matteo Leonetti, Jivko Sinapov, Matthew E. Taylor, Peter Stone. *JMLR* 21(181):1–50, 2020. | JMLR (open PDF on page): <https://jmlr.org/papers/v21/20-212.html> · arXiv: <https://arxiv.org/abs/2003.04960> · PDF: <https://arxiv.org/pdf/2003.04960> |
| 22 | **Task-Agnostic Continual Reinforcement Learning: Gaining Insights and Overcoming Challenges.** Massimo Caccia, Jonas Mueller, Taesup Kim, Laurent Charlin, Rasool Fakoor. *CoLLAs* 2023 (arXiv 2022). | arXiv: <https://arxiv.org/abs/2205.14495> · PDF: <https://arxiv.org/pdf/2205.14495> · See §6.5 — title note, **flagged**. |

### 6.5 Verification notes — entries to double-check before relying on the exact metadata

All 22 titles/authors/venues were web-confirmed. Four entries carry a caveat the user should know about before downloading or citing:

1. **#1 McCloskey & Cohen (1989)** — Title, authors, venue (*Psychology of Learning and Motivation* vol. 24, pp. 109–165) confirmed across multiple sources. **No free, legal PDF exists** — it is a 1989 book chapter behind ScienceDirect/Elsevier. The DOI link is correct but paywalled. *Action: obtain via institutional access or library; there is no arXiv equivalent.*
2. **#9 Kumar et al. (2021), "Implicit Under-Parameterization"** — Title, all four authors, and ICLR 2021 venue confirmed. The arXiv link (`2010.14498`) is the standard arXiv ID for this paper but was **inferred from the known arXiv posting, not re-fetched in this session** — verify the `abs` page resolves before bulk download. The authoritative landing page <https://agarwl.github.io/iup/> and OpenReview (ICLR 2021) are reliable fallbacks.
3. **#20 Cui et al. (2025)** — Title, lead authors, and arXiv ID `2505.22617` confirmed; the full author line is 18 people (truncated here to first five + "et al."). **Venue is an arXiv 2025 preprint** — the [[curriculum_underperformed_baseline_plasticity_vs_budget]] critique cites it as "NeurIPS 2025," but I could *not* independently confirm formal NeurIPS 2025 acceptance from the search results. *Action: treat as a 2025 preprint unless acceptance is verified; the arXiv PDF is the safe download.*
4. **#22 Caccia et al. (2022/2023)** — The arXiv version (`2205.14495`, all three versions) is titled **"…Gaining Insights and Overcoming Challenges."** The subtitle **"In Praise of a Simple Baseline"** used in our critique memo is the **Amazon Science publication title** of the same work (<https://www.amazon.science/publications/task-agnostic-continual-reinforcement-learning-in-praise-of-a-simple-baseline>). Same authors, same paper, two titles. *Action: download from arXiv under the "Gaining Insights" title; both titles refer to one paper.*

No other entry had a title or link I was unable to verify.

---

## 7. Suggested reading order for someone new to the topic

A path that builds the two-failure-modes intuition before diving into mechanisms:

1. **French (1999)** [#2] — the cleanest statement of catastrophic forgetting *and* the stability–plasticity dilemma. Start here for the framing.
2. **Kirkpatrick et al. (2017), EWC** [#4] — the canonical Phase-1 fix; understand the regularization family from its flagship.
3. **Dohare et al. (2024), Nature** [#18] — jump straight to the capstone of the *other* failure mode. Nature-level exposition makes loss of plasticity vivid before you study its sub-mechanisms. This is the pivot of the whole story.
4. **Abbas et al. (2023)** [#13] — the RL continual-task setup closest to ours; grounds the abstract phenomenon in a concrete RL benchmark with dormant units + rank collapse.
5. **Sokar et al. (2023), ReDo** [#14] and **Nikishin et al. (2022), Primacy Bias** [#12] — the two most practical fixes (recycle dormant units; reset). Read together as the "corrective" toolkit.
6. **Lyle et al. (2024), Disentangling** [#17] — the synthesis: there is no single cause, so there is no single fix. Read last among the core set; it recontextualizes everything before it.
7. **Then, for our project specifically:** Cui (2025) [#20], Narvekar (2020) [#21], Caccia (2022) [#22], and the critique [[curriculum_underperformed_baseline_plasticity_vs_budget]] that applies all of this to our curriculum result.

*Optional deepening:* Ash & Adams (2020) [#8] (why warm-starting hurts — the supervised root of the RL reset line), Kumar (2021) [#9] and Lyle (2022) [#11] (the rank/capacity vocabulary), Nikishin (2023) [#15] (plasticity injection as a *diagnostic*). The remaining Phase-1 entries (#3, #5, #6, #7) are reference material on the three forgetting-fix families — read on demand, not front-to-back.
