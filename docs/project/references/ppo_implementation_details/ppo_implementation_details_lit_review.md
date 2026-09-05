---
title: "PPO Implementation Details — Master Reference Review (3 works, 4 PDFs)"
topic: ppo_implementation_details
status: active
created: 2026-09-04
last_updated: 2026-09-04
works_reviewed: 3
source_files: 4
related:
  - ../modulation_in_rl/ppo_return_normalization_survey.md
scope: |
  The three works the reinforcement-learning field itself cites when arguing about
  what inside a PPO implementation is load-bearing: the two large ablation studies
  (Andrychowicz et al., ICLR 2021; Engstrom et al., ICLR 2020) and the community
  reference catalogue (Huang et al., ICLR Blog Track 2022). Four PDFs, because the
  Andrychowicz work is held in both its camera-ready and its 48-page preprint form
  and only the preprint contains the experimental appendix.

  This folder was assembled to answer five specific questions arising from a live
  dispute in this project about where normalization belongs in PPO: (1) what was
  actually measured about advantage normalization, (2) what was measured about
  reward / return / value-target normalization, (3) how strong the shared-vs-separate
  network finding is, (4) what the policy last-layer initialization claim rests on,
  and (5) which findings survive the move from continuous-control MuJoCo with
  feed-forward networks to this project's discrete-action, recurrent, shared-trunk
  setting. The answers, including several explicit statements that a paper is silent
  on a question, are in section 7.
---

# PPO Implementation Details — Master Reference Review

## 1. Plain-English entry point

### What this document is

Proximal Policy Optimization (PPO) is the reinforcement-learning algorithm this project's agent is
trained with. Its published description takes about a page of equations. The code that produces the
published results contains **dozens of additional steps that are not in that page** — rescaling the
numbers that come out of the environment, choosing how the network's weights start out, deciding
whether the "actor" (which picks actions) and the "critic" (which predicts how well things are going)
share a network or get one each. This document reviews the three works that the field uses to argue
about which of those extra steps actually matter.

It exists for a specific reason. This project has an open disagreement about **where rescaling belongs
in PPO** — specifically, whether the critic's training target should be rescaled batch-by-batch, or
whether it is the *advantage* (the "was this action better than expected?" signal that drives the
actor) that should be rescaled. Rather than argue from intuition, this review goes to the field's own
experimental evidence and reports, question by question, what was measured and what was not.

### The three works, in one line each

- **Andrychowicz et al. (ICLR 2021)** built one PPO-like agent with more than fifty of these hidden decisions turned into switches, trained **over 250,000 agents** flipping them at random, and reported which switches moved the score. It is the largest study of its kind.
- **Engstrom et al. (ICLR 2020)** showed that the gap between PPO and its predecessor TRPO is mostly produced by these hidden code details rather than by PPO's headline idea, and that PPO's famous "clipping" trick is not what makes it work.
- **Huang et al. (ICLR Blog Track 2022)** is not a study at all — it is a **catalogue**: 37 implementation details read line by line out of the official code, each with a link to the exact source line, plus a from-scratch re-implementation that reproduces the official results. Its authors say explicitly that they are not doing ablations.

A practical warning before anything else: **the Andrychowicz paper changed its title between the
preprint and the published version** ("What Matters In On-Policy Reinforcement Learning?" became "What
Matters For On-Policy Deep Actor-Critic Methods?"), so a literature search on one title will not find
the other. They are one work. Section 2.1 documents exactly what differs. Separately, the Engstrom PDF
misprints its own venue as "ICLR 2019" in the page header; it is an **ICLR 2020** paper, and section
2.2 gives the evidence, because that misprint has already propagated into at least one other paper's
reference list.

### The five answers, in plain language

**1. Rescaling the advantage (the actor's learning signal), batch by batch.** The 250,000-agent study
did measure this. Its conclusion, in its own words, is that it *"seems not to affect the performance
too much"*. Turning it on helped a little on three of five tasks and hurt a little on the other two,
and every error bar overlapped. It is not in their list of recommendations and their own default
setting leaves it switched **off**. **Correction to what has been said in this project:** describing
this as "no significant effect" is wrong on a technicality that matters — **they never ran a
significance test**, and among their best-performing agents there is in fact a mild preference for
having it on (53.5% versus 47%). The defensible sentence is "a small, task-dependent effect that their
error bars cannot separate from zero, neither recommended nor warned against". The other two works do
not test it at all: Engstrom's paper never discusses advantages, and Huang's catalogue only documents
that the official code does it (per small batch, not per whole batch) before citing Andrychowicz for
whether it helps.

**2. Rescaling the rewards, the returns, or the critic's target.** Engstrom's team ablated **reward
scaling** — dividing every reward by a slowly-updated estimate of how spread out the accumulated
rewards have been, with no shifting — and found it was **one of three things PPO could not do without**
to reach its best scores. Andrychowicz's team never studied reward scaling at all, but did study
rescaling **the critic's training target** using a slowly-updated mean and spread, converting the
critic's output back into ordinary reward units whenever it is read. That one turned out to be a very
large effect **in both directions**: worth +78% on one task and −42% on another, from which the
authors draw the deliberately cautious recommendation "check whether it helps on your environment".
And the direct answer to this project's question: **nobody in this corpus rescales the critic's
training target using statistics recomputed from the current batch.** Where the target is rescaled at
all, the statistics move slowly and the rescaling is undone before the number is used anywhere else.

**3. One shared network, or one each for actor and critic.** Andrychowicz's team found separate
networks better on four of five tasks, and their architecture advice says to keep the critic's network
wide and unshared. But the comparison is weaker than it sounds: the two arms of the experiment were
given different sets of things to tune, and in particular the shared arm had to guess a "how much does
the critic's error count?" weight that was drawn at random across five orders of magnitude — and the
authors then **deleted the shared arm from the experiment before ever reporting that weight's
effect**. Huang's catalogue contributes the only comparison outside robot-locomotion tasks (separate
networks clearly better on two simple discrete-action tasks), and offers the reason as a
one-sentence guess: "probably due to the competing objectives of the policy and value functions".
**On the specific mechanism this project has been invoking** — that in a shared network the critic's
error dominates and the actor ends up merely reading out features shaped by the critic — **none of the
three works tests it.** Andrychowicz removes the relevant arm; Engstrom's networks never share
anything; Huang states it as a conjecture and points to a different paper that is not in this corpus.
And the same corpus contains a counterweight: the official code **shares** its network for Atari
games, Huang reproduces the published Atari scores with it, and the only recurrent (memory-equipped)
PPO anywhere in this corpus is built on that shared network.

**4. How the actor's final layer is initialized.** This is reported as one of the largest effects in
the whole 250,000-agent study, and it is rarely mentioned in papers. The recipe is to shrink the
weights of the actor network's last layer by a factor of about 100, so that at the start of training
the agent's behaviour is a small random wiggle that ignores what it is looking at. On the hardest task
this alone was worth 66% more score. Two things to note: the relationship is a **cliff, not a slope** —
shrinking by 10, 100 or 1000 all work about equally well, and only leaving the layer at its default
scale is bad — and the same shrinking applied to the *critic's* last layer does almost nothing, which
is what makes the explanation credible. Engstrom's team corroborate that initialization matters, but
they tested the initialization scheme as one indivisible switch and never isolated the last layer.

**5. Does any of this transfer to our setting?** Only partly, and the differences must be stated every
time. All the *ablations* in this corpus were run on continuous-control robot-locomotion simulations
with plain feed-forward networks and fully visible state. This project's agent chooses from a discrete
set of actions, carries a memory (a GRU), shares one trunk between actor and critic, cannot see the
whole world, and is rewarded for keeping its internal variables in a healthy range rather than for
running fast. Findings about **input rescaling**, about **not clipping the critic's loss**, and about
**how advantages are estimated** are likely to carry over. Findings about **shared versus separate
networks** should not be quoted as a design instruction. The whole apparatus around **reward
rescaling** was built to cope with a reward that grows without limit as the agent improves, which is
not the situation here. And the corpus is **completely silent** on partial observability, on
homeostatic rewards, and on measuring performance in survival time rather than accumulated reward.
Section 8 is the standing caveat table.

### The discipline applied here

This review was written under an instruction to report what the papers measured and concluded, and to
say plainly where they are silent rather than reaching. Section 7 therefore contains several explicit
"the paper is silent on this" entries, and section 9 records four refinements and three citation
corrections to this project's own earlier survey — none of which overturn it. Where this review
derives a consequence that a paper does not state (three times: what advantage normalization does to
PPO's clipping geometry, what target normalization does to the value loss, and how reward scaling
changes the balance between the two loss terms), the derivation is explicitly labelled as
reviewer-supplied and marked "do not cite the paper for this".

---

## Table of contents

1. [Plain-English entry point](#1-plain-english-entry-point)
2. [Corpus manifest — three works, four files](#2-corpus-manifest--three-works-four-files)
   - [2.1 The Andrychowicz title change](#21-the-andrychowicz-title-change--read-this-before-searching-the-literature)
   - [2.2 The Engstrom running-header misprint](#22-the-engstrom-running-header-misprint)
   - [2.3 The Huang et al. format caveat](#23-the-huang-et-al-format-caveat)
3. [Reading order and how to cite this corpus](#3-reading-order-and-how-to-cite-this-corpus)
4. [Andrychowicz et al. (2021) — *What Matters For On-Policy Deep Actor-Critic Methods?*](#4-andrychowicz-et-al-2021--what-matters-for-on-policy-deep-actor-critic-methods-a-large-scale-study)
   - [4.1 Fixed extraction block](#41-fixed-extraction-block)
   - [4.2 The three findings this project came for, verified against the figures](#42-the-three-findings-this-project-came-for-verified-against-the-figures)
   - [4.3 Everything else the paper concluded, compressed](#43-everything-else-the-paper-concluded-compressed)
   - [4.4 Phase 1 — Foundational overview](#44-phase-1--foundational-overview)
   - [4.5 Phase 2 — Graduate-level deep dive](#45-phase-2--graduate-level-deep-dive)
   - [4.6 Relevance and transfer to this project](#46-relevance-and-transfer-to-this-project)
   - [4.7 Appendix: Section-by-Section Backbone](#47-appendix-section-by-section-backbone)
5. [Engstrom et al. (2020) — *Implementation Matters in Deep Policy Gradients*](#5-engstrom-et-al-2020--implementation-matters-in-deep-policy-gradients-a-case-study-on-ppo-and-trpo)
   - [5.1 Fixed extraction block](#51-fixed-extraction-block)
   - [5.2 The nine code-level optimizations](#52-the-nine-code-level-optimizations-as-the-paper-defines-them-3)
   - [5.3 Reward scaling — the answer to this project's question 2](#53-reward-scaling--the-answer-to-this-projects-question-2)
   - [5.4 Phase 1 — Foundational overview](#54-phase-1--foundational-overview)
   - [5.5 Phase 2 — Graduate-level deep dive](#55-phase-2--graduate-level-deep-dive)
   - [5.6 A terminological trap between the two ablation papers](#56-a-terminological-trap-between-the-two-ablation-papers)
   - [5.7 Relevance and transfer to this project](#57-relevance-and-transfer-to-this-project)
   - [5.8 Appendix: Section-by-Section Backbone](#58-appendix-section-by-section-backbone)
6. [Huang et al. (2022) — *The 37 Implementation Details of Proximal Policy Optimization*](#6-huang-dossa-raffin-kanervisto--wang-2022--the-37-implementation-details-of-proximal-policy-optimization)
   - [6.1 Fixed extraction block](#61-fixed-extraction-block)
   - [6.2 The catalogue, restricted to the details this project asked about](#62-the-catalogue-restricted-to-the-details-this-project-asked-about)
   - [6.3 The one comparison they run themselves — shared vs. separate networks, on discrete actions](#63-the-one-comparison-they-run-themselves--shared-vs-separate-networks-on-discrete-actions)
   - [6.4 Reproduction fidelity](#64-reproduction-fidelity--what-this-post-is-genuinely-authoritative-about)
   - [6.5 Phase 1 — Foundational overview](#65-phase-1--foundational-overview)
   - [6.6 Phase 2 — Graduate-level deep dive](#66-phase-2--graduate-level-deep-dive)
   - [6.7 Relevance and transfer to this project](#67-relevance-and-transfer-to-this-project)
   - [6.8 Appendix: Section-by-Section Backbone](#68-appendix-section-by-section-backbone)
7. [Evidence ledger — the five questions this corpus was read to answer](#7-evidence-ledger--the-five-questions-this-corpus-was-read-to-answer)
   - [Q1 — Advantage normalization](#q1--advantage-normalization-what-did-andrychowicz-et-al-actually-measure)
   - [Q2 — Reward, return and value normalization](#q2--reward-return-and-value-normalization)
   - [Q3 — Shared vs. separate policy and value networks](#q3--shared-vs-separate-policy-and-value-networks)
   - [Q4 — Initialization of the policy's last layer](#q4--initialization-of-the-policys-last-layer)
   - [Q5 — Scope limits](#q5--scope-limits-what-transfers-to-a-discrete-recurrent-shared-trunk-homeostatic-reward-setting)
8. [Scope-limit register](#8-scope-limit-register--what-may-and-may-not-be-quoted-at-this-projects-architecture)
9. [Checks against this project's own survey](#9-checks-against-this-projects-own-survey)
10. [Provenance and method](#10-provenance-and-method)

---

## 2. Corpus manifest — three works, four files

| # | Work | File in `sources/` | Pages | Status |
|---|---|---|---|---|
| 1 | **Andrychowicz, Raichuk, Stańczyk, Orsini, Girgin, Marinier, Hussenot, Geist, Pietquin, Michalski, Gelly & Bachem (2021).** *What Matters For On-Policy Deep Actor-Critic Methods? A Large-Scale Study.* ICLR 2021. | `Andrychowicz et al. 2021 - ... (ICLR camera-ready).pdf` | 10 | **Version of record — cite this.** Main text only; no appendices. |
| 1b | Same work, earlier preprint. **Title as published on arXiv: *What Matters In On-Policy Reinforcement Learning? A Large-Scale Empirical Study*** (arXiv:2006.05990v1, 10 June 2020). | `Andrychowicz et al. 2021 - ... (preprint, 48pp with full appendix).pdf` | 48 | **Evidence source.** All 38 pages of per-choice experimental appendix (App. A–K, Figures 1–92) exist **only** here. |
| 2 | **Engstrom, Ilyas, Santurkar, Tsipras, Janoos, Rudolph & Madry (2020).** *Implementation Matters in Deep Policy Gradients: A Case Study on PPO and TRPO.* **ICLR 2020.** | `Engstrom et al. 2020 - ... (ICLR camera-ready).pdf` | 14 | Camera-ready + appendix. |
| 3 | **Huang, Dossa, Raffin, Kanervisto & Wang (2022).** *The 37 Implementation Details of Proximal Policy Optimization.* ICLR Blog Track 2022. | `Huang et al. 2022 - ... (ICLR Blog Track).pdf` | 68 | Peer-reviewed blog post, rendered from HTML to PDF. |

### 2.1 The Andrychowicz title change — read this before searching the literature

The paper was **retitled between the preprint and the camera-ready**, and the two titles are
different enough that a literature search on one will not surface the other:

- **arXiv v1 (June 2020):** "What Matters **In On-Policy Reinforcement Learning**? A Large-Scale **Empirical** Study"
- **ICLR 2021 camera-ready:** "What Matters **For On-Policy Deep Actor-Critic Methods**? A Large-Scale Study"

They are **one work, not two.** The abstract was reframed in the same direction — "a unified
on-policy RL framework" became "a unified on-policy **deep actor-critic** framework", and
"recommendations for on-policy training of RL agents" became "recommendations for the training of
on-policy **deep actor-critic** RL agents". The narrowing is honest: every experiment in the paper
is actor-critic, so the camera-ready title claims less than the preprint title did.

**What actually differs between the two files, verified by text diff:**

| Aspect | Finding |
|---|---|
| Main-text §3.3 (*Normalization and clipping*) | **Character-identical** apart from heading capitalisation and footnote renumbering (diff similarity 0.961, with every difference located in the heading or a footnote index). |
| Main-text §3.2 (*Networks architecture*) | Identical apart from one typo fix — preprint "The key recipe **appears is** to initialize", camera-ready "The key recipe **is** to initialize" — and footnote renumbering. |
| Abstract / introduction | Reworded to the "deep actor-critic" framing described above; no change to any number. |
| Appendices A–K (36 pages, Figures 1–92, Tables 2–9) | **Present only in the preprint.** The camera-ready ends at the reference list on p. 10. |

**Practical rule used throughout this review:** cite the **camera-ready** for what the authors chose
to *assert*; read the **preprint** for the *evidence* behind any assertion. Every figure number
referenced below (Fig. 15, 24, 33–38, 43, …) is a preprint figure. The camera-ready refers to those
same figure numbers in its main text but does not contain the figures.

### 2.2 The Engstrom running-header misprint

The Engstrom PDF's running header reads **"Published as a conference paper at ICLR 2019"** on every
page. This is a **misprint in the authors' LaTeX** (a stale `iclrconf` year), not a statement of
venue. The evidence that it is an ICLR **2020** paper:

- the arXiv comment for [arXiv:2005.12729](https://arxiv.org/abs/2005.12729) states this is the "ICLR 2020 version";
- the PDF's own embedded creation timestamp is **2020-04-13 16:46 (UTC−4)**, i.e. eleven months after ICLR 2019 concluded, and two weeks before ICLR 2020 convened;
- Andrychowicz et al.'s reference list ([27] in the camera-ready) dates it "International Conference on Learning Representations, **2019**" — i.e. a second paper **inherited the same misprint**, which is exactly how such an error propagates through a literature. (Andrychowicz's entry also gives an earlier title, "Implementation Matters in Deep **RL**: A Case Study on PPO and TRPO", rather than the camera-ready's "Implementation Matters in Deep **Policy Gradients**" — so that reference is wrong on both the title and the year, while Huang et al.'s bibliography gets both right.)

**Cite it as ICLR 2020.** Do not "correct" the citation back to 2019 on the strength of the header.

### 2.3 The Huang et al. format caveat

The Huang PDF is a **print rendering of an HTML blog post**. Page breaks, the floating table of
contents, the "Back to top" chrome, the citation widget and the Disqus block are artefacts of that
rendering and carry no meaning. The blog *is* a real peer-reviewed publication (ICLR 2022 Blog Track,
which ran a review process), and it is unusually well sourced — but it is a **practitioner
walkthrough with reproduction experiments**, not a controlled study. It is weighted accordingly
throughout: it is authoritative on *what reference implementations do* and on *reproduction of
published scores*, and it is **not** evidence about *which choices matter*, because it does not run
the ablations that would establish that.

## 3. Reading order and how to cite this corpus

**If you have twenty minutes** and want the answers rather than the papers: read §1, then §7 (the five
questions), then §8 (what transfers). Everything else is support.

**If you are about to cite one of these works in a plan, a design doc or a paper**, check it against
this table first. Getting these wrong is the most common failure mode with this corpus, and two of the
three errors below already exist in published documents.

| If you want to say… | Cite | Do **not** cite |
|---|---|---|
| "Per-minibatch advantage normalization has a small, sign-inconsistent effect" | Andrychowicz et al. 2021, §3.3 and Fig. 35 (preprint) | Engstrom (silent); Huang (documents, does not test) |
| "Advantage normalization has *no significant effect*" | — **nobody**; no significance test was run | Andrychowicz (this misquotes them) |
| "Reward scaling by a running return-std was necessary for PPO's best MuJoCo scores" | Engstrom et al. 2020, Fig. 1 and App. A.2 | Andrychowicz (does not study it) |
| "Value-target normalization has a very large, environment-dependent effect" | Andrychowicz et al. 2021, §3.3 and Fig. 37 — **and quote the sign flip**, or the citation is misleading | Engstrom (does not normalize the target) |
| "The critic's target should be normalized" | — **nobody in this corpus recommends this unconditionally**; Andrychowicz's own recommendation is "check if it improves performance" | anyone |
| "Separate policy and value networks outperformed a shared trunk" | Andrychowicz et al. 2021, §3.2 and Fig. 15 (4/5 MuJoCo tasks); Huang et al. 2022, core detail 13 (2/3 classic-control tasks) | either paper, without the confounds in §4.2.3 and §6.3 |
| "…because the value loss dominates the shared trunk's gradient" | — **nobody**; Huang states it as a conjecture and points to Phasic Policy Gradient, which is outside this corpus | all three |
| "Shrinking the policy's last-layer initialization is one of the largest effects" | Andrychowicz et al. 2021, §3.2 and Fig. 24 | Engstrom (tests the initialization *scheme*, not the last layer) |
| "PPO-style value-loss clipping does not help" | Andrychowicz C13 (it *hurts*, at every threshold) **and** Engstrom Fig. 1 (absent from what mattered) — two independent sources | — |
| "PPO's clipping is not what makes PPO work" | Engstrom et al. 2020, §5 and Table 3 | Andrychowicz (finds the PPO loss best among six, a different question) |
| "The reference PPO implementation does X at line Y" | Huang et al. 2022 — best-sourced document of its kind | the ablation papers, which describe rather than link |
| "Recurrent PPO requires state resets at episode boundaries, sequential minibatches, and state reconstruction at training time" | Huang et al. 2022, LSTM details 1–5 | Andrychowicz or Engstrom (neither runs a recurrent network) |

**Version discipline for Andrychowicz.** Cite the **ICLR 2021 camera-ready** for assertions; read the
**arXiv:2006.05990v1 preprint** for evidence. Every figure number in this review (Fig. 15, 19, 24,
33–38, 43, 53, 58, 65, 76–77) is a **preprint** figure — the camera-ready refers to them by the same
numbers but does not contain them.

## 4. Andrychowicz et al. (2021) — *What Matters For On-Policy Deep Actor-Critic Methods? A Large-Scale Study*

### 4.1 Fixed extraction block

| Field | Value |
|---|---|
| **Venue** | ICLR 2021 (camera-ready, 10 pp.). Preprint arXiv:2006.05990v1, 10 June 2020, 48 pp., under the earlier title *What Matters In On-Policy Reinforcement Learning? A Large-Scale Empirical Study*. |
| **PDFs** | `docs/project/references/ppo_implementation_details/sources/Andrychowicz et al. 2021 - What matters for on-policy deep actor-critic methods (ICLR camera-ready).pdf` and `... (preprint, 48pp with full appendix).pdf` |
| **Object of study** | >50 configurable "choices" (C1–C68) inside one unified on-policy deep actor-critic implementation, built on the SEED RL code base. |
| **Environments** | Hopper-v1, Walker2d-v1, HalfCheetah-v1, Ant-v1, Humanoid-v1 (OpenAI Gym / MuJoCo 2.0). **Continuous action spaces only.** |
| **Networks** | Multilayer perceptrons only. No recurrence anywhere in the paper (the string "recurrent", "LSTM" and "GRU" do not occur). Appendix A explicitly says "To simplify the exposition we assume in this section that the environment is fully observable." |
| **Action distribution** | Gaussian throughout (App. B.8). **No discrete-action / categorical policy is ever trained.** |
| **Budget** | >250,000 agents; 3 seeds per choice configuration; 1M env steps (Hopper, HalfCheetah, Walker2d) or 2M (Ant, Humanoid). |
| **Score** | Average of undiscounted 100-episode evaluation returns taken every 100k steps, i.e. **proportional to area under the learning curve** — it rewards fast learning, not just final performance. Per configuration, the **median across the 3 seeds**. |
| **Two reported statistics** | (a) **conditional 95th percentile** of that score among all sampled configurations that used a given value, with a binomial 95% confidence interval; (b) **share of the top-5% configurations** that used that value — 0.5 means "no preference", because values were sampled uniformly. |
| **Experiment structure** | Eight thematic groups (App. D–K). Within a group, the group's choices are sampled uniformly at random and **everything outside the group is pinned to the default configuration in App. C, Table 2**. Adam learning rate (C24) is included in every group. |
| **Not studied at all** | Reward scaling / reward normalization (the OpenAI-Baselines `VecNormalize` mechanism). The strings "reward scaling" and "reward normalization" **do not occur in the paper**. |

### 4.2 The three findings this project came for, verified against the figures

The camera-ready asserts these in §3.2 and §3.3 in a single paragraph each. The evidence sits in
preprint Appendix E (architecture, Figs. 15–30) and Appendix F (normalization and clipping,
Figs. 31–38). Bar heights and confidence intervals below were read off the rendered figure pages at
600 dpi; they are approximate to roughly the width of a bar's error whisker and are reported to three
significant figures at most.

#### 4.2.1 Per-minibatch advantage normalization (choice C67)

**Definition, verbatim (App. B.9):** "Per minibatch advantage normalization (choice C67). We
normalize advantages in each minibatch by subtracting their mean and dividing by their standard
deviation for the policy loss."

**Finding, verbatim (§3.3):** "In contrast to observation and value function normalization,
per-minibatch advantage normalization (C67) **seems not to affect the performance too much** (Fig. 35)."

**Default in the base configuration (App. C, Table 2):** `C67 Per minibatch advantage normalization: False`.

**Absent from the §3.3 recommendation**, which reads in full: "Always use observation normalization
and check if value function normalization improves performance. Gradient clipping might slightly
help but is of secondary importance."

**Figure 35, left panel** — conditional 95th percentile of the performance score, C67 `False` vs `True`,
with 95% binomial confidence intervals:

| Environment | C67 = False | C67 = True | Direction |
|---|---|---|---|
| Hopper-v1 | ≈1625 [1575, 1670] | ≈1670 [1650, 1685] | True higher, CIs overlap |
| Humanoid-v1 | ≈2030 [1905, 2270] | ≈2250 [2050, 2510] | True higher, CIs overlap heavily |
| Walker2d-v1 | ≈1585 [1440, 1725] | ≈1525 [1440, 1640] | **False** higher, CIs overlap |
| HalfCheetah-v1 | ≈1780 [1600, 1965] | ≈1690 [1485, 1850] | **False** higher, CIs overlap |
| Ant-v1 | ≈2255 [2185, 2360] | ≈2330 [2210, 2420] | True higher, CIs overlap |

**Figure 35, right panel** — share of the top-5% configurations using each value (0.5 = no preference):

| | Hopper | Humanoid | Walker2d | HalfCheetah | Ant | **all** |
|---|---|---|---|---|---|---|
| False | 0.38 | 0.42 | 0.535 | 0.548 | 0.46 | **0.47** |
| True | 0.62 | 0.58 | 0.47 | 0.445 | 0.54 | **0.535** |

**Verdict on the phrasing "no significant effect".** That phrasing needs correcting on two counts,
one of substance and one of statistical hygiene.

1. **The paper never performed a significance test on C67**, so "no significant effect" attributes to
   the authors a claim they did not make. What they report is a percentile estimate with a binomial
   confidence interval, and their own wording is the deliberately soft "seems not to affect the
   performance too much".
2. **It is not literally "no effect" in their data.** The direction is inconsistent across the five
   environments (True ahead on 3, behind on 2), every confidence interval overlaps its counterpart,
   and the largest gap is Humanoid at roughly +11% for `True` — inside its own error bar. But the
   top-5% panel shows a **mild and consistent-in-aggregate tilt toward `True`** (0.535 vs 0.47
   pooled), driven by Hopper (0.62) and Humanoid (0.58) and opposed by Walker2d and HalfCheetah.

**The defensible sentence** is therefore: *Andrychowicz et al. found per-minibatch advantage
normalization to have a small, environment-dependent, sign-inconsistent effect that their confidence
intervals cannot separate from zero; they neither recommend it nor warn against it, and they left it
off in their own default configuration.* Anything stronger than that in either direction over-reads
Figure 35.

#### 4.2.2 Value-function normalization (choice C66)

**Definition, verbatim (App. B.9):** "Value function normalization (choice C66). Similarly to
observations, we also maintain the empirical mean $v_\mu$ and standard deviation $v_\rho$ of value
function targets (See Sec. B.2). The value function network predicts normalized targets
$(\hat V - v_\mu)/\max(v_\rho, 10^{-6})$ and its outputs are denormalized accordingly to obtain
predicted values: $\hat V = v_\mu + V_{\text{out}} \max(v_\rho, 10^{-6})$ where $V_{\text{out}}$ is
the value network output."

Three properties of that definition matter and are easy to lose:

- the statistics are **running** — "the empirical mean and standard deviation of value function targets", i.e. accumulated over targets seen so far, not recomputed from the current minibatch;
- the network output is **denormalized at read-out**, so every downstream consumer (including the advantage computation, which subtracts $V(s_t)$ from a return estimate) sees a value in **raw reward units**;
- the choice has exactly two settings in the experiment: `Average` (on) and `None` (off).

**Finding, verbatim (§3.3):** "Quite surprisingly, value function normalization (C66) **also
influences the performance very strongly** — it is crucial for good performance on HalfCheetah and
Humanoid, helps slightly on Hopper and Ant and **significantly hurts the performance on Walker2d**
(Fig. 37). We are not sure why the value function scale matters that much but suspect that it affects
the performance by **changing the speed of the value function fitting**."

**Figure 37, left panel** (conditional 95th percentile, 95% CI):

| Environment | C66 = Average (on) | C66 = None (off) | Change | CIs overlap? |
|---|---|---|---|---|
| Hopper-v1 | ≈1690 [1665, 1715] | ≈1620 [1590, 1660] | +4% | no |
| Humanoid-v1 | ≈2710 [2510, 2880] | ≈1520 [1460, 1590] | **+78%** | no |
| Walker2d-v1 | ≈1035 [975, 1090] | ≈1780 [1720, 1840] | **−42%** | no |
| HalfCheetah-v1 | ≈1905 [1755, 2060] | ≈1375 [1290, 1660] | +39% | no |
| Ant-v1 | ≈2365 [2295, 2420] | ≈2150 [2075, 2230] | +10% | no |

**Figure 37, right panel** (share of top-5% configurations):

| | Hopper | Humanoid | Walker2d | HalfCheetah | Ant | **all** |
|---|---|---|---|---|---|---|
| Average (on) | 0.68 | 0.96 | **0.01** | 0.67 | 0.635 | 0.59 |
| None (off) | 0.32 | 0.04 | **0.99** | 0.33 | 0.37 | 0.415 |

This is a genuinely large effect **with a sign flip**: on Walker2d, essentially every top-5%
configuration had value normalization *off*, and on Humanoid essentially every one had it *on*. The
recommendation the authors derive from this is deliberately conditional — "**check if** value function
normalization improves performance", not "use it".

**Two documented ambiguities in this experiment**, both of which limit how hard the C66/C67 numbers
can be leaned on:

1. **Was PPO-style value clipping on or off?** App. F.1 says "All the other choices were set to the
   default values as described in Appendix C", and App. C Table 2 lists `C13 PPO-style value clipping ε: 0.2`.
   But footnote 11 of the same section says: "Another explanation could be the interaction between the
   value function normalization and PPO-style value clipping (C13). We have, however, disable[d] the
   value clipping in this experiment to avoid this interaction. The disabling of the value clipping
   could also explain why our conclusions are different from [Engstrom et al.] where a form of value
   normalization improved the performance on Walker." The footnote and the design statement
   contradict each other; the footnote is the more specific claim and is presumably right.
2. **Was the trunk shared or separate?** App. C Table 2 lists `C47 Shared MLPs?: Shared` — but then
   lists the sub-choices that are only active under *separate* networks (`C49` policy width 64,
   `C50` value width 64, `C52` policy depth 2, `C53` value depth 2) and lists **none** of the
   sub-choices that are active under *shared* (`C48` shared width, `C51` shared depth, `C54` baseline
   cost). The table's own caption says "We only list sub-choices that are active." The table is
   therefore self-inconsistent, and **the paper does not let a reader determine whether the C66 and
   C67 numbers were measured on a shared trunk or on two separate networks.** For a project running a
   shared trunk this is exactly the wrong ambiguity to have.

#### 4.2.3 Shared vs. separate policy and value networks (choice C47)

**Definition, verbatim (App. B.7):** "We use multilayer perceptrons (MLPs) to represent policies and
value functions. We either use separate networks for the policy and value function, or use a single
network with two linear heads, one for the policy and one for the value function (choice C47). …
If we use the shared MLP, we further add a hyperparameter `Baseline cost (shared)` (C54) that
rescales the contribution of the value loss to the full objective function. This is important in this
case as the shared layers of the MLP affect the loss terms related to both the policy and the value
function."

**Finding, verbatim (§3.2):** "Separate value and policy networks (C47) appear to lead to better
performance on four out of five environments (Fig. 15). To avoid analyzing the other choices based on
bad models, we thus focus for the rest of this experiment only on agents with separate value and
policy networks." The App. E.1 design section adds: "After running the experiment described above we
noticed (Fig. 15) that separate policy and value function networks (C47) perform better and we have
**rerun the experiment with only this variant present**."

**Recommendation, verbatim (§3.2, final clause):** "Use a wide value MLP (**no layers shared with the
policy**) but tune the policy width (it might need to be narrower than the value MLP)."

**Figure 15, left panel** (conditional 95th percentile, 95% CI):

| Environment | separate | shared | Change | CIs overlap? |
|---|---|---|---|---|
| Hopper-v1 | ≈1360 [1305, 1430] | ≈1190 [1085, 1265] | +14% | no (just) |
| Humanoid-v1 | ≈2670 [2470, 2840] | ≈2330 [2110, 2560] | +15% | yes |
| Walker2d-v1 | ≈660 [630, 710] | ≈685 [640, 780] | **−4% (shared wins)** | yes |
| HalfCheetah-v1 | ≈2320 [2230, 2470] | ≈2000 [1800, 2200] | +16% | no (just) |
| Ant-v1 | ≈2680 [2600, 2790] | ≈2340 [2240, 2450] | +15% | no |

**Figure 15, right panel** (share of top-5%): separate / shared = 0.65/0.35 (Hopper), 0.61/0.39
(Humanoid), **0.48/0.52 (Walker2d)**, 0.64/0.355 (HalfCheetah), 0.66/0.34 (Ant), **0.61/0.395 (all)**.

**How strong is this finding? Moderately strong as a marginal, and it does not identify a mechanism.**
Four qualifications, all readable off the paper's own design:

1. **It is a marginal over a confounded comparison.** Under `separate`, App. E.1 samples four
   sub-choices (policy width, policy depth, value width, value depth). Under `shared`, it samples
   three *different* ones (shared width, shared depth, **baseline cost** $c_V \in \{0.001, 0.1, 1.0, 10.0, 100.0\}$).
   The two arms therefore differ in parameter count, in the number of hyperparameters that must come
   out right, and in the presence of a loss-weighting term. A shared configuration must additionally
   have drawn a workable $c_V$: four of the five sampled values are one-to-five orders of magnitude
   away from 1.0, so a large fraction of the shared arm is handicapped by a value-loss weight that a
   practitioner would never choose. **A marginal that pools over that sampling is not a clean
   shared-vs-separate contrast.**
2. **The paper never reports the effect of the value-loss weight.** Appendix E's per-choice figures
   are Figs. 17–30, covering C63, C58, C59, C57, C60, C61, C56, C49, C50, C52, C53, C55, C62, C24.
   There is **no figure for C48, C51 or C54** — because the experiment was rerun with the shared arm
   removed. So the quantity that would test the mechanism most directly, the balance between policy
   and value loss in a shared trunk, is **measured nowhere in this paper**.
3. **The effect is not uniform.** Walker2d prefers shared on both panels, and Humanoid's confidence
   intervals overlap. "Four out of five environments" is the paper's own honest phrasing.
4. **Everything downstream is conditioned on `separate`.** Because of the rerun, every other
   architecture result in Appendix E — including the last-layer-initialization result below — was
   measured on agents with **separate policy and value networks**.

**On the mechanism this project has been invoking** (that in a shared trunk the value loss dominates
the summed gradient, so the actor becomes a readout of features shaped by value regression):
Andrychowicz et al. **provide no evidence for or against it**. They report an outcome difference, they
name the loss-weighting hyperparameter that the mechanism would predict to be critical, and then they
delete that arm of the experiment before analysing it. The paper is a legitimate citation for
*"separate networks outperformed a shared trunk on 4/5 MuJoCo tasks in a large random search"*. It is
**not** a citation for *why*.

#### 4.2.4 Initialization of the policy's last layer (choice C57)

**Definition (App. B.7):** "For the initialization of both the last layer in the policy MLP / the
policy head (choice C57) and the last layer in the value MLP / the value head (choice C58), we
further consider a hyperparameter that **rescales the network weights of these layers after
initialization**." Sampled over $\{0.001, 0.01, 0.1, 1.0\}$ in App. E.1; the base configuration uses
`C57 = 0.01`, `C58 = 1.0`.

**Claim, verbatim (§3.2 and the "Most surprising finding" box on p. 2):** "the policy initialization
scheme significantly influences the performance while it is rarely even mentioned in RL publications.
In particular, we have found that initializing the network so that the initial action distribution
has zero mean, a rather low standard deviation and is independent of the observation significantly
improves the training speed." And: "This can be achieved by initializing the policy MLP with smaller
weights in the last layer (C57, Fig. 24, **this alone boosts the performance on Humanoid by 66%**) so
that the initial action distribution is almost independent of the observation and by introducing an
offset in the standard deviation of actions (C61)."

**Figure 24, left panel** (conditional 95th percentile by C57 value):

| Environment | 0.001 | 0.01 | 0.1 | 1.0 | best vs 1.0 |
|---|---|---|---|---|---|
| Hopper-v1 | ≈1405 | ≈1350 | ≈1340 | ≈1315 | +7% |
| Humanoid-v1 | ≈2560 | ≈2590 | ≈2340 | **≈1500** | **+73%** |
| Walker2d-v1 | ≈615 | ≈675 | ≈665 | ≈610 | +11% |
| HalfCheetah-v1 | ≈2260 | ≈2220 | ≈2225 | ≈1780 | +27% |
| Ant-v1 | ≈2530 | ≈2565 | ≈2620 | ≈2135 | +23% |

**Read the shape, not just the headline.** The relationship is **not monotone-in-smallness**; it is a
cliff at 1.0. Values 0.001, 0.01 and 0.1 are within each other's error bars on every environment; only
leaving the last layer at its default initialization scale (1.0) is clearly bad, and even that is
mild on Hopper and Walker2d. The paper's "66% on Humanoid" is the same effect as the +73% in the
table above, measured on the hardest environment where the score range is largest; the two numbers
differ because the paper's comparison and this one need not pick the same pair of bars, so treat 66%
as the authoritative figure and the table as its shape.

**The contrast that makes it interesting** is with the *value* head. Figure 19 (C58, last **value**
layer scaling, same four values) gives: Hopper ≈1305/1340/1370/1375 (mildly *increasing* in scale),
Humanoid ≈2510/2495/2370/2065, Walker2d ≈690/622/625/625, HalfCheetah ≈2225/1930/2035/2280 (a U
shape), Ant ≈2520/2510/2450/2455. Direction is inconsistent and magnitudes are small — matching the
camera-ready text: "The scale of the last layer initialization matters much less for the value MLP
(C58) than for the policy MLP (Fig. 19)."

**Two conditions to carry with the citation.** (a) The finding is about a **Gaussian continuous-action
policy** whose last layer emits the *mean* of an action distribution; "the action distribution is
almost independent of the observation at initialization" is a statement about a Gaussian mean, and
the recipe is explicitly a package with C61 (initial action standard deviation, where 0.5 was best on
four of five environments) and C63 (`tanh` vs clipping to bound actions). (b) It was measured **only
on separate-network agents**, per the App. E.1 rerun.

### 4.3 Everything else the paper concluded, compressed

Kept because two of these bear directly on the project's normalization dispute.

| Group | Choice | Finding (paper's own words, condensed) | Recommendation |
|---|---|---|---|
| Policy losses (§3.1) | C14 | PPO beats PG, V-trace, AWR, V-MPO and RPA on 4/5 envs; trust-region enforcement is "crucial for good sample complexity" | Use the PPO loss; start at clipping threshold 0.25 |
| Normalization (§3.3) | C64 observation normalization | "crucial for good performance on all environments apart from Hopper" | **Always** use it |
| Normalization (§3.3) | C68 gradient clipping | "small performance boost with the exact clipping threshold making little difference" | Secondary importance |
| Normalization (§3.3) | C65 observation clipping | "little evidence that clipping normalized observations helps" | Only as a divergence guard |
| Advantage estimation (§3.4) | C6, C8 | GAE and V-trace both beat N-step; no significant difference between GAE and V-trace; $\lambda = 0.9$ works everywhere | Use GAE, $\lambda = 0.9$ |
| Advantage estimation (§3.4) | **C13 PPO-style value-loss clipping** | "**hurts the performance regardless of the clipping threshold**" (footnote: "consistent with prior work [Engstrom et al.]") | **Do not use it** |
| Advantage estimation (§3.4) | C11 Huber value loss | "performed worse than MSE in all environments" | Use MSE |
| Training setup (§3.5) | C5 minibatch handling | Stale advantages (recomputed once per iteration, as in OpenAI Baselines) hurt; recomputing advantages once **per data pass** performs best of all four variants | Shuffle transitions, recompute advantages per epoch |
| Timesteps (§3.6) | C20 | $\gamma$ is "one of the most important hyperparameters"; 0.99 a safe start | Tune per environment |
| Optimizers (§3.7) | C23–C31 | Adam vs RMSprop makes little difference; LR 3e-4 safe; linear decay helps slightly (15% on Ant) | Adam, $\beta_1 = 0.9$, tuned LR |
| Regularization (§3.8) | C32–C46 | No evidence that entropy/KL regularization helps, except HalfCheetah — and there the benefit is independent of the constraint threshold, suggesting it comes from the initial penalty strength, not the constraint | Skip it (given PPO's trust region + careful init) |

### 4.4 Phase 1 — Foundational overview

**The problem.** Two research groups publish two reinforcement-learning algorithms and report
different scores. Some of that difference comes from the *idea* in each paper — a new loss function,
a new way of estimating how good an action was. But some of it comes from dozens of small decisions
buried in the code: how the network's weights were set at the start, whether the numbers fed into the
network were rescaled, how the training data was cut into batches. Those decisions are rarely written
down, so nobody can tell which part of the score belongs to the idea. That makes it impossible to
know whether the field is making progress.

**What they did.** They rebuilt an on-policy actor-critic agent — the family that includes PPO — from
scratch, but with **every one of these buried decisions exposed as a switch**: more than fifty
switches, labelled C1 through C68. Then they trained over a quarter of a million agents on five
simulated robot-locomotion tasks, randomly flipping the switches in one thematic group at a time
while holding the rest at a sensible default. For each switch they ask two questions: *if I set the
switch this way and tune everything else in the group with a modest random search, how good does my
best agent get?* and *among the very best agents that came out of this experiment, how often was the
switch set this way?*

**The main results, in plain terms.**

- **The initialization of the policy's final layer is one of the biggest levers in the whole study**, and almost nobody writes it down. If you shrink the weights of that last layer so the agent's initial behaviour is essentially "the same small random wiggle regardless of what I see", you learn much faster. Leaving that layer at its default scale cost 40% of the score on the hardest task.
- **Rescaling the critic's regression targets matters enormously, but not in a consistent direction.** Keeping a running mean and standard deviation of the value targets, training the critic in those normalized units, and converting back to real units at read-out was worth +78% on one task and −42% on another. The authors say outright that they do not know why.
- **Rescaling the advantages inside each minibatch — the single most widely implemented "trick" in PPO code — barely moved the needle.** It is not in their recommendation list, and their own default configuration leaves it off.
- **Two networks beat one shared network** on four of the five tasks, and their architecture recommendation says to keep the value network wide and unshared.
- Several things "everyone does" turned out to be harmful: clipping the value loss the way the reference PPO implementation does made results **worse** on every task and at every threshold.

**Initial takeaway.** The study's real contribution is a discipline, not a recipe: it separates
*choices that are load-bearing* (observation normalization, policy last-layer initialization, the
discount factor, the policy loss) from *choices that are merely traditional* (advantage
normalization, value-loss clipping, Huber losses, entropy regularization). The one caution to carry
away is that a switch can be load-bearing **and** have opposite signs on different tasks — value
normalization is the study's own example, and the honest recommendation it produces is "check
whether it helps on your environment", not "do this".

### 4.5 Phase 2 — Graduate-level deep dive

#### 4.5.1 Setting and notation

Following App. A: an environment $(\mathcal S, \mathcal A, p(s_0), r, p(s_{t+1}|s_t,a_t), \mathcal T(s_t,a_t), \gamma)$,
a policy $\pi(\cdot|s)$, return $R_t = \sum_{i \ge t} \gamma^{i-t} r_i$, value
$V^\pi(s_t) = \mathbb E_\pi[R_t | s_t]$, action-value $Q^\pi(s_t,a_t) = \mathbb E_\pi[R_t|s_t,a_t]$,
advantage $A^\pi(s_t,a_t) = Q^\pi(s_t,a_t) - V^\pi(s_t)$. Two networks are maintained: the policy
$\pi_\theta$ and an approximator $V_\phi \approx V^\pi$.

Advantage estimators (App. B.2), written as the paper writes them. $N$-step:

$$\hat V^{(N)}_t = \sum_{i=t}^{t+N-1} \gamma^{i-t} r_i + \gamma^N V(s_{t+N}), \qquad \hat A^{(N)}_t = \hat V^{(N)}_t - V(s_t).$$

GAE($\lambda$) as an exponentially weighted average of those:

$$\hat V^{\text{GAE}}_t = (1-\lambda)\sum_{N>0} \lambda^{N-1} \hat V^{(N)}_t, \qquad \hat A^{\text{GAE}}_t = \hat V^{\text{GAE}}_t - V(s_t).$$

The PPO policy loss, in the paper's own asymmetric form (App. B.3, footnote 19 — they use
$1/(1+\epsilon)$ as the lower clip bound rather than the original paper's $1-\epsilon$, "as it is
more symmetric"):

$$\mathcal L^{\epsilon}_{\text{PPO}} = -\min\!\left[\rho_t \hat A^\pi_t,\; \operatorname{clip}\!\left(\rho_t, \tfrac{1}{1+\epsilon}, 1+\epsilon\right)\hat A^\pi_t\right], \qquad \rho_t = \frac{\pi(a_t|s_t)}{\mu(a_t|s_t)},$$

with $\mu$ the behavioural policy that generated the data.

#### 4.5.2 What C67 does to the PPO objective — a derivation

*This subsection derives consequences of the paper's definitions. The consequences are standard, and
the paper does not state them; they are included because they explain why "advantage normalization is
just a learning-rate rescale" is only half true, which in turn constrains how Figure 35 may be cited.*

Let a minibatch $B$ of size $m$ (their default $m = 64$) carry advantage estimates
$\{\hat A_i\}_{i \in B}$ with empirical moments

$$\bar A = \frac{1}{m}\sum_{i \in B} \hat A_i, \qquad s_A^2 = \frac{1}{m}\sum_{i \in B}(\hat A_i - \bar A)^2 .$$

C67 replaces $\hat A_i$ by $\tilde A_i = (\hat A_i - \bar A)/s_A$. Decompose this into a scaling step
and a centring step and treat them separately, because they behave completely differently.

**Scaling commutes with the clip.** For any $c > 0$ and any reals $x, y$, $\min(cx, cy) = c\min(x,y)$.
Hence

$$\mathcal L^{\epsilon}_{\text{PPO}}\!\left(\tfrac{\hat A_t}{s_A}\right) = \frac{1}{s_A}\,\mathcal L^{\epsilon}_{\text{PPO}}(\hat A_t),$$

so dividing by the batch standard deviation multiplies the whole minibatch loss — and therefore its
gradient — by the scalar $1/s_A$. **On this part alone, C67 is exactly a per-minibatch adaptive
learning rate.** With a per-parameter adaptive optimizer such as Adam (their default, C23), a
constant rescale of the gradient is very nearly absorbed by the second-moment normalization
$\hat m/(\sqrt{\hat v} + \epsilon_{\text{Adam}})$ with $\epsilon_{\text{Adam}} = 10^{-7}$, so the
*expected* effect of the scaling half of C67 under Adam is small. That is consistent with, though not
proof of, the flatness of Figure 35.

**Centring does not commute with the clip.** Expand the loss by the sign of the advantage. For
$\hat A_t > 0$, the $\min$ selects the smaller of $\rho_t$ and its clipped version:

$$\mathcal L_{\text{PPO}} = -\hat A_t \min\!\left(\rho_t, \min(\rho_t, 1+\epsilon)\right) = -\hat A_t \min(\rho_t, 1+\epsilon),$$

whose gradient in $\theta$ vanishes once $\rho_t > 1+\epsilon$ — the update is switched off when the
policy has already increased this action's probability too much. For $\hat A_t < 0$, multiplying by a
negative number reverses the $\min$ into a $\max$:

$$\mathcal L_{\text{PPO}} = -\hat A_t \max\!\left(\rho_t, \tfrac{1}{1+\epsilon}\right),$$

whose gradient vanishes once $\rho_t < 1/(1+\epsilon)$. **The side on which the objective is switched
off depends on $\operatorname{sign}(\hat A_t)$.** Subtracting $\bar A$ flips that sign for every sample
with $0 < \hat A_i < \bar A$ (or $\bar A < \hat A_i < 0$), so centring changes *which* samples are
pessimistically clipped and in *which direction* — it is a change of objective, not a rescale.

Two consequences for how Figure 35 may be used:

- C67's two halves have different characters, and Figure 35 measures only their *combination*. A finding of "not much effect" for the pair does not license a claim about either half separately.
- With their default $m = 64$ drawn from an iteration of 2048 transitions under `batch_mode = Shuffle transitions`, $\bar A$ is a 64-sample estimate of a quantity that GAE has already centred in expectation (since $\hat A^{\text{GAE}}_t$ subtracts $V(s_t)$). So the centring is a **noise term of mean roughly zero**, which is precisely the regime in which one should *expect* to measure "not much".

The paper itself notices the sign-flipping property in a different context: in App. B.3 it remarks
that V-MPO's "top half of advantages" rule and RPA's $[\hat A_t > 0]$ rule "become even more similar
if advantage normalization is used" — because centring moves the zero to the batch mean.

#### 4.5.3 What C66 does to the value objective — a derivation

Write $\sigma := \max(v_\rho, 10^{-6})$ and let $y$ denote a value target (the return estimate from
B.2). Without normalization the critic minimizes

$$\mathcal L^{\text{raw}}_V = \tfrac12\,\mathbb E\big[(V_\phi(s) - y)^2\big].$$

With C66 the network emits $V_{\text{out}}$, the prediction is read out as
$\hat V = v_\mu + \sigma V_{\text{out}}$, and the regression is performed in normalized units:

$$\mathcal L^{\text{norm}}_V = \tfrac12\,\mathbb E\!\left[\left(V_{\text{out}}(s) - \frac{y - v_\mu}{\sigma}\right)^{\!2}\right] = \frac{1}{2\sigma^2}\,\mathbb E\!\left[\big(\underbrace{v_\mu + \sigma V_{\text{out}}(s)}_{=\;\hat V(s)} - y\big)^2\right] = \frac{1}{\sigma^2}\,\mathcal L^{\text{raw}}_V .$$

So **as a function of the read-out value $\hat V$, C66 is exactly the raw squared loss reweighted by
$1/\sigma^2$** — and this is the arithmetic behind the authors' guess that it "affects the performance
by changing the speed of the value function fitting".

But the identity above is a statement about the loss *as a function of $\hat V$*, and the optimizer
acts on $\phi$, not on $\hat V$. Two distinct mechanisms follow, and they are not equally plausible:

1. **A gradient-magnitude mechanism.** Differentiating the middle expression above and using $\partial \hat V/\partial V_{\text{out}} = \sigma$,

$$\nabla_\phi \mathcal L^{\text{norm}}_V = \frac{1}{\sigma^2}\,\mathbb E\big[(\hat V - y)\,\sigma\,\nabla_\phi V_{\text{out}}\big] = \frac{1}{\sigma}\,\mathbb E\big[(\hat V - y)\,\nabla_\phi V_{\text{out}}\big],$$

   i.e. the $1/\sigma^2$ from the loss prefactor and the $\sigma$ from the chain rule leave a net $1/\sigma$ relative to the raw-unit gradient $\mathbb E[(V_\phi - y)\nabla_\phi V_\phi]$. Under **separate** networks and Adam, a constant rescale of one loss is largely absorbed per-parameter by the second-moment normalization, so this mechanism should be weak — unless $\sigma$ drifts fast enough that the second-moment estimate never catches up, which is plausible early in training while the value scale is still growing. Under a **shared** trunk it is *not* absorbed: the trunk parameter $\psi$ receives $\nabla_\psi \mathcal L_\pi + c_V \nabla_\psi \mathcal L_V$ and Adam normalizes the **sum**, so changing $\sigma$ changes the *mixing ratio* between the two terms before the optimizer ever sees them.
2. **A representational mechanism.** With $\sigma \approx 1$ the network must emit outputs of order 1; with raw targets on Humanoid it must emit outputs of order $10^3$. Under their fixed initialization (orthogonal, gain 1.41, last value layer scaled by C58 = 1.0) and `tanh` hidden units, producing outputs three orders of magnitude larger requires the final linear layer's weights to grow by the same factor, which takes many steps at a fixed learning rate. This mechanism is independent of the optimizer's scale invariance and survives the Adam objection.

**The paper adjudicates neither.** Its exact words are "We are not sure why the value function scale
matters that much but suspect that it affects the performance by changing the speed of the value
function fitting", plus footnote 11 raising a possible interaction with value clipping. Both
mechanisms above are this reviewer's reconstruction from the paper's own definitions and are labelled
as such; **do not cite Andrychowicz et al. for either of them.**

Note finally the property that separates C66 from a per-batch z-score of the critic target: because
$\hat V = v_\mu + \sigma V_{\text{out}}$ is computed at read-out, the advantage
$\hat A_t = \hat V^{\text{GAE}}_t - V(s_t)$ is formed entirely in **raw reward units** regardless of
whether C66 is on. Normalizing the target and normalizing the advantage are, in their implementation,
strictly independent knobs — which is exactly why C66 and C67 are separate switches in the same
experiment.

#### 4.5.4 The shared-trunk objective, and the quantity the study did not measure

Under `C47 = shared`, App. B.7 gives one MLP with two linear heads and a scalar
`Baseline cost (shared)` $c_V$ (C54) that "rescales the contribution of the value loss to the full
objective function". Writing $\psi$ for trunk parameters:

$$\mathcal L_{\text{total}}(\psi) = \mathcal L_\pi(\psi) + c_V\,\mathcal L_V(\psi), \qquad \nabla_\psi \mathcal L_{\text{total}} = \nabla_\psi \mathcal L_\pi + c_V\,\nabla_\psi \mathcal L_V .$$

Combining this with §4.5.3, the trunk gradient under a shared network with value normalization is

$$\nabla_\psi \mathcal L_{\text{total}} = \nabla_\psi \mathcal L_\pi + \frac{c_V}{\sigma^2}\,\nabla_\psi \mathcal L^{\text{raw}}_V,$$

i.e. **$c_V$ and $\sigma^{-2}$ enter the shared trunk through the same slot.** That is a formal
statement about their implementation, and it is the reason the C54 marginal would have been the
informative measurement for any claim about value-loss dominance in a shared trunk.

**That marginal does not exist in the paper.** C54 is sampled in App. E.1 and then removed from the
analysis when the experiment is rerun with `separate` only; no figure in Appendices D–K reports it.
Any claim of the form "in a shared trunk the value loss dominates the summed gradient and the actor
degenerates into a readout of value features" must therefore be sourced elsewhere — Andrychowicz et
al. is silent on it, and the outcome-level Figure 15 does not substitute, because it pools over five
values of $c_V$ spanning five orders of magnitude.

### 4.6 Relevance and transfer to this project

**Setting mismatch, stated once so it can be attached to every citation.** Andrychowicz et al. train
feed-forward Gaussian policies on five fully-observed continuous-control MuJoCo tasks, with observation
normalization on by default, episodes capped at 1000 steps, and a score that rewards fast learning.
This project runs a **discrete-action, recurrent (GRU), shared-trunk** agent in a partially observed
grid world with a **homeostatic** reward and a survival-step performance measure. Four of the paper's
structural assumptions are violated at once, and the paper contains no experiment that varies any of
them.

| Finding | Transfer expectation | Reasoning |
|---|---|---|
| Observation normalization is crucial (C64) | **Likely transfers** | The mechanism is about network input conditioning, which is architecture- and action-space-agnostic. |
| Advantage normalization has little effect (C67) | **Weak transfer; treat as "no strong prior either way"** | The derivation in §4.5.2 is action-space-agnostic, but the *magnitude* of the effect depends on advantage scale and minibatch size, both of which differ here. Their $m=64$ over 2048 transitions is not this project's batching. |
| Value-target normalization matters strongly but with inconsistent sign (C66) | **Transfers as a warning, not as a direction** | The only defensible transfer is the paper's own recommendation: *check empirically on your environment*. It emphatically does **not** license "normalize the critic target". |
| Separate networks beat a shared trunk (C47) | **Do not transfer as a design instruction** | Confounded with capacity and with $c_V$ sampling (§4.2.3); measured on MLPs where "separate" is cheap. With a recurrent trunk, duplicating the GRU changes memory, compute and the credit-assignment path — a different trade-off entirely, which this paper never faced. |
| Small policy last-layer init (C57) | **Partially transfers, with a translation problem** | Their mechanism is "make the initial *Gaussian mean* observation-independent and low-variance". The discrete-action analogue is "make the initial *categorical logits* near-uniform and observation-independent", which is the same intent and is what a scaled-down final layer does. But the C61/C63 half of their recipe (initial action standard deviation, `tanh` squashing) has **no discrete-action counterpart at all**, and their evidence comes from separate-network agents. |
| PPO-style value-loss clipping hurts (C13) | **Likely transfers** | Corroborated independently by Engstrom et al. (§5); the mechanism is about the value objective, not the action space. |
| GAE with $\lambda = 0.9$; MSE not Huber | **Plausibly transfers** | Standard estimator behaviour; but $\lambda$ interacts with episode length and reward density, both of which differ here. |
| Anything about reward scaling | **Not addressed** | The paper does not study it. |
| Anything about recurrence, partial observability, or discrete actions | **Not addressed** | No such experiment exists in the paper. |

**One implementation note, flagged for the record and not acted on here.** §3.5 reports that the
OpenAI-Baselines convention of computing advantages **once per iteration** and then shuffling
transitions across minibatches produces measurably worse results than **recomputing advantages at the
start of every epoch**, and that the latter "performs best among all variants" (Figs. 53 and 58). If
this project's trainer computes advantages once per rollout and then performs multiple epochs, this
is a cheap, independently-evidenced change. Evaluating whether that is the case is a code question,
not a literature question — recommend routing it to `senior-developer` for an `issue_plan` rather
than treating this paragraph as a design decision.

### 4.7 Appendix: Section-by-Section Backbone

Section order is the paper's own. Camera-ready section numbers; appendix content from the preprint.

| § | Content |
|---|---|
| **Abstract** | >50 choices implemented in a unified on-policy deep actor-critic framework; >250,000 agents; five continuous control environments; insights and practical recommendations. |
| **1 Introduction** | Published algorithm descriptions diverge from their implementations; this makes attribution of progress impossible. Three contributions: (1) implement >50 choices in one code base (SEED RL), (2) train >250,000 agents, (3) analyse and recommend. Flags the "most surprising finding": policy initialization matters and is rarely mentioned. |
| **2 Study design** | Setting: on-policy policy-iteration actor-critic for continuous control; five Gym/MuJoCo tasks. How the unified agent was built (survey prior work → implement → verify that the OpenAI-Baselines PPO settings reproduce published PPO scores). Why one-choice-at-a-time is inadequate (bad random configurations learn nothing; choices interact). Design: thematic groups, random sampling within group, everything else pinned to a competitive base configuration (≈PPOv2 defaults at 256 envs). Two statistics: conditional 95th percentile with binomial CI, and share of choice values among top-5% configurations. Performance score = mean of periodic 100-episode evaluations ∝ area under learning curve; median over 3 seeds. |
| **3.1 Policy losses** (App. D) | Compares PG, V-trace, PPO, AWR, V-MPO, RPA and sweeps their hyperparameters. PPO best on 4/5, still best under per-loss optimal hyperparameters on the two hardest tasks. PG and V-trace suffer from multi-epoch off-policyness. Recommendation: PPO loss, clipping threshold ≈0.25. |
| **3.2 Networks architecture** (App. E) | Separate > shared on 4/5 (Fig. 15); experiment rerun with `separate` only. Policy width is task-dependent and two-sided; value width has no downside; depth 2 works for both; `tanh` best activation, `relu` worst. Policy last-layer scaling is a large effect (+66% on Humanoid); initial action std 0.5 best on 4/5; `tanh` action squashing slightly better than clipping. Value last-layer scaling matters much less; initializer family (beyond last-layer scale) matters little except He is suboptimal. Recommendation: 100× smaller last policy layer; softplus std with negative offset; `tanh` everywhere; wide unshared value MLP. |
| **3.3 Normalization and clipping** (App. F) | C64 observation normalization crucial except Hopper. C66 value-function normalization very strong but sign-flipping (crucial on HalfCheetah/Humanoid, slight help on Hopper/Ant, significant harm on Walker2d); cause unknown, suspected to be value-fitting speed; footnote raises value-clipping interaction and the disagreement with Engstrom et al. on Walker. C67 per-minibatch advantage normalization "seems not to affect the performance too much". C65 observation clipping: little evidence of benefit. C68 gradient clipping: small boost, threshold-insensitive. Recommendation: always normalize observations; **check** value normalization; gradient clipping secondary. |
| **3.4 Advantage estimation** (App. G) | GAE and V-trace both beat N-step; no significant GAE-vs-V-trace difference; $\lambda = 0.9$ good everywhere. PPO-style value-loss clipping hurts at every threshold. Huber value loss worse than MSE at every threshold. Recommendation: GAE $\lambda = 0.9$, MSE, no value clipping. |
| **3.5 Training setup** (App. H) | Multiple epochs over the data is crucial. More parallel environments hurts sample complexity (shorter fragments, earlier bootstrapping) but helps wall-clock. Larger minibatches do not hurt in the range tested. Iteration size matters significantly. Of the four `batch_mode` variants, "shuffle transitions with advantages recomputed each epoch" is best; stale advantages hurt. |
| **3.6 Timesteps handling** (App. I) | Discount factor is among the most important hyperparameters (0.99 a safe start). Frame skip helps on 2/5. Special handling of time-limit-abandoned episodes makes no difference at a 1000-step limit. |
| **3.7 Optimizers** (App. J) | Adam vs RMSprop: no consistent winner. LR strongly matters; 3e-4 safe for Adam. Adam benefits from momentum 0.9; RMSprop centering and both $\epsilon$'s make no difference. Linear LR decay helps slightly (15% on Ant). |
| **3.8 Regularization** (App. K) | No evidence that entropy or KL regularization helps except on HalfCheetah, where the benefit is threshold-independent — attributed to initial penalty strength rather than the constraint. Conjecture: PPO's trust region and careful initialization already supply what regularization would. |
| **4 Related work** | Islam et al., Henderson et al. (reproducibility); Tucker et al. (gains attributable to implementation details); Engstrom et al. as closest prior work; parallels to large-scale studies in GANs, NLP, disentanglement, model-based RL, convnets. |
| **5 Conclusions** | >250,000 experiments; recommendations delivered; initial action distribution highlighted as a fruitful direction. |
| **App. A** (preprint only) | RL background and notation; explicitly assumes full observability. |
| **App. B** (preprint only) | The complete choice catalogue. B.1 data collection and the four `batch_mode` variants; B.2 advantage estimators and value losses; B.3 the six policy losses with equations; B.4 timestep handling; B.5 optimizers; B.6 policy regularizers; B.7 network architecture incl. shared-vs-separate and the shared baseline cost; B.8 action-distribution parameterization; **B.9 data normalization and clipping — the definitions of C64–C68**. |
| **App. C** (preprint only) | Table 2, the default configuration used in every experiment. Source for `C67 = False`, `C66 = Average`, `C47 = Shared` (with the internal inconsistency documented in §4.2.2), `C57 = 0.01`, `C13 = 0.2`. |
| **App. D–K** (preprint only) | Per-experiment design (which choices were sampled, over which grids) and results: aggregate performance quantiles, training curves, and one per-choice figure per choice. Figures 1–92. The evidence for every claim in §3. |

## 5. Engstrom et al. (2020) — *Implementation Matters in Deep Policy Gradients: A Case Study on PPO and TRPO*

### 5.1 Fixed extraction block

| Field | Value |
|---|---|
| **Venue** | **ICLR 2020.** The running header on every page misprints "ICLR 2019" — see §2.2 for why that is a misprint and how it propagated into Andrychowicz et al.'s reference list. |
| **PDF** | `docs/project/references/ppo_implementation_details/sources/Engstrom et al. 2020 - Implementation matters in deep policy gradients (ICLR camera-ready).pdf` (14 pp. incl. Appendix A) |
| **Authors / affiliation** | Engstrom\*, Ilyas\*, Santurkar, Tsipras (MIT), Janoos (Two Sigma), Rudolph (MIT/Two Sigma), Mądry (MIT). \*equal contribution. |
| **Object of study** | Nine "code-level optimizations" present in the OpenAI Baselines PPO implementation but absent or under-described in the PPO paper, and their effect relative to the *algorithmic* difference between PPO and TRPO. |
| **Environments** | MuJoCo Humanoid-v2, Walker2d-v2, Hopper-v2. **Continuous control only.** |
| **Networks** | **Separate** policy and value MLPs, both `[64, 64]`, with **separate learning rates** (App. A.1, Tables 4–6: "Policy LR (Adam)" and "Value network LR" are distinct entries). No recurrence; the strings "recurrent", "LSTM", "shared" do not appear. |
| **Budget** | Figure 1: $2^4 = 16$ on/off configurations of the four ablated optimizations × 5 seeds = **80 agents per environment**, with a learning-rate grid search inside each configuration and the best learning rate selected before the agents enter the histogram. Tables 2–3: **at least 80 agents per cell**, 95% confidence intervals from a 1000-sample bootstrap. |
| **Code** | `https://github.com/MadryLab/implementation-matters` |
| **Not studied at all** | **Advantage normalization.** The word "advantage" occurs exactly three times in the paper — once in the phrase "one disadvantage of TRPO", once listing "advantage estimation methods" among components, once in the GAE citation. **None concerns standardizing advantages.** Also not studied: shared-vs-separate trunks; discrete action spaces; recurrence. |

### 5.2 The nine code-level optimizations, as the paper defines them (§3)

Transcribed because the list is the paper's most-cited object and is often misquoted.

1. **Value function clipping.** The PPO paper suggests $\mathcal L_V = (V_{\theta_t} - V_{\text{targ}})^2$; the reference implementation instead uses $\mathcal L_V = \min\big[(V_{\theta_t} - V_{\text{targ}})^2, \big(\operatorname{clip}(V_{\theta_t}, V_{\theta_{t-1}}-\varepsilon, V_{\theta_{t-1}}+\varepsilon) - V_{\text{targ}}\big)^2\big]$, with $\varepsilon$ tied to the policy-ratio clipping constant.
2. **Reward scaling.** "Rather than feeding the rewards directly from the environment into the objective, the PPO implementation performs a certain discount-based scaling scheme. In this scheme, **the rewards are divided through by the standard deviation of a rolling discounted sum of the rewards (without subtracting and re-adding the mean)**."
3. **Orthogonal initialization and layer scaling** — "an orthogonal initialization scheme with scaling that varies from layer to layer", in place of the framework default.
4. **Adam learning rate annealing.**
5. **Reward clipping** to a preset range, "usually $[-5,5]$ or $[-10,10]$".
6. **Observation normalization** to mean-zero, variance-one.
7. **Observation clipping**, "usually $[-10,10]$".
8. **Hyperbolic tan activations.**
9. **Global gradient clipping** on the $\ell_2$ norm of the concatenated gradients, threshold 0.5.

**Only the first four were ablated**, and the paper says so in a footnote: "Due to restrictions on
computational resources, we could only perform a full ablation on the first four of the identified
optimizations." Anyone citing this paper for "the nine optimizations matter" is over-reading: the
evidence covers four.

### 5.3 Reward scaling — the answer to this project's question 2

**Exact mechanism (Appendix A.2, Algorithm 1, transcribed):**

```
procedure INITIALIZE-SCALING()
    R_0 ← 0
    RS ← RunningStatistics()          # tracks mean, standard deviation

procedure SCALE-OBSERVATION(r_t)      # input: a reward r_t
    R_t ← γ R_{t-1} + r_t             # γ is the reward discount
    ADD(RS, R_t)
    return r_t / STANDARD-DEVIATION(RS)
```

Three structural facts follow directly from those five lines, and they decide how this result may be
used:

- **The divisor is a running statistic** over a *discounted accumulator* $R_t$, updated every step and never reset per batch. It is not recomputed from the current rollout.
- **The numerator is the per-step reward**, not a return and not an advantage. The scaling is applied **upstream of everything** — before the return, before the value target, before the advantage.
- **There is no mean subtraction.** The paper says so parenthetically and Algorithm 1 confirms it: only `STANDARD-DEVIATION(RS)` is used. A reward of constant sign stays constant in sign.

**How it is separated from the other optimizations.** Figure 1 is a $2^4$ full factorial over
{value clipping, reward scaling, orthogonal initialization, learning-rate annealing}. For each of the
16 configurations a learning-rate grid search is run, the best learning rate is chosen by average
reward over 5 seeds, and the 5 resulting agents enter a pool of 80. Figure 1 then plots, for each
optimization separately, the **survival function** $1 - \text{CDF}(\text{reward})$ of that pool,
partitioned by whether that one optimization was on or off — marginalizing over the other three. So
the separation is a **marginal over a full factorial with the learning rate re-tuned inside every
cell**, which is a genuinely strong design; what it is not is an estimate of an individual effect
size, because no number is ever reported for it.

**Finding, verbatim (Figure 1 caption):** "Our results show that **reward normalization, Adam
annealing, and network initialization each significantly impact the rewards landscape with respect to
hyperparameters, and were necessary for attaining the highest PPO reward within the tested
hyperparameter grid.**"

**What the figure looks like, read qualitatively** (the paper prints no numbers for it, and the plot
legends partially occlude the curves, so nothing more precise is available):

| Panel | Humanoid-v2 | Walker2d-v2 |
|---|---|---|
| `norm_rewards` (`returns` vs `none`) | The two survival curves separate early and stay separated; `none` has effectively no mass above ≈800 reward while `returns` extends past 1200. **Largest visual separation of the four panels.** | `none` has essentially no mass above ≈2500–3000; `returns` carries the distribution out to ≈5000. |
| `anneal_lr` (True/False) | True dominates through the tail; False ends near 800. | True dominates above ≈2800. |
| `initialization` (orthogonal/xavier) | orthogonal dominates through the tail. | orthogonal dominates above ≈2500. |
| `value_clipping` (True/False) | Curves nearly coincide; **False is marginally ahead** in the 600–900 band, True has a thin extreme tail. | **False is visibly ahead** through the 2500–4000 band; True has a thin extreme tail. |

**Value clipping is conspicuously missing from the caption's list of what mattered**, and the panels
show why. This is an independent corroboration of Andrychowicz et al.'s stronger finding (their C13,
"hurts the performance regardless of the clipping threshold"), and Andrychowicz's footnote cites this
paper as consistent prior work.

**The one numeric table bundles everything.** Table 2 does not isolate reward scaling; it compares
whole algorithms. `PPO-M` ("PPO-MINIMAL") is defined as PPO that "uses the standard value network
loss, **no reward scaling**, the default network initialization, and Adam with a fixed learning rate",
and `TRPO+` is TRPO with PPO's code-level optimizations found via grid search. Both are trained with
≥80 agents per cell and 95% bootstrap confidence intervals:

| | Walker2d-v2 | Hopper-v2 | Humanoid-v2 |
|---|---|---|---|
| PPO | 3292 [3157, 3426] | 2513 [2391, 2632] | 806 [785, 827] |
| PPO-M | 2735 [2602, 2866] | 2142 [2008, 2279] | 674 [656, 695] |
| TRPO | 2791 [2709, 2873] | 2043 [1948, 2136] | 586 [576, 596] |
| TRPO+ | 3050 [2976, 3126] | 2466 [2381, 2549] | 1030 [979, 1083] |
| **AAI** (algorithmic) | 242 | 99 | 224 |
| **ACLI** (code-level) | 557 | 421 | 444 |

with the two summary metrics defined in §5 as

$$\text{AAI} = \max\{\,|\text{PPO} - \text{TRPO+}|,\; |\text{PPO-M} - \text{TRPO}|\,\}, \qquad \text{ACLI} = \max\{\,|\text{PPO} - \text{PPO-M}|,\; |\text{TRPO+} - \text{TRPO}|\,\}.$$

Code-level effects exceed algorithmic effects by 2.3× (Walker2d), 4.3× (Hopper) and 2.0× (Humanoid).
**Reward scaling is one of four things varied between PPO and PPO-M**, so its individual contribution
to those gaps is not identified by this table — only Figure 1 speaks to it individually, and only as
a distribution shift.

**A third setting the main text never mentions.** The hyperparameter tables (App. A.1, Tables 4–6)
list `Reward normalization` with values `returns`, `rewards` and `none`. `returns` is Algorithm 1
above; `rewards` evidently divides by the standard deviation of the raw rewards rather than of the
discounted accumulator. The grid search selected `rewards` for PPO-NoClip on Walker2d and Hopper, and
`returns` for PPO and TRPO+ everywhere. No comparison of the two is reported.

**Does anyone here normalize the critic's regression target per batch?** No. Engstrom et al. do not
normalize the value target at all: their value-side intervention is *clipping* the value loss, and
their scale-side intervention acts on the reward, upstream of both the target and the advantage. This
is worth stating flatly because of a terminological trap described in §5.6.

### 5.4 Phase 1 — Foundational overview

**The question.** PPO scores better than TRPO on standard benchmarks. TRPO enforces a "trust region"
— a hard limit on how far each update may move the policy — using an expensive calculation; PPO
replaces that with a cheap trick that clips the objective. Everyone assumed the clipping trick was
why PPO wins. This paper asks whether that is true.

**What they did.** They read the reference PPO implementation line by line and found nine
modifications that are in the code but not (or barely) in the paper: rescaling rewards, rescaling and
clipping observations, a particular way of setting the network's initial weights, decaying the
learning rate, clipping the value loss, and so on. They then built four algorithms covering every
combination of *which core step you take* (PPO's or TRPO's) and *whether you use these extra code
tricks*, and trained at least eighty agents in each cell.

**The three results.**

1. **The code tricks matter more than the algorithm.** Switching between the PPO and TRPO update rules moved the score by 99–242 points depending on the task; adding or removing the code tricks moved it by 421–557 points, two to four times as much.
2. **The clipping trick is not what makes PPO work.** An ablation on four of the tricks found that reward rescaling, learning-rate annealing and the initialization scheme were each necessary to reach PPO's best scores, while clipping the value loss was not on that list. Separately, a version of PPO with the clipping removed entirely (but the code tricks kept and tuned) beat the version with clipping but without the tricks, on all three tasks.
3. **The tricks change what the algorithm *is*, not just how well it scores.** PPO's clipping is supposed to keep successive policies close together. They measured the actual distance and found the clipping does **not** enforce the limit it appears to promise — the policy ratio exceeded the supposed bound by up to 17× on one task and 30× on another. What keeps successive policies close is a side effect of *how* the objective is optimized, which the code tricks change.

**Initial takeaway.** The paper's lasting contribution is a warning about attribution: when two RL
methods are compared, the difference reported may be a difference in their codebases rather than in
their ideas. Its specific empirical result relevant here is that **rescaling rewards by a running
estimate of the return's spread was one of the things PPO could not do without** — and note that
*reward* rescaling is a different operation from the *advantage* rescaling this project has been
debating; this paper never tests the latter.

### 5.5 Phase 2 — Graduate-level deep dive

#### 5.5.1 The two objectives, as the paper writes them

TRPO (their Eq. 1), with the state-wise KL constraint that is in practice replaced by a mean over
observed states and a second-order approximation:

$$\max_\theta \; \mathbb E_{(s_t,a_t)\sim\pi}\!\left[\frac{\pi_\theta(a_t|s_t)}{\pi(a_t|s_t)}\hat A^\pi(s_t,a_t)\right] \quad \text{s.t.} \quad D_{\text{KL}}\big(\pi_\theta(\cdot|s)\,\|\,\pi(\cdot|s)\big) \le \delta \;\; \forall s.$$

PPO (their Eqs. 2–3), with the **symmetric** $1\pm\varepsilon$ clip — note this differs from
Andrychowicz et al.'s $[1/(1+\epsilon),\,1+\epsilon]$ variant:

$$\max_\theta \; \mathbb E_{(s_t,a_t)\sim\pi}\!\left[\min\Big(\operatorname{clip}(\rho_t, 1-\varepsilon, 1+\varepsilon)\,\hat A^\pi(s_t,a_t),\; \rho_t\,\hat A^\pi(s_t,a_t)\Big)\right], \qquad \rho_t = \frac{\pi_\theta(a_t|s_t)}{\pi(a_t|s_t)}.$$

#### 5.5.2 Why clipping does not enforce a trust region — the paper's argument

Define the unclipped and clipped surrogates

$$L_\theta := \mathbb E_{(s,a)\in\tau\sim\pi}\!\left[\rho\,A^\pi(s,a)\right], \qquad L^C_\theta := \mathbb E_{(s,a)\in\tau\sim\pi}\!\left[\operatorname{clip}(\rho, 1-\varepsilon, 1+\varepsilon)\,A^\pi(s,a)\right].$$

The paper then states the per-sample gradient of the PPO objective as

$$\nabla_\theta L_{\text{PPO}} = \begin{cases}\nabla_\theta L_\theta & \text{if } \dfrac{\pi_\theta(a|s)}{\pi(a|s)} \in [1-\epsilon,\,1+\epsilon] \;\text{ or }\; L^C_\theta < L_\theta,\\[2mm] 0 & \text{otherwise.}\end{cases}$$

Two consequences the authors draw:

1. **The first update is unconstrained.** At the start of each policy-improvement phase $\pi_\theta = \pi$, so every ratio equals 1, every sample is inside the clip window, and the first gradient step is *identical* to a step on the unclipped surrogate. Nothing in the objective bounds its size.
2. **The realized trust region is a property of the optimizer, not the objective.** Because the objective is flat (zero gradient) only *after* a ratio has already left the window, "the size of step we take is determined solely be[y] the steepness of the surrogate landscape … and we can end up moving arbitrarily far from the trust region."

**Measurement (Figure 2, Humanoid-v2; Figure 3, Walker2d-v2 and Hopper-v2).** They log, per training
step, mean reward, $\max_t \rho_t$, and mean KL between successive policies, for TRPO, PPO and PPO-M:

- **All three algorithms violate the ratio-based trust region.** The maximum ratio reaches ≈17.5 on Humanoid and ≈30 on Hopper against a nominal bound of $1+\varepsilon = 1.2$. The violation occurs for PPO and PPO-M *despite* both being trained with the ratio-clipping objective.
- **All three maintain a mean-KL trust region** at their tuned hyperparameters (compare TRPO's bound of 0.07).
- **PPO and PPO-M maintain it differently**: "while PPO-M KL trends up as the number of iterations increases, PPO KL peaks halfway through training before trending down again." Since PPO and PPO-M share the identical core objective, that difference is produced by the code-level optimizations alone.
- The same measurements on a **held-out** set of state-action pairs are "qualitatively nearly identical" (Fig. 4), so this is not an artefact of evaluating on the training batch.

#### 5.5.3 PPO without clipping (Table 3)

`PPO-NOCLIP` keeps the code-level optimizations (with their configuration found by grid search) and
removes clipping entirely — implemented, per Tables 4–6, by setting `PPO Clipping ε = 1e+32`.

| | Walker2d-v2 | Hopper-v2 | Humanoid-v2 |
|---|---|---|---|
| PPO | 3292 [3157, 3426] | 2513 [2391, 2632] | 806 [785, 827] |
| PPO (Baselines reference) | 3424 | 2316 | — |
| PPO-M | 2735 [2602, 2866] | 2142 [2008, 2279] | 674 [656, 695] |
| PPO-NOCLIP | 2867 [2701, 3024] | 2371 [2316, 2424] | **831 [798, 869]** |

PPO-NOCLIP beats PPO-M on all three tasks and beats full PPO on Humanoid. Their conclusion, verbatim
from the abstract and §5: the code-level optimizations "are responsible for most of PPO's gain in
cumulative reward over TRPO", and "the clipping mechanism is not necessary to achieve high
performance". The paper's own caveat, in footnote 6, is that PPO-NOCLIP explores a strict subset of
PPO's configuration space (PPO can always set $\varepsilon$ large), so this is a statement about what
is *necessary*, not about what is *optimal*.

#### 5.5.4 What reward scaling does to the two loss terms — a derivation

*Reviewer-supplied, from the paper's own definitions. The paper does not perform this analysis; it is
included because it is the cleanest way to see how Engstrom's intervention differs in kind from
per-batch advantage or return normalization, which is the distinction this project needs.*

Let $c > 0$ be the running scale $\text{STANDARD-DEVIATION}(RS)$ and suppose it is momentarily
frozen, so the environment's rewards are transformed as $r_t \mapsto r_t / c$. Because the return, the
value target and the advantage are all **positively homogeneous of degree 1** in the rewards:

$$R_t = \sum_{i\ge t}\gamma^{i-t}r_i \;\mapsto\; R_t/c, \qquad V^\pi \;\mapsto\; V^\pi/c, \qquad \hat A_t \;\mapsto\; \hat A_t/c .$$

Now apply this to each loss term. The PPO policy loss is homogeneous of degree 1 in $\hat A$
(the $\min$ commutes with positive scaling, exactly as in §4.5.2):

$$\mathcal L_\pi \;\mapsto\; \frac{1}{c}\,\mathcal L_\pi .$$

The value regression is homogeneous of degree **2** in its residual, and both $V$ and $V_{\text{targ}}$
scale together:

$$\mathcal L_V = \big(V_\theta - V_{\text{targ}}\big)^2 \;\mapsto\; \frac{1}{c^2}\,\mathcal L_V .$$

**Therefore reward scaling changes the *ratio* of the two loss terms by a factor $1/c$**, in addition
to changing each one's absolute magnitude. This is the formal sense in which reward scaling is *not*
equivalent to advantage normalization: advantage normalization touches $\mathcal L_\pi$ only and
leaves $\mathcal L_V$ untouched, whereas reward scaling moves both, and by different powers.

Two scope conditions on that observation, both from this paper's own setup:

- In **Engstrom's** experiments policy and value are **separate networks with separate Adam learning rates** (Tables 4–6). A change in the ratio of two losses that never enter the same parameter's gradient is therefore *not* the channel through which their reward scaling can act. Whatever makes `norm_rewards = returns` matter in Figure 1 must operate through per-network effects — the scale of the value regression target relative to a fixed initialization, the scale of the advantage relative to Adam's $\epsilon$ and to the entropy/gradient-clipping thresholds, and the interaction with the fixed reward-clipping range $[-10,10]$, which becomes a *different* clipping operation once rewards have been divided by $c$.
- The scale $c$ is **not** frozen in reality: it is a running statistic over an accumulator that grows as the agent improves. So the transformation above is applied with a slowly drifting $c$, which is a non-stationarity of exactly the kind PopArt was later designed to absorb. The paper does not discuss this.

**Neither this derivation nor the paper supports any claim about matched critic/advantage scale in a
shared trunk.** Engstrom et al. never ran a shared trunk. The derivation says only what happens to
two loss *functions*; whether that matters depends on an architecture this paper did not test.

### 5.6 A terminological trap between the two ablation papers

Andrychowicz et al.'s footnote on their C66 result reads: "The disabling of the value clipping could
also explain why our conclusions are different from **[Engstrom et al.]** where **a form of value
normalization improved the performance on Walker**."

That sentence identifies Engstrom's **reward scaling** (`norm_rewards = returns`) as "a form of value
normalization" — reasonably, since dividing rewards by the running standard deviation of a discounted
accumulator *is* a way of controlling the scale of the value function. But it means the two papers use
overlapping words for **three genuinely different operations**, and a citation that does not
disambiguate them will be wrong:

| Operation | Who does it | Statistic | Applied to | Advantage ends up in | Critic target ends up in |
|---|---|---|---|---|---|
| **Reward scaling** | Engstrom §3.2 / Alg. 1; OpenAI Baselines `VecNormalize` | running std of a discounted reward accumulator; **no mean subtraction** | the per-step **reward**, upstream of everything | scaled units | scaled units |
| **Value-function normalization** | Andrychowicz C66 | running mean **and** std of **value targets** | the **critic's regression target**, with **de-normalized read-out** | **raw** units | normalized units *inside the network only* |
| **Per-minibatch advantage normalization** | Andrychowicz C67 | mean and std of the **current minibatch** | the **advantage**, in the policy loss only | standardized units | untouched |

And the substantive disagreement the footnote points at is real: **on Walker2d, Engstrom's Figure 1
shows the reward-scaling intervention helping, while Andrychowicz's Figure 37 shows value-target
normalization hurting by 42%.** These are different interventions, so this is not a strict
contradiction — but anyone tempted to write "the literature says normalizing helps on Walker" or
"…hurts on Walker" is choosing which of two papers to quote, and should say which operation they mean.

### 5.7 Relevance and transfer to this project

| Question this project asked | What Engstrom et al. contribute |
|---|---|
| Advantage normalization | **Nothing. The paper is silent.** Do not cite it in either direction. |
| Reward / return / value normalization | Reward scaling (running std of a discounted accumulator, applied to rewards, no mean subtraction) was **necessary for PPO's best scores** in a $2^4$ ablation with the learning rate re-tuned per cell, on Humanoid-v2 and Walker2d-v2. No effect size is reported; the evidence is a distribution shift in Figure 1. They do **not** normalize the critic's regression target, and nothing here is per-batch. |
| Shared vs separate networks | **Silent.** Their setup is separate networks with separate learning rates throughout; the alternative is never run. |
| Policy last-layer initialization | **Partially relevant, and often over-cited.** They ablate `initialization ∈ {orthogonal, xavier}` as a *scheme*, including "scaling that varies from layer to layer" as part of the package, and find it necessary for top scores. They do **not** isolate the last-layer scale, do not sweep it, and report no numbers. For the last-layer claim specifically, Andrychowicz's C57 (§4.2.4) is the citation; this paper corroborates only the weaker "initialization matters". |
| Scope | Three MuJoCo continuous-control tasks; `[64,64]` feed-forward MLPs; Gaussian policies; no recurrence; no partial observability; no discrete actions. |

**Transfer assessment for this project's setting** (discrete actions, GRU, shared trunk, homeostatic
reward, survival-step evaluation):

- **Likely to transfer:** the *methodological* lesson (a comparison between two agents that differ in implementation details is not a comparison of algorithms); the finding that PPO-style **value-loss clipping is not carrying its weight**, which agrees with Andrychowicz's stronger C13 result.
- **Transfers only with the mechanism unstated:** reward scaling. The reason it helps is not established by the paper, and every candidate mechanism named in §5.5.4 depends on quantities this project sets differently — reward magnitude, clipping ranges, whether the trunk is shared. A homeostatic reward whose scale is bounded by construction is a different regime from a MuJoCo locomotion reward that grows without bound as the agent improves, and the running-std divisor is precisely a device for coping with the latter.
- **Does not transfer:** the trust-region measurements are about a Gaussian policy's KL in a continuous action space. The qualitative point (clipping does not bound the realized step) is a property of the objective and should hold for a categorical policy too, but the *numbers* (max ratio 17.5, 30; KL against a 0.07 bound) are setting-specific.
- **Cannot be cited at our architecture at all:** anything about the interaction of policy and value losses. Their networks never share a parameter.

### 5.8 Appendix: Section-by-Section Backbone

| § | Content |
|---|---|
| **Abstract** | Case study of PPO and TRPO; investigates "code-level optimizations: algorithm augmentations found only in implementations or described as auxiliary details". Two claims: they (a) "are responsible for most of PPO's gain in cumulative reward over TRPO" and (b) "fundamentally change how RL methods function". |
| **1 Introduction** | Deep RL is brittle, hard to reproduce, sometimes beaten by random search. Motivating question: how do the many mechanisms in deep RL training algorithms affect agent behaviour? Contribution: analyse behaviour both by cumulative reward and by finer-grained algorithmic properties, via a PPO/TRPO case study. Footnote 1 distinguishes "code-level optimizations" (which change the algorithm's operation) from "implementation choices" like PyTorch vs TensorFlow (which do not). |
| **2 Related work** | REINFORCE → policy gradient theorem; TRPO's lineage through Kakade's natural policy gradient and relative-entropy policy search; Henderson et al. on codebase-dependent results (the observation this paper builds on); Rajeswaran and Mania on random search matching PG methods; Tucker et al. on misattributed gains from action-dependent baselines. |
| **3 Attributing success in PPO** | The nine code-level optimizations, enumerated with equations for value clipping and a prose definition of reward scaling. Figure 1: full $2^4$ ablation of the first four, per-cell learning-rate grid search, 5 seeds, survival-function histograms partitioned by each optimization. Footnote 3 concedes the ablation was limited to four by compute. Defines PPO-M (PPO with none of the optimizations) and Table 1 (the five algorithms studied and their properties). |
| **4 Code-level optimizations have algorithmic effects** | Trust-region background; TRPO's constrained objective (Eq. 1); PPO's clipped objective (Eqs. 2–3). Analytic gradient of the PPO objective and the argument that the first step of each phase is unconstrained and the realized step size is set by the optimizer, not the objective. Figure 2 (Humanoid) and Figure 3 (Walker2d, Hopper): reward, max ratio, mean KL over training for TRPO/PPO/PPO-M. Findings: all three violate the ratio trust region; all three respect a mean-KL trust region; PPO and PPO-M differ in the *shape* of their KL trajectory despite sharing a core objective. Figure 4 repeats on held-out state-action pairs with no qualitative change. |
| **5 Identifying roots of algorithmic progress** | Constructs TRPO+ (TRPO plus the code-level optimizations, plus a KL-decay schedule as the analogue of LR annealing) so that all four combinations of {PPO, TRPO} × {with, without optimizations} exist. Table 2 with 95% bootstrap CIs, ≥80 agents/cell; defines AAI and ACLI; code-level effects dominate algorithmic effects on all three tasks. Then Table 3: PPO-NOCLIP beats PPO-M everywhere and PPO on Humanoid; "the clipping mechanism is not necessary to achieve high performance", with footnote 6 conceding PPO-NOCLIP is a strict subset of PPO's configuration space. |
| **6 Conclusion** | Code-level optimizations have a drastic effect on performance and change algorithm operation "in ways unpredicted by the conceptual policy gradient framework", especially the nature of the enforced trust region. Calls for modular design and for moving beyond benchmark-driven evaluation. |
| **App. A.1 Experimental setup** | All hyperparameters from grid searches; PPO's optimization settings taken from OpenAI Baselines; per-algorithm grids described. Tables 4–6 (Walker2d, Humanoid, Hopper): full hyperparameters per algorithm, including separate policy/value networks (`[64,64]` each), separate learning rates, `Reward normalization ∈ {returns, rewards, none}`, reward and state clipping ranges, gradient clipping, and `PPO Clipping ε = 1e+32` for PPO-NOCLIP. All error bars are bootstrapped 95% CIs. |
| **App. A.2** | Algorithm 1, the reward-scaling procedure (five lines, transcribed in §5.3). |
| **App. A.3** | Figure 3 (Walker2d, Hopper training-set trust-region measurements) and Figure 4 (held-out measurements). |

## 6. Huang, Dossa, Raffin, Kanervisto & Wang (2022) — *The 37 Implementation Details of Proximal Policy Optimization*

### 6.1 Fixed extraction block

| Field | Value |
|---|---|
| **Venue** | ICLR Blog Track, 25 March 2022. The post's own BibTeX gives `booktitle = {ICLR Blog Track}, year = {2022}`. |
| **PDF** | `docs/project/references/ppo_implementation_details/sources/Huang et al. 2022 - The 37 implementation details of PPO (ICLR Blog Track).pdf` (68 pp., rendered from HTML by headless Chrome — page breaks are artificial; see §2.3) |
| **What it is, in the authors' words** | "Instead of doing ablation studies and making recommendations on which details matter, this blog post takes a step back and focuses on **reproductions** of PPO's results in all accounts." And, from the conclusion: "Instead of introducing additional improvements or doing further ablation studies, this blog post … focuses on delivering a thorough reproduction of PPO in all accounts, as well as aggregating, documenting, and cataloging its most salient implementation details." |
| **Reference implementation studied** | `openai/baselines` `ppo2` at commit `ea25b9e` (31 Jan 2020), chosen after a genealogy analysis of the repository's revisions, because it scores well on both Atari and MuJoCo and supports LSTM and `MultiDiscrete` action spaces. |
| **The 37** | 13 core details + 9 Atari-specific + 9 for continuous-action robotics + 5 LSTM + 1 `MultiDiscrete`. Plus 4 "auxiliary"/situational details not in the reference implementation. |
| **Domains actually run** | Classic control (**CartPole-v1, Acrobot-v1, MountainCar-v0** — discrete), Atari (Breakout, Pong, BeamRider — discrete, visual), MuJoCo (continuous), Gym-μRTS (`MultiDiscrete`), Procgen, and Atari + LSTM. **The only paper in this corpus that runs anything other than MuJoCo.** |
| **Protocol** | "For each category (except the first one), we benchmark our implementation against the original implementation in three environments, each with three random seeds." Every experiment is a *reproduction* comparison — their PyTorch re-implementation vs `openai/baselines` — not a controlled ablation. |
| **Evidential weight** | **High** for "what does the reference implementation actually do, and at which source line". **Low** for "which choices matter", because it runs no ablations; where it makes such claims it cites Engstrom or Andrychowicz. **One exception**, §6.3 below, where it does run its own comparison. |

### 6.2 The catalogue, restricted to the details this project asked about

Each entry gives the detail as Huang et al. state it, plus what they cite for it. Where their citation
is inaccurate, this review says so.

| # | Detail | What the reference implementation does | Their cited evidence |
|---|---|---|---|
| Core 2 | **Orthogonal initialization of weights, constant initialization of biases** | Hidden layers: orthogonal with scale $\sqrt2$, biases 0. **"However, the policy output layer weights are initialized with the scale of `0.01`. The value output layer weights are initialized with the scale of `1`."** | Engstrom et al. (orthogonal beats Xavier); Andrychowicz et al. C57 ("centering the action distribution around 0"). Their code pointer: `common/policies.py#L49-L63`. |
| Core 5 | **Generalized advantage estimation** | GAE, with value bootstrapping for non-terminated sub-environments. **"PPO implements the return target as `returns = advantages + values`, which corresponds to TD($\lambda$) for value estimation (where Monte Carlo estimation is a special case when $\lambda = 1$)."** They also flag, as a known wart, that the implementation does *not* bootstrap the value of time-limit-truncated episodes and that it arguably should — "for high-fidelity reproduction, we did not implement the correct handling for truncated environments". | Andrychowicz C6 (GAE beats N-step). |
| Core 6 | **Minibatch updates** | Shuffles the indices of the $N\times M$ training batch and splits into minibatches. Names two common mis-implementations: using the whole batch, and sampling minibatches with replacement. | — |
| Core 7 | **Normalization of advantages** | "After calculating the advantages based on GAE, PPO normalizes the advantages by subtracting their mean and dividing them by their standard deviation. In particular, **this normalization happens at the minibatch level instead of the whole batch level!**" | Andrychowicz C67 — reported by Huang as "find per-minibatch advantage normalization to not affect performance much (figure 35)". **No ablation of their own.** |
| Core 9 | **Value function loss clipping** | Implemented, with the target defined as `V_targ = returns = advantages + values`. | Engstrom ("no evidence that value function loss clipping helps"), Andrychowicz C13 ("even hurts"). "We implemented this detail because this work is more about high-fidelity reproduction of prior results." |
| Core 10 | **Overall loss and entropy bonus** | `loss = policy_loss - entropy * entropy_coefficient + value_loss * value_coefficient`. **"Note that the policy parameters and value parameters share the same optimizer."** | Mnih et al. for the entropy bonus. **Citation error**: they write "Andrychowicz, et al. (2021) overall find no evidence that the entropy term improves performance on continuous control environments (**decision C13**, figure 76 and 77)". Figures 76–77 are correct, but C13 is PPO-style *value clipping*; the regularization choice is **C32** (with C40/C46 for entropy specifically). |
| Core 11 | **Global gradient clipping** | Rescales so the global $\ell_2$ norm of all concatenated gradients does not exceed 0.5. | Andrychowicz C68 (small boost). |
| Core 13 | **Shared and separate MLP networks** | See §6.3 — this is the one place they run their own comparison. | — |
| Atari 8 | **Shared Nature-CNN for policy and value** | For Atari, the reference implementation **shares** the convolutional trunk and attaches a policy head (init scale 0.01) and a value head (init scale 1). Huang et al. keep it, noting only that "such a parameter-sharing paradigm obviously computes faster when compared to setting completely separate networks", and that "recent work suggests balancing the competing policy and value objective could be problematic, which is what methods like Phasic Policy Gradient are trying to address (Cobbe et al., 2021)". | — |
| Cont. 4 | **Separate MLP networks for policy and value** | For continuous control the reference implementation uses `value_network='copy'`, i.e. two separate `[64,64]` tanh MLPs, with the value head init scale 1.0 and the policy-mean head init scale 0.01. | Andrychowicz C47. |
| Cont. 6–7 | **Observation normalization and clipping** | `VecNormalize` subtracts a running mean and divides by (the text says) the variance; then clips to $[-10,10]$. | Andrychowicz C64 (helpful), C65 (clipping did not help). |
| Cont. 8 | **Reward scaling** | "The `VecNormalize` also applies a certain discount-based scaling scheme, where the rewards are divided by the standard deviation of a rolling discounted sum of the rewards (without subtracting and re-adding the mean)." Identical to Engstrom's Algorithm 1. | "Engstrom, Ilyas, et al., (2020) reported that reward scaling can significantly affect the performance of the algorithm and **recommends the use of reward scaling**." **Paraphrase drift**: Engstrom et al. make no recommendations section and never write "we recommend"; their claim is that reward normalization "significantly impact[s] the rewards landscape … and w[as] necessary for attaining the highest PPO reward within the tested hyperparameter grid". |
| Cont. 9 | **Reward clipping** | The scaled reward is further clipped to $[-10,10]$. | "There is currently **no clear evidence** that Reward Clipping after Reward Scaling can help with learning." |
| LSTM 1–5 | **Recurrent PPO** | (1) LSTM layer weights initialized with `std=1`, biases 0. (2) Hidden and cell states initialized to zeros. (3) States reset to zero at end of episode, via an end-of-episode flag passed to the agent during **both** rollout and training. (4) **"Under the non-LSTM setting, the mini-batches fetch randomly-indexed training data because the ordering of the training data doesn't matter. However, the ordering of the training data does matter in the LSTM setting. As a result, the mini-batches fetch the sequential training data from sub-environments."** (5) The initial LSTM state is saved before the rollout and the states are sequentially reconstructed during training, "ensur[ing] that we reconstructed the probability distributions used in rollouts". | — (no external citation; this is documentation of `a2c/utils.py`). |
| Aux 3 | **Early stopping of policy optimization on KL** | Tracks approximate mean KL between pre- and post-update policy; stops updating when it exceeds a threshold. From `spinningup`/`modular_rl`, not from `baselines`. Included in their code as `--target-kl 0.01`, off by default. **"Note, however, that while openai/spinningup only early stops the updates to the policy, our implementation early stops both the policy and the value network updates."** | Dossa et al. |

**Two of these bear directly on this project's normalization dispute and are worth isolating:**

- **Core 5** states the reference implementation's critic target explicitly: `returns = advantages + values`, left in raw reward units, and identifies **Monte-Carlo return as the $\lambda = 1$ special case of the same construction**. Nothing anywhere in the post normalizes that target.
- **Core 7** states that advantage normalization is applied **per minibatch, not per batch**, with the exclamation mark in the original. This is a documentation claim about `ppo2/model.py#L139`, not a claim about whether it helps.

### 6.3 The one comparison they run themselves — shared vs. separate networks, on discrete actions

This is the only place in the corpus where shared-vs-separate is compared **outside MuJoCo**, and it is
therefore the single most relevant experiment in these three works to this project's architecture.

**Setup.** They build `ppo.py` (322 lines, separate networks) and `ppo_shared.py` (317 lines, shared
trunk with two heads) differing by about ten lines, and run both against `openai/baselines`' shared
and separate variants — **four curves** — on **CartPole-v1, Acrobot-v1 and MountainCar-v0**, three
random seeds each, roughly 500k steps.

**Result, verbatim:** "While shared-network architecture is the default setting in PPO, the
**separate-networks architecture clearly outperforms in simpler environments**. The shared-network
architecture performs worse **probably due to the competing objectives of the policy and value
functions**. For this reason, we implement the separate-networks architecture in the video tutorial."

**The curves, read off the rendered figures** (episodic return vs. environment steps; shaded bands are
across-seed spread):

| Environment | separate (theirs / baselines) at ≈500k | shared (theirs / baselines) at ≈500k | Comment |
|---|---|---|---|
| CartPole-v1 | ≈490 / ≈500 (both at the 500-step cap) | ≈275 / ≈210 | Separate is roughly **2×** shared and reaches the cap by ≈250k steps while shared is still climbing at 500k. |
| Acrobot-v1 | ≈−85 / ≈−85 (both converged by ≈150k) | ≈−380 / ≈−360 | Very large gap; the shared arms have enormous seed bands (some seeds sit at the −500 floor for most of training). |
| MountainCar-v0 | ≈−200 | ≈−200 | **Uninformative** — all four configurations sit flat at the failure return for the whole run. |

**How much weight this carries.** More than one might expect, and less than it looks.

*In its favour:* the action space is **discrete**; the comparison is direct (two implementations,
same hyperparameters, both arms); it replicates across two independent codebases (theirs and
`openai/baselines`), which rules out a bug in either; and the effect size on Acrobot is far larger
than the seed spread of the separate arms.

*Against:* three seeds; three environments of which one is uninformative; both classic-control tasks
are trivially easy, which is the regime where a shared trunk has least to gain from sharing; **no
hyperparameter re-tuning per arm**, so the shared arm runs with a value-loss coefficient it never had
a chance to tune (contrast Engstrom, who re-grids the learning rate inside every ablation cell); and
the numbers behind the curves live in a Weights & Biases panel that the PDF cannot render, so nothing
more precise than the plot is available here.

*On the mechanism:* "probably due to the competing objectives of the policy and value functions" is
offered as a **conjecture**, in one sentence, with no measurement of gradient magnitudes, loss ratios
or feature usage. It is the same conjecture this project has been invoking, stated by a credible
practitioner, and it is **not evidence for it**. The pointer they give — Phasic Policy Gradient
(Cobbe et al., 2021), which separates policy and value optimization into alternating phases — is the
place where that mechanism is actually studied, and PPG is **not in this corpus**.

**The counterweight in the same post, which must be quoted alongside it.** For **Atari**, the same
authors keep the **shared** Nature-CNN trunk, reproduce `openai/baselines`' published Atari scores
with it — their own re-implementation's Atari curves track `openai/baselines`' `ppo2` (`ea25b9e`),
which they themselves measure at Breakout 409.265 ± 30.98 — and offer no criticism of sharing beyond
"balancing the competing objective could be problematic". And their **recurrent** implementation,
`ppo_atari_lstm.py`, is built by modifying `ppo_atari.py` — so **the only recurrent PPO anywhere in
this corpus runs on a shared trunk**, and reproduces the reference implementation's LSTM results with
it (their Fig.: PPO-LSTM on Breakout and Pong, their curves tracking `openai/baselines`').

The honest summary is therefore: **the field's own convention is architecture-conditional** — separate
networks where the trunk is a cheap two-layer MLP over a low-dimensional observation, shared where the
trunk is an expensive feature extractor (a CNN over pixels; a recurrent core). Huang et al. follow both
conventions in the same post without treating them as contradictory.

### 6.4 Reproduction fidelity — what this post is genuinely authoritative about

The genealogy section is the post's underrated contribution. `openai/baselines`' PPO has four
milestone revisions (`da99706` `pposgd`/`ppo1` on 2017-07-20; `2dd7d30` introducing `ppo2` on
2017-11-16; `ea68f3b` producing the MuJoCo benchmark on 2018-08-10; `7bfbcf1` producing the Atari
benchmark on 2018-10-04; `ea25b9e`, the final merge, on 2020-01-31), and **they do not all score the
same**. Their table of best-reported Atari scores across libraries:

| Source | Breakout | Pong | BeamRider |
|---|---|---|---|
| Baselines `ppo1` (`da99706`), from the PPO paper | 274.8 | 20.7 | 1590 |
| Baselines `ppo2` (`7bfbcf1`/`ea68f3b`), from the docs | 114.26 | 13.68 | 1299.25 |
| Baselines `ppo2` (`ea25b9e`), measured in this post | **409.265 ± 30.98** | 20.59 ± 0.40 | 2627.96 ± 625.75 |
| Stable-Baselines3 | 398.03 ± 33.28 | 20.98 ± 0.10 | 3397.00 ± 1662.36 |
| CleanRL | ~402 | ~20.39 | ~2131 |
| Tianshou | ~400 | ~20 | — |
| Ray/RLlib | 201 | — | 4480 |

Their observations: the revisions "are not without performance consequences … even the original
implementation could produce inconsistent results"; libraries that match `ea25b9e`'s details report
similar results while others diverge; and many libraries report MuJoCo but not Atari. This is the
strongest available evidence that **"PPO" names a family of implementations rather than an
algorithm** — which is the same conclusion Engstrom et al. reach by a different route.

Their debugging recommendations are worth recording because they are cheap and checkable in any
implementation: seed everything and diff against a reference; **check that the probability ratio is
exactly 1 on the first epoch's first minibatch** (if not, the training-time forward pass has not
reconstructed the rollout-time distribution — the single most common recurrent-PPO bug); watch that
approximate KL stays below ≈0.02; compare policy- and value-loss curves against a reference, not just
returns; and use "400 episodic return on Breakout" as a fidelity smoke test.

### 6.5 Phase 1 — Foundational overview

**The problem.** PPO is described in its paper in about a page of equations. The code that actually
produces the published scores contains dozens of extra steps that are not in that page. Two earlier
papers (Engstrom's and Andrychowicz's, both reviewed above) had shown that those extra steps matter,
but neither showed *how they fit together*, and both studied only robot-locomotion tasks. So a
practitioner who wants to reproduce PPO's Atari scores, or to add a memory (recurrent) network, still
has nowhere to look.

**What they did.** They picked one specific snapshot of the official code as *the* reference, read it
line by line, and wrote out every implementation decision it makes — 37 of them, each with a
permanent link to the exact source line. Then they re-implemented PPO from scratch in PyTorch as a set
of self-contained single files (one for classic control, one for Atari, one for continuous control,
one for recurrent networks, one for compound action spaces) and showed that each one reproduces the
official implementation's learning curves on three environments with three seeds.

**Why it is useful and where its limits are.** It is the best available answer to "what does PPO
actually do, and where is that in the code". It is *not* an answer to "which of these things
matter" — the authors say so themselves in the opening and again in the conclusion, and where they
comment on importance they are quoting the other two papers. There is one exception: they compare a
shared network against two separate networks on three simple control tasks and find separate clearly
better, offering "competing objectives between the policy and the value function" as a guess at why.

**Initial takeaway.** Treat this as the field's reference card, not as evidence. Its most valuable
content for a project outside MuJoCo is the material the two ablation papers do not have at all:
what changes when the action space is discrete or compound, and the five specific things that must be
right for PPO with a recurrent network — reset the memory at episode boundaries, feed minibatches as
ordered sequences rather than shuffled transitions, and rebuild the memory states at training time so
that the probabilities computed during learning match the ones used when acting.

### 6.6 Phase 2 — Graduate-level deep dive

#### 6.6.1 The value-loss clipping equation — a discrepancy between two papers in this corpus

Both Engstrom et al. and Huang et al. transcribe the same operation from the same source file
(`ppo2/model.py#L68-L75`), and **they disagree on the outer operator.**

Engstrom et al. §3, item 1, print

$$\mathcal L^V = \min\Big[(V_{\theta_t} - V_{\text{targ}})^2,\; \big(\operatorname{clip}(V_{\theta_t}, V_{\theta_{t-1}} - \varepsilon, V_{\theta_{t-1}} + \varepsilon) - V_{\text{targ}}\big)^2\Big].$$

Huang et al., core detail 9, print

$$\mathcal L^V = \max\Big[(V_{\theta_t} - V_{\text{targ}})^2,\; \big(\operatorname{clip}(V_{\theta_t}, V_{\theta_{t-1}} - \varepsilon, V_{\theta_{t-1}} + \varepsilon) - V_{\text{targ}}\big)^2\Big]$$

— though in the PDF rendering this equation **overflows its column and is cut off** at
"$-\,V_{\text{ta}}\ldots$", which is a defect of the HTML-to-PDF conversion, not of the post.

**Which is right?** The third paper in this corpus adjudicates it on wording: Andrychowicz et al.
(App. B.2) describe the same mechanism as "an additional **pessimistic** clipping in the value loss
function". A pessimistic combination of two candidate losses takes the **larger** one, i.e. $\max$.
Under $\min$ the operation would be optimistic and would weaken rather than constrain the value
update, which is not what any of the three papers describe it as doing. **Read Engstrom's `min` as a
typographical error and Huang's `max` as the correct transcription.** This matters if anyone
implements from the equation rather than the code: the two differ in sign of effect, not in
magnitude.

#### 6.6.2 Why recurrence changes the minibatch story — reading detail 5 against LSTM detail 4

The corpus contains a genuine, unremarked tension that only appears once all three works are on the
table.

Andrychowicz et al. §3.5 recommend `batch_mode = Shuffle transitions (recompute advantages)`:
break the rollout into individual transitions, assign them randomly to minibatches for maximum
within-batch diversity, and recompute advantages at the start of each epoch to avoid staleness. This
is their best-performing variant.

Huang et al.'s **LSTM detail 4** says the opposite is mandatory once the network is recurrent: "the
ordering of the training data does matter in the LSTM setting. As a result, the mini-batches fetch the
**sequential** training data from sub-environments."

There is no contradiction — Andrychowicz never ran a recurrent network — but the practical
consequence is that **Andrychowicz's minibatch recommendation does not apply to a recurrent agent**,
and the reason is structural rather than empirical. Write the recurrent policy as
$\pi_\theta(a_t | o_{\le t}) = \pi_\theta(a_t \mid h_t)$ with $h_t = f_\theta(h_{t-1}, o_t)$. Then
the training-time probability of a stored action is not a function of the stored observation alone; it
is a function of the entire prefix through $h_t$. Shuffling transitions destroys the prefix, so
$h_t$ cannot be reconstructed, so the ratio

$$\rho_t = \frac{\pi_\theta(a_t \mid h_t)}{\mu(a_t \mid h_t^{\text{rollout}})}$$

is not computable — and in particular is not equal to 1 at the first gradient step of an epoch, which
is the invariant Huang et al.'s debugging recommendation 2 tells you to check. Their LSTM details 2,
3 and 5 exist precisely to make $h_t^{\text{rollout}}$ reconstructible: save the state at the start of
the rollout, replay forward in order, and zero it at episode boundaries.

This is the one place where the corpus speaks *directly* to a recurrent agent, and what it says is
about correctness, not about performance.

#### 6.6.3 The overall loss, and where a shared trunk actually appears in it

Huang et al.'s core detail 10 states the objective as implemented:

$$\mathcal L = \mathcal L_{\text{policy}} - c_H\,\mathcal H[\pi] + c_V\,\mathcal L_{\text{value}},$$

with the note "the policy parameters and value parameters **share the same optimizer**". Two
observations follow, and it is worth being precise about which is architectural and which is not:

- **Sharing an optimizer is not sharing a trunk.** Adam maintains per-parameter first and second moments, so a policy parameter and a value parameter in *separate* networks receive updates that are independent of each other's gradient magnitudes, whatever $c_V$ is. In the separate-network configuration, $c_V$ therefore acts almost purely as a (per-parameter-normalized) rescaling of the value network's effective step size.
- **Sharing a trunk is different in kind.** For a trunk parameter $\psi$, the gradient is $\nabla_\psi \mathcal L_{\text{policy}} - c_H \nabla_\psi \mathcal H + c_V \nabla_\psi \mathcal L_{\text{value}}$ — a *sum computed before* Adam sees it. Adam normalizes the sum, not the parts, so the ratio of the terms in that sum is a real degree of freedom that has no analogue in the separate configuration.

That distinction is what makes "the value loss dominates the shared trunk" a coherent hypothesis
rather than a category error. **It is also, in this corpus, an entirely untested one:** Huang et al.
do not vary $c_V$; Andrychowicz et al. sample $c_V$ (their C54) and then discard the arm before
analysing it (§4.5.4); Engstrom et al. never share parameters at all. The only pointer any of the
three gives is Huang et al.'s citation of Phasic Policy Gradient, which is outside this corpus.

#### 6.6.4 Reward scaling in a discrete, non-MuJoCo domain

One small piece of evidence in the post extends Engstrom's reward-scaling result beyond continuous
control: their **Procgen** configuration uses `VecNormalize(venv=env, ob=False)` — i.e. reward
normalization on, observation normalization off — and they note that in this configuration
"1. Learning rate annealing is turned off by default. 2. Reward scaling and reward clipping is used."
Procgen is a discrete-action, image-observation, procedurally generated benchmark. So the reference
configuration for a discrete-action domain **does** use the running-return-std reward scaling that
Engstrom et al. found necessary in MuJoCo.

This is a **usage** fact, not an effect-size measurement: Huang et al. run no ablation of it on
Procgen and report only that their reproduction matches the reference. It removes the objection
"reward scaling is a MuJoCo-only convention"; it does not establish that it helps in a discrete
domain, and it says nothing about a homeostatic reward.

### 6.7 Relevance and transfer to this project

| Question this project asked | What Huang et al. contribute |
|---|---|
| Advantage normalization | **Documentation only.** They record that the reference implementation standardizes advantages **per minibatch** (not per whole batch) at `ppo2/model.py#L139`, and they cite Andrychowicz's C67 for whether it helps. **No independent test.** |
| Reward / return / value normalization | **Documentation, plus one scope extension.** Reward scaling defined identically to Engstrom's Algorithm 1; the critic target defined as `returns = advantages + values` in **raw** units, with MC named as the $\lambda=1$ case; reward scaling shown to be part of the reference Procgen (discrete, visual) configuration. **Nobody here normalizes the critic's regression target, per batch or otherwise.** |
| Shared vs separate networks | **The corpus's only discrete-action comparison** (§6.3): separate clearly better on CartPole-v1 and Acrobot-v1, uninformative on MountainCar-v0, 3 seeds, no per-arm tuning, mechanism offered as a one-sentence conjecture. **Counterweighted by their own practice**: they keep a shared CNN trunk for Atari and build their recurrent PPO on top of it. |
| Policy last-layer initialization | **Corroborates the convention.** The reference implementation initializes the **policy** output layer at scale 0.01 and the **value** output layer at scale 1.0 — exactly Andrychowicz's C57/C58 defaults — and Huang et al. apply it to a **`Categorical`** policy head (`layer_init(Linear(64, n_actions), std=0.01)`), which is the discrete-action translation this project needs and which Andrychowicz never ran. Note this is a *usage* corroboration, not an effect measurement. |
| Recurrence | **The only source in the corpus.** Five concrete requirements (§6.2, LSTM 1–5), of which three are correctness conditions rather than tuning choices, plus the "ratio must equal 1 on the first minibatch of the first epoch" self-check. Worth reading against this project's recurrent trainer. |
| Scope | Broadest of the three: discrete classic control, discrete visual (Atari, Procgen), continuous MuJoCo, compound discrete (Gym-μRTS), and recurrent. Still no partially-observed continuous-state task with a homeostatic reward, and still no ablations. |

**How to cite it.** For "the reference PPO implementation does X, at this source line" — cite freely;
it is the best-sourced document of its kind and was peer reviewed. For "X matters" — do **not** cite
it; cite whichever of Engstrom or Andrychowicz it is itself citing, and check the citation, because
two of the ones checked in this review were inaccurate (the C13-for-C32 mix-up in core detail 10, and
the "recommends the use of reward scaling" paraphrase of Engstrom in continuous detail 8).

### 6.8 Appendix: Section-by-Section Backbone

Section order as rendered in the PDF.

| § | Content |
|---|---|
| **Opening narrative** | A framing dialogue between "Jon" (a first-year master's student) and "Sam". Establishes the gap: the two ablation papers explain *which* details matter but not *how they are coded*, "their conclusions are in MuJoCo tasks and do not necessarily transfer to other games such as Atari", and neither covers PPO + LSTM or `MultiDiscrete` action spaces. States the post's stance: reproduction, not ablation. Lists four contributions: genealogy analysis, video tutorials + single-file implementations, a 37-item checklist with permanent source links, high-fidelity reproduction, plus 4 situational details. |
| **Background** | PPO as a refinement of TRPO. The revision history of `openai/baselines` PPO (`pposgd`/`ppo1` → `ppo2`, with four milestone commits). Table of best-reported Atari scores across nine libraries, with per-library footnotes on step budgets, MuJoCo version and seed counts. Three observations: revisions changed performance; libraries matching `ea25b9e` agree with each other; most libraries report MuJoCo but not Atari. Declares `ppo2 (ea25b9e)` the reference. |
| **13 core implementation details** | (1) vectorized architecture with pseudocode, including its use for two-player self-play; (2) orthogonal init $\sqrt2$ with policy head 0.01 and value head 1.0; (3) Adam $\epsilon = 10^{-5}$ (vs PyTorch $10^{-8}$, TensorFlow $10^{-7}$); (4) Adam LR annealing (2.5e-4→0 Atari, 3e-4→0 MuJoCo); (5) GAE, value bootstrapping, the truncation wart, `returns = advantages + values`; (6) minibatch updates and two common mis-implementations; (7) per-minibatch advantage normalization; (8) the clipped surrogate objective; (9) value-function loss clipping; (10) overall loss, entropy bonus, shared optimizer; (11) global gradient clipping at 0.5; (12) five debug variables incl. `clipfrac` and two `approxkl` estimators; (13) shared vs separate MLPs, with the classic-control comparison. |
| **9 Atari-specific details** | `NoopResetEnv`, `MaxAndSkipEnv`, `EpisodicLifeEnv`, `FireResetEnv`, `WarpFrame`, `ClipRewardEnv`, `FrameStack`, **shared Nature-CNN trunk with policy/value heads**, image scaling to [0,1]. Benchmarked on three Atari games against the reference. |
| **9 continuous-action details** | Normal action distribution; state-independent log-std initialized to 0; independent action components; **separate** policy/value MLPs (`value_network='copy'`); action clipping with the unclipped action stored; observation normalization; observation clipping; **reward scaling**; reward clipping. Lists the reference MuJoCo hyperparameters verbatim. |
| **5 LSTM details** | Layer init `std=1`, biases 0; zero-initialized hidden/cell states; reset at episode end; **sequential** minibatches; reconstruction of LSTM states during training from a saved `initial_lstm_state`. Benchmarked on Breakout and Pong with frame stacking removed. |
| **1 `MultiDiscrete` detail** | Factorized action spaces treated as probabilistically independent components; cites AlphaStar and OpenAI Five. Benchmarked on Gym-μRTS. |
| **4 auxiliary details** | Clip-range annealing; parallelized gradient update (`ppo1` only); KL-based early stopping (from `spinningup`, off by default, and applied to both policy *and* value in their version); invalid action masking (logits set to $-\infty$ before softmax, which zeroes the corresponding gradients). Invalid action masking is benchmarked on Gym-μRTS. |
| **Results** | "our implementations match the results of the original implementation closely", extending to policy and value losses, with an interactive comparison. |
| **Recommendations** | Five debugging techniques (seed everything and diff; check ratio = 1; watch approximate KL below ≈0.02; compare loss curves not just returns; "400 on Breakout" as a fidelity rule of thumb) and four reproducibility recommendations (enumerate the implementation details you used; release locked source; track experiments; adopt single-file implementations). |
| **Discussions** | "Does modularity help RL libraries?" — modularity disperses implementation details across files and makes them hard to see. Then accelerated vectorized environments (Envpool, Procgen's C++ vec envs, Isaac Gym, Brax), with an Atari + Envpool reproduction and a "Pong in 5 minutes on 24 CPUs and an RTX 2060" result compared against IMPALA-style distributed setups. |
| **Request for Research** | Three directions: alternative choices (different Atari preprocessing; Beta / squashed-Gaussian / full-covariance action distributions; state-dependent std; **different LSTM initializations, and GRU cells instead of LSTM**); vectorized architectures for replay-based methods; value-function optimization (citing Phasic Policy Gradient's separate value optimization and prioritized replay). |
| **Conclusion** | Reproduction, not ablation; prior works "are not structured as tutorials and only focus on details concerning robotics tasks". |
| **Bibliography** | Notably cites Engstrom et al. correctly as **ICLR 2020** (see §2.2), and cites Andrychowicz et al. by the **preprint title** ("What matters in on-policy reinforcement learning? a large-scale empirical study") with the **camera-ready venue** (ICLR 2021) — a hybrid citation that is itself an artefact of the retitling described in §2.1. |

## 7. Evidence ledger — the five questions this corpus was read to answer

Each question gets: the answer, the citation, and — where the corpus does not answer it — an explicit
statement of silence. "Silent" below means *the experiment does not exist in the paper*, not *the
paper found no effect*. The two are routinely confused and the distinction is the whole point of this
section.

### Q1 — Advantage normalization: what did Andrychowicz et al. actually measure?

**Measured object (their choice C67, defined App. B.9):** the advantages within **each minibatch** are
shifted to zero mean and divided by their standard deviation, **for the policy loss only**. The
critic's target is untouched. Sampled over $\{$False, True$\}$ in a 2000-configuration random search
per environment (App. F.1), 3 seeds each, alongside PPO $\epsilon$, observation normalization and
clipping, gradient clipping, Adam learning rate, and value-function normalization.

**Their exact finding (§3.3):** *"In contrast to observation and value function normalization,
per-minibatch advantage normalization (C67) **seems not to affect the performance too much**
(Fig. 35)."* It is **absent from the §3.3 recommendation**, and their own base configuration sets
`C67 = False` (App. C, Table 2).

**Correction to the phrasing "no significant effect".** Two problems, detailed with the figure data in
§4.2.1:

- **They ran no significance test on C67.** The paper reports a conditional 95th percentile with a binomial confidence interval and a top-5% share. "No significant effect" attributes a statistical claim to authors who did not make one.
- **"No effect" also overstates the data slightly.** Turning C67 on was ahead on Hopper, Humanoid and Ant and behind on Walker2d and HalfCheetah; every confidence interval overlaps; but pooled over all five environments, `True` accounts for **53.5%** of the top-5% configurations against `False`'s **47%** — a mild tilt in favour of normalizing, driven by Hopper (0.62) and Humanoid (0.58).

**Recommended phrasing:** *"Andrychowicz et al. (2021), the largest such study run, found per-minibatch
advantage normalization to have a small, environment-dependent, sign-inconsistent effect that their
confidence intervals cannot separate from zero. They neither recommend it nor warn against it, and
their own default leaves it off."*

**Does Engstrom test it?** **No — completely silent.** The word "advantage" occurs three times in the
paper and never in connection with standardization (§5.1). Do not cite Engstrom on this in either
direction.

**Does Huang test it?** **No.** He documents it as core detail 7 — including the point that the
reference implementation normalizes **per minibatch rather than per whole batch** — and then cites
Andrychowicz's C67 for whether it matters (§6.2). No ablation.

**One structural point the corpus does not make, derived in §4.5.2 and flagged as reviewer-supplied:**
the two halves of C67 behave differently under PPO's clipped objective. Dividing by the batch standard
deviation commutes with the $\min$ and is therefore exactly a per-minibatch learning-rate rescale
(largely absorbed by Adam). Subtracting the batch mean does **not** commute: it flips the sign of some
advantages, and the clip is asymmetric in that sign, so centring changes which samples get
pessimistically clipped. Figure 35 measures only the pair, so it licenses no claim about either half.

### Q2 — Reward, return and value normalization

**(a) What Engstrom et al. ablate, and how it is separated.** Their "reward scaling" is the OpenAI
Baselines `VecNormalize` mechanism, and Appendix A.2 Algorithm 1 gives it in five lines: maintain a
discounted accumulator $R_t \leftarrow \gamma R_{t-1} + r_t$, add $R_t$ to a running-statistics
object, and return $r_t / \text{std}(RS)$. So: a **running** divisor, taken from the standard deviation
of a **discounted return accumulator**, applied to the **per-step reward**, **with no mean
subtraction**. It sits upstream of the return, the value target and the advantage, which are all
scaled together.

Separation method: a $2^4 = 16$ full factorial over {value clipping, reward scaling, orthogonal
initialization, LR annealing}, with a **learning-rate grid search re-run inside every cell** and the
best LR chosen before agents enter the pool ($5 \times 2^4 = 80$ agents per environment). Figure 1
then plots the survival function of final rewards, partitioned by each optimization, marginalizing
over the other three.

Finding, verbatim: *"reward normalization, Adam annealing, and network initialization each
significantly impact the rewards landscape with respect to hyperparameters, and were necessary for
attaining the highest PPO reward within the tested hyperparameter grid."* **No effect size is
reported** — Figure 1 is a distribution plot, and the paper's one numeric table (Table 2) bundles all
four optimizations into the PPO-vs-PPO-M contrast. See §5.3.

**(b) What Andrychowicz finds on value-function normalization (their C66).** A **running** mean and
standard deviation of **value targets**; the network predicts
$(\hat V - v_\mu)/\max(v_\rho, 10^{-6})$ and its outputs are **de-normalized at read-out**, so the
advantage is always formed in raw reward units. Finding, verbatim: it *"also influences the
performance very strongly — it is crucial for good performance on HalfCheetah and Humanoid, helps
slightly on Hopper and Ant and **significantly hurts the performance on Walker2d**"*, and *"We are not
sure why the value function scale matters that much"*. Their recommendation is conditional: *"check
if value function normalization improves performance."* Effect sizes and the sign flip are tabulated
in §4.2.2 (+78% on Humanoid, −42% on Walker2d; on Walker2d **99%** of top-5% configurations had it
**off**).

**Andrychowicz does not study reward scaling at all** — the strings "reward scaling" and "reward
normalization" do not occur in the paper.

**(c) Does anyone normalize the critic's regression target per batch?** **No. Nobody in this corpus.**
The closest thing is Andrychowicz's C66, and it differs on both counts that matter: the statistic is
**running**, not per-batch, and the output is **de-normalized at read-out**, so the quantity the
advantage is computed from never leaves raw units. Engstrom et al. do not touch the value target at
all (their value-side intervention is *clipping the loss*). Huang et al. record the reference
implementation's target as `returns = advantages + values` in raw units, and note that Monte-Carlo
return is the $\lambda = 1$ special case of that same construction; **normalization of the value
target appears nowhere among the 37 details**.

**(d) A terminology warning, because these three operations get conflated.** Andrychowicz's own
footnote describes Engstrom's *reward scaling* as "a form of value normalization". The full
three-way distinction is tabulated in §5.6. In one line: **reward scaling** rescales everything
upstream with a running divisor and no centring; **value normalization** rescales the critic's target
with a running mean and std and undoes it at read-out; **advantage normalization** rescales and
centres the advantage per minibatch and touches nothing else. A claim that names "normalization"
without naming which one is not a citable claim.

### Q3 — Shared vs. separate policy and value networks

**Andrychowicz's finding (C47), stated precisely.** In a 4000-configuration random search per
environment, agents with **separate** policy and value MLPs reached a higher conditional 95th
percentile than agents with a **shared** MLP + two heads on **four of five** environments (Walker2d
preferred shared), and separate accounted for **61%** of the top-5% configurations pooled across
environments. Their architecture recommendation ends "Use a wide value MLP (**no layers shared with
the policy**)". Numbers in §4.2.3.

**How strong is it? Moderate as an outcome, and it identifies no mechanism.** Four qualifications,
all from the paper's own design:

1. The two arms sample **different sub-choices** — separate draws four (policy/value width and depth), shared draws three (shared width, shared depth, and the **value-loss coefficient** $c_V \in \{0.001, 0.1, 1, 10, 100\}$). Four of the five sampled $c_V$ values are one to five orders of magnitude from 1.0, so a large fraction of the shared arm is handicapped by a loss weight no practitioner would pick.
2. **The paper never reports the marginal for $c_V$** (their C54). The experiment was **rerun with the shared arm deleted**, so Appendix E contains no figure for C48, C51 or C54.
3. The effect reverses on Walker2d and its confidence intervals overlap on Humanoid.
4. **Every other architecture result in the paper — including the last-layer initialization finding — was measured on separate-network agents only**, because of that rerun.

**Huang et al. add the corpus's only discrete-action comparison** (§6.3): separate networks clearly
beat a shared trunk on CartPole-v1 (≈490 vs ≈275 at 500k steps) and Acrobot-v1 (≈−85 vs ≈−380), with
MountainCar-v0 uninformative (all arms fail), 3 seeds, replicated across two codebases, but with **no
per-arm hyperparameter tuning** and no variation of the value-loss coefficient.

**On the specific mechanism this project has been invoking** — that in a shared trunk the value loss
dominates the summed gradient and the actor becomes a readout of features shaped only by value
regression:

- **Andrychowicz et al. are silent.** They name the hyperparameter the mechanism would predict to be critical, then remove it from the analysis before measuring it.
- **Engstrom et al. are silent by construction.** Their policy and value networks never share a parameter (separate `[64,64]` MLPs with separate learning rates), so the quantity in question does not exist in their experiments.
- **Huang et al. state the mechanism as a one-sentence conjecture** — "probably due to the competing objectives of the policy and value functions" — with no measurement of gradients, loss ratios or feature usage, and point to **Phasic Policy Gradient (Cobbe et al., 2021)** as the work that addresses it. PPG is **not in this corpus**.

**And the corpus contains a direct counterweight that must be quoted alongside.** For Atari, the
reference implementation **shares** a Nature-CNN trunk between policy and value heads; Huang et al.
keep it, reproduce the reference implementation's Atari performance with it (they measure that
reference at Breakout 409.265 ± 30.98 and show their own curves tracking it), and build their
**recurrent** PPO on top of it. **The only recurrent PPO in this corpus runs on a shared trunk.** The
field's actual convention is architecture-conditional: separate where the trunk is a cheap two-layer
MLP, shared where the trunk is an expensive feature extractor.

**Verdict for citation purposes.** "Separate networks outperformed a shared trunk in a large MuJoCo
random search (4/5 tasks) and in a small discrete classic-control comparison (2/3 tasks)" is
supportable. "…because the value loss dominates the shared trunk's gradient" is **not supported by
anything in this corpus** — it is a conjecture that one practitioner states in passing and that no
included paper measures. The formal statement that $c_V$ and the value-normalization scale
$\sigma^{-2}$ enter a shared trunk through the same slot is derived in §4.5.4 and is a statement about
the *implementation*, not evidence about its *consequences*.

### Q4 — Initialization of the policy's last layer

**The claim (Andrychowicz, §3.2 and the "most surprising finding" box).** Initializing the policy so
that the initial action distribution is **centred at zero, low-variance and observation-independent**
markedly improves training speed. The concrete lever is C57, a post-initialization rescaling of the
**policy** output layer's weights: *"initializing the policy MLP with smaller weights in the last
layer (C57, Fig. 24, **this alone boosts the performance on Humanoid by 66%**)"*. Their default is
`C57 = 0.01`, i.e. 100× smaller.

**The evidence (Fig. 24, values $\{0.001, 0.01, 0.1, 1.0\}$; full table in §4.2.4).** Read the shape:
the relationship is **a cliff at 1.0, not a gradient**. 0.001, 0.01 and 0.1 are within each other's
error bars on every environment; only leaving the layer at default scale is clearly bad, most
dramatically on Humanoid (≈2590 at 0.01 vs ≈1500 at 1.0, i.e. +73% in this reading of the figure
against the paper's stated 66%) and mildly on Hopper and Walker2d.

**The contrast that makes it credible.** The same rescaling applied to the **value** output layer
(C58, Fig. 19) has small and sign-inconsistent effects. So the finding is specific to the *policy*
head, which is what the "observation-independent initial action distribution" story predicts.

**Corroboration.** Engstrom et al. ablate `initialization ∈ {orthogonal, xavier}` and find it one of
three optimizations "necessary for attaining the highest PPO reward" — but they ablate the **whole
scheme**, including per-layer scaling, as a single switch, and never isolate the last layer or sweep
its scale. Cite them for "initialization matters", not for the last-layer claim. Huang et al.
corroborate the **convention** rather than the effect: the reference implementation initializes the
policy output layer at scale 0.01 and the value output layer at 1.0, and they apply that to a
`Categorical` head for discrete actions.

**Two conditions to attach.** (a) Andrychowicz's mechanism is about a **Gaussian mean**, and it is a
package with C61 (initial action standard deviation; 0.5 best on 4/5) and C63 (`tanh` squashing) — the
latter two have no discrete-action counterpart. (b) The evidence comes from **separate-network**
agents only.

### Q5 — Scope limits: what transfers to a discrete, recurrent, shared-trunk, homeostatic-reward setting?

The full per-finding table is §8. The one-paragraph version: **all three works study feed-forward
Gaussian policies on fully-observed continuous-control MuJoCo tasks, except Huang et al., who add
discrete and recurrent settings but run no ablations there.** Every quantitative claim about *which
choice matters* in this corpus therefore comes from MuJoCo with a feed-forward network. There is no
experiment anywhere in these three works that varies the action space, adds recurrence, or introduces
partial observability while measuring the effect of a normalization or architecture choice. That is
not a criticism of the papers — Huang et al. make the point themselves, noting that prior works'
"conclusions are in MuJoCo tasks and do not necessarily transfer to other games such as Atari" — but
it is a hard limit on what may be quoted at this project's architecture.

## 8. Scope-limit register — what may and may not be quoted at this project's architecture

This project's agent is **discrete-action**, **recurrent (GRU)**, **shared-trunk**, **partially
observed**, with a **homeostatic** reward and a **survival-step** performance measure. The corpus
studies none of those jointly. This table is the standing caveat that should travel with every
citation drawn from these three works.

### 8.1 What each paper's evidence base actually is

| | Andrychowicz et al. 2021 | Engstrom et al. 2020 | Huang et al. 2022 |
|---|---|---|---|
| Action space | Continuous (Gaussian) only | Continuous (Gaussian) only | Discrete, continuous, and `MultiDiscrete` |
| Observation | Fully observed proprioceptive vectors; "we assume … the environment is fully observable" (App. A) | Fully observed proprioceptive vectors | Vectors, images, and game state |
| Network | Feed-forward MLPs; **no recurrence anywhere** | Feed-forward `[64,64]` MLPs; no recurrence | MLPs, CNNs, **and LSTM** |
| Trunk | Shared vs separate compared, then **separate only** | **Separate only** | Separate for MLP tasks, **shared for Atari and for the LSTM agent** |
| Reward | MuJoCo locomotion (unbounded, grows with skill) | MuJoCo locomotion | MuJoCo, Atari (clipped), Procgen, μRTS |
| Evaluation | Undiscounted return, area-under-curve weighted | Final cumulative reward | Episodic return vs a reference implementation |
| Ablations run? | **Yes**, >250k agents | **Yes**, on four of nine optimizations | **No**, except shared-vs-separate |

### 8.2 Transfer register

| Finding | Source | Transfer to our setting | Why |
|---|---|---|---|
| Observation normalization is crucial | Andry. C64; Huang cont. 6 | **Likely transfers** | A statement about network input conditioning; independent of action space and recurrence. |
| PPO-style value-loss clipping is not helping | Andry. C13; Engstrom Fig. 1 | **Likely transfers** | Two independent studies agree; the mechanism concerns the value objective, not the policy parameterization. Also the cheapest change to test. |
| GAE beats N-step; $\lambda \approx 0.9$; MSE beats Huber | Andry. C6/C8/C11 | **Plausibly transfers, with tuning** | Estimator behaviour is generic, but $\lambda$ interacts with episode length and reward density, which differ here. |
| Small policy last-layer init | Andry. C57 | **Intent transfers; recipe does not** | "Observation-independent, low-entropy initial action distribution" translates to near-uniform categorical logits. The companion choices (initial action std, `tanh` squashing) have no discrete analogue. Evidence is from separate-network agents. Huang et al. show the reference implementation already applies scale 0.01 to a `Categorical` head. |
| Per-minibatch advantage normalization has little effect | Andry. C67 | **Weak transfer — a weak prior, not a result** | Direction inconsistent even within MuJoCo; effect depends on advantage scale and minibatch size, both set differently here. Their $m = 64$ from a 2048-step iteration is not our batching. |
| Value-target normalization matters strongly | Andry. C66 | **Transfers as a warning, not a direction** | It helped by 78% on one task and hurt by 42% on another *within the same benchmark suite*. The transferable content is the authors' own recommendation: check empirically. It is **not** a licence to normalize the critic target, and it is **not** a per-batch operation. |
| Reward scaling by a running return-std is necessary | Engstrom Fig. 1 | **Mechanism unestablished; regime differs** | The device exists to cope with a reward whose scale grows without bound as the agent improves. A homeostatic reward is a different regime. Huang et al. show the reference Procgen (discrete, visual) config uses it, so it is not MuJoCo-only — but nobody ablates it outside MuJoCo. |
| Separate networks beat a shared trunk | Andry. C47; Huang core 13 | **Do not transfer as a design instruction** | Confounded with capacity and with the value-loss-coefficient sampling; measured on cheap MLP trunks. Duplicating a GRU changes memory, compute and the credit-assignment path — a trade-off none of these papers faced. Counter-evidence in the same corpus: the reference implementation shares a trunk for Atari and for LSTM. |
| "The value loss dominates a shared trunk" | — | **Not evidenced anywhere in this corpus** | Andrychowicz deletes the arm before measuring; Engstrom never shares parameters; Huang states it as a conjecture and points to PPG, which is outside this corpus. |
| Recompute advantages once per data pass rather than once per iteration | Andry. C5 §3.5 | **Untested under recurrence** | Their best variant *also* shuffles transitions, which is impossible for a recurrent agent (Huang LSTM detail 4). Whether the "recompute" half is separable from the "shuffle" half is not reported. |
| Five requirements for recurrent PPO | Huang LSTM 1–5 | **Directly applicable; correctness, not tuning** | Zero-initialize and reset hidden states at episode boundaries; feed minibatches as ordered sequences from the same sub-environment; reconstruct hidden states at training time from a saved rollout-initial state; and verify that the policy ratio equals exactly 1 on the first minibatch of the first epoch. |
| Anything about partial observability | — | **Silent** | No paper in this corpus runs a partially observed task with an ablation. |
| Anything about homeostatic / interoceptive reward structure | — | **Silent** | Out of scope for all three. |
| Anything evaluated in survival steps | — | **Silent** | All three evaluate in cumulative reward. |

### 8.3 A note on what "silent" licenses

Three of the rows above say the corpus is silent. Silence is **not** evidence that a choice does not
matter, and it is **not** evidence that it does. The correct use of a silent row is to stop citing
these papers for that question and either find a paper that ran the experiment (PPG for
policy/value-objective interference; PopArt and its descendants for target-scale non-stationarity —
both already covered in this project's own survey, [[ppo_return_normalization_survey]]) or run the
experiment here.

## 9. Checks against this project's own survey

The project already holds a source-code survey with an ablation-paper section:
[`docs/project/references/modulation_in_rl/ppo_return_normalization_survey.md`](../modulation_in_rl/ppo_return_normalization_survey.md)
([[ppo_return_normalization_survey]], 2026-09-01). Every claim it makes about these three works was
re-checked against the PDFs. The result is: **no contradictions on the substance; four refinements
and three citation-hygiene corrections.**

### 9.1 Confirmed — verbatim quotations and numbers

| Survey claim | Status |
|---|---|
| Andrychowicz C67 quote — "seems not to affect the performance too much" | **Verbatim correct** (§3.3). |
| Andrychowicz C67 default is `False` in their base configuration | **Correct** (App. C, Table 2, verified against the rendered table). |
| C67 is absent from the §3.3 recommendation | **Correct**; the recommendation is quoted in full in §4.2.1. |
| Andrychowicz C66 definition and the long quote about HalfCheetah / Humanoid / Walker2d | **Verbatim correct** (App. B.9 and §3.3). |
| C66 uses a **running** statistic with **de-normalized read-out** | **Correct**, and this review adds the derivation showing the advantage therefore stays in raw units (§4.5.3). |
| "Reward scaling is not studied in Andrychowicz at all — zero occurrences" | **Correct**; independently re-grepped. |
| Andrychowicz on PPO-style value-loss clipping (C13) hurting at every threshold | **Verbatim correct** (§3.4). |
| Engstrom catalogues **nine** code-level optimizations and ablated only the **first four** for compute reasons | **Correct** (§3 and footnote 3). |
| Engstrom Figure 1 caption quote | **Verbatim correct**. |
| Engstrom's reward-scaling description and the Appendix A.2 pseudocode characterisation | **Correct**. |
| Engstrom Table 2 numbers, PPO-M definition, AAI / ACLI values | **All correct**, checked digit by digit. |
| "Engstrom never discusses advantage normalization; the word 'advantage' appears three times" | **Correct**; independently re-grepped, three occurrences, none about standardization. |
| "Value-target normalization appears nowhere among Huang's 37 details" | **Correct**. |
| Huang's advantage-normalization detail happens "at the minibatch level instead of the whole batch level" | **Verbatim correct** (core detail 7). |

### 9.2 Refinements — true, but imprecise in a way that matters for citation

1. **"Roughly cosmetic" understates the top-5% panel.** The survey's summary table says advantage
   normalization is "closer to cosmetic than load-bearing". That is a fair reading of Figure 35's left
   panel, but the right panel — which the survey does not mention — shows `True` at **53.5%** of
   top-5% configurations pooled across environments against `False`'s **47%**, reaching 0.62 on
   Hopper. The tilt is mild and sign-inconsistent across environments, but it is not zero. Suggested
   replacement wording is in §7 (Q1).

2. **"Its value is mainly step-size insensitivity, not final score" is the survey's own gloss, not a
   sourced claim.** No paper in this corpus offers that rationale. This review's derivation (§4.5.2)
   partially supports it and partially undercuts it: the *scaling* half of C67 is exactly a
   per-minibatch step-size rescale, but the *centring* half changes which samples the PPO clip
   switches off, which is a change of objective rather than of step size. If the phrase is kept, it
   should be attributed to the survey rather than to Andrychowicz.

3. **The "no significant effect" phrasing that has been used in conversation should be retired.**
   Andrychowicz et al. ran no significance test on C67. See §7 (Q1) for the correction and for
   wording that the paper supports.

4. **The survey's Andrychowicz-vs-Engstrom picture can be sharpened.** The survey lists the two
   papers' normalization findings in adjacent sections without noting that Andrychowicz's own footnote
   describes Engstrom's *reward scaling* as "a form of value normalization" and flags the two papers'
   **disagreement on Walker2d** (Engstrom's Figure 1: reward scaling helps; Andrychowicz's Figure 37:
   value-target normalization hurts by 42%). That is not a contradiction — the interventions differ —
   but it is exactly the place where a careless citation would produce one. The three-way distinction
   is tabulated in §5.6.

### 9.3 Citation-hygiene corrections

1. **The Huang detail number for reward scaling is wrong.** The survey cites it as "Detail #8, Reward
   Scaling", parallel to "Detail #7, Normalization of Advantages". Detail 7 is indeed the 7th of the
   **13 core** details, but reward scaling is the **8th of the 9 continuous-action (robotics)
   details** — the 8th core detail is the clipped surrogate objective. Correct citation: *Huang et al.
   (2022), continuous-action detail 8.*

2. **The Huang reward-scaling quotation is a close paraphrase presented as verbatim.** The survey
   quotes *"The `VecNormalize` applies a discount-based scaling scheme, where rewards are divided by
   …"*. The post reads: *"The `VecNormalize` **also** applies a **certain** discount-based scaling
   scheme, where **the** rewards are divided by the standard deviation of a rolling discounted sum of
   the rewards (without subtracting and re-adding the mean)."* Substance unchanged; the quotation
   marks should be loosened or the text corrected.

3. **The survey's Andrychowicz heading uses the preprint title with the camera-ready year.** It reads
   "Andrychowicz et al. (2021), *What Matters In On-Policy Reinforcement Learning?*". Post-§2.1, the
   correct forms are either *What Matters For On-Policy Deep Actor-Critic Methods? A Large-Scale
   Study*, ICLR 2021, or *What Matters In On-Policy Reinforcement Learning? A Large-Scale Empirical
   Study*, arXiv:2006.05990 (2020). Mixing them is what makes the same work look like two. (Huang et
   al.'s bibliography makes the identical mix, so the survey is in good company — but the fix is
   cheap.)

### 9.4 One correction that is not about the survey — a discrepancy between two papers

Engstrom et al. §3 item 1 prints the value-clipping loss with an outer **`min`**; Huang et al. core
detail 9 prints the same loss, from the same source line, with an outer **`max`**. Andrychowicz et al.
(App. B.2) describe the mechanism as "an additional **pessimistic** clipping in the value loss
function", and pessimism means taking the larger of the two candidate losses. **Read Huang's `max` as
correct and Engstrom's `min` as a typo** (§6.6.1). This matters only for someone implementing from the
printed equation rather than from the code, but that is exactly what a reader of a review does.

### 9.5 Net effect on the survey

The survey stands. Its central conclusion — that no mainstream implementation per-batch z-scores the
critic's regression target, and that every principled target-side scheme keeps the statistic slow or
fixed, de-normalizes at read-out, or preserves outputs — is **corroborated** rather than qualified by
a close reading of these three works: Andrychowicz's C66 is the running, de-normalized version;
Engstrom's reward scaling is upstream, running and centring-free; Huang's catalogue records the
reference target as `returns = advantages + values` in raw units and contains no target normalization
at all. Nothing found here weakens the survey's adjudication.

**What this corpus does *not* do is settle the mechanism.** If the project's live hypothesis is that
matched critic and advantage scale drives learning in a shared recurrent trunk, then the honest
report is: **these three papers do not test it.** Andrychowicz measured the outcome of shared vs.
separate and then removed the arm that carried the relevant hyperparameter; Engstrom never shared a
parameter; Huang offered the mechanism as a conjecture in a single sentence and pointed elsewhere.
Reading any of the three as confirming or refuting that hypothesis would be over-reading, in both
directions.

---

## 10. Provenance and method

**Extraction.** All four PDFs were read from their text layer, and every quantitative claim drawn from
a figure was verified by **rendering the figure page at 400–600 dpi and reading the plot**, because
the text layer of a figure yields only its axis ticks. The figure pages read this way were:
Andrychowicz preprint pp. 23 (Fig. 15, shared vs separate), 24 (Fig. 19, value last-layer scale),
25 (Fig. 24, policy last-layer scale) and 29 (Figs. 33–37, the normalization group); Engstrom p. 4
(Fig. 1, the $2^4$ ablation) and p. 3 (the value-clipping equation); Huang p. 19 (core detail 9's
equation) and p. 23 (the shared-vs-separate curves). Bar heights read from a rendered plot are
reported with an explicit "≈" and are accurate to roughly the width of an error whisker; they are
never used to support a claim that the paper's own prose does not already make.

**Two extraction hazards worth recording**, because they change numbers:

- **Superscripts are lost in PDF text extraction.** Engstrom's Figure 1 caption reads "the 24 possible configurations" and "a total of 5 × 24 agents" in the text layer; the rendered page shows $2^4$ and $5 \times 2^4$. Any tool-assisted reading of this caption will get the experiment's size wrong by a factor of 1.5 and misdescribe its design.
- **The Huang PDF's text layer inserts spurious spaces inside words** (an artefact of the headless-Chrome render), so every quotation from it in this review was verified against a rendered page image rather than copied from the text layer.

**Cross-checks performed.** Section 3.2 and section 3.3 of the Andrychowicz camera-ready were
mechanically diffed against the preprint to establish that they are the same text (§2.1). Every claim
this project's own survey makes about these three works was re-checked against the PDFs (§9). Word
counts used for "the paper never discusses X" claims were produced by grep over the full extracted
text, not by recall.

**What this review deliberately does not do.** It does not evaluate this project's code, does not
propose a change to `src/`, `configs/` or `scripts/`, and does not adjudicate the project's live
hypothesis about matched critic and advantage scale. Two observations surfaced that imply possible
code changes — the advantage-recomputation-per-epoch point (§4.6) and the five recurrent-PPO
correctness conditions (§8.2, from Huang's LSTM details) — and both are flagged as **hand-offs to
`senior-developer` for an `issue_plan`**, not as decisions.

**Related project documents.**

- [[ppo_return_normalization_survey]] — `docs/project/references/modulation_in_rl/ppo_return_normalization_survey.md`, the ten-implementation source-code survey this review was checked against (§9).
- [[mc_return_units_bug_severity_and_repair]] — `docs/project/critiques/`, the severity analysis that motivated the survey.

**Handoff status.** All three works carry a section-by-section backbone appendix (§4.7, §5.8, §6.8),
a Phase 1 foundational overview (§4.4, §5.4, §6.5) and a Phase 2 graduate-level deep dive (§4.5,
§5.5, §6.6). The table of contents is current as of this document's `last_updated` date. If the corpus
grows, append the new work as §10 onward and move this provenance section to the end.
