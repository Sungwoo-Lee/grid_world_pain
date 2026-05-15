---
title: "Neuromodulated attention and goal-driven perception in uncertain domains"
authors: ["Xinyun Zou", "Soheil Kolouri", "Praveen K. Pilly", "Jeffrey L. Krichmar"]
year: 2020
venue: "Neural Networks 125:56–69 (Elsevier)"
slug: zou_2020_neuromodulated_attention
source_pdf: "sources/Zou et al. 2020 - Neuromodulated attention and goal-driven perception in uncertain domains.pdf"
topic: neuromodulatory_algorithms
---

## Plain-English entry point

This paper asks: when a system has to attend to a *goal* in a noisy world, but the system *doesn't know which goal is currently active*, how should it decide what to look at? In animals, two neuromodulator systems are believed to share this work. **Acetylcholine (ACh)** — released by the basal forebrain — boosts attention to a known target and suppresses attention to distractors. **Norepinephrine (NE)** — released by the locus coeruleus — fires phasically (a burst of activity) when something unexpected happens, and resets the network so it can quickly switch strategies. Yu and Dayan (2005) made this a formal Bayesian story: ACh tracks **expected uncertainty** (you know the world is noisy, you've already taken the noise into account); NE tracks **unexpected uncertainty** (the rules changed without warning).

The authors port that two-modulator story onto a deep network for attention. They use **contrastive Excitation Backprop (c-EB)** — a method by Zhang et al. (2018) that runs the gradient backwards from a chosen output neuron and produces a saliency map showing which input pixels drove the prediction, while a parallel inhibitory backward pass cancels out neurons shared with competing classes so only the goal-relevant pixels survive. They train a small MLP on pairs of noisy MNIST digits with four goals — *even*, *odd*, *low (0–4)*, *high (5–9)* — and a goal validity (reward probability) that drifts during the run. They wrap this network with a small **ACh + NE neuromodulatory head**: four ACh neurons (one per goal) updated by simple multiplicative rules on each trial, plus one NE neuron that detects when the goal seems to have switched and triggers a network reset.

The headline claim: the ACh-NE head lets the system **autonomously discover the current goal**, attend to the right pixels via c-EB, recover within ~20–50 trials when the goal switches, and explore versus exploit gracefully depending on goal validity. The same architecture generalizes to a Toyota Human Support Robot using GoogLeNet + COCO objects, with the four goals being actions ("eat", "work-on-computer", "read", "say-hi").

## Section-ordered backbone

### Abstract
In uncertain domains, goals are unknown and must be predicted. The authors apply **contrastive Excitation Backprop (c-EB)** to two goal-driven perception tasks: (i) noisy MNIST digit pairs with parity/magnitude goals; (ii) a robot action-attention scenario. Because the valid goal is unknown, an **online learning model based on the cholinergic and noradrenergic neuromodulatory systems** predicts a noisy goal (expected uncertainty) and re-adapts when the goal changes (unexpected uncertainty). The biologically plausible model shows how neuromodulators predict goals in uncertain domains and how attention enhances perception of those goals.

### 1. Introduction
Classical attention is largely context-invariant (Itti & Koch 2000; Tsotsos et al. 2015). Animals attend selectively to context- and goal-relevant features, learn the goal from experience, and adapt fast when conditions shift. **ACh** drives both bottom-up and top-down attention (Avery, Dutt, & Krichmar 2014; Baxter & Chiba 1999; Oros et al. 2014) — increment attention to task-relevant stimuli, decrement to distractors. **NE** responds phasically to surprise / large deviations from priors (Yu & Dayan 2005; Bouret & Sara 2005; Grella et al. 2019), causing a network reset. **Contrastive EB** (Zhang et al. 2018) increments via an excitation mask + decrements via an inhibition mask, conceptually mirroring ACh. The paper modifies c-EB for goal-driven perception, adds an ACh+NE head, and tests it on noisy MNIST pairs (Experiment 1) and on a HSR robot with COCO objects (Experiment 2).

### 2. Methods
**2.1 Network architecture (Experiment 1).** Two-digit MNIST pairs (28×28 each, total 1568 input neurons) with uniform [0, 0.7] pixel noise added then re-normalized. Architecture: two sequential FC layers (800 → 600) → two parallel FC layers (400 each) → outputs. Each parallel head outputs per-side: two parity neurons (even/odd) + ten digit neurons. Activation: ReLU. During training, the final digit prediction averages the two parallel heads; during testing, only the parallel head matching the cued goal class is used.

**2.1.1 c-EB modification.** Standard EB (Zhang et al. 2018) propagates a probabilistic Winner-Take-All Marginal Winning Probability (MWP) backward:
$$
P(a_n) = \sum_{a_m \in (L_0, \dots, L_{l-1})} P(a_n \mid a_m) \cdot P(a_m),
$$
with conditional
$$
P(a_n \mid a_m) = \begin{cases} \frac{\hat a_n \cdot w_{nm}}{\sum_{n: w_{nm}\ge 0} \hat a_n \cdot w_{nm}} & \text{if } w_{nm}\ge 0 \\ 0 & \text{otherwise} \end{cases}.
$$
**c-MWP** subtracts an inhibition-mask backward pass (weights flipped sign):
$$
A - \bar A = P_0 \cdot (P_1 - \bar P_1) \cdot P_1 \cdot \dots \cdot P_{l-1}.
$$
The authors additionally route c-EB through the parallel hidden layer corresponding to the cued goal class only, so per-goal attention maps differ even when the target digit is the same.

**2.1.2 Training and testing.** Adam ($\alpha=10^{-3}$), 4400 training steps × 256 noisy MNIST pairs/step = 1,126,400 training pairs. Validation: 2000 pairs / 200 steps. Test: 10,000 pairs, no digit overlap with training. Test pairs always have opposite parity *and* opposite high/low to avoid ambiguous-goal pairs.

**2.2 Neuromodulated goal-driven perception.**
- **K = 4 ACh neurons**, one per goal (even/odd/low/high).
- **1 NE neuron**.
- ACh activities go through softmax (temperature $1/\beta$) to choose a guessed goal each trial:
  $$
  p(\text{goal})_i = \frac{\exp(\beta \cdot \text{ACh}_i)}{\sum_{j=1}^K \exp(\beta \cdot \text{ACh}_j)}.
  $$
- After each trial, update the *guessed-goal* ACh and the global NE:
  $$
  (\text{ACh}_g)_t = \begin{cases} \min(ch_{\text{correct}}\cdot(\text{ACh}_g)_{t-1},\, ch_{\max}) & \text{if correct} \\ \max(ch_{\text{wrong}}\cdot(\text{ACh}_g)_{t-1},\, ch_{\min}) & \text{otherwise} \end{cases}
  $$
  $$
  \text{NE}_t = \begin{cases} \max(ne_{\text{correct}}\cdot\text{NE}_{t-1},\, ne_{\min}) & \text{if correct} \\ \min(ne_{\text{wrong}}\cdot\text{NE}_{t-1},\, ne_{\max}) & \text{otherwise} \end{cases}
  $$
  Constraints: $ch_{\text{correct}}, ne_{\text{wrong}} \in [1,2)$; $ch_{\text{wrong}}, ne_{\text{correct}} \in (0,1]$.
- **Reset threshold** (Yu & Dayan 2005):
  $$
  \theta_{\text{reset}} = \frac{\sum_{i=1}^K \text{ACh}_i / K}{0.5 + \sum_{i=1}^K \text{ACh}_i / K}.
  $$
  When $\text{NE} > \theta_{\text{reset}}$, all ACh neurons reset to $ch_{\text{reset}}$ and NE to $ne_{\text{reset}}$.

Default parameters: $\beta=0.7$, num_switches=10, K=4, trial_interval=400, trial_range=30, $ne_{\text{reset}}=0.25$, $ne_{\min}=0.25$, $ne_{\max}=1.0$, $ch_{\text{reset}}=1.0$, $ch_{\min}=0$, $ch_{\max}=10.0$, $ne_{\text{correct}}=0.70$, $ne_{\text{wrong}}=1.10$, $ch_{\text{correct}}=1.40$, $ch_{\text{wrong}}=0.90$.

**2.2.2 Goal selection setup.** Each run: 10 major-goal switches every $400 \pm 30$ trials. Major-goal validity drawn from $\{0.99, 0.85, 0.70\}$ (consistent with Yu & Dayan 2005); minor goal validity is $(1 - \text{major})$ from the *same goal class* (e.g., major=high → minor=low).

**2.3 Action-based attention on Toyota HSR.** Four actions: eat / work-on-computer / read / say-hi. Each action linked to a list of related COCO objects (Table 1). Underlying classifier: GoogLeNet trained on Microsoft COCO via Caffe. c-EB routes through `pool5/7x7_s1` → `pool3/3x3_s2`. Per trial: HSR takes three view-angle captures, c-EB produces saliency, threshold 0.1, the highest-attention region is kept (rest blacked out), forward pass classifies the surviving region. Action validity = 1; trial_interval = 50; $\beta=10$; $ne_{\text{correct}}=0.75$, $ne_{\text{wrong}}=1.15$, $ch_{\text{correct}}=1.35$, $ch_{\text{wrong}}=0.95$. User feedback YES/NO drives the modulator update.

### 3. Results
**3.1 Digit prediction with c-EB (Table 2).** Without neuromodulator overhead: goal prediction >99%; digit prediction 87–95% depending on goal task. c-EB filters background noise and the wrong-side digit.

**3.2 Goal-driven perception with uncertainties (Figs. 8, 9; Table 3).** With $p_{\text{valid}}=0.99$: 86.1% correct major-goal selection, 21-trial lag after a switch. With $p_{\text{valid}}=0.85$: 73.0%, 29-trial lag. With $p_{\text{valid}}=0.70$: 57.9%, 48-trial lag. Lower validity → more frequent NE bursts, more exploration, longer lag, occasionally correct minor-goal predictions. Predictions tend to *stay within the correct goal class* (parity vs magnitude).

**3.3 Ablations (Table 4, Fig. 10).** Ablating NE: lag rises from 30 → 54 trials (no reset → slow tracking). Ablating ACh: collapses to random guessing (19.8% major-goal, lag 400). Ablating both: same. **Both** modulators are necessary; ACh carries the choice signal; NE carries the switch detector.

**3.4 Goal-selection method comparison (Table 5).** Neuromodulated softmax ($\beta=0.7$): 75.1% major-goal, lag 30. Neuromodulated WTA: 76.4% major-goal, lag 24. "Random-or-fixed" baseline (random until match, then stay until mismatch): 63.1% major-goal, lag 23 — *but* the high minor-goal rate is mostly random noise across all four goals, so the baseline loses the *correct-goal-class* property. Softmax gives more flexible minor-goal selection than WTA when validity drops, at the cost of slightly longer lag.

**3.5 Robot results (Section 3.5).** HSR experiment: average goal selection error 23.8% (action validity =1, trial_interval=50, parameters as above). c-EB object prediction error 30.6%, lag 13 trials. Uncertainties addressed: object location switches, object removals/additions, multiple instances, view-angle variation.

### 4. Discussion
**4.1 Main findings.** Neuromodulated c-EB tracks context and shifts attention flexibly. ACh-like incrementing/decrementing of attention aligns with Baxter & Chiba 1999; Oros et al. 2014. Goal validity 0.99 → 0.85 → 0.70 spans a wide spectrum from confident exploitation to active exploration. NE-driven reset clears the *goal prior* but not the trained representations.

**4.1.2 Exploration / probability matching.** Humans probability-match instead of always picking the best option (Wozny et al. 2010; Craig et al. 2016). Rats seek uncertainty via ACh (Naude et al. 2016). The model reproduces this: probabilistic minor-goal sampling, not WTA.

**4.2 Related work.** EB / c-EB (Zhang et al. 2018), CAM / Grad-CAM (Zhou et al. 2016; Selvaraju et al. 2017). The model differs by adding a neuromodulatory layer for goal discovery in dynamic settings.

**4.3 Future directions.** Handling new goal classes; alternative top-down attention mechanisms (CAM, Grad-CAM); broader AI applications (self-driving, human-support); biology predictions — NE phasic activity after a goal switch; possibly random ACh activity to V1 after NE reset.

### 5. Conclusions
Online ACh + NE neuromodulation atop c-EB discovers unknown goals, attends to goal-relevant features, and rapidly adapts when goal contingencies change.

## Phase 1 — Undergraduate-level synthesis

**Key idea.** A neural network is shown a pair of noisy handwritten digits and told to pay attention to one of four kinds of goal: "even", "odd", "low (≤4)", or "high (≥5)". A separate, very small "neuromodulator" module on top has to *figure out which goal is active right now* without being told. Two competing chemicals do the work:

- **Acetylcholine (ACh)** — four neurons, one per goal. Each represents "how confident I am the current goal is X". After a correct guess, the chosen one's strength multiplies up; after a wrong guess, it multiplies down.
- **Norepinephrine (NE)** — one neuron, a "surprise alarm". It goes up after wrong guesses, down after right guesses. When it crosses a threshold, the whole module resets ACh to a flat starting point — like raising a hand and saying "I have no idea anymore, let's start over".

A separate machinery called **contrastive Excitation Backprop** (c-EB) is what actually paints attention onto pixels. You feed in the noisy digit pair; you tell c-EB "the goal is 'odd'"; c-EB runs the network *backwards* twice — once exciting the path to "odd" outputs, once inhibiting that path — and subtracts the two backward signals. What survives is a heat map highlighting only the pixels that mattered for picking the odd digit and ignoring the noise.

**Setup.** Train the network on a million noisy MNIST digit pairs to be good at all four goal tasks at once. Then run "test sessions": every ~400 trials the *true* goal silently changes, and the *validity* (probability that the rewarded goal really is the announced major goal) can be 0.99, 0.85, or 0.70. The neuromodulator has to track this.

**Result.** With 99% validity, the system locks onto the right goal within ~20 trials of a switch and gets it right ~86% of the time. With 70% validity, it takes ~50 trials and gets it right ~58% — but it spends more time exploring the other goal, which is the *right* behavior because the "wrong" goal really does pay off 30% of the time. Knock out NE: the system is slow to switch (lag doubles). Knock out ACh: the system collapses to random guessing. You need both.

**Concrete instantiation.** A robot in a classroom (the Toyota HSR) sees objects through a camera. A user types "I want to eat". The neuromodulator guesses the action is "eat", c-EB highlights bananas / sandwiches / apples in the scene (ignoring laptops and books), GoogLeNet identifies the highest-attention object, the robot moves to it and shows it to the user. User says "no" → ACh "eat" gets multiplied down, NE goes up. Eventually NE crosses threshold and resets — the robot starts over, tries "work-on-computer", maybe lands on the laptop, gets "yes", ACh "work-on-computer" climbs.

## Phase 2 — Graduate-level deep dive

### Contrastive Excitation Backprop in formal detail

**Standard EB (Zhang et al. 2018).** For a feedforward network with non-negative excitatory weights, define the Marginal Winning Probability (MWP) of neuron $a_n$ in layer $L_l$ recursively from layer 0:

$$
P(a_n) = \sum_{a_m \in (L_0, L_1, \dots, L_{l-1})} P(a_n \mid a_m)\, P(a_m), \tag{1}
$$

with conditional sampling probability

$$
P(a_n \mid a_m) = \begin{cases} \dfrac{\hat a_n \cdot w_{nm}}{\sum_{\hat n:\, w_{\hat n m} \ge 0} \hat a_{\hat n} \cdot w_{\hat n m}} & \text{if } w_{nm} \ge 0, \\[6pt] 0 & \text{otherwise.} \end{cases} \tag{2}
$$

EB is therefore a *forward sampling* in reverse: starting from the chosen output neuron, sample a parent neuron in the previous layer proportional to (activation × non-negative weight), recurse to layer 0, and the cumulative MWP at the input layer is a saliency map. It is *probabilistic*; the WTA is implicit in the sampling.

**Contrastive EB.** Run EB twice with opposite sign masks at the top layer. The "inhibition mask" $\bar P_1$ uses the *negative* of the original top-layer weights (the threshold condition in Eq. 2 is reversed). The c-MWP for target layer $L_l$ is:

$$
A - \bar A = P_0 \cdot (P_1 - \bar P_1) \cdot P_1 \cdot P_2 \cdots P_{l-1}. \tag{3}
$$

The subtraction in $(P_1 - \bar P_1)$ cancels neurons that are winners under both the goal and the anti-goal — *common-mode rejection*. What remains are neurons whose forward path is selective for the chosen goal vs. its competitors. Hence the c-MWP highlights pixels that distinguish the goal digit from the distractor, not pixels that are merely active.

**The authors' contribution to c-EB.** Instead of routing the contrastive mask through a single hidden layer, they route through the parallel hidden layer corresponding to the *cued goal class* (parity vs. magnitude). This decouples per-class attention so the saliency map for "odd" digit 4 differs from the saliency map for "low" digit 4 — important because the cued *goal class* changes how the same pixel pattern should be interpreted.

### The ACh/NE neuromodulator as a Bayesian uncertainty tracker

**Yu & Dayan (2005) decomposition.** Let $p(g_t = i \mid \text{history})$ be the posterior over the discrete goal identity $i \in \{1, \dots, K\}$. Decompose total uncertainty as:

$$
\underbrace{\text{Total uncertainty}}_{H[g_t]} = \underbrace{\text{Expected uncertainty}}_{\text{noise within a known goal}} + \underbrace{\text{Unexpected uncertainty}}_{\text{the goal itself changed}}.
$$

ACh tracks the first; NE tracks the second. In Zou et al., the ACh state is a $K$-vector $\mathbf{ACh} \in \mathbb{R}^K_{\ge 0}$, and the NE state is a scalar $\text{NE} \in \mathbb{R}_{\ge 0}$.

**Goal-selection softmax** (Eq. 4):

$$
p(\text{goal})_i = \frac{\exp(\beta \cdot \text{ACh}_i)}{\sum_{j=1}^K \exp(\beta \cdot \text{ACh}_j)}, \qquad i = 1, \dots, K. \tag{4}
$$

This is **probability matching** in the high-uncertainty limit and **argmax** in the low-uncertainty limit, controlled by inverse temperature $\beta$. The default $\beta = 0.7$ is deliberately soft so the minor goal can be sampled.

**Update rules** (Eqs. 5, 6) are *multiplicative*, with separate up- and down-gain factors:

$$
(\text{ACh}_g)_t = \begin{cases} \min(ch_{\text{correct}} (\text{ACh}_g)_{t-1},\, ch_{\max}) & \text{if correct} \\ \max(ch_{\text{wrong}} (\text{ACh}_g)_{t-1},\, ch_{\min}) & \text{otherwise} \end{cases} \tag{5}
$$

$$
\text{NE}_t = \begin{cases} \max(ne_{\text{correct}} \text{NE}_{t-1},\, ne_{\min}) & \text{if correct} \\ \min(ne_{\text{wrong}} \text{NE}_{t-1},\, ne_{\max}) & \text{otherwise} \end{cases} \tag{6}
$$

Only the *guessed* goal's ACh updates; the other three ACh values drift only through subsequent trials when they are selected.

**Reset rule** (Eq. 7):

$$
\theta_{\text{reset}} = \frac{\bar{\text{ACh}}}{0.5 + \bar{\text{ACh}}}, \qquad \bar{\text{ACh}} = \frac{1}{K}\sum_{i=1}^K \text{ACh}_i. \tag{7}
$$

This threshold scales with the mean ACh. If $\bar{\text{ACh}}$ is small (low confidence, possibly fresh after a reset), $\theta_{\text{reset}}$ is small and even a modest NE rise can trigger another reset; if $\bar{\text{ACh}}$ is large (system confidently exploiting a stable goal), $\theta_{\text{reset}}$ approaches 1 and NE has to climb high to trigger reset. This makes the modulator hysteretic: hard to disturb when confident, easy to disturb when uncertain.

**Derivation of $\theta_{\text{reset}}$ shape.** The form $x / (c + x)$ with $c=0.5$ is a saturating Michaelis–Menten-style nonlinearity. As $x \to 0$, $\theta_{\text{reset}} \to 0$; as $x \to \infty$, $\theta_{\text{reset}} \to 1$; at $x = 0.5$, $\theta_{\text{reset}} = 0.5$. Combined with $\text{NE} \in [ne_{\min}, ne_{\max}] = [0.25, 1.0]$, the reset is feasible only when $\bar{\text{ACh}}$ is at least roughly comparable to $\text{NE}$.

### Lag-length analysis after a goal switch

After a goal switch, the network's true major goal is now (say) "odd" but the system still believes it's "even". On each trial, the system samples a goal from softmax over ACh; if it picks the new correct goal, ACh for that goal *multiplies up* by $ch_{\text{correct}} = 1.4$; if wrong, the chosen ACh *multiplies down* by $ch_{\text{wrong}} = 0.9$. NE rises on misses (×1.1) and falls on hits (×0.7).

Setting aside the reset for a moment, the convergence time to a stable correct major-goal preference can be analyzed in expectation. With $K=4$ and an initially uniform softmax distribution after a reset, the expected number of trials to first hit the new correct goal is $K = 4$. Each correct hit pushes that ACh up by a factor 1.4; to reach the ceiling $ch_{\max}=10$ from $ch_{\text{reset}}=1$ takes $\lceil \ln 10 / \ln 1.4 \rceil = 7$ correct hits. With validity 0.99 the lag observed is ~20 trials, comfortably above $K\cdot 7 / 0.99 \approx 28$ if the system stayed uniform — meaning the softmax bias toward an emerging winner accelerates locking-in. With validity 0.70, of the 7 needed correct hits, ~30% are misses that down-multiply the winner by 0.9, requiring ~$7/(0.7-0.3)$ ≈ 17 net hits, consistent with the measured 48-trial lag.

### Two NE roles, one neuron

A single scalar $\text{NE}$ does two jobs:
1. **Surprise signal** — rises on each miss.
2. **Reset signal** — when above $\theta_{\text{reset}}$, triggers a hard reset of all ACh and NE.

Because the reset compares NE against an ACh-weighted threshold, the same neuron implements both "this trial felt surprising" (Bouret & Sara 2005) and "the world is no longer the world I learned" (Grella et al. 2019). The reset zeros out the *prior* (ACh) but not the *parameters* of the trained perceptual network.

### Why softmax beats WTA at low validity

With WTA, once one ACh wins by a margin it is always selected, and a low-validity major goal (say 70%) means 30% of trials are "wrong" → drives that ACh down, but also triggers more NE bursts → reset → uniform reset → re-explore. With softmax ($\beta=0.7$), even at the winner, the minor goal has probability $\exp(\beta \cdot \text{ACh}_{\min}) / Z > 0$ and gets occasionally sampled, picking up its own rewards — so the system *probability-matches* the validity instead of fighting it. Net effect at validity 0.70: WTA never realizes the minor goal pays off, softmax does.

### Parameter table (Experiment 1 defaults)

| Symbol | Value | Role |
|---|---|---|
| $K$ | 4 | Number of ACh neurons (goals) |
| $\beta$ | 0.7 | Softmax inverse temperature |
| trial_interval | 400 | Mean trials between major-goal switches |
| trial_range | 30 | $\pm$ jitter on switch interval |
| $ne_{\text{reset}}$ | 0.25 | NE value after a reset |
| $ne_{\min}, ne_{\max}$ | 0.25, 1.0 | NE bounds |
| $ch_{\text{reset}}$ | 1.0 | ACh value after a reset |
| $ch_{\min}, ch_{\max}$ | 0, 10.0 | ACh bounds |
| $ne_{\text{correct}}$ | 0.70 | NE down-gain on correct |
| $ne_{\text{wrong}}$ | 1.10 | NE up-gain on wrong |
| $ch_{\text{correct}}$ | 1.40 | ACh up-gain on correct |
| $ch_{\text{wrong}}$ | 0.90 | ACh down-gain on wrong |

### Mapping to Doya 2002

- **DA = TD error / reward signal**: Zou et al. have correct/wrong feedback per trial — the analog of a reward signal driving the multiplicative ACh/NE updates.
- **ACh = memory time-constant / signal-to-noise**: explicitly modeled here as four neurons tracking expected uncertainty per goal.
- **NE = exploration / randomness**: explicitly modeled here as the reset/switch detector.
- **5-HT = temporal discounting**: not modeled in this paper.

The Zou et al. instantiation operationalizes Yu & Dayan's two-modulator story as a *neural-network-friendly outer loop* on top of a c-EB attention map, complementing the Doya framework's single-cell view.

## Connections

**Direct references inside this corpus:**

- **[doya_2002_metalearning_neuromodulation](doya_2002_metalearning_neuromodulation.md)** — foundational. Zou et al. operationalize a particular slice (ACh + NE) of the Doya framework. Not cited by name in this paper but the conceptual lineage is unmistakable.
- **[avery_krichmar_2017_models_neuromodulation](avery_krichmar_2017_models_neuromodulation.md)** — cited as Avery et al. 2014 (related earlier work) for the basal-forebrain enhancement of attention.
- **[hwu_krichmar_2020_schemas_memory](hwu_krichmar_2020_schemas_memory.md)** — same lab and same DARPA L2M contract (FA8750-18-C-0103). Both papers share Krichmar 2008's neuromodulation-as-survival framework. Hwu & Krichmar's mPFC-HPC indexing + neuromodulated replay solves continual learning of schemas; Zou et al.'s ACh-NE solves online goal discovery. Hwu & Krichmar's acknowledgments thank Xinyun Zou for collaboration on the precursor Telluride 2017 project.
- **[xing_2020_neuromodulated_patience](xing_2020_neuromodulated_patience.md)** — same lab, same year. Xing 2020 puts neuromodulation (5-HT) in the *action-selection / patience* loop; Zou 2020 puts neuromodulation (ACh + NE) in the *attention / perception* loop. Xinyun Zou is second author on Xing 2020.
- **[krichmar_hwu_2022_design_principles_neurorobotics](krichmar_hwu_2022_design_principles_neurorobotics.md)** — Krichmar & Hwu's later synthesis explicitly uses Zou 2020 as a worked example of "principle: ACh and NE complementary modulation".
- **[chiba_krichmar_2020_self_monitoring](chiba_krichmar_2020_self_monitoring.md)** — companion review article from the same lab on neurobiologically inspired self-monitoring, citing the same ACh/NE framework.

**External anchors cited.** Yu & Dayan 2005 (ACh/NE Bayesian uncertainty decomposition); Bouret & Sara 2005 (NE-driven network reset); Grella et al. 2019 (LC phasic activation drives global remapping); Naude et al. 2016 (ACh-mediated uncertainty seeking in rats); Zhang et al. 2018 (c-EB); Selvaraju et al. 2017 (Grad-CAM); LeCun et al. 1998 (MNIST); Lin et al. 2014 (COCO); Szegedy et al. 2015 (GoogLeNet); Yamamoto et al. 2018 (HSR robot).

**Forward citations expected in this corpus.** Lee et al. 2024, Vecoven et al. 2020, Ben-Iwhiwhu et al. 2022, Wang et al. 2024 — later papers extending neuromodulatory gating into deep RL and meta-learning sit in the same "online uncertainty-driven adaptation" theme. Espino et al. 2024 and Alonso & Krichmar 2023 are in the related continual-learning + sparse-memory cluster.
