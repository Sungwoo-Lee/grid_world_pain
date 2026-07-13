> **Per-paper review — continual-learning corpus, paper 13 of 22.**
> Extracted from the [master review](../continual_learning_lit_review.md) (§13); content is identical. Field-evolution primer: [[continual_learning_field_evolution]].

# 13. Sokar et al. 2023 — The Dormant Neuron Phenomenon in Deep RL (ReDo)

**PDF:** `docs/project/references/continual_learning/sources/Sokar et al. 2023 - The Dormant Neuron Phenomenon (ReDo).pdf`
**Venue:** ICML 2023, PMLR 202 · **arXiv:** 2302.12902
**Authors:** Ghada Sokar, Rishabh Agarwal, Pablo Samuel Castro, Utku Evci (Eindhoven / Google DeepMind / Mila)

## <a id="redo-phase-1"></a>Phase 1 — Foundational Overview (undergraduate level)

**The plain-language question.** Abbas showed that units in an RL network go silent over training. Sokar asks: what *exactly* is the atomic unit of that decay, can we measure it precisely, and can we reverse it *without* the drastic step of wiping the whole network? Their answer names a concrete object — the **dormant neuron** — and offers a surgical fix, **ReDo** ("Recycle Dormant neurons").

**What a dormant neuron is.** A neuron is *dormant* if its activation, averaged over inputs and normalized against the other neurons in its layer, is essentially zero — it has stopped contributing to the network's output. The paper's central empirical fact, the **dormant neuron phenomenon**: as an RL agent trains, the *number* of dormant neurons steadily *grows*, and once a neuron goes dormant it tends to *stay* dormant. The network is quietly shrinking its own usable size even though it is nominally over-parameterized.

**Three diagnostic findings.**
1. **It's caused by moving targets, not moving data.** RL has two kinds of non-stationarity: the *input* distribution shifts (the agent's own policy changes what it sees) and the *target* shifts (the network bootstraps off its own changing estimate). Using controlled CIFAR-10 and offline-RL experiments, Sokar shows the *target* non-stationarity is the primary culprit — dormancy grows when learning targets keep moving, and barely grows with fixed targets, even when the input data is fixed.
2. **More gradient updates → more dormant neurons.** Cranking up the **replay ratio** (updates per environment step) increases dormancy — which explains why naively training harder on the same data collapses performance. This is the link to D'Oro's replay-ratio work.
3. **Dormancy directly damages future learning.** A pre-trained network full of dormant neurons is measurably *worse* than a fresh random network at fitting a new target — dormancy is not cosmetic, it degrades the learner.

**The fix — ReDo.** Every so often during training, scan every layer; any neuron below a dormancy threshold $\tau$ gets its *incoming* weights re-randomized and its *outgoing* weights zeroed. Zeroing the outgoing weights means the recycled neuron initially changes the network's output *not at all* (a "do no harm" property) — but its incoming weights are now fresh, so it can start learning again. Result: dormancy stays low, network capacity is maintained, and performance improves — especially at high replay ratios, where ReDo *avoids the performance collapse* that normally caps how hard you can train.

**Initial takeaway.** ReDo is the *surgical* member of the primer's "reset / recycle capacity" family (primer §4a): rather than resetting whole layers (Nikishin 2022) or perpetually reinitializing the least-used units (Dohare's continual backprop), it recycles *only* the units that have actually gone dormant, and does so without disturbing the current output. Directly relevant to the project's replay-ratio study: ReDo is a concrete lever for pushing the replay ratio up without collapse.

## <a id="redo-phase-2"></a>Phase 2 — Graduate-Level Deep Dive

**Formal definition of a dormant neuron.** Given an input distribution $D$, let $h_i^\ell(x)$ be the activation of neuron $i$ in layer $\ell$ under input $x\in D$, and $H^\ell$ the number of neurons in layer $\ell$. The neuron's **score** is its expected absolute activation, normalized by the layer's average:

$$
s_i^\ell \;=\; \frac{\mathbb{E}_{x\in D}\,\lvert h_i^\ell(x)\rvert}{\tfrac{1}{H^\ell}\sum_{k\in h}\mathbb{E}_{x\in D}\,\lvert h_k^\ell(x)\rvert}.
$$

A neuron is **$\tau$-dormant** if $s_i^\ell \le \tau$. The denominator normalizes scores to sum (in expectation) to a constant within a layer, making neurons *across layers of different width* comparable — a key design choice, since a raw activation magnitude means different things in a 512-wide vs. 3136-wide layer.

**Definition of the phenomenon.** An algorithm *exhibits the dormant neuron phenomenon* if the count of $\tau$-dormant neurons increases steadily throughout training. Such a network under-utilizes its capacity, and the under-utilization worsens over time. (Early analyses use the strict $\tau=0$; benchmarking loosens to $\tau=0.1$.)

**Why $\tau$-dormancy matters even for small $\tau$.** Low-activation neurons could in principle still shape the learned function, but their contribution — and the disruption from recycling them — is bounded by their small activation magnitude. So recycling a $\tau$-dormant neuron with small $\tau$ perturbs the output only slightly; with $\tau=0$, the output is left *exactly* unchanged.

**Evidence chain for the cause (target non-stationarity):**
- *Baseline phenomenon.* DQN dormant-neuron fraction rises steadily across gradient steps (DemonAttack, Asterix; Fig. 2), consistent across algorithms (DrQ($\epsilon$), SAC) and domains (Atari, MuJoCo).
- *Supervised control (Fig. 3).* CIFAR-10 with **fixed** labels → dormancy *decreases* over time; with **shuffled/non-stationary** labels → dormancy *increases*, with sharp jumps exactly at label-shuffle points. Isolates target non-stationarity.
- *Offline-RL control (Fig. 4).* Fixed dataset (removes *input* non-stationarity) but standard moving TD targets → phenomenon persists. Ablating to fixed *random* targets → dormancy drops. Therefore **target** non-stationarity (bootstrapping off a moving estimate) is primary; *input* non-stationarity is not a major factor.
- *Persistence (Figs. 5–6).* The overlap coefficient $\text{overlap}(X,Y)=\frac{\lvert X\cap Y\rvert}{\min(\lvert X\rvert,\lvert Y\rvert)}$ between the current and historical dormant sets *rises* → dormant neurons rarely reactivate. Explicitly **pruning** all-time-dormant neurons does **not** hurt performance → confirms they are functionally inert.
- *Replay-ratio dependence (Fig. 7).* Higher replay ratio (RR ∈ {0.25, 0.5, 1, 2}) → strictly more dormant neurons, correlating with the known performance drop at high RR. This is the mechanistic bridge to [D'Oro's replay-ratio barrier](#15-doro-et-al-2023--sample-efficient-rl-by-breaking-the-replay-ratio-barrier).
- *Causal harm (Fig. 8).* Distilling a dormant-heavy pre-trained DQN toward a well-performing target *degrades* over training and its dormancy keeps climbing, while a randomly-initialized network improves and stays stable-dormancy. Dormancy is a *cause* of impaired new-task learning, not a mere correlate.

**The ReDo algorithm (Algorithm 1).**

```
Input: parameters θ, threshold τ, training steps T, frequency F
for t = 1 to T:
    Update θ with the regular RL loss
    if t mod F == 0:
        for each neuron i:
            if s_i^ℓ ≤ τ:
                Reinitialize incoming weights of neuron i   (from the original init distribution)
                Set outgoing weights of neuron i to 0
```

**Why the two-sided reinit is the right design.** Reinitializing *incoming* weights gives the neuron a fresh, non-degenerate learning direction. Zeroing *outgoing* weights guarantees the recycled neuron's *immediate* contribution to the next layer is zero, so at the moment of recycling the network's function is (for $\tau=0$) unchanged and (for small $\tau$) barely perturbed — ReDo restores plasticity *without* the abrupt performance drop and re-exploration that a full-layer reset (Nikishin 2022) causes. Ablations (App. C.2): scaling incoming weights by the mean non-dormant norm ≈ same as using the init distribution; randomizing outgoing weights ≈ same or worse. So the simple recipe is not fragile.

**Is it a ReLU-specific problem?** RL nets typically use ReLU, which saturates at zero output (zero gradient). App. C.1 measures dormancy under a *different* activation and finds a *mild* decrease but the phenomenon persists — so dormancy is not purely a ReLU artifact (contrast Abbas, who fixes it *at the activation level* with CReLU; ReDo instead fixes it *at the weight level* with recycling — the two are complementary members of families (c) and (a)).

**Empirical results.**
- **Setup.** DQN on 17 ALE games (default CNN + IMPALA ResNet); DrQ($\epsilon$) on the 26-game Atari 100k; SAC on 4 MuJoCo tasks. Dopamine framework. Default $\tau=0.1$ (beat $\tau=0$ and $\tau=0.025$). Metric: Interquartile Mean (IQM, Agarwal 2021) with 95% stratified-bootstrap CIs, 5–10 seeds.
- **§5.1 Sample efficiency.** Across RR ∈ {0.25, 0.5, 1, 2} (DQN default 0.25), ReDo *avoids the performance collapse* at high RR and even *benefits* from higher RR. Holds with n-step=3 returns, with the ResNet architecture, and for DrQ($\epsilon$) at RR ∈ {1,2,4,8} on Atari 100k.
- **§5.2 Learning-rate scaling.** A reduced LR at high RR partially mitigates but does not match ReDo — ReDo is not just "the LR was too high."
- **Headline (Fig. 1).** On 17 Atari games at RR=1, DQN+ReDo beats DQN, DQN+Reset (Nikishin 2022), and DQN+WeightDecay in IQM human-normalized score.

**Relation to the primer & sibling papers.** ReDo operationalizes the "dormant unit" that Abbas observed as a *fraction* into a per-neuron, thresholded, recyclable object; it targets the *dormant count* directly (primer §4a). Versus Nikishin 2022 resets: more surgical, output-preserving, buffer-independent. Versus Dohare continual-backprop: ReDo triggers on *dormancy* ($s_i^\ell\le\tau$), continual-backprop on *low utility* — different selection criteria for the same "recycle the least-useful units" idea. The replay-ratio finding is the direct handshake to D'Oro.

## <a id="redo-appendix-section-by-section-backbone"></a>Appendix: Section-by-Section Backbone

- **Abstract.** Identifies the dormant neuron phenomenon (growing count of inactive neurons hurting expressivity) across algorithms/environments; proposes ReDo to recycle dormant neurons, maintaining expressivity and improving performance.
- **§1 Introduction.** Deep nets as RL function approximators enable scaling but bring RL-specific training pathologies. Scaling laws in supervised learning: performance ↑ with parameters; but in RL, networks *lose* expressivity/target-fitting ability despite over-parameterization (Kumar 2021; Lyle 2021), partly mitigated by perturbing/resetting params (Igl 2020; Nikishin 2022) — but resets are drastic (forget + slow recovery). Central question: *do RL agents use their parameters to full potential?* Track dormant neurons → they grow with training (unlike supervised learning). Contributions: demonstrate the phenomenon; investigate causes + negative effect; propose ReDo; show effectiveness.
- **§2 Background.** MDP $\langle S,A,R,P,\gamma\rangle$; $Q^\pi$, $Q^*$; deep $Q_\theta$; TD loss $L_\theta = Q_\theta(s,a)-Q_\theta^T(s,a)$ with bootstrap target $Q^T(s,a)=[R(s,a)+\gamma\max_{a'}Q_{\tilde\theta}(s',a')]$ and target net $Q_{\tilde\theta}$. **Replay ratio** = gradient updates per env step; higher RR → sample efficiency but training instability/collapse (Nikishin 2022). Two non-stationarities: **input** (online data collection under changing $\pi$) and **target** (bootstrapping off changing $Q_{\tilde\theta}$).
- **§3 The Dormant Neuron Phenomenon.** Def. 3.1 (score $s_i^\ell$, $\tau$-dormant); Def. 3.2 (phenomenon = steadily growing $\tau$-dormant count). Evidence: present in DQN (Fig. 2); **target non-stationarity exacerbates** (CIFAR fixed vs shuffled, Fig. 3); **input non-stationarity not major** (offline RL still shows it, Fig. 4); **dormant neurons remain dormant** (overlap coefficient ↑, Fig. 5; pruning them harmless, Fig. 6); **more updates → more dormant** (RR sweep, Fig. 7); **dormancy makes new-task learning harder** (distillation experiment, Fig. 8).
- **§4 Recycling Dormant Neurons (ReDo).** Algorithm 1 (periodic check, reinit incoming, zero outgoing; $\tau=0$ leaves output unchanged, small $\tau$ slightly changed). Design discussion: alternate recycling strategies (mean-norm scaling ≈ init distribution); alternate init (random outgoing ≈ worse); "Are ReLUs to blame?" — different activation shows mild decrease but phenomenon persists.
- **§5 Empirical Evaluations.** Agents/architectures/environments (DQN 17 games CNN+ResNet; DrQ($\epsilon$) Atari 100k; SAC MuJoCo); Dopamine; $\tau=0.1$; IQM + 95% CIs. §5.1 Consequences for sample efficiency (RR sweep, avoids collapse, benefits from high RR; n-step; ResNet; DrQ). §5.2 Learning-rate scaling (low-LR partial, ReDo better). (Further: related-methods comparison, ablations in appendices.)
- **Appendices.** A: CIFAR non-stationary-target details. B: phenomenon in DrQ($\epsilon$), SAC. C.1: activation-function ablation. C.2: recycling/init-strategy ablations.

---
