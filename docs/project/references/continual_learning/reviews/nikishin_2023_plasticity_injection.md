> **Per-paper review — continual-learning corpus, paper 14 of 22.**
> Extracted from the [master review](../continual_learning_lit_review.md) (§14); content is identical. Field-evolution primer: [[continual_learning_field_evolution]].

# 14. Nikishin et al. 2023 — Deep RL with Plasticity Injection

**PDF:** `docs/project/references/continual_learning/sources/Nikishin et al. 2023 - Deep RL with Plasticity Injection.pdf`
**Venue:** NeurIPS 2023 (also ICLR 2023 Reincarnating RL Workshop) · **arXiv:** 2305.15555
**Authors:** Evgenii Nikishin, Junhyuk Oh, Georg Ostrovski, Clare Lyle, Razvan Pascanu, Will Dabney, André Barreto (DeepMind)

## <a id="inject-phase-1"></a>Phase 1 — Foundational Overview (undergraduate level)

**The plain-language question.** Suppose an RL agent has plateaued — it stopped improving. *Why?* It could be that its network lost plasticity (can't learn anymore), **or** it could be that it simply can't *explore* well enough to find better behavior. These two causes look identical from the outside (flat learning curve) but need opposite fixes. This paper introduces a single minimal trick — **plasticity injection** — that both (a) *diagnoses* which cause it is, and (b) if it's plasticity, *fixes* it, cheaply.

**The trick, intuitively.** At any moment you can *freeze* the current network (so it stops learning but keeps its knowledge) and bolt on a *fresh, randomly-initialized* network whose job is to learn a *correction* to the frozen network's outputs. It's engineered so that at the instant you attach it, the correction is exactly **zero** — the agent's predictions and behavior are completely unchanged. But now there are *fresh* trainable weights, full of plasticity, ready to keep improving. Critically, the total number of *trainable* parameters is kept the same (the old head is frozen), and the predictions are not disturbed — so any change in performance afterward is attributable *only* to the added plasticity, with exploration and capacity held constant.

**As a diagnostic.** Take a plateaued agent, inject plasticity, and compare the training curves with vs. without the injection:
- If performance **jumps** → the agent *was* plasticity-limited (Phoenix: injection doubles the final return; Space Invaders: post-injection learns faster).
- If performance **doesn't budge** → the plateau is *not* about plasticity — it's exploration (Assault: stuck because a new action becomes necessary at score ~2800) or the agent is simply healthy (Robotank: no pathology). Varying *when* you inject even pinpoints *when* plasticity was lost (Phoenix ~25M frames; Space Invaders ~100M frames).

**As a practical tool.** It also saves compute. You can start training with a *small* network (cheap) and, partway through, inject plasticity to effectively *grow* into a larger network — reaching the same score as training the big network from the start, but saving ~20 GPU-hours because the small net was used for the first 50M frames. It also "reincarnates" agents: improve a long-trained agent that's out of plasticity *without* retraining from scratch — beating shrink-and-perturb, resets, and naive width-scaling on aggregate Atari score.

**Initial takeaway.** Plasticity injection is the primer's family-(a) member that *adds* fresh capacity rather than *recycling* old (primer §4a) — a residual/boosting-flavored intervention. Its distinctive value to the project is the **clean diagnostic protocol**: it disentangles "can't learn" from "can't explore," a confound that plagues any interpretation of a flat RL curve — exactly the ambiguity the project's curriculum plateau raises.

## <a id="inject-phase-2"></a>Phase 2 — Graduate-Level Deep Dive

**Design desiderata (what makes it a clean intervention).** Two properties are demanded up front:
1. **Unaffected predictions** — the agent's outputs must be identical immediately after injection, so exploration is not perturbed and cannot confound the analysis.
2. **Preserved trainable-parameter count** — so representational *capacity* (in the trainable sense) is held fixed and cannot confound.

**The construction.** Let the approximator be $h_\theta(x)$ (e.g. an action-value head). At the injection moment, freeze $\theta$ and introduce a fresh random init $\theta'$, kept in **two copies** $\theta_1'$ (trainable) and $\theta_2'$ (permanently frozen), with $\theta_1'=\theta_2'$ at injection. The post-injection prediction is:

$$
\underbrace{h_\theta(x)}_{\text{frozen}} \;+\; \underbrace{h_{\theta_1'}(x)}_{\text{trained}} \;-\; \underbrace{h_{\theta_2'}(x)}_{\text{frozen}}. \tag{1}
$$

**Two properties by construction, derived:**
- **Zero initial change.** At $t=$ injection, $\theta_1'=\theta_2'\Rightarrow h_{\theta_1'}(x)=h_{\theta_2'}(x)$, so the last two terms cancel and (1) reduces to $h_\theta(x)$ — the *exact* pre-injection prediction. No abrupt jump, no induced exploration. (Contrast a hard reset, which *does* jump.)
- **Constant trainable count.** Only $\theta_1'$ is trainable; $\theta$ and $\theta_2'$ are frozen. If $\theta_1'$ has the same shape as the frozen head, the number of *trainable* params is unchanged (total params rise, which costs memory/time but not trainable capacity).

As training proceeds, $\theta_1'$ drifts from $\theta_2'$, and $h_\theta(x)-h_{\theta_2'}(x)$ becomes a *constant bias term* (both frozen) added to the freshly-learning $h_{\theta_1'}(x)$. So the trainable network $h_{\theta_1'}$ learns a *residual* on top of a fixed offset — a residual-learning / boosting view the authors make explicit.

**Applying to only part of the network.** Injecting into *all* parameters would force relearning the entire representation from scratch. Instead, split the net into an encoder $\phi(\cdot)$ (first $k$ layers) and a head $h_\theta(\cdot)$ (remaining layers), and apply (1) only to the head. The encoder $\phi$ is *shared* across the three heads and keeps learning — importantly, gradients from the *frozen* heads are **not** stopped; they still flow into $\phi$, so the encoder is refined by all three output paths. (In experiments: 5-layer conv net, $k=3$ encoder, last 2 layers = head. Target network gets the same intervention.)

**Why not just reset (Nikishin 2022)?** A hard reset of the head abruptly changes predictions → temporary performance drop + induced exploration effect. Analytically that abruptness makes it *impossible* to isolate the plasticity effect from the exploration effect; practically, reset relies on the *replay buffer* to relearn, whereas injection does *not* need the buffer (§5.3 shows this can be decisive).

**Diagnostic protocol (the counterfactual).** For a suboptimal/plateaued agent: save a checkpoint, inject plasticity, and compare curves with vs. without injection — answering the counterfactual *"what would performance be if the network had more plasticity?"* Four canonical outcomes (Fig. 3):
- **Phoenix** — flat baseline; injection *doubles* final return → **catastrophic plasticity loss** (extra interactions weren't translating to learning). Earlier injection helps → plasticity lost ~25M frames.
- **Space Invaders** — baseline still learning; injection accelerates late learning → *gradual* plasticity decline; injection timing doesn't matter until ~100M → onset ~100M.
- **Assault** — plateau *not* fixed by injection → **exploration** limit (a new action becomes necessary at ~2800; App. D).
- **Robotank** — healthy; injection does nothing → no pathology.
The argument is careful/nuanced: plasticity is broadly defined and hard to measure, so "the post-injection agent learns further *because* plasticity was the bottleneck" is the *most likely* interpretation under a design built to hold other factors fixed — not a proof.

**What controls the degree of plasticity loss (Fig. 5).** Measuring the IQM improvement from a 50M-frame injection across regimes, the effect size:
- **increases monotonically with replay ratio (RR)** — more updates burn plasticity faster (consistent with Sokar's RR→dormancy finding);
- **increases monotonically with learning rate (LR)**;
- **decreases with network size** — bigger nets retain plasticity longer;
- **is smaller but still positive with spectral normalization (SN)** applied to the penultimate layer.
These double as *recommendations* for controlling plasticity loss: lower RR/LR, larger nets, normalization.

**Computational-efficiency applications (§5.3).**
- **Reincarnating RL.** Improve an already-trained-out agent without retraining. Across 57 Atari games, injection beats shrink-and-perturb (Ash & Adams), resets (Nikishin 2022), and naive width-scaling in aggregate IQM (paper reports ~20% improvement over other dynamic methods).
- **Dynamic growth to save compute.** Start with a small network; inject plasticity at 50M frames to grow (matching the parameter budget of $\phi + h_\theta + h_{\theta_1'}$). Reaches the same IQM as using the large net from the start while saving ~20 hours of A100 wall-clock, since the small net trains cheaply up to 50M and fewer params are updated afterward — supporting the hypothesis that a large net's *full* capacity isn't needed early, only later for plasticity.

**Illustration of plasticity loss (§3).** A didactic supervised sequence: train a Double-DQN agent on Up'n'Down 200M frames, snapshot policies every 10M, then for each policy build a Monte-Carlo value-regression task (states + MC value estimates) — a sequence of related prediction problems mimicking an online RL agent (Dabney 2021). *"Reset every task"* (random init each) fits every task; *"reset never"* (init from previous task's final params) takes **longer and longer** to fit each subsequent task — the *opposite* of the transfer-learning intuition that related pre-training accelerates. Clean demonstration that warm-starting *degrades* future learnability.

**Limitations.** Extra memory/training time; benefit varies a lot per game (aggregate positive on Atari, individually mixed); can't handle parameter *divergence* (another loss-of-plasticity mode) without more drastic tools; does *not* identify the *causal* factors driving plasticity loss — only diagnoses and mitigates.

**Relation to primer & siblings.** Family-(a) *additive* recycling; conceptually a simplified progressive network (Rusu 2016) motivated by *within-task* plasticity without prediction change; kin to residual learning, boosting, mixtures-of-experts, LoRA (frozen backbone + trainable low-rank addition). Whereas Sokar/ReDo *recycles dormant* units and Nikishin 2022 *resets* layers, injection *appends* fresh trainable capacity while freezing the old. It cites Abbas and Sokar as the "saturation of neurons" line, and notes Lyle 2023's caveat that saturation alone can't fully characterize plasticity loss.

## <a id="inject-appendix-section-by-section-backbone"></a>Appendix: Section-by-Section Backbone

- **Abstract.** Neural nets in deep RL gradually lose plasticity; analysis is hampered by the plasticity–exploration–performance confound. Introduces plasticity injection: increases plasticity without changing trainable-parameter count or biasing predictions. Two uses: (1) diagnostic — if injection helps, the agent was plasticity-limited (identifies plateau-causing Atari envs); (2) efficiency — grow the net dynamically or reincarnate without retraining. Stronger + more efficient than alternatives on Atari.
- **§1 Introduction.** Biological loss of plasticity (aging) has no reason to apply to artificial agents, yet RL agents lose learning ability (Dohare 2021, Lyle 2022, Nikishin 2022). Mechanisms poorly understood; performance confounded by exploration; proxy measures (saturated ReLUs, feature rank) may not capture it (Gulcehre 2022). Contributions: the intervention; complementary existence evidence; a diagnostic protocol; a dynamic-growth efficiency method.
- **§2 Related Work.** *Plasticity in continual learning* (McCloskey-Cohen stability–plasticity dilemma; French forgetting; Ash & Adams warm-start damage; Berariu gradient-noise-reduction conjecture; Dohare reduced train-error-minimization). *Loss of plasticity in deep RL* (Lyle capacity loss; Kumar implicit under-parameterization/rank; Gulcehre weak rank–performance correlation; Sokar/Abbas neuron saturation; Lyle 2023 saturation-insufficient; Nikishin 2022 primacy bias + resets; Igl distillation). *Architectures* (progressive nets Rusu 2016 = closest; MoE/modular nets; growing nets Fahlman-Lebiere, Net2Net; LoRA; residual learning; boosting).
- **§3 An Illustration of Plasticity Loss.** Up'n'Down policy-evaluation sequence; "reset never" fits progressively slower vs "reset every task"; opposite of transfer intuition. Notes RL's distinctive **exploration confounder** (agent shapes its own future data) motivating the injection design.
- **§4 Plasticity Injection.** Desiderata (unaffected predictions; preserved trainable count). Construction Eq. (1) with $\theta$, $\theta_1'$ (trained), $\theta_2'$ (frozen); zero-initial-change + residual-bias derivation. Encoder/head split ($\phi$, $k$ layers) to avoid full representation relearning; gradients flow from frozen heads into $\phi$. Contrast with resets (abrupt change, buffer dependence).
- **§5 Experiments.** §5.1 Setup (Double-DQN, 200M frames, 57 Atari; 5-layer conv, $k=3$; single injection @50M default; IQM, 3 seeds). §5.2 Diagnostic tool (Phoenix/Space Invaders/Assault/Robotank; injection-timing pinpoints onset; Fig. 4 across 57 games; Fig. 5 sensitivity to RR↑/LR↑/size↓/SN). §5.3 Computational efficiency (Reincarnating RL — beats SnP/resets/width-scaling; dynamic growth — matches larger net, saves ~20 A100-hours).
- **§6 Limitations.** Memory/time overhead; per-game variance; parameter divergence unaddressed; no causal identification of plasticity-loss drivers.
- **§7 Discussion & Conclusion.** A clean study of the phenomenon; the proposed version prioritizes simplicity over optimality (a "blueprint"); architecture-agnostic (ResNet blocks, Transformer decoder blocks); open questions — can plasticity loss be solved completely? which properties of fresh nets give high plasticity?

---
