# DreamerV3 implementation reference — architectural-soundness review

One-line summary: the doc's framing, structure, and deviation ranking are mostly correct; recommend three small reframings (make the imagination-as-sample-efficiency argument explicit; promote the mixture-sampling × replay-ratio × sequence-length triple to a single coupled top item; add a one-paragraph "see also" naming TWM / IRIS / TD-MPC2 / SPR-as-auxiliary).

Reviewer: `professor-rl-bayesian-dl`. Reviewed file: [`docs/project/concepts/dreamer_v3_implementation.md`](../concepts/dreamer_v3_implementation.md) (778 lines as of 2026-05-09). This pass is **architectural-soundness only** — code-fidelity (`code-reviewer`) and paper-equation-fidelity (`math-reviewer`) are out of scope.

## Summary

- §1 framing: **PASS** — three-pillar story is present and correctly emphasised; one missing beat (the *why* of imagination-time training).
- §2 paper-canonical structure: **PASS** — leads with RSSM, isolates the V3-specific scale-invariance stack at §2.2, names KL balancing intuitively.
- §3 implementation map structure: **PASS-WITH-WARN** — grouping is sound but the slow-target-critic and the percentile-`Moments` are buried in places where a reader hunting "what stabilises this thing" will not find them on first pass.
- §6 deviation prioritization: **PASS-WITH-WARN** — items 1, 2, 3 are individually correctly ranked but should be presented as a *coupled triple* affecting the world-model training distribution; one omission (KL-balancing preprint-vs-Nature swap is item 7 but its empirical importance is understated).
- Lineage / related-work coverage: **WARN** — DreamerV1 / V2 / PlaNet / Dreamer-4 are present; TWM, IRIS, TD-MPC2, and SPR-as-auxiliary are absent and at least one of them deserves a "see also" mention given the project's reward-head failure history.

**Verdict: ACCEPT-WITH-REFRAMINGS.** The doc is architecturally sound. The reframings below are non-blocking but high-leverage for a reader who has never seen DreamerV3.

---

## §1 framing review

### What the doc gets right

The three-pillar story is present and correctly named in the opening paragraph (L5):

1. RSSM-backed learned world model trained from replay,
2. actor and critic trained entirely *inside* multi-step imagined rollouts,
3. the three robustness tricks (symlog, two-hot, percentile return scaling).

This is the canonical pitch and the doc lands it cleanly. The "one set of hyperparameters across 150+ tasks" claim is in the first sentence — that is the right hook for a domain-naive reader, because it is the single fact that distinguishes V3 from V2 and from every other model-based method on the leaderboard.

The lineage signposting is also correct: PlaNet → DreamerV1 → V2 → V3 is named (L9), and the preprint-vs-Nature distinction is flagged inline in the same sentence with the four substantive deltas (loss-weight schedule, ML framing of critic, replay-value loss, explicit symexp grid). A reader walks away knowing two things they need to know — "this is the *third* Dreamer; there are *two* canonical references; our codebase implements the *preprint* (revealed in §3 / §6)".

### What the doc accidentally undersells

**The cross-task uniformity claim is asserted, not motivated.** The opening sentence says one hyperparameter set spans 150+ tasks, but the doc never connects this *empirical claim* to the *architectural rationale*: the three robustness tricks **exist precisely** to remove the per-domain scaling knobs (reward clipping, return normalisation running stats, return-scale-dependent entropy bonus) that made earlier Dreamer versions need per-domain tuning. A reader seeing §1 today could come away thinking "DreamerV3 is just DreamerV2 with a few tricks bolted on", which misses the engineering thesis that animates the entire paper.

The fix is one sentence inserted between the third-generation-Dreamer claim and the three-pillar list. Suggested phrasing:

> The motivation for the three tricks is not generic regularisation; it is to remove every per-domain knob that earlier Dreamer versions needed (reward clipping, running-normalisation stats, scale-dependent entropy bonus), so that the same hyperparameter set transfers across domains whose return distributions differ by orders of magnitude.

Without this sentence, §6's deviation ranking loses interpretive context: a reader cannot judge "is item 7 (DYN_SCALE = 0.5 preprint vs 1.0 Nature) deliberate engineering or accidental drift?" without knowing that V3's organising principle is *robustness across return scales*.

**The why of imagination is not stated.** §1 says "actor and critic trained entirely *inside* multi-step rollouts of the world model's imagination" but never says **why** imagination-time training is the central design decision. The argument is sample efficiency: real rollouts cost env steps (expensive on Atari, catastrophic on Minecraft, irrelevant on a fast simulator); imagined rollouts are essentially free once the world model exists. A reader who has only seen model-free RL (PPO, DQN, SAC) will read "imagination" as a cute name for something they already understand, and will not realise that this is *the* design lever that makes all of model-based RL work.

§2.4 step 1 ("Take all `B * T` posterior states from the replay batch as start states") implicitly conveys this — every replayed posterior is a **free start state for an imagined rollout**, multiplying the effective gradient signal by `T_imag` per real env step. But §1 should make the argument explicit.

Suggested addition: one sentence in §1 after the three-pillar list:

> The imagination-time training is what gives DreamerV3 its sample efficiency: every replayed posterior state launches a `T_imag = 15` step imagined rollout in latent space, multiplying the actor-critic gradient signal per real env step by roughly `T_imag` without further env interaction.

### Misframings to check (none found)

The doc is **not** mis-framing DreamerV3 as a Bayesian RL method, a planning method, or a model-free method — these are real failure modes in pop-science write-ups but the doc avoids all three. The phrase "model-based reinforcement learning algorithm" appears in the first sentence, the phrase "no posterior, just the prior" at imagination time (L53) makes clear that there is no test-time inference (no MCTS, no MPC, no rollout-based action selection — the actor is a pure feed-forward policy at inference). No "Bayesian" terminology appears anywhere in §1–§2 except in the natural sense of "prior" and "posterior" over the categorical latent, where these terms are correct.

### Verdict for §1

**PASS** with one suggested addition (the *why* of cross-task uniformity, and the *why* of imagination-time training). The current §1 is intuitively correct and architecturally honest; the additions strengthen the framing rather than fixing a defect.

---

## §2 paper-canonical structure review

### Ordering

The §2 ordering is:

- §2.1 RSSM (carried over from V2)
- §2.2 Three robustness tricks (the V3 contribution)
- §2.3 World-model loss (with KL balancing)
- §2.4 Behaviour learning in imagination (with λ-returns)
- §2.5 Other ingredients (unimix, sub-sequence replay, architecture, replay ratio)
- §2.6 Hyperparameter table

**This is the right ordering.** A domain expert teaching DreamerV3 from scratch would lead with the RSSM (because that is the part of the algorithm that the rest of the math attaches to), then immediately distinguish "what's V3-specific" before getting into the loss, exactly because the three tricks (symlog, two-hot, percentile) are needed to read the loss equations. The doc does this.

The §2.2 framing — "Three robustness tricks (the DreamerV3 contribution)" — is the strongest single sentence in §2. It tells a reader who already knows V2 exactly what is new, and it explicitly names "remove every per-domain knob" as the design intent (L26). This is precisely the framing §1 needs to inherit (see §1 review above).

### KL balancing intuition

§2.3 explains KL balancing both formally and intuitively. The intuitive part is at L46:

> The asymmetric weights ... generalise DreamerV2's KL balancing — the prior is updated more aggressively than the posterior is regularised. The Nature version pushes this asymmetry from 5× to 10×.

This is the right intuition (the prior should chase the posterior more than the posterior should be flattened toward the prior — otherwise the latent collapses), and the "5× vs. 10×" framing is the kind of empirical anchor a reader can carry forward. **PASS.**

The free-bits floor is also explained intuitively (L45–46): "the loss is silenced once each KL has fallen below 1 nat, preventing the regulariser from collapsing already-informative latents". This is correct and well-positioned.

### What §2 mis-emphasises (minor)

§2.5 lumps unimix, sub-sequence replay, architecture, and replay ratio together as "other paper-canonical ingredients". One of these is load-bearing in a way the others are not: the **replay ratio** is the single biggest data-efficiency knob in the entire algorithm, and DreamerV3's scaling experiments (Hafner 2023 Fig. 6, Hafner 2025 Fig. 6) sweep it from 1/16 to 64. Treating it as a category-mate of "unimix" understates this.

Recommended reframing: split §2.5 into "§2.5 Categorical machinery (unimix, sub-sequence replay, architecture)" and "§2.6 Compute–data trade-off (replay ratio)" — the latter sub-section being one paragraph naming the 1/16 default for Atari/DMC, the up-to-64 sweep, and the empirical claim that a higher replay ratio buys sample efficiency at the cost of WM saturation on the modal policy. This matters for §6: replay_ratio is the doc's deviation item 2, and its 8× deviation from canonical is far easier to interpret if §2 has primed the reader on what the knob does.

### What §2 does NOT mis-emphasise

The doc does not over-spend on side-features. There is no extended discussion of architecture profiles (XS / S / M / L / XL), no extended discussion of per-domain reward shaping, no extended discussion of CNN encoder details — all correct omissions for a paper-canonical writeup whose downstream §3 maps to a vector-observation MLP-only codebase.

### Verdict for §2

**PASS.** The structural ordering is what a domain expert would write. One minor reframing recommended: pull replay ratio out of §2.5 into its own sub-section to prime §6 item 2.

---

## §3 implementation map structure review

### Grouping

The §3 grouping is:

- §3.1 RSSM
- §3.2 Encoder
- §3.3 Decoder
- §3.4 Heads (reward, continue, actor, critic, slow-target critic, replay-value-loss-or-not)
- §3.5 Loss terms (recon, reward, continue, KL, λ-returns, critic, actor, aggregate)
- §3.6 Buffer + sampling (mixture sub-section)
- §3.7 Numerical-stability tricks (symlog, two-hot, unimix, percentile-`Moments`, free-nats, eps)
- §3.8 Optimiser + schedule

**Most of this is the right grouping.** RSSM-encoder-decoder-heads-losses-buffer-numerics-optimiser is the canonical decomposition of a model-based-RL trainer; a reader chasing any one component knows where to look.

### Two structural concerns

**Concern 1: the slow-target critic is buried at §3.4.5 — and it is the load-bearing stability trick.**

The slow-target critic (Polyak/EMA copy used as the value bootstrap inside λ-returns, while the online critic is used as the actor's baseline) is the single most important stabiliser in DreamerV3's behaviour-learning loop, because without it the bootstrap target moves at the same speed as the regression target and the critic diverges. It is the same reason DDPG / SAC / DQN-with-target-net use a Polyak target — but in DreamerV3 the recurrence amplifies the instability because the bootstrap is at imagination-step T while the regression is the cumulative λ-return from steps 0..T.

The doc correctly describes this at §3.4.5 — "online critic is the **baseline** in the actor advantage; target critic is the **value bootstrap** in λ-return computation" — but the reader has to find §3.4.5 to see it. A reader who jumps from §1's "actor and critic trained inside imagined rollouts" to §3.5.5 (λ-return computation) sees the formula `R^λ_t = r + γc[(1-λ)v + λR^λ_{t+1}]` with `v` simply called "the target critic" and may not realise this is a *different* critic from the one whose loss is computed two sub-sections later in §3.5.6.

Recommended reframing: either (a) promote §3.4.5 to its own §3.5 sub-section ("§3.5 Stability machinery — slow-target critic + percentile-`Moments`") between heads and losses, or (b) at the top of §3.5.5 (λ-return computation) add a one-line forward reference: "uses the slow-target critic for `v` (§3.4.5), not the online critic". Option (b) is cheaper and equally effective.

**Concern 2: the percentile-`Moments` is filed under "numerical-stability tricks" but it is architecturally a *risk-sensitivity* / *return-scaling* mechanism.**

§3.7.4 places the percentile-`Moments` block under §3.7 "numerical-stability tricks" alongside symlog and free-nats. This is technically true — the moments stabilise the policy gradient — but it understates the architectural role. The percentile normalisation is what gives DreamerV3 its **single-entropy-coefficient** property: with `η = 3e-4` fixed, the entropy bonus has comparable strength relative to the policy gradient regardless of whether the domain has dense rewards (returns of order 100s) or sparse rewards (returns of order 1). Without percentile-`Moments`, V3 would need per-domain entropy tuning, which would defeat the entire single-config thesis from §1.

A reader who understands DreamerV3 as "the algorithm that doesn't need per-domain entropy tuning" should be able to find this fact in the implementation map under a heading that signposts it. Recommended: rename §3.7.4 from "`Moments` percentile normalisation" to "`Moments` percentile return scaling (entropy-coefficient invariance)" or similar — the rename costs five words, gains a reader's correct mental model.

### What §3 gets architecturally right

- The decision to put §3.6.2 (mixture sampling — DreamerV4-inspired positive-reward buffer) in its own sub-section, separated from §3.6.1 (paper-canonical buffer container), is correct: the mixture sampler is the single largest structural deviation in the codebase and should be visually separable from the paper-canonical recipe.
- The decision to keep §3.5 monolithic (recon, reward, continue, KL, λ-returns, critic, actor) is correct: these are all summed into a single optimiser step (or two — WM and behaviour) and reading them adjacently makes the loss-balance decisions visible.
- The decision to put §3.4.6 ("Replay-value loss (Nature only) — NOT IMPLEMENTED") inside §3.4 rather than as a §6-only entry is correct: a reader reviewing the heads should see "this paper-spec component is intentionally absent in our build" at the place where they expect it.

### Verdict for §3

**PASS-WITH-WARN.** The grouping is sound. Two small reframings: forward-reference the slow-target critic from §3.5.5, and rename §3.7.4 to surface its architectural (not just numerical) role.

---

## §6 deviation prioritization review

### Top-tier ranking

The current top-8:

1. Mixture sampling + positive-reward buffer
2. `replay_ratio: 0.5` (paper 0.0625, 8× deviation)
3. `sequence_length: 128` (paper 64, 2× deviation)
4. Block-aligned (not uniform) sub-sequence sampling
5. MLP hidden 128 throughout (paper 512+)
6. No replay-value loss (Nature `β_repval = 0.3`)
7. `DYN_SCALE = 0.5` (Nature `β_dyn = 1.0`)
8. WM grad-clip 1000 (Nature 100)

### Concern: items 1, 2, 3, 4 are not independent — they are one coupled deviation in the world-model training distribution

Items 1, 2, 3, and 4 all change *what the world model sees per gradient step*, and they interact multiplicatively:

- Item 1 (mixture sampling) changes which transitions enter a batch (5/16 positive-biased + 5/16 recent-biased + 6/16 uniform).
- Item 2 (replay ratio 0.5) changes how often gradient steps fire per env step (8× more often than canonical).
- Item 3 (sequence length 128) changes how many transitions per gradient step (2× canonical).
- Item 4 (block-aligned sampling) changes which sub-sequence starts are reachable (a coarser grid than canonical uniform sampling).

The total per-iteration WM-update budget in our codebase is therefore `0.5 * 128 = 64` gradient steps × `16 * 128 = 2048` transitions seen per iteration — which the doc itself notes at §6 item 3 ("compounds with item 2"). But the *interaction with item 1* is invisible in the current ranking: a high replay ratio combined with a positive-biased mixture sampler means the WM is seeing the **same set of positive-reward blocks repeatedly** across many gradient steps, in a regime where the canonical V3 buffer would have shown those blocks once or twice.

This matters because the conventional-fixes battery already showed (per the user's task brief) that `replay_ratio = 0.5` fixed the NoPred collapse but did not recover predator-task survival. That negative result is more interpretable if you treat items 1+2+3+4 as a coupled triple: **lowering item 2 alone tests "did the gradient steps overfit?", but does not test "did the positive-biased sampling lock the WM onto a non-representative coverage of the state space?"** The coupling is the load-bearing fact.

Recommended reframing: instead of items 1, 2, 3, 4 as four separate top-tier entries, present them as a single "**Item 1: World-model training distribution (4 coupled deviations)**" with sub-items 1a (mixture sampling), 1b (replay ratio), 1c (sequence length), 1d (block-aligned sampling). The ranking of the rest of the list shifts accordingly. This makes the design of the next experiment ("what if we hold mixture and block-aligned fixed but only vary replay ratio?") read as a natural single-knob test against the coupled-triple view, rather than as "we already tested item 2 in isolation, what's left?".

### Concern: item 7 (DYN_SCALE preprint vs Nature) is empirically more important than its rank suggests

Item 7 — `DYN_SCALE = 0.5` (preprint) vs Nature's `β_dyn = 1.0` — is currently ranked *below* item 6 (no replay-value loss). The doc's own annotation at §3.5.4 reads:

> Per the Nature ablation (Fig. 6), this is the second-most-impactful loss-weight change in the algorithm.

If Hafner-2025's own ablation ranks `β_dyn = 0.5 → 1.0` as the second-most-impactful loss-weight change in the algorithm, then a doc whose entire purpose is "what is non-standard about our DreamerV3" is undershooting by ranking it #7. The replay-value loss (item 6) is documented as a stabiliser for hard-prediction domains — its impact is contingent on whether the domain triggers it. The KL-balancing asymmetry change is a structural change to the latent-dynamics objective that fires every gradient step.

Recommended reframing: swap items 6 and 7 in §6, and lift item 7's "second-most-impactful per Hafner-2025 ablation" annotation from §3.5.4 into the §6 item description so a reader skimming §6 sees the empirical anchor.

### Items the doc may have missed

**Free-nats per-group vs. per-step.** The doc flags this at §3.5.4 as `MATCHES PAPER` (the official implementation also clips per stochastic group). Architecturally this is correct, but the doc should note that the **lower bound on KL** that this implies — `(0.5 + 0.1) * 32 = 19.2` nats per (B, T) entry — is a non-trivial floor on the regularisation pressure the WM experiences. For our small encoder (item 5), this floor is a larger fraction of the total WM loss than it would be for the paper's S-profile encoder. Worth a one-line note in §6 even though it is not strictly a deviation.

**Bin layout symmetry between reward head and critic head.** Both use 255 bins on `symlog([-20, 20])`. The reward head's bin range is appropriate for raw rewards in `[-20, 20]` (which is the entire useful range of our reward shaping). The critic's bin range, however, must accommodate **λ-returns**, which can be on the order of `1 / (1 - γ * λ) = 1 / (1 - 0.997 * 0.95) ≈ 16.4` times the per-step reward. With per-step rewards in `[-1, 1]`, λ-returns sit comfortably inside the symlog-`[-20, 20]` range; with per-step rewards approaching `±1`, λ-returns at the bin edges become possible. Worth checking whether our reward distribution ever pushes λ-returns into the saturation regime — this would be a *configuration-dependent* deviation invisible to the static comparison the doc currently performs.

**No KL warmup / annealing.** The doc says "no LR schedule" but does not flag the absence of KL annealing. Hafner 2023 and Hafner 2025 both use *constant* `β_dyn`, `β_rep` from step 0; this is `MATCHES PAPER`. But it is worth a sentence in §6 noting that KL warmup / β-VAE annealing is a known mitigation for posterior collapse, and that DreamerV3 deliberately does *not* use it (relying on free-nats and KL-balancing instead). Architecturally this is a deliberate design choice worth a reader's attention.

### Tone of deviation flags

The doc's `EXTENSION (deliberate research choice)` vs `MAJOR DEVIATION (deliberate)` vs `MINOR DEVIATION (suspected unjustified)` taxonomy is correct and the assignments are mostly right. Two notes:

- **`KL_SCALE = 1.0` declared but never used (item 9)** is correctly flagged as `MINOR DEVIATION (suspected unjustified)`. The tone is constructive (calls it a "cleanup opportunity").
- **`agent.unimix` YAML key never read (item 10)** is correctly flagged as `MINOR DEVIATION (suspected unjustified)` and the doc explicitly notes the latent-bug risk if a future reader edits the YAML expecting it to take effect. Excellent flagging.

The accusatory-vs-constructive split is well-judged throughout. No reframing needed on tone.

### Verdict for §6

**PASS-WITH-WARN.** Three concrete reframings: (a) coalesce items 1–4 into one coupled-deviation top entry with four sub-items; (b) swap items 6 and 7 to reflect the Hafner-2025-ablation ranking; (c) add a sentence on the implied KL floor (free-nats per-group × stoch_dim = 19.2 nats) as a configuration-sensitive interaction with the small encoder.

---

## Lineage and related-work coverage

### What the doc covers

- DreamerV1 (Hafner 2020, "Dream to Control"): named in references and at §2 ("the discounted-λ-return actor-critic objective from DreamerV1").
- DreamerV2 (Hafner 2021, "Mastering Atari with Discrete World Models"): named in references and at §2.1 / §2.3 (categorical RSSM, KL balancing).
- PlaNet: named at §2 opening sentence as the backbone source for the RSSM lineage.
- Dreamer-4 (Hafner 2025, "Training Agents Inside of Scalable World Models"): named at §3.6.2 and §6 item 1 as the source of the mixture-sampling backport.
- Preprint vs Nature DreamerV3: comprehensively distinguished throughout.

This is the *intra-Dreamer* lineage. **PASS** on intra-lineage.

### What the doc does NOT cover

The following are peer / cousin architectures that a reader trying to place DreamerV3 in the model-based-RL landscape would benefit from at least seeing named:

- **TWM (Robine et al., "Transformer-based World Models Are Happy with 100k Interactions", 2023).** A transformer-based competitor to RSSM. Useful contrast: TWM replaces the GRU+categorical RSSM with a Transformer over a sequence of discrete latent tokens, achieving comparable Atari-100k performance with a different attentional inductive bias. A reader asking "is the GRU-vs-Transformer choice live?" should at least know TWM exists.
- **IRIS (Micheli, Alonso, & Fleuret, "Transformers are Sample-Efficient World Models", 2023).** Discrete-VAE latents + Transformer dynamics. The closest direct competitor to DreamerV3 on Atari-100k. Architecturally instructive because IRIS shows that *categorical latents* (which V3 inherits from V2) are not RSSM-specific — the categorical inductive bias is separable from the recurrent inductive bias.
- **TD-MPC2 (Hansen, Su, & Wang, 2024).** Latent-dynamics model with model-predictive-control-style action selection at *test time*. Useful contrast precisely because it is the *anti-DreamerV3*: V3 trains a policy in imagination and uses it directly at test time; TD-MPC2 trains a value function and runs short MPC rollouts at test time. A reader needs to know "test-time planning vs. amortised policy" is a live design axis.
- **SPR (Schwarzer et al., 2021) / BYOL-Explore-style auxiliary losses.** Self-supervised auxiliary objectives that add a contrastive / predictive auxiliary head to model-based or model-free RL agents. Directly relevant to this project's reward-head failure history: if the reward head is the bottleneck, an SPR-style auxiliary objective on the latent-dynamics representation is the nearest published mitigation — and a reader of the doc finishing §6 should know this avenue exists.

The doc does not have to *include* all four. A "see also" subsection in §7 (or a short paragraph at the end of §6) naming two of the four — recommend **TWM** (transformer cousin) and **SPR-style auxiliary loss** (relevant to project's reward-head failure mode) — would close the gap.

### Verdict on lineage / related-work

**WARN.** Intra-Dreamer lineage is exemplary; cross-architecture peers are entirely absent. Recommend a 3–5 line "see also" paragraph in §7 naming TWM, IRIS, TD-MPC2, and SPR, with one-line each on what each one does differently. Cost: 5 lines. Benefit: a reader who finishes the doc knows where to look next.

---

## Cross-cutting questions

### 1. Does the doc make the right claim about why DreamerV3 was designed for cross-task uniform hyperparameters?

**Partially.** The empirical claim (one config across 150+ tasks) is in §1 sentence 1. The architectural rationale (the three robustness tricks exist *to enable* that uniformity) is in §2.2 ("the three robustness ingredients that make the single-config claim work" at L24). But §1 itself does not connect the two — a reader who only reads §1 sees the claim without the rationale.

Recommended fix: see §1 review above. One sentence inserted between the third-generation claim and the three-pillar list.

### 2. Does the doc properly handle the preprint-vs-Nature distinction?

**Yes — exemplary.** Every loss weight, every grad clip, every equation reference distinguishes preprint from Nature. §1 names both papers. §2 footnotes the four substantive deltas. §3 flags every per-component preprint-vs-Nature deviation. §5.2 has a dedicated row for each. §6 item 6 (no replay-value loss) and item 7 (DYN_SCALE = 0.5) each carry the preprint-vs-Nature annotation.

A reader walking through this doc will know exactly which version our codebase implements and which deltas are deliberate (preprint-canonical) vs. accidental (un-adopted Nature update). **PASS, no changes needed.**

### 3. Is the §6 deviation framing constructive or accusatory?

**Constructive throughout.** The taxonomy `MATCHES PAPER` / `MINOR DEVIATION (justified)` / `MINOR DEVIATION (suspected unjustified)` / `MAJOR DEVIATION (deliberate)` / `EXTENSION (not in paper)` is precisely the right granularity, and the assignments are calibrated:

- Deliberate research choices (mixture sampling, replay ratio, sequence length, MLP width, no replay-value loss, preprint loss weights) are flagged `MAJOR DEVIATION (deliberate)` or `EXTENSION (not in paper)` — the tone signals "we know we differ, here is why".
- Suspected accidents (dead `KL_SCALE`, dead `unimix` YAML, dead `train_steps` YAML, dead unimodal_overrides) are flagged `MINOR DEVIATION (suspected unjustified)` with explicit "cleanup opportunity" language — the tone is repair, not blame.
- Justified minor deviations (the stricter `is_first` reset, the three-LayerNorm bottleneck) are flagged `MINOR DEVIATION (justified)` with the rationale inline.

**PASS, no changes needed.** This is the correct way to write a deviation list for a research codebase.

---

## Recommended additions or reframings

In rough priority order:

1. **§1 — add one sentence** connecting "one set of hyperparameters across 150+ tasks" (the claim) to "the three tricks exist to remove every per-domain knob" (the rationale). Cost: one sentence. Benefit: §6 deviation list becomes interpretable.

2. **§1 — add one sentence** stating that imagination-time training is the sample-efficiency lever (every replayed posterior launches a free `T_imag`-step rollout). Cost: one sentence. Benefit: a reader from a model-free-RL background understands *why* the architecture is shaped the way it is.

3. **§3.5.5 (λ-return computation) — forward-reference the slow-target critic.** One line: "uses the slow-target critic for `v` (§3.4.5), not the online critic". Cost: one line. Benefit: a reader does not have to backtrack to §3.4.5 to disambiguate which critic produces the bootstrap.

4. **§3.7.4 — rename to surface the architectural role.** Suggested: "`Moments` percentile return scaling — entropy-coefficient invariance across return scales". Cost: title rename. Benefit: a reader looking for "why doesn't V3 need per-domain entropy tuning" finds it.

5. **§6 — coalesce items 1–4 into one coupled-triple (or quadruple) top entry** "Item 1: World-model training distribution (4 coupled deviations)" with sub-items 1a–1d. Cost: structural reorganisation of §6 top tier. Benefit: the experiment design implication ("hold three knobs fixed, vary one") becomes legible from §6 alone.

6. **§6 — swap items 6 and 7.** `DYN_SCALE = 0.5` (preprint) vs. Nature's 1.0 is rated by Hafner-2025's own ablation as the second-most-impactful loss-weight change in the algorithm; the replay-value loss is documented as a stabiliser for hard-prediction domains (contingent impact). Lift the "second-most-impactful per Hafner-2025 ablation" annotation from §3.5.4 into the §6 item description. Cost: a swap and a sentence move. Benefit: deviation ranking matches the published ablation.

7. **§6 — add one sentence on the implied KL floor.** Free-nats per-group × `stoch_dim = 32` × `(β_dyn + β_rep) = 0.6` gives a lower bound of 19.2 nats per (B, T) on the KL contribution — a non-trivial floor on the regularisation pressure the WM experiences, especially relevant when combined with the small encoder (item 5). Cost: one sentence in §6. Benefit: a configuration-sensitive interaction surfaces.

8. **§7 — add a 3–5 line "see also" paragraph** naming TWM (transformer cousin), IRIS (closest direct V3 competitor on Atari-100k), TD-MPC2 (test-time-planning anti-V3), and SPR-style auxiliary losses (relevant to reward-head failure history). Cost: 5 lines. Benefit: a reader finishing the doc knows where to look next for cross-architecture context.

None of these are blocking. All eight together would extend the doc by roughly 15 lines and would not change a single deviation flag or implementation-map entry.

---

## Verdict

**ACCEPT-WITH-REFRAMINGS.**

The doc is architecturally sound. §1 framing is correct on the three pillars and the lineage; §2 leads with the right backbone and isolates the V3-specific contribution at §2.2; §3's grouping is sensible; §6's deviation taxonomy is well-calibrated and constructive. The eight reframings above are non-blocking enhancements rather than fixes for defects.

The single most important reframing is **#5** (coalescing §6 items 1–4 into a coupled-triple top entry): the current §6 reads as if four independent knobs each contribute one unit of deviation, but architecturally they form one coupled deviation in the world-model training distribution, and treating them coupled is what makes the next experiment design legible.

The single most important addition is **#8** (the "see also" paragraph): a doc that names the four close cousins (TWM, IRIS, TD-MPC2, SPR-as-auxiliary) gives the reader an exit ramp for further reading, and it costs five lines.

Recommend the doc author incorporate items 1, 2, 5, 8 (the four highest-leverage reframings) before stamping the doc as finalised; the remaining four (items 3, 4, 6, 7) are nice-to-have polish and can be deferred.

---

## Next steps

- **Doc author / `senior-developer`**: incorporate reframings 1, 2, 5, 8 (highest leverage). Items 3, 4, 6, 7 are polish; defer if time-pressed.
- **`literature-curator`** or **`literature-reviewer`**: if reframing 8 is adopted, the "see also" paragraph needs three short citations (TWM, IRIS, TD-MPC2; SPR is in the project's existing knowledge base). Either fetch via `WebSearch` to confirm citation strings or pull from the existing Dreamer lit review under `docs/project/references/Dreamer/`.
- **`experiment-designer`**: if reframing 5 is adopted, the coupled-triple framing implies a cleaner experiment design: hold mixture-sampling and block-aligned sampling fixed at canonical, vary replay ratio and sequence length together — the existing `dreamer_v3_rr06.yaml` is the first leg of that two-by-two but not the full design.
- **`professor-rl-bayesian-dl`** (this agent): no further follow-up needed unless the author wants a fuller writeup of the SPR-style-auxiliary-loss avenue (cross-link with the project's reward-head failure history and the existing FiLM / precision-modulation work).
