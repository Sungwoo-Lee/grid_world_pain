# Modality-aware observation encoding and reconstruction loss in our Dreamer world model

> **One line:** Declare the 7 sensory modalities as 7 observation keys using the mechanism our reference implementation already has — it is provably loss-neutral, costs almost nothing, and buys per-modality diagnostics — but be clear that it buys you *no* encoder inductive bias at all, and do not reweight the modalities on the first attempt, because the original DreamerV3 trains a 1-dimensional hunger signal alongside a 12,288-dimensional camera image at equal per-number weight and still solves Minecraft.

---

## 1. Question, in plain language

We are giving our Dreamer agent's internal world model a sense of *which sensor a number came from*. Right now the agent receives its body-and-surroundings state as one undifferentiated list of 27 numbers, and the world model treats all 27 identically. Our other agent (the recurrent-PPO one) already knows the 27 numbers are really 7 different senses — hunger, felt pain, seen threat, smell, bumping into things, its own last movement, and vision — and we want the world model to know that too.

Two design questions came with the request. **First**, DreamerV3 squashes every input number through a gentle compression function called *symlog* before feeding it in. Should we keep doing that to all seven senses, or exempt the ones whose values are already small and bounded? **Second**, the world model is trained partly by asking it to reproduce the 27 numbers it just saw. Reproducing 8 vision numbers currently counts eight times as much as reproducing the single hunger number, even though hunger matters far more for survival. Should we correct that imbalance?

**Verdict.** Keep symlog on everything (it is already nearly a no-op for small values, and where it is not a no-op it happens to nudge the balance in the direction we want). Split into 7 keys using the built-in mechanism, not a bespoke architecture. Do **not** reweight on the first attempt. And understand that the built-in mechanism gives us the bookkeeping we want but *not* the modality-aware encoder — that part is genuinely bespoke, and it is the lower-confidence half of the proposal.

---

## 2. What the reference implementations actually do — verified, not recalled

Three things were checked directly against source, because the whole recommendation turns on them.

### 2.1 sheeprl (our port's upstream)

| Component | File | Behaviour |
|---|---|---|
| `MLPEncoder.forward` | `vendor/sheeprl/sheeprl/algos/dreamer_v3/agent.py:150` | `torch.cat([symlog(obs[k]) for k in self.keys], -1)` → **one shared MLP over the concatenation**. Multiple keys are concatenated and lose their identity immediately. |
| `MLPDecoder.__init__` | same file, `:274` | `self.heads = nn.ModuleList([nn.Linear(dense_units, mlp_dim) for mlp_dim in self.output_dims])` → shared trunk, **one linear head per key**. |
| observation loss | `vendor/sheeprl/sheeprl/algos/dreamer_v3/loss.py:61` | `-sum([po[k].log_prob(observations[k]) for k in po.keys()])` — **unweighted sum across keys**. |
| per-key distribution | `vendor/sheeprl/sheeprl/algos/dreamer_v3/dreamer_v3.py:152-162` | `MSEDistribution` for image keys, `SymlogDistribution` for vector keys. |
| aggregation inside a key | `vendor/sheeprl/sheeprl/utils/distribution.py:189-190` | `agg="sum"` → `distance.sum(dims)`. **Sum over that key's dimensions.** |

So the user's stated prior fact is **confirmed**: upstream is *flat concatenating encoder + per-key decoder heads*.

### 2.2 The official DreamerV3 (danijar/dreamerv3, `main`)

Checked because sheeprl is a re-implementation and could have drifted. It has not.

- **Encoder** (`dreamerv3/rssm.py:216-219`): `nn.DictConcat(vspace, 1, squish=nn.symlog)` — every vector key concatenated, symlog applied uniformly. The **only** per-key differentiation on the encoder side is in `embodied/jax/nets.py:488-496`: a key declared as a *discrete* space is one-hot encoded and **not** symlogged; every continuous key is symlogged.
- **Decoder** (`dreamerv3/rssm.py:298-303`): shared MLP trunk, then `DictHead(spaces, outputs)` with `outputs = {k: 'categorical' if v.discrete else 'symlog_mse'}` — per-key head, with the **output distribution family chosen per key by space type**.
- **Aggregation** (`embodied/jax/heads.py:91`): `outs.Agg(output, len(space.shape), jnp.sum)` — **sum** over the key's feature dims.
- **Per-key weighting** (`dreamerv3/agent.py:80-83`):
  ```python
  scales = self.config.loss_scales.copy()
  rec = scales.pop('rec')
  scales.update({k: rec for k in dec_space})
  ```
  A per-observation-key loss-scale dictionary **exists structurally** — and is then **overwritten so every key gets the same value**, `rec = 1.0` (`dreamerv3/configs.yaml:86`). Final combination at `agent.py:240`: `loss = sum([v.mean() * self.scales[k] for k, v in losses.items()])`.

**Conclusion:** the mechanism for per-modality reconstruction weighting is present in the official code and is deliberately not used. This is a designed uniformity, not an oversight.

### 2.3 The decisive precedent: DreamerV3 on Minecraft

From the paper's environment appendix (p. 21 of *Mastering Diverse Domains through World Models*), the Minecraft observation is:

> a 64×64×3 image, a vector with one entry for each of the game's over 400 items, the vector of maximum inventory counts since episode begin, a one-hot vector indicating the equipped item, and **scalar inputs for the health, hunger, and breath levels**.

That is roughly 13,000 reconstruction terms, of which **health and hunger are one term each** — a dimensional imbalance of about $10^4{:}1$, versus our worst case of $8{:}1$. All terms are summed unweighted. This configuration is the first system to collect diamonds in Minecraft from scratch. Health and hunger are the *exact structural analogue* of our satiation and interoceptive-nociception channels: 1-dimensional, behaviourally critical, drowned by dimensionality.

This is the single most useful empirical fact in this memo. It says our $8{:}1$ worry, taken on its own terms, is a $10^4{:}1$ problem that has already been survived.

---

## 3. Q1 — symlog placement and per-modality treatment

### 3.1 What symlog is for (from the paper)

$$\operatorname{symlog}(x) \doteq \operatorname{sign}(x)\ln(|x|+1), \qquad \operatorname{symexp}(y) \doteq \operatorname{sign}(y)\bigl(e^{|y|}-1\bigr)$$

with the prediction loss $\mathcal L(\theta) = \tfrac12\bigl(f(x,\theta) - \operatorname{symlog}(y)\bigr)^2$ and readout $\hat y = \operatorname{symexp}(f(x,\theta))$ (Eqs. 1–2, DreamerV3).

The paper's own statement of purpose, verbatim (p. 4):

> "The symlog function compresses the magnitudes of both large positive and negative values. … **Symlog approximates the identity around the origin so that it does not affect learning of targets that are already small enough.**"

and the *Nature* (2025) restatement adds the gradient-side reason:

> "transforming vector observations using the symlog function … **prevents large inputs and large reconstruction gradients**, further stabilizing the trade-off with the representation loss."

So symlog has two jobs: (i) scale-robustness for inputs of unknown magnitude, and (ii) bounding the reconstruction-gradient magnitude so it does not destabilise the balance against the KL/free-bits term. Job (ii) matters even when job (i) is vacuous.

The ablation (Fig. D.1, `NoObsSymlog`) removes both the encoder squash and the symlog decoder target; it *slows* learning on proprioceptive control. Note that the ablation is joint — the paper does not separate "symlog the input" from "symlog the target".

### 3.2 Is symlog harmful on bounded / one-hot / small values? No.

Since symlog is elementwise, "before vs. after splitting" is numerically a **no-op**; the real question is per-modality treatment. Local behaviour:

$$\operatorname{symlog}'(x) = \frac{1}{1+|x|}, \qquad \operatorname{symlog}(x) = x - \tfrac{1}{2}x|x| + O(x^3)$$

| $x$ | 0 | 0.1 | 0.5 | 1 | 2 | 3 |
|---|---|---|---|---|---|---|
| $\operatorname{symlog}(x)$ | 0 | 0.0953 | 0.405 | 0.693 | 1.099 | 1.386 |
| relative compression | 0% | 4.7% | 19% | 31% | 45% | 54% |

For a one-hot channel $x\in\{0,1\}$, symlog gives $\{0, 0.693\}$: **still perfectly separated, still exactly invertible, contrast reduced 31%**. There is no loss of representational range and no distortion of small values — a monotone bijection cannot destroy information, and for $|x|\ll 1$ it is the identity to first order. For our normalised interoceptive drives in $[0,1]$ it is close to the identity throughout. **Non-issue, as suspected.**

### 3.3 The one real effect — and it points the way we want

Symlog *is* a mild reweighting of the reconstruction loss, because it compresses the target. A channel whose raw values reach 3 (our aggregated visual channel counts, `sense_visual` at `src/environment/sensor.py:146` sums per-entity property vectors over visible cells) is compressed to 1.386, so its worst-case squared error is $\approx(1.386)^2 = 1.92$ instead of $9$ — a $4.7\times$ shrink in that channel's share of the gradient. A satiation channel in $[0,1]$ is shrunk only $2.1\times$. So **symlog already partially offsets the visual-vs-interoceptive imbalance the user is worried about**, in the right direction, for free.

That is an argument *against* exempting bounded modalities: exempting the one-hot-ish visual channels from symlog would *increase* their share of the reconstruction gradient relative to the interoceptive scalars.

### 3.4 Recommendation — Q1

> **Keep symlog applied uniformly to the raw 27-vector before any branching, exactly as now. Do not exempt any modality; do not add per-modality normalisation.**
>
> **Confidence: high (0.9).**

Rationale, in priority order: (1) it is a no-op for our small/bounded channels by the paper's own design statement; (2) where it is not a no-op it moves the balance toward the interoceptive scalars; (3) it is the only configuration the reference hyperparameters (free bits = 1 nat, $\beta_{\text{rep}}=0.1$) have ever been tuned against, and DreamerV3's whole selling point is that this bundle works untouched; (4) per-modality normalisation would reintroduce the running-statistics non-stationarity that symlog exists to avoid ("normalizing targets based on running statistics introduces non-stationarity into the optimization", p. 3).

**One legitimate per-modality refinement, borrowed from the official code, not invented here:** if a modality is genuinely *categorical* — our proprioception channel is a one-hot of the previous action (`jax.nn.one_hot(state.last_action, params.action_dim)`, `src/environment/sensor.py:332`) — the official implementation declares it a discrete space, **skips symlog**, and gives it a **categorical (cross-entropy) decoder head** instead of symlog-MSE. That is defensible and precedented. It is also a strictly optional refinement; symlog-MSE on a one-hot vector is what sheeprl does and it works.

**What would falsify the recommendation:** a run in which the observation loss on the visual channels plateaus at a floor while the interoceptive channels are reconstructed to near-zero error *and* survival stalls — that would suggest the compression is costing us visual fidelity. Per-key loss curves (§4) make this observable, which is the main reason to split.

---

## 4. Q2(a) — Is splitting into 7 keys loss-neutral? Yes, exactly.

**Claim.** Let the 27 dimensions be partitioned into disjoint index sets $I_1,\dots,I_7$ with $\bigcup_k I_k = \{1,\dots,27\}$. Then the summed per-key negative log-probability equals the single-key negative log-probability.

**Proof.** Our port's observation distribution (`src/algorithms/dreamer_srl/loss.py:83-86`) is

$$\log p_k(x_{I_k}) = -\sum_{i \in I_k} \rho\bigl(\hat x_i - \operatorname{symlog}(x_i)\bigr), \qquad \rho(u) = u^2 \cdot \mathbb{1}[u^2 \ge \tau], \ \tau = 10^{-8}$$

Then

$$\mathcal L_{\text{obs}}^{\text{split}} = -\sum_{k=1}^{7}\log p_k(x_{I_k}) = \sum_{k=1}^{7}\sum_{i\in I_k}\rho(\cdot) = \sum_{i=1}^{27}\rho(\cdot) = \mathcal L_{\text{obs}}^{\text{flat}} \qquad \blacksquare$$

Three points that make this stronger than the generic "yes if same family and unit variance" intuition:

1. **It does not depend on the distribution family.** It holds for *any* per-dimension-factorised likelihood aggregated by summation — which is what `agg="sum"` and `Agg(..., jnp.sum)` both are. It would fail only for `agg="mean"` (then splitting *does* reweight, by $1/d_k$ per key) or for a non-diagonal covariance.
2. **The tolerance clamp survives.** $\tau$ is applied elementwise *before* the sum, so partitioning cannot move a term across the threshold.
3. **It is not bit-identical, only numerically indistinguishable.** Seven partial `sum`s then a `sum` associates floating-point additions differently from one 27-term `sum`. In fp32 the discrepancy is $O(27\varepsilon) \approx 3\times10^{-6}$ relative, non-systematic. If a bit-parity regression test exists on the observation loss, expect it to need a tolerance, not a fix.

**Answer to (a):** the split is loss-neutral. The flat-vs-hierarchical ablation stays unconfounded **on the loss side**. It does *not* stay unconfounded on the parameter-initialisation side if the encoder is branched (§6), which is the real confound to watch.

---

## 5. Q2(b) — Should modalities be weighted?

### 5.1 What is standard

| System | Multi-key vector obs? | Per-key reconstruction weight |
|---|---|---|
| DreamerV3 (Hafner et al. 2023; *Nature* 2025) | Yes — DMC proprio, BSuite, Minecraft (`inventory`, `equipped`, `health`, `hunger`, `breath` + image) | **No.** Mechanism exists in code, hard-set uniform at $\beta_{\text{pred}}=1$. Fixed hyperparameters across 8 benchmark suites. |
| sheeprl DreamerV3 (our upstream) | Yes, `mlp_keys` list | **No.** Plain `sum` over keys. |
| DayDreamer (Wu et al. 2022, real robots: proprioception + images) | Yes | **No** per-key weight reported. |
| Multi-Modal World Model for Physical Robot Interactions (arXiv 2304.11193, visuo-tactile) | Yes | **Yes** — explicit $\alpha, \beta$ modality loss weights, both set to 1.0 in the reported experiments, with reweighting flagged as future work. |
| Combining Reconstruction and Contrastive Methods for Multimodal Representations in RL (arXiv 2302.05342) | Yes | Argues reconstruction in multimodal RL is **dominated by the high-dimensional modality** and replaces reconstruction with a contrastive objective rather than reweighting it. |

**Standard practice is unweighted.** The only systems that reweight are outside the Dreamer line, and the one paper that takes the imbalance most seriously (2302.05342) concludes that the fix is to change the *objective*, not to tune a scalar — which is a much larger commitment than we want here.

### 5.2 Why unweighted is defensible in our specific setup

The reconstruction loss is not the only pressure shaping the latent state. The prediction loss is

$$\mathcal L_{\text{pred}}(\phi) = -\ln p_\phi(x_t\mid z_t,h_t) - \ln p_\phi(r_t\mid z_t,h_t) - \ln p_\phi(c_t\mid z_t,h_t)$$

The reward term and the continue term are **full-weight, 1-dimensional-target** losses on quantities that are direct functions of satiation and nociception in our environment. They are also *not* squared errors — the reward head is a 255-bin two-hot cross-entropy, whose gradient magnitude is decoupled from the signal scale by construction. So the behaviourally critical scalars already receive first-class, scale-independent gradient pressure through a channel that has nothing to do with the 27-dim reconstruction budget. This is precisely why Minecraft's health/hunger scalars are not lost at $10^4{:}1$.

### 5.3 The failure mode any reweighting must avoid

DreamerV3's stability rests on a balance the paper calls out explicitly: free bits clip the dynamics/representation KL **below an absolute floor of 1 nat**, and $\beta_{\text{rep}}=0.1$ is small. The *Nature* text says symlog on vector observations exists partly to keep reconstruction gradients from destabilising "the trade-off with the representation loss."

The free-bits floor is **absolute, not relative**. Therefore any reweighting that changes $\|\mathcal L_{\text{obs}}\|$ silently re-tunes the KL-vs-reconstruction balance, and does so in a regime where the reference hyperparameters were never validated. Naïvely equalising the seven modalities (i.e. taking a mean within each key, $w_k = 1/d_k$) shrinks the total observation loss by roughly $27/7 \approx 3.9\times$, which is a large, unintended KL upweighting.

**If we reweight at all, the weights must be magnitude-preserving:**

$$\mathcal L_{\text{obs}} = \sum_{k=1}^{K} w_k \sum_{i\in I_k}\rho(\cdot), \qquad w_k = \frac{D/K}{d_k}, \qquad \sum_k w_k d_k = D = 27$$

With $K=7$, $D=27$: each modality contributes $27/7\approx 3.86$ "dimension-equivalents", the total stays 27, and the balance against KL, reward, and continue is untouched. Concretely $w \approx \{$Satiation 3.86, Intero-Nocic. 3.86, Extero-Nocic. 3.86, Olfaction 0.77, Collision 0.77, Proprioception 0.64, Visual 0.48$\}$.

### 5.4 Recommendation — Q2(b)

> **First attempt: unweighted ($w_k = 1$ for all $k$), matching both reference implementations. Do not reweight.**
>
> **Confidence: high (0.85)** that unweighted is not the binding constraint on our Dreamer's performance.
>
> **Reserve** the magnitude-preserving equal-per-modality weighting of §5.3 as a *single, pre-specified* second arm, to be used only if the per-key diagnostics (§8) show the interoceptive channels are actually badly reconstructed. Confidence that this arm would help *if* that diagnostic fires: moderate (0.5). Confidence that it helps *unconditionally*: low (0.2).

**What would falsify "unweighted is fine":** per-key observation-loss curves in which the Satiation / Interoceptive-Nociception / Extero-Nociception keys plateau at a normalised error near their marginal variance (i.e. the decoder is predicting the channel mean and nothing more) while the Visual and Collision keys drop steadily. That is the signature of a starved modality, and it is exactly what splitting into 7 keys makes visible. Absent that signature, reweighting is tuning noise.

**Against reweighting, one further consideration specific to our project:** [[evaaa_vs_gridworld_algorithm_inversion]] establishes that our grid world does not need a world model to reach current performance, and that DreamerV3's advantage here is in *experience efficiency*, not final score. A reconstruction-weighting knob is a low-ceiling intervention on a component that is not the bottleneck. Spending the "one configuration that succeeds" budget on it is poor expected value relative to the encoder question.

---

## 6. Q2(c) — Does a per-modality decoder head change anything?

### 6.1 sheeprl's per-key head is an exact reparameterisation — a literal no-op

Single 27-dim head: $\hat y = W z + b$, $W\in\mathbb R^{27\times512}$, $b\in\mathbb R^{27}$.
Seven per-key heads: $\hat y_k = W_k z + b_k$, $W_k\in\mathbb R^{d_k\times512}$, $\sum_k d_k = 27$.

Stacking $W = [W_1; \dots; W_7]$, $b = [b_1;\dots;b_7]$ recovers the single head exactly. Same function class, same parameter count ($27\times512 + 27$), and — critically — **the same gradients**, because

$$\frac{\partial \mathcal L}{\partial W_k} = \frac{\partial \mathcal L}{\partial \hat y_k}\, z^\top$$

is precisely the corresponding row-block of $\partial\mathcal L/\partial W$. A dense linear layer's output rows are *already* independent: row $i$'s weights receive gradient only from output $i$. Splitting a linear layer along its output dimension changes nothing about conditioning, capacity allocation, or gradient flow.

Initialisation is also identical in distribution: PyTorch's `nn.Linear` uses $\mathcal U(-1/\sqrt{\text{fan\_in}}, +1/\sqrt{\text{fan\_in}})$ with $\text{fan\_in}=512$ in both cases. Only the random-number draw *ordering* differs.

> **Answer to (c): sheeprl's per-key decoder head changes nothing beyond the RNG stream. Not parameter count, not gradient conditioning, not capacity allocation. Confidence: very high (0.95).**

The decoder's capacity allocation happens entirely in the **shared trunk**, which the per-key head does not touch.

### 6.2 What *would* change learning dynamics

A genuinely **branched** decoder — per-modality *hidden* layers before the head, not just a per-modality output row-block — is a real architectural change, and it cuts both ways:

- **For:** each modality gets private nonlinear capacity; a modality with idiosyncratic structure is not forced to share features with an unrelated one.
- **Against:** a private branch for a 1-dim modality receives gradient from exactly one scalar error term. In fp32 with a shared 512-unit trunk feeding a private 512-unit branch, that branch is trained by $1/27$ of the reconstruction signal and will be badly under-determined. This *worsens* the imbalance the branching was meant to fix.
- **Against, structurally:** the shared bottleneck is doing useful work. It forces the latent state $z_t$ to carry features that serve *all* modalities, which is exactly the cross-modal binding a world model should learn. Branching removes that pressure.

The same asymmetry applies on the encoder side, with the sign flipped: a per-modality encoder branch *is* the inductive bias we want (it lets the network learn "these 5 numbers are one sensor" without having to discover it), and the encoder branch for a 1-dim modality is cheap and well-conditioned because it is a *feature extractor*, not a predictor — it receives gradient from the entire downstream world model, not from one scalar loss term.

**Practical consequence:** if we branch, **branch the encoder, keep the decoder trunk shared with per-key heads.** That is the asymmetric design, and it is not what "mirrored decoder" in the request implies. Flagging this explicitly as a disagreement with the stated plan.

---

## 7. The headline verdict — native multi-key vs. bespoke hierarchical encoder

This is the question the request said would be most useful, so it gets a direct answer.

**The native multi-key mechanism and the modality-aware encoder are two different changes, and the native mechanism delivers only one of them.**

| What we want | Native `mlp_keys` gives it? |
|---|---|
| Per-modality reconstruction-loss curves (diagnostics) | **Yes**, fully, free |
| A per-modality loss-weight hook, if we ever want it | **Yes**, one line |
| Per-modality output distribution family (categorical for the one-hot proprioception key) | **Yes**, precedented in official DreamerV3 |
| Loss-neutrality vs. current runs | **Yes**, provably (§4) |
| Deviation risk from the validated reference | **Zero** — it is the reference's own configuration |
| **A modality-aware encoder inductive bias** | **No. Zero. None.** The encoder concatenates and runs one MLP. |

So: **use the native mechanism — but do not expect it to be the thing that makes Dreamer work.** It is bookkeeping. It is *excellent* bookkeeping, and it should be done first because it is free and it tells us whether the bespoke half is even needed. But if we ship only the native path and call it "modality-aware encoding", we will have shipped a relabelling.

**Recommended sequencing.**

1. **Step 1 (do now, high confidence, ~zero risk).** Declare the 7 modalities as 7 observation keys via the native mechanism. Unweighted. Symlog on all continuous keys. Optionally declare proprioception discrete → categorical head. Expected performance change: **none** (that is the point — §4 proves it). Expected diagnostic gain: large. This is the change that tells us whether Step 2 is worth doing.

2. **Step 2 (bespoke, lower confidence, only if warranted).** A per-modality encoder branch: $e_k = \mathrm{MLP}_k(\operatorname{symlog}(x_{I_k}))$ for each modality, then fuse $\bigl[e_1;\dots;e_7\bigr] \to \mathrm{MLP}_{\text{shared}}$. Keep the decoder trunk shared (§6.2). This is genuinely off-reference and carries real risk; it is also the only part that supplies the inductive bias motivating the whole exercise.

Given the constraint that we do **not** need a controlled comparison and are hunting for *one configuration that succeeds*, Steps 1 and 2 can be shipped together in a single run to save wall-clock — Step 1 is provably inert, so bundling costs no interpretability. What must **not** be bundled in is the reweighting of §5.3; that one changes the loss magnitude and would confound everything.

**Confidence that the branched encoder (Step 2) materially improves survival:** **low-to-moderate (0.35).** Honest statement of why: DreamerV3's flat vector encoder handles Minecraft's heterogeneous $10^4$-dimensional multi-key observation without branching, and our 27-dim observation is a far easier case. A 512-unit MLP over 27 symlogged inputs has ample capacity to learn the block structure from data. The strongest argument *for* branching is not capacity but **sample efficiency of discovering the block structure**, and that argument is weaker in our low-dimensional setting than in a high-dimensional one. This memo does not oppose the change — it is cheap and the parity argument with our recurrent-PPO agent is legitimate — but the implementer should not carry a strong prior that it is the fix.

---

## 8. Empirical signatures — what to log, and what each pattern means

The whole point of Step 1 is to make these observable. Per-key observation loss, normalised by each key's marginal variance in symlog space (so keys are comparable):

$$\tilde{\mathcal L}_k = \frac{\frac{1}{d_k}\sum_{i\in I_k}\mathbb E\bigl[(\hat x_i - \operatorname{symlog}(x_i))^2\bigr]}{\frac{1}{d_k}\sum_{i\in I_k}\operatorname{Var}\bigl[\operatorname{symlog}(x_i)\bigr]}$$

| Signature | Interpretation | Action |
|---|---|---|
| $\tilde{\mathcal L}_k \to 0$ for all $k$ | World model reconstructs everything; obs loss is not the constraint. | Look elsewhere for the bottleneck. |
| $\tilde{\mathcal L}_k \approx 1$ for the interoceptive keys, $\ll 1$ for Visual/Collision | Starved modality — decoder predicts the channel mean. **This is the one signature that justifies reweighting.** | Try the magnitude-preserving weights of §5.3. |
| $\tilde{\mathcal L}_k \approx 1$ for *all* keys | Latent state carries no observation information at all — a KL/free-bits collapse, not a modality problem. | Check the dynamics/representation KL against the 1-nat floor; reweighting will not help. |
| Total obs loss falls but survival does not improve | Reconstruction is being solved but is not the binding constraint on control. Consistent with the finding in [[evaaa_vs_gridworld_algorithm_inversion]] that this environment does not require a world model. | Stop investing in the world-model observation path. |
| Dynamics KL pinned at the 1-nat floor throughout | Free bits disabled the regulariser entirely; obs-loss changes are being absorbed by an inactive term. | Any reweighting is a no-op; do not interpret it. |

Also worth logging: the **fraction of reconstruction gradient norm attributable to each key**, $\|\partial\mathcal L_{\text{obs}}/\partial \hat x_{I_k}\|^2 / \|\partial\mathcal L_{\text{obs}}/\partial\hat x\|^2$. This measures the *actual* imbalance, which is what §5 is arguing about, rather than the dimensional imbalance, which is only a proxy for it. If the Visual key turns out to hold only 25% of the gradient rather than the naïve $8/27 = 30\%$ (symlog compression, §3.3), the imbalance is even smaller than feared.

---

## 9. Relation to the four causes of the v8 null

This memo touches the return-estimator / representation side only indirectly. It speaks to the "the world model is not learning a useful state" family of explanations, and offers a cheap discriminating test (§8) rather than a fix. It does **not** speak to the modulator-identifiability or reward-signal causes. Nothing here should be read as a candidate explanation for the v8 null; it is a world-model hygiene and instrumentation memo.

---

## 10. Failure modes to flag to the implementer

1. **Changing $\|\mathcal L_{\text{obs}}\|$ silently re-tunes the KL balance.** The free-bits floor is absolute (1 nat). Any weighting scheme must preserve the total. This is the single most likely way to break a working configuration.
2. **`agg="mean"` instead of `agg="sum"` inside a key** would make splitting *not* loss-neutral, reweighting every key by $1/d_k$ implicitly. Our port uses `sum` (`loss.py:86`); do not change it as a side effect of the refactor.
3. **A bit-parity regression test on the observation loss will fail by $\sim10^{-6}$** after splitting, from floating-point associativity, not from a bug (§4, point 3).
4. **Branching the decoder as well as the encoder starves the 1-dim branches** (§6.2). Mirror-symmetry is aesthetically appealing and here it is the wrong call.
5. **Exempting bounded modalities from symlog would increase their gradient share** (§3.3) — the opposite of the stated intent.
6. **Bundling the reweighting with the encoder change** makes the run uninterpretable even under the relaxed "find one that works" standard, because a failure would not tell us which half to revert.

---

## 11. Open questions

- Is our reward head's gradient actually carrying the interoceptive signal, as the Minecraft argument assumes? That is checkable directly (reward-loss curve vs. observation-loss curve) and would substantially firm up the §5 recommendation.
- Does the official implementation's **categorical head for discrete keys** matter for our one-hot proprioception channel? Cheap to test, precedented, and orthogonal to everything else here.
- Symlog's encoder-squash and decoder-target roles are only ablated *jointly* in the paper (Fig. D.1). If we ever care about which one carries the benefit, that is an unanswered question in the literature, not just in our project.

---

## 12. References

**In the project library** (`docs/project/references/Dreamer/sources/`):
- Hafner et al. 2023, *Mastering Diverse Domains through World Models* — symlog (§"Symlog Predictions", pp. 3–4, Eqs. 1–2), world-model losses (Eqs. 4–5, p. 5), `NoObsSymlog` ablation (Fig. D.1 + p. 20), Minecraft observation spec (p. 21), fixed hyperparameters (Table W.1, p. 38).
- Hafner et al. 2025 (*Nature*), *Mastering diverse control tasks through world models* — the gradient-stability rationale for symlog on vector observations (p. 1).

**Named but not in the library — recommend acquiring:**
- Wu et al. 2022, *DayDreamer: World Models for Physical Robot Learning* — closest precedent for multi-sensor Dreamer on real hardware; no per-key weighting.
- arXiv 2302.05342, *Combining Reconstruction and Contrastive Methods for Multimodal Representations in RL* — the strongest published statement of the exact worry in Q2(b), with a different answer (change the objective, don't reweight).
- arXiv 2304.11193, *Multi-Modal World Model for Physical Robot Interactions* — the one multimodal world model that carries explicit per-modality loss weights $\alpha,\beta$.
- Seo et al. 2023, *Multi-View Masked World Models for Visual Robotic Manipulation* (MV-MWM) — multi-view rather than multi-modal, but the closest Dreamer-line treatment of heterogeneous inputs.

**Source read directly (not a paper):** `danijar/dreamerv3` at `main` — `dreamerv3/rssm.py` (Encoder/Decoder), `dreamerv3/agent.py:80-83,240` (loss scales), `embodied/jax/nets.py:467-500` (`DictConcat`), `embodied/jax/heads.py:44-130` (`DictHead`, `Head`), `dreamerv3/configs.yaml:86` (`loss_scales`).

**Cross-links:** [[evaaa_vs_gridworld_algorithm_inversion]] — establishes that this environment does not require a world model to reach current performance, which bounds the expected value of every recommendation above.

---

## 13. Next steps

- **`senior-developer`** — Step 1 (native 7-key declaration) is a contained change: the observation breakdown is static (`get_observation_breakdown`), so the split is a slice-by-static-offsets in the encoder and a dict-of-heads in the decoder. Step 2 (branched encoder) is invasive and should be a separately revertible commit. Please do **not** implement §5.3 reweighting in this pass. Note the §10.3 tolerance issue for any observation-loss parity test.
- **`experiment-designer`** — if per-key diagnostics fire the "starved modality" signature in §8, the magnitude-preserving weighting of §5.3 is the pre-specified second arm; it needs its own seeds and must not be bundled with the encoder change.
- **`professor-dl-theory`** — the branched-encoder-with-shared-fusion design in §7 Step 2 is a conditional/modular architecture question; the fusion-layer design (concat vs. gated vs. attention over modality embeddings) is theirs, not mine.
- **`math-reviewer`** — §4's loss-neutrality proof is the load-bearing claim; worth an independent check against `src/algorithms/dreamer_srl/loss.py:83-86` and `vendor/sheeprl/sheeprl/utils/distribution.py:177-192`.
