# DreamerV3 implementation reference — paper-equation fidelity review

> Review target: [`docs/project/concepts/dreamer_v3_implementation.md`](../project/concepts/dreamer_v3_implementation.md) (778 lines, sections §1–§8)
> Reviewer: `math-reviewer` (Phase 4 of pilot)
> Sources of truth: Hafner et al. 2023 (preprint, arXiv:2301.04104) and Hafner et al. 2025 (Nature, doi:10.1038/s41586-025-08744-2). Cross-checked against published code at github.com/danijar/dreamerv3 (`rssm.py`, `embodied/jax/heads.py`, `configs.yaml`).

## Summary

- **Forward pass**: 31 distinct paper-equation / hyperparameter / definitional claims walked through, comparing the doc's paper-spec lines against the cited PDFs. **8 mismatches found**, of which **3 are BLOCKERs** (wrong bin-layout claim; wrong "MATCHES PAPER" classification on the use of `target_critic` for the λ-return bootstrap; wrong paper-default for WM Adam $\epsilon$). Two are CONCERNs (wrong horizon for Nature; wrong "Atari/DMC default 0.0625" universalisation). Three are NITs (the lit-review's preprint vs. Nature horizon conflation, the doc's per-stoch-group free-bits arithmetic, and the LayerNorm vs. RMSNorm slip for Nature).
- **Reverse pass**: re-walked Hafner 2023 §"DreamerV3" (pages 3–8), Tables A.1, B.1, W.1, Appendix C–E (pages 18–22); Hafner 2025 §"World model learning", §"Critic learning", §"Actor learning", §"Robust predictions", §"Methods" (pages 2–9), Extended Data Tables 4 and 5. **Three load-bearing equations / hyperparameter ingredients in the papers are missing or misrepresented in the doc**: the Nature optimiser swap (LaProp + AGC, not Adam + global-norm); the Nature unified learning rate ($4{\cdot}10^{-5}$, not the split $10^{-4}/3{\cdot}10^{-5}$); and the Nature replay-capacity change ($5{\cdot}10^6$, not "typically 1M"). Two procedure steps are missing: where the Nature critic distributional loss and replay-value term sit in the gradient flow, and where the percentile-EMA ordering goes (computed on the imagined-rollout returns BEFORE the actor gradient — not "outside the gradient" the way our code stages it).
- **Symmetry check**: 3 paper equations cited without a code pair (Nature replay-value Eq., Nature distributional ML critic Eq., percentile-EMA Eq.); 0 code refs without a paper pair (the doc is exhaustive on the active code path).
- **Verdict**: **ROUTE-BACK-TO-AUTHOR** — 3 BLOCKERs need correcting before the doc is sound as a paper-canonical reference. F1, F2, F3 below each propagate to multiple sections (§3, §5, §6) and to the §6 deviations ranking; F2 in particular elevates one item from "MATCHES PAPER" to TOP-TIER deviation, which would change the §6 priority list materially.

---

## Equations under review

For ground-truth referencing throughout the findings, here are the canonical equations from the source papers as I extracted them.

### Eq. P1 — Symlog squashing (preprint Eq. 1–2 / Nature pp. 4)

$$\text{symlog}(x) \doteq \text{sign}(x)\,\ln(|x|+1), \qquad \text{symexp}(x) \doteq \text{sign}(x)\,(\exp(|x|) - 1)$$

### Eq. P2 — RSSM components (preprint Eq. 3 / Nature page 2)

$$
\begin{aligned}
\text{Sequence model:}\quad & h_t = f_\phi(h_{t-1}, z_{t-1}, a_{t-1}) \\
\text{Encoder:}\quad & z_t \sim q_\phi(z_t \mid h_t, x_t) \\
\text{Dynamics predictor:}\quad & \hat z_t \sim p_\phi(\hat z_t \mid h_t) \\
\text{Reward predictor:}\quad & \hat r_t \sim p_\phi(\hat r_t \mid h_t, z_t) \\
\text{Continue predictor:}\quad & \hat c_t \sim p_\phi(\hat c_t \mid h_t, z_t) \\
\text{Decoder:}\quad & \hat x_t \sim p_\phi(\hat x_t \mid h_t, z_t)
\end{aligned}
$$

### Eq. P3 — World-model loss (preprint Eq. 4–5; Nature uses $\beta_{\text{dyn}}=1$ instead of $0.5$)

$$\mathcal L(\phi) \doteq \mathbb{E}_{q_\phi}\Big[\sum_{t=1}^T \big(\beta_{\text{pred}}\,\mathcal L_{\text{pred}}(\phi) + \beta_{\text{dyn}}\,\mathcal L_{\text{dyn}}(\phi) + \beta_{\text{rep}}\,\mathcal L_{\text{rep}}(\phi)\big)\Big]$$

with
$$
\begin{aligned}
\mathcal L_{\text{pred}}(\phi) &\doteq -\ln p_\phi(x_t\mid z_t,h_t) - \ln p_\phi(r_t\mid z_t,h_t) - \ln p_\phi(c_t\mid z_t,h_t) \\
\mathcal L_{\text{dyn}}(\phi) &\doteq \max\!\big(1,\,\text{KL}\big[\text{sg}(q_\phi(z_t\mid h_t,x_t)) \,\|\, p_\phi(z_t\mid h_t)\big]\big) \\
\mathcal L_{\text{rep}}(\phi) &\doteq \max\!\big(1,\,\text{KL}\big[q_\phi(z_t\mid h_t,x_t) \,\|\, \text{sg}(p_\phi(z_t\mid h_t))\big]\big)
\end{aligned}
$$

Hyperparameters: $\beta_{\text{pred}}{=}1$, $\beta_{\text{dyn}}{=}0.5$ (preprint) / $1$ (Nature), $\beta_{\text{rep}}{=}0.1$.

### Eq. P4 — λ-return (preprint Eq. 7; Nature page 3, identical formula)

$$R^\lambda_t \doteq r_t + \gamma c_t\big[(1-\lambda)\,v_\psi(s_{t+1}) + \lambda R^\lambda_{t+1}\big], \qquad R^\lambda_T \doteq v_\psi(s_T)$$

with $\gamma = 0.997$, $\lambda = 0.95$. **Critical**: $v_\psi$ is the **fast (online) critic**, not the EMA-target critic (Nature page 3 verbatim: "regularizing the critic towards predicting the outputs of an exponentially moving average of its own parameters... allows us to compute returns using the **current critic network**"; preprint Appendix D.2 confirms by ablating "SlowTarget" and finding it offers no benefit over fast-critic + self-EMA).

### Eq. P5 — Critic loss (preprint Eq. 10 / Nature page 3 ML form)

Preprint:
$$\mathcal L_{\text{critic}}(\psi) \doteq -\sum_{t=1}^T y_t^\top \ln p_\psi(\cdot\mid s_t), \qquad y_t = \text{sg}\big(\text{twohot}(\text{symlog}(R^\lambda_t))\big)$$

Nature (re-framed as max-likelihood of the return distribution, plus the new replay-value term):
$$\mathcal L(\psi) \doteq -\sum_{t=1}^T \big[\beta_{\text{val}}\,\ln p_\psi(R^\lambda_t \mid s_t) + \beta_{\text{repval}}\,\ln p_\psi(R^{\lambda,\text{replay}}_t \mid s^{\text{replay}}_t)\big]$$

with $\beta_{\text{val}}=1$, $\beta_{\text{repval}}=0.3$ (Nature only).

### Eq. P6 — Twohot encoding (preprint Eq. 9)

For a continuous target $x$ in **symlog space** and bin centres $b_1,\dots,b_K$ also in **symlog space**:

$$\text{twohot}(x)_i \doteq \begin{cases} |b_{k+1} - x| / |b_{k+1} - b_k| & i = k \\ |b_k - x| / |b_{k+1} - b_k| & i = k+1 \\ 0 & \text{else} \end{cases}, \quad k \doteq \sum_{j=1}^{|B|} \delta(b_j < x)$$

Point-estimate readout: $\hat y \doteq \text{symexp}\big(p_\psi(\cdot\mid s_t)^\top B\big)$.

**Critical**: the bin grid $B$ is `linspace(-20, +20, 255)` **in symlog space** (preprint Eq. 8: $B = (-20\dots+20)^\top$ with the readout $\text{symexp}(p^\top B)$ — only consistent if $B$ is in symlog space). Equivalently, **raw-space bin centres** are `symexp(linspace(-20, 20, 255))`, spanning $\pm(\exp(20)-1) \approx \pm 4.85{\cdot}10^8$. Nature page 4 makes this explicit: "$B \doteq \text{symexp}([-20\dots+20])^\top$".

### Eq. P7 — Actor loss with percentile-EMA scaling (preprint Eq. 11–12 / Nature page 3)

$$\mathcal L(\theta) \doteq -\sum_{t=1}^T \Big[\,\text{sg}\big((R^\lambda_t - v_\psi(s_t))/\max(1, S)\big)\,\ln\pi_\theta(a_t\mid s_t) - \eta\,\mathbb H[\pi_\theta(a_t\mid s_t)]\,\Big]$$

with $\eta = 3{\cdot}10^{-4}$ and

$$S \doteq \text{EMA}\!\big(\text{Per}(R^\lambda, 95) - \text{Per}(R^\lambda, 5),\;\rho{=}0.99\big)$$

(Nature explicit; preprint is the same up to the $\max(1,\cdot)$ clamp and the entropy-sign convention).

### Eq. P8 — Unimix (preprint §A.1 / Nature page 3)

$$p_{\text{unimix}}(\cdot) = (1 - \alpha)\cdot \text{softmax}(\text{logits}) + \alpha \cdot \text{Uniform}, \qquad \alpha = 0.01$$

Applied to RSSM posterior $q_\phi(z_t\mid h_t, x_t)$, RSSM prior $p_\phi(z_t\mid h_t)$, and actor $\pi_\theta(a_t\mid s_t)$.

---

## Forward-pass findings (paper-equation fidelity)

### F1 — BLOCKER. Twohot bin layout claim is wrong: doc's bins span $\pm 20$ in raw space; paper bins span $\approx \pm 4.85{\cdot}10^8$.

**Doc claims** (§2.2):
> "The preprint says 'equally spaced over [−20, +20]' — disambiguated by the Nature paper as `B = symexp(linspace(-20, 20, 255))`, i.e. equally spaced *in symlog space*, exponentially spaced *in raw return space*."

And (§3.7.2):
> "Bins are linearly spaced **in symlog space** per the comment at `util.py:25–29`, which matches the official-implementation convention. ... `MATCHES PAPER` for the symlog-spaced bin layout (preprint convention; equivalent to Nature's `B = symexp(linspace(-20, 20, 255))` framing)."

**Paper canonical** ([Eq. P6](#eq-p6--twohot-encoding-preprint-eq-9), preprint Eq. 8 + Nature page 4): bin grid is `linspace(-20, +20, 255)` **in symlog space** (equivalently, raw-space bin centres are $\text{symexp}(\text{linspace}(-20, 20, 255))$ spanning approximately $\pm 4.85{\cdot}10^8$).

**Code at `src/models/dreamer_v3_util.py:67–71`** ([from_twohot](../../src/models/dreamer_v3_util.py)):
```python
bottom = symlog(jnp.array(min_v))   # symlog(-20)  ≈ -3.045
top    = symlog(jnp.array(max_v))   # symlog(+20)  ≈ +3.045
bucket_vals = jnp.linspace(bottom, top, num_buckets)
```

The code's bin centres in symlog space are `linspace(-3.045, +3.045, 255)`. After `symexp(...)`, the raw-space bin centres span only **$[-20, +20]$** — fully **eight orders of magnitude smaller** than the paper.

**Cross-check against Hafner's published code** (`embodied/jax/heads.py`, lines 87–97, fetched live):
```python
half = jnp.linspace(-20, 0, (self.bins - 1) // 2 + 1, dtype=f32)
half = nets.symexp(half)
bins = jnp.concatenate([half, -half[:-1][::-1]], 0)
```

The Hafner code's `half` is in symlog space from $-20$ to $0$, then `symexp` maps it to raw values from $-\exp(20)+1 \approx -4.85{\cdot}10^8$ to $0$. Mirrored: full bins span $\pm 4.85{\cdot}10^8$ in raw space.

**Severity**: this is a $4{\cdot}10^7\!\times$ range mismatch in raw-return support. For the project's grid-world reward magnitudes (homeostasis bonus $|r| \lesssim 1$, episodic injury $|r| \sim$ few units) the practical effect is small, because clipping at $\pm 20$ in raw space never triggers. **But the doc's "MATCHES PAPER" classification is wrong**, and any future scaling exercise that hits returns above $\pm 20$ would silently saturate. The deviation is `MAJOR DEVIATION (suspected unjustified)` — not a `MATCHES PAPER` and not even mentioned in §6.

**Suggested fix**:
- Rewrite §2.2 disambiguation to: "the linspace is over `[-20, +20]` *in symlog space*; raw-space bin centres are `symexp(linspace(-20, 20, 255))` spanning approximately $\pm 4.85{\cdot}10^8$".
- Reclassify §3.7.2 from `MATCHES PAPER` to `MAJOR DEVIATION (suspected unjustified)` — our $\pm 20$ raw-space support is dramatically narrower than paper $\pm 4.85{\cdot}10^8$.
- Add a §6 entry alongside item 5 (small width) at the TOP-TIER level: "Twohot bin support narrowed from $\pm 4.85{\cdot}10^8$ to $\pm 20$ in raw space" — for low-magnitude reward domains the effect is benign, but it is a documented deviation.
- Update §5.2 row "Two-hot `min_v`/`max_v`": current "matches paper" → flag as "DEVIATION — paper bin range is `symexp([-20,+20])`, our bins clip raw values at $\pm 20$".

---

### F2 — BLOCKER. λ-return bootstrap uses the EMA-target critic; both preprint and Nature explicitly use the fast critic.

**Doc claims** (§3.4.5):
> "Hafner 2023 preprint uses a self-EMA regulariser on the critic; Hafner 2025 Nature uses a separate **'slow target critic'** updated by EMA at every gradient step, **used as the value bootstrap for λ-returns**. ... `MATCHES PAPER` (Nature wording)."

**Paper canonical** (Hafner 2025, page 3, verbatim):
> "Because the critic regresses targets that depend on its own predictions, we stabilize learning by regularizing the critic towards predicting the outputs of an exponentially moving average of its own parameters. This is similar to target networks used previously in reinforcement learning **but allows us to compute returns using the current critic network**."

And the equation immediately below (Nature page 3):
$$\mathcal L(\psi) \doteq -\sum_{t=1}^T \ln p_\psi(R^\lambda_t \mid s_t), \qquad R^\lambda_t \doteq r_t + \gamma c_t\big((1-\lambda)v_t + \lambda R^\lambda_{t+1}\big)$$

with $v_t$ being the value of the **same** critic $\psi$ (no subscript distinguishing target from online).

**Cross-check against the preprint** (Appendix D.2, preprint page 20): the figure D.2 ablation includes a `SlowTarget` variant defined as: "Instead of using the fast critic for computing returns and training it towards the slow critic, use the slow critic for computing returns" — and the appendix caption reports: "**Using the slow critic for computing targets offers no benefit over using the fast critic and regularizing it towards its own EMA**."

Therefore: **both preprint and Nature** compute λ-returns using the **fast critic** $v_\psi$, and the slow EMA only enters as a regularisation target on the loss.

**Code at `src/models/dreamer_v3_trainer.py:367, 388, 411`**:
```python
val = from_twohot(self.target_critic(next_feat))   # line 367 (modulated branch)
val = from_twohot(self.target_critic(next_feat))   # line 388 (vanilla branch)
v_start = from_twohot(self.target_critic(start_feat))  # line 411
```

All three bootstrap-value sites for the λ-return computation use `self.target_critic`, **not** `critic`. The online critic is used only as the **baseline** in the actor advantage (line 433 `baseline = from_twohot(v_pred_logits)`) and as the **prediction** in the critic loss (line 427).

**Severity**: this is one of the three components the preprint explicitly ablates as the **`SlowTarget` variant** that "offers no benefit". Our code is running the ablation, not the published recipe. The doc's `MATCHES PAPER (Nature wording)` classification is incorrect for both papers. This belongs in §6 TOP-TIER (alongside the replay-value loss item §6 #6) because:
1. It is a deliberate-or-unintentional structural deviation, NOT covered in any current §6 item.
2. It changes the gradient-flow topology: the bootstrap value is one update behind the critic, with `target_critic` updated as `0.98 t + 0.02 c` per step.

**Suggested fix**:
- Rewrite §3.4.5 to say: "Hafner 2023 preprint AND Hafner 2025 Nature both compute λ-returns using the **fast (online) critic**, regularising it toward an EMA of itself only inside the critic loss (preprint Appendix D.2 explicitly ablates the slow-target alternative as offering no benefit). Our code uses `target_critic` for bootstrap values — the ablated `SlowTarget` variant of the preprint."
- Reclassify the deviation flag from `MATCHES PAPER (Nature wording)` to `MAJOR DEVIATION (suspected unjustified — matches the ablated SlowTarget recipe)`.
- Add a new §6 TOP-TIER entry: "λ-return bootstrap uses target_critic, not online critic — corresponds to the preprint's SlowTarget ablation."
- §3.5.5 caller text ("`v_start = from_twohot(target_critic(start_feat))`") is currently presented as canonical without flag — also needs the deviation cross-link.

---

### F3 — BLOCKER. Preprint Adam $\epsilon$ is **asymmetric** (1e-8 WM / 1e-5 AC), not "uniform 1e-5" as the doc's paper-spec line claims.

**Doc claims** (§3.8.1 paper-spec line):
> "Adam, no weight decay, no LR schedule, **`eps = 1e-5` uniform**. World-model LR `1e-4`, actor LR `3e-5`, critic LR `3e-5`."

And (§2.6 hyperparameter table row):
> "| Adam epsilon | 1e-5 (uniform across heads) | Nature supplementary |"

And (§3.7.6 deviation flag):
> "Adam `eps` split: `1e-8` for WM, `1e-5` for actor and critic ... Hafner-2025 uses `1e-5` everywhere; this is `MINOR DEVIATION (suspected unjustified)` for the WM optimiser."

And (§5.2 row):
> "| WM Adam eps | 1e-8 | trainer.py:96 | **1e-5** | MINOR DEVIATION |"

And (§6 item 15):
> "WM Adam `eps = 1e-8` (paper 1e-5 uniform). MINOR DEVIATION (suspected unjustified)."

**Paper canonical** (Hafner 2023 Table W.1, p. 38, verbatim):
- World Model section: "Adam epsilon | $\epsilon$ | $10^{-8}$"
- Actor Critic section: "Adam epsilon | $\epsilon$ | $10^{-5}$"

So the **preprint** uses an **asymmetric** $\epsilon$ split — exactly the same as our code (1e-8 for WM, 1e-5 for AC). This is **`MATCHES PAPER`**, not a deviation, against the preprint baseline that the rest of the doc uses.

The Nature paper does change this — but not toward "1e-5 uniform". Nature Extended Data Table 5 reports the optimiser as **LaProp(\epsilon = 10^{-20})**, fully replacing Adam. So neither paper version supports the doc's "1e-5 uniform" claim. The 1e-5 uniform attribution to "Nature supplementary" in §2.6 is a fabrication / misread.

**Severity**: BLOCKER because:
1. The §3.8.1 paper-spec line is the canonical reference for what the paper requires; downstream readers will believe "paper says uniform 1e-5".
2. §6 item 15 is a false-positive deviation that would be removed; the §6 list shrinks by one and the priority order may shift.
3. §2.6 row is also wrong, and §5.2 row.

**Suggested fix**:
- §3.8.1 paper-spec: "Adam, no weight decay, no LR schedule. **Preprint Table W.1: WM $\epsilon = 10^{-8}$, actor/critic $\epsilon = 10^{-5}$ (asymmetric split).** Nature replaced Adam entirely with LaProp $(\epsilon = 10^{-20})$ + AGC(0.3) — so the matching baseline is the preprint."
- Remove the "minor deviation" flag at §3.7.6 (3rd bullet) and §5.2 WM-eps row → reclassify as `MATCHES PAPER (preprint)`.
- Remove §6 item 15.
- §2.6 row: replace "1e-5 (uniform across heads) | Nature supplementary" with "$\epsilon = 10^{-8}$ (WM) / $10^{-5}$ (AC) preprint Table W.1; Nature uses LaProp $\epsilon = 10^{-20}$ — different optimiser entirely".

---

### F4 — CONCERN. Nature imagination horizon is $H=15$, not $16$.

**Doc claims** (§2.6 hyperparameter table):
> "| Imagination horizon `T_imag` | 15 (preprint) / **16** (Nature) | both |"

And (§5.2 row):
> "| `HORIZON` | 15 | `trainer.py:21` | 15 (preprint) / **16** (Nature) | matches preprint |"

**Paper canonical**:
- Hafner 2023 Table W.1 (p. 38): "Imagination horizon | $H$ | **15**"
- Hafner 2025 Extended Data Table 5 (p. 17, image-rendered): "Imagination horizon | $H$ | **15**"

Both papers have $H = 15$. The doc's "$T_\text{imag} = 16$ (Nature)" appears to come from misreading the prose at preprint page 6 / Nature page 3:
> "To consider rewards beyond the prediction horizon $T = 16$, the critic learns to predict the return..."

Here $T = 16$ is the **batch / sequence prediction horizon**, NOT the imagination horizon $H$ of the actor-critic loop. (Note: this is also inconsistent with both papers' Table where batch length $T = 64$ — the prose's "$T = 16$" is also itself a mis-reuse of $T$ for what is clearly meant to be a separate "prediction horizon for the critic". This is an artefact of the paper's notation; both Tables W.1 / Ext.5 are the source of truth and both say $H = 15$.)

**Severity**: misleading but not behaviour-changing — our code uses `HORIZON = 15` and the deviation flag in §5.2 says "matches preprint", which is correct on the value. The CONCERN is that the doc invents a non-existent Nature deviation.

**Suggested fix**:
- §2.6 row: "| Imagination horizon `T_imag` | **15 (both versions, per preprint Table W.1 and Nature Ext. Data Table 5)** | both |"
- §5.2 row: paper-default column "**15 (both versions)**".
- Optionally add a note: "The preprint and Nature prose both use `T = 16` ambiguously for the *critic's prediction horizon*, but this is distinct from the imagination horizon $H = 15$ in both Tables."

The lit-review companion at line 1585 (`imagination horizon T=16`) carries the same conflation; if the lit-review is the source the doc is following, that's the upstream root cause. Worth flagging back to `literature-curator`.

---

### F5 — CONCERN. "Atari/DMC default 0.0625" is wrong: DMC default in the preprint is 0.5 — equal to our value.

**Doc claims** (§2.6 hyperparameter table, §3.8.2, §6 item 2):
> "Replay ratio (default) | 1/16 ≈ 0.0625 (Atari/DMC) | both"

> "Hafner-2023 default for **Atari 200M and DMC**: `1/16 ≈ 0.0625`."

> "**`agent.replay_ratio: 0.5` (paper 0.0625).** `MAJOR DEVIATION (deliberate)`. ... 8× the paper-canonical Atari/DMC default of 0.0625."

**Paper canonical** (Hafner 2023 Table A.1, p. 18):

| Benchmark | Train Ratio | Action Repeat |
|---|---|---|
| DMC Proprio | **512** | 2 |
| DMC Vision | **512** | 2 |
| Crafter | 512 | 1 |
| BSuite | 1024 | 1 |
| Atari 100K | 1024 | 4 |
| Atari 200M | **64** | 4 |
| DMLab | 64 | 4 |
| Minecraft | 16 | 1 |

Footnote (verbatim): "The train ratio is the number of replayed steps per policy steps rather than environment steps, and thus unaware of the action repeat."

Convert to gradient-step / env-step (using batch shape $B \times L = 16 \times 64 = 1024$):

- Atari 200M: $\text{ratio} = 64 / 1024 / 4 = 1/64 \approx 0.0156$ grad-steps per env-step (further down than 0.0625; the doc's "1/16" doesn't match this either).
- DMC Proprio/Vision: $\text{ratio} = 512 / 1024 / 2 = 0.25$ grad-steps per env-step.

Or, using the Nature-style "replay ratio = replayed-steps / env-step" convention (Nature page 9 footnote, verbatim: "Dividing the replay ratio by the time steps in a minibatch and by action repeat yields the ratio of gradient steps to environment steps"):

- DMC has train ratio 512 and action repeat 2 → 512/2 = 256 replayed-steps per env-step → in our YAML semantics where `replay_ratio = grad-steps/env-step ≈ replay-step/(L) / action_repeat`: $256 / 64 = 4$ in pure replay-step-per-env-step, then divided by minibatch length $L=64$: $4 / 64 = 0.0625$. So **DMC ≈ 0.0625** in our YAML semantics, equal to `dreamer_v3_rr06.yaml`. By the same convention DMC's actor-critic update rate = 0.0625 grad-step / env-step.

**The arithmetic depends on what `replay_ratio` means in our YAML**. If it is "grad-steps per `Ratio.__call__` invocation interval" (which trainer.py code suggests with `train_steps = ratio_scaled_updates(global_step // num_steps)`), then it's 8× faster than DMC default. If it's "grad-steps / env-step", then DMC = 0.0625 and our 0.5 is 8× DMC.

Either way: **the doc conflates Atari and DMC into a single "0.0625" default, but the preprint Table A.1 has Atari 200M at the lowest end (train ratio 64 → effective replay 0.0156) and DMC at the middle (train ratio 512 → effective replay 0.25 in grad/env, or 0.0625 in normalized units). They are NOT both 0.0625.**

**Severity**: CONCERN — the §6 item 2 deviation classification ("8× the Atari/DMC default") oversimplifies into a single number that hides the per-domain spread (factor-of-8 between DMC and Atari200M in the preprint). The numerical magnitude of our deviation depends on which benchmark the project is comparing against; for DMC-style task density our `replay_ratio = 0.5` is `8x` over the YAML-normalized 0.0625, but for Atari-200M it's `32x`. The §6 prioritisation as "TOP-TIER" item is still likely correct, but the magnitude estimate is benchmark-dependent.

**Suggested fix**:
- §2.6 row: "Replay ratio | **0.0625 (Atari 200M, normalized; DMC effective 0.25 grad/env in preprint Table A.1; varies $\times 16$ across benchmarks)**".
- §3.8.2 paper-spec: "Hafner 2023 Table A.1 train-ratios per benchmark: Atari 200M 64, DMC 512, BSuite/Atari100K 1024, Minecraft 16. In Nature `replay_ratio = replay-steps / env-step` semantics: Atari 200M $\approx$ 0.0156 grad/env, DMC $\approx$ 0.25 grad/env."
- §6 item 2: replace "8× the paper-canonical Atari/DMC default" with "8× the preprint DMC default of 0.5 grad/env (or $32\times$ Atari 200M's 0.0156)".

---

### F6 — NIT. Free-bits floor description: doc says "per stoch group"; code (and Hafner-published code) apply per-state.

**Doc claims** (§3.5.4 paragraph after the code excerpt, and §3.7.5):
> "Free-nats is applied **per-stoch-group AFTER summing over the 32 discrete classes**, then clipped, then averaged across batch+time. So the clipping floor is 1.0 nat per `(B, T, stoch_dim=32)` entry — i.e. the lower bound of `dyn_kl + rep_kl` summed over groups would be `(0.5 + 0.1) * 32 = 19.2` nats per `(B, T)`."

> §3.7.5 "Code spec ... `dyn_kl/rep_kl` going into `max(., 1.0)` is shape `(B, T, stoch_dim)` after the per-class sum, so the floor is per stoch group."

**Code at `src/models/dreamer_v3_trainer.py:230–249`**:
```python
def kl_div_categ(p_logits, q_logits):
    return jnp.sum(p_dist * (p_log - q_log), axis=-1)   # (B, T, stoch=32, discrete=32) → (B, T, stoch=32)
...
dyn_kl = jnp.sum(dyn_kl, axis=-1)              # (B, T, stoch=32) → (B, T)
rep_kl = jnp.sum(rep_kl, axis=-1)
dyn_kl = jnp.maximum(dyn_kl, FREE_NATS)        # clip applied to (B, T) tensor
```

Free-nats is applied to a `(B, T)` tensor, **after summing over both the discrete-class axis and the stoch-group axis**. The code-reviewer's F1 finding is correct on this. The doc's `(B, T, stoch_dim)` shape claim is wrong.

**Cross-check against published Hafner code** (`dreamerv3/rssm.py`, fetched live):
```python
out = embodied.jax.outs.OneHot(logits, self.unimix)
out = embodied.jax.outs.Agg(out, 1, jnp.sum)
return out
```
The `Agg(out, 1, jnp.sum)` wrapper aggregates the `OneHot.kl(...)` over **1 axis** (the stoch-group axis) **before** the `jnp.maximum(dyn, free_nats)` clip in the loss. So Hafner's reference clips per `(B, T)` state too. **Our code matches Hafner's published implementation** on the axis of free-bits application.

**Severity**: NIT — the doc's verbal description and arithmetic are wrong, but the deviation classification `MATCHES PAPER` is **correct** at the level of "per-state free-bits clipping". The lower-bound arithmetic is wrong: at floor `FREE_NATS = 1.0` and shapes `(B, T)`,
$$\mathcal L_{\text{kl}}^{\min} = 0.5 \cdot 1.0 + 0.1 \cdot 1.0 = 0.6 \;\text{nats (averaged over } B,T\text{)}$$
not 19.2 nats. **Concur with code-reviewer's F1 on the description; reject F1's suggested deviation reclassification** because Hafner's published code agrees with our axis choice.

**Suggested fix** (corroborates code-reviewer F1 with a math angle):
- §3.5.4 after-code prose: "Free-nats is applied **per-state $(B, T)$**, after summing over both the 32-class discrete axis and the 32-group stoch axis. Floor is 1 nat per `(B, T)` entry, lower bound of `loss_kl ≥ 0.5 \cdot 1.0 + 0.1 \cdot 1.0 = 0.6` nats averaged."
- §3.7.5 "Code spec ... `dyn_kl/rep_kl` going into `max(., 1.0)` is shape `(B, T)` after summing over both class and group axes."
- Keep deviation classification `MATCHES PAPER` — Hafner's `Agg(out, 1, jnp.sum)` wrapper around `OneHot.kl(...)` produces the same per-state axis behaviour.

---

### F7 — NIT. Nature activation is RMSNorm + SiLU, not LayerNorm + SiLU.

**Doc claims** (§2.5):
> "**Architecture**: layer normalisation, SiLU activations, same-padded stride-2 kernel-3 convolutions for vision."

**Paper canonical**:
- Preprint Table W.1: "Activation: LayerNorm + SiLU"
- Nature Extended Data Table 5: "Activation: **RMSNorm** + SiLU"

The preprint matches our code (LayerNorm). Nature swapped to RMSNorm. The doc lists this as a paper-canonical ingredient without distinguishing which version. Since the doc otherwise treats the preprint as the primary baseline, and our code uses LayerNorm, this is a minor accuracy issue.

**Severity**: NIT — the §2.5 entry is in the "paper-canonical ingredients" list, where conflation between versions matters. Our code matches preprint; the deviation flag for "LayerNorm vs Nature RMSNorm" is missing from §6.

**Suggested fix**:
- §2.5 row: "**Architecture**: layer normalisation (preprint) / RMSNorm (Nature), SiLU activations, same-padded stride-2 kernel-3 convolutions for vision."
- Add §6 item: "LayerNorm vs Nature RMSNorm — `MATCHES PAPER (preprint)`, `MAJOR DEVIATION (deliberate)` against Nature."

---

### F8 — NIT. Doc's percentile-EMA gradient-flow timing: paper computes EMA on the rollout-batch returns; doc says "moments stale by one step".

**Doc claims** (§3.5.8):
> "**Moments update is OUTSIDE the gradient.** `self.moments.update(lambda_returns)` runs after `nnx.grad`, so the moments used inside the gradient are stale by one step (`trainer.py:336–340, 495`). The comment at L337–338 calls this out as deliberate to avoid tracing through `self.moments` inside `nnx.grad`, which causes OOM."

**Paper canonical** ([Eq. P7](#eq-p7--actor-loss-with-percentile-ema-scaling-preprint-eq-1112--nature-page-3) percentile EMA):
$$S \leftarrow \rho\,S + (1-\rho)\big(\text{Per}(R^\lambda, 95) - \text{Per}(R^\lambda, 5)\big)$$

The preprint and Nature both compute $S$ on the **current imagined-rollout batch**, then divide the actor advantage by $\max(1, S)$. There is no explicit comment about whether $S$ uses the current-batch percentiles directly or the post-update EMA value. From Hafner's published source code semantics (`Moments` reads `self.low.value` / `self.high.value` for the gradient-step normalization, then updates AFTER the gradient step) — this matches our code's "stale by one" pattern. **`MATCHES PAPER` in spirit**.

**Severity**: NIT, recording for completeness. Our code behaviour is correct; the doc's framing as "stale by one step" is technically precise but the reader should know it's the standard practice in Hafner's code too.

**Suggested fix**: tweak §3.5.8 to "Same as Hafner's published implementation: $S$ is computed from the EMA state at gradient-step entry, then `update` is called afterward — read-then-update ordering, not write-then-read."

---

## Reverse-pass findings (paper content the doc misses)

I walked the source PDFs and lit-review independently. The following paper content is **not** covered by the doc:

### R1 — Nature optimiser: LaProp + AGC, not Adam + global-norm

**Source**: Hafner 2025 page 9 (verbatim):
> "We employ adaptive gradient clipping, which clips per-tensor gradients if they exceed 30% of the L2 norm of the weight matrix they correspond to, with its default $\epsilon = 10^{-3}$. Adaptive gradient clipping decouples the clipping threshold from the loss scales, allowing to change loss functions or loss scales without adjusting the clipping threshold. We apply the clipped gradients using the LaProp optimizer with $\epsilon = 10^{-20}$ and its default parameters $\beta_1 = 0.9$ and $\beta_2 = 0.99$. LaProp normalizes gradients by RMSProp and then smoothes them by momentum, instead of computing both momentum and normalizer on raw gradients as Adam does. This simple change allows for a smaller epsilon and avoids occasional instabilities that we observed under Adam."

And Extended Data Table 5: "Gradient clipping: AGC(0.3); Optimizer: LaProp($\epsilon = 10^{-20}$); Learning rate: $4{\cdot}10^{-5}$ (single, uniform across all components)."

**What the doc says**: §3.8.1 paper-spec line says "Adam, no weight decay, no LR schedule, `eps = 1e-5` uniform. World-model LR `1e-4`, actor LR `3e-5`, critic LR `3e-5`. Grad clip global-norm 1000 (preprint) or 100 (Nature)."

**Impact**: the doc's Nature claims for both optimiser type, $\epsilon$, LR, and grad-clip are all wrong. The doc says "Nature uses 100.0 grad clip" — actually Nature uses **AGC at 30%** (a per-tensor relative clip, not a global L2 clip).

**Suggested fix**:
- §3.8.1 paper-spec: "Preprint Table W.1: Adam, asymmetric $\epsilon$ ($10^{-8}$ WM / $10^{-5}$ AC), split LR ($10^{-4}$ WM, $3{\cdot}10^{-5}$ AC), global-norm grad-clip $1000$ (WM) / $100$ (AC). Nature Ext. Table 5: **LaProp** ($\epsilon = 10^{-20}$), **single LR $4{\cdot}10^{-5}$**, **AGC(0.3)** per-tensor adaptive clip."
- §6 item 8 (`WM grad-clip 1000`): keep `MATCHES PAPER (preprint)` but reframe Nature deviation: "Nature uses AGC(0.3), not global-norm; we are not approximating either way".
- Add new §6 entry: "Nature optimiser swap (LaProp + AGC) — entirely separate path; we run preprint Adam recipe."

---

### R2 — Nature replay capacity is $5{\cdot}10^6$, not "typically 1M"

**Source**: Hafner 2025 Extended Data Table 5: "Replay capacity: $5 \times 10^6$".

**What the doc says**: §5.1 row "buffer_capacity | typically 1M | 1_000_000 | matches paper".

**Impact**: our 1M buffer is **5× smaller** than Nature's $5\times 10^6$. Whether this matters depends on env-step throughput, but the doc says "matches paper" which is wrong against Nature.

**Suggested fix**: §5.1 row: "buffer_capacity | $10^6$ (preprint Table W.1 implicit) / $5 \times 10^6$ (Nature Ext. Table 5) | $10^6$ | matches preprint; $5\times$ smaller than Nature".

---

### R3 — Nature critic loss form (max-likelihood) and replay-value term

**Source**: Hafner 2025 page 3, [Eq. P5](#eq-p5--critic-loss-preprint-eq-10--nature-page-3-ml-form) Nature form:
$$\mathcal L(\psi) \doteq -\sum_t [\beta_{\text{val}}\ln p_\psi(R^\lambda_t\mid s_t) + \beta_{\text{repval}}\ln p_\psi(R^{\lambda,\text{replay}}_t \mid s^{\text{replay}}_t)], \quad \beta_{\text{val}}=1, \beta_{\text{repval}}=0.3$$

The replay-side λ-return $R^{\lambda,\text{replay}}_t$ is computed by recursively unrolling on the **replay reward sequence**, with the imagined-rollout $R^\lambda$ at the final replay-step state as the bootstrap value.

**What the doc says**: §3.4.6 — "**NOT IMPLEMENTED**", flagged as `MAJOR DEVIATION (deliberate)`.

**Impact**: the doc correctly flags absence, but the explanation lacks the **bootstrapping mechanism** — readers don't know how the replay-value term is computed. Add a brief gloss: "Replay critic loss bootstraps off the imagined $R^\lambda$ at the start state — i.e. it uses the imagination rollout as on-policy value annotations for the replay trajectory. No imagination during the loss computation." (Verbatim from Nature page 3.)

This is a CONCERN-level completeness gap, not a deviation gap.

---

### R4 — "Free nats" appears as a single hyperparameter row in Nature Ext. Table 5

**Source**: Nature Extended Data Table 5: "Free nats: 1" (a single row, no axis specification).

This corroborates F6: the paper does not specify the axis, but the published code applies the floor per-state (i.e. after summing over groups). Our code agrees. Worth noting that the **paper itself is ambiguous** on this — the disambiguation comes only from the published source.

**Suggested fix**: §3.7.5 paper-spec: "Both papers prescribe `max(1, KL)` per state — Nature Ext. Table 5 lists 'Free nats: 1' as a single floor without axis specification; the published code applies it per-state after summing over the stoch-group axis."

---

### R5 — Nature actor unimix is 1% (same as preprint), and **batch length T is 64 (both versions)** — so our 128 deviates from BOTH

**Source**: Both preprint Table W.1 ("Batch length T = 64") and Nature Ext. Table 5 ("Batch length T = 64"). Doc's §6 item 3 already flags this as `MAJOR DEVIATION (deliberate)`. **No correction needed; just confirming**.

---

### R6 — Symmetric symexp_twohot bin construction: published code uses HALF-bins + mirror

**Source**: `embodied/jax/heads.py` lines 87–97 (fetched live):
```python
half = jnp.linspace(-20, 0, (self.bins - 1) // 2 + 1, dtype=f32)
half = nets.symexp(half)
bins = jnp.concatenate([half, -half[:-1][::-1]], 0)
```

Bins are constructed as: 128 symexp'd negative half-bins + 127 mirrored positive bins = 255 total. **The bin spacing is symmetric around zero in raw space**, with denser spacing near zero (because symexp curves rapidly).

**Our code's `from_twohot`**: `bucket_vals = jnp.linspace(symlog(-20), symlog(+20), 255)` — uniform spacing in symlog space, **then symexp gives non-uniform raw spacing** (also denser near zero).

The qualitative spacing curve matches (denser near zero), but the **raw range is dramatically different** (F1: $\pm 20$ vs. $\pm 4.85{\cdot}10^8$). 

**Suggested fix**: covered by F1 fix.

---

### R7 — Preprint Appendix C "Summary of Differences" is the canonical doc-to-source crosscheck, and the doc misses an item

**Source**: preprint Appendix C, page 19. Lists 8 deltas vs. DreamerV2:
1. Symlog predictions (encoder + decoder + reward + critic)
2. World model regularizer (KL balance + free bits)
3. Policy regularizer (percentile + max(1, S))
4. Unimix categoricals
5. Architecture (LayerNorm + SiLU + same-pad stride-2 kernel-3)
6. **Critic EMA regularizer** — "We compute λ-returns using the **fast critic** network and regularize the critic outputs towards those of its own weight EMA instead of computing returns using the slow critic. However, both approaches perform similarly in practice."
7. Replay buffer (subsequence sampling)
8. Hyperparameters tuned for visual + Atari simultaneously

**What the doc says**: items 1, 2, 4, 5, 7 are covered. **Item 6 is the critical disagreement** that F2 flags: the preprint **explicitly says the slow critic is NOT used for computing returns**, and item 6's "However, both approaches perform similarly in practice" is the appendix justification for the preprint Author's choice. The doc's §3.4.5 reverses this.

**Suggested fix**: covered by F2 fix; reference preprint Appendix C item 6 as the canonical claim.

---

## Symmetry findings

### Equations cited in §2 / §3 with no code pair

- **Nature replay-value loss (§2.4 last sentence + §3.4.6)** — paper Eq. cited; correctly flagged as NOT IMPLEMENTED. Symmetry: the paper-spec line in §3.4.6 names $\beta_{\text{val}} = 1, \beta_{\text{repval}} = 0.3$; no corresponding code exists. ✓ documented (matches Reverse R3).
- **Percentile-EMA Eq. (§2.2 last sentence)** — paper Eq. 12 cited; the corresponding code is `Moments` (`util.py:113-159`). ✓ paired.
- **Nature distributional ML critic Eq. (§2.4 step 4)** — paper re-framing cited; the corresponding code is `loss_critic` (`trainer.py:421-430`). The Eq. cited is the Nature wording of the cross-entropy loss; numerically identical to the preprint cross-entropy form, so matching the code is correct on the loss form, but the **bootstrap-value site** uses target_critic (F2 BLOCKER) — equation does not pair cleanly with code at the bootstrap level.

### Code references cited with no paper pair

- **None**: every §3 code reference cited in the doc has a paper-spec line above it. Symmetry is materially complete (the same finding the code-reviewer flagged for the active code path).

---

## Coordination with code-reviewer F1 (free-nats)

The code-reviewer's F1 BLOCKER asserts:
- **Description error** in §3.5.4/§3.7.5: free-nats applied per `(B, T, stoch=32)`, NOT per `(B, T)` as the prose says. **Math reviewer concurs** — see F6 above. The doc's `(B, T, stoch_dim)` shape claim is wrong; the code clamps a `(B, T)` tensor.
- **Lower-bound arithmetic error**: doc says "(0.5 + 0.1) * 32 = 19.2 nats per (B, T)" — should be 0.6 nats. **Math reviewer concurs**.
- **Deviation reclassification suggestion**: code-reviewer suggests changing `MATCHES PAPER` to e.g. `MAJOR DEVIATION (intent unclear — stricter than paper)`, on the grounds that "Hafner's paper... applies free-bits per stochastic group → lower bound `1 nat × stoch_dim = 32` nats per state".

**Math reviewer expansion / partial dissent on the reclassification**: I checked the **published Hafner code** (github.com/danijar/dreamerv3, `dreamerv3/rssm.py`) for the actual axis of free-bits application. The code uses `out = embodied.jax.outs.Agg(out, 1, jnp.sum)` on the `OneHot` distribution **before** the `kl()` call, then applies `jnp.maximum(dyn, free_nats)` to the post-aggregated KL. The `Agg(..., 1, jnp.sum)` reduces over the stoch-group axis. Therefore:

**The published Hafner code applies free-bits per `(B, T)` state, after summing over the stoch-group axis** — exactly matching our code. The paper's Eq. 5 wording $\max(1, \text{KL}[\dots])$ is ambiguous on the axis, but the published implementation disambiguates in favour of per-state.

So the deviation classification `MATCHES PAPER` is **correct** at the level of "matches Hafner's published code". The doc's verbal description is wrong (clamp axis), but the deviation-flag conclusion is right. I would **keep the `MATCHES PAPER` classification** but **fix the description and lower-bound arithmetic**.

This is a minor disagreement with code-reviewer's F1 final recommendation: the **description** must be fixed (math reviewer agrees), but the **deviation reclassification is unwarranted** given Hafner's published implementation.

---

## Verdict

- **All forward-pass mismatches resolved?** **FAIL** — 3 BLOCKERs (F1 bin layout, F2 target_critic for bootstrap, F3 Adam $\epsilon$ paper-spec) and 2 CONCERNs (F4 horizon, F5 replay-ratio Atari/DMC conflation) remain.
- **All paper content covered in doc?** **FAIL** — R1 (Nature optimiser swap to LaProp + AGC), R2 (Nature replay capacity $5{\cdot}10^6$), R3 (Nature replay-value bootstrap mechanism gloss), R7 (preprint Appendix C item 6 disagreement) need addressing.
- **All §3 entries paper-paired?** **PASS** — every §3 entry has both a paper-spec line and a code-spec line; symmetry is materially complete.
- **Recommendation: ROUTE-BACK-TO-AUTHOR**. Three BLOCKERs each propagate to multiple sections (§2, §3, §5, §6) and one (F2) reorders the §6 priority list. The doc is fixable with surgical edits — no structural overhaul — but the changes are non-trivial and need authorial review before re-circulation. Once F1, F2, F3 are corrected and R1, R2, R3, R7 are added, the doc should be re-routed to math-reviewer for a confirmation pass; with those changes the doc would be ACCEPTED as a paper-canonical implementation reference.

**Urgency**: F2 (target_critic bootstrap) is potentially the most consequential. If the project has been running ablations under the assumption "MATCHES PAPER (Nature wording)" but the bootstrap path is actually the preprint's documented `SlowTarget` ablation (which the preprint's own Appendix D.2 says "offers no benefit"), this is a signal that should be tested independently — possibly compounding with the small-width and high-replay-ratio deviations that §6 already flags as TOP-TIER. Recommend escalating to `senior-developer` and `experiment-designer` for a follow-up controlled comparison once the doc is corrected.

---

## Files audited

- **Doc under review**: `docs/project/concepts/dreamer_v3_implementation.md` (778 lines)
- **Source PDFs (read in full or in load-bearing pages)**:
  - `docs/project/references/Dreamer/sources/Hafner et al. 2023 - Mastering Diverse Domains through World Models.pdf` — pages 1–10 (algorithm), 18–22 (Tables A.1, B.1, Appendix C–E), 38 (Table W.1)
  - `docs/project/references/Dreamer/sources/Hafner et al. 2025 - Mastering diverse control tasks through world models.pdf` — pages 2–10 (algorithm), 16–17 (Extended Data Tables 4, 5)
- **Lit review companion (navigation aid only)**: `docs/project/references/Dreamer/dreamer_lit_review.md` — Phase 5a (lines 1308–1626) and Phase 5b (lines 1627–1763)
- **Code (math-side spot checks)**:
  - `src/models/dreamer_v3_trainer.py` lines 17–23, 28–51, 122, 213–252, 336–504
  - `src/models/dreamer_v3_util.py` lines 6–17, 19–76, 78–110, 113–159
- **Coordinated review**: `docs/reviews/dreamer_v3_implementation_code_review.md` — corroborated F1 free-nats per-(B,T) finding; partially dissented on deviation reclassification suggestion based on Hafner's published code.
- **Hafner's published code (live web fetch via WebFetch)**:
  - `github.com/danijar/dreamerv3/blob/main/dreamerv3/rssm.py` — verified the `Agg(out, 1, jnp.sum)` aggregation pattern that establishes per-state free-bits axis
  - `github.com/danijar/dreamerv3/blob/main/embodied/jax/heads.py` lines 87–97 — verified twohot bin construction `linspace(-20, 0, ...)` + `symexp` + mirror pattern (BLOCKER F1)
  - `github.com/danijar/dreamerv3/blob/main/dreamerv3/configs.yaml` — verified `bins: 255`, `output: symexp_twohot` for both reward and value heads

---

Reviewed by: math-reviewer
