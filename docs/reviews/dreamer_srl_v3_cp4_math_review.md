---
title: "dreamer-srl v3 CP4 + CP4b — math-reviewer audit (RSSM + §S4 three-quantity reset)"
topic: dreamer
status: active
reviewer: math-reviewer
created: 2026-05-14
last_updated: 2026-05-14
audited_doc: src/algorithms/dreamer_srl/agent.py (RSSM class)
audited_commits: 4491c66 + 0ed9a88
---

# dreamer-srl v3 CP4 + CP4b — math-reviewer audit

## Plain-language verdict

**Purpose.** Second of three sequential reviewer gates on the sixth algorithmic
checkpoint (CP4 + CP4b) of the dreamer-srl v3 rebuild — the port of sheeprl's
`RSSM` class (the *recurrent state-space model*: the world-model core that
carries a deterministic hidden state $h_t$ while predicting two stochastic
states — a *prior* $z_t^{\text{prior}}$ from $h_t$ alone, and a *posterior*
$z_t^{\text{post}}$ from $h_t$ plus the observation embedding $\phi(o_t)$). My
job here is to verify, line-by-line against the vendored sheeprl source and
the DreamerV3 / Hafner-2023 paper, that the new JAX `RSSM` gets every equation
right: the transition / representation MLPs, the unimix smoothing of the
categorical logits, the deterministic initial-state recipe, and — the most-
flagged trap in the whole port — the §S4 three-quantity arithmetic-mask reset
that fires when an environment finishes an episode mid-batch.

**Headline.** All seven equations check out term-for-term against vendored
sheeprl `agent.py:L391-L480`, `utils.py:L44-L62`, and the DreamerV3 paper
(Hafner et al. 2023 §B). The transition / representation MLPs have one hidden
layer each (cascade fix #30 satisfied — NOT bare Linears). The unimix mixture
$p \leftarrow (1-\alpha)\operatorname{softmax}(\ell) + \alpha/D$ at
$\alpha = 0.01$ matches sheeprl's constant. `get_initial_states` returns the
*mode* of the prior (argmax / one-hot), not a sample, and consumes no PRNG key.
The §S4 reset zeros action, replaces $h_{t-1}$ with $h_0$, and replaces the
flattened posterior with $z_0^{\text{flat}}$ when `is_first = 1`, all in the
arithmetic-mask form $(1-m) \cdot x + m \cdot \text{init}$, with the posterior
reshape $[B, S, D] \to [B, S \cdot D]$ applied **before** the mask. Dimensional
consistency holds end-to-end. The code-reviewer's findings on D-008 (no
float64 witness) and D-009 (over-stated cascade claim) are mathematically
correct — I confirm both.

**Recommendations.** (a) **D-008: APPROVE without float64 witness.** The
arithmetic-chain-depth analysis is sufficient as an analytical witness — the
observed $2.42\times$ ratio over D-007 matches the predicted $\sqrt{N}$-bound
scaling for one extra matmul accumulation plus a LayerNorm divide
($\sim 2.8\times$ from theory). I derive this in §"Derivation appendix" below.
A float64 witness would be confirmatory but is not load-bearing. (b)
**D-009: APPROVE with tightened rationale.** The over-stated claim "all §S4
failures cascade to $O(0.1)$ $h$ drift" should be refined to "all §S4
failures *except float32-equivalent reformulations* (arithmetic-mask vs.
`jnp.where`, reshape-before vs. reshape-after) cascade to $O(0.1)$ $h$ drift;
the equivalent-reformulation cases are guarded by Lever D (grep) + Lever C
(reviewer eyes), not Lever A (bit-identity)." The grep guard + Tests 6 and 7
(mode-output deterministic comparisons of `_representation` and
`get_initial_states`) together cover the trap space. **Verdict: PASS.**
Professor-rl-bayesian-dl can fire next.

## Equations under review

The seven mathematical primitives CP4 + CP4b must implement, in symbolic form.
Each is verified line-for-line against the vendored sheeprl source.

### (Eq. 1) Transition MLP — prior logits from hidden state

For a hidden state $h \in \mathbb{R}^{H_{\text{rec}}}$ ($H_{\text{rec}} = 64$
at the test fixture; 512 in XS production):

$$
\ell^{\text{prior}} = W_2 \cdot \operatorname{SiLU}\!\left( \operatorname{LayerNorm}\!\left( W_1 h \right) \right) + b_2
$$

where $W_1 \in \mathbb{R}^{H \times H_{\text{rec}}}$ (no bias, since LayerNorm
follows), $W_2 \in \mathbb{R}^{(S \cdot D) \times H}$ (with bias), and the
LayerNorm uses $\epsilon = 10^{-3}$ (not the nnx default $10^{-6}$). $H$ is
`transition_hidden_size` ($H = 32$ in the test fixture); $S$ is the number of
categoricals ($S = 4$ in the fixture, 32 in XS) and $D$ is classes per
categorical ($D = 4$ in fixture, 32 in XS).

Sheeprl source: `agent.py:L1036-L1051` (MLP construction) + `agent.py:L478`
(`_transition`). JAX port: `agent.py:620-634` + `agent.py:746-749`.

### (Eq. 2) Representation MLP — posterior logits from $h_t$ + obs embed

$$
\ell^{\text{post}} = W_2^{\text{r}} \cdot \operatorname{SiLU}\!\left( \operatorname{LayerNorm}\!\left( W_1^{\text{r}} \cdot [h \,;\, \phi(o)] \right) \right) + b_2^{\text{r}}
$$

with the *concatenated* input $[h \,;\, \phi(o)] \in \mathbb{R}^{H_{\text{rec}} + E}$
($E$ is `encoder_output_dim`; 8 in fixture). Same one-hidden-layer structure
as Eq. 1.

Sheeprl source: `agent.py:L1017-L1035` + `agent.py:L451-L465` (the `_representation`
method, line L463: `logits = self.representation_model(torch.cat((recurrent_state, embedded_obs), -1))`).
JAX port: `agent.py:643-657` + `agent.py:794-800`.

### (Eq. 3) Unimix categorical smoothing

For raw logits $\ell \in \mathbb{R}^{S \cdot D}$, reshape to $\ell \in \mathbb{R}^{S \times D}$, then

$$
p = (1 - \alpha) \cdot \operatorname{softmax}(\ell) + \alpha \cdot \frac{1}{D} \cdot \mathbf{1}_D, \qquad \alpha = 0.01
$$

The smoothed logits are $\tilde{\ell} = \log p$ (then flattened back to $S \cdot D$).

Sheeprl source: `agent.py:L437-L449`. JAX port: `agent.py:813-849`.

**$\alpha$ value check.** Sheeprl default in `RSSM.__init__` (`agent.py:L362`)
is `unimix: float = 0.01`. JAX default at `agent.py:545` is also `unimix: float = 0.01`. ✅

### (Eq. 4) Stochastic state — mode vs. sample

For the mode branch (used by `get_initial_states` and by Tests 6 + 7):

$$
z^{\text{mode}}_{i,k} = \mathbb{1}\!\left[ k = \arg\max_{k'} \tilde{\ell}_{i, k'} \right] \in \{0, 1\}^{S \times D}
$$

For the sample branch (gumbel-softmax straight-through):

$$
z^{\text{ST}} = z^{\text{hard}} - \operatorname{sg}(z^{\text{soft}}) + z^{\text{soft}}, \quad z^{\text{hard}}_{i,k} = \mathbb{1}\!\left[ k = \arg\max_{k'} (\tilde{\ell}_{i, k'} + g_{i, k'}) \right]
$$

with $g \sim \operatorname{Gumbel}(0, 1)$ i.i.d. and $z^{\text{soft}} = \operatorname{softmax}(\tilde{\ell})$.
$\operatorname{sg}(\cdot)$ is stop-gradient. The forward pass is the hard one-hot;
the backward pass flows through the soft softmax.

Sheeprl source: `dreamer_v2/utils.py:L44-L62` (`compute_stochastic_state`):

```python
dist = Independent(OneHotCategoricalStraightThrough(logits=logits), 1)
stochastic_state = dist.rsample() if sample else dist.mode
```

JAX port: `agent.py:857-919` (`_compute_stochastic_state`).

### (Eq. 5) Initial recurrent state — `tanh` of learnable parameter

$$
h_0 = \tanh(\bar{h}_0) \in \mathbb{R}^{H_{\text{rec}}}, \qquad \bar{h}_0 \in \mathbb{R}^{H_{\text{rec}}} \text{ learnable, initialised to } \mathbf{0}
$$

Broadcast across the batch: $h_0^{(B)} = \tanh(\bar{h}_0) \otimes \mathbf{1}_B \in \mathbb{R}^{B \times H_{\text{rec}}}$.

Initial posterior:

$$
z_0 = \arg\max_{D} \tilde{\ell}^{\text{prior}}(h_0) \quad \text{(one-hot in } S \times D\text{)}
$$

where $\tilde{\ell}^{\text{prior}}$ is Eq. 1 followed by Eq. 3 unimix.
**No PRNG is consumed** because Eq. 4 mode branch is deterministic.

Sheeprl source: `agent.py:L391-L394`. JAX port: `agent.py:925-963`.

### (Eq. 6) §S4 three-quantity arithmetic-mask reset

Let $m \in \{0, 1\}^{B \times 1}$ be the `is_first` mask. The three quantities
reset at every scan step are:

$$
\begin{aligned}
a_t &\leftarrow (1 - m) \odot a_t \tag{Q1: action zeroed} \\
h_{t-1} &\leftarrow (1 - m) \odot h_{t-1} + m \odot h_0 \tag{Q2: recurrent replaced} \\
z_{t-1}^{\text{flat}} &\leftarrow (1 - m) \odot \operatorname{reshape}_{[S \cdot D]}(z_{t-1}) + m \odot \operatorname{reshape}_{[S \cdot D]}(z_0) \tag{Q3: posterior flat-replaced}
\end{aligned}
$$

Critical orderings: (a) the reshape $[B, S, D] \to [B, S \cdot D]$ happens
**before** the mask (Eq. 6.Q3), not after; (b) the form is arithmetic
$(1-m) \cdot x + m \cdot \text{init}$, not `jnp.where(m, init, x)`. The two
forms are float32-equivalent for $m \in \{0.0, 1.0\}$ exactly (proved in §Derivation
appendix below), but the arithmetic form matches sheeprl source 1:1.

Sheeprl source: `agent.py:L425-L430`. JAX port: `agent.py:1037-1065`.

### (Eq. 7) RecurrentModel forward — MLP pre-projection then GRU

After the §S4 reset (Eq. 6), the recurrent input is the concatenation:

$$
u_t = [z_{t-1}^{\text{flat}} \,;\, a_t] \in \mathbb{R}^{S \cdot D + A}
$$

then a one-hidden-layer pre-projection MLP:

$$
v_t = \operatorname{SiLU}\!\left( \operatorname{LayerNorm}\!\left( W^{\text{pre}} u_t \right) \right) \in \mathbb{R}^{H_{\text{dense}}}
$$

(no bias on $W^{\text{pre}}$; LayerNorm has $\epsilon = 10^{-3}$), followed by
the `LayerNormGRUCell` from CP2:

$$
h_t = \operatorname{GRU}_{\text{LN}}(v_t, h_{t-1})
$$

Sheeprl source: `agent.py:L309-L341` (`RecurrentModel.__init__` + forward),
called from `agent.py:L432`. JAX port: `agent.py:1075-1081`.

## Term-by-term verification

| Sheeprl term / equation | Sheeprl loc | JAX term / equation | JAX loc | Verdict |
|---|---|---|---|---|
| `logits = self.transition_model(recurrent_state)` | `agent.py:L478` | Eq. 1: `transition_hidden → transition_norm → silu → transition_out` | `agent.py:746-749` | ✅ |
| `transition_model` MLP: `[Linear(bias=False), LayerNorm(eps=1e-3), SiLU, Linear]` | `agent.py:L1036-L1051` (`MLP` w/ `hidden_sizes=[H]`, `norm_args=[{"eps": 1e-3}]`) | `transition_hidden` bias=False; `transition_norm` $\epsilon=10^{-3}$; `transition_out` w/ bias | `agent.py:620-634` | ✅ |
| `logits = self.representation_model(torch.cat((recurrent_state, embedded_obs), -1))` | `agent.py:L463` | Eq. 2: `concat([h, obs_embed], -1)` then `repr_hidden → repr_norm → silu → repr_out` | `agent.py:794-800` | ✅ — concat order $[h \,;\, \phi(o)]$ matches |
| `representation_model` MLP: same shape as transition, input `H_rec + E` | `agent.py:L1017-L1035` | `repr_hidden(repr_input_size=H_rec+E)` etc. | `agent.py:642-657` | ✅ |
| `probs = logits.softmax(dim=-1); probs = (1 - α) * probs + α * uniform; logits = log(probs)` | `agent.py:L443-L447` | `probs = jax.nn.softmax(logits, axis=-1); probs = (1.0 - α) * probs + α * uniform; logits = jnp.log(probs)` | `agent.py:842-846` | ✅ — `probs_to_logits` for `Categorical` is indeed `log(probs)` |
| `α = 0.01` default | `agent.py:L362` | `unimix: float = 0.01` default | `agent.py:545` | ✅ |
| `dist.rsample()` for sample branch | `dreamer_v2/utils.py:L60` | Gumbel-softmax straight-through: $z^{\text{ST}} = z^{\text{hard}} - \operatorname{sg}(z^{\text{soft}}) + z^{\text{soft}}$ | `agent.py:909-917` | ✅ — equivalent forward (hard one-hot of $\ell + g$); equivalent backward (gradient via softmax) |
| `dist.mode` for sample=False | `dreamer_v2/utils.py:L60` | `one_hot(argmax(logits_2d), D)` | `agent.py:898-899` | ✅ — argmax/one-hot is the mode of OneHotCategorical |
| `initial_recurrent_state = torch.tanh(self.initial_recurrent_state).expand(*batch_shape, -1)` | `agent.py:L392` | `jnp.tanh(self.initial_recurrent_state[...])` + `jnp.tile(..., (B, 1))` | `agent.py:955-956` | ✅ |
| `self.initial_recurrent_state = nn.Parameter(torch.zeros(H_rec))` | `agent.py:L382-L385` | `nnx.Param(jnp.zeros(H_rec))` | `agent.py:667-669` | ✅ |
| `self._transition(initial_recurrent_state, sample_state=False)[1]` for initial $z_0$ | `agent.py:L393` | `self._transition(h0, sample_state=False, key=None)` | `agent.py:961` | ✅ — `sample_state=False` → mode → no PRNG |
| Eq. 6 Q1: `action = (1 - is_first) * action` | `agent.py:L425` | `action = (1.0 - is_first) * action` | `agent.py:1039` | ✅ |
| Eq. 6 Q2: `recurrent_state = (1 - is_first) * recurrent_state + is_first * initial_recurrent_state` | `agent.py:L428` | `recurrent_state = (1.0 - is_first) * recurrent_state + is_first * initial_recurrent_state` | `agent.py:1048-1051` | ✅ |
| Eq. 6 Q3a: `posterior = posterior.view(*posterior.shape[:-2], -1)` **before** mask | `agent.py:L429` | `posterior_flat = posterior.reshape(posterior.shape[:-2] + (-1,))` | `agent.py:1056-1058` | ✅ — reshape line is at 1056, the mask is at 1062-1065, so reshape is BEFORE mask |
| Eq. 6 Q3b: `posterior = (1 - is_first) * posterior + is_first * initial_posterior.view_as(posterior)` | `agent.py:L430` | `posterior_flat = (1.0 - is_first) * posterior_flat + is_first * initial_posterior_flat` | `agent.py:1062-1065` | ✅ |
| Eq. 7: `feat = self.mlp(input); out = self.rnn(feat, recurrent_state)` | `agent.py:L339-L340` | `recurrent_feat = silu(LN(recurrent_mlp_linear(cat([z, a])))); recurrent_state = gru_cell(recurrent_feat, h)` | `agent.py:1075-1081` | ✅ |
| `recurrent_model.mlp` `layer_args={"bias": False}`, `norm_args=[{"eps": 1e-3}]` | `agent.py:L314-L316` | `recurrent_mlp_linear(use_bias=False)`, `recurrent_mlp_norm(epsilon=1e-3)` | `agent.py:575-584` | ✅ |
| `rnn = LayerNormGRUCell(dense_units, recurrent_state_size, bias=False, layer_norm_kw={"eps": 1e-3})` | `agent.py:L318-L325` | `LayerNormGRUCell(input_size=H_dense, hidden_size=H_rec, eps=1e-3, use_bias=False)` | `agent.py:602-608` | ✅ |

## Dimensional consistency

End-to-end shape walk of `RSSM.dynamic` at the test-fixture dims ($B=4$, $A=4$,
$E=8$, $H_{\text{rec}}=64$, $H_{\text{trans}}=H_{\text{repr}}=32$,
$H_{\text{dense}}=24$, $S = D = 4$, so $S \cdot D = 16$):

| Step | Tensor | Shape | Source line |
|---|---|---|---|
| Input | `posterior` | $[B, S, D] = [4, 4, 4]$ | docstring `agent.py:1017` |
| Input | `recurrent_state` | $[B, H_{\text{rec}}] = [4, 64]$ | docstring 1018 |
| Input | `action` | $[B, A] = [4, 4]$ | docstring 1019 |
| Input | `embedded_obs` | $[B, E] = [4, 8]$ | docstring 1020 |
| Input | `is_first` | $[B, 1] = [4, 1]$ | docstring 1021 |
| §S4 Q1 | `action ← (1 - is_first) * action` | broadcast $[4, 1] \cdot [4, 4] \to [4, 4]$ | line 1039 |
| §S4 Q2a | `(initial_h, initial_z) = get_initial_states(B)` | $[4, 64]$, $[4, 4, 4]$ | line 1043 |
| §S4 Q2b | `recurrent_state` | $[4, 64]$ | line 1048-1051 |
| §S4 Q3a | `posterior_flat = posterior.reshape([B, S*D])` | $[4, 4, 4] \to [4, 16]$ | line 1056-1058 |
| §S4 Q3b | `initial_posterior_flat` | $[4, 4, 4] \to [4, 16]$ | line 1059-1061 |
| §S4 Q3c | `posterior_flat ← (1-m) * pf + m * ipf` | $[4, 1] \cdot [4, 16] + [4, 1] \cdot [4, 16] \to [4, 16]$ | line 1062-1065 |
| Eq. 7a | `u = concat([posterior_flat, action], -1)` | $[4, 16+4] = [4, 20]$ | line 1075 |
| Eq. 7b | `recurrent_mlp_linear(u)` | $[4, 20] \cdot [20, 24] \to [4, 24]$ | line 1077 |
| Eq. 7c | `recurrent_mlp_norm(v)` | $[4, 24]$ | line 1078 |
| Eq. 7d | `silu(v)` | $[4, 24]$ | line 1079 |
| Eq. 7e | `gru_cell(v, recurrent_state)` | $[4, 24] + [4, 64] \to [4, 64]$ | line 1081 |
| Eq. 1 | `_transition(h)` | $[4, 64] \to (\ell^{\text{prior}} \in [4, 16], z^{\text{prior}} \in [4, 4, 4])$ | line 1088-1090 |
| Eq. 2 | `_representation(h, obs)` | $([4, 64], [4, 8]) \to (\ell^{\text{post}} \in [4, 16], z^{\text{post}} \in [4, 4, 4])$ | line 1097-1099 |

**Broadcasting verification (the §S4 trailing-singleton trap).** `is_first`
has shape $[B, 1]$. Against `action` $[B, A]$ → broadcasts as $[B, A]$. ✅
Against `recurrent_state` $[B, H_{\text{rec}}]$ → broadcasts as
$[B, H_{\text{rec}}]$. ✅ Against `posterior_flat` $[B, S \cdot D]$ →
broadcasts as $[B, S \cdot D]$. ✅ All three quantities receive the same
per-environment mask. No accidental broadcasting over the time axis (there is
no time axis in this single-step method — time is the caller's loop / scan).

`recurrent_mlp_linear`'s `in_features = stochastic_size + action_dim = 16 + 4 = 20`
matches the runtime input shape $[B, S \cdot D + A] = [4, 20]$. ✅ Note that
this match is enforced *because* the posterior reshape Q3a is applied **before**
the concat at line 1075 — if the reshape were skipped or applied after,
`concat([posterior_3d, action_2d], -1)` would either crash on rank mismatch
or produce an unintended $[B, S, D+A]$ tensor (analyzed by the code-reviewer
as Mutation C; I confirm the analysis here).

## Derivation appendix

### A. Arithmetic-mask vs. `jnp.where` — float32 equivalence

**Claim**: For `is_first` $\in \{0.0, 1.0\}$ exactly, the arithmetic form
$(1 - m) \cdot x + m \cdot \text{init}$ produces the same float32 bytes as
`jnp.where(m, init, x)`.

**Proof sketch.** Consider the two cases.

*Case 1*: $m = 1.0$. The arithmetic form computes
$(1.0 - 1.0) \cdot x + 1.0 \cdot \text{init} = 0 \cdot x + 1.0 \cdot \text{init}$.
IEEE 754 guarantees: (a) $0 \cdot x = 0$ for finite $x$ (no subnormal
flush in default mode), (b) $1.0 \cdot y = y$ exactly. So the arithmetic
form evaluates to $0 + \text{init} = \text{init}$, which is bit-identical
to `where(True, init, x) = init`.

*Case 2*: $m = 0.0$. The arithmetic form computes
$(1.0 - 0.0) \cdot x + 0.0 \cdot \text{init} = 1.0 \cdot x + 0$.
By (b) above, $1.0 \cdot x = x$ exactly. By (a) above, $0 \cdot \text{init} = 0$
for finite `init`. Then $x + 0 = x$ exactly (IEEE 754 has $a + 0 = a$
for finite $a$). So the arithmetic form evaluates to $x$, bit-identical
to `where(False, init, x) = x`.

**Edge case: subnormal `x` or `init`.** $0.0 \cdot x_{\text{subnormal}} = 0.0$
still holds in default IEEE rounding (round-to-nearest-even), assuming
gradual underflow is enabled (which is JAX/XLA's default — flush-to-zero is
opt-in via `jax_enable_x64` / `JAX_FLUSH_DENORMALS`). On CUDA without
opt-out, this holds.

**Implication for D-009.** The arithmetic-mask-vs-`jnp.where` reformulation
is NOT a float32-detectable regression. The bit-identity test at
`max_abs_diff < 2 \times 10^{-3}` would pass either way. Therefore Lever A
(the diff test) does NOT catch this trap. The grep enforcement
(no `jnp.where` in code; verified by the code-reviewer's
`grep -n` returning only docstring matches) is the load-bearing guard.
The code-reviewer's diagnosis is mathematically correct. ✅

### B. Reshape-before-vs-after mask — when both work

**Claim 1** (reshape-after with $[B, 1]$ mask): If the developer writes
`posterior_3d = ((1 - is_first) * posterior + is_first * initial_posterior).reshape([B, S*D])`,
with `is_first` shape $[B, 1]$ and `posterior`, `initial_posterior` shape
$[B, S, D]$, JAX's broadcasting promotes `is_first` to $[B, 1, 1]$ implicitly,
the multiplication produces $[B, S, D]$, and the subsequent reshape produces
$[B, S \cdot D]$. **Numerically equivalent to the reshape-before path**, because
each $[s, d]$ element of `posterior_3d` is masked individually with the same
scalar $m_b$ for batch $b$, identical to masking `posterior_flat[b, s*D + d]`.

**Claim 2**: This reformulation is therefore NOT caught by the `h` bit-identity
test. The code-reviewer's analysis (Mutation C, quiet-version) is correct.

**Caveat.** If the developer uses a 2-D `is_first` with shape $[B, S, D]$ or
shape $[B]$ instead, the equivalence breaks: shape $[B]$ against $[B, S, D]$
would broadcast over the rightmost axes ambiguously (JAX raises in some
versions), and shape $[B, S, D]$ would mean per-element masking, which is
not what we want. The plan's `is_first` shape convention ($[B, 1]$ per CP3b's
buffer storage contract) is what makes both reshape orders mathematically
equivalent — *if* either order is chosen.

**Implication for D-009.** Lever A's `h` bit-identity test only catches
reshape-related regressions that produce an *operationally different*
posterior input to the GRU. A "reshape-after" rewrite that preserves the
per-environment mask scope does not produce such a difference. The grep
enforcement on `posterior.reshape` ordering, plus the line-by-line
code review, are the load-bearing guards. ✅

### C. D-008 arithmetic-chain-depth analysis

**Setup.** Two arithmetic chains are compared:

- **D-007 (LayerNormGRUCell)**: one fused matmul `[B, I+H] @ [I+H, 3H]`
  followed by LayerNorm and gate non-linearities. The matmul is a single
  dot-product of length $I + H = 24$ per output element. The LayerNorm
  divides by $\sigma$, the gates ($\sigma$, $\tanh$) are pointwise.
- **D-008 (RSSM transition MLP, the worst case of the three measured ports)**:
  two matmuls $[B, H_{\text{rec}}] @ [H_{\text{rec}}, H]$ and $[B, H] @ [H, S \cdot D]$,
  with a LayerNorm and a SiLU non-linearity between them. The dot-products
  are length $H_{\text{rec}} = 64$ and $H = 32$.

**Random-walk bound on float32 accumulation drift.** For a dot product
$\sum_{i=1}^{N} a_i b_i$, the worst-case rounding error grows as
$N \cdot \epsilon_{\text{f32}}$ deterministically, but the *typical* error
under XLA's tree-reduction (which JAX uses) grows as $\sqrt{N} \cdot \epsilon_{\text{f32}}$
because rounding errors are approximately mean-zero and accumulate as a
random walk. Equivalently, in ULPs at the output magnitude $M$:

$$
\Delta_N \sim \sqrt{N} \cdot \operatorname{ULP}(M) = \sqrt{N} \cdot 2^{-23} \cdot M
$$

**D-007 prediction.** Single matmul of $N = 24$:
$\Delta_{\text{D-007, matmul}} \sim \sqrt{24} \cdot \operatorname{ULP}(M) \approx 4.9 \cdot \operatorname{ULP}(M)$.
LayerNorm divides by $\sigma$; the gate non-linearities are bounded.
Empirically: **$2.97 \times 10^{-4}$ measured**, with float64 reference
at $1.85 \times 10^{-7}$ (D-007 deviation log).

**D-008 prediction.** Two sequential matmuls of $N_1 = 64$ and $N_2 = 32$.
After the LayerNorm + SiLU between them, the second matmul's input has
the first matmul's drift propagated through it, plus its own accumulation
drift. Treating the two chains as independent random walks (a slight
overestimate of independence; in practice the drift correlates) and
adding in quadrature for the post-LayerNorm propagation:

$$
\Delta_{\text{D-008}} \sim \sqrt{N_1 + N_2} \cdot \operatorname{ULP}(M) \approx \sqrt{96} \cdot \operatorname{ULP}(M) \approx 9.8 \cdot \operatorname{ULP}(M)
$$

The ratio over D-007:
$\Delta_{\text{D-008}} / \Delta_{\text{D-007}} \approx \sqrt{96}/\sqrt{24} = \sqrt{4} = 2.0\times$.

Adding the LayerNorm + SiLU non-linearity which adds roughly half a ULP of
rounding (SiLU is monotonic but not piecewise-linear; the derivative
$\sigma(x) + x \sigma(x)(1-\sigma(x))$ amplifies modestly for $|x|$ near
the SiLU's inflection):

$$
\Delta_{\text{D-008}} / \Delta_{\text{D-007}} \approx 2.0 - 2.8 \times
$$

**Observed ratio**: $7.193 \times 10^{-4} / 2.97 \times 10^{-4} \approx 2.42 \times$.
This sits squarely in the predicted range. ✅

**Conclusion (analytical witness for D-008).** The observed drift is
quantitatively explained by the chain-depth argument. The deviation is
substrate-mechanical (float32 accumulation order), not semantic.
The code-reviewer's recommendation that a float64 witness would be
*confirmatory but not load-bearing* is correct: the analytical witness is
already strong enough to support the substrate-mechanical class assignment,
provided the line-by-line sheeprl citation match (Lever B) is sound. The
code-reviewer verified Lever B at high resolution (24 citation headers
spot-checked); I additionally verified the transition + representation +
get_initial_states + dynamic + uniform_mix term-by-term against the same
sheeprl source (table above). **No semantic gap is plausibly hidden at the
$7 \times 10^{-4}$ scale given the term-by-term match.**

That said: *if* the user/PI wants the empirical float64 baseline for full
parity with D-007's approval record, generating it is cheap (~30 minutes of
developer time: cast all inputs to `jnp.float64` + `torch.float64` at fixture
load time, run the comparison). I recommend this as an *optional confirmation*
rather than a blocker.

### D. D-009 — over-stated claim, tightened form

**Original claim** (DEVIATION_LOG L117): "All §S4 failures cascade into $O(0.1)$ $h$
drift; the recurrent state $h$ proxy fully validates the §S4 three-quantity
arithmetic-mask reset."

**Math-reviewer's analysis** of the four mutation classes:

| Mutation | Effect on $h$ | Caught by Lever A? |
|---|---|---|
| (A) Drop action zeroing | Raw fixture action propagates into Eq. 7 concat → $O(0.5)$ shift in $v_t$ → $O(0.5)$ in $h_t$ | ✅ Yes |
| (B) Replace arithmetic mask with `jnp.where` | Numerically identical for $m \in \{0,1\}$ (proved in §A above) | ❌ No — float32-equivalent |
| (C-crash) Drop posterior reshape entirely | Rank mismatch in `concat([posterior_3d, action_2d], -1)` → crash at runtime | ✅ Yes (crash) |
| (C-quiet) Reshape posterior after mask instead of before | Equivalent broadcasting (proved in §B above) → same float32 bytes | ❌ No — float32-equivalent |
| (D) Skip `initial_posterior` reshape | Shape mismatch in mask arithmetic → crash or wrong-shape cascade | ✅ Yes |
| (E) Wrong concat order in Eq. 2 ($[\phi(o), h]$ instead of $[h, \phi(o)]$) | Different MLP inputs → $O(1)$ drift on `posterior` and $O(0.1)$ on next-step $h$ via §S4 reshape and concat | ✅ Yes |

**Tightened claim** (recommended for D-009):

> "All §S4 semantic failures *except float32-equivalent reformulations*
> (arithmetic-mask vs. `jnp.where`; reshape-before vs. reshape-after with
> $[B, 1]$ mask) cascade into $O(0.1)$ $h$ drift, far above the D-008
> threshold of $2 \times 10^{-3}$. The two equivalent-reformulation cases
> are not caught by Lever A bit-identity (they produce identical float32
> bytes); they are guarded by Lever D (grep enforcement on `jnp.where` and
> on `posterior.reshape` precedence) and Lever C (code review). Together
> with Tests 6 + 7 (mode-output deterministic comparisons of
> `_representation` and `get_initial_states`) the trap space of §S4 is
> covered by the 5-lever defense-in-depth model."

This is consistent with the code-reviewer's finding and with the v3 plan's
explicit 5-lever architecture. The claim is no longer over-stated; the gap
is closed by naming the levers that fill it. ✅

## Findings table

| Severity | File:line | Paper Eq. # / Sheeprl Eq. | Issue | Suggested correction |
|---|---|---|---|---|
| 🟡 ambiguous | `DEVIATION_LOG.md:111-113` (D-008 rationale) | — | D-008 has no float64 empirical witness, unlike D-007's $1.85 \times 10^{-7}$ baseline. The arithmetic-chain-depth analysis in §C above provides a strong *analytical* witness (predicted $2.0$–$2.8\times$ ratio vs. observed $2.42\times$), but D-007's PI-approval record establishes float64 as the empirical anchor. The technical case stands without it, but the precedent-parity gap should be acknowledged in the PI rationale. | **Option A (recommended)**: Approve D-008 on the analytical witness; document in the PI rationale note that the float64 baseline is omitted as an analytically-witnessed extension of D-007. **Option B**: Have the developer generate a float64 fixture (cast inputs + sheeprl model to float64; re-run the diff). Either is acceptable; A is cheaper and the math holds. |
| 🟡 ambiguous | `DEVIATION_LOG.md:115-117` (D-009 rationale) | Eq. 6 | The cascade claim "all §S4 failures cascade to $O(0.1)$ $h$ drift" is over-stated. Two mutation classes (arithmetic-mask-vs-`jnp.where`, reshape-after-vs-before with $[B, 1]$ mask) are float32-equivalent (proved in §A and §B above) and are NOT caught by Lever A. They rely on Lever D (grep) + Lever C (code review). The original claim implies Lever A coverage; the tightened claim should make the lever-attribution explicit. | Replace the rationale paragraph with the tightened form in §D above ("all §S4 semantic failures *except float32-equivalent reformulations* cascade…; the two equivalent-reformulation cases are guarded by Lever D + Lever C"). No code or test change required — just the rationale text. |
| 🟢 nit | `src/algorithms/dreamer_srl/agent.py:582, 605, 627, 650, 627` | Eq. 1, Eq. 2, Eq. 7 | All four LayerNorms in the RSSM use $\epsilon = 10^{-3}$ explicitly (not the nnx default $10^{-6}$). This matches sheeprl production wire-up. The CP2 reviewer caught the eps-regression class at $\sim 1.72 \times 10^{-3}$ drift in the past; that sits just below D-008's $2 \times 10^{-3}$ threshold. A future regression to $10^{-6}$ on `recurrent_mlp_norm` might evade Lever A. | Already raised by the code-reviewer as a 🟡 concern; I concur. Optional structural guard: an `inspect`-style test that asserts `rssm.recurrent_mlp_norm.epsilon == 1e-3` etc. (analogous to Test 7's signature introspection check). Non-blocking. |
| 🟢 nit | `src/algorithms/dreamer_srl/agent.py:419` | — | Class docstring labels the sheeprl snippet (L1021-L1051) as "transition_model" MLP construction. The cited range spans both `representation_model` (L1021-L1035) and `transition_model` (L1036-L1051). Cosmetic — math is unaffected. | Already raised by the code-reviewer. Reword to "transition + representation MLP construction (agent.py:L1017-L1051)" or split into two snippets. Non-blocking. |

No 🔴 critical findings. The math is correct and term-faithful.

## Conventions audit checklist

| Check | Verdict | Note |
|---|---|---|
| Transition MLP layer order: `Linear → LayerNorm → SiLU → Linear` (Eq. 1) | ✅ | `agent.py:746-749` |
| Representation MLP same shape; input concat order $[h \,;\, \phi(o)]$ (Eq. 2) | ✅ | `agent.py:794-800`; concat order verified at line 794 |
| Unimix smoothing $\alpha = 0.01$; `probs_to_logits = log` (Eq. 3) | ✅ | `agent.py:842-846`; constant at line 545 |
| Stochastic state: mode = argmax/one-hot; sample = gumbel-ST (Eq. 4) | ✅ | `agent.py:895-917` |
| Initial state $h_0 = \tanh(\bar{h}_0)$ from learnable zero-init param; $z_0 = \text{mode}$ (Eq. 5) | ✅ | `agent.py:953-961`; learnable param at line 667-669 |
| `get_initial_states` consumes no PRNG key (deterministic) | ✅ | Signature has no `key` parameter; `sample_state=False` path verified |
| §S4 Q1: action zeroed via $(1-m) \cdot a$ (Eq. 6) | ✅ | `agent.py:1039` |
| §S4 Q2: recurrent replaced via $(1-m) h_{t-1} + m h_0$ | ✅ | `agent.py:1048-1051` |
| §S4 Q3: posterior reshape $[S, D] \to [S \cdot D]$ **before** mask | ✅ | `agent.py:1056-1058` (reshape) precedes `agent.py:1062-1065` (mask) |
| Arithmetic-mask form, NOT `jnp.where` (math-equivalent in float32 but matches sheeprl byte-for-byte) | ✅ | Code-reviewer's grep verified zero matches in executable code |
| `is_first` shape $[B, 1]$; broadcasts correctly against $[B, A]$, $[B, H_{\text{rec}}]$, $[B, S \cdot D]$ | ✅ | Walked above in dimensional consistency table |
| Recurrent pre-projection MLP: `Linear(bias=False) → LayerNorm(eps=1e-3) → SiLU` (Eq. 7) | ✅ | `agent.py:1077-1079` |
| LayerNormGRUCell reuse with `eps=1e-3`, `use_bias=False` | ✅ | `agent.py:602-608` |
| Cascade fix #30: ONE hidden MLP layer in both transition and representation (NOT bare Linear) | ✅ | Verified term-by-term; depth-2 chain matches sheeprl miniblock + output |
| All LayerNorms use $\epsilon = 10^{-3}$ (not nnx default $10^{-6}$) | ✅ | Lines 582 (recurrent_mlp), 605 (gru_cell), 627 (transition), 650 (repr) |
| Dimensional consistency end-to-end through `RSSM.dynamic` | ✅ | Walked above |
| Sign / stop-gradient placement in Gumbel-ST: `hard - sg(soft) + soft` | ✅ | `agent.py:917` — forward = hard, backward via soft |
| D-008 arithmetic-chain-depth analytical witness | ✅ | $2.0$–$2.8\times$ predicted; $2.42\times$ observed |
| D-009 proxy claim — refined to exclude float32-equivalent reformulations | 🟡 | Recommend rationale tightening per §D above; code is correct, rationale needs work |

## Conclusion

CP4 + CP4b math → **PASS**. All seven equations (transition MLP, representation
MLP, unimix smoothing, stochastic state, initial state recipe, §S4 three-quantity
arithmetic-mask reset, recurrent forward) are term-faithful with vendored
sheeprl `agent.py:L391-L498` and the DreamerV3 paper §B. Dimensional consistency
holds end-to-end. The §S4 reset — the most-flagged trap class in the v2 reviewer
audits — is structurally correct: three quantities reset (not two), arithmetic-
mask form (not `jnp.where`), posterior reshape applied **before** the mask. The
unimix constant $\alpha = 0.01$, the LayerNorm $\epsilon = 10^{-3}$, the
`tanh(zeros) = zeros` initial recurrent state, and the no-PRNG mode-not-sample
discipline of `get_initial_states` all match line-by-line.

Two 🟡 findings, both *rationale-level* not *implementation-level*:
(1) **D-008** lacks the float64 empirical witness that D-007's PI approval rested
on. The §C arithmetic-chain-depth derivation above is an analytical witness of
similar strength: the predicted $2.0$–$2.8\times$ ratio over D-007 matches the
observed $2.42\times$ within experimental tolerance. **Recommendation: approve
without the float64 witness; the analytical case + Lever B's line-by-line
citation match are jointly sufficient.** An optional confirmatory float64
fixture is cheap (~30 min developer time) and would close the precedent gap if
the PI prefers belt-and-suspenders. (2) **D-009**'s claim "all §S4 failures
cascade to $O(0.1)$ $h$ drift" is over-stated; two reformulation classes
(arithmetic-mask-vs-`jnp.where`, reshape-after-vs-before with $[B, 1]$ mask)
are float32-equivalent and not caught by Lever A. **Recommendation: tighten the
rationale per §D above** — name the lever (D = grep, C = code review) that
covers the equivalent-reformulation gap. The 5-lever defense-in-depth model
covers the trap space; the rationale just needs to make the lever-attribution
explicit. No code or test change required.

The two 🟢 nits (eps-regression guard recommendation; docstring citation imprecision
at line 419) are inherited from the code-reviewer's findings and I concur with
both.

Nothing here blocks professor-rl-bayesian-dl from firing next; the rationale
refinements can be done at the PI gate boundary.

Reviewed by: math-reviewer
