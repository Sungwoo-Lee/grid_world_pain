"""dreamer-srl loss utilities — TwoHotEncoding distribution ported from sheeprl@33b6366.

All functions in this module carry source-citation docstring headers (Lever B)
naming the vendored sheeprl file and exact line range. Every function is
paired with a bit-identity test in tests/algorithms/dreamer_srl/test_loss.py
that asserts max-absolute-difference < 1e-6 against the sheeprl PyTorch side.

Isolation rule (v2 Risks §13 / NNX_CONVENTIONS.md): this module does NOT import
from the legacy dreamer-v3 models in src/models/ (dreamer_v3_nnx, dreamer_v3_trainer,
or any other file in src.models). The only allowed shared import is
src.algorithms.dreamer_srl.utils (symlog / symexp).

SYMLOG-SPACE BIN DISCIPLINE — triple-consistency contract
---------------------------------------------------------
These three locations all say the same thing about where bins live:

  1. v2 archived plan cascade item #2:
       "Two-hot bins as linspace(-20, +20, 255) stored in SYMLOG space"
     (docs/develop/archive/dreamer_srl/IMPLEMENTATION_PLAN.md)

  2. v3 plan Checkpoint 5 spec:
       "bins[0]=-20, bins[127]≈0, bins[254]=+20 (in symlog space)"
     (docs/develop/active/dreamer_srl_v3/IMPLEMENTATION_PLAN.md §CP5 row)

  3. This file (TwoHotEncoding class):
       self.bins = jnp.linspace(-20.0, 20.0, 255)  — SYMLOG-space grid
       symexp is applied ONLY at consumption (mean/mode), NOT at storage.

The math-reviewer will spot-check this triple. If any site says otherwise,
that is the historical bug recurring.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp

from src.algorithms.dreamer_srl.utils import symexp, symlog


class TwoHotEncoding:
    """Two-hot encoding for reward / critic head distributions.

    Ported from sheeprl@33b6366:sheeprl/utils/distribution.py:L224-L276
    (TwoHotEncodingDistribution class).

    CRITICAL — bin grid lives in SYMLOG space, NOT real reward space.
    The 255-bin linspace(-20, +20, 255) is stored in symlog space; the
    bin CENTERS in real space are symexp(bins). symexp is applied ONLY
    at consumption (mean/mode via transbwd); the stored grid is never
    symexp-transformed. Targets are symlog-encoded BEFORE bin lookup
    (transfwd = symlog inside log_prob).

    Historical scar: the v1 cascade stored the bin grid in real reward
    space (self.bins = symexp(linspace(-20, 20, 255))). Losses decreased
    to the wrong basin. Static plan review did not catch it. Lever A
    bit-identity tests + Lever B source citations exist to prevent the
    recurrence. See docs/develop/archive/dreamer_srl/IMPLEMENTATION_PLAN.md
    cascade item #2 for the full history.

    Triple-consistency check (math-reviewer will verify all three sites):
      - v2 archived plan cascade item #2 — "symlog space"
      - v3 plan CP5 row — "bins[0]=-20, bins[127]≈0, bins[254]=+20"
      - This class docstring + __init__ — "SYMLOG space, NOT real"

    GOTCHA: bins[127] is the float32 linspace midpoint of linspace(-20, 20, 255).
    Due to float32 rounding it is 7.45e-8, not exactly 0.0. The bit-identity
    test asserts |bins[127]| < 1e-6, matching sheeprl's actual float32 value.
    Do NOT "fix" this to exactly 0.0 — that would diverge from sheeprl.

    Args:
        logits: [... , 255] unnormalized logits for the 255-bin distribution.
        dims  : number of event dimensions to reduce over in log_prob (default 0).
                Use dims=1 for reward/critic losses where the event dim has size 1.
        low   : lower endpoint of the bin grid in symlog space (default -20,
                verbatim from sheeprl's TwoHotEncodingDistribution default).
        high  : upper endpoint of the bin grid in symlog space (default +20,
                verbatim from sheeprl's TwoHotEncodingDistribution default).

    Bit-identity tests:
        tests/algorithms/dreamer_srl/test_loss.py::test_twohot_bins_endpoints_match_sheeprl
        tests/algorithms/dreamer_srl/test_loss.py::test_twohot_encode_matches_sheeprl
        tests/algorithms/dreamer_srl/test_loss.py::test_twohot_log_prob_matches_sheeprl
    """

    def __init__(
        self,
        logits: jax.Array,
        dims: int = 0,
        low: int = -20,
        high: int = 20,
    ) -> None:
        """Initialize the distribution.

        Ported from sheeprl@33b6366:sheeprl/utils/distribution.py:L225-L243
        (TwoHotEncodingDistribution.__init__).

        CRITICAL: self.bins is linspace(low, high, n_bins) in SYMLOG SPACE.
        Do NOT apply symexp here. symexp is for the consumption side only (mean/mode).
        This is the exact storage form sheeprl uses at L237:
            self.bins = torch.linspace(low, high, logits.shape[-1], device=logits.device)
        """
        # sheeprl L234: self.logits = logits
        self.logits = logits
        # sheeprl L235: self.probs = F.softmax(logits, dim=-1)
        self.probs = jax.nn.softmax(logits, axis=-1)
        # sheeprl L236: self.dims = tuple([-x for x in range(1, dims + 1)])
        self.dims = tuple([-x for x in range(1, dims + 1)])
        # sheeprl L237: self.bins = torch.linspace(low, high, logits.shape[-1])
        # SYMLOG SPACE — this is the critical line. bins[0]=-20, bins[254]=+20 in symlog.
        # Do NOT wrap with symexp. symexp belongs at consumption (mean/mode), not storage.
        self.bins = jnp.linspace(low, high, logits.shape[-1])
        self.low = low
        self.high = high
        # Note: transfwd and transbwd are fixed to symlog/symexp (sheeprl L231-L232 defaults)
        # and are not stored as callables — we call symlog/symexp directly in the methods.

    @property
    def mean(self) -> jax.Array:
        """Mean of the distribution in REAL space (symexp applied at consumption).

        Ported from sheeprl@33b6366:sheeprl/utils/distribution.py:L245-L247
        (TwoHotEncodingDistribution.mean property).

        sheeprl L247:
            return self.transbwd((self.probs * self.bins).sum(dim=self.dims, keepdim=True))

        CRITICAL: symexp (transbwd) is applied HERE at consumption, not at storage.
        self.bins is in symlog space; (probs * bins).sum() is a symlog-space
        expected value; symexp maps it back to real space.

        Bit-identity test: tests/algorithms/dreamer_srl/test_loss.py::test_twohot_log_prob_matches_sheeprl
        """
        # keepdims=True matches sheeprl's keepdim=True
        return symexp(jnp.sum(self.probs * self.bins, axis=self.dims, keepdims=True))

    @property
    def mode(self) -> jax.Array:
        """Mode of the distribution in REAL space (identical to mean for two-hot).

        Ported from sheeprl@33b6366:sheeprl/utils/distribution.py:L249-L251
        (TwoHotEncodingDistribution.mode property).

        sheeprl L251:
            return self.transbwd((self.probs * self.bins).sum(dim=self.dims, keepdim=True))
        """
        return symexp(jnp.sum(self.probs * self.bins, axis=self.dims, keepdims=True))

    def log_prob(self, x: jax.Array) -> jax.Array:
        """Log-probability of target x under the two-hot distribution.

        Ported from sheeprl@33b6366:sheeprl/utils/distribution.py:L253-L276
        (TwoHotEncodingDistribution.log_prob method).

        GOTCHA: x is the RAW target in real reward space. The first step is
        symlog-encoding x (transfwd = symlog). The bin lookup then happens in
        SYMLOG SPACE — this is bit-identical with sheeprl's computation.
        If you skip symlog(x) here, the bin indices are computed in real space
        against a symlog-space bin grid, which is the historical bug.

        sheeprl L253-L276 (verbatim structure):
            x = self.transfwd(x)                         # symlog-encode
            below = (self.bins <= x).sum(-1, keepdim=True) - 1   # left bin index
            above = below + 1                             # right bin index
            above = torch.minimum(above, ...)             # clamp to [0, n_bins-1]
            below = torch.maximum(below, ...)             # clamp to [0, n_bins-1]
            equal = below == above
            dist_to_below = where(equal, 1, |bins[below] - x|)
            dist_to_above = where(equal, 1, |bins[above] - x|)
            total = dist_to_below + dist_to_above
            weight_below = dist_to_above / total          # NOTE: cross-weight (dist_to_above → weight_below)
            weight_above = dist_to_below / total          # NOTE: cross-weight
            target = one_hot(below) * weight_below + one_hot(above) * weight_above
            log_pred = logits - logsumexp(logits)
            return (target * log_pred).sum(dim=self.dims)

        GOTCHA: the cross-weight assignment (dist_to_above gives weight_below) is
        NOT a bug — it is the correct linear interpolation: the closer to the
        ABOVE bin center (large dist_to_below), the more weight goes to BELOW,
        and vice versa. Match sheeprl exactly.

        Args:
            x: [... , 1] target in real reward space (pre-symlog).

        Returns:
            log_prob: [...] scalar log-probability (event dims summed out).

        Bit-identity test: tests/algorithms/dreamer_srl/test_loss.py::test_twohot_log_prob_matches_sheeprl
        """
        n_bins = self.bins.shape[0]

        # sheeprl L254: x = self.transfwd(x)
        x = symlog(x)  # encode target to symlog space before bin lookup

        # sheeprl L256: below in [-1, len(self.bins) - 1]
        # (self.bins <= x) counts how many bins are <= x; subtract 1 → left bin index
        below = (self.bins <= x).astype(jnp.int32).sum(axis=-1, keepdims=True) - 1

        # sheeprl L258: above = below + 1
        above = below + 1

        # sheeprl L261: above = torch.minimum(above, torch.full_like(above, len(self.bins) - 1))
        above = jnp.minimum(above, n_bins - 1)

        # sheeprl L263: below = torch.maximum(below, torch.zeros_like(below))
        below = jnp.maximum(below, 0)

        # sheeprl L265: equal = below == above
        equal = below == above

        # sheeprl L266-L267: distances from x to the adjacent bin centers
        # When equal (x on or outside grid boundary): both distances are 1 (equal weight)
        dist_to_below = jnp.where(equal, jnp.ones_like(x), jnp.abs(self.bins[below] - x))
        dist_to_above = jnp.where(equal, jnp.ones_like(x), jnp.abs(self.bins[above] - x))

        # sheeprl L268: total = dist_to_below + dist_to_above
        total = dist_to_below + dist_to_above

        # sheeprl L269-L270: cross-weights (correct linear interpolation)
        weight_below = dist_to_above / total  # closer to above → more weight to below
        weight_above = dist_to_below / total  # closer to below → more weight to above

        # sheeprl L271-L274: two-hot target via one_hot scatter + squeeze
        # JAX equivalent of: F.one_hot(below, n_bins) * weight_below[..., None]
        #                   + F.one_hot(above, n_bins) * weight_above[..., None]
        # then .squeeze(-2)
        target = (
            jax.nn.one_hot(below, n_bins) * weight_below[..., None]
            + jax.nn.one_hot(above, n_bins) * weight_above[..., None]
        )
        target = jnp.squeeze(target, axis=-2)  # remove the keepdim=True axis from below/above

        # sheeprl L275: log_pred = self.logits - torch.logsumexp(self.logits, dim=-1, keepdims=True)
        log_pred = self.logits - jax.scipy.special.logsumexp(self.logits, axis=-1, keepdims=True)

        # sheeprl L276: return (target * log_pred).sum(dim=self.dims)
        return (target * log_pred).sum(axis=self.dims)
