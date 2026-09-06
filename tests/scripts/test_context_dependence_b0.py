"""B0 (bush-entry rate) in `scripts/analysis/context_dependence.py`, pinned against a
hand-built trajectory store whose correct answer is known by inspection.

What B0 is, in one sentence: of the times the agent was standing in the open and *decided to
take a step*, how often did that step land it in a concealing bush.

Why this file exists
--------------------
B0 has one structural trap and two semantic ones, and all three yield a plausible wrong number
rather than an error, so none of them would be caught by running the script and looking at the
output.

1. **Row alignment.** In the trajectory store, row `t` carries the environment state *at* `t`
   together with the action that *arrived at* `t` (`docs/environment/TRAJECTORY_STORE_SCHEMA.md`
   §1). An entry event is therefore `in_bush[t] == False` **and** `action[t+1]` is a move **and**
   `in_bush[t+1] == True` — the deciding row is `t`, but both the action and the outcome are read
   off row `t+1`. Reading the action from row `t` instead, or the outcome from row `t`, still
   produces a number.
2. **Freezing must leave the denominator, not enter the numerator.** An agent that rests in a
   bush is not entering it. The whole point of B0 over a bush-occupancy measure is that a frozen
   agent contributes nothing in either direction.
3. **`Eat` is excluded too.** The denominator is "the agent decided to move", not "the agent was
   not resting".

The fixture below is built so that each of those mistakes changes the answer. Episodes 4's
bush flips on a `Rest`/`Eat` row are physically impossible (nothing but a move can change which
cell the agent occupies) and are deliberately synthetic: they are tripwires that only a broken
implementation can stand on.

The fixture is also checked to be *discriminating*: `test_fixture_discriminates_misalignment`
recomputes B0 with the row alignment shifted by one and asserts it disagrees with the reference
values, so this file cannot silently degrade into a test that passes on a broken implementation.
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))), "scripts", "analysis"))
import context_dependence as cd  # noqa: E402

# One predator (slot 0), one neutral animal (slot 1) that is inactive all episode.
LAY = {"pred": [0], "neutral": [1]}
FAR = (9, 9)      # Chebyshev distance from every agent cell used below is > NEAR_D

# t, in_bush, action, rested, agent_row, agent_col, predator_row, predator_col
EPISODES = {
    # --- ep 1, start injury 0.0 -> bin "injury 0". Predator always far. -------------------
    #  t=0 open  -> move -> bush            : DENOMINATOR + ENTRY
    #  t=1 bush                             : excluded (already in cover)
    #  t=2 bush, rests, still bush          : excluded — freezing in a bush is not an entry
    #  t=3 open  -> Rest                    : excluded — froze in the open, leaves denominator
    #  t=4 open  -> move that does NOT move : DENOMINATOR (action variant only), no entry
    #  t=5 last row, no successor           : excluded
    1000: (0.0, [
        (0, False, -1, False, 0, 0, *FAR),
        (1, True,   0, False, 0, 1, *FAR),
        (2, True,   4, True,  0, 1, *FAR),
        (3, False,  1, False, 0, 2, *FAR),
        (4, False,  4, True,  0, 2, *FAR),
        (5, False,  2, False, 0, 2, *FAR),   # blocked move: chosen, but the cell is unchanged
    ]),
    # --- ep 2, start injury 60.0 -> bin ">=50". Predator always far. ----------------------
    #  t=0 open -> Rest        : excluded (froze in the open)
    #  t=1 open -> move -> bush: DENOMINATOR + ENTRY   <- the off-by-one probe
    #  t=3 bush, rests, bush   : excluded
    1001: (60.0, [
        (0, False, -1, False, 5, 0, *FAR),
        (1, False,  4, True,  5, 0, *FAR),
        (2, True,   0, False, 5, 1, *FAR),
        (3, True,   4, True,  5, 1, *FAR),
        (4, False,  1, False, 5, 2, *FAR),
    ]),
    # --- ep 3, start injury 60.0 -> bin ">=50". Predator adjacent at t=0 only. ------------
    1002: (60.0, [
        (0, False, -1, False, 2, 2, 2, 3),   # predator at Chebyshev 1 -> "near"
        (1, False,  0, False, 2, 1, *FAR),
        (2, False,  0, False, 2, 0, *FAR),
    ]),
    # --- ep 4, start injury 30.0 -> bin "25-50". TRIPWIRES ONLY. -------------------------
    # Both bush flips arrive on a non-movement action, which cannot happen in the real
    # environment. Correct B0 has ZERO denominator steps here; an implementation that forgets
    # to exclude `Eat` scores 100%, and one that excludes neither `Eat` nor `Rest` scores 100%
    # on two steps.
    1003: (30.0, [
        (0, False, -1, False, 7, 0, *FAR),
        (1, True,   5, False, 7, 0, *FAR),   # Eat
        (2, False,  4, True,  7, 0, *FAR),   # Rest
        (3, True,   4, True,  7, 0, *FAR),   # Rest
        (4, False,  0, False, 7, 1, *FAR),
    ]),
    # --- ep 5, start injury 0.0 -> bin "injury 0". 30 rows, so the two measurement -------
    # windows differ: the single entry happens at t=26, i.e. OUTSIDE the first-25-step window.
    1004: (0.0, [
        (t, t in (27, 28), -1 if t == 0 else 0, False, 3, t, *FAR) for t in range(30)
    ]),
}

# ---------------------------------------------------------------- expected, by hand ----
# (window, variant, predator condition) -> {bin: (denominator, entries)}
EXPECTED = {
    ("whole_episode", "action", "far"): {
        "injury 0": (29, 2),    # ep1: 2 den / 1 entry  +  ep5: 27 den / 1 entry
        "0-25":     (0, 0),
        "25-50":    (0, 0),     # ep4's two flips both arrive on Eat / Rest
        ">=50":     (2, 1),     # ep2 t=1 (entry)  +  ep3 t=1
    },
    ("whole_episode", "action", "near"): {
        "injury 0": (0, 0), "0-25": (0, 0), "25-50": (0, 0), ">=50": (1, 0),  # ep3 t=0
    },
    ("first_25", "action", "far"): {
        "injury 0": (28, 1),    # ep1: 2/1  +  ep5: 26 den / 0 entries (its entry is at t=26)
        "0-25":     (0, 0), "25-50": (0, 0), ">=50": (2, 1),
    },
    ("whole_episode", "displacement", "far"): {
        "injury 0": (28, 2),    # ep1's blocked move drops out: 1 den / 1 entry  +  ep5: 27/1
        "0-25":     (0, 0), "25-50": (0, 0), ">=50": (2, 1),
    },
    ("first_25", "displacement", "far"): {
        "injury 0": (27, 1), "0-25": (0, 0), "25-50": (0, 0), ">=50": (2, 1),
    },
    ("whole_episode", "displacement", "near"): {
        "injury 0": (0, 0), "0-25": (0, 0), "25-50": (0, 0), ">=50": (1, 0),
    },
}


# ------------------------------------------------------------------------ fixture ----
def _write_store(root) -> str:
    """Write the hand-built episodes above as a one-shard trajectory store."""
    store = os.path.join(root, "")
    os.makedirs(root, exist_ok=True)
    seeds, t, act, rested, bush, inj, ar, ac, anr, anc = ([] for _ in range(10))
    for sd, (inj0, rows) in EPISODES.items():
        for (tt, bu, a, rs, r, c, pr, pc) in rows:
            seeds.append(sd); t.append(tt); act.append(a); rested.append(rs); bush.append(bu)
            # injury_level is a per-step state column; only its t=0 value is used (the
            # environment's randomised starting draw). Give it a decaying trajectory so a
            # binner that reads the CONTEMPORANEOUS value instead lands in a different bin.
            inj.append(inj0 if tt == 0 else max(inj0 - 20.0 * tt, 0.0))
            ar.append(r); ac.append(c)
            anr.append([pr, 0]); anc.append([pc, 0])
    pq.write_table(pa.table({
        "episode_seed": pa.array(seeds, pa.int64()), "t": pa.array(t, pa.int16()),
        "action": pa.array(act, pa.int8()), "rested": pa.array(rested, pa.bool_()),
        "agent_in_bush": pa.array(bush, pa.bool_()),
        "injury_level": pa.array(inj, pa.float32()),
        "agent_row": pa.array(ar, pa.int16()), "agent_col": pa.array(ac, pa.int16()),
        "animal_row": pa.array(anr, pa.list_(pa.int16())),
        "animal_col": pa.array(anc, pa.list_(pa.int16())),
    }), os.path.join(root, "steps_00000.parquet"))
    pq.write_table(pa.table({
        "episode_seed": pa.array(list(EPISODES), pa.int64()),
        # slot 0 (the predator) active, slot 1 (the neutral) inactive, every episode
        "animal_active": pa.array([[True, False]] * len(EPISODES), pa.list_(pa.bool_())),
    }), os.path.join(root, "episodes_00000.parquet"))
    return store


@pytest.fixture(scope="module")
def swept(tmp_path_factory):
    store = _write_store(str(tmp_path_factory.mktemp("b0store")))
    return cd.sweep_entry([store], LAY, verbose=False)


# -------------------------------------------------------------------------- tests ----
def test_counts_match_hand_computation(swept):
    """Every (window, variant, predator) cell equals what the fixture says by inspection."""
    ENT, _, _ = swept
    wi = {w: i for i, w in enumerate(cd.WINDOWS)}
    vi = {v: i for i, v in enumerate(cd.VARIANTS)}
    ci = {"far": 0, "near": 1}
    for (w, v, cond), per_bin in EXPECTED.items():
        A = ENT[wi[w], vi[v], :, ci[cond], :]
        for bi, lab in enumerate(cd.ENTRY_LABELS):
            den, num = per_bin[lab]
            assert (A[bi].sum(), A[bi, 1]) == (den, num), \
                f"{w}/{v}/{cond} bin {lab}: got den={A[bi].sum():.0f} num={A[bi,1]:.0f}, " \
                f"expected den={den} num={num}"


def test_freezing_in_a_bush_is_never_an_entry(swept):
    """Episode 2 t=3 (rests while in a bush) and episode 1 t=2 must contribute nothing.

    Both would be counted by any implementation whose denominator is "the agent was not
    resting" rather than "the agent was in the open and chose to move"."""
    ENT, _, _ = swept
    total_den = ENT[cd.WINDOWS.index("whole_episode"), cd.VARIANTS.index("action")].sum()
    # ep1: t=0, t=4 · ep2: t=1 · ep3: t=0, t=1 · ep4: none · ep5: t=0..26
    assert total_den == 2 + 1 + 2 + 0 + 27


def test_rest_and_eat_are_both_excluded(swept):
    """Episode 4 is entirely tripwire: correct B0 has no denominator steps in its bin."""
    ENT, _, _ = swept
    bi = cd.ENTRY_LABELS.index("25-50")
    assert ENT[:, :, bi, :, :].sum() == 0


def test_predator_condition_is_read_at_the_deciding_row(swept):
    """Episode 3's predator is adjacent only at t=0, so exactly one 'near' denominator step
    exists — read at row t, not at row t+1."""
    ENT, _, _ = swept
    near = ENT[cd.WINDOWS.index("whole_episode"), cd.VARIANTS.index("action"), :, 1, :]
    assert near.sum() == 1 and near[cd.ENTRY_LABELS.index(">=50")].sum() == 1


def test_window_split_is_not_a_no_op(swept):
    """Episode 5's only entry lands at t=26, so it is in the whole-episode window and not in
    the first-25-step one. A window that silently pooled would hide this."""
    ENT, _, _ = swept
    bi = cd.ENTRY_LABELS.index("injury 0"); vi = cd.VARIANTS.index("action")
    assert ENT[cd.WINDOWS.index("first_25"), vi, bi, 0, 1] == 1
    assert ENT[cd.WINDOWS.index("whole_episode"), vi, bi, 0, 1] == 2


def test_binning_uses_the_randomised_start_not_the_current_value(swept):
    """The fixture's injury decays 20 per step, so episode 2 (start 60) spends most of its rows
    at injury 0. Its single denominator step must still be filed under '>=50'."""
    ENT, _, _ = swept
    A = ENT[cd.WINDOWS.index("whole_episode"), cd.VARIANTS.index("action"), :, 0, :]
    assert A[cd.ENTRY_LABELS.index(">=50")].sum() == 2      # ep2 t=1 and ep3 t=1
    assert A[cd.ENTRY_LABELS.index("injury 0")].sum() == 29  # ep1 + ep5, never ep2/ep3


def test_action_index_convention_is_verified_not_assumed(swept):
    """The script must derive which action is `Rest` from the store's own `rested` column."""
    _, _, diag = swept
    assert diag["rest_action_values"] == [4]
    assert set(diag["action_values"]) == {-1, 0, 1, 2, 4, 5}
    assert diag["displaced_without_move_action"] == 0
    assert diag["n_episodes"] == len(EPISODES)


def test_delta_b0_and_ci(swept):
    """Δ_B0 = top populated bin − bottom populated bin, in percentage points."""
    ENT, _, _ = swept
    m = cd.entry_metrics(ENT, min_n=1)["whole_episode|action|far"]
    assert m["bins_used"] == ["injury 0", ">=50"]
    assert not m["substituted"]
    assert m["delta_b0"] == pytest.approx(100 * (1 / 2 - 2 / 29), abs=1e-9)
    assert m["delta_b0_ci95"] > 0
    # B3 = B0(near) − B0(far) inside a bin; only ">=50" has both sides populated here.
    b3 = cd.entry_metrics(ENT, min_n=1)["whole_episode|action|B3"]
    assert [r["bin"] for r in b3["rows"]] == [">=50"]
    assert b3["rows"][0]["threat_response"] == pytest.approx(0.0 - 50.0)


def test_min_n_blanks_and_substitutes_bins(swept):
    """A bin below the qualifying-step floor drops out and the next bin inward is used."""
    ENT, _, _ = swept
    m = cd.entry_metrics(ENT, min_n=3)["whole_episode|action|far"]
    assert m["bins_used"] == ["injury 0"] or np.isnan(m["delta_b0"])
    rows = {r["bin"]: r for r in m["rows"]}
    assert np.isnan(rows[">=50"]["b0"])          # only 2 denominator steps
    assert np.isfinite(rows["injury 0"]["b0"])   # 29


def test_b1_rest_rate(swept):
    """B1 is the fraction of ACTING rows (t >= 1) on which `Rest` was chosen, per bin."""
    _, REST, _ = swept
    R = cd.entry_rest_metrics(REST)
    whole = {r["bin"]: r for r in R["whole_episode"]["rows"]}
    assert whole["injury 0"]["rest_rate"] == pytest.approx(100 * 2 / (5 + 29))
    assert whole[">=50"]["rest_rate"] == pytest.approx(100 * 2 / (4 + 2))
    assert whole["25-50"]["rest_rate"] == pytest.approx(100 * 2 / 4)


def test_fixture_discriminates_misalignment(swept):
    """The guard on this file: a row alignment shifted by one must give a DIFFERENT answer.

    Without this, `test_counts_match_hand_computation` could be passing for the wrong reason —
    a fixture on which the correct and the broken estimator happen to agree proves nothing.
    Here both plausible slips are checked: reading the action from the deciding row instead of
    the arrival row, and reading the outcome from the deciding row.
    """
    ENT, _, _ = swept
    ref = ENT[cd.WINDOWS.index("whole_episode"), cd.VARIANTS.index("action"), :, 0, :]

    # The independent reference must first reproduce the implementation exactly, otherwise a
    # "they differ" assertion below would prove nothing about the alignment.
    assert np.array_equal(_b0_counts_shifted(False, False), ref)

    for shift_action, shift_outcome in ((False, True), (True, False), (True, True)):
        got = _b0_counts_shifted(shift_action, shift_outcome)
        assert not np.array_equal(got, ref), (
            f"fixture does not discriminate a shift (action={shift_action}, "
            f"outcome={shift_outcome}); it would pass on a broken implementation")


def _b0_counts_shifted(shift_action: bool, shift_outcome: bool):
    """Independent, pure-Python recomputation of the action-variant B0 counts, `far` rows only.

    `shift_*` = True means "read it from the deciding row `t` instead of from row `t+1`" — the
    off-by-one an implementer makes when they forget the store's arrival convention. With both
    flags False this is the correct estimator and must reproduce `sweep_entry`.

    Returns counts shaped (injury bin, 2), last axis [did not enter, entered], matching
    `sweep_entry`'s ENT[..., 0/1].
    """
    out = np.zeros((len(cd.ENTRY_LABELS), 2))
    for _sd, (inj0, rows) in EPISODES.items():
        b = int(np.digitize(inj0, cd.INJ_EDGES))
        for i in range(len(rows) - 1):
            _t, bush_now, _a, _r, ar, ac, pr, pc = rows[i]
            if max(abs(pr - ar), abs(pc - ac)) <= cd.NEAR_D:
                continue                                    # 'near' row — this helper is far-only
            a = rows[i][2] if shift_action else rows[i + 1][2]
            outcome = rows[i][1] if shift_outcome else rows[i + 1][1]
            if (not bush_now) and a in cd.MOVE_ACTIONS:
                out[b, 1 if outcome else 0] += 1
    return out
