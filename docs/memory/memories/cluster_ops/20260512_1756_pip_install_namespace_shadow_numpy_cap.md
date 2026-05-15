---
id: 20260512_1756_pip_install_namespace_shadow_numpy_cap
date: 2026-05-12
time: "17:56"
folder: cluster_ops
tags: [meta, learned_lesson, training_runner, decision]
summary: "Two pip-install gotchas surfaced during the node-114 sheeprl_bridge env rebuild: outer/inner namespace package shadow (fix: --config-settings editable_mode=compat); sheeprl@33b6366 declares spurious numpy<2.0 cap that conflicts with env's numpy 2.4.4 (fix: --no-deps on the affected installs)."
related: []
session_origin: claude_code
session_label: "dreamer_srl plan + PI pivot to sheeprl-direct"
importance: medium
status: settled
supersedes: []
raw_source: claude_data/.claude/projects/-media-nas01-projects-Interoceptive-AI-grid-world-pain/268d07a3-2eac-4772-858a-c44fb812d10d.jsonl
raw_completeness: full
---

# Two pip-install gotchas during shared-env rebuild: namespace shadow + spurious numpy cap

## Key conclusion

When installing a new in-repo Python package (`pytorch_agents/`) plus a git-pinned third-party framework (`sheeprl@33b6366`) into an existing shared conda env (`sheeprl_bridge` on node 114), two distinct pip-install failure modes surfaced — both have clean workarounds, both are worth recognizing on sight because they recur in other shared-env contexts. (1) **Outer/inner namespace package shadow**: standard setuptools editable install puts the package finder at the END of `sys.meta_path`, so Python's default `PathFinder` finds the OUTER `pytorch_agents/` directory (no `__init__.py` → namespace package) BEFORE the editable finder reaches the inner `pytorch_agents/pytorch_agents/` package. Fix: `pip install -e pytorch_agents/ --config-settings editable_mode=compat` — flips the finder priority. (2) **Spurious upstream version cap**: sheeprl@33b6366's pyproject hard-caps `numpy<2.0` (defensive), but the env's existing `numpy 2.4.4` was already there from prior installs AND sheeprl's runtime actually works fine with numpy>=2.0 (proved by `jzgkcep4` smoke run). Fix: `pip install ... --no-deps` to bypass pip's clean resolution while accepting the runtime works.

## Evidence, measurements, facts

- **Gotcha 1 — namespace shadow** discovered during developer's CP-v2-1 install: `pip install -e pytorch_agents/` succeeded but `python -c "import pytorch_agents"` failed because the outer `pytorch_agents/` directory (without `__init__.py`) shadowed the inner package. Adding `--config-settings editable_mode=compat` fixed it.
- Why "compat" works: it falls back to the older setuptools editable mechanism (`MetaPathFinder` injected at `sys.meta_path[0]` instead of appended at the end). The new "strict" mode finder is correct in clean cases but loses to default `PathFinder` when there's a same-named outer directory.
- **Gotcha 2 — spurious numpy cap** discovered during the env rebuild step on node 114 (replaying developer's documented sequence): `pip install -e pytorch_agents --config-settings editable_mode=compat` triggered pip's full resolution, which tried to install sheeprl@33b6366 as a dep. Sheeprl declares `numpy<2.0`; env has `numpy 2.4.4`; `jax 0.10.0` (already in env) requires `numpy>=2.0`. Disjoint constraint set → `ResolutionImpossible`.
- The spurious-cap diagnosis: `jzgkcep4` smoke run ran sheeprl with numpy 2.4.4 successfully (validated survival ~500). Sheeprl's runtime does NOT actually require numpy<2.0 — the cap is defensive in pyproject metadata, not enforced at runtime.
- Workaround used:
  ```bash
  pip install -e /media/nas01/projects/Interoceptive-AI/grid_world_pain/pytorch_agents \
      --no-deps --config-settings editable_mode=compat
  pip install "sheeprl @ git+https://github.com/Eclectic-Sheep/sheeprl@33b636681fd8b5340b284f2528db8821ab8dcd0b" \
      --no-deps
  ```
- Post-install verification: `python -c 'import torch, jax, sheeprl, wandb, pytorch_agents'` succeeded; all versions reported correctly (torch 2.5.0+cu121, jax 0.10.0, sheeprl 0.5.8.dev0, pytorch_agents 0.1.0).
- CP-v2-2 Hydra dry-run, CP-v2-3 smoke launch (PID 24725, completed 2000 steps, WandB `z8350nmn`), CP-v2-4 parity launch (PID 25542, WandB `i4ulpn95`, running) — all confirmed the `--no-deps` install runs cleanly at training time.

## Decisions and actions

- The `--no-deps` workaround is the current install path on node 114. Documented in `docs/develop/active/diagnosis/sheeprl_training_howto.md` §5 and in the `training-runner` profile pre-flight notes.
- The `--config-settings editable_mode=compat` flag is documented for any node setting up the `sheeprl_bridge` env, since the outer/inner namespace issue will recur on every fresh install.
- The two workarounds are tactical, not permanent. Senior-developer follow-up named in `pytorch_agents/` plan's residual nits: either bump the sheeprl pin to a numpy-2.0-compatible commit (preferred), or restructure `pytorch_agents/pyproject.toml` to not declare jax-vs-numpy-conflicting transitives.
- For future shared-env work: recognize the namespace-shadow class of bug on `import` failures right after editable install. Recognize the spurious-cap class on `ResolutionImpossible` errors when the runtime evidence says the constraint isn't real.

## Open questions and follow-ups

- Senior-developer task: pick the permanent fix for the numpy<2.0 cap. Options: (a) bump sheeprl pin to a more recent commit that allows numpy>=2.0 (preferred — preserves "pip install just works" on fresh nodes); (b) restructure `pytorch_agents/pyproject.toml` to remove the conflicting jax declaration (the env wrapper would assume jax is pre-installed via the main project).
- Investigate whether sheeprl's upstream has fixed the numpy cap in a later commit. The pinned `33b6366` is from 2024-07-12; over a year has passed.
- The PyTorch deprecation warning `index_put_ on expanded tensors` from `sheeprl/algos/dreamer_v3/agent.py:658` (RSSM reset state internals) is benign for now but will become a hard error in a future PyTorch version — another reason to consider bumping the sheeprl pin.

## References

- Related insight (same session): [`20260512_1755_pytorch_agents_pip_dep_layout`](20260512_1755_pytorch_agents_pip_dep_layout.md) — the broader restructure that surfaced these gotchas
- Related insight (same session): [`20260512_1754_sheeprl_direct_pivot_jax_dreamer_abandoned`](../dreamer_diagnosis/20260512_1754_sheeprl_direct_pivot_jax_dreamer_abandoned.md) — the pivot that drove the env rebuild
- Install path documented in: [`docs/develop/active/diagnosis/sheeprl_training_howto.md`](../../../docs/develop/active/diagnosis/sheeprl_training_howto.md) §5
- Setuptools editable mode reference: <https://setuptools.pypa.io/en/latest/userguide/development_mode.html> (search "editable_mode=compat" — explains the strict/lax/compat trichotomy)
- Pip resolution backtracking guidance: <https://pip.pypa.io/warnings/backtracking>
- WandB smoke run that validated the install: [`z8350nmn`](https://wandb.ai/sungwoolee/grid_world_pain/runs/z8350nmn)
- Raw conversation: synced via `./sync-agent-data.sh claude push`. To read on another node: `./sync-agent-data.sh claude pull`, then either `claude --resume 268d07a3-2eac-4772-858a-c44fb812d10d` (re-enter the session) or `python scripts/claude_jsonl_to_md.py <jsonl> /tmp/<id>.md` (one-shot markdown view).

<!-- BACKLINKS — auto-generated by scripts/regen_memory_graph.py; do not edit -->
## Backlinks
- [[20260513_1417_jax_vmap_no_speedup_tiny_env]] (dreamer_diagnosis, 2026-05-13) — JAX-vmap parallel env over our 5×5 NoPred gridworld delivered no speedup vs shee
- [[20260513_2309_merge_path_manifest_tripwire]] (cluster_ops, 2026-05-13) — When a project carries 165GB of gitignored training data (results 39GB, wandb 11
<!-- END BACKLINKS -->
