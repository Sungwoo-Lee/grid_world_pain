---
id: 20260512_1755_pytorch_agents_pip_dep_layout
date: 2026-05-12
time: "17:55"
folder: cluster_ops
tags: [meta, decision, design, training_runner, learned_lesson]
summary: "Third-party RL frameworks (e.g., sheeprl) should be integrated as pip-installed git-pinned dependencies with our extensions in a sibling in-repo package (e.g., pytorch_agents/), not as edited `tmp/` clones; `tmp/` is gitignored and silently loses edits."
related: []
session_origin: claude_code
session_label: "dreamer_srl plan + PI pivot to sheeprl-direct"
importance: high
status: settled
supersedes: []
raw_source: claude_data/.claude/projects/-media-nas01-projects-Interoceptive-AI-grid-world-pain/268d07a3-2eac-4772-858a-c44fb812d10d.jsonl
raw_completeness: full
---

# Integrate third-party RL frameworks as pip dep + in-repo extension package, not as `tmp/` clones

## Key conclusion

When integrating a third-party RL framework (here: sheeprl) where we want to add our own env wrappers and config overrides on top, the right structural pattern is: install the framework as a `pip`-installed git-pinned dependency, and keep ONLY our additions as a sibling in-repo Python package (here: `pytorch_agents/`) with its own `pyproject.toml`. Do NOT clone the framework into `tmp/` and edit files in place. `tmp/` is gitignored, so edits live on NAS-mounted disk only and are silently lost on a fresh clone or `git clean -x`. The sibling-package pattern uses the framework's own extension hooks (here: sheeprl's `SHEEPRL_SEARCH_PATH` Hydra plugin) to load our configs and modules without touching upstream source. Layout chosen: `pytorch_agents/` (not `pytorch_dreamer/` or `sheeprl_ext/`) — accommodates the user's long-term intent to migrate rPPO and other agents to PyTorch without forcing a rename.

## Evidence, measurements, facts

- Earlier setup: `tmp/sheeprl/` was a clone of `Eclectic-Sheep/sheeprl@33b6366` with 4 net-new files (env wrapper, env Hydra config, exp Hydra config, logger WandB config) and zero modifications to upstream-tracked files. Senior-developer survey confirmed via `git -C tmp/sheeprl diff --stat HEAD` empty.
- Because `tmp/sheeprl/` was gitignored (`.gitignore:23`), all 4 of our additions lived on NAS-only — invisible to git history.
- Sheeprl ships a Hydra `SearchPathPlugin` at `tmp/sheeprl/hydra_plugins/sheeprl_search_path.py` that reads `SHEEPRL_SEARCH_PATH` (semicolon-separated `pkg://...` paths) — designed exactly for this case.
- New layout: `pytorch_agents/pytorch_agents/{envs,configs/{env,exp,logger}}/` with `pytorch_agents/pyproject.toml` declaring `sheeprl @ git+https://github.com/Eclectic-Sheep/sheeprl@33b636681fd8b5340b284f2528db8821ab8dcd0b` as a dep.
- Hydra resolves our configs via `SHEEPRL_SEARCH_PATH="pkg://pytorch_agents.configs"`. The 4 `configs/*/__init__.py` files are MANDATORY for the `pkg://` resolution to work.
- The `_target_` in our env config changed from `sheeprl.envs.grid_world_pain.GridWorldPainWrapper` (namespace squat under upstream sheeprl) → `pytorch_agents.envs.grid_world_pain.GridWorldPainWrapper` (own namespace, no upstream collision).
- 4 commits landed the restructure: `0067721` (skeleton + 4-file move), `7a2cc83` (`launch_sheeprl.sh` rewrite), `e8b8ed9` (docs + training-runner profile), `f8f00b1` (plan report).
- `tmp/sheeprl/` deleted after CPs passed (backed up to `/tmp/tmp_sheeprl_bk_<epoch>` first per the git-safety rule about gitignored data).
- CP-v2-2 Hydra dry-run on node 114 confirmed: `SHEEPRL_SEARCH_PATH='pkg://pytorch_agents.configs' python -m sheeprl --cfg job exp=dreamer_v3_grid_world_pain` composes the experiment cleanly.

## Decisions and actions

- Adopted Option B (pip dep + in-repo extension package) over Option A (vendor everything in-repo) and Option C (git submodule). Option A rejected because we have zero upstream mods — vendoring costs ~600 files of maintenance for no benefit. Option C rejected because no other submodules in the project; UX cost unjustified.
- Folder name: `pytorch_agents/` — chosen specifically because the user signaled long-term intent to migrate rPPO and other agents to PyTorch ("I will move on to all pytorch usages, including rppo"). Narrower names like `pytorch_dreamer/` or `sheeprl_ext/` would force a rename when rPPO arrives.
- Repo-root `pyproject.toml` (JAX/Flax stack) stays untouched. The two pyprojects coexist because they install into different conda envs (`grid_world_pain` for JAX, `sheeprl_bridge` for PyTorch).
- The 4 in-repo files (`pytorch_agents/pytorch_agents/envs/grid_world_pain.py` + 3 YAML configs under `pytorch_agents/pytorch_agents/configs/`) are now first-class git-tracked code, not NAS-only.
- Install command for new nodes: `pip install -e pytorch_agents/ --config-settings editable_mode=compat` (see related insight on the `editable_mode=compat` requirement).

## Open questions and follow-ups

- The `pytorch_agents/pyproject.toml` declares `jax[cpu]>=0.9.0` because the env wrapper imports the JAX env from `src/environments/grid_world_pain/`. This creates a dep-resolution conflict with sheeprl's `numpy<2.0` cap (see related insight). Senior-developer follow-up: decide whether to keep jax declared (and bump sheeprl pin to a numpy-2.0-compatible commit) or remove it (and rely on the env having jax pre-installed via the main project).
- Future rPPO migration: when rPPO ports to PyTorch, it lands under `pytorch_agents/pytorch_agents/agents/` or similar — folder name already accommodates this.
- `launch_sheeprl.sh` does not yet have a `--noise on/off` flag pass-through. Developer task for a future session.

## References

- Active plan: [`docs/develop/active/sheeprl_bridge/IMPLEMENTATION_PLAN.md`](../../../docs/develop/active/sheeprl_bridge/IMPLEMENTATION_PLAN.md) — §"Layout (v2)" captures the option comparison and feasibility checks
- Related insight (same session): [`20260512_1754_sheeprl_direct_pivot_jax_dreamer_abandoned`](../dreamer_diagnosis/20260512_1754_sheeprl_direct_pivot_jax_dreamer_abandoned.md) — the strategic decision that triggered this restructure
- Related insight (same session): [`20260512_1756_pip_install_namespace_shadow_numpy_cap`](20260512_1756_pip_install_namespace_shadow_numpy_cap.md) — the two pip-install gotchas surfaced during node-114 rebuild
- Restructure commits: `0067721`, `7a2cc83`, `e8b8ed9`, `f8f00b1`
- Sheeprl search-path plugin (upstream, for reference): `sheeprl/hydra_plugins/sheeprl_search_path.py`
- Raw conversation: synced via `./sync-agent-data.sh claude push`. To read on another node: `./sync-agent-data.sh claude pull`, then either `claude --resume 268d07a3-2eac-4772-858a-c44fb812d10d` (re-enter the session) or `python scripts/claude_jsonl_to_md.py <jsonl> /tmp/<id>.md` (one-shot markdown view).
