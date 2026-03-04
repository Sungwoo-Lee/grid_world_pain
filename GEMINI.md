# Grid World Pain — Implementation Agent Instructions

## Documentation Protocol (CRITICAL)

When implementing a plan from `docs/`, you MUST update that document in real-time:

1. **On start**: Set `Status` to `IN PROGRESS`. Fill `Implemented by: Gemini` and `Date` in the Implementation Report. Always use `Gemini` as the agent name in `Implemented by` and `Verified by` fields.
2. **Each checkpoint**: After verifying, mark `- [ ]` → `- [x]` and append a one-line result (e.g., `- [x] No NaN — confirmed, 100 steps clean`).
3. **Each file change**: Log what was done under Implementation Report immediately — not at the end. Include any deviations from the plan and why.
4. **On error/blocker**: Append to Implementation Report with `**BLOCKER:**` prefix. Do not silently skip.
5. **On completion**: Set `Status` to `COMPLETED`. Fill the Verification Report table. Run final verification steps.
6. **On session interrupt**: The doc should already reflect all progress up to the last completed step — this is the resume point.

> The goal: if the session dies mid-task, another agent can read the doc and continue from where you left off.

## Environment

- **Python**: `/home/vncuser/miniconda3/envs/grid_world_pain/bin/python` (always use absolute path)
- **Framework**: JAX + Flax NNX. Ensure conda env is active.

## Configuration Protocol

- **No fallback defaults** for critical params in `train.py` / `evaluation.py`.
- Use `config.get_mandatory('key')` for all agent parameters.
- Missing YAML key → `ValueError`. Never hardcode defaults.

## Training & Evaluation

- **Debug flag**: Use `--debug` for troubleshooting. Always use `configs/environment/default.yaml` for debug runs (not ablation configs).
- **WandB**: Job type = `debugging`.
- **SPS check**: If changes may affect speed, measure and report training SPS before/after.

## Code Standards

- **JAX states**: All env logic must be compatible with JAX state management and `vmap`.
- **Config parity**: Algorithm changes require matching YAML config updates in `configs/models/`.
