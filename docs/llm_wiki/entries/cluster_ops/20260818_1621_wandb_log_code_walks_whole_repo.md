---
id: 20260818_1621_wandb_log_code_walks_whole_repo
date: 2026-08-18
time: "16:21"
folder: cluster_ops
tags: [learned_lesson, training_runner, decision, wandb, meta]
summary: "wandb.run.log_code('.') walked the ENTIRE repo before every training run: wandb's filtered_dir() iterates os.walk(root) and DISCARDS the dirnames list, so it never prunes directories and stats every file under results/ (~372k recordings on CIFS). include_fn/exclude_fn do NOT help - they run AFTER the walk has visited each file. Cost scaled with accumulated eval scratch: ~8 min (07-03) -> 45-54 min (08-10) -> >1h40m (08-16), holding GPUs idle. Fixed (657c87a) with ANCHORED globs + wandb's InternalArtifact: 1.37s."
related: ["20260721_0421_eval_sweep_cpu_bound_not_nas", "20260805_0120_node114_nas_hang_sustained_io", "20260806_0306_gpu_claim_and_nas_git_lock_protocol"]
session_origin: claude_code
session_label: "rest-premium sweep + no-hiding-predator replicate + log_code fix"
importance: high
status: settled
valid_until: null
confidence: high
supersedes: []
raw_source: claude_data/.claude/projects/-media-nas01-projects-Interoceptive-AI-grid-world-pain/4efbe660-28c2-4643-b231-d3c6d2635b5a.jsonl
raw_completeness: full
---

# wandb log_code(".") walked the whole repo; startup cost grew with results/

## Key conclusion
Every training launch spent its first phase neither compiling nor training, but crawling `results/` over the NAS. `wandb.run.log_code(".")` snapshots source code by walking from the given root, and wandb never prunes directories, so `"."` means all ~372,000 files in the repo - almost all of them `.rec.gz` recordings. The cost is not constant: it grows every time an eval sweep leaves scratch recordings behind, so a latent 8-minute annoyance became a >1h40m stall that held ten GPUs idle at 0% utilisation.

## Evidence, measurements, facts
- ROOT CAUSE (wandb 0.24.0, `wandb/sdk/lib/filenames.py::filtered_dir`): it iterates `os.walk(root)` binding the dirnames element to `_` and discarding it. Pruning a walk requires mutating that list in place; wandb does not. So `include_fn` / `exclude_fn` are evaluated per FILE after the walk already visited it - they reduce UPLOADS, not TRAVERSAL. (I first proposed `exclude_fn` as the fix; it would have changed nothing. Reasoning about the API was wrong, reading its source was right.)
- MEASURED: `os.walk(".")` = 37,952 files in 45s **and still going** (of ~372k). `src/` = 193 files in 0.02s. `scripts/` = 77 files in 0.01s.
- ESCALATION: ~8 min (2026-07-03, first noted as a "log_code hang") -> ~45-54 min (2026-08-10, 8 concurrent arms) -> **>1h40m** (2026-08-16, 10 restpremNH arms). Stall signature: process alive, state `D`, wchan `wait_for_response` (CIFS), CPU advancing only ~13% of wall-clock, and an open fd deep inside an unrelated July `_scratch/.../recordings` tree.
- FIX (`657c87a`): build the same artifact from ANCHORED globs - `*.py` plus `src/**/*.py` plus `scripts/**/*.py`. Verified end-to-end against an offline wandb run: **1.37s**, 122 files, train.py + src/ 61 + scripts/ 49 captured, ZERO results/ paths, `code_path` set so the WandB Code tab still links.
- GOTCHA caught only by TESTING: the public `wandb.Artifact(name, type="code")` raises "Artifact type 'code' is reserved for internal use". wandb's own log_code uses `wandb.sdk.artifacts._internal_artifact.InternalArtifact`; the fix uses that with a public `type="source"` fallback. The first patch attempt crashed on this - a syntax/compile check would never have caught it.
- SILENT-FAILURE TRAP, now written into the code comment: an UNANCHORED `glob("**/*.py", recursive=True)` restores the full walk AND fails invisibly - `results/` holds no `.py`, so the consuming loop never ticks while ~372k paths are scanned (empirically: zero output in 60s).

## Decisions and actions
- Shipped the anchored-glob snapshot in `train.py` (`657c87a`). Running jobs unaffected - they had already loaded train.py into memory.
- Left `results/` untouched: gitignored training data, not mine to prune. Archiving finished sweeps' `_scratch` trees is a separate, complementary cleanup - git, backups and `du` are still slow (`du -sh` on one subtree timed out at 2 min).

## Open questions and follow-ups
- Archive or prune the old `_scratch` recording trees (preserve the data, move it out of the working tree).
- No persistent JAX compile cache is configured for TRAINING (the eval pipeline sets one). Adding `JAX_COMPILATION_CACHE_DIR` would remove the remaining genuine compile cost on repeat launches.

## References
- Explains why ten correctly-launched runs looked idle to monitoring (270 MiB / 0% util, `gpu_status` reporting FREE): the claim-detection rule is [[20260806_0306_gpu_claim_and_nas_git_lock_protocol]]. Note also that lab-node PIDs are container-namespaced and never match `nvidia-smi` host PIDs, which defeats naive process-to-GPU correlation.
- Related NAS/CIFS pathologies: [[20260805_0120_node114_nas_hang_sustained_io]], [[20260721_0421_eval_sweep_cpu_bound_not_nas]].
- Raw conversation: synced via `./sync-agent-data.sh claude push`; `claude --resume 4efbe660-28c2-4643-b231-d3c6d2635b5a`.

<!-- BACKLINKS — auto-generated by scripts/regen_wiki_graph.py; do not edit -->
## Backlinks
- _no inbound links yet_
<!-- END BACKLINKS -->
