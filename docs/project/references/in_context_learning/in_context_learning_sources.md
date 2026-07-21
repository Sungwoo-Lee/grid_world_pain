# In-context learning — reference corpus (source manifest)

**Type**: reference-topic source manifest + download record
**Created**: 2026-07-21
**Cross-links**: [[Hypernetwork]] (the project's hypernetwork reference topic — group C below is the hypernetwork↔ICL bridge); [[continual_learning_field_evolution]] (sibling reference topic, same folder conventions).

## Purpose — what this topic is

This folder collects the papers behind **in-context learning (ICL)** — the ability of a
sequence model (a Transformer) to "learn" a new task at inference time purely from
examples in its prompt, with *no weight updates*. The corpus is organised around one
throughline the group cares about: **ICL is a form of Bayesian inference / amortized
inference**, and it connects to **hypernetworks** (a network that outputs another
network's weights). The 15 papers split into three groups (below): the Mila/Lajoie–Bengio
line that frames ICL via Occam's razor and latent-variable inference; the broader
"ICL = implicit Bayesian inference" lineage; and the newer hypernetwork↔ICL equivalence
papers.

## Sourcing policy (version of record vs. preprint)

PDFs were fetched with the `academic-pdf-fetch` skill, **preferring the published version
of record** (PMLR / NeurIPS open-access) and falling back to arXiv otherwise. Each row
records which version is on disk. Four papers are published at **ICLR**, whose version of
record lives on OpenReview (Cloudflare-walled); their camera-ready is content-identical to
arXiv, so the arXiv PDF is stored — the exact OpenReview camera-ready can be fetched via the
skill's Tier-3 browser path on request. Four papers have **no published version** (arXiv
preprints), two of which (#11, #14) were flagged uncertain by the requester but were
**confirmed to exist** on arXiv with matching titles.

All PDFs live in `./sources/` and are magic-byte verified.

## Group A — Mila / Lajoie–Bengio group (Elmoznino & collaborators)

| # | Paper | Venue | arXiv | On-disk version |
|---|---|---|---|---|
| 1 | Elmoznino et al. — In-context learning and Occam's razor | **ICML 2025** (PMLR v267) | 2410.14086 | **published** |
| 2 | Mittal et al. — Does learning the right latent variables necessarily improve ICL? | **ICML 2025** (PMLR v267) | 2405.19162 | **published** |
| 3 | Hu et al. — Amortizing intractable inference in large language models | **ICLR 2024** | 2310.04363 | arXiv (≈ OpenReview camera-ready) |

## Group B — ICL = Bayesian-inference lineage

| # | Paper | Venue | arXiv | On-disk version |
|---|---|---|---|---|
| 4 | Xie et al. — An Explanation of ICL as Implicit Bayesian Inference | **ICLR 2022** | 2111.02080 | arXiv (≈ OpenReview) |
| 5 | Müller et al. — Transformers Can Do Bayesian Inference (PFNs) | **ICLR 2022** | 2112.10510 | arXiv (≈ OpenReview) |
| 6 | Garg et al. — What Can Transformers Learn In-Context? (simple function classes) | **NeurIPS 2022** | 2208.01066 | **published** |
| 7 | von Oswald et al. — Transformers Learn In-Context by Gradient Descent | **ICML 2023** (PMLR v202) | 2212.07677 | **published** |
| 8 | Reuter et al. — Can Transformers Learn Full Bayesian Inference in Context? | **ICML 2025** (PMLR v267) | 2501.16825 | **published** |
| 9 | Mittal et al. — Amortized In-Context Bayesian Posterior Estimation | preprint (2025) | 2502.06601 | arXiv (preprint) |
| 10 | Mittal et al. — In-Context Parametric Inference: Point or Distribution Estimators? | preprint (2025) | 2502.11617 | arXiv (preprint) |
| 11 | Kang, Lee & Cheng — Transformers Can Learn Posterior Predictive Distributions In-Context | preprint (2026) | 2605.26713 | arXiv (preprint) |

## Group C — Hypernetwork ↔ in-context learning

| # | Paper | Venue | arXiv | On-disk version |
|---|---|---|---|---|
| 12 | Schug et al. — Attention as a Hypernetwork | **ICLR 2025 (Oral)** | 2406.05816 | arXiv (≈ OpenReview) |
| 13 | Schlag, Irie & Schmidhuber — Linear Transformers Are Secretly Fast Weight Programmers | **ICML 2021** (PMLR v139) | 2102.11174 | **published** |
| 14 | Mittal et al. — Iterative Amortized Inference: Unifying ICL and Learned Optimizers | preprint (2025) | 2510.11471 | arXiv (preprint) |
| 15 | Chen et al. — Exact Conversion of ICL to Model Weights in Linearized-Attention Transformers | **ICML 2024** (PMLR v235) | 2406.02847 | **published** |

## Status

15/15 downloaded and verified. **7 published version-of-record** (PMLR/NeurIPS: #1, #2, #6, #7, #8, #13, #15); **4 ICLR** stored as arXiv camera-ready-equivalent (#3, #4, #5, #12); **4 arXiv preprints** with no published version (#9, #10, #11, #14). No per-paper review has been produced yet — this manifest is the source record only.
