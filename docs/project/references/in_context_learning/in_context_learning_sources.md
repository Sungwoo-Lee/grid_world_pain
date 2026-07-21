# In-context learning — reference corpus (source manifest)

**Type**: reference-topic source manifest + download record
**Created**: 2026-07-21 · **Last updated**: 2026-07-21 (expanded to 39 papers)
**Cross-links**: [[Hypernetwork]] (§5 is the hypernetwork↔ICL bridge to that topic); [[continual_learning_field_evolution]] (sibling reference topic, same conventions).

## Purpose — what this topic is

Papers behind **in-context learning (ICL)** — a Transformer's ability to "learn" a task at
inference time from prompt examples alone, with **no weight updates**. The corpus is built
around three interlocking readings the group cares about: **ICL as (implicit / amortized)
Bayesian inference**, **ICL as a learning algorithm run inside the forward pass** (gradient
descent / algorithm selection), and **ICL as weight generation** (the hypernetwork /
fast-weight-programmer view — the bridge to the project's hypernetwork/FiLM work).

## Sourcing policy (version of record vs. preprint)

Fetched with the `academic-pdf-fetch` skill. Published open-access PDFs (PMLR / NeurIPS)
were preferred where directly available; otherwise the **arXiv** version is stored (for the
many ICLR/NeurIPS/ICML papers here the arXiv camera-ready is content-identical to the version
of record, whose canonical copy sits on the Cloudflare-walled OpenReview). Every row notes the
venue + arXiv id + which version is on disk. All PDFs live in `./sources/`, magic-byte verified.

**Two known gaps (no legal OA copy):**
- **Schmidhuber (1992), "Learning to Control Fast-Weight Memories"** — *Neural Computation* 4(1),
  MIT Press; no arXiv, `isOA:false` everywhere. Obtainable only via a campus MIT-Press subscription
  through the Tier-3 browser path (subscription coverage of a 1992 issue is uncertain). **Not on disk.**

## §1 — ICL as Bayesian inference / model averaging

| # | Paper | Venue | arXiv | On-disk |
|---|---|---|---|---|
| 1 | Xie, Raghunathan, Liang, Ma — An Explanation of ICL as Implicit Bayesian Inference | ICLR 2022 | 2111.02080 | arXiv |
| 2 | Wang, Zhu, Saxon, Steyvers, Wang — LLMs Are Latent Variable Models (good ICL demonstrations) | NeurIPS 2023 | 2301.11916 | arXiv |
| 3 | Panwar, Ahuja, Goyal — In-Context Learning through the Bayesian Prism | ICLR 2024 | 2306.04891 | arXiv |
| 4 | Wies, Levine, Shashua — The Learnability of In-Context Learning | NeurIPS 2023 | 2303.07895 | arXiv |
| 5 | Elmoznino et al. — In-context learning and Occam's razor | ICML 2025 (v267) | 2410.14086 | **published** |
| 6 | Mittal et al. — Does learning the right latent variables necessarily improve ICL? | ICML 2025 (v267) | 2405.19162 | **published** |
| 7 | Hu et al. — Amortizing intractable inference in large language models | ICLR 2024 | 2310.04363 | arXiv |

## §2 — ICL as gradient descent / learning algorithms inside transformers

| # | Paper | Venue | arXiv | On-disk |
|---|---|---|---|---|
| 8 | Garg, Tsipras, Liang, Valiant — What Can Transformers Learn In-Context? (function classes) | NeurIPS 2022 | 2208.01066 | **published** |
| 9 | von Oswald et al. — Transformers Learn In-Context by Gradient Descent | ICML 2023 (oral) | 2212.07677 | **published** |
| 10 | Akyürek, Schuurmans, Andreas, Ma, Zhou — What learning algorithm is ICL? (linear models) | ICLR 2023 (oral) | 2211.15661 | arXiv |
| 11 | Dai et al. — Why Can GPT Learn In-Context? (secretly gradient descent as meta-optimizers) | ACL 2023 Findings | 2212.10559 | arXiv |
| 12 | Bai, Chen, Wang, Xiong, Mei — Transformers as Statisticians (in-context algorithm selection) | NeurIPS 2023 (oral) | 2306.04637 | arXiv |
| 13 | Ahn, Cheng, Daneshmand, Sra — Transformers learn to implement preconditioned gradient descent | NeurIPS 2023 | 2306.00297 | arXiv |
| 14 | Li, Ildiz, Papailiopoulos, Oymak — Transformers as Algorithms (generalization & stability) | ICML 2023 | 2301.07067 | arXiv |
| 15 | Zhang, Frei, Bartlett — Trained Transformers Learn Linear Models In-Context | JMLR 2024 (v25) | 2306.09927 | arXiv |

## §3 — Amortized Bayesian inference: PFN / TabPFN / neural processes / posterior estimation

| # | Paper | Venue | arXiv | On-disk |
|---|---|---|---|---|
| 16 | Müller, Hollmann, Pineda Arango, Grabocka, Hutter — Transformers Can Do Bayesian Inference (PFNs) | ICLR 2022 | 2112.10510 | arXiv |
| 17 | Hollmann, Müller, Eggensperger, Hutter — TabPFN | ICLR 2023 (oral) | 2207.01848 | arXiv |
| 18 | Garnelo et al. — Conditional Neural Processes | ICML 2018 | 1807.01613 | arXiv |
| 19 | Kim et al. — Attentive Neural Processes | ICLR 2019 | 1901.05761 | arXiv |
| 20 | Nguyen, Grover — Transformer Neural Processes | ICML 2022 | 2207.04179 | arXiv |
| 21 | Reuter, Rudner, Fortuin, Rügamer — Can Transformers Learn Full Bayesian Inference in Context? | ICML 2025 (v267) | 2501.16825 | **published** |
| 22 | Mittal et al. — Amortized In-Context Bayesian Posterior Estimation | preprint (2025) | 2502.06601 | arXiv (preprint) |
| 23 | Mittal et al. — In-Context Parametric Inference: Point or Distribution Estimators? | preprint (2025) | 2502.11617 | arXiv (preprint) |
| 24 | Kang, Lee, Cheng — Transformers Can Learn Posterior Predictive Distributions In-Context | preprint (2026) | 2605.26713 | arXiv (preprint) |

## §4 — Emergence / task-diversity / latent-variable conditions for ICL

| # | Paper | Venue | arXiv | On-disk |
|---|---|---|---|---|
| 25 | Chan et al. — Data Distributional Properties Drive Emergent ICL in Transformers | NeurIPS 2022 | 2205.05055 | arXiv |
| 26 | Raventós, Paul, Chen, Ganguli — Pretraining task diversity & emergence of non-Bayesian ICL | NeurIPS 2023 | 2306.15063 | arXiv |

## §5 — Hypernetworks & fast-weight programmers (attention ↔ weight generation)

| # | Paper | Venue | arXiv | On-disk |
|---|---|---|---|---|
| 27 | Ha, Dai, Le — HyperNetworks | ICLR 2017 | 1609.09106 | arXiv |
| 28 | Schmidhuber — Learning to Control Fast-Weight Memories | Neural Computation 4(1), 1992 | — | **❌ not OA (MIT Press paywall)** |
| 29 | Ba, Hinton, Mnih, Leibo, Ionescu — Using Fast Weights to Attend to the Recent Past | NeurIPS 2016 | 1610.06258 | arXiv |
| 30 | Munkhdalai, Yu — Meta Networks | ICML 2017 | 1703.00837 | arXiv |
| 31 | Schlag, Irie, Schmidhuber — Linear Transformers Are Secretly Fast Weight Programmers | ICML 2021 | 2102.11174 | **published** |
| 32 | Irie, Schlag, Csordás, Schmidhuber — Going Beyond Linear Transformers with Recurrent FWPs | NeurIPS 2021 | 2106.06295 | arXiv |
| 33 | Irie, Schlag, Csordás, Schmidhuber — A Modern Self-Referential Weight Matrix | ICML 2022 | 2202.05780 | arXiv |
| 34 | von Oswald, Henning, Sacramento, Grewe — Continual learning with hypernetworks | ICLR 2020 | 1906.00695 | arXiv |
| 35 | Zhmoginov, Sandler, Vladymyrov — HyperTransformer (few-shot model generation) | ICML 2022 | 2201.04182 | arXiv |
| 36 | Chen, Wang — Transformers as Meta-Learners for Implicit Neural Representations | ECCV 2022 | 2208.02801 | arXiv |
| 37 | Schug et al. — Attention as a Hypernetwork | ICLR 2025 (oral) | 2406.05816 | arXiv |
| 38 | Chen, Hu, Jin, Lee, Kawaguchi — Exact Conversion of ICL to Model Weights (linearized attention) | ICML 2024 (v235) | 2406.02847 | **published** |

## Status

**39 papers in the corpus; 38 PDFs on disk** (magic-byte verified), **1 gap** (#28 Schmidhuber 1992,
MIT-Press paywall). 8 stored as published version of record (PMLR/NeurIPS: #5, #6, #8, #9, #21, #31, #38 —
and NeurIPS #8); the rest as arXiv (ICLR/NeurIPS/ICML camera-ready-equivalent, or preprint for #22/#23/#24).

**Review status:** a per-paper literature review is in progress for the **original batch of 15**
(§ Mila/Lajoie group + Bayesian-lineage + hypernetwork subset from the first request). The **24 papers
added on 2026-07-21 are not yet reviewed** — they need a follow-up review pass.
