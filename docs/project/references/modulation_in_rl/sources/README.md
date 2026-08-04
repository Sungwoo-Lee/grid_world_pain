# `modulation_in_rl/sources/` — raw source PDFs

**Read-only.** Reviews live at the topic root, not here:
[`../modulation_in_rl_lit_review.md`](../modulation_in_rl_lit_review.md).

17 PDFs are held. **Ten implement FiLM-style scale-and-shift modulation and are
reviewed in full**; six are retained as context only and are **not** reviewed; one
high-priority paper is deliberately absent.

## Reviewed in full (10)

| File | Venue as printed in PDF | Review |
|---|---|---|
| `Yuan 2024 - Unpacking the individual components of diffusion policy (preprint).pdf` | unstated (arXiv 2412.00084v1) | §6 |
| `Reuss et al. 2025 - FLOWER - democratizing generalist robot policies.pdf` | **CoRL 2025** | §7 |
| `Yoon et al. 2026 - PAPL - phase-aware policy learning via FiLM.pdf` | unstated (arXiv 2602.09370v2) | §8 |
| `Zhu et al. 2025 - EquAct - SE(3) equivariant FiLM.pdf` | unstated — "Preprint. Under review." | §9 |
| `Li et al. 2025 - CogVLA - modulation as routing.pdf` | **NeurIPS 2025** | §10 |
| `NVIDIA et al. 2025 - GR00T N1 (preprint).pdf` | unstated (industrial report) | §11 |
| `Marquis et al. 2026 - Hypernetwork-conditioned RL under actuator failures (preprint).pdf` | unstated (arXiv 2604.03392v1) | §12 |
| `Kang et al. 2026 - SplitAdapter - two-source factorised FiLM (preprint).pdf` | unstated (arXiv 2606.03297v1) | §13 |
| `Guo et al. 2026 - GEAR - drone aerobatics (preprint).pdf` | unstated (arXiv 2602.10997v1) | §14 |
| `Guo et al. 2026 - MoE-ACT (preprint).pdf` | unstated (arXiv 2603.15265v1) | §15 |

## Held as context only — NOT reviewed (6)

These do not implement scale-and-shift modulation. Where one bears on a question
*about FiLM* it appears in the review's §17 "Adjacent evidence" appendix, and nowhere
else; their own contributions are not assessed.

| File | Mechanism | Why excluded |
|---|---|---|
| `Tessera et al. 2024 - HyperMARL - adaptive hypernetworks for multi-agent RL.pdf` | full weight generation | hypernetwork. **Retained in §17** — principal counter-evidence on self-conditioning |
| `Zhang et al. 2025 - DyMoDreamer - world modeling with dynamic modulation.pdf` | 32×32 categorical latents **concatenated** into an RSSM state | not affine, and not modulation — concatenation |
| `Huang et al. 2024 - MENTOR - mixture of experts for single-task visual RL.pdf` | mixture-of-experts backbone | conditional computation, not conditional gain |
| `Grooten et al. 2025 - SPARC.pdf` | plain concatenation | negative control. **Retained in §17** |
| `Black et al. 2024 - pi0 - a vision-language-action flow model (preprint).pdf` | decoder-only MoE + cross-attention | **Retained in §17** as displacement evidence |
| `Li et al. 2025 - Neuro-Vesicles - critique of FiLM as neuromodulation (preprint).pdf` | position paper, no experiments | no evidence to contribute |

## Known gap

**e-nmRNN** — *Volume Transmission Implements Context Factorization to Target Online
Credit Assignment and Enable Compositional Generalization*, NeurIPS 2025 Poster,
OpenReview `S9Y89poypx`. **Deliberately not downloaded**, pending a separate decision.
It was the earlier surveys' provisional candidate for a grouped-modulation precedent,
so the review's answer to that question is stated over the reviewed corpus and does
**not** include it. Under the narrowed scope it would in any case have been
context-only (biophysically derived modulation, not affine).

Also uncovered: *Hyper-GoalNet* (NeurIPS 2025) and *HyPoGen* (ICLR 2025) — full-weight
hypernetwork papers, never downloaded, out of scope under the narrowed definition.

## Conventions

- Filenames follow `<Author> <year> - <short title>.pdf`; the year is the **arXiv v1**
  year and may differ from the venue year printed inside (e.g. HyperMARL's file says
  2024, the held PDF is v4 and states NeurIPS 2025).
- Text extracts used during review were written to `tmp/` and are not committed.
