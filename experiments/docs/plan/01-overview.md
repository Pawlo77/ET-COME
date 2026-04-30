# Overview & Motivation

## What this is and why it is different

Oversampling methods fail in a specific, diagnosable way: they decide *how* to generate new points before knowing *where* the model actually needs help. SMOTE (Chawla et al., *JAIR* 16, 2002) picks two random minority neighbours and blends them. Borderline-SMOTE (Han et al., ICIC 2005) targets geometric proximity to the boundary, which has no connection to what the current ensemble actually gets wrong. The result is synthesis in regions the model may already handle well, and silence in regions where it is genuinely confused.

ET-COME reorders these decisions. It first identifies where the ensemble is uncertain for learnable (not irreducible) reasons, then moves synthetic mass toward those regions via optimal transport, then discards any generated point the ensemble's own out-of-bag scores (Breiman, *Machine Learning* 45(1), 2001) flag as implausible.

### Why this should outperform class-weighted LightGBM

Recent tabular imbalance surveys (e.g. arXiv:2402.03819, 2024 — class weighting vs. resampling) establish that class weighting often beats naive oversampling. Class weighting adjusts the loss function but leaves the training distribution unchanged — the model sees the same geometry, just penalised differently. At low-to-moderate imbalance ratios this is often sufficient. At high IR (> 30), the minority class occupies a small, fragmented feature-space region, and the decision boundary is shaped almost entirely by majority geometry. Class weighting cannot fix this because the boundary itself is drawn from the wrong distribution.

ET-COME modifies the training distribution, adding minority mass specifically where the ensemble is confused. We hypothesize that this mechanistic design leads to advantage over class-weighted GBDT that increases monotonically with imbalance ratio (IR), because learnable uncertainty becomes the dominant failure mode at high IR. This hypothesis will be tested via stratified benchmark results (see [08-evaluation.md](08-evaluation.md)); if not supported, the paper will investigate whether assumptions fail at high IR or whether design rationale requires revision.

---

## Novelty claim (must be validated in Phase 2 literature gate — gates Phase 5 implementation)

Prior work has addressed synthesis location (conformal synthesis; arXiv:2312.08999), uncertainty-guided admissibility (UQDIR), and OT-based placement (P²OT; arXiv:2401.09266) independently. **To our knowledge**, ET-COME is the first method in supervised tabular imbalanced learning to jointly condition synthesis location, placement, and admission on a *shared OOB uncertainty infrastructure*: the same OOB probability matrix that identifies where to synthesise (Module A) also drives what to synthesise (Module B) and whether to accept each candidate (Module C). This architectural coupling — not the individual techniques — is the proposed contribution.

> A systematic literature search was conducted across arXiv (cs.LG, stat.ML), Google Scholar, and major ML venue proceedings (ICML/NeurIPS/ICLR/ECML/PAKDD/AAAI/CIKM, 2016–2026) using five query strings: {"imbalanced learning" + "OOB"}, {"imbalanced learning" + "epistemic uncertainty"}, {"imbalanced learning" + "optimal transport"}, {"oversampling" + "conformal"}, {"minority synthesis" + "ensemble uncertainty"}. The search evidence log is in `docs/search_log_2026-04-29.md`.

**No paper found that combines any two of: (a) OOB-based uncertainty decomposition for synthesis targeting, (b) OT-based placement, (c) OOB-based admission screening.** The three-way combination from a shared infrastructure is novel.

**Closest single-component predecessors — each requires explicit differentiation in the paper:**

| Component | Closest prior work | Key differentiator |
|---|---|---|
| Module A: uncertainty-guided targeting | CGMOS (CIKM 2016, arXiv:1607.06525) | Uses generic classifier certainty, not OOB; no epistemic/aleatoric decomposition |
| Module A: uncertainty-guided targeting | D'souza et al. AAAI 2025 (arXiv:2412.15657) | Uses RF cross-validation folds (not OOB); identifies overlap zone rather than learnable uncertainty; no OT, no screening |
| Module A: RF decomposition framework | Shaker & Hüllermeier DS 2020 (arXiv:2001.00893) | Formally decomposes RF uncertainty into epistemic/aleatoric — but never applied to synthesis or imbalanced learning |
| Module B: OT placement | OTOS (AAAI 2019) | Only prior paper using true OT coupling for minority synthesis on tabular data; uses loss signal (not OOB); no uncertainty decomposition; no conformal screening |
| Module C: synthetic admission screening | SMOGAN (arXiv:2504.21152, 2025) | Uses GAN discriminator for screening (not OOB ensemble); requires training a generative model; no OT placement |
| Module C: conformal filtering of synthetic data | "Filtering with Confidence" (arXiv:2509.21479, 2025) | Uses held-out calibration set (not OOB); not imbalance-specific; no OT placement |

**Required paper framing:** the novelty is the *shared OOB infrastructure* that enables three-way coupling without a separate validation set, calibration set, or generative model. Each individual component has a partial predecessor; the integration architecture and the OOB-only design do not.

The related work table in [09-related-work.md](09-related-work.md) has been updated with all six predecessors. The "to our knowledge" qualifier and this search evidence summary appear in the paper's related work section.

---

## Module interdependence (architectural vs. incidental)

A reviewer will ask whether this is three methods glued together. The proposed answer is architectural interdependence: Module A's admissibility mask restricts Module B's transport support, Module C reuses Module A's OOB predictions for a different purpose, and Module E's stopping rule requires both B and C to be interpretable.

**Figure requirement:** One diagram in the main paper showing **E⁰ → shared OOB matrix → Module A → Module B → Module C** with Module E feeding back — see [07-ablation.md](07-ablation.md) execution-order table.

**Counterfactual experiment:** Ablation alone does not answer "sequential pipeline with *independent* ensemble signals." **Baseline-Sequential** ([07-ablation.md](07-ablation.md)) uses **E_A with T₀ trees** matching ET-COME’s E⁰ (fair variant); a secondary **equal-total** split isolates tree-count effects. If the **AUPRC gap** between ET-COME-ABC and **fair Baseline-Sequential** is **≤ ~2%**, treat architectural coupling as **weak / incidental** unless pre-registered robustness checks pass.

This claim will be validated empirically via ablation (see [07-ablation.md](07-ablation.md)). If Baseline-AB alone beats SMOTE significantly, or Baseline-AC alone is sufficient, then the claimed interdependence is weakened and the paper must acknowledge modular value separately.

---

## What changed from v1 (architectural redesign, not a patch)

v1 identifies bad-performance regions using the validation set (`_identify_bp` runs on `X_val, y_val` inside the training loop — leakage at every step). v1 also uses fixed scalar thresholds (`br_threshold`, `bp_theta`) with no principled interpretation.

v2 is an architectural redesign, not a corrected version of v1. The uncertainty estimation mechanism is replaced wholesale: validation-set bad-performance region detection → OOB mutual-information decomposition into epistemic and aleatoric components (Module A). A synthesis admission criterion absent from v1 is introduced: OOB-consistency screening (Module C). The placement mechanism changes from neighbour interpolation to risk-targeted Sinkhorn OT (Module B). The stopping signal changes from a fixed iteration count to a dual convergence criterion on transport plan stability and OOB interval width (Module E). Adaptive majority undersampling (Module G) is entirely new.

These changes are not interchangeable with v1 equivalents: the OOB infrastructure works as designed precisely because all modules consume the same OOB probability matrix from the same ensemble. Framing v2 as "v1 with the leakage fixed" would misrepresent the scope of redesign and would obscure the architectural coupling that is the claimed contribution. ENN pre-cleaning (Wilson, *IEEE Trans. SMC* 2(3), 1972) is kept unchanged — it runs once, uses no validation labels, and improves graph quality.
