# Engineering Roadmap & Pre-Registration

## Linear phase schedule (single numbering — no fractional phases)

| Phase | What gets built | Duration | Buffer | Notes |
|-------|-----------------|----------|--------|-------|
| **1 — Access & compute gate** | **Day 1 (blocking):** Hardware spec, OSF credential check; **manifest of N open binary datasets** (URLs, licenses, checksums); fit-time benchmark on **5 largest OBS datasets** by n×d; contingency if median fit > 60 s | **3 days** | — | **Nothing else starts until Gate 1 completes** — compute & data access gate the entire budget ([06-hyperparameters.md](06-hyperparameters.md)) |
| **2 — Literature & statistics** | Systematic lit search (days 2–5); novelty freeze; **target venue declared** (gates multi-class scope); **OSF:** fallback title strings + title governance; power σ-branches; Bayesian test verification; **2D synthetic decomposition gate** (complete **before Phase 5** — same artefact as [02-method.md](02-method.md)) | **1 week** | — | Synthetic gate + venue are **blocking** for decomposition narrative and Phase 13 relevance |
| **Gate A — Empirical de-risk** | **Blocks Phase 3.** Minimal competitive evidence **before** package/layout spend: (1) deliver **2D synthetic** figure + go/no-go log ([02-method.md](02-method.md)); (2) **3 KEEL datasets** (low/med/high IR), **untuned** ET-COME vs SMOTE+RF, same folds — table + script hash; (3) **termination split** (dual vs backstop) on those runs. **Stop/pivot if:** synthetic fails decomposition go/no-go **or** ET-COME loses on **≥ 2/3** pilots **or** dual-criterion rate **< 50%** with no bug — document OSF pivot before continuing ([08-evaluation.md](08-evaluation.md) POC pilots). | **≤ 1 week** | — | Addresses “no preliminary results” risk **before** ~24-week commitment scales |
| **3** | Package layout; StandardScaler + HNSW; SCORING updates; baseline runners; tree infrastructure; complete OSF pre-registration | 1 week | — | — |
| **4** | Debug infrastructure (Module F): artifact engine, Streamlit GUI, integration tests | 1 week | — | Required for pilot diagnosis |
| **5** | Module A; Module C; Module G (inertia + γ* bisection); acceptance-rate-vs-IR plots on 10 pilots (2 per IR stratum) | **3 weeks** | — | Pilot gate: if collapse or acceptance < 10%, pause |
| **6** | Spearman ρ validation (ρ > 0.75 gate); cost matrix; log-domain Sinkhorn (Module B) | 3 weeks | +1 week | Tight timeline |
| **7** | Full loop; Module E stopping; convergence plots; leakage audit | 2 weeks | — | — |
| **8 — Checkpoint** | Convergence/stability plots; if monotone on ≥ 8/10 pilots, proceed | **0.5 week** | — | Prevent benchmark on broken method |
| **9** | IR-stratified Optuna (TPE, 300 trials/stratum, 20-dataset corpus); sensitivity plots | **2.5 weeks** | +0.5 week | Start with 100 trials on 10 datasets first |
| **10 — Checkpoint** | Interim results; if AUPRC gain < 1% vs. SMOTE, stop and investigate | **0.5 week** | — | Early-exit gate |
| **11** | Ablations (all configurations in [07-ablation.md](07-ablation.md)); Module D + G checks | 2 weeks | +0.5 week | Includes Sequential + Baseline-A-simple |
| **12** | Full **OBS** benchmark (**N** datasets); Tier 0–3 baselines; statistical tests | **4.5 weeks** | +0.5 week | Checkpoint at 25 and 50 datasets |
| **13** | Multi-class infrastructure; 15+ KEEL datasets; per-class Sinkhorn; HP search — **see scope gate** ([03-multiclass.md](03-multiclass.md)) | 5 weeks | +1 week | Full codebase is binary at every layer — rewrite is substantial |
| **14** | Paper draft | 2 weeks | — | — |
| **15** | Diagnosis buffer | 2 weeks | — | — |
| **Total** | | **~25.5 weeks** | **+7 weeks (40%)** | +**Gate A ≤1 week**; includes debug infra, early-exit gates, contingency |

**Legacy alias (do not use in new prose):** old P0a → Phase 1–2; P0 → Phase 3; P0.5 → Phase 4; P3.5 → Phase 8; P4.5 → Phase 10; P6 → Phase 12; P7 → Phase 13.

---

## Immediate next steps

1. **Phase 1 — Day 1:** Hardware specification, **open dataset manifest** (N datasets, public URLs), fit-time on five largest OBS datasets. **Negative outcome protocol:** If median fit > 60 s or N < 60 feasible → immediately revise Phase 9 trial counts, wall-clock estimate, or OBS scope; document in OSF amendment — **do not** defer to end of week.
2. **Phase 2:** Complete systematic literature search; novelty framing frozen before Phase 5 implementation. See [01-overview.md](01-overview.md).
3. Pre-register: venue; **working title = fallback** (see [08-evaluation.md](08-evaluation.md) title governance); 20 tuning corpus + 20 ablation holdout + Criterion 0 decision tree + all 12 numbered minimum bar thresholds + power **σ-branches** + Criterion **4** spec + **2D synthetic artefact** deadline (**before Phase 5**) + **Baseline-Sequential OSF one-liner** ([07-ablation.md](07-ablation.md))
4. **Gate A — empirical de-risk:** Must complete **before Phase 3** — synthetic + 3 pilot wins vs SMOTE+RF + termination audit ([08-evaluation.md](08-evaluation.md)). **Do not** merge Phase 3 scaffolding until Gate A passes or OSF documents a pivot.
5. **Phase 3:** Package layout; StandardScaler + HNSW; baseline runners — **only after Gate A passes** (see **Gate A** row in the phase table above)
6. **Phase 4:** Implement debug artifact engine + Streamlit GUI
7. Implement EasyEnsemble and BalancedRandomForest baseline runners; add class-weighted ExtraTrees as Tier 0 baseline
8. Implement incremental tree-addition infrastructure
9. **Pilot gate:** Run Module A + C + G on 10 pilots (2 per IR stratum); if admissibility collapse or acceptance < 10%, pause
10. Validate ν★ surrogate ranking (**Spearman ρ > 0.75 on ≥ 8 of 10 pilots**); if soft band [0.65, 0.75], dual-report OT vs. SMOTE-within-A per [02-method.md](02-method.md); **STOP** primary OT claims if hard-stop triggers
11. **Checkpoint (Phase 8):** Confirm convergence on pilots (monotone on ≥ 8/10)
12. Start grid search with limited scope (100 trials, 10 datasets). **Checkpoint (Phase 10):** If gain < 1%, pause
13. Do **not** proceed to full benchmark until Phase 9 defaults are finalized and ablation confirms module value

Document all pivots, early-stops, and diagnoses in a supplementary "Method Development Log."

---

## Pre-Registration Checklist

### Phase 1 complete (hardware & open data — gates Phase 2 & Gate A prep)

- [ ] Exact hardware (CPU model, cores, RAM) documented in OSF
- [ ] **N ≥ 60** binary datasets in OBS with public URLs + licenses + checksums (KEEL + supplementary public only)
- [ ] Fit-time benchmark on 5 largest OBS datasets at T₀ = 50 trees; median recorded
- [ ] If median fit > 60 s: revised CPU-hour estimate and timeline amendment filed **before** Phase 9 Optuna at full scale

### Gate A complete (empirical de-risk — **blocks Phase 3**)

- [ ] **2D synthetic** artefact + go/no-go logged ([02-method.md](02-method.md))
- [ ] **3 KEEL pilots** (untuned ET-COME vs SMOTE+RF): results table + script/commit hash
- [ ] **Termination audit:** dual vs backstop counts on those runs
- [ ] **Pivot or proceed** documented in OSF if gate fails ([08-evaluation.md](08-evaluation.md))

### Literature search completed (Phase 2 — gates Phase 5)

- [x] Systematic lit search completed per protocol in [01-overview.md](01-overview.md) — **completed 2026-04-29**
- [x] Search evidence log saved (query strings, result counts, examined papers, outcome decision)
- [x] Novelty framing confirmed: three-way combination is novel; 6 single-component predecessors documented in [09-related-work.md](09-related-work.md) and [01-overview.md](01-overview.md)
- [x] **Phase 5 implementation may begin after Phases 3–4**

### Open datasets locked down

- [ ] 20 tuning corpus datasets (5 per IR stratum) with **public** download links
- [ ] 20 ablation holdout datasets (disjoint from tuning corpus)
- [ ] 15+ KEEL multi-class datasets identified and accessible
- [ ] Verify no dataset overlap between tuning corpus, ablation holdout, OBS benchmark list, and KEEL multi-class
- [ ] All datasets registered with IDs, URLs, IR values, and split ratios

### Pre-registration platform

- [ ] AsPredicted or OSF registration filed with:
  1. Exact dataset lists (OBS **N** datasets)
  2. Optuna search protocol (TPE, 300 trials/stratum, mean AUPRC, 5-fold CV)
  3. Criterion 0 decision tree and all 12 numbered minimum bar thresholds
  4. IR strata definitions and **Criterion 4** regression specification (**R² > 0.15**, slope CI > 0)
  5. Statistical test specifications
  6. All ablation configurations ([07-ablation.md](07-ablation.md)) including **fair Baseline-Sequential** (E_A = T₀)
  7. Tier 3 deterministic inclusion rule and claim governance
  8. **Power analysis** numeric worksheet ([08-evaluation.md](08-evaluation.md)) — include **δ = 0.02**, **σ** (pilot estimate), **N**, target power **≥ 85%**, **σ̂ > 0.07 branches**
  9. **Subjective priors for asymmetric win rates:** documented expectations vs. LightGBM / EasyEnsemble / SMOTE+RF (numeric or interval), locked **before** Phase 12 — satisfies non-circular threshold rationale ([08-evaluation.md](08-evaluation.md))
  10. **Fallback reporting pledge:** ET-COME_min vs ET-COME_full and module-active table ([08-evaluation.md](08-evaluation.md) §Contribution accounting)
  11. **Title governance:** primary + **fallback** title strings; **Equilibrium** rule ([08-evaluation.md](08-evaluation.md))
  12. **Compute branch 4:** N=40 + Optuna 150 if cluster denied ([06-hyperparameters.md](06-hyperparameters.md))

- [ ] Tier 3 availability evidence log frozen before Phase 9 full Optuna

### Code & reproducibility

- [ ] GitHub repository with dummy baseline implementations
- [ ] Dataset download scripts tested (open URLs only)
- [ ] Optuna trial logging infrastructure (save all trials as JSON)
- [ ] Interim results dashboard template (weekly updates, no p-hacking)

### Statistical infrastructure

- [ ] Bayesian signed-rank test implementation verified
- [ ] Power analysis: **σ-branches** (**σ̂ > 0.07**) locked **before Phase 5**; worksheet completed with δ, N, trials, power — update σ once after pilots **before Phase 12** ([08-evaluation.md](08-evaluation.md))
- [ ] Random seed strategy: Optuna seed=42, CV seeds [0,1,2,3,4], model seeds matched

### Pilot planning

- [ ] 10 KEEL pilot datasets chosen (2 per IR stratum); list locked before Phase 5 begins
- [ ] Pilot metrics defined: entropy sanity checks, Spearman ρ (**≥ 8/10 at ρ > 0.75**), convergence plots, acceptance rates, wall-clock, ENN removal rate
