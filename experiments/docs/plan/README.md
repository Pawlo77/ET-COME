# ET-COME Research Plan

**Method name:** ET-COME — *Equilibrium Transport with Conformal Minority Expansion*
**Scope:** Binary and multi-class classification on tabular data. **Benchmark policy:** [open-source datasets only](08-evaluation.md) (public URLs, redistribution-friendly licenses).

**Paper naming:** Use **execution-order headers** in prose (Setup → Step 1 …); letter labels **(Module A)** only after first mention — see [07-ablation.md](07-ablation.md).

---

## Document Structure

| File | Contents |
|------|----------|
| [01-overview.md](01-overview.md) | Motivation, novelty claim, v1→v2 changes |
| [02-method.md](02-method.md) | Core algorithm: Setup, Modules A, B, C, D, G, E |
| [03-multiclass.md](03-multiclass.md) | Multi-class extension & Phase 13 scope gate |
| [04-debugging.md](04-debugging.md) | Module F: debug artifacts & Streamlit dashboard |
| [05-complexity.md](05-complexity.md) | Computational complexity analysis |
| [06-hyperparameters.md](06-hyperparameters.md) | Parameters, search grid, protocol, experiment budget |
| [07-ablation.md](07-ablation.md) | Ablation design (Baseline-A-simple, Sequential, …) |
| [08-evaluation.md](08-evaluation.md) | Benchmarks (OBS), metrics, baselines, minimum bar, power analysis |
| [09-related-work.md](09-related-work.md) | Related work positioning |
| [10-risks.md](10-risks.md) | Risks, mitigations, regime, pivot rules |
| [11-roadmap.md](11-roadmap.md) | Linear engineering phases (1–15), pre-registration checklist |
| [12-references.md](12-references.md) | Full bibliography |

---

## Quick Summary

ET-COME reorders the oversampling decision process (see [07-ablation.md](07-ablation.md) for **execution order vs. letter labels**):

1. **Once at setup — Module G:** optional inertia-based majority undersampling → `E⁰_final`
2. **Where** should synthesis focus? → **Module A** (OOB entropy partition — heuristic; validated vs. Baseline-A-simple)
3. **How** should mass move? → **Module B** (risk-targeted OT, or SMOTE-within-A if ν★ validation fails)
4. **Whether** candidates are plausible → **Module C** (OOB consistency screening)
5. **When** to stop → **Module E** (dual criterion or max_iter — report which)
6. Optional **Module D** (entropic positions); **Module F** = tooling only

All modules share a single OOB probability infrastructure when active (no validation set in the loop).

---

## Engineering phases

Single linear numbering — see [11-roadmap.md](11-roadmap.md). **Phase 1 (Day 1)** is hardware + open dataset manifest + fit-time — **nothing else gates without it**.

---

## Critical review — closure checklist (traceability)

| Topic | Where addressed |
|-------|-----------------|
| **Fair Sequential (E_A = T₀)** | [07-ablation.md](07-ablation.md), [01-overview.md](01-overview.md) |
| **Criterion 4** R² > 0.15 + slope CI + reviewer rationale vs token R² | [08-evaluation.md](08-evaluation.md) |
| **Criterion 0** ≥ 65% | [08-evaluation.md](08-evaluation.md) |
| **Power σ̂ rule** (Phase 5 lock) | [08-evaluation.md](08-evaluation.md) |
| **Compute ×1.5–2 + branch 4** (N=40, Optuna 150) | [06-hyperparameters.md](06-hyperparameters.md) |
| **Module G** main vs appendix | [02-method.md](02-method.md) §Module G — Decision rule (**Outcomes 1–3**); [08-evaluation.md](08-evaluation.md) Criterion 7 (pointer only) |
| **Title “Equilibrium”** governance + **default working title = fallback** | [08-evaluation.md](08-evaluation.md) |
| **Gate A** (synthetic + 3 pilots **before Phase 3**) | [11-roadmap.md](11-roadmap.md), [08-evaluation.md](08-evaluation.md) |
| **2D synthetic go/no-go** + decomposition fallback | [02-method.md](02-method.md); **Gate A** |
| **HNSW majority-neighbour diagnostic** | [02-method.md](02-method.md) Setup |
| **Proof-of-concept 3-dataset pilots** | [08-evaluation.md](08-evaluation.md) (**Gate A**) |
| **Venue before Phase 5** | [11-roadmap.md](11-roadmap.md), [03-multiclass.md](03-multiclass.md) |
| **Module G bisection in complexity** | [05-complexity.md](05-complexity.md) |
| **Prominent correlation diagnostic** | [08-evaluation.md](08-evaluation.md) |
