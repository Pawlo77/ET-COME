# Computational Complexity

This is a question any reviewer will raise. The answer must be in the paper.

## Per-operation breakdown

| Operation | When | Complexity |
|-----------|------|------------|
| ENN cleaning | Once | O(n × k) |
| StandardScaler | Once | O(n × d) |
| HNSW graph build | Once | O(n log n) |
| Initial ensemble E⁰ | Once | O(T₀ × n log n × d) |
| Module G inertia scoring | Once (after E⁰) | O(n_maj × log n) — one HNSW query per majority point |
| Module G adaptive γ* search | Once (2–3 bisection refits) | O(3 × T₀ × n' log n' × d) where n' < n |
| Module A entropy decomp. | Per iteration, incremental | O(n × step_size) — only new trees' OOB |
| Module B cost matrix | Per iteration | O(n_minority × k) binary; O(C_minority × n̄_c × k) multi-class |
| Module B Sinkhorn | Per iteration, warm-started | O(C_minority × n̄_c × k × T_sinkhorn) |
| Module C screening | Per iteration | O(n_synthetic) binary; O(C_minority × n̄_synthetic_c) multi-class |
| New tree training | Per iteration | O(step_size × n' log n' × d) where n' = n − |R_γ*| |

---

## Total training cost

Per-iteration dominant term: **O(T₀ + L × step_size) × O(n' log n' × d)**, where n' = n − |R_γ*| after Module G.

**Additive one-time overhead (Module G bisection):** **+ O(3 × T₀ × n log n × d)** for **2–3 full E⁰ refits** over progressively smaller training sets — **not** folded into the per-iteration expression; include explicitly in complexity discussion and wall-clock budgets ([06-hyperparameters.md](06-hyperparameters.md) multiplier).

If the total tree budget is held fixed (e.g., T₀ + L × step_size = 100), the cost is *less than* a 100-tree RF on the original data when γ* > 0, because each tree trains on n' < n points.

The Sinkhorn overhead per iteration is O(n_minority × k × 20) — for n_minority = 100 and k = 15, that is roughly 30,000 weighted updates per iteration, which remains small relative to tree training.

---

## Module G efficiency gain

For n = 5,000, IR = 10 (n_maj = 4,500), γ* = 0.30:
- n' = 5,000 − 1,350 = 3,650
- Tree training cost scales as n' log n' / (n log n) ≈ 0.71 — a **29% per-tree speedup**
- Over L = 5 iterations with step_size = 5 (25 incremental trees): saves ~29% × 25 tree-fits
- One-time overhead (3 bisection refits): adds < 10% to initial E⁰ cost

---

## Practical overhead

For n = 5,000, T₀ = 50, step_size = 5, L = 5 (total 75 trees), with Module G removing γ* = 0.30 of majority (n_maj = 4,500 → remove 1,350 → n' = 3,650):

- Incremental tree training dominates but trains on 3,650 + synthetics rather than 5,000 + synthetics
- Sinkhorn + Module A + Module C together add < 5% to per-iteration wall-clock
- Module G adds one-time cost of ~3 E⁰ refits on progressively smaller data (< 10% overhead on E⁰)
- HNSW build is a one-time cost amortised over all iterations

The method is computationally comparable to training a 75-tree RF on ~73% of the original data, with one-time HNSW build and Module G bisection added. Report wall-clock against SMOTE+RF at equal total tree counts, not against SMOTE alone.

---

## Where the speed claim does NOT hold

For KEEL-era small datasets (n < 500), SMOTE's k-NN search is already negligible and the HNSW build adds proportionally more. Report times honestly, broken by dataset size. Do not claim a blanket speedup.
