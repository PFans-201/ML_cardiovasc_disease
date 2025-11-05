# Milestone M3: Inference in Bayesian Networks

**Deadline:** 7 November, 23:59  
**Status:** � In progress (exact inference implemented; approximate inference pending)

---

## 🧭 Scope & Goal

In M2 you designed a Bayesian Network (BN) and estimated CPDs over the imputed and discretized dataset. In M3 you will use that BN to perform inference and answer clinically meaningful questions such as: probability of disease given symptoms, likely treatment choices, and chances of good outcomes under specific therapies.

You will start with exact inference (Variable Elimination and Belief Propagation), then implement one sampling-based approximate method to compare accuracy and computational efficiency.

Constraints:
- Use the BN structure and CPDs derived in M2 (or re-estimate CPDs from the same imputed + discretized data if needed).  
- Use the imputed + discretized dataset from M2 (not the normalized one).

---

## ✅ Tasks (What to deliver)

### 8.1 Exact inference
Using pgmpy, compute posterior probabilities for variables of interest. For each query run BOTH Variable Elimination (VE) and Belief Propagation (BP). The posteriors should be nearly identical. If they differ, report and discuss possible reasons (e.g., loops, numerical issues, factorization choices).

Queries (adapt if your BN differs, but keep the intent: one diagnosis, one treatment, one outcome):
- a) $P(state = Healthy | chest_pain = 1, fatigue = 1)$
- b) $P(treatment | chest_pain = 1, fatigue = 1)$  → report most likely treatment
- c) $P(utility = High | risk_score = 4, treatment = DrugA)$

Expected artifacts:
- A compact table for each query with VE and BP posteriors side by side and a numerical equality check (e.g., max absolute diff < 1e-6).

### 8.2 Approximate inference
Implement one sampling-based approach for the same queries (choose one):
- Likelihood-Weighted Sampling (recommended for evidence)
- Gibbs Sampling
- Rejection Sampling (for comparison only; can be inefficient)

For each query:
- Compare the approximate posterior against exact (distance metric such as L1/L2/KL divergence; plus a bar plot).
- Study convergence vs number of samples (e.g., 1e3, 5e3, 1e4, 5e4): accuracy vs runtime plot.  
- For query (b), fix one treatment value and compare the univariate posterior (as required).

### 8.3 Query design (2–3 custom queries)
Pose 2–3 additional clinically meaningful queries and compute posteriors using both exact and approximate methods. Example: “P(state = Advanced | cholesterol = High, shortness_of_breath = 1)”. Discuss any interesting differences between methods.

---

## 📦 Deliverables

- Notebook: `./M3_G14.ipynb` containing:
  1) Tools/imports  
  2) Short recap of M2 preprocessing and discretization (or link to M2)  
  3) BN structure and CPD check (model.check_model())  
  4) Exact inference (VE & BP) for (a)–(c) with side-by-side results  
  5) Approximate inference implementation + comparisons  
  6) 2–3 custom queries (exact vs approximate)  
  7) Discussion: accuracy vs cost, limitations, and clinical interpretation

- Optional utility module (if code grows): `../../src/project/bn_sampling.py` with reusable sampling/evaluation helpers.

- Figures (auto-saved or exported from notebook) to `../../reports/figures/` when relevant.

---

## 🧪 Acceptance criteria

- VE and BP are executed for all three required queries; posteriors are reported and compared.
- At least one sampling-based method is implemented and evaluated for the same queries.
- Convergence analysis: accuracy improves with more samples; provide timing and distance-to-exact plots/tables.
- 2–3 additional queries are answered (both exact and approximate) and briefly discussed.
- Reproducibility: fixed random seeds, versioned environment, and no hardcoded local paths.

---

## ▶️ How to run

1) Environment
- Ensure pgmpy is installed in your active environment (see project `requirements.txt` or `environment.yml`).

2) Data
- Place/access raw CSVs at `../../data/raw/patients.csv` and `../../data/raw/encounters.csv` (or adjust paths in the notebook).  
- Use the M2 imputation + discretization strategy (the notebook includes a minimal recap). If you produced a processed dataset in M2, you can load that directly.

3) Notebook
- Open `M3_G14.ipynb` and run top-to-bottom.  
- Parameters to consider: number of samples (e.g., 1_000 → 50_000), random seed, and which sampler to use (LW / Gibbs).

4) Outputs
- The notebook prints the VE/BP results and generates comparison plots for approximate inference. Export any plots you need for the report to `../../reports/figures/`.

---

## � Notes

- If VE and BP disagree noticeably, verify the DAG contains no undirected cycles (or acknowledge polytree assumption is violated) and re-check CPDs/finite state names.  
- For likelihood weighting, ensure evidence variables are clamped and use normalized weighted counts to estimate posteriors.  
- When using Gibbs, warm-up and thinning help reduce autocorrelation; report effective sample size if possible.

---

## ✅ Quick checklist

- [ ] VE and BP implemented and compared for (a), (b), (c)  
- [ ] One sampler implemented (LW/Gibbs/Rejection)  
- [ ] Accuracy vs samples experiment + timing  
- [ ] 2–3 custom queries explored  
- [ ] Seeds set; paths relative; code cells clean  
- [ ] Figures saved when relevant  
- [ ] Short discussion written

---

## 🔗 Related

- Notebook in this folder: `M3_G14.ipynb`  
- Upstream data docs: `../../data/raw/README.md`  
- Project guidelines: `../../docs/guidelines.md`

---

## 🗓️ Status tracker (team)

- Exact inference (VE/BP): In progress / Done  
- Approximate inference (LW/Gibbs): Pending  
- Custom queries: Pending  
- Plots & write-up: Pending

---

Let’s wrap M3 with clear, reproducible inference results that connect back to clinical questions.
