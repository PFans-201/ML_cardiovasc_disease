# Milestone M4: Learning Bayesian Network Structure

**Deadline:** November 10, 23:59  
**Status:** 📋 Planned

---

## 📋 Objectives

1. **Structure Learning Algorithms**
   - Implement constraint-based structure learning (PC algorithm)
   - Implement score-based structure learning (Hill Climbing with BIC/AIC)
   - Apply algorithms to cardiovascular disease dataset
   - Compare learned structures with M2 hand-designed network

2. **Algorithm Comparison**
   - Analyze differences between constraint-based and score-based approaches
   - Evaluate computational efficiency of each method
   - Assess sensitivity to sample size and data quality
   - Document advantages and limitations of each approach

3. **Structure Validation**
   - Compare learned structures with medical domain knowledge
   - Evaluate clinical plausibility of discovered relationships
   - Test inference performance on learned vs designed networks
   - Analyze which structure better captures data dependencies

4. **Causal Discovery Analysis**
   - Interpret learned edges in terms of potential causal relationships
   - Discuss limitations of observational data for causal inference
   - Compare discovered dependencies with known medical relationships
   - Identify surprising or counterintuitive discoveries

---

## 📊 Specific Tasks

### 9.1. Constraint-Based Learning (PC Algorithm)
- [ ] Implement PC algorithm using pgmpy's `PC` estimator
- [ ] Apply conditional independence tests (Chi-square, G-test)
- [ ] Tune significance threshold (alpha) parameter
- [ ] Generate learned network structure and visualize
- [ ] Document independence test results and decisions

### 9.2. Score-Based Learning (Hill Climbing)
- [ ] Implement Hill Climbing with BIC scoring
- [ ] Try alternative scoring functions (AIC, K2)
- [ ] Set appropriate search constraints (max parents, forbidden edges)
- [ ] Generate learned network structure and visualize
- [ ] Track search process and score improvements

### 9.3. Structure Comparison
- [ ] Compare learned structures (PC vs Hill Climbing vs M2 hand-designed)
- [ ] Calculate structural similarity metrics (SHD, precision, recall)
- [ ] Identify common edges across all methods
- [ ] Analyze edges unique to each approach
- [ ] Visualize structure differences side-by-side

### 9.4. Performance Evaluation
- [ ] Estimate CPTs for each learned structure
- [ ] Compare inference accuracy across structures
- [ ] Evaluate predictive performance (cross-validation)
- [ ] Measure computational cost of learning algorithms
- [ ] Test robustness to data subsampling

---

## 🎯 Key Requirements

- **Two Algorithms**: Must implement both constraint-based (PC) and score-based (Hill Climbing)
- **Structure Comparison**: Quantitative comparison with M2 hand-designed network
- **Clinical Validation**: Medical interpretation of learned relationships
- **Performance**: Inference and predictive accuracy comparison
- **Computational**: Runtime and scalability analysis

---

## 📁 Deliverables

1. **Jupyter Notebook**: `M4_G14.ipynb` with complete structure learning analysis
2. **Algorithm Implementation**: PC and Hill Climbing with parameter tuning
3. **Structure Visualization**: Clear network graphs for all learned structures
4. **Comparison Analysis**: Quantitative and qualitative structure comparison
5. **Clinical Interpretation**: Medical evaluation of discovered relationships

---

## 🔗 Connections to Other Milestones

- **M2**: Compares against hand-designed network structure
- **M3**: Uses same inference methods to evaluate learned structures
- **M5**: Structure learning insights inform HMM state design
- **M6**: Discovered treatment-outcome relationships guide RL reward design

---

## 🛠 Technical Notes

- **Data Requirements**: Use imputed and discretized dataset from M2
- **Independence Tests**: Chi-square for discrete variables, appropriate significance levels
- **Search Constraints**: Limit max parents to avoid overfitting, consider domain constraints
- **Scoring Functions**: BIC penalizes complexity, AIC less conservative
- **Visualization**: Use networkx or pgmpy plotting for clear structure comparison
- **Validation**: Cross-validation for predictive performance, medical literature for clinical validity