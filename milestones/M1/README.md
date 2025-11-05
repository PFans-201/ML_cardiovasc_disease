# Milestone M1: Gaussian Mixture Models

**Deadline:** October 10, 23:59  
**Status:** 📋 Planned

---

## 📋 Objectives

1. **Feature Selection and Preprocessing**
   - Select appropriate features from encounters.csv for clustering
   - Implement missing data imputation strategy from M0
   - Standardize continuous variables for comparable scales
   - Prepare encounter-level dataset for independent observation analysis

2. **Gaussian Mixture Model Implementation**
   - Fit GMM with varying numbers of clusters (2-8)
   - Use information criteria (AIC/BIC) for optimal cluster selection
   - Handle mixed data types (continuous and binary features)
   - Assign encounters to most likely clusters

3. **Cluster Analysis and Interpretation**
   - Characterize clusters by symptom and lab feature distributions
   - Compare discovered clusters to true disease states
   - Identify clinical interpretations for each cluster
   - Assess alignment with medical knowledge

4. **Treatment and Outcome Analysis**
   - Analyze treatment variation across discovered clusters
   - Examine utility score patterns within clusters
   - Evaluate clinical relevance of cluster-based groupings

---

## 📊 Specific Tasks

### 6.1. Feature Selection and Preprocessing
- [ ] Select features from encounters.csv for clustering analysis
- [ ] Treat each encounter as independent observation
- [ ] Apply imputation method designed in M0
- [ ] Standardize/normalize continuous variables for scale comparability
- [ ] Document preprocessing decisions and rationale

### 6.2. Model Fitting
- [ ] Implement GMM with 2-8 clusters using sklearn
- [ ] Handle binary features as numeric 0/1 in GMM approximation
- [ ] Apply AIC/BIC criteria for optimal cluster number selection
- [ ] Assign each encounter to most probable cluster
- [ ] Validate model convergence and stability

### 6.3. Cluster Characterization
- [ ] Analyze symptom and lab feature distributions within clusters
- [ ] Compare cluster assignments to true disease states (Healthy, Early, Advanced)
- [ ] Quantify cluster-state alignment using appropriate metrics
- [ ] Identify whether clusters capture different groupings than disease states

### 6.4. Interpretation
- [ ] Provide clinical interpretation for each discovered cluster
- [ ] Analyze treatment distribution patterns across clusters
- [ ] Examine utility value variations by cluster membership
- [ ] Create summary tables and visualizations for cluster profiles

---

## 🎯 Key Requirements

- **Independent Encounters**: Each row in encounters.csv treated as separate observation
- **Mixed Data Handling**: Approximate binary features as numeric for GMM compatibility
- **Model Selection**: Use information criteria for principled cluster number choice
- **Clinical Interpretation**: Connect mathematical clustering results to medical insights
- **Comparison Analysis**: Systematic comparison between clusters and true disease states

---

## 📁 Deliverables

1. **Jupyter Notebook**: `M1_G14.ipynb` with complete GMM analysis
2. **Preprocessing Pipeline**: Documented and reusable data preparation code
3. **Cluster Analysis Report**: Detailed characterization of discovered groups
4. **Clinical Interpretation**: Medical reasoning for cluster patterns
5. **Visualization Portfolio**: Cluster distributions, feature profiles, and comparisons

---

## 🔗 Connections to Future Milestones

- **M2**: Cluster insights will inform variable selection for Bayesian network design
- **M3**: Understanding of patient groupings will guide inference query formulation
- **M4**: Cluster patterns will provide context for comparing hand-designed vs. learned BN structures
- **M5**: Discovered phenotypes may reveal relevant features for HMM temporal modeling

---

## 🛠 Technical Notes

- **GMM Approximation**: Binary symptoms treated as numeric 0/1 (acknowledge this approximation)
- **Feature Scaling**: Essential for GMM convergence with mixed-scale variables
- **Cluster Stability**: Consider multiple random initializations for robust results
- **Missing Data**: Must implement and document chosen imputation strategy
- **Interpretability**: Balance mathematical rigor with clinical reasoning
   - Visualize model performance
   - Prepare for M2 optimization

---

## 📦 Deliverables

### Required Outputs

- [ ] **Feature Engineering Report** (`M1_feature_engineering.ipynb`)
  - New features created with rationale
  - Feature transformation methods (log, polynomial, interactions, etc.)
  - Feature importance analysis
  - Feature selection results (if applicable)
  - Domain knowledge integration

- [ ] **Model Training Notebook** (`M1_models.ipynb`)
  - At least 3 different model types:
    - Example: Logistic Regression, Random Forest, Gradient Boosting
    - Or: SVM, Neural Network, KNN
  - Hyperparameter configurations documented
  - Training curves (loss/accuracy over epochs/iterations)
  - Training time comparisons

- [ ] **Model Comparison & Evaluation** (`M1_evaluation.ipynb`)
  - Cross-validation setup (k-fold or stratified k-fold)
  - Performance metrics table:
    - Accuracy, Precision, Recall, F1-score
    - AUC-ROC, AUC-PR (for classification)
    - MSE, RMSE, MAE, R² (for regression)
  - Confusion matrices for all models
  - ROC curves and Precision-Recall curves
  - Model selection justification

- [ ] **Visualizations** (saved to `../../reports/figures/`)
  - Feature importance plots
  - Model comparison charts
  - Confusion matrices
  - ROC/PR curves
  - Training curves

- [ ] **Code Modules** (in `../../src/project/`)
  - `features.py` - Feature engineering functions
  - `models.py` - Model training utilities
  - `evaluation.py` - Evaluation metrics and plots

---

## 📊 Key Deliverable Checklist

### Feature Engineering
- [ ] Domain-specific features created
- [ ] Feature transformations applied (log, sqrt, polynomial, etc.)
- [ ] Interaction features explored
- [ ] Feature importance calculated
- [ ] Feature selection performed (if needed)
- [ ] All features documented with descriptions

### Model Training
- [ ] At least 3 different algorithms implemented
- [ ] Baseline hyperparameters documented
- [ ] Training process logged (time, iterations, convergence)
- [ ] Models saved with versioning
- [ ] Random seeds set for reproducibility

### Cross-Validation
- [ ] K-fold or stratified k-fold setup (k=5 or k=10)
- [ ] CV strategy justified
- [ ] Mean and std of CV scores reported
- [ ] Overfitting/underfitting assessed

### Model Comparison
- [ ] Performance metrics table created
- [ ] Confusion matrices generated
- [ ] ROC curves plotted
- [ ] Statistical significance tested (optional)
- [ ] Best model(s) identified for M2

---

## 🔗 Related Files & Notebooks

### Existing Work (Parent Directory)
- `../../M1_G14.ipynb` - Group 14 M1 submission

### New/Organized Work (This Directory)
- `M1_feature_engineering.ipynb` - Feature creation and analysis
- `M1_models.ipynb` - Model training experiments
- `M1_evaluation.ipynb` - Model comparison and evaluation
- `M1_report.md` or `M1_report.pdf` - Milestone summary

### Code Modules
- `../../src/project/features.py` - Feature engineering utilities
- `../../src/project/models.py` - Model training functions
- `../../src/project/evaluation.py` - Evaluation utilities

### Outputs
- `../../reports/figures/M1_feature_importance.png`
- `../../reports/figures/M1_confusion_matrices.png`
- `../../reports/figures/M1_roc_curves.png`
- `../../reports/M1_results_summary.csv`

---

## 📈 Expected Outcomes

By the end of M1, we should have:

1. **Enriched feature set** with engineered features
2. **Multiple trained models** with documented configurations
3. **Comprehensive evaluation** showing model strengths/weaknesses
4. **Clear direction** for M2 optimization
5. **Reproducible pipeline** for feature engineering and model training

---

## 🎯 Success Criteria

M1 is considered complete when:

- [ ] All models show improvement over M0 baseline
- [ ] Feature engineering rationale is documented
- [ ] Model comparison is thorough and objective
- [ ] Code is modular and reusable
- [ ] Results are reproducible with random seeds
- [ ] Figures are publication-ready

---

## 📝 Notes & Tips

### Feature Engineering Best Practices
- **Start with domain knowledge:** Consult domain experts or literature
- **Test incrementally:** Add features one group at a time
- **Check for leakage:** Ensure no future information in features
- **Monitor complexity:** More features ≠ better model

### Model Training Best Practices
- **Start simple:** Begin with linear/tree models before deep learning
- **Use defaults first:** Baseline hyperparameters before tuning
- **Monitor overfitting:** Compare train vs validation performance
- **Save everything:** Models, configs, logs, plots

### Evaluation Best Practices
- **Multiple metrics:** Don't rely on accuracy alone
- **Confusion matrix:** Understand error types
- **Cross-validation:** Essential for reliable estimates
- **Statistical tests:** Use paired t-tests if needed

---

## 🚀 Next Steps (M2)

After completing M1, we will:
- Select top 1-2 models for hyperparameter tuning
- Perform systematic optimization (Grid/Bayesian search)
- Conduct detailed error analysis
- Investigate model interpretability
- Assess robustness and generalization
