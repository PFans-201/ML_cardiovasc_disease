# Milestone M0: Exploratory Data Analysis

**Deadline:** September 26, 23:59  
**Status:** 🔄 In Progress

---

## 📋 Objectives

1. **Dataset Overview**
   - Understand the structure and scale of patient and encounter data
   - Quantify the longitudinal nature of the dataset
   - Establish baseline statistics for the population

2. **Baseline Characteristics Analysis**
   - Analyze patient demographics and risk factor distributions
   - Examine the computed risk score across the population
   - Understand initial disease state distributions

3. **Disease State Pattern Discovery**
   - Investigate disease state distributions across encounters
   - Identify patient progression patterns with illustrative examples
   - Assess stability vs. progression in the patient population

4. **Clinical Variable Analysis**
   - Examine symptom frequencies and laboratory value distributions
   - Investigate relationships between clinical variables and disease states
   - Identify potential predictive patterns

5. **Treatment and Outcome Analysis**
   - Analyze treatment distribution patterns across encounters and states
   - Examine utility score distributions and relationships
   - Understand treatment-outcome associations

6. **Data Quality Assessment**
   - Identify missing value patterns and frequencies
   - Propose strategies for handling missing data
   - Prepare data preprocessing pipeline for future milestones

---

## 📊 Specific Tasks

### 5.1. Dataset Overview
- Count total patients and encounters
- Calculate average encounters per patient
- Verify longitudinal structure (8 time points per patient)

### 5.2. Baseline Characteristics
- Distribution analysis: age, sex, BMI, smoking, family history
- Risk score distribution across population
- Demographic summaries with visualizations

### 5.3. Disease States
- [x] Proportion of encounters in each state (Healthy, Early, Advanced)
- [x] Patient trajectory examples showing stability vs. progression
- [x] Transition patterns between consecutive visits

### 5.4. Symptoms and Laboratory Values
- [x ] Binary symptom frequencies (chest pain, fatigue, shortness of breath)
- [ ] Lab value statistics (blood pressure, cholesterol, glucose, troponin)
- [ ] Clinical variable differences by disease state

### 5.5. Treatments and Outcomes
- [ ] Treatment distribution across encounters and disease states
- [ ] Utility score distribution and summary statistics
- [ ] Treatment-outcome relationship analysis

### 5.6. Data Quality Check
- [ ] Missing value assessment by variable and frequency
- [ ] Missing data pattern visualization
- [ ] Proposed missing data handling strategies

---

## 🎯 Key Requirements

- **Independent Observations**: Treat each encounter (row in encounters.csv) as independent
- **Static vs. Dynamic**: Use patients.csv only for baseline summaries
- **Interpretation Focus**: Reason about patterns in medical context to motivate future probabilistic models
- **Visualization**: Include both summary statistics and informative plots
- **Clinical Reasoning**: Connect observed patterns to potential medical interpretations

---

## 📁 Deliverables

1. **Jupyter Notebook**: `M0.ipynb` with complete analysis
2. **Data Summary Report**: Key findings and insights
3. **Missing Data Strategy**: Documented approach for M1 preprocessing
4. **Visualization Portfolio**: Comprehensive EDA plots

---

## 🔗 Connections to Future Milestones

- **M1**: Missing data strategy will be implemented for GMM preprocessing
- **M2**: Clinical patterns will inform Bayesian network structure design
- **M3**: Identified relationships will guide inference query design
- **M5**: Disease progression patterns will inform HMM temporal modeling
- **M6**: Treatment-outcome patterns will inform RL reward design

---

## 📦 Deliverables

### Required Outputs

- [ ] **Problem Statement Document** (`M0_problem_statement.md` or section in notebook)
  - ML task definition
  - Success metrics with justification
  - Dataset description and source
  - Known limitations and assumptions

- [ ] **EDA Notebook** (`M0_eda.ipynb`)
  - Data loading and initial inspection
  - Univariate analysis (distributions, summary statistics)
  - Bivariate/multivariate analysis (correlations, relationships)
  - Missing values and outliers report
  - Visualizations (histograms, box plots, correlation heatmaps)

- [ ] **Preprocessing Pipeline** (`M0_preprocessing.ipynb` or in `src/project/preprocessing.py`)
  - Missing value imputation code
  - Feature scaling implementation
  - Categorical encoding
  - Data splitting function

- [ ] **Baseline Models** (`M0_baseline.ipynb`)
  - Simple baseline implementation and results
  - Classical ML baseline implementation
  - Performance metrics (accuracy, F1, etc.)
  - Comparison table

- [ ] **Data Documentation** (`../../data/README.md`)
  - Dataset description
  - Feature dictionary
  - Data source and collection method
  - Preprocessing steps applied

---

## 📊 Key Deliverable Checklist

### Problem Definition
- [ ] ML task clearly defined (classification, regression, etc.)
- [ ] Target variable identified
- [ ] Success metrics chosen and justified
- [ ] Evaluation protocol defined

### Data Understanding
- [ ] Dataset size and shape documented
- [ ] Feature types identified (numeric, categorical, ordinal)
- [ ] Missing value patterns analyzed
- [ ] Outliers detected and documented
- [ ] Class distribution analyzed (if classification)
- [ ] Feature correlations explored

### Preprocessing
- [ ] Missing value strategy chosen and implemented
- [ ] Feature scaling method selected
- [ ] Categorical encoding approach defined
- [ ] Train/val/test split implemented (e.g., 70/15/15)
- [ ] No data leakage in splitting process

### Baseline Models
- [ ] Simple baseline implemented (e.g., most frequent class)
- [ ] Classical ML baseline trained (e.g., Logistic Regression, Random Forest)
- [ ] Baseline metrics recorded
- [ ] Results compared and documented

---

## 🔗 Related Files & Notebooks

### Existing Work (Parent Directory)
- `../../M0.ipynb` - Initial M0 exploration
- `../../M0_G14.ipynb` - Group 14 M0 submission
- `../../M0_almost.ipynb` - Draft M0 work
- `../../Project_M0.ipynb` - Project M0 version 1
- `../../Project_M0_v2.ipynb` - Project M0 version 2

### New/Organized Work (This Directory)
- `M0_eda.ipynb` - Exploratory data analysis
- `M0_preprocessing.ipynb` - Data preprocessing pipeline
- `M0_baseline.ipynb` - Baseline model experiments
- `M0_report.md` or `M0_report.pdf` - Milestone summary report

### Code Modules
- `../../src/project/eda.py` - Reusable EDA utilities
- `../../src/project/preprocessing.py` - Preprocessing functions

### Data
- `../../data/raw/` - Original datasets (not committed)
- `../../data/interim/` - Cleaned/preprocessed data
- `../../data/README.md` - Data documentation

---

## 📈 Expected Outcomes

By the end of M0, we should have:

1. **Clear problem definition** with measurable success criteria
2. **Comprehensive understanding** of the data structure and quality
3. **Reproducible preprocessing pipeline** ready for model training
4. **Baseline performance** to compare future models against
5. **Documentation** sufficient for team members and external reviewers

---

## 🎯 Success Criteria

M0 is considered complete when:

- [ ] All team members understand the problem and data
- [ ] Preprocessing pipeline runs without errors
- [ ] Baseline models produce interpretable results
- [ ] Documentation is clear and reproducible
- [ ] Code is committed to GitHub with proper structure

---

## 📝 Notes & Tips

- **Start simple:** Don't over-engineer preprocessing in M0
- **Document decisions:** Explain why you chose specific approaches
- **Visualize extensively:** Good plots reveal insights
- **Baseline is important:** It's your performance floor
- **Test set untouched:** Only use train/validation in M0

---

## 🚀 Next Steps (M1)

After completing M0, we will:
- Engineer new features based on EDA insights
- Implement multiple ML models
- Compare model performance systematically
- Refine preprocessing based on model needs
