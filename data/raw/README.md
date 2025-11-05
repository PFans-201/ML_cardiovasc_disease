# Raw Data Documentation

## Dataset Overview

**Source:** Synthetic Medical Diagnosis and Treatment Dataset  
**Domain:** Cardiovascular Disease Longitudinal Study  
**Purpose:** Educational dataset for probabilistic modeling, inference, and treatment optimization  
**Creation Date:** 2024/2025 Academic Year  
**License:** Educational use only (synthetic data)

---

## Clinical Scenario

This dataset simulates a longitudinal clinical study conducted by a cardiology clinic monitoring patients at risk for cardiovascular disease. The study tracks patients over multiple years through regular clinical visits, recording:

- Demographic and risk factors (baseline)
- Disease state progression over time
- Symptoms and laboratory measurements
- Treatment decisions and outcomes

**Key characteristics:**
- **Longitudinal design:** Each patient followed over 8 time points (0-7)
- **Mixed data types:** Static patient profiles + dynamic encounter records
- **Hidden states:** True disease state provided as ground truth for teaching purposes
- **Treatment decisions:** Simulates clinical decision-making environment

---

## Files

### 1. `patients.csv`

**Description:** Static patient characteristics recorded at baseline (time of enrollment)

**Dimensions:** 3,000 patients × 9 features

**Separator:** Semicolon (`;`)

**Column Descriptions:**

| Column | Type | Range/Values | Description |
|--------|------|--------------|-------------|
| `patient_id` | Integer | 1-3000 | Unique identifier for each patient |
| `age` | Integer | 18-85 | Patient's age in years at baseline |
| `sex` | Categorical | M, F | Biological sex (Male/Female) |
| `bmi` | Float | 15.0-45.0 | Body Mass Index (weight/height²) |
| `smoker` | Binary | 0, 1 | Current smoker status (0=No, 1=Yes) |
| `family_history` | Binary | 0, 1 | Family history of cardiovascular disease (0=No, 1=Yes) |
| `hypertension` | Binary | 0, 1 | Previous diagnosis of high blood pressure (0=No, 1=Yes) |
| `risk_score` | Integer | 0-7 | Synthetic risk index combining all risk factors (higher=greater risk) |
| `initial_state` | Categorical | Healthy, Early, Advanced | Patient's baseline disease state |

**Missing Values:** None

**Notes:**
- All patient characteristics are **fixed** across time
- `risk_score` is a derived feature summarizing combined risk factors
- Higher risk scores correlate with worse initial states

**Example:**
```csv
patient_id;age;sex;bmi;smoker;family_history;hypertension;risk_score;initial_state
1;61;M;26.05;0;0;0;1;Early
2;53;M;19.51;1;0;1;3;Healthy
```

---

### 2. `encounters.csv`

**Description:** Time-varying information collected at each clinical follow-up visit

**Dimensions:** 24,000 encounters (3,000 patients × 8 time points)

**Separator:** Semicolon (`;`)

**Column Descriptions:**

| Column | Type | Range/Values | Description |
|--------|------|--------------|-------------|
| `patient_id` | Integer | 1-3000 | Links to patient in `patients.csv` |
| `time` | Integer | 0-7 | Follow-up time point (0=baseline, 7=final) |
| `state` | Categorical | Healthy, Early, Advanced | **Ground truth** disease state at this visit |
| `treatment` | Categorical | None, DrugA, DrugB, Lifestyle | Treatment administered at visit |
| `chest_pain` | Binary | 0, 1, NaN | Presence of chest pain symptom |
| `fatigue` | Binary | 0, 1, NaN | Presence of fatigue symptom |
| `shortness_of_breath` | Binary | 0, 1, NaN | Presence of shortness of breath symptom |
| `systolic_bp` | Float | 80-200 | Systolic blood pressure (mmHg) |
| `cholesterol` | Float | 120-300 | Total cholesterol (mg/dL) |
| `glucose` | Float | 70-140 | Blood glucose level (mg/dL) |
| `troponin` | Float | 0.0-0.1 | Cardiac troponin (biomarker for heart damage) |
| `utility` | Float | 0.0-1.0 | Patient outcome quality score (higher=better outcome) |

**Missing Values:** Yes (see Preprocessing section in M1)

**Notes:**
- **Disease state progression:** `state` can have the values between of `Healthy`, `Early` and `Advanced`
- **Treatment values:**
  - `None`: No treatment (may appear as NaN in raw file)
  - `DrugA`, `DrugB`: Pharmacological interventions
  - `Lifestyle`: Non-pharmacologic (diet/exercise/smoking cessation)
- **Utility:** Synthetic reward signal encoding quality of life given state and treatment
  - Higher utility = favorable outcome (stable, treatment effective)
  - Lower utility = poor outcome (advanced disease, poor symptom control)
- **Symptoms and labs:** May have missing values due to incomplete testing

**Example:**
```csv
patient_id;time;state;treatment;chest_pain;fatigue;shortness_of_breath;systolic_bp;cholesterol;glucose;troponin;utility
1;0;Early;DrugA;1;1;0;113.68;;88.58;;0.449
1;1;Early;Lifestyle;0;;0;130.6;226.68;112.56;0.0139;0.471
```

---

## Data Relationships

### Patient-Encounter Linkage

Each patient has exactly **8 encounters** (time points 0-7):
```
patients.patient_id (1:N) ←→ encounters.patient_id
```

### Temporal Structure Example

```
Time:    0 ------→ 1 ----→ 2 -------→ ... ----→ 7
State:   Healthy → Early → Advanced → ... ----→ Advanced
```

Disease states form a **Markov chain** where transitions depend on:
- Patient risk factors (from `patients.csv`)
- Previous state
- Treatment administered

---

## Preprocessing Required

### 1. Treatment Missing Values

`None` treatments may appear as `NaN` in pandas. **Fix with:**

```python
import pandas as pd

patients = pd.read_csv('patients.csv', sep=';')
encounters = pd.read_csv('encounters.csv', sep=';')

# Fix treatment missing values
encounters['treatment'] = encounters['treatment'].fillna('None')
```

### 2. Laboratory Missing Values

Lab results (`systolic_bp`, `cholesterol`, `glucose`, `troponin`) and symptoms have **realistic missingness** simulating incomplete testing. The missing values can be handled with:

- **Imputation:** Forward and backward fill, mean/median/mode, KNN
- **Indicator variables:** Create `lab_missing` flags
- **Model-based:** Use models robust to missing data

### 3. Data Merging

Combine static and dynamic data:

```python
df = encounters.merge(patients, on='patient_id', how='left')
```

---

## Data Loading Template

```python
import pandas as pd
import numpy as np

# Load datasets
patients = pd.read_csv('data/raw/patients.csv', sep=';')
encounters = pd.read_csv('data/raw/encounters.csv', sep=';')

# Fix treatment missing values
encounters['treatment'] = encounters['treatment'].fillna('None')

# Merge for analysis
df = encounters.merge(patients, on='patient_id', how='left')

# Basic statistics
print(f"Total patients: {patients.shape[0]}")
print(f"Total encounters: {encounters.shape[0]}")
print(f"Encounters per patient: {encounters.groupby('patient_id').size().mean():.1f}")
print(f"\nDisease state distribution:\n{encounters['state'].value_counts()}")
print(f"\nTreatment distribution:\n{encounters['treatment'].value_counts()}")
```

---

## Dataset Statistics - NEEDED TO BE CONFIRMED

### Patient Demographics (Baseline)
- **Total patients:** 3,000
- **Age range:** 18-85 years (median ~55)
- **Sex distribution:** ~50% Male, ~50% Female
- **Smokers:** ~30%
- **Family history:** ~20%
- **Hypertension:** ~25%
- **Initial states:**
  - Healthy: ~70%
  - Early disease: ~25%
  - Advanced disease: ~5%

### Encounter Statistics
- **Total encounters:** 24,000
- **Encounters per patient:** 8 (consistent)
- **State distribution across all visits:**
  - Healthy: ~40%
  - Early: ~35%
  - Advanced: ~25%
- **Treatment distribution:**
  - None: ~35%
  - Lifestyle: ~25%
  - DrugA: ~20%
  - DrugB: ~20%
- **Missing data rates:**
  - Symptoms: 10-20% per symptom
  - Labs: 5-15% per lab

---

## Use Cases

This dataset supports multiple machine learning tasks:

### 1. **Classification**
- Predict disease state from symptoms and labs
- Predict treatment effectiveness
- Identify high-risk patients

### 2. **Clustering**
- Discover patient subgroups with similar disease trajectories
- Identify treatment response patterns

### 3. **Probabilistic Modeling**
- **Bayesian Networks:** Model causal relationships between risk factors, symptoms, and disease
- **Hidden Markov Models (HMM):** Model disease state transitions over time
- **Markov Decision Processes (MDP):** Optimize treatment policies

### 4. **Time Series Analysis**
- Predict disease progression
- Forecast future lab values
- Model symptom evolution

### 5. **Reinforcement Learning**
- Optimize treatment strategies using `utility` as reward
- Learn policies for treatment selection
- Evaluate counterfactual treatment scenarios

---

## Important Limitations

⚠️ **This is synthetic data for educational purposes only.**

### Limitations:
1. **Not real patients:** All records are simulated
2. **Simplified disease model:** Only 3 discrete states (real disease is continuous)
3. **Limited treatments:** Only 4 treatment options (real clinics have many more)
4. **Utility is artificial:** Real outcomes are multidimensional and hard to quantify
5. **No external validity:** Do NOT draw medical conclusions for real patients
6. **Educational only:** Strictly a teaching tool for ML methods

### What the data IS good for:
✅ Learning probabilistic modeling techniques  
✅ Practicing ML workflow and evaluation  
✅ Experimenting with treatment optimization  
✅ Understanding causal inference challenges  
✅ Developing reproducible analysis pipelines  

### What the data is NOT good for:
❌ Making real medical decisions  
❌ Publishing clinical findings  
❌ Training production healthcare models  
❌ Validating real treatment efficacy  

---

## Data Quality Notes

- **No duplicates:** Each `(patient_id, time)` pair is unique
- **Consistent structure:** All patients have exactly 8 time points
- **Plausible ranges:** All values fall within medically realistic bounds
- **Realistic missingness:** Missing values follow realistic clinical patterns
- **Ground truth included:** `state` variable provided for teaching purposes (unavailable in real scenarios)
- **Synthetic randomness:** Data generated with stochastic processes to include noise and variability and mimic real-world data distributions

