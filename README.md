# 🏥 Cardiovascular Disease Analysis: Probabilistic Modeling & Decision Making

> **Advanced Machine Learning Project**: Systematic exploration of probabilistic methods for medical data analysis - from mixture models to reinforcement learning

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://python.org)
[![License](https://img.shields.io/badge/License-Educational-green)](LICENSE)
[![Data](https://img.shields.io/badge/Data-Synthetic-orange)](data/raw/README.md)

## 🎯 Project Overview

This project systematically explores **probabilistic modeling techniques** applied to cardiovascular disease data, progressing from basic clustering to advanced sequential decision making. Each milestone builds upon previous work, culminating in a reinforcement learning system for treatment optimization.

### Core Learning Path
- 📊 **M0**: Exploratory Data Analysis & Statistical Foundations
- 🎯 **M1**: Gaussian Mixture Models for Patient Clustering  
- 🕸️ **M2-M4**: Bayesian Networks (Design → Inference → Learning)
- ⏱️ **M5**: Hidden Markov Models for Disease Progression
- 🤖 **M6**: Reinforcement Learning for Treatment Policies

### Key Features
- 📚 **Systematic Methodology**: Six milestones covering core probabilistic ML techniques
- 🔬 **Hands-on Implementation**: Build models from scratch using pgmpy, hmmlearn, custom RL
- � **Real-world Complexity**: Handle missing data, discretization, temporal dependencies
- 🎓 **Educational Focus**: Deep understanding over black-box solutions

## 🏗️ Project Structure

```
├── data/
│   ├── raw/                    # 📁 Original datasets (INCLUDED in repo)
│   │   ├── patients.csv        # Static patient demographics & risk factors (3,000 patients)
│   │   ├── encounters.csv      # Longitudinal clinical encounters (24,000 visits)
│   │   └── README.md           # Comprehensive data documentation
│   ├── interim/                # Processed data (imputed, discretized)
│   └── processed/              # Final model-ready datasets
├── milestones/                 # 📋 Six milestone deliverables
│   ├── M0/                     # Exploratory Data Analysis
│   ├── M1/                     # Gaussian Mixture Models
│   ├── M2/                     # Bayesian Network Design
│   ├── M3/                     # BN Inference (Exact & Approximate)
│   ├── M4/                     # BN Learning from Data
│   ├── M5/                     # Hidden Markov Models
│   └── M6/                     # Reinforcement Learning
├── src/project/                # 🔧 Reusable Python modules
│   ├── data_loader.py          # Standardized data loading utilities
│   ├── visualizations.py       # Comprehensive plotting toolkit
│   ├── gmm_utils.py            # Gaussian Mixture Model utilities
│   ├── bayesian_networks.py    # BN modeling and inference
│   ├── hmm_utils.py            # Hidden Markov Model tools
│   └── rl_environment.py       # Reinforcement Learning setup
├── notebooks/                  # 🧪 Development & demonstration notebooks
├── reports/                    # 📊 Final outputs & visualizations
└── docs/                       # 📚 Documentation & guidelines
```

## 🚀 Quick Start

### 1. Environment Setup
```bash
# Clone repository
git clone <repository-url>
cd Project

# Create virtual environment
python -m venv adv_ml_venv
source adv_ml_venv/bin/activate  # Linux/Mac
# adv_ml_venv\Scripts\activate   # Windows

# Install dependencies
pip install -r requirements.txt
```

### 2. Data Overview
```python
import pandas as pd

# Load datasets (included in repo!)
patients = pd.read_csv('data/raw/patients.csv', sep=';')
encounters = pd.read_csv('data/raw/encounters.csv', sep=';')

print(f"📊 {len(patients):,} patients, {len(encounters):,} encounters")
print(f"⏱️  {encounters.groupby('patient_id').size().iloc[0]} time points per patient")
```

### 3. Start Exploring
```bash
# Run milestone notebooks
jupyter lab milestones/M0/M0.ipynb     # Data exploration
jupyter lab milestones/M1/M1_G14.ipynb # Feature engineering & modeling
```

## 🎓 Learning Objectives

This project systematically builds expertise in probabilistic machine learning:

| **Milestone** | **Core Technique** | **Learning Focus** | **Deliverable** |
|---------------|-------------------|-------------------|-----------------|
| **M0** | Exploratory Data Analysis | Data understanding, visualization | Statistical summaries & insights |
| **M1** | Gaussian Mixture Models | Unsupervised clustering, EM algorithm | Patient phenotype discovery |
| **M2** | Bayesian Network Design | Graphical models, conditional independence | Hand-designed probabilistic model |
| **M3** | BN Inference | Variable elimination, belief propagation | Clinical query answering |
| **M4** | BN Parameter/Structure Learning | Maximum likelihood, score-based search | Data-driven model comparison |
| **M5** | Hidden Markov Models | Temporal modeling, Viterbi decoding | Disease progression analysis |
| **M6** | Reinforcement Learning | Q-learning, policy optimization | Treatment recommendation system |

### Key Skills Developed
- **🧮 Probabilistic Reasoning**: Understanding uncertainty and conditional dependencies
- **📊 Graphical Models**: Designing and interpreting Bayesian networks
- **⏱️ Temporal Modeling**: Capturing disease progression with HMMs
- **🎯 Decision Making**: Optimizing treatment policies with reinforcement learning
- **🔬 Model Evaluation**: Comparing hand-designed vs. learned models

## 📈 Milestones & Progress

### 📊 **M0: Exploratory Data Analysis** (Deadline: Sept 26)
- ✅ Dataset overview and baseline characteristics
- ✅ Disease state distributions and patient trajectories  
- ✅ Treatment patterns and outcome analysis
- ✅ Missing data assessment and handling strategies

### 🎯 **M1: Gaussian Mixture Models** (Deadline: Oct 10)
- 📋 Feature selection and preprocessing for clustering
- 📋 GMM fitting with optimal cluster selection (AIC/BIC)
- 📋 Cluster characterization and clinical interpretation
- 📋 Comparison with true disease states

### �️ **M2: Bayesian Network Design** (Deadline: Oct 20)
- 📋 Structure design with clinical justification
- 📋 Variable discretization for categorical BNs
- 📋 CPT estimation from encounter data
- 📋 Conditional independence analysis

### 🧠 **M3: Bayesian Network Inference** (Deadline: Nov 7)
- 📋 Exact inference (Variable Elimination, Belief Propagation)
- 📋 Approximate inference with sampling methods
- 📋 Clinical query design and interpretation

### 📚 **M4: Learning Bayesian Networks** (Deadline: Nov 21)
- 📋 Parameter learning with train/test patient splits
- 📋 Structure learning with score-based search
- 📋 Model comparison: hand-designed vs. learned
- 📋 Expert knowledge vs. data-driven trade-offs

### ⏱️ **M5: Hidden Markov Models** (Deadline: Dec 1)
- 📋 Temporal modeling with troponin + symptom features
- 📋 Baum-Welch parameter learning
- 📋 Viterbi decoding and state sequence analysis
- 📋 Comparison with true disease progression

### 🤖 **M6: Reinforcement Learning** (Deadline: Dec 12)
- 📋 MDP environment setup with state/action/reward
- 📋 Tabular Q-learning for treatment policies
- 📋 Policy evaluation vs. random/heuristic baselines
- 📋 Clinical interpretation of learned strategies

## 💡 Key Innovations

1. **Multi-task Learning**: Simultaneous disease prediction + treatment optimization
2. **Temporal Modeling**: Account for disease state transitions over time  
3. **Missing Data Realism**: Handle realistic clinical data gaps
4. **Utility-based Evaluation**: Beyond accuracy - optimize patient outcomes
5. **Causal Treatment Effects**: Estimate counterfactual treatment scenarios

## 🔬 Data Highlights

- **Domain**: Synthetic cardiovascular clinical study
- **Design**: Longitudinal (8 time points over ~2 years)
- **Patients**: 3,000 diverse synthetic patients
- **Variables**: Demographics, risk factors, symptoms, labs, treatments, outcomes
- **Missing Data**: Realistic patterns (10-20% symptoms, 5-15% labs)
- **Ground Truth**: Disease states provided for educational purposes

> 📖 **Full data documentation**: [`data/raw/README.md`](data/raw/README.md)

## 🤝 Contributing

See [`CONTRIBUTING.md`](CONTRIBUTING.md) for development workflow, branching strategy, and code standards.

## ⚠️ Important Notes

- **🎓 Educational Data**: Synthetic dataset for learning purposes only
- **❌ Not for Clinical Use**: Do not apply insights to real patients  
- **📚 Research Only**: Suitable for methodology development and education
- **✅ Reproducible**: All code, data, and results version controlled
