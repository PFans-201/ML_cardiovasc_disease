# 🎯 M3 Collaborative Workflow Guide

## Current Status: Working on Milestone M3 - Bayesian Network Inference

**GitHub Repository**: https://github.com/PFans-201/ML_cardiovasc_disease.git  
**Target Deliverable**: Single `M3_G14.ipynb` notebook with complete BN inference implementation

---

## 📋 M3 Task Division Strategy

### **Branch Structure for M3**

```
🏠 main (stable, contains M0-M2 completed work)
│
├── 🔧 milestone/m3-development (integration branch for M3)
    ├── 🧮 m3/exact-inference (Variable Elimination & Belief Propagation)
    ├── 🎲 m3/approximate-inference (Sampling methods & convergence)
    ├── 🏥 m3/clinical-queries (Medical scenario analysis)
    └── ⚡ m3/performance-analysis (Computational efficiency comparison)
```

### **Parallel vs Sequential Task Dependencies**

**✅ PARALLEL TASKS** (can work simultaneously):
1. **Exact Inference Implementation** (`m3/exact-inference`)
   - Variable Elimination algorithm
   - Belief Propagation algorithm
   - Basic query execution

2. **Clinical Query Design** (`m3/clinical-queries`)
   - Design 2-3 medically relevant queries
   - Research clinical scenarios
   - Prepare evidence sets

**⏳ SEQUENTIAL TASKS** (need previous work):
3. **Approximate Inference** (`m3/approximate-inference`) 
   - **Depends on**: Exact inference results for comparison
   - Likelihood-weighted sampling
   - Convergence analysis vs exact results

4. **Performance Analysis** (`m3/performance-analysis`)
   - **Depends on**: Both exact and approximate implementations
   - Runtime comparisons
   - Accuracy vs efficiency trade-offs

---

## 🚀 Workflow for Team Members

### **Person 1: Exact Inference Lead**

```bash
# 1. Start work on exact inference
git checkout m3/exact-inference
git pull origin milestone/m3-development

# 2. Work on Variable Elimination & Belief Propagation
# Create notebook sections:
# - 8.1 Exact inference implementation
# - Query (a): P(state = Healthy | chest_pain = 1, fatigue = 1)
# - Query (b): P(treatment | chest_pain = 1, fatigue = 1)
# - Query (c): P(utility = High | risk_score = 4, treatment = DrugA)

# 3. Commit progress regularly
git add milestones/M3/
git commit -m "feat(M3): implement Variable Elimination for exact inference

- Add VE algorithm using pgmpy
- Implement queries (a), (b), (c) with VE
- Add result validation and comparison tables"

# 4. Push work
git push origin m3/exact-inference
```

### **Person 2: Clinical Queries Specialist**

```bash
# 1. Start work on clinical analysis
git checkout m3/clinical-queries

# 2. Work on custom queries and medical interpretation
# Create notebook sections:
# - 8.3 Custom clinical queries (2-3 additional)
# - Medical scenario development
# - Clinical interpretation of results

# 3. Commit and push
git add milestones/M3/
git commit -m "feat(M3): design custom clinical queries for BN inference

- Add 3 medically relevant query scenarios
- Include clinical context and interpretation
- Prepare evidence sets for realistic cases"

git push origin m3/clinical-queries
```

### **Person 3: Approximate Inference** (waits for Person 1)

```bash
# 1. Wait for exact inference to be completed
# 2. Merge exact inference work
git checkout m3/approximate-inference
git merge m3/exact-inference

# 3. Implement sampling methods
# Create notebook sections:
# - 8.2 Approximate inference implementation
# - Likelihood-weighted sampling
# - Convergence analysis vs exact results
# - Distance metrics (L1, L2, KL divergence)

# 4. Commit and push
git commit -m "feat(M3): implement approximate inference with convergence analysis

- Add likelihood-weighted sampling
- Compare against exact inference results
- Include convergence plots and accuracy metrics"
```

### **Person 4: Performance Analysis** (waits for Persons 1, 3)

```bash
# 1. Wait for both exact and approximate inference
# 2. Merge all previous work
git checkout m3/performance-analysis
git merge m3/exact-inference
git merge m3/approximate-inference

# 3. Add performance comparison
# Create notebook sections:
# - Runtime comparison plots
# - Accuracy vs sample size analysis
# - Computational efficiency discussion

# 4. Final integration work
```

---

## 🔄 Integration Workflow

### **Daily Sync Process**

```bash
# Every team member should sync daily
git checkout milestone/m3-development
git pull origin milestone/m3-development

# Merge latest changes into your branch
git checkout m3/your-task-branch
git merge milestone/m3-development

# Continue your work...
```

### **Weekly Integration**

Every few days, merge completed work into the development branch:

```bash
# Person 1 completes exact inference
git checkout milestone/m3-development
git merge m3/exact-inference
git push origin milestone/m3-development

# Person 2 adds clinical queries
git checkout milestone/m3-development
git merge m3/clinical-queries
git push origin milestone/m3-development

# And so on...
```

### **Final M3 Assembly**

When all tasks are complete:

```bash
# Merge everything into the final M3 notebook
git checkout milestone/m3-development

# Merge all task branches
git merge m3/exact-inference
git merge m3/approximate-inference  
git merge m3/clinical-queries
git merge m3/performance-analysis

# Resolve any conflicts and create final M3_G14.ipynb
# Final commit and merge to main
git checkout main
git merge milestone/m3-development
```

---

## 📝 M3 Notebook Structure Template

Your final `M3_G14.ipynb` should have this structure:

```markdown
# M3: Bayesian Network Inference

## Introduction & Setup
- Import libraries
- Load BN model from M2
- Data preprocessing recap

## 8.1 Exact Inference (Person 1)
- Variable Elimination implementation
- Belief Propagation implementation
- Queries (a), (b), (c) with both methods
- VE vs BP comparison table

## 8.2 Approximate Inference (Person 3)
- Likelihood-weighted sampling implementation
- Same queries with sampling
- Convergence analysis (sample size vs accuracy)
- Distance metrics vs exact results

## 8.3 Custom Clinical Queries (Person 2)
- 2-3 additional medically relevant queries
- Clinical scenario descriptions
- Medical interpretation of results

## 8.4 Performance Analysis (Person 4)
- Runtime comparison (exact vs approximate)
- Accuracy vs efficiency trade-offs
- Scalability discussion
- Recommendations for different scenarios

## Conclusion
- Summary of key findings
- Clinical insights
- Computational recommendations
```

---

## 🛠 Essential Commands for M3 Work

### Quick Status Check
```bash
git status                           # See current changes
git log --oneline -3                # Recent commits
git branch                          # Current branch
```

### Syncing with Team
```bash
git fetch origin                    # Get all latest changes
git checkout milestone/m3-development
git pull origin milestone/m3-development
git checkout m3/your-branch
git merge milestone/m3-development  # Get team updates
```

### Handling Conflicts in Notebook
```bash
# If M3_G14.ipynb has conflicts:
git status                          # See conflicted files
# Edit M3_G14.ipynb manually to resolve conflicts
# Look for <<<<<<< ======= >>>>>>> markers
git add milestones/M3/M3_G14.ipynb
git commit -m "resolve: merge conflicts in M3 notebook"
```

### Emergency Recovery
```bash
git stash                           # Save current work
git checkout milestone/m3-development
git pull origin milestone/m3-development
git checkout m3/your-branch
git stash pop                       # Restore your work
```

---

## 🎯 Success Criteria for M3

- ✅ **VE and BP implemented** and produce identical results
- ✅ **Sampling method works** with reasonable accuracy
- ✅ **Convergence analysis** shows improvement with more samples
- ✅ **Custom queries** demonstrate clinical relevance
- ✅ **Performance comparison** provides clear recommendations
- ✅ **Single cohesive notebook** that runs end-to-end
- ✅ **All team contributions** integrated seamlessly

---

## 🚨 Important Notes

1. **Never work directly on main** - use task branches
2. **Commit early and often** - don't lose work
3. **Test notebook execution** before merging
4. **Clear conflict resolution** - communicate when merging
5. **Consistent code style** - follow established patterns
6. **Document your sections** - add markdown explanations

---

**Target Completion**: M3 deadline  
**Final Deliverable**: Single `M3_G14.ipynb` with all inference implementations

Let's make M3 awesome! 🚀