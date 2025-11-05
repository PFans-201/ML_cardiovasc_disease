# 🎯 M3 Task Assignment Template

## Team Member Assignments for M3

**Deadline**: [Insert M3 deadline]  
**Target**: Single `M3_G14.ipynb` notebook with complete BN inference

---

### 👥 **Team Member Roles**

#### **Person 1: [Name] - Exact Inference Lead**
- **Branch**: `m3/exact-inference`
- **Tasks**: 
  - ✅ Implement Variable Elimination using pgmpy
  - ✅ Implement Belief Propagation using pgmpy
  - ✅ Execute required queries (a), (b), (c)
  - ✅ Create VE vs BP comparison table
  - ✅ Validate results and check for numerical differences
- **Dependencies**: None (can start immediately)
- **Estimated Time**: 3-4 days
- **Deliverable**: Notebook section 8.1 with working exact inference

#### **Person 2: [Name] - Clinical Analysis Specialist**
- **Branch**: `m3/clinical-queries`
- **Tasks**:
  - ✅ Design 2-3 additional medically relevant queries
  - ✅ Research realistic clinical scenarios
  - ✅ Prepare evidence sets for queries
  - ✅ Write medical interpretation of results
  - ✅ Connect BN inference to clinical decision-making
- **Dependencies**: None (can start immediately)
- **Estimated Time**: 2-3 days
- **Deliverable**: Notebook section 8.3 with custom clinical queries

#### **Person 3: [Name] - Approximate Inference Developer**
- **Branch**: `m3/approximate-inference`
- **Tasks**:
  - ✅ Implement Likelihood-Weighted Sampling
  - ✅ Apply sampling to same queries as exact inference
  - ✅ Compare approximate vs exact results
  - ✅ Analyze convergence (sample size vs accuracy)
  - ✅ Calculate distance metrics (L1, L2, KL divergence)
- **Dependencies**: **WAIT FOR Person 1** (needs exact results for comparison)
- **Estimated Time**: 3-4 days
- **Deliverable**: Notebook section 8.2 with sampling implementation

#### **Person 4: [Name] - Performance Analysis Lead**
- **Branch**: `m3/performance-analysis`
- **Tasks**:
  - ✅ Measure runtime of exact vs approximate methods
  - ✅ Create accuracy vs efficiency plots
  - ✅ Analyze scalability implications
  - ✅ Provide recommendations for different scenarios
  - ✅ Final notebook integration and polishing
- **Dependencies**: **WAIT FOR Persons 1 & 3** (needs both implementations)
- **Estimated Time**: 2-3 days
- **Deliverable**: Notebook section 8.4 and final integration

---

### 📅 **Timeline (Example for 2-week milestone)**

#### **Week 1 (Days 1-7)**
- **Days 1-2**: Person 1 & 2 start work (parallel)
  - Person 1: Begin exact inference implementation
  - Person 2: Design clinical queries and scenarios
- **Days 3-4**: Person 1 completes exact inference
  - Merge exact inference to development branch
  - Person 3 can start approximate inference
- **Days 5-7**: 
  - Person 2 finishes clinical queries
  - Person 3 works on sampling implementation

#### **Week 2 (Days 8-14)**
- **Days 8-10**: Person 3 completes approximate inference
  - Merge to development branch
  - Person 4 starts performance analysis
- **Days 11-12**: Person 4 performance analysis
- **Days 13-14**: Final integration and notebook polishing

---

### 🔄 **Daily Standup Template**

Use this format for daily team check-ins:

```
**Date**: [Date]
**Person**: [Name]
**Branch**: m3/[your-branch]

**Yesterday**:
- What did you complete?
- Any blockers encountered?

**Today**:
- What will you work on?
- Do you need anything from teammates?

**Blockers**:
- What's preventing progress?
- Who can help unblock?

**Status**: On track / Behind / Ahead of schedule
```

---

### 🛠 **Getting Started Commands**

Each team member should run these commands:

```bash
# 1. Get latest repository state
git clone https://github.com/PFans-201/ML_cardiovasc_disease.git
cd ML_cardiovasc_disease

# 2. Set up environment
python -m venv adv_ml_venv
source adv_ml_venv/bin/activate
pip install -r requirements.txt

# 3. Switch to your assigned branch
git checkout m3/exact-inference          # Person 1
git checkout m3/clinical-queries         # Person 2
git checkout m3/approximate-inference    # Person 3
git checkout m3/performance-analysis     # Person 4

# 4. Start working in the M3 notebook
cd milestones/M3/
# Create or edit M3_G14.ipynb
```

---

### 📝 **Communication Channels**

- **Daily Updates**: [Team chat/Slack/Discord]
- **Code Reviews**: GitHub Pull Requests
- **Issues/Bugs**: GitHub Issues
- **Quick Questions**: [WhatsApp/Telegram group]
- **Weekly Sync**: [Meeting time/platform]

---

### 🚨 **Important Reminders**

1. **Test your code** before committing
2. **Document your sections** with markdown explanations
3. **Sync daily** with development branch
4. **Ask for help early** if blocked
5. **Respect dependencies** - don't start if you need someone else's work
6. **Keep commits small** and descriptive
7. **Run entire notebook** before final submission

---

### ✅ **Definition of Done**

M3 is complete when:
- [ ] All 4 sections implemented and working
- [ ] Single `M3_G14.ipynb` runs end-to-end without errors
- [ ] VE and BP produce nearly identical results
- [ ] Sampling converges to exact results with sufficient samples
- [ ] Clinical queries provide meaningful medical insights
- [ ] Performance analysis gives clear recommendations
- [ ] All team members' contributions integrated
- [ ] Final notebook uploaded to main branch

---

**Let's make M3 exceptional!** 🚀