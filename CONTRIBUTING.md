# Contributing to Cardiovascular Disease ML Project

This guide covers the complete Git workflow for collaborative development on this Advanced Machine Learning project.

## 🚀 Initial Repository Setup

### Step 1: Create GitHub Repository

1. **Go to GitHub** and click "New repository"
2. **Repository name**: `ML_cardiovasc_disease` 
3. **Description**: `Advanced Machine Learning for Cardiovascular Disease Prediction & Treatment Optimization`
4. **Visibility**: Private (recommended for academic work) or Public
5. **Initialize**: Do NOT initialize with README, .gitignore, or license (we already have these)
6. **Click "Create repository"**

### Step 2: Initialize Local Git Repository

```bash
# Navigate to your project directory
cd /home/pfanyka/Desktop/MASTERS/ADV_ML/Project

# Initialize Git repository
git init

# Add all files to staging
git add .

# Create initial commit
git commit -m "feat: initial project setup with probabilistic ML structure

- Add comprehensive milestone structure (M0-M6)
- Implement specialized utilities (GMM, BN, HMM, RL)
- Include professional documentation and guidelines
- Set up data pipeline and reproducible environment"

# Add remote repository (replace with your actual GitHub URL)
git remote add origin https://github.com/PFans-201/ML_cardiovasc_disease.git

# Push to GitHub
git push -u origin main
```

### Step 3: Set Up Branch Protection (GitHub Web Interface)

1. Go to **Settings** → **Branches** in your GitHub repository
2. Click **Add rule**
3. **Branch name pattern**: `main`
4. Enable:
   - ✅ Require a pull request before merging
   - ✅ Require approvals (set to 1)
   - ✅ Dismiss stale PR approvals when new commits are pushed
   - ✅ Require branches to be up to date before merging
5. **Save changes**

---

## 📋 Collaborative Workflow

### Creating Feature Branches

```bash
# Always start from main branch
git checkout main
git pull origin main

# Create and switch to new feature branch
git checkout -b feature/milestone-m1-gmm-implementation

# Alternative branch naming conventions:
# git checkout -b feature/m2-bayesian-networks
# git checkout -b fix/data-loading-bug
# git checkout -b docs/update-readme
# git checkout -b analysis/m3-inference-experiments
```

### Daily Development Workflow

```bash
# 1. Start work session - sync with main
git checkout main
git pull origin main
git checkout your-feature-branch
git merge main  # or: git rebase main

# 2. Make changes and commit frequently
git add src/project/gmm_utils.py milestones/M1/M1_G14.ipynb
git commit -m "feat(M1): implement GMM patient clustering

- Add CVDGaussianMixture class with model selection
- Implement cluster analysis and visualization
- Add clinical risk level prediction
- Include comprehensive docstrings and examples"

# 3. Push your work regularly
git push origin your-feature-branch

# 4. Create Pull Request when ready (via GitHub web interface)
```

### Commit Message Convention

Use **Conventional Commits** format:

```
<type>(<scope>): <description>

[optional body]

[optional footer(s)]
```

**Types:**
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation changes
- `style`: Code style changes (formatting, etc.)
- `refactor`: Code refactoring
- `test`: Adding or updating tests
- `chore`: Maintenance tasks

**Examples:**
```bash
git commit -m "feat(M2): implement Bayesian Network structure design"
git commit -m "fix(data): resolve CSV parsing issue with semicolon separator"
git commit -m "docs(M1): add GMM interpretation guidelines"
git commit -m "refactor(utils): optimize data loading performance"
git commit -m "test(BN): add unit tests for CPD estimation"
```

---

## 🔧 Essential Git Commands

### Basic Operations

```bash
# Check repository status
git status

# View commit history
git log --oneline --graph

# View changes
git diff                    # Unstaged changes
git diff --staged          # Staged changes
git diff main..your-branch # Compare branches

# Undo changes
git checkout -- filename   # Discard unstaged changes
git reset HEAD filename     # Unstage file
git reset --soft HEAD~1     # Undo last commit (keep changes)
git reset --hard HEAD~1     # Undo last commit (discard changes)
```

### Branch Management

```bash
# List branches
git branch -a              # All branches
git branch -r              # Remote branches

# Switch branches
git checkout branch-name
git switch branch-name     # Modern alternative

# Delete branches
git branch -d feature-branch           # Delete local branch
git push origin --delete feature-branch # Delete remote branch

# Rename branch
git branch -m old-name new-name
```

### Collaborative Features

```bash
# Fetch latest changes without merging
git fetch origin

# Pull changes from main
git pull origin main

# Merge vs Rebase
git merge main             # Creates merge commit
git rebase main           # Replays commits on top of main

# Stash changes temporarily
git stash                 # Save current changes
git stash pop            # Restore stashed changes
git stash list           # List all stashes
```

---

## 📊 Project-Specific Workflow

### Milestone Development Process

1. **Create Milestone Branch**
   ```bash
   git checkout -b feature/milestone-m1-gmm
   ```

2. **Work on Milestone Tasks**
   - Implement required algorithms in `src/project/`
   - Create/update milestone notebook
   - Add visualizations to `reports/figures/`
   - Update milestone README.md

3. **Commit Progress Regularly**
   ```bash
   git add milestones/M1/M1_G14.ipynb src/project/gmm_utils.py
   git commit -m "feat(M1): implement GMM cluster analysis

   - Add patient risk stratification using GMM
   - Include model selection with BIC/AIC
   - Create visualization utilities
   - Add clinical interpretation framework"
   ```

4. **Sync with Main Branch**
   ```bash
   git checkout main
   git pull origin main
   git checkout feature/milestone-m1-gmm
   git rebase main  # Keep clean history
   ```

5. **Create Pull Request**
   - Push branch: `git push origin feature/milestone-m1-gmm`
   - Go to GitHub and create PR
   - Add descriptive title and detailed description
   - Request review from collaborators

### Handling Merge Conflicts

```bash
# When conflicts occur during merge/rebase
git status                 # See conflicted files

# Edit conflicted files, remove conflict markers:
# <<<<<<< HEAD
# Your changes
# =======
# Their changes
# >>>>>>> branch-name

# Mark conflicts as resolved
git add conflicted-file.py

# Continue merge/rebase
git rebase --continue     # For rebase
git commit               # For merge
```

---

## 🎯 Best Practices

### Code Organization

- **Atomic commits**: Each commit should represent one logical change
- **Clear messages**: Describe what and why, not just what
- **Small PRs**: Easier to review and less likely to have conflicts
- **Regular pushes**: Don't lose work, push frequently

### Collaboration Guidelines

1. **Communication**
   - Create GitHub issues for planned work
   - Use descriptive PR titles and descriptions
   - Tag collaborators for reviews: `@username`

2. **Code Review Process**
   - Review code for correctness and style
   - Test notebooks end-to-end before approval
   - Check that figures are saved to `reports/figures/`
   - Verify documentation is updated

3. **Branch Management**
   - Delete feature branches after merging
   - Keep main branch clean and deployable
   - Use descriptive branch names

### Data Management

- **Never commit large files** (>50MB)
- **Raw data goes in `data/raw/`** (add to .gitignore)
- **Processed data** can be committed if small
- **Document data sources** in `data/README.md`

### Notebook Guidelines

- **Clear outputs**: Remove or minimize cell outputs before committing
- **Self-contained**: Include imports and path setup
- **Save figures**: Export plots to `reports/figures/`
- **Add markdown**: Document analysis steps and findings

```python
# Standard notebook setup
import sys, os
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), '..', '..', 'src')))

from project.data_loader import CVDDataLoader
from project.gmm_utils import CVDGaussianMixture
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# Set random seed for reproducibility
np.random.seed(42)
```

---

## 🔍 Troubleshooting Common Issues

### Authentication Issues

```bash
# Use personal access token instead of password
git remote set-url origin https://YOUR_USERNAME:YOUR_TOKEN@github.com/YOUR_USERNAME/repo-name.git

# Or set up SSH keys (recommended)
ssh-keygen -t ed25519 -C "your_email@example.com"
# Add SSH key to GitHub account
git remote set-url origin git@github.com:YOUR_USERNAME/repo-name.git
```

### Large File Issues

```bash
# If you accidentally committed large files
git filter-branch --force --index-filter \
'git rm --cached --ignore-unmatch data/raw/large_file.csv' \
--prune-empty --tag-name-filter cat -- --all

# Then force push (be careful!)
git push origin --force --all
```

### Sync Issues

```bash
# If your branch is behind
git checkout main
git pull origin main
git checkout your-branch
git rebase main

# If you need to reset to remote state
git fetch origin
git reset --hard origin/main
```

---

## 📚 Useful Resources

- **Git Cheat Sheet**: https://education.github.com/git-cheat-sheet-education.pdf
- **Interactive Git Tutorial**: https://learngitbranching.js.org/
- **Conventional Commits**: https://www.conventionalcommits.org/
- **GitHub Flow**: https://guides.github.com/introduction/flow/

---

## 🎓 Academic Collaboration Notes

- **Milestone deadlines**: Create milestone-specific branches well before deadlines
- **Code review**: All team members should review each other's work
- **Documentation**: Maintain clear documentation for reproducibility
- **Backup strategy**: Regular pushes ensure no work is lost
- **Final submission**: Create a release tag for submission versions

```bash
# Create release for milestone submission
git tag -a v1.0-M1-submission -m "Milestone 1 submission: GMM analysis complete"
git push origin v1.0-M1-submission
```
