# Git Quick Reference for CVD ML Project

## 🚀 Getting Started

```bash
# Run the setup script
./setup_git.sh

# Connect to GitHub (replace with your repository URL)
git remote add origin https://github.com/YOUR_USERNAME/adv-ml-cvd-project.git
git push -u origin main
```

## 📋 Daily Workflow

### Start Working on a Milestone
```bash
# Switch to milestone branch
git checkout feature/milestone-m1-gmm

# Sync with latest changes
git pull origin main
git merge main
```

### Make and Commit Changes
```bash
# Check what changed
git status
git diff

# Add specific files
git add milestones/M1/M1_G14.ipynb src/project/gmm_utils.py

# Or add all changes
git add .

# Commit with descriptive message
git commit -m "feat(M1): implement GMM cluster analysis

- Add patient risk stratification
- Include model selection with BIC/AIC  
- Create visualization utilities"

# Push to GitHub
git push origin feature/milestone-m1-gmm
```

### Create Pull Request
1. Go to GitHub repository
2. Click "New Pull Request"
3. Select your feature branch
4. Add title: `feat(M1): Implement GMM Patient Clustering`
5. Add description with what you implemented
6. Request review from collaborators

## 🔧 Common Commands

### Branch Management
```bash
# List all branches
git branch -a

# Create new branch
git checkout -b feature/new-feature

# Switch branches
git checkout main
git checkout feature/milestone-m2-bayesian-networks

# Delete branch (after merging)
git branch -d feature/completed-feature
```

### Syncing with Team
```bash
# Get latest changes from main
git checkout main
git pull origin main

# Update your feature branch
git checkout your-feature-branch
git merge main

# Alternative: rebase (cleaner history)
git rebase main
```

### Undoing Changes
```bash
# Discard unstaged changes
git checkout -- filename.py

# Unstage file
git reset HEAD filename.py

# Undo last commit (keep changes)
git reset --soft HEAD~1

# Undo last commit (discard changes)
git reset --hard HEAD~1
```

## 📊 Project-Specific Tips

### Milestone Development
```bash
# Work on M1
git checkout feature/milestone-m1-gmm

# Work on M2  
git checkout feature/milestone-m2-bayesian-networks

# Work on documentation
git checkout docs/project-documentation
```

### Commit Message Examples
```bash
# Feature implementation
git commit -m "feat(M1): implement GMM clustering with risk stratification"

# Bug fix
git commit -m "fix(data): resolve CSV parsing issue with semicolons"

# Documentation
git commit -m "docs(M2): add Bayesian Network design guidelines"

# Refactoring
git commit -m "refactor(utils): optimize data loading performance"

# Testing
git commit -m "test(BN): add unit tests for CPD estimation"
```

### Working with Notebooks
```bash
# Before committing notebooks, clean outputs (optional)
jupyter nbconvert --ClearOutputPreprocessor.enabled=True --inplace notebook.ipynb

# Always save figures
plt.savefig('../../reports/figures/m1_gmm_clusters.png', dpi=300, bbox_inches='tight')
```

## 🆘 Emergency Commands

### Large File Accidentally Committed
```bash
# Remove from Git history (be careful!)
git filter-branch --force --index-filter \
'git rm --cached --ignore-unmatch data/raw/large_file.csv' \
--prune-empty --tag-name-filter cat -- --all

git push origin --force --all
```

### Merge Conflicts
```bash
# When conflicts occur
git status  # See conflicted files

# Edit files to resolve conflicts, then:
git add resolved_file.py
git commit -m "resolve: merge conflict in resolved_file.py"
```

### Reset to Remote State
```bash
# Nuclear option - reset to remote main
git fetch origin
git reset --hard origin/main
```

## 🎯 Collaboration Best Practices

### Before Starting Work
1. `git checkout main`
2. `git pull origin main`
3. `git checkout your-feature-branch`
4. `git merge main`

### Before Creating PR
1. Test your code thoroughly
2. Update documentation
3. Clean notebook outputs
4. Save figures to `reports/figures/`
5. Write descriptive commit messages

### Code Review Checklist
- [ ] Code runs without errors
- [ ] Notebooks execute end-to-end
- [ ] Figures are saved appropriately
- [ ] Documentation is updated
- [ ] Commit messages are clear
- [ ] No sensitive data committed

## 📚 Help Resources

- **Git Basics**: https://git-scm.com/book
- **GitHub Flow**: https://guides.github.com/introduction/flow/
- **Conventional Commits**: https://www.conventionalcommits.org/
- **Interactive Tutorial**: https://learngitbranching.js.org/

## 🎓 Academic Notes

### For Milestone Submissions
```bash
# Create submission tag
git tag -a v1.0-M1 -m "Milestone 1 submission: GMM analysis complete"
git push origin v1.0-M1

# Create release branch for final submission
git checkout -b release/final-submission
git push origin release/final-submission
```

### Team Coordination
- Use GitHub Issues for task planning
- Tag teammates in PR reviews: `@username`
- Use project boards for milestone tracking
- Regular team sync meetings to avoid conflicts