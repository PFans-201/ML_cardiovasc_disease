#!/bin/bash
# Git Repository Setup Script for Advanced ML CVD Project
# Run this script to initialize your Git repository and set up collaboration

echo "🚀 Setting up Git repository for Advanced ML CVD Project..."

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_step() {
    echo -e "${BLUE}[STEP]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if we're in the right directory
if [[ ! -f "README.md" ]] || [[ ! -d "milestones" ]]; then
    print_error "Please run this script from the project root directory"
    exit 1
fi

# Step 1: Initialize Git repository
print_step "Initializing Git repository..."
if [[ ! -d ".git" ]]; then
    git init
    print_success "Git repository initialized"
else
    print_warning "Git repository already exists"
fi

# Step 2: Create .gitignore if it doesn't exist
print_step "Setting up .gitignore..."
if [[ ! -f ".gitignore" ]]; then
    cat > .gitignore << 'EOF'
# Byte-compiled / optimized / DLL files
__pycache__/
*.py[cod]
*$py.class

# C extensions
*.so

# Distribution / packaging
.Python
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
wheels/
*.egg-info/
.installed.cfg
*.egg
MANIFEST

# PyInstaller
*.manifest
*.spec

# Installer logs
pip-log.txt
pip-delete-this-directory.txt

# Unit test / coverage reports
htmlcov/
.tox/
.nox/
.coverage
.coverage.*
.cache
nosetests.xml
coverage.xml
*.cover
.hypothesis/
.pytest_cache/

# Jupyter Notebook
.ipynb_checkpoints

# IPython
profile_default/
ipython_config.py

# pyenv
.python-version

# Environments
.env
.venv
env/
venv/
ENV/
env.bak/
venv.bak/
adv_ml_venv/

# Spyder project settings
.spyderproject
.spyproject

# Rope project settings
.ropeproject

# mkdocs documentation
/site

# mypy
.mypy_cache/
.dmypy.json
dmypy.json

# Data files (keep structure, ignore content)
data/raw/*.csv
data/raw/*.xlsx
data/raw/*.json
data/raw/*.parquet
data/interim/*.csv
data/interim/*.xlsx
data/processed/*.csv
data/processed/*.xlsx
*.h5
*.hdf5

# Large model files
models/*.pkl
models/*.joblib
models/*.h5
models/*.hdf5

# OS generated files
.DS_Store
.DS_Store?
._*
.Spotlight-V100
.Trashes
ehthumbs.db
Thumbs.db

# IDE
.vscode/
.idea/
*.swp
*.swo
*~

# Temporary files
*.tmp
*.temp
EOF
    print_success ".gitignore created"
else
    print_warning ".gitignore already exists"
fi

# Step 3: Add all files to staging
print_step "Adding files to Git staging area..."
git add .
print_success "Files added to staging"

# Step 4: Create initial commit
print_step "Creating initial commit..."
if ! git diff --cached --quiet; then
    git commit -m "feat: initial project setup with probabilistic ML structure

- Add comprehensive milestone structure (M0-M6)
- Implement specialized utilities (GMM, BN, HMM, RL) 
- Include professional documentation and guidelines
- Set up data pipeline and reproducible environment
- Add collaborative development workflow"
    print_success "Initial commit created"
else
    print_warning "No changes to commit"
fi

# Step 5: Instructions for GitHub setup
echo ""
echo "📋 Next steps to complete GitHub setup:"
echo ""
echo "1. Create a new repository on GitHub:"
echo "   - Go to https://github.com/new"
echo "   - Repository name: ML_cardiovasc_disease"
echo "   - Description: Advanced Machine Learning for Cardiovascular Disease Prediction & Treatment Optimization"
echo "   - Choose Private or Public"
echo "   - Do NOT initialize with README, .gitignore, or license"
echo ""
echo "2. Connect your local repository to GitHub:"
echo "   ${YELLOW}git remote add origin https://github.com/YOUR_USERNAME/adv-ml-cvd-project.git${NC}"
echo "   ${YELLOW}git branch -M main${NC}"
echo "   ${YELLOW}git push -u origin main${NC}"
echo ""
echo "3. Set up branch protection (on GitHub web interface):"
echo "   - Go to Settings → Branches"
echo "   - Add rule for 'main' branch"
echo "   - Enable 'Require a pull request before merging'"
echo "   - Set 'Require approvals' to 1"
echo ""

# Step 6: Create initial branch structure
print_step "Creating development branches..."

# Create branches for each milestone
branches=(
    "feature/milestone-m0-eda"
    "feature/milestone-m1-gmm"
    "feature/milestone-m2-bayesian-networks"
    "feature/milestone-m3-bn-inference"
    "feature/milestone-m4-structure-learning"
    "feature/milestone-m5-hmm"
    "feature/milestone-m6-reinforcement-learning"
    "docs/project-documentation"
)

for branch in "${branches[@]}"; do
    git checkout -b "$branch" main
    print_success "Created branch: $branch"
done

# Return to main branch
git checkout main
print_success "Returned to main branch"

echo ""
echo "🎉 Git repository setup complete!"
echo ""
echo "📊 Current branch structure:"
git branch --list
echo ""
echo "🔗 Useful commands to get started:"
echo ""
echo "   Work on Milestone 1:"
echo "   ${YELLOW}git checkout feature/milestone-m1-gmm${NC}"
echo ""
echo "   Make changes and commit:"
echo "   ${YELLOW}git add .${NC}"
echo "   ${YELLOW}git commit -m \"feat(M1): implement GMM patient clustering\"${NC}"
echo "   ${YELLOW}git push origin feature/milestone-m1-gmm${NC}"
echo ""
echo "   Return to main branch:"
echo "   ${YELLOW}git checkout main${NC}"
echo ""
echo "📖 For detailed workflow, see CONTRIBUTING.md"