# Exercise 2: Advanced Git Workflow for AI
# Objective: Master version control for AI projects with data and models

echo -e "\n=== Exercise 2: Advanced Git Workflow for AI ==="

# Create a sample AI project with proper Git setup
mkdir -p ai-git-workflow-demo
cd ai-git-workflow-demo

# Initialize Git repository
git init
echo "Initializing Git repository for AI project..."

# Configure Git LFS for common AI file types
git lfs track "*.pkl"
git lfs track "*.h5"
git lfs track "*.pth"
git lfs track "*.onnx"
git lfs track "*.joblib"
git lfs track "*.model"
git lfs track "*.csv"
git lfs track "*.parquet"
git lfs track "*.json"
git lfs track "data/raw/*"
git lfs track "data/processed/*"
git lfs track "models/trained/*"

# Create comprehensive .gitignore for AI projects
cat > .gitignore << 'EOF'
# Byte-compiled / optimized / DLL files
__pycache__/
*.py[cod]
*$py.class

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
pip-wheel-metadata/
share/python-wheels/
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
*.py,cover
.hypothesis/
.pytest_cache/

# Jupyter Notebook
.ipynb_checkpoints

# IPython
profile_default/
ipython_config.py

# pyenv
.python-version

# pipenv
Pipfile.lock

# PEP 582
__pypackages__/

# Celery stuff
celerybeat-schedule
celerybeat.pid

# SageMath parsed files
*.sage.py

# Environments
.env
.venv
env/
venv/
ENV/
env.bak/
venv.bak/

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

# Pyre type checker
.pyre/

# AI/ML specific
*.pkl
*.h5
*.pth
*.onnx
*.joblib
*.model
data/raw/
data/processed/
data/external/
models/trained/
models/checkpoints/
logs/
runs/
outputs/
artifacts/
mlruns/
.mlflow/
wandb/
.wandb/

# Large datasets
*.csv
*.parquet
*.feather
*.hdf5
*.npz

# IDE
.vscode/
.idea/
*.swp
*.swo
*~

# OS
.DS_Store
.DS_Store?
._*
.Spotlight-V100
.Trashes
ehthumbs.db
Thumbs.db
EOF

# Create project structure
mkdir -p {data/{raw,processed,external},notebooks/{01_data_exploration,02_feature_engineering,03_modeling,04_evaluation},src/{data,features,models,visualization},models/{trained,checkpoints},reports,tests}

# Create sample files to demonstrate the workflow
cat > README.md << 'EOF'
# AI Project with Git LFS Demo

This project demonstrates proper Git workflow for AI/ML projects.

## Structure
- `data/`: Data files (tracked with Git LFS)
- `notebooks/`: Jupyter notebooks for exploration and experimentation
- `src/`: Source code for production
- `models/`: Trained models (tracked with Git LFS)
- `reports/`: Analysis reports and documentation

## Git LFS Setup
This project uses Git LFS to handle large files:
- Model files (.pkl, .pth, .h5, .onnx)
- Data files (.csv, .parquet)
- Checkpoints and artifacts

## Workflow
1. `main` branch: Stable, production-ready code
2. `develop` branch: Integration branch for features
3. `experiment/*` branches: Individual experiments
4. `feature/*` branches: New features and improvements
EOF

# Create sample experiment tracking file
cat > experiments.md << 'EOF'
# Experiment Log

## Experiment 1: Baseline Model
- **Branch**: experiment/baseline-model
- **Date**: 2024-06-26
- **Objective**: Establish baseline performance
- **Results**: Accuracy: 0.85, F1: 0.82
- **Notes**: Simple logistic regression, minimal preprocessing

## Experiment 2: Feature Engineering
- **Branch**: experiment/advanced-features
- **Date**: 2024-06-27
- **Objective**: Improve performance with better features
- **Results**: Accuracy: 0.89, F1: 0.86
- **Notes**: Added polynomial features and PCA
EOF

# Create sample source code
cat > src/models/train_model.py << 'EOF'
"""
Model training utilities for AI project
"""
import pickle
import logging
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

def train_and_save_model(model, X, y, model_path, test_size=0.2, random_state=42):
    """Train model and save to disk"""
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    
    # Train model
    logging.info("Training model...")
    model.fit(X_train, y_train)
    
    # Evaluate
    y_pred = model.predict(X_test)
    report = classification_report(y_test, y_pred, output_dict=True)
    
    # Save model
    model_path = Path(model_path)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)
    
    logging.info(f"Model saved to {model_path}")
    
    return model, report
EOF

# Create sample notebook
cat > notebooks/01_data_exploration/data_overview.ipynb << 'EOF'
{
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# Data Exploration\n",
    "\n",
    "This notebook provides an overview of the dataset and initial exploration."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "source": [
    "import pandas as pd\n",
    "import numpy as np\n",
    "import matplotlib.pyplot as plt\n",
    "import seaborn as sns\n",
    "\n",
    "# Load data\n",
    "# df = pd.read_csv('../data/raw/dataset.csv')\n",
    "print('Data exploration notebook ready!')"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.8.5"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 4
}
EOF

# Add all files to git
git add .gitattributes  # Git LFS tracking file
git add .
git commit -m "Initial project setup with Git LFS configuration"

# Create and demonstrate branching strategy
echo "Demonstrating AI project branching strategy..."

# Create develop branch
git checkout -b develop
echo "# Development branch for integrating features" >> README.md
git add README.md
git commit -m "Setup develop branch"

# Create experiment branch
git checkout -b experiment/baseline-model
echo "Working on baseline model..." > experiments/baseline_log.txt
git add experiments/
git commit -m "Add baseline model experiment"

# Create feature branch
git checkout develop
git checkout -b feature/data-preprocessing
mkdir -p src/data
cat > src/data/preprocessing.py << 'EOF'
"""
Data preprocessing utilities
"""
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder

def clean_data(df):
    """Clean and preprocess data"""
    # Remove duplicates
    df = df.drop_duplicates()
    
    # Handle missing values
    df = df.fillna(df.mean())
    
    return df

def encode_categorical_features(df, categorical_columns):
    """Encode categorical features"""
    encoders = {}
    df_encoded = df.copy()
    
    for col in categorical_columns:
        encoder = LabelEncoder()
        df_encoded[col] = encoder.fit_transform(df[col])
        encoders[col] = encoder
    
    return df_encoded, encoders
EOF

git add src/data/preprocessing.py
git commit -m "Add data preprocessing utilities"

# Merge feature back to develop
git checkout develop
git merge feature/data-preprocessing --no-ff -m "Merge data preprocessing feature"

# Show branch structure
echo "Git branch structure:"
git log --oneline --graph --all --decorate

echo "✅ Exercise 2 Complete: Advanced Git workflow with LFS and branching strategy set up"

# Return to original directory
cd ..
