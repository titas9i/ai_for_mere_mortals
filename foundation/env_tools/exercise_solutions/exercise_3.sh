
# Exercise 3: Custom Jupyter Kernel Setup
# Objective: Configure specialized Jupyter kernels for different AI domains

echo -e "\n=== Exercise 3: Custom Jupyter Kernel Setup ==="

# Create specialized environments with custom kernels

# Data Science Kernel (for exploration)
echo "Creating Data Science kernel environment..."
cat > ds_kernel_environment.yml << 'EOF'
name: ds-kernel
channels:
  - conda-forge
  - defaults
dependencies:
  - python=3.10
  - ipykernel
  - pandas
  - numpy
  - matplotlib
  - seaborn
  - plotly
  - jupyter
  - scikit-learn
  - statsmodels
  - scipy
  - pip
  - pip:
    - pandas-profiling
    - sweetviz
    - dtale
EOF

# Deep Learning Kernel (for model training)
cat > dl_kernel_environment.yml << 'EOF'
name: dl-kernel
channels:
  - conda-forge
  - pytorch
  - defaults
dependencies:
  - python=3.10
  - ipykernel
  - pytorch
  - torchvision
  - torchaudio
  - jupyter
  - matplotlib
  - tensorboard
  - pip
  - pip:
    - wandb
    - lightning
    - torchmetrics
    - timm
EOF

# Production Kernel (for inference)
cat > prod_kernel_environment.yml << 'EOF'
name: prod-kernel
channels:
  - conda-forge
  - defaults
dependencies:
  - python=3.10
  - ipykernel
  - numpy
  - pandas
  - scikit-learn
  - joblib
  - fastapi
  - uvicorn
  - pydantic
  - jupyter
  - pip
  - pip:
    - mlflow
    - evidently
EOF

# Function to create environment and register kernel using Mamba
create_and_register_kernel() {
    local env_file=$1
    local env_name=$2
    local display_name=$3
    
    echo "Creating environment with Mamba: $env_name"
    mamba env create -f $env_file
    
    echo "Registering Jupyter kernel: $display_name"
    mamba activate $env_name
    python -m ipykernel install --user --name $env_name --display-name "$display_name"
    mamba deactivate
}

# Create all kernels
create_and_register_kernel "ds_kernel_environment.yml" "ds-kernel" "Data Science (Exploration)"
create_and_register_kernel "dl_kernel_environment.yml" "dl-kernel" "Deep Learning (Training)"
create_and_register_kernel "prod_kernel_environment.yml" "prod-kernel" "Production (Inference)"

# List available kernels
echo "Available Jupyter kernels:"
jupyter kernelspec list

# Create kernel management script
cat > manage_kernels.py << 'EOF'
#!/usr/bin/env python3
"""
Kernel Management Script for AI Development
"""
import subprocess
import json
import sys

def list_kernels():
    """List all available Jupyter kernels"""
    result = subprocess.run(['jupyter', 'kernelspec', 'list', '--json'], 
                          capture_output=True, text=True)
    kernels = json.loads(result.stdout)
    
    print("Available Jupyter Kernels:")
    print("-" * 40)
    for name, info in kernels['kernelspecs'].items():
        print(f"Name: {name}")
        print(f"Display Name: {info['spec']['display_name']}")
        print(f"Language: {info['spec']['language']}")
        print(f"Location: {info['resource_dir']}")
        print()

def create_kernel_from_env(env_name, display_name=None):
    """Create Jupyter kernel from mamba environment"""
    if not display_name:
        display_name = env_name
    
    # Activate environment and install kernel
    commands = [
        f"mamba activate {env_name}",
        f"python -m ipykernel install --user --name {env_name} --display-name '{display_name}'"
    ]
    
    for cmd in commands:
        subprocess.run(cmd, shell=True, check=True)
    
    print(f"Kernel '{display_name}' created from environment '{env_name}'")

def remove_kernel(kernel_name):
    """Remove a Jupyter kernel"""
    subprocess.run(['jupyter', 'kernelspec', 'uninstall', kernel_name, '-f'], check=True)
    print(f"Kernel '{kernel_name}' removed")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage:")
        print("  python manage_kernels.py list")
        print("  python manage_kernels.py create <env_name> [display_name]")
        print("  python manage_kernels.py remove <kernel_name>")
        sys.exit(1)
    
    command = sys.argv[1]
    
    if command == "list":
        list_kernels()
    elif command == "create" and len(sys.argv) >= 3:
        env_name = sys.argv[2]
        display_name = sys.argv[3] if len(sys.argv) > 3 else env_name
        create_kernel_from_env(env_name, display_name)
    elif command == "remove" and len(sys.argv) >= 3:
        kernel_name = sys.argv[2]
        remove_kernel(kernel_name)
    else:
        print("Invalid command or missing arguments")
        sys.exit(1)
EOF

chmod +x manage_kernels.py

# Create kernel testing notebook
cat > test_kernels.ipynb << 'EOF'
{
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# Kernel Testing Notebook\n",
    "\n",
    "Use this notebook to test different kernels by changing the kernel from the Jupyter menu.\n",
    "\n",
    "## Data Science Kernel Test"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Test Data Science packages\n",
    "try:\n",
    "    import pandas as pd\n",
    "    import numpy as np\n",
    "    import matplotlib.pyplot as plt\n",
    "    import seaborn as sns\n",
    "    print(\"✅ Data Science kernel working!\")\n",
    "    print(f\"Pandas: {pd.__version__}\")\n",
    "    print(f\"NumPy: {np.__version__}\")\n",
    "except ImportError as e:\n",
    "    print(f\"❌ Data Science packages not available: {e}\")"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Deep Learning Kernel Test"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Test Deep Learning packages\n",
    "try:\n",
    "    import torch\n",
    "    import torchvision\n",
    "    print(\"✅ Deep Learning kernel working!\")\n",
    "    print(f\"PyTorch: {torch.__version__}\")\n",
    "    print(f\"CUDA available: {torch.cuda.is_available()}\")\n",
    "except ImportError as e:\n",
    "    print(f\"❌ Deep Learning packages not available: {e}\")"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Production Kernel Test"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Test Production packages\n",
    "try:\n",
    "    import fastapi\n",
    "    import mlflow\n",
    "    import joblib\n",
    "    print(\"✅ Production kernel working!\")\n",
    "    print(f\"FastAPI: {fastapi.__version__}\")\n",
    "    print(f\"MLflow: {mlflow.__version__}\")\n",
    "except ImportError as e:\n",
    "    print(f\"❌ Production packages not available: {e}\")"
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
   "version": "3.10.0"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 4
}
EOF

echo "✅ Exercise 3 Complete: Custom Jupyter kernels created and configured with Mamba"

# Summary and cleanup instructions
cat > exercise_summary.md << 'EOF'
# Module 1.2 Exercise Solutions Summary (Using Mamba)

## Exercise 1: Multi-Project Environment Management ✅
- Created specialized environments for Computer Vision and NLP projects using Mamba
- Each environment has domain-specific packages (OpenCV vs Transformers)
- Demonstrated environment switching and package verification
- Experienced Mamba's speed advantage over traditional conda

**Files created:**
- `cv_environment.yml` - Computer Vision environment specification
- `nlp_environment.yml` - NLP environment specification

**Commands to verify:**
```bash
mamba env list  # Should show cv-projects and nlp-projects
mamba activate cv-projects
python -c "import cv2; print(cv2.__version__)"
mamba activate nlp-projects  
python -c "import transformers; print(transformers.__version__)"
```

## Exercise 2: Advanced Git Workflow for AI ✅
- Set up Git LFS for handling large AI files (models, datasets)
- Created comprehensive .gitignore for AI projects
- Demonstrated branching strategy for AI experiments

**Directory created:** `ai-git-workflow-demo/`
**Key features:**
- Git LFS tracking for .pkl, .pth, .h5, .csv files
- Proper project structure for AI development
- Branching strategy: main → develop → feature/experiment branches

**Commands to explore:**
```bash
cd ai-git-workflow-demo
git log --oneline --graph --all --decorate
git lfs ls-files
```

## Exercise 3: Custom Jupyter Kernel Setup ✅
- Created three specialized Jupyter kernels for different AI workflows using Mamba
- Data Science kernel: exploration and analysis
- Deep Learning kernel: model training and experimentation  
- Production kernel: inference and deployment

**Files created:**
- `ds_kernel_environment.yml` - Data Science kernel environment
- `dl_kernel_environment.yml` - Deep Learning kernel environment
- `prod_kernel_environment.yml` - Production kernel environment
- `manage_kernels.py` - Kernel management utility (updated for mamba)
- `test_kernels.ipynb` - Kernel testing notebook

**Commands to verify:**
```bash
jupyter kernelspec list  # Should show all three custom kernels
python manage_kernels.py list  # Alternative kernel listing
jupyter lab test_kernels.ipynb  # Test kernels interactively
```

## Key Learnings

1. **Mamba Speed Advantage**: Mamba significantly outperforms conda for complex AI environments
2. **Environment Isolation**: Each AI project should have its own environment to prevent package conflicts
3. **Version Control for AI**: Large files need special handling with Git LFS
4. **Specialized Kernels**: Different AI workflows benefit from tailored development environments
5. **Reproducibility**: Environment files ensure consistent setups across different machines

## Mamba vs Conda Benefits Observed

- **Faster Environment Creation**: 2-5x faster dependency solving
- **Parallel Downloads**: Multiple packages downloaded simultaneously
- **Better Error Messages**: Clearer conflict resolution information
- **Same Syntax**: Drop-in replacement for conda commands

## Next Steps

1. Practice switching between environments for different types of AI projects
2. Use the Git LFS workflow for your own AI projects with large datasets
3. Experiment with the custom Jupyter kernels for different development phases
4. Customize the environments further based on your specific AI interests
5. Consider using mamba for all future AI environment management

## Troubleshooting

If you encounter issues:

1. **Environment creation fails**: Check internet connection and conda-forge channel availability
2. **Kernel not appearing**: Run `jupyter kernelspec list` to verify installation
3. **Git LFS issues**: Ensure Git LFS is installed with `git lfs install`
4. **Import errors**: Verify you're in the correct mamba environment
5. **Mamba not found**: Ensure Miniforge is properly installed or install mamba via `conda install mamba -c conda-forge`

For more help, refer to the respective documentation:
- [Mamba Documentation](https://mamba.readthedocs.io/)
- [Conda Documentation](https://docs.conda.io/)
- [Jupyter Kernels](https://jupyter-client.readthedocs.io/en/stable/kernels.html)
- [Git LFS](https://git-lfs.github.io/)
EOF

echo -e "\n=== All Exercises Complete with Mamba! ==="
echo "Check 'exercise_summary.md' for a complete overview of what was accomplished."
echo "Your AI development environment is now professionally configured with lightning-fast Mamba! 🚀"