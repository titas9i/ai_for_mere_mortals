
# Exercise 1: Multi-Project Environment Management
# Objective: Learn to manage multiple AI projects with different requirements

echo "=== Exercise 1: Multi-Project Environment Management ==="

# Create Computer Vision Environment
echo "Creating Computer Vision environment..."
cat > cv_environment.yml << 'EOF'
name: cv-projects
channels:
  - conda-forge
  - pytorch
  - defaults
dependencies:
  - python=3.10
  - pytorch
  - torchvision
  - torchaudio
  - opencv
  - pillow
  - matplotlib
  - numpy
  - jupyter
  - scikit-image
  - albumentations
  - pip
  - pip:
    - ultralytics  # YOLO models
    - timm        # Vision transformer models
    - wandb       # Experiment tracking
EOF

# Create NLP Environment
echo "Creating NLP environment..."
cat > nlp_environment.yml << 'EOF'
name: nlp-projects
channels:
  - conda-forge
  - pytorch
  - huggingface
  - defaults
dependencies:
  - python=3.10
  - pytorch
  - transformers
  - datasets
  - tokenizers
  - numpy
  - pandas
  - matplotlib
  - jupyter
  - nltk
  - spacy
  - pip
  - pip:
    - langchain
    - openai
    - sentence-transformers
    - evaluate
    - accelerate
EOF

# Create the environments using Mamba (much faster than conda!)
echo "Building Computer Vision environment with Mamba (notice the speed!)..."
mamba env create -f cv_environment.yml

echo "Building NLP environment with Mamba..."
mamba env create -f nlp_environment.yml

# Test environment switching
echo "Testing environment switching..."

echo "Activating CV environment and checking packages..."
mamba activate cv-projects
python -c "
import cv2
import torch
import torchvision
print(f'CV Environment Active:')
print(f'  OpenCV: {cv2.__version__}')
print(f'  PyTorch: {torch.__version__}')
print(f'  TorchVision: {torchvision.__version__}')
"

echo "Switching to NLP environment..."
mamba activate nlp-projects
python -c "
import transformers
import torch
import nltk
print(f'NLP Environment Active:')
print(f'  Transformers: {transformers.__version__}')
print(f'  PyTorch: {torch.__version__}')
"

echo "✅ Exercise 1 Complete: Successfully created and tested CV and NLP environments with Mamba"
