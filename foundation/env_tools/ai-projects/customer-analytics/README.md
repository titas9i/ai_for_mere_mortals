## Project Structure
├── data/
│   ├── raw/          # Original, immutable data
│   ├── processed/    # Cleaned, feature-engineered data
│   └── external/     # External datasets and references
├── notebooks/
│   ├── exploratory/  # Data exploration and analysis
│   └── modeling/     # Model development and evaluation
├── src/             # Source code for production
├── models/          # Trained models and serialized objects
├── reports/         # Analysis reports and documentation
└── tests/           # Unit tests for your code

## Getting Started
1. Create environment: `mamba env create -f environment.yml`
2. Activate environment: `mamba activate project-name`
3. Install in development mode: `pip install -e .`
