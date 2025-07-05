import logging
import pickle
import json
from pathlib import Path
import pandas as pd
from typing import Any, Dict, List

def setup_logging(config: Dict) -> None:
    """Setup logging configuration for the project"""
    import logging.config
    logging.config.dictConfig(config)
def save_model(model: Any, filepath: Path) -> None:
    """Save trained model to disk"""
    filepath.parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, 'wb') as f:
        pickle.dump(model, f)
    logging.info(f"Model saved to {filepath}")
def load_model(filepath: Path) -> Any:
    """Load trained model from disk"""
    with open(filepath, 'rb') as f:
        model = pickle.load(f)
    logging.info(f"Model loaded from {filepath}")
    return model
def save_experiment_results(results: Dict, filepath: Path) -> None:
    """Save experiment results as JSON"""
    filepath.parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    logging.info(f"Results saved to {filepath}")
def load_data(filepath: Path) -> pd.DataFrame:
    """Load data with automatic format detection"""
    if filepath.suffix == '.csv':
        return pd.read_csv(filepath)
    elif filepath.suffix == '.parquet':
        return pd.read_parquet(filepath)
    elif filepath.suffix == '.json':
        return pd.read_json(filepath)
    else:
        raise ValueError(f"Unsupported file format: {filepath.suffix}")
def validate_environment() -> Dict[str, str]:
    """Validate that all required packages are installed"""
    import sys
    import importlib.util
    
    required_packages = [
        'pandas', 'numpy', 'scikit-learn', 'matplotlib', 
        'seaborn', 'torch', 'jupyter'
    ]
    
    results = {}
    for package in required_packages:
        spec = importlib.util.find_spec(package)
        if spec is not None:
            try:
                module = importlib.import_module(package)
                version = getattr(module, '__version__', 'unknown')
                results[package] = version
            except ImportError:
                results[package] = 'import_error'
        else:
            results[package] = 'not_found'
    
    results['python'] = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
    return results