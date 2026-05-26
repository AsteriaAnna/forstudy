"""Utility functions for PHM2010 project."""

from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np


# Project root directory
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RESULTS_DIR = PROJECT_ROOT / "results"

# Data subdirectories
DATA_RAW = DATA_DIR / "raw"
DATA_INTERIM = DATA_DIR / "interim"
DATA_INTERIM_FEATURES = DATA_INTERIM / "features"
DATA_INTERIM_SAMPLES = DATA_INTERIM / "samples"
DATA_PROCESSED = DATA_DIR / "processed"


def ensure_dir(path: str | Path) -> None:
    """Create directory if it doesn't exist.
    
    Args:
        path: Directory path to create
    """
    Path(path).mkdir(parents=True, exist_ok=True)


def get_run_id() -> str:
    """Generate run ID with auto-incrementing run number.
    
    Returns:
        Run ID in format: run_001_20250524_1530
        (run_NNN_YYYYMMDD_HHMM, run number is zero-padded 3 digits)
    """
    now = datetime.now()
    date_str = now.strftime("%Y%m%d")
    time_str = now.strftime("%H%M")
    
    # Track run counter in a file within results directory
    counter_file = RESULTS_DIR / ".run_counter"
    
    # Read current counter
    current_counter = 0
    if counter_file.exists():
        try:
            content = counter_file.read_text().strip()
            if content:
                current_counter = int(content)
        except (ValueError, IOError):
            current_counter = 0
    
    # Increment counter
    current_counter += 1
    
    # Ensure results directory exists for counter file
    ensure_dir(RESULTS_DIR)
    
    # Write updated counter
    counter_file.write_text(str(current_counter))
    
    run_num_str = f"{current_counter:03d}"
    return f"run_{run_num_str}_{date_str}_{time_str}"


def get_data_dir(run_id: str) -> str:
    """Get data directory path for a run.
    
    Args:
        run_id: Run identifier
        
    Returns:
        Path to run's data directory: results/{run_id}/data/
    """
    return str(RESULTS_DIR / run_id / "data")


def get_results_dir(run_id: str) -> str:
    """Get results directory path for a run.
    
    Args:
        run_id: Run identifier
        
    Returns:
        Path to run's results directory: results/{run_id}/
    """
    return str(RESULTS_DIR / run_id)


def save_signal(
    signal: np.ndarray,
    cut_unique_id: str,
    run_id: str,
    prefix: str
) -> str:
    """Save signal array to .npy file.
    
    Args:
        signal: Signal numpy array to save
        cut_unique_id: Unique identifier for the cut
        run_id: Run identifier
        prefix: Prefix for the signal file (e.g., 'raw', 'filtered')
        
    Returns:
        Path to the saved file
    """
    data_dir = get_data_dir(run_id)
    ensure_dir(data_dir)
    
    filename = f"{prefix}_{cut_unique_id}.npy"
    filepath = Path(data_dir) / filename
    np.save(filepath, signal)
    
    return str(filepath)


def load_signal(
    cut_unique_id: str,
    run_id: str,
    prefix: str
) -> np.ndarray:
    """Load signal array from .npy file.
    
    Args:
        cut_unique_id: Unique identifier for the cut
        run_id: Run identifier
        prefix: Prefix for the signal file (e.g., 'raw', 'filtered')
        
    Returns:
        Loaded signal numpy array
        
    Raises:
        FileNotFoundError: If the signal file doesn't exist
    """
    data_dir = get_data_dir(run_id)
    filename = f"{prefix}_{cut_unique_id}.npy"
    filepath = Path(data_dir) / filename
    
    if not filepath.exists():
        raise FileNotFoundError(f"Signal file not found: {filepath}")
    
    return np.load(filepath)


def get_cache_path(run_id: str, stage: int | str) -> str:
    """Get cache directory path for a processing stage.
    
    Args:
        run_id: Run identifier
        stage: Stage number or name (e.g., 1, 2, 3, 'stage1', 'features')
        
    Returns:
        Path to the cache directory for the stage
    """
    stage_mapping = {
        1: "stage1_preprocessing",
        2: "stage2_feature_extraction",
        3: "stage3_dataset_creation",
        4: "features",  # maps to data/interim/features/
        5: "samples",   # maps to data/interim/samples/
        "stage1": "stage1_preprocessing",
        "stage2": "stage2_feature_extraction",
        "stage3": "stage3_dataset_creation",
        "features": "features",
        "samples": "samples",
    }
    
    # Handle both int and string stage input
    if isinstance(stage, int):
        stage_name = stage_mapping.get(stage, f"stage{stage}")
    else:
        stage_name = stage_mapping.get(stage, stage)
    
    # Stages 1-3 go to results/{run_id}/data/
    # Stages 4+ (features, samples) go to data/interim/
    if isinstance(stage, int) and stage <= 3:
        cache_path = Path(get_data_dir(run_id)) / "cache" / stage_name
    elif stage in ("features", 4):
        cache_path = DATA_INTERIM_FEATURES / run_id
    elif stage in ("samples", 5):
        cache_path = DATA_INTERIM_SAMPLES / run_id
    else:
        cache_path = Path(get_data_dir(run_id)) / "cache" / stage_name
    
    ensure_dir(cache_path)
    return str(cache_path)


def get_logs_dir(run_id: str) -> str:
    """Get logs directory path for a run.
    
    Args:
        run_id: Run identifier
        
    Returns:
        Path to run's logs directory: results/{run_id}/logs/
    """
    logs_dir = RESULTS_DIR / run_id / "logs"
    ensure_dir(logs_dir)
    return str(logs_dir)


def get_figures_dir(run_id: str) -> str:
    """Get figures directory path for a run.
    
    Args:
        run_id: Run identifier
        
    Returns:
        Path to run's figures directory: results/{run_id}/figures/
    """
    figures_dir = RESULTS_DIR / run_id / "figures"
    ensure_dir(figures_dir)
    return str(figures_dir)


def get_interim_dir(subdir: Optional[str] = None) -> str:
    """Get interim data directory path.
    
    Args:
        subdir: Optional subdirectory (e.g., 'features', 'samples')
        
    Returns:
        Path to interim data directory or subdirectory
    """
    if subdir:
        return str(DATA_INTERIM / subdir)
    return str(DATA_INTERIM)


def get_raw_data_dir() -> str:
    """Get raw data directory path.
    
    Returns:
        Path to raw data directory: data/raw/
    """
    return str(DATA_RAW)


def get_processed_data_dir() -> str:
    """Get processed data directory path.
    
    Returns:
        Path to processed data directory: data/processed/
    """
    return str(DATA_PROCESSED)
