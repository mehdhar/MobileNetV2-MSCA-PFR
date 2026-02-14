import os

# -------------------------------------------------------------------------
# Path configuration module
# Centralizes all filesystem paths used across the project to ensure clean,
# maintainable and reproducible experiments.
# -------------------------------------------------------------------------

# Project root directory (folder above /utils)
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Dataset root directory (update this if you place the dataset elsewhere)
DATASET_ROOT = r"C:/Users/admin/Desktop/Datasets/MSID"

# Dataset subsets
TRAIN_DIR = os.path.join(DATASET_ROOT, "train")
VAL_DIR = os.path.join(DATASET_ROOT, "val")
TEST_DIR = os.path.join(DATASET_ROOT, "test")

# Directory where model checkpoints will be saved
CHECKPOINT_DIR = os.path.join(PROJECT_ROOT, "checkpoints")
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

# Directory for evaluation outputs (plots, reports, metrics files)
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)


def show_paths():
    """Utility function for printing all configured directories."""
    print("PROJECT_ROOT:", PROJECT_ROOT)
    print("DATASET_ROOT:", DATASET_ROOT)
    print("TRAIN_DIR:", TRAIN_DIR)
    print("VAL_DIR:", VAL_DIR)
    print("TEST_DIR:", TEST_DIR)
    print("CHECKPOINT_DIR:", CHECKPOINT_DIR)
    print("OUTPUT_DIR:", OUTPUT_DIR)
