"""
Configuration module for Whisper fine-tuning on Minangkabau language.
Contains all paths, hyperparameters, and configuration settings.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# =============================================================================
# PATHS
# =============================================================================
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "Data"
TRAIN_METADATA = DATA_DIR / "metadata_train.csv"
TEST_METADATA = DATA_DIR / "metadata_test.csv"
AUDIO_ROOT = DATA_DIR

# Output directories
OUTPUT_DIR = BASE_DIR / "outputs"
CHECKPOINT_DIR = OUTPUT_DIR / "checkpoints"
PROCESSED_AUDIO_DIR = OUTPUT_DIR / "processed_audio"

# Create directories if they don't exist
OUTPUT_DIR.mkdir(exist_ok=True)
CHECKPOINT_DIR.mkdir(exist_ok=True)
PROCESSED_AUDIO_DIR.mkdir(exist_ok=True)

# =============================================================================
# WANDB CONFIGURATION
# =============================================================================
WANDB_API_KEY = os.getenv("API_KEY")
WANDB_PROJECT = "whisper-minangkabau"
WANDB_GROUP = "whisper-minang-CV-freeze-encoder-v2"

# =============================================================================
# MODEL CONFIGURATION
# =============================================================================
MODEL_NAME = "openai/whisper-base"
# Use Indonesian ("id") as proxy for Minangkabau (not natively supported)
LANGUAGE = "id"  # ISO 639-1 code for Indonesian
LANGUAGE_FULL = "indonesian"  # Full name for processor
DATA_LANGUAGE = "min"  # For filtering CSV (actual language code in dataset)
TASK = "transcribe"

# =============================================================================
# AUDIO CONFIGURATION
# =============================================================================
SAMPLE_RATE = 16000
MIN_DURATION_SECONDS = 0.0
MAX_DURATION_SECONDS = 50.0

# =============================================================================
# TRAINING CONFIGURATION
# =============================================================================
NUM_FOLDS = 5
RANDOM_STATE = 42

# Training arguments (optimized for lower WER)
TRAINING_ARGS = {
    "output_dir": str(CHECKPOINT_DIR),
    "per_device_train_batch_size": 16,
    "per_device_eval_batch_size": 8,
    "gradient_accumulation_steps": 2,
    "learning_rate": 1e-5,
    "warmup_steps": 50,
    "max_steps": 400,  # Increased for cosine scheduler
    "lr_scheduler_type": "cosine",  # Cosine annealing for better convergence
    "gradient_checkpointing": True,
    "bf16": True,
    "dataloader_num_workers": 4,
    "dataloader_pin_memory": True,
    # Regularization
    "weight_decay": 0.01,
    # Evaluation
    "eval_strategy": "steps",
    "eval_steps": 50,
    "save_steps": 50,
    "logging_steps": 10,
    "load_best_model_at_end": True,
    "metric_for_best_model": "wer",
    "greater_is_better": False,
    "save_total_limit": 3,
    # Other
    "report_to": "wandb",
    "push_to_hub": False,
    "predict_with_generate": True,
    "generation_max_length": 225,
    "torch_compile": True,
}

# Model dropout configuration (increased to fight overfitting)
MODEL_DROPOUT_CONFIG = {
    "dropout": 0.2,  # Increased from 0.1
    "attention_dropout": 0.2,  # Increased from 0.1
    "activation_dropout": 0.1,
}

# Generation config for beam search (improves sentence structure)
GENERATION_CONFIG = {
    "num_beams": 5,  # Beam search for better inference
    "max_length": 225,
    "language": LANGUAGE,
    "task": TASK,
}

# =============================================================================
# AUGMENTATION CONFIGURATION
# =============================================================================
AUGMENTATION_CONFIG = {
    "speed_perturbation": [0.85, 0.9, 0.95, 1.0, 1.05, 1.1, 1.15],
    "noise_snr_range": (10, 25),
    "specaugment_time_mask": 100,
    "specaugment_freq_mask": 40,
}

# =============================================================================
# CSV COLUMN NAMES (no headers in original CSV)
# =============================================================================
CSV_COLUMNS = ["audio_path", "language_code", "speaker_id", "transcript"]
