"""
Main entry point for Whisper fine-tuning on Minangkabau language.
Implements 5-fold cross-validation with WandB integration.
"""

import os
import warnings
import logging
import numpy as np
import torch
import wandb
from sklearn.model_selection import KFold

# Suppress all warnings BEFORE importing transformers
warnings.filterwarnings("ignore")
os.environ["TOKENIZERS_PARALLELISM"] = "false"
logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("torch").setLevel(logging.ERROR)

from config import (
    WANDB_API_KEY,
    WANDB_PROJECT,
    WANDB_GROUP,
    NUM_FOLDS,
    RANDOM_STATE,
    CHECKPOINT_DIR,
)
from data_loader import (
    load_and_merge_datasets,
    dataframe_to_hf_dataset,
    get_train_eval_split,
)
from dataset import (
    prepare_dataset_for_training,
    prepare_dataset_for_evaluation,
    create_data_collator,
)
from augmentation import AudioAugmenter
from trainer import (
    load_processor,
    load_model,
    train_fold,
)


def check_gpu():
    """Check and display GPU information."""
    print("\n[GPU CHECK]")
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"✅ GPU Available: {gpu_name}")
        print(f"   Memory: {gpu_memory:.1f} GB")
        return True
    else:
        print("❌ No GPU available - training will use CPU (very slow)")
        return False


def setup_wandb():
    """Initialize WandB with API key from .env"""
    if WANDB_API_KEY:
        os.environ["WANDB_API_KEY"] = WANDB_API_KEY
        print("WandB API key loaded from .env")
    else:
        print("Warning: WANDB_API_KEY not found in .env")


def main():
    """Main training pipeline with 5-fold cross-validation."""
    
    print("=" * 70)
    print("WHISPER FINE-TUNING FOR MINANGKABAU LANGUAGE")
    print("=" * 70)
    
    # Check GPU
    check_gpu()
    
    # Setup WandB
    setup_wandb()
    
    # ==========================================================================
    # STEP 1: Load data (uses pre-converted WAV metadata if available)
    # ==========================================================================
    print("\n[STEP 1] Loading data...")
    
    # Load datasets (data_loader will use pre-converted metadata if it exists)
    merged_df = load_and_merge_datasets()
    
    print(f"Final dataset size: {len(merged_df)} samples")
    
    # Convert to HuggingFace Dataset
    dataset = dataframe_to_hf_dataset(merged_df)
    
    # ==========================================================================
    # STEP 2: Setup processor and augmenter
    # ==========================================================================
    print("\n[STEP 2] Setting up processor and augmenter...")
    
    processor = load_processor()
    augmenter = AudioAugmenter()
    
    # Create a dummy model for data collator
    dummy_model = load_model()
    data_collator = create_data_collator(processor, dummy_model)
    del dummy_model  # Free memory
    
    # ==========================================================================
    # STEP 3: 5-Fold Cross-Validation
    # ==========================================================================
    print("\n[STEP 3] Starting 5-Fold Cross-Validation...")
    
    kfold = KFold(n_splits=NUM_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    
    fold_results = []
    
    for fold_idx, (train_indices, eval_indices) in enumerate(kfold.split(range(len(dataset)))):
        
        # Initialize WandB for this fold
        wandb.init(
            project=WANDB_PROJECT,
            group=WANDB_GROUP,
            name=f"fold-{fold_idx}",
            reinit=True,
        )
        
        print(f"\n--- Fold {fold_idx + 1}/{NUM_FOLDS} ---")
        print(f"Train samples: {len(train_indices)}")
        print(f"Eval samples: {len(eval_indices)}")
        
        # Split dataset
        train_dataset, eval_dataset = get_train_eval_split(
            dataset, 
            train_indices.tolist(), 
            eval_indices.tolist()
        )
        
        # Prepare datasets
        print("Preparing training dataset with augmentation...")
        train_dataset_prepared = prepare_dataset_for_training(
            train_dataset, processor, augmenter
        )
        
        print("Preparing evaluation dataset...")
        eval_dataset_prepared = prepare_dataset_for_evaluation(
            eval_dataset, processor
        )
        
        # Train fold
        fold_metrics = train_fold(
            fold_idx=fold_idx,
            train_dataset=train_dataset_prepared,
            eval_dataset=eval_dataset_prepared,
            processor=processor,
            data_collator=data_collator,
            output_dir=str(CHECKPOINT_DIR),
        )
        
        fold_results.append(fold_metrics)
        
        # Log to WandB
        wandb.log({
            "fold": fold_idx,
            "fold_wer": fold_metrics["wer"],
            "fold_cer": fold_metrics["cer"],
        })
        
        # Finish WandB run for this fold
        wandb.finish()
    
    # ==========================================================================
    # STEP 4: Calculate and print final statistics
    # ==========================================================================
    print("\n" + "=" * 70)
    print("FINAL RESULTS - 5-FOLD CROSS-VALIDATION")
    print("=" * 70)
    
    wer_scores = [r["wer"] for r in fold_results]
    cer_scores = [r["cer"] for r in fold_results]
    
    print("\nPer-Fold Results:")
    print("-" * 40)
    for i, result in enumerate(fold_results):
        print(f"  Fold {i + 1}: WER = {result['wer']:.4f}, CER = {result['cer']:.4f}")
    
    print("\n" + "-" * 40)
    print(f"WER: {np.mean(wer_scores):.4f} ± {np.std(wer_scores):.4f}")
    print(f"CER: {np.mean(cer_scores):.4f} ± {np.std(cer_scores):.4f}")
    print("-" * 40)
    
    # Log summary to WandB
    wandb.init(
        project=WANDB_PROJECT,
        group=WANDB_GROUP,
        name="summary",
        reinit=True,
    )
    wandb.log({
        "mean_wer": np.mean(wer_scores),
        "std_wer": np.std(wer_scores),
        "mean_cer": np.mean(cer_scores),
        "std_cer": np.std(cer_scores),
    })
    wandb.finish()
    
    print("\n✅ Training complete!")
    print(f"Checkpoints saved to: {CHECKPOINT_DIR}")
    
    return fold_results


if __name__ == "__main__":
    results = main()
