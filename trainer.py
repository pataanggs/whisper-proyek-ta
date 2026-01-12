"""
Trainer module for Whisper fine-tuning.
Implements model initialization, freeze encoder strategy, and training loop.
"""

import torch
import evaluate
from typing import Dict, Any, Optional
from transformers import (
    WhisperForConditionalGeneration,
    WhisperProcessor,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
)

from config import MODEL_NAME, LANGUAGE, TASK, TRAINING_ARGS, MODEL_DROPOUT_CONFIG


def load_model() -> WhisperForConditionalGeneration:
    """
    Load and initialize Whisper model with dropout for regularization.
    
    Returns:
        WhisperForConditionalGeneration model
    """
    model = WhisperForConditionalGeneration.from_pretrained(MODEL_NAME)
    
    # Configure model for transcription
    model.generation_config.language = LANGUAGE
    model.generation_config.task = TASK
    model.generation_config.forced_decoder_ids = None
    
    # Apply dropout configuration to prevent overfitting
    model.config.dropout = MODEL_DROPOUT_CONFIG["dropout"]
    model.config.attention_dropout = MODEL_DROPOUT_CONFIG["attention_dropout"]
    model.config.activation_dropout = MODEL_DROPOUT_CONFIG["activation_dropout"]
    
    return model


def load_processor() -> WhisperProcessor:
    """
    Load Whisper processor (tokenizer + feature extractor).
    
    Returns:
        WhisperProcessor
    """
    processor = WhisperProcessor.from_pretrained(MODEL_NAME, language=LANGUAGE, task=TASK)
    return processor


def freeze_encoder(model: WhisperForConditionalGeneration) -> WhisperForConditionalGeneration:
    """
    Freeze the encoder weights; only train the decoder.
    
    Args:
        model: Whisper model
        
    Returns:
        Model with frozen encoder
    """
    # Freeze encoder
    model.model.encoder.requires_grad_(False)
    
    # Ensure decoder is trainable
    model.model.decoder.requires_grad_(True)
    
    # Count trainable parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Frozen parameters: {total_params - trainable_params:,}")
    print(f"Trainable %: {100 * trainable_params / total_params:.2f}%")
    
    return model


def create_compute_metrics(processor: WhisperProcessor):
    """
    Create metrics computation function for WER and CER.
    
    Args:
        processor: WhisperProcessor for decoding
        
    Returns:
        Function to compute metrics
    """
    wer_metric = evaluate.load("wer")
    cer_metric = evaluate.load("cer")
    
    def compute_metrics(pred) -> Dict[str, float]:
        """
        Compute WER and CER for predictions.
        
        Args:
            pred: Prediction output from Trainer
            
        Returns:
            Dictionary with 'wer' and 'cer' metrics
        """
        pred_ids = pred.predictions
        label_ids = pred.label_ids
        
        # Replace -100 with pad token id
        label_ids[label_ids == -100] = processor.tokenizer.pad_token_id
        
        # Decode predictions and labels
        pred_str = processor.tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
        label_str = processor.tokenizer.batch_decode(label_ids, skip_special_tokens=True)
        
        # Compute metrics
        wer = wer_metric.compute(predictions=pred_str, references=label_str)
        cer = cer_metric.compute(predictions=pred_str, references=label_str)
        
        return {"wer": wer, "cer": cer}
    
    return compute_metrics


def create_training_arguments(
    output_dir: str,
    run_name: str,
    **kwargs
) -> Seq2SeqTrainingArguments:
    """
    Create training arguments for Seq2SeqTrainer.
    
    Args:
        output_dir: Directory to save checkpoints
        run_name: Name for this training run
        **kwargs: Override default training arguments
        
    Returns:
        Seq2SeqTrainingArguments
    """
    # Merge default args with overrides
    args = TRAINING_ARGS.copy()
    args.update(kwargs)
    args["output_dir"] = output_dir
    args["run_name"] = run_name
    
    return Seq2SeqTrainingArguments(**args)


def create_trainer(
    model: WhisperForConditionalGeneration,
    processor: WhisperProcessor,
    training_args: Seq2SeqTrainingArguments,
    train_dataset,
    eval_dataset,
    data_collator,
) -> Seq2SeqTrainer:
    """
    Create Seq2SeqTrainer for Whisper fine-tuning.
    
    Args:
        model: Whisper model
        processor: WhisperProcessor
        training_args: Training arguments
        train_dataset: Training dataset
        eval_dataset: Evaluation dataset
        data_collator: Data collator
        
    Returns:
        Seq2SeqTrainer instance
    """
    compute_metrics = create_compute_metrics(processor)
    
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        processing_class=processor,
    )
    
    return trainer


def train_fold(
    fold_idx: int,
    train_dataset,
    eval_dataset,
    processor: WhisperProcessor,
    data_collator,
    output_dir: str,
) -> Dict[str, float]:
    """
    Train a single fold.
    
    Args:
        fold_idx: Fold index (0-4)
        train_dataset: Training dataset for this fold
        eval_dataset: Evaluation dataset for this fold
        processor: WhisperProcessor
        data_collator: Data collator
        output_dir: Base output directory
        
    Returns:
        Dictionary with evaluation metrics
    """
    print(f"\n{'='*60}")
    print(f"TRAINING FOLD {fold_idx + 1}/5")
    print(f"{'='*60}")
    
    # CRITICAL: Re-initialize model for each fold to prevent weight leakage
    print("Loading fresh model...")
    model = load_model()
    model = freeze_encoder(model)
    
    # Create training arguments
    fold_output_dir = f"{output_dir}/fold_{fold_idx}"
    training_args = create_training_arguments(
        output_dir=fold_output_dir,
        run_name=f"fold-{fold_idx}",
    )
    
    # Create trainer
    trainer = create_trainer(
        model=model,
        processor=processor,
        training_args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
    )
    
    # Train
    print(f"\nStarting training for fold {fold_idx + 1}...")
    trainer.train()
    
    # Evaluate
    print(f"\nEvaluating fold {fold_idx + 1}...")
    eval_results = trainer.evaluate()
    
    print(f"\nFold {fold_idx + 1} Results:")
    print(f"  WER: {eval_results['eval_wer']:.4f}")
    print(f"  CER: {eval_results['eval_cer']:.4f}")
    
    return {
        "wer": eval_results["eval_wer"],
        "cer": eval_results["eval_cer"],
    }


if __name__ == "__main__":
    print("Testing Trainer Module")
    print("-" * 50)
    
    # Load model and processor
    processor = load_processor()
    model = load_model()
    
    print(f"\nModel: {MODEL_NAME}")
    print(f"Language: {LANGUAGE}")
    print(f"Task: {TASK}")
    
    # Test freeze encoder
    print("\nFreezing encoder...")
    model = freeze_encoder(model)
