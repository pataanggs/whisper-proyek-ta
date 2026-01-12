"""
Visualization script for Whisper fine-tuning training results.
Shows detailed training curves, WER/CER per epoch, and fold comparison.
"""

import os
import json
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Dict, Optional
import glob


def load_wandb_history(run_dir: str) -> pd.DataFrame:
    """Load training history from WandB run directory."""
    history_file = Path(run_dir) / "files" / "wandb-summary.json"
    if history_file.exists():
        with open(history_file, 'r') as f:
            return json.load(f)
    return {}


def load_trainer_state(checkpoint_dir: str) -> Dict:
    """Load trainer state from checkpoint."""
    state_file = Path(checkpoint_dir) / "trainer_state.json"
    if state_file.exists():
        with open(state_file, 'r') as f:
            return json.load(f)
    return {}


def find_all_trainer_states(base_dir: str = "outputs/checkpoints") -> List[Dict]:
    """Find all trainer states from all folds."""
    states = []
    for fold_dir in sorted(glob.glob(f"{base_dir}/fold_*")):
        # Find the latest checkpoint in each fold
        checkpoints = sorted(glob.glob(f"{fold_dir}/checkpoint-*"))
        if checkpoints:
            state = load_trainer_state(checkpoints[-1])
            if state:
                state['fold'] = int(Path(fold_dir).name.split('_')[1])
                states.append(state)
    return states


def extract_training_logs(trainer_state: Dict) -> pd.DataFrame:
    """Extract training logs from trainer state."""
    log_history = trainer_state.get('log_history', [])
    if not log_history:
        return pd.DataFrame()
    
    df = pd.DataFrame(log_history)
    return df


def plot_training_curves(trainer_states: List[Dict], save_path: str = "training_curves.png"):
    """Plot training curves for all folds."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    colors = plt.cm.tab10(np.linspace(0, 1, 5))
    
    for i, state in enumerate(trainer_states):
        fold = state.get('fold', i)
        df = extract_training_logs(state)
        
        if df.empty:
            continue
        
        # Training Loss
        if 'loss' in df.columns:
            train_df = df[df['loss'].notna()]
            axes[0, 0].plot(train_df['step'], train_df['loss'], 
                          label=f'Fold {fold}', color=colors[fold], alpha=0.7)
        
        # Eval Loss
        if 'eval_loss' in df.columns:
            eval_df = df[df['eval_loss'].notna()]
            axes[0, 1].plot(eval_df['step'], eval_df['eval_loss'], 
                          label=f'Fold {fold}', color=colors[fold], marker='o', markersize=3)
        
        # Eval WER
        if 'eval_wer' in df.columns:
            eval_df = df[df['eval_wer'].notna()]
            axes[1, 0].plot(eval_df['step'], eval_df['eval_wer'] * 100, 
                          label=f'Fold {fold}', color=colors[fold], marker='o', markersize=3)
        
        # Eval CER
        if 'eval_cer' in df.columns:
            eval_df = df[df['eval_cer'].notna()]
            axes[1, 1].plot(eval_df['step'], eval_df['eval_cer'] * 100, 
                          label=f'Fold {fold}', color=colors[fold], marker='o', markersize=3)
    
    # Configure axes
    axes[0, 0].set_title('Training Loss', fontsize=12, fontweight='bold')
    axes[0, 0].set_xlabel('Step')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    axes[0, 1].set_title('Evaluation Loss', fontsize=12, fontweight='bold')
    axes[0, 1].set_xlabel('Step')
    axes[0, 1].set_ylabel('Loss')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    axes[1, 0].set_title('Word Error Rate (WER)', fontsize=12, fontweight='bold')
    axes[1, 0].set_xlabel('Step')
    axes[1, 0].set_ylabel('WER (%)')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    axes[1, 1].set_title('Character Error Rate (CER)', fontsize=12, fontweight='bold')
    axes[1, 1].set_xlabel('Step')
    axes[1, 1].set_ylabel('CER (%)')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.suptitle('Whisper Fine-tuning Training Curves (5-Fold CV)', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"✅ Saved training curves to: {save_path}")
    plt.show()


def plot_fold_comparison(trainer_states: List[Dict], save_path: str = "fold_comparison.png"):
    """Plot final metrics comparison across folds."""
    folds = []
    wer_scores = []
    cer_scores = []
    
    for state in trainer_states:
        df = extract_training_logs(state)
        if df.empty:
            continue
        
        # Get final eval metrics
        eval_df = df[df['eval_wer'].notna()]
        if not eval_df.empty:
            folds.append(f"Fold {state.get('fold', len(folds))}")
            wer_scores.append(eval_df['eval_wer'].iloc[-1] * 100)
            cer_scores.append(eval_df['eval_cer'].iloc[-1] * 100)
    
    if not folds:
        print("No evaluation data found")
        return
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    x = np.arange(len(folds))
    width = 0.6
    
    # WER comparison
    bars1 = axes[0].bar(x, wer_scores, width, color='steelblue', edgecolor='black')
    axes[0].axhline(y=np.mean(wer_scores), color='red', linestyle='--', 
                    label=f'Mean: {np.mean(wer_scores):.2f}% ± {np.std(wer_scores):.2f}%')
    axes[0].set_ylabel('WER (%)')
    axes[0].set_title('Word Error Rate by Fold', fontweight='bold')
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(folds)
    axes[0].legend()
    axes[0].grid(True, alpha=0.3, axis='y')
    
    # Add value labels
    for bar, val in zip(bars1, wer_scores):
        axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5, 
                    f'{val:.1f}%', ha='center', va='bottom', fontsize=10)
    
    # CER comparison
    bars2 = axes[1].bar(x, cer_scores, width, color='coral', edgecolor='black')
    axes[1].axhline(y=np.mean(cer_scores), color='red', linestyle='--',
                    label=f'Mean: {np.mean(cer_scores):.2f}% ± {np.std(cer_scores):.2f}%')
    axes[1].set_ylabel('CER (%)')
    axes[1].set_title('Character Error Rate by Fold', fontweight='bold')
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(folds)
    axes[1].legend()
    axes[1].grid(True, alpha=0.3, axis='y')
    
    # Add value labels
    for bar, val in zip(bars2, cer_scores):
        axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                    f'{val:.1f}%', ha='center', va='bottom', fontsize=10)
    
    plt.suptitle('5-Fold Cross-Validation Results', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"✅ Saved fold comparison to: {save_path}")
    plt.show()


def plot_learning_rate(trainer_states: List[Dict], save_path: str = "learning_rate.png"):
    """Plot learning rate schedule."""
    fig, ax = plt.subplots(figsize=(10, 5))
    
    for i, state in enumerate(trainer_states):
        df = extract_training_logs(state)
        if df.empty or 'learning_rate' not in df.columns:
            continue
        
        lr_df = df[df['learning_rate'].notna()]
        if i == 0:  # Only plot one fold since LR schedule is same
            ax.plot(lr_df['step'], lr_df['learning_rate'] * 1e5, 
                   color='purple', linewidth=2)
            break
    
    ax.set_xlabel('Step')
    ax.set_ylabel('Learning Rate (×10⁻⁵)')
    ax.set_title('Cosine Learning Rate Schedule', fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"✅ Saved learning rate plot to: {save_path}")
    plt.show()


def print_summary(trainer_states: List[Dict]):
    """Print summary statistics."""
    wer_scores = []
    cer_scores = []
    
    print("\n" + "="*60)
    print("TRAINING SUMMARY")
    print("="*60)
    
    for state in trainer_states:
        df = extract_training_logs(state)
        if df.empty:
            continue
        
        fold = state.get('fold', len(wer_scores))
        eval_df = df[df['eval_wer'].notna()]
        
        if not eval_df.empty:
            final_wer = eval_df['eval_wer'].iloc[-1] * 100
            final_cer = eval_df['eval_cer'].iloc[-1] * 100
            best_wer = eval_df['eval_wer'].min() * 100
            best_cer = eval_df['eval_cer'].min() * 100
            
            wer_scores.append(final_wer)
            cer_scores.append(final_cer)
            
            print(f"\nFold {fold}:")
            print(f"  Final WER: {final_wer:.2f}% | Best WER: {best_wer:.2f}%")
            print(f"  Final CER: {final_cer:.2f}% | Best CER: {best_cer:.2f}%")
    
    if wer_scores:
        print("\n" + "-"*60)
        print("OVERALL RESULTS (5-Fold CV)")
        print("-"*60)
        print(f"WER: {np.mean(wer_scores):.2f}% ± {np.std(wer_scores):.2f}%")
        print(f"CER: {np.mean(cer_scores):.2f}% ± {np.std(cer_scores):.2f}%")
        print("="*60)


def main():
    """Main visualization function."""
    print("="*60)
    print("WHISPER FINE-TUNING VISUALIZATION")
    print("="*60)
    
    # Find all trainer states
    trainer_states = find_all_trainer_states()
    
    if not trainer_states:
        print("❌ No training data found in outputs/checkpoints/")
        print("   Run training first with: python main.py")
        return
    
    print(f"Found {len(trainer_states)} fold(s)")
    
    # Print summary
    print_summary(trainer_states)
    
    # Create visualizations
    print("\nGenerating visualizations...")
    plot_training_curves(trainer_states)
    plot_fold_comparison(trainer_states)
    plot_learning_rate(trainer_states)
    
    print("\n✅ Visualization complete!")


if __name__ == "__main__":
    main()
