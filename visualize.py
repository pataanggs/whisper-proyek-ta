"""
Visualization script for Whisper fine-tuning training results.
High-precision metrics visualization with individual plots per metric.
"""

import os
import json
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Dict
import glob
from datetime import datetime


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
    return pd.DataFrame(log_history)


def create_run_folder(base_dir: str = "visualizations") -> Path:
    """Create a new folder for this visualization run."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_folder = Path(base_dir) / f"run_{timestamp}"
    run_folder.mkdir(parents=True, exist_ok=True)
    return run_folder


def plot_single_metric(trainer_states: List[Dict], save_dir: Path, 
                       metric_name: str, y_label: str, title: str,
                       is_eval: bool = False, multiply_100: bool = False,
                       multiply_1e5: bool = False):
    """Plot a single metric with high precision - each step visible."""
    fig, ax = plt.subplots(figsize=(16, 8))
    colors = plt.cm.tab10(np.linspace(0, 1, len(trainer_states)))
    
    for i, state in enumerate(trainer_states):
        fold = state.get('fold', i)
        df = extract_training_logs(state)
        if df.empty or metric_name not in df.columns:
            continue
        
        if is_eval:
            plot_df = df[df['eval_loss'].notna()]
        else:
            plot_df = df[df['loss'].notna()]
        
        if plot_df.empty:
            continue
            
        y_values = plot_df[metric_name]
        if multiply_100:
            y_values = y_values * 100
        if multiply_1e5:
            y_values = y_values * 1e5
        
        # Plot with markers for each step
        ax.plot(plot_df['step'], y_values, 
               label=f'Fold {fold}', color=colors[i], 
               marker='o', markersize=4, linewidth=1.5, alpha=0.8)
        
        # Add value annotations for first, min, and last
        if len(y_values) > 0:
            # First value
            ax.annotate(f'{y_values.iloc[0]:.3f}', 
                       (plot_df['step'].iloc[0], y_values.iloc[0]),
                       textcoords="offset points", xytext=(0,10), 
                       fontsize=7, alpha=0.7)
            # Last value
            ax.annotate(f'{y_values.iloc[-1]:.3f}', 
                       (plot_df['step'].iloc[-1], y_values.iloc[-1]),
                       textcoords="offset points", xytext=(0,10), 
                       fontsize=7, alpha=0.7)
    
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xlabel('Step', fontsize=12)
    ax.set_ylabel(y_label, fontsize=12)
    ax.legend(loc='best', fontsize=10)
    ax.grid(True, alpha=0.3, which='both')
    ax.minorticks_on()
    ax.grid(True, alpha=0.15, which='minor')
    
    # Set x-axis to show each step
    ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True, nbins=20))
    
    plt.tight_layout()
    safe_name = metric_name.replace('/', '_')
    save_path = save_dir / f"{safe_name}.png"
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    print(f"✅ Saved: {save_path}")
    plt.close()


def plot_fold_comparison(trainer_states: List[Dict], save_dir: Path):
    """Plot final WER and CER comparison across folds."""
    folds, wer_scores, cer_scores = [], [], []
    
    for state in trainer_states:
        df = extract_training_logs(state)
        if df.empty:
            continue
        eval_df = df[df['eval_wer'].notna()]
        if not eval_df.empty:
            folds.append(f"Fold {state.get('fold', len(folds))}")
            wer_scores.append(eval_df['eval_wer'].iloc[-1] * 100)
            cer_scores.append(eval_df['eval_cer'].iloc[-1] * 100)
    
    if not folds:
        return
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    x = np.arange(len(folds))
    width = 0.6
    
    bars1 = axes[0].bar(x, wer_scores, width, color='steelblue', edgecolor='black')
    axes[0].axhline(y=np.mean(wer_scores), color='red', linestyle='--', linewidth=2,
                    label=f'Mean: {np.mean(wer_scores):.2f}% ± {np.std(wer_scores):.2f}%')
    axes[0].set_ylabel('WER (%)', fontsize=12)
    axes[0].set_title('Word Error Rate by Fold', fontsize=14, fontweight='bold')
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(folds)
    axes[0].legend(fontsize=10)
    axes[0].grid(True, alpha=0.3, axis='y')
    for bar, val in zip(bars1, wer_scores):
        axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3, 
                    f'{val:.2f}%', ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    bars2 = axes[1].bar(x, cer_scores, width, color='coral', edgecolor='black')
    axes[1].axhline(y=np.mean(cer_scores), color='red', linestyle='--', linewidth=2,
                    label=f'Mean: {np.mean(cer_scores):.2f}% ± {np.std(cer_scores):.2f}%')
    axes[1].set_ylabel('CER (%)', fontsize=12)
    axes[1].set_title('Character Error Rate by Fold', fontsize=14, fontweight='bold')
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(folds)
    axes[1].legend(fontsize=10)
    axes[1].grid(True, alpha=0.3, axis='y')
    for bar, val in zip(bars2, cer_scores):
        axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.2,
                    f'{val:.2f}%', ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    plt.suptitle('5-Fold Cross-Validation Results', fontsize=16, fontweight='bold')
    plt.tight_layout()
    save_path = save_dir / "fold_comparison.png"
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    print(f"✅ Saved: {save_path}")
    plt.close()


def save_summary(trainer_states: List[Dict], save_dir: Path) -> Dict:
    """Save and return summary statistics."""
    wer_scores, cer_scores = [], []
    summary_lines = []
    
    summary_lines.append("=" * 60)
    summary_lines.append("TRAINING SUMMARY")
    summary_lines.append("=" * 60)
    
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
            
            summary_lines.append(f"\nFold {fold}:")
            summary_lines.append(f"  Final WER: {final_wer:.2f}% | Best WER: {best_wer:.2f}%")
            summary_lines.append(f"  Final CER: {final_cer:.2f}% | Best CER: {best_cer:.2f}%")
    
    if wer_scores:
        summary_lines.append("\n" + "-" * 60)
        summary_lines.append("OVERALL RESULTS (5-Fold CV)")
        summary_lines.append("-" * 60)
        summary_lines.append(f"WER: {np.mean(wer_scores):.2f}% ± {np.std(wer_scores):.2f}%")
        summary_lines.append(f"CER: {np.mean(cer_scores):.2f}% ± {np.std(cer_scores):.2f}%")
        summary_lines.append("=" * 60)
    
    for line in summary_lines:
        print(line)
    
    summary_path = save_dir / "summary.txt"
    with open(summary_path, 'w') as f:
        f.write("\n".join(summary_lines))
    print(f"\n✅ Saved: {summary_path}")
    
    return {"wer_mean": np.mean(wer_scores), "wer_std": np.std(wer_scores),
            "cer_mean": np.mean(cer_scores), "cer_std": np.std(cer_scores)}


def save_run_config(save_dir: Path, trainer_states: List[Dict], summary: Dict):
    """Save run configuration as JSON."""
    config = {"timestamp": datetime.now().isoformat(), "num_folds": len(trainer_states), "results": summary}
    with open(save_dir / "run_config.json", 'w') as f:
        json.dump(config, f, indent=2)
    print(f"✅ Saved: {save_dir / 'run_config.json'}")


def main():
    """Main visualization function."""
    print("=" * 60)
    print("WHISPER FINE-TUNING VISUALIZATION (HIGH PRECISION)")
    print("=" * 60)
    
    trainer_states = find_all_trainer_states()
    
    if not trainer_states:
        print("❌ No training data found in outputs/checkpoints/")
        return
    
    print(f"Found {len(trainer_states)} fold(s)")
    
    run_folder = create_run_folder()
    print(f"\n📁 Saving visualizations to: {run_folder}")
    
    summary = save_summary(trainer_states, run_folder)
    
    print("\nGenerating individual metric plots...")
    
    # Training metrics
    plot_single_metric(trainer_states, run_folder, 'epoch', 'Epoch', 'train/epoch')
    plot_single_metric(trainer_states, run_folder, 'grad_norm', 'Gradient Norm', 'train/grad_norm')
    plot_single_metric(trainer_states, run_folder, 'learning_rate', 'Learning Rate (×10⁻⁵)', 'train/learning_rate', multiply_1e5=True)
    plot_single_metric(trainer_states, run_folder, 'loss', 'Loss', 'train/loss')
    
    # Evaluation metrics
    plot_single_metric(trainer_states, run_folder, 'eval_cer', 'CER (%)', 'eval/cer', is_eval=True, multiply_100=True)
    plot_single_metric(trainer_states, run_folder, 'eval_loss', 'Loss', 'eval/loss', is_eval=True)
    plot_single_metric(trainer_states, run_folder, 'eval_runtime', 'Runtime (s)', 'eval/runtime', is_eval=True)
    plot_single_metric(trainer_states, run_folder, 'eval_samples_per_second', 'Samples/s', 'eval/samples_per_second', is_eval=True)
    plot_single_metric(trainer_states, run_folder, 'eval_steps_per_second', 'Steps/s', 'eval/steps_per_second', is_eval=True)
    plot_single_metric(trainer_states, run_folder, 'eval_wer', 'WER (%)', 'eval/wer', is_eval=True, multiply_100=True)
    
    # Fold comparison
    plot_fold_comparison(trainer_states, run_folder)
    
    save_run_config(run_folder, trainer_states, summary)
    
    print(f"\n✅ Visualization complete! All files saved to: {run_folder}")


if __name__ == "__main__":
    main()
