"""
Visualization module for Whisper fine-tuning metrics.
Creates comprehensive plots for training metrics, evaluation results,
learning rate schedules, and cross-validation summaries.
"""

import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd
from datetime import datetime


# Set style
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['figure.figsize'] = (14, 10)
plt.rcParams['font.size'] = 11
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['axes.labelsize'] = 12

# Color palette
COLORS = {
    'train_loss': '#2ecc71',      # Green
    'eval_loss': '#e74c3c',       # Red
    'wer': '#3498db',             # Blue
    'cer': '#9b59b6',             # Purple
    'lr': '#f39c12',              # Orange
    'grad_norm': '#1abc9c',       # Teal
    'folds': ['#3498db', '#e74c3c', '#2ecc71', '#9b59b6', '#f39c12'],  # For multiple folds
}


class MetricsVisualizer:
    """
    Visualize training metrics from the metrics logger output.
    """
    
    def __init__(self, metrics_dir: str, output_dir: str = None):
        """
        Initialize visualizer with metrics directory.
        
        Args:
            metrics_dir: Path to the run directory (e.g., outputs/metrics/run_20260122_160012)
            output_dir: Optional output directory for saving plots
        """
        self.metrics_dir = Path(metrics_dir)
        self.output_dir = Path(output_dir) if output_dir else self.metrics_dir / "visualizations"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Load cross-validation summary if exists
        self.cv_summary = self._load_json(self.metrics_dir / "cross_validation_summary.json")
        
        # Detect folds
        self.fold_dirs = sorted([
            d for d in self.metrics_dir.iterdir() 
            if d.is_dir() and d.name.startswith("fold_")
        ])
        self.num_folds = len(self.fold_dirs)
        
        print(f"📊 MetricsVisualizer initialized")
        print(f"   Metrics directory: {self.metrics_dir}")
        print(f"   Output directory: {self.output_dir}")
        print(f"   Found {self.num_folds} folds")
    
    def _load_json(self, path: Path) -> Optional[Dict]:
        """Load JSON file if exists."""
        if path.exists():
            with open(path, 'r') as f:
                return json.load(f)
        return None
    
    def _load_fold_data(self, fold_idx: int) -> Dict:
        """Load all data for a specific fold."""
        fold_dir = self.metrics_dir / f"fold_{fold_idx}"
        return {
            "training_logs": self._load_json(fold_dir / "training_logs.json") or [],
            "evaluation_logs": self._load_json(fold_dir / "evaluation_logs.json") or [],
            "learning_rate_logs": self._load_json(fold_dir / "learning_rate_logs.json") or [],
            "epoch_summaries": self._load_json(fold_dir / "epoch_summaries.json") or [],
            "sample_predictions": self._load_json(fold_dir / "sample_predictions.json") or [],
            "final_results": self._load_json(fold_dir / "final_results.json") or {},
            "training_config": self._load_json(fold_dir / "training_config.json") or {},
            "model_config": self._load_json(fold_dir / "model_config.json") or {},
        }
    
    def plot_training_loss(self, save: bool = True) -> plt.Figure:
        """Plot training loss over steps for all folds."""
        fig, ax = plt.subplots(figsize=(12, 6))
        
        for fold_idx in range(self.num_folds):
            data = self._load_fold_data(fold_idx)
            training_logs = data["training_logs"]
            
            if training_logs:
                steps = [log["step"] for log in training_logs]
                losses = [log["loss"] for log in training_logs if log["loss"] is not None]
                steps = steps[:len(losses)]
                
                ax.plot(steps, losses, 
                       color=COLORS['folds'][fold_idx % len(COLORS['folds'])],
                       alpha=0.7, linewidth=1.5,
                       label=f'Fold {fold_idx + 1}')
        
        ax.set_xlabel('Training Step')
        ax.set_ylabel('Loss')
        ax.set_title('Training Loss Across All Folds')
        ax.legend(loc='upper right')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save:
            path = self.output_dir / "training_loss.png"
            fig.savefig(path, dpi=150, bbox_inches='tight')
            print(f"✅ Saved: {path}")
        
        return fig
    
    def plot_evaluation_metrics(self, save: bool = True) -> plt.Figure:
        """Plot WER and CER over evaluation steps for all folds."""
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        for fold_idx in range(self.num_folds):
            data = self._load_fold_data(fold_idx)
            eval_logs = data["evaluation_logs"]
            
            if eval_logs:
                steps = [log["step"] for log in eval_logs]
                wers = [log["wer"] * 100 for log in eval_logs]  # Convert to percentage
                cers = [log["cer"] * 100 for log in eval_logs]
                
                color = COLORS['folds'][fold_idx % len(COLORS['folds'])]
                
                axes[0].plot(steps, wers, color=color, alpha=0.7, 
                            linewidth=1.5, marker='o', markersize=3,
                            label=f'Fold {fold_idx + 1}')
                axes[1].plot(steps, cers, color=color, alpha=0.7,
                            linewidth=1.5, marker='o', markersize=3,
                            label=f'Fold {fold_idx + 1}')
        
        axes[0].set_xlabel('Training Step')
        axes[0].set_ylabel('WER (%)')
        axes[0].set_title('Word Error Rate (WER) During Training')
        axes[0].legend(loc='upper right')
        axes[0].grid(True, alpha=0.3)
        
        axes[1].set_xlabel('Training Step')
        axes[1].set_ylabel('CER (%)')
        axes[1].set_title('Character Error Rate (CER) During Training')
        axes[1].legend(loc='upper right')
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save:
            path = self.output_dir / "evaluation_metrics.png"
            fig.savefig(path, dpi=150, bbox_inches='tight')
            print(f"✅ Saved: {path}")
        
        return fig
    
    def plot_learning_rate_schedule(self, save: bool = True) -> plt.Figure:
        """Plot learning rate schedule."""
        fig, ax = plt.subplots(figsize=(12, 5))
        
        # Just use fold 0 since LR schedule is the same
        data = self._load_fold_data(0)
        lr_logs = data["learning_rate_logs"]
        
        if lr_logs:
            steps = [log["step"] for log in lr_logs]
            lrs = [log["learning_rate"] for log in lr_logs]
            
            ax.plot(steps, lrs, color=COLORS['lr'], linewidth=2)
            ax.fill_between(steps, lrs, alpha=0.3, color=COLORS['lr'])
        
        ax.set_xlabel('Training Step')
        ax.set_ylabel('Learning Rate')
        ax.set_title('Learning Rate Schedule (Cosine Annealing with Warmup)')
        ax.grid(True, alpha=0.3)
        ax.ticklabel_format(style='scientific', axis='y', scilimits=(0,0))
        
        plt.tight_layout()
        
        if save:
            path = self.output_dir / "learning_rate_schedule.png"
            fig.savefig(path, dpi=150, bbox_inches='tight')
            print(f"✅ Saved: {path}")
        
        return fig
    
    def plot_gradient_norm(self, save: bool = True) -> plt.Figure:
        """Plot gradient norm over training."""
        fig, ax = plt.subplots(figsize=(12, 5))
        
        for fold_idx in range(self.num_folds):
            data = self._load_fold_data(fold_idx)
            training_logs = data["training_logs"]
            
            if training_logs:
                steps = [log["step"] for log in training_logs if log.get("grad_norm")]
                grad_norms = [log["grad_norm"] for log in training_logs if log.get("grad_norm")]
                
                if steps:
                    ax.plot(steps, grad_norms,
                           color=COLORS['folds'][fold_idx % len(COLORS['folds'])],
                           alpha=0.7, linewidth=1,
                           label=f'Fold {fold_idx + 1}')
        
        ax.set_xlabel('Training Step')
        ax.set_ylabel('Gradient Norm')
        ax.set_title('Gradient Norm During Training')
        ax.legend(loc='upper right')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save:
            path = self.output_dir / "gradient_norm.png"
            fig.savefig(path, dpi=150, bbox_inches='tight')
            print(f"✅ Saved: {path}")
        
        return fig
    
    def plot_loss_comparison(self, save: bool = True) -> plt.Figure:
        """Plot training vs evaluation loss."""
        fig, ax = plt.subplots(figsize=(12, 6))
        
        for fold_idx in range(self.num_folds):
            data = self._load_fold_data(fold_idx)
            training_logs = data["training_logs"]
            eval_logs = data["evaluation_logs"]
            
            color = COLORS['folds'][fold_idx % len(COLORS['folds'])]
            
            if training_logs:
                steps = [log["step"] for log in training_logs]
                losses = [log["loss"] for log in training_logs if log["loss"] is not None]
                ax.plot(steps[:len(losses)], losses, 
                       color=color, alpha=0.3, linewidth=1,
                       label=f'Train Fold {fold_idx + 1}' if fold_idx == 0 else None)
            
            if eval_logs:
                steps = [log["step"] for log in eval_logs]
                eval_losses = [log["eval_loss"] for log in eval_logs if log.get("eval_loss")]
                ax.plot(steps[:len(eval_losses)], eval_losses,
                       color=color, alpha=0.9, linewidth=2, marker='o', markersize=4,
                       label=f'Eval Fold {fold_idx + 1}')
        
        ax.set_xlabel('Training Step')
        ax.set_ylabel('Loss')
        ax.set_title('Training vs Evaluation Loss')
        ax.legend(loc='upper right', ncol=2)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save:
            path = self.output_dir / "loss_comparison.png"
            fig.savefig(path, dpi=150, bbox_inches='tight')
            print(f"✅ Saved: {path}")
        
        return fig
    
    def plot_cross_validation_summary(self, save: bool = True) -> plt.Figure:
        """Plot cross-validation results summary."""
        if not self.cv_summary:
            print("⚠️ No cross-validation summary found")
            return None
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        
        # Get per-fold results
        fold_results = self.cv_summary.get("per_fold_results", [])
        folds = [f"Fold {r['fold'] + 1}" for r in fold_results]
        wers = [r["wer_percent"] for r in fold_results]
        cers = [r["cer_percent"] for r in fold_results]
        
        cv_stats = self.cv_summary.get("cross_validation_summary", {})
        
        # WER Bar Chart
        bars1 = axes[0].bar(folds, wers, color=COLORS['wer'], alpha=0.7, edgecolor='black')
        axes[0].axhline(y=cv_stats.get("mean_wer_percent", 0), color='red', 
                       linestyle='--', linewidth=2, label=f'Mean: {cv_stats.get("mean_wer_percent", 0):.2f}%')
        axes[0].fill_between(
            range(-1, len(folds) + 1),
            cv_stats.get("mean_wer_percent", 0) - cv_stats.get("std_wer_percent", 0),
            cv_stats.get("mean_wer_percent", 0) + cv_stats.get("std_wer_percent", 0),
            alpha=0.2, color='red', label=f'±1 Std: {cv_stats.get("std_wer_percent", 0):.2f}%'
        )
        axes[0].set_ylabel('WER (%)')
        axes[0].set_title('Word Error Rate by Fold')
        axes[0].legend(loc='upper right')
        axes[0].set_xlim(-0.5, len(folds) - 0.5)
        
        # Add value labels on bars
        for bar, val in zip(bars1, wers):
            axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                        f'{val:.1f}%', ha='center', va='bottom', fontsize=10)
        
        # CER Bar Chart
        bars2 = axes[1].bar(folds, cers, color=COLORS['cer'], alpha=0.7, edgecolor='black')
        axes[1].axhline(y=cv_stats.get("mean_cer_percent", 0), color='red',
                       linestyle='--', linewidth=2, label=f'Mean: {cv_stats.get("mean_cer_percent", 0):.2f}%')
        axes[1].fill_between(
            range(-1, len(folds) + 1),
            cv_stats.get("mean_cer_percent", 0) - cv_stats.get("std_cer_percent", 0),
            cv_stats.get("mean_cer_percent", 0) + cv_stats.get("std_cer_percent", 0),
            alpha=0.2, color='red', label=f'±1 Std: {cv_stats.get("std_cer_percent", 0):.2f}%'
        )
        axes[1].set_ylabel('CER (%)')
        axes[1].set_title('Character Error Rate by Fold')
        axes[1].legend(loc='upper right')
        axes[1].set_xlim(-0.5, len(folds) - 0.5)
        
        # Add value labels on bars
        for bar, val in zip(bars2, cers):
            axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                        f'{val:.2f}%', ha='center', va='bottom', fontsize=10)
        
        plt.tight_layout()
        
        if save:
            path = self.output_dir / "cross_validation_summary.png"
            fig.savefig(path, dpi=150, bbox_inches='tight')
            print(f"✅ Saved: {path}")
        
        return fig
    
    def plot_comprehensive_dashboard(self, save: bool = True) -> plt.Figure:
        """Create a comprehensive dashboard with all key metrics."""
        fig = plt.figure(figsize=(18, 14))
        gs = gridspec.GridSpec(3, 3, figure=fig, hspace=0.3, wspace=0.3)
        
        # 1. Training Loss (top-left, spans 2 columns)
        ax1 = fig.add_subplot(gs[0, :2])
        for fold_idx in range(self.num_folds):
            data = self._load_fold_data(fold_idx)
            training_logs = data["training_logs"]
            if training_logs:
                steps = [log["step"] for log in training_logs]
                losses = [log["loss"] for log in training_logs if log["loss"] is not None]
                ax1.plot(steps[:len(losses)], losses,
                        color=COLORS['folds'][fold_idx], alpha=0.7,
                        label=f'Fold {fold_idx + 1}')
        ax1.set_xlabel('Step')
        ax1.set_ylabel('Loss')
        ax1.set_title('Training Loss')
        ax1.legend(loc='upper right', fontsize=8)
        ax1.grid(True, alpha=0.3)
        
        # 2. Learning Rate (top-right)
        ax2 = fig.add_subplot(gs[0, 2])
        data = self._load_fold_data(0)
        lr_logs = data["learning_rate_logs"]
        if lr_logs:
            steps = [log["step"] for log in lr_logs]
            lrs = [log["learning_rate"] for log in lr_logs]
            ax2.plot(steps, lrs, color=COLORS['lr'], linewidth=2)
            ax2.fill_between(steps, lrs, alpha=0.3, color=COLORS['lr'])
        ax2.set_xlabel('Step')
        ax2.set_ylabel('LR')
        ax2.set_title('Learning Rate Schedule')
        ax2.ticklabel_format(style='scientific', axis='y', scilimits=(0,0))
        ax2.grid(True, alpha=0.3)
        
        # 3. WER Progress (middle-left)
        ax3 = fig.add_subplot(gs[1, 0])
        for fold_idx in range(self.num_folds):
            data = self._load_fold_data(fold_idx)
            eval_logs = data["evaluation_logs"]
            if eval_logs:
                steps = [log["step"] for log in eval_logs]
                wers = [log["wer"] * 100 for log in eval_logs]
                ax3.plot(steps, wers, color=COLORS['folds'][fold_idx], 
                        alpha=0.7, marker='o', markersize=2,
                        label=f'Fold {fold_idx + 1}')
        ax3.set_xlabel('Step')
        ax3.set_ylabel('WER (%)')
        ax3.set_title('WER During Training')
        ax3.legend(loc='upper right', fontsize=8)
        ax3.grid(True, alpha=0.3)
        
        # 4. CER Progress (middle-center)
        ax4 = fig.add_subplot(gs[1, 1])
        for fold_idx in range(self.num_folds):
            data = self._load_fold_data(fold_idx)
            eval_logs = data["evaluation_logs"]
            if eval_logs:
                steps = [log["step"] for log in eval_logs]
                cers = [log["cer"] * 100 for log in eval_logs]
                ax4.plot(steps, cers, color=COLORS['folds'][fold_idx],
                        alpha=0.7, marker='o', markersize=2,
                        label=f'Fold {fold_idx + 1}')
        ax4.set_xlabel('Step')
        ax4.set_ylabel('CER (%)')
        ax4.set_title('CER During Training')
        ax4.legend(loc='upper right', fontsize=8)
        ax4.grid(True, alpha=0.3)
        
        # 5. Gradient Norm (middle-right)
        ax5 = fig.add_subplot(gs[1, 2])
        for fold_idx in range(self.num_folds):
            data = self._load_fold_data(fold_idx)
            training_logs = data["training_logs"]
            if training_logs:
                steps = [log["step"] for log in training_logs if log.get("grad_norm")]
                grads = [log["grad_norm"] for log in training_logs if log.get("grad_norm")]
                if steps:
                    ax5.plot(steps, grads, color=COLORS['folds'][fold_idx],
                            alpha=0.5, linewidth=1)
        ax5.set_xlabel('Step')
        ax5.set_ylabel('Grad Norm')
        ax5.set_title('Gradient Norm')
        ax5.grid(True, alpha=0.3)
        
        # 6. Cross-Validation Results (bottom, spans all columns)
        ax6 = fig.add_subplot(gs[2, :])
        if self.cv_summary:
            fold_results = self.cv_summary.get("per_fold_results", [])
            cv_stats = self.cv_summary.get("cross_validation_summary", {})
            
            folds = [f"Fold {r['fold'] + 1}" for r in fold_results]
            wers = [r["wer_percent"] for r in fold_results]
            cers = [r["cer_percent"] for r in fold_results]
            
            x = np.arange(len(folds))
            width = 0.35
            
            bars1 = ax6.bar(x - width/2, wers, width, label='WER (%)', color=COLORS['wer'], alpha=0.7)
            bars2 = ax6.bar(x + width/2, cers, width, label='CER (%)', color=COLORS['cer'], alpha=0.7)
            
            # Add mean lines
            ax6.axhline(y=cv_stats.get("mean_wer_percent", 0), color=COLORS['wer'],
                       linestyle='--', linewidth=2, alpha=0.8)
            ax6.axhline(y=cv_stats.get("mean_cer_percent", 0), color=COLORS['cer'],
                       linestyle='--', linewidth=2, alpha=0.8)
            
            ax6.set_xlabel('Fold')
            ax6.set_ylabel('Error Rate (%)')
            ax6.set_title(f'Cross-Validation Results | Mean WER: {cv_stats.get("mean_wer_percent", 0):.2f}% ± {cv_stats.get("std_wer_percent", 0):.2f}% | Mean CER: {cv_stats.get("mean_cer_percent", 0):.2f}% ± {cv_stats.get("std_cer_percent", 0):.2f}%')
            ax6.set_xticks(x)
            ax6.set_xticklabels(folds)
            ax6.legend(loc='upper right')
            ax6.grid(True, alpha=0.3, axis='y')
            
            # Add value labels
            for bar, val in zip(bars1, wers):
                ax6.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                        f'{val:.1f}', ha='center', fontsize=9)
            for bar, val in zip(bars2, cers):
                ax6.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                        f'{val:.1f}', ha='center', fontsize=9)
        
        # Add title
        fig.suptitle('Whisper Fine-tuning Dashboard - Minangkabau Language', 
                    fontsize=16, fontweight='bold', y=0.98)
        
        if save:
            path = self.output_dir / "comprehensive_dashboard.png"
            fig.savefig(path, dpi=150, bbox_inches='tight')
            print(f"✅ Saved: {path}")
        
        return fig
    
    def plot_single_fold_details(self, fold_idx: int = 0, save: bool = True) -> plt.Figure:
        """Create detailed visualization for a single fold."""
        data = self._load_fold_data(fold_idx)
        
        fig, axes = plt.subplots(2, 3, figsize=(16, 10))
        fig.suptitle(f'Fold {fold_idx + 1} Detailed Analysis', fontsize=14, fontweight='bold')
        
        training_logs = data["training_logs"]
        eval_logs = data["evaluation_logs"]
        lr_logs = data["learning_rate_logs"]
        
        # 1. Training Loss
        if training_logs:
            steps = [log["step"] for log in training_logs]
            losses = [log["loss"] for log in training_logs if log["loss"] is not None]
            axes[0, 0].plot(steps[:len(losses)], losses, color=COLORS['train_loss'], linewidth=1.5)
            axes[0, 0].set_xlabel('Step')
            axes[0, 0].set_ylabel('Loss')
            axes[0, 0].set_title('Training Loss')
            axes[0, 0].grid(True, alpha=0.3)
        
        # 2. Evaluation Loss
        if eval_logs:
            steps = [log["step"] for log in eval_logs]
            eval_losses = [log["eval_loss"] for log in eval_logs]
            axes[0, 1].plot(steps, eval_losses, color=COLORS['eval_loss'], 
                          linewidth=2, marker='o', markersize=4)
            axes[0, 1].set_xlabel('Step')
            axes[0, 1].set_ylabel('Loss')
            axes[0, 1].set_title('Evaluation Loss')
            axes[0, 1].grid(True, alpha=0.3)
        
        # 3. Learning Rate
        if lr_logs:
            steps = [log["step"] for log in lr_logs]
            lrs = [log["learning_rate"] for log in lr_logs]
            axes[0, 2].plot(steps, lrs, color=COLORS['lr'], linewidth=2)
            axes[0, 2].fill_between(steps, lrs, alpha=0.3, color=COLORS['lr'])
            axes[0, 2].set_xlabel('Step')
            axes[0, 2].set_ylabel('Learning Rate')
            axes[0, 2].set_title('Learning Rate')
            axes[0, 2].ticklabel_format(style='scientific', axis='y', scilimits=(0,0))
            axes[0, 2].grid(True, alpha=0.3)
        
        # 4. WER Progress
        if eval_logs:
            steps = [log["step"] for log in eval_logs]
            wers = [log["wer"] * 100 for log in eval_logs]
            axes[1, 0].plot(steps, wers, color=COLORS['wer'], linewidth=2, marker='o', markersize=4)
            axes[1, 0].set_xlabel('Step')
            axes[1, 0].set_ylabel('WER (%)')
            axes[1, 0].set_title(f'WER (Best: {min(wers):.2f}%)')
            axes[1, 0].grid(True, alpha=0.3)
        
        # 5. CER Progress
        if eval_logs:
            steps = [log["step"] for log in eval_logs]
            cers = [log["cer"] * 100 for log in eval_logs]
            axes[1, 1].plot(steps, cers, color=COLORS['cer'], linewidth=2, marker='o', markersize=4)
            axes[1, 1].set_xlabel('Step')
            axes[1, 1].set_ylabel('CER (%)')
            axes[1, 1].set_title(f'CER (Best: {min(cers):.2f}%)')
            axes[1, 1].grid(True, alpha=0.3)
        
        # 6. Gradient Norm
        if training_logs:
            steps = [log["step"] for log in training_logs if log.get("grad_norm")]
            grads = [log["grad_norm"] for log in training_logs if log.get("grad_norm")]
            if steps:
                axes[1, 2].plot(steps, grads, color=COLORS['grad_norm'], linewidth=1, alpha=0.7)
                axes[1, 2].set_xlabel('Step')
                axes[1, 2].set_ylabel('Gradient Norm')
                axes[1, 2].set_title('Gradient Norm')
                axes[1, 2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save:
            path = self.output_dir / f"fold_{fold_idx}_details.png"
            fig.savefig(path, dpi=150, bbox_inches='tight')
            print(f"✅ Saved: {path}")
        
        return fig
    
    def generate_all_plots(self):
        """Generate all available plots."""
        print("\n📊 Generating all visualizations...\n")
        
        self.plot_training_loss()
        self.plot_evaluation_metrics()
        self.plot_learning_rate_schedule()
        self.plot_gradient_norm()
        self.plot_loss_comparison()
        self.plot_cross_validation_summary()
        self.plot_comprehensive_dashboard()
        
        # Generate per-fold details
        for fold_idx in range(self.num_folds):
            self.plot_single_fold_details(fold_idx)
        
        print(f"\n✅ All visualizations saved to: {self.output_dir}")
    
    def print_summary(self):
        """Print a text summary of the results."""
        if not self.cv_summary:
            print("⚠️ No cross-validation summary available")
            return
        
        cv_stats = self.cv_summary.get("cross_validation_summary", {})
        config = self.cv_summary.get("training_config", {})
        
        print("\n" + "="*60)
        print("📊 TRAINING RESULTS SUMMARY")
        print("="*60)
        
        print(f"\n📈 Cross-Validation Results ({self.num_folds} Folds):")
        print("-"*40)
        print(f"  WER: {cv_stats.get('mean_wer_percent', 0):.2f}% ± {cv_stats.get('std_wer_percent', 0):.2f}%")
        print(f"  CER: {cv_stats.get('mean_cer_percent', 0):.2f}% ± {cv_stats.get('std_cer_percent', 0):.2f}%")
        print(f"  Best WER: {cv_stats.get('min_wer', 0)*100:.2f}%")
        print(f"  Best CER: {cv_stats.get('min_cer', 0)*100:.2f}%")
        
        print(f"\n⚙️ Training Configuration:")
        print("-"*40)
        print(f"  Learning Rate: {config.get('learning_rate', 'N/A')}")
        print(f"  Batch Size: {config.get('batch_size', 'N/A')}")
        print(f"  Max Steps: {config.get('max_steps', 'N/A')}")
        print(f"  Warmup Ratio: {config.get('warmup_ratio', 'N/A')}")
        print(f"  Weight Decay: {config.get('weight_decay', 'N/A')}")
        print(f"  LR Scheduler: {config.get('lr_scheduler_type', 'N/A')}")
        
        print("\n" + "="*60)


def visualize_run(run_dir: str, output_dir: str = None):
    """
    Convenience function to visualize a training run.
    
    Args:
        run_dir: Path to the run directory
        output_dir: Optional custom output directory
    """
    visualizer = MetricsVisualizer(run_dir, output_dir)
    visualizer.print_summary()
    visualizer.generate_all_plots()
    return visualizer


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Visualize Whisper fine-tuning metrics")
    parser.add_argument(
        "--run-dir", "-r",
        type=str,
        default="outputs/metrics/run_20260122_160012",
        help="Path to the run directory containing metrics"
    )
    parser.add_argument(
        "--output-dir", "-o",
        type=str,
        default=None,
        help="Optional output directory for plots"
    )
    
    args = parser.parse_args()
    
    visualize_run(args.run_dir, args.output_dir)
