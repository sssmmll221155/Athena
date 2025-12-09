"""
ATHENA RL Evaluation - Compare RL vs Baseline

Comprehensive evaluation comparing RL-optimized thresholds vs fixed baseline.
Provides detailed metrics: precision, recall, F1, accuracy, ROC-AUC.
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from typing import Dict, Any, Tuple, List
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_auc_score, roc_curve
)

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from rl.environment import create_environment
from rl.agent import load_latest_agent

sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (14, 10)


class RLEvaluator:
    """
    Evaluates and compares RL agent vs baseline approaches.
    """

    def __init__(self, commits_csv_path: str = "commits_export.csv"):
        """
        Initialize evaluator.

        Args:
            commits_csv_path: Path to commits CSV
        """
        self.commits_csv_path = commits_csv_path
        self.env = None
        self.rl_agent = None

        # Results storage
        self.baseline_results = {}
        self.rl_results = {}

    def setup(self):
        """Setup environment and load RL agent."""
        print("=" * 80)
        print("SETTING UP EVALUATION")
        print("=" * 80)

        # Create environment
        print("\nCreating environment...")
        self.env = create_environment(self.commits_csv_path)

        # Load RL agent
        print("Loading RL agent...")
        self.rl_agent = load_latest_agent(self.env)

        if self.rl_agent is None:
            raise FileNotFoundError("No trained RL agent found. Train first with: python rl/trainer.py")

        print("\nSetup complete!")

    def evaluate_baseline(
        self,
        threshold: float = 0.5,
        n_episodes: int = 50
    ) -> Dict[str, Any]:
        """
        Evaluate baseline with fixed threshold.

        Args:
            threshold: Fixed threshold value
            n_episodes: Number of episodes

        Returns:
            Evaluation metrics
        """
        print(f"\nEvaluating baseline (threshold={threshold})...")

        all_predictions = []
        all_ground_truths = []
        all_probabilities = []
        episode_rewards = []

        for ep in range(n_episodes):
            obs, info = self.env.reset()
            episode_reward = 0
            done = False

            while not done:
                # Fixed threshold action
                action = np.array([threshold, 1.0])  # Medium priority
                obs, reward, terminated, truncated, info = self.env.step(action)
                episode_reward += reward
                done = terminated or truncated

                # Record prediction
                all_predictions.append(info['prediction'])
                all_ground_truths.append(info['ground_truth'])
                all_probabilities.append(obs[0])  # prediction probability is first element

            episode_rewards.append(episode_reward)

        # Calculate metrics
        metrics = self._calculate_metrics(
            all_predictions,
            all_ground_truths,
            all_probabilities,
            episode_rewards
        )

        self.baseline_results[threshold] = metrics
        return metrics

    def evaluate_rl(self, n_episodes: int = 50) -> Dict[str, Any]:
        """
        Evaluate RL agent.

        Args:
            n_episodes: Number of episodes

        Returns:
            Evaluation metrics
        """
        print("\nEvaluating RL agent...")

        all_predictions = []
        all_ground_truths = []
        all_probabilities = []
        all_thresholds = []
        episode_rewards = []

        for ep in range(n_episodes):
            obs, info = self.env.reset()
            episode_reward = 0
            done = False

            while not done:
                # RL policy action
                action, _ = self.rl_agent.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, info = self.env.step(action)
                episode_reward += reward
                done = terminated or truncated

                # Record prediction
                all_predictions.append(info['prediction'])
                all_ground_truths.append(info['ground_truth'])
                all_probabilities.append(obs[0])
                all_thresholds.append(info['threshold'])

            episode_rewards.append(episode_reward)

        # Calculate metrics
        metrics = self._calculate_metrics(
            all_predictions,
            all_ground_truths,
            all_probabilities,
            episode_rewards
        )

        # Add RL-specific metrics
        metrics['mean_threshold'] = np.mean(all_thresholds)
        metrics['std_threshold'] = np.std(all_thresholds)
        metrics['min_threshold'] = np.min(all_thresholds)
        metrics['max_threshold'] = np.max(all_thresholds)

        self.rl_results = metrics
        return metrics

    def _calculate_metrics(
        self,
        predictions: List[int],
        ground_truths: List[int],
        probabilities: List[float],
        rewards: List[float]
    ) -> Dict[str, Any]:
        """
        Calculate comprehensive evaluation metrics.

        Args:
            predictions: Binary predictions
            ground_truths: True labels
            probabilities: Prediction probabilities
            rewards: Episode rewards

        Returns:
            Metrics dictionary
        """
        metrics = {
            # Reward metrics
            'mean_reward': np.mean(rewards),
            'std_reward': np.std(rewards),

            # Classification metrics
            'accuracy': accuracy_score(ground_truths, predictions),
            'precision': precision_score(ground_truths, predictions, zero_division=0),
            'recall': recall_score(ground_truths, predictions, zero_division=0),
            'f1_score': f1_score(ground_truths, predictions, zero_division=0),

            # AUC-ROC
            'roc_auc': roc_auc_score(ground_truths, probabilities) if len(set(ground_truths)) > 1 else 0.5,

            # Confusion matrix elements
            'confusion_matrix': confusion_matrix(ground_truths, predictions).tolist(),

            # Sample counts
            'n_samples': len(predictions),
            'n_positive': sum(ground_truths),
            'n_negative': len(ground_truths) - sum(ground_truths)
        }

        return metrics

    def compare(self, baseline_thresholds: List[float] = [0.3, 0.5, 0.7]) -> pd.DataFrame:
        """
        Compare RL vs multiple baseline thresholds.

        Args:
            baseline_thresholds: List of thresholds to test

        Returns:
            Comparison DataFrame
        """
        print("\n" + "=" * 80)
        print("COMPARING RL vs BASELINE")
        print("=" * 80)

        if self.env is None or self.rl_agent is None:
            self.setup()

        # Evaluate baselines
        for threshold in baseline_thresholds:
            self.evaluate_baseline(threshold=threshold, n_episodes=50)

        # Evaluate RL
        self.evaluate_rl(n_episodes=50)

        # Create comparison table
        rows = []

        for threshold, metrics in self.baseline_results.items():
            rows.append({
                'Method': f'Baseline ({threshold})',
                'Mean Reward': metrics['mean_reward'],
                'Accuracy': metrics['accuracy'],
                'Precision': metrics['precision'],
                'Recall': metrics['recall'],
                'F1 Score': metrics['f1_score'],
                'ROC-AUC': metrics['roc_auc']
            })

        rows.append({
            'Method': 'RL (Adaptive)',
            'Mean Reward': self.rl_results['mean_reward'],
            'Accuracy': self.rl_results['accuracy'],
            'Precision': self.rl_results['precision'],
            'Recall': self.rl_results['recall'],
            'F1 Score': self.rl_results['f1_score'],
            'ROC-AUC': self.rl_results['roc_auc']
        })

        df = pd.DataFrame(rows)

        print("\n" + "=" * 80)
        print("COMPARISON RESULTS")
        print("=" * 80)
        print()
        print(df.to_string(index=False))

        return df

    def plot_comparison(self, save_path: str = "rl/plots/comparison.png"):
        """
        Create comprehensive comparison plots.

        Args:
            save_path: Path to save plot
        """
        print("\nGenerating comparison plots...")

        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('RL vs Baseline Performance Comparison', fontsize=16, fontweight='bold')

        # Prepare data
        methods = []
        rewards = []
        accuracies = []
        precisions = []
        recalls = []
        f1_scores = []
        roc_aucs = []

        for threshold, metrics in sorted(self.baseline_results.items()):
            methods.append(f'Base\n({threshold})')
            rewards.append(metrics['mean_reward'])
            accuracies.append(metrics['accuracy'])
            precisions.append(metrics['precision'])
            recalls.append(metrics['recall'])
            f1_scores.append(metrics['f1_score'])
            roc_aucs.append(metrics['roc_auc'])

        methods.append('RL')
        rewards.append(self.rl_results['mean_reward'])
        accuracies.append(self.rl_results['accuracy'])
        precisions.append(self.rl_results['precision'])
        recalls.append(self.rl_results['recall'])
        f1_scores.append(self.rl_results['f1_score'])
        roc_aucs.append(self.rl_results['roc_auc'])

        colors = ['#3498db'] * (len(methods) - 1) + ['#2ecc71']

        # Plot 1: Mean Reward
        axes[0, 0].bar(methods, rewards, color=colors)
        axes[0, 0].set_ylabel('Mean Reward')
        axes[0, 0].set_title('Episode Rewards')
        axes[0, 0].grid(True, alpha=0.3, axis='y')
        for i, v in enumerate(rewards):
            axes[0, 0].text(i, v + max(rewards) * 0.02, f'{v:.1f}', ha='center', fontsize=9)

        # Plot 2: Accuracy
        axes[0, 1].bar(methods, accuracies, color=colors)
        axes[0, 1].set_ylabel('Accuracy')
        axes[0, 1].set_title('Prediction Accuracy')
        axes[0, 1].set_ylim(0, 1)
        axes[0, 1].grid(True, alpha=0.3, axis='y')
        for i, v in enumerate(accuracies):
            axes[0, 1].text(i, v + 0.02, f'{v:.2%}', ha='center', fontsize=9)

        # Plot 3: Precision
        axes[0, 2].bar(methods, precisions, color=colors)
        axes[0, 2].set_ylabel('Precision')
        axes[0, 2].set_title('Precision (Bug Predictions)')
        axes[0, 2].set_ylim(0, 1)
        axes[0, 2].grid(True, alpha=0.3, axis='y')
        for i, v in enumerate(precisions):
            axes[0, 2].text(i, v + 0.02, f'{v:.2%}', ha='center', fontsize=9)

        # Plot 4: Recall
        axes[1, 0].bar(methods, recalls, color=colors)
        axes[1, 0].set_ylabel('Recall')
        axes[1, 0].set_title('Recall (Bug Detection Rate)')
        axes[1, 0].set_ylim(0, 1)
        axes[1, 0].grid(True, alpha=0.3, axis='y')
        for i, v in enumerate(recalls):
            axes[1, 0].text(i, v + 0.02, f'{v:.2%}', ha='center', fontsize=9)

        # Plot 5: F1 Score
        axes[1, 1].bar(methods, f1_scores, color=colors)
        axes[1, 1].set_ylabel('F1 Score')
        axes[1, 1].set_title('F1 Score')
        axes[1, 1].set_ylim(0, 1)
        axes[1, 1].grid(True, alpha=0.3, axis='y')
        for i, v in enumerate(f1_scores):
            axes[1, 1].text(i, v + 0.02, f'{v:.2%}', ha='center', fontsize=9)

        # Plot 6: ROC-AUC
        axes[1, 2].bar(methods, roc_aucs, color=colors)
        axes[1, 2].set_ylabel('ROC-AUC')
        axes[1, 2].set_title('ROC-AUC Score')
        axes[1, 2].set_ylim(0, 1)
        axes[1, 2].grid(True, alpha=0.3, axis='y')
        for i, v in enumerate(roc_aucs):
            axes[1, 2].text(i, v + 0.02, f'{v:.3f}', ha='center', fontsize=9)

        plt.tight_layout()

        # Save plot
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Comparison plot saved to: {save_path}")

        plt.close()

    def plot_confusion_matrices(self, save_path: str = "rl/plots/confusion_matrices.png"):
        """
        Plot confusion matrices for baseline and RL.

        Args:
            save_path: Path to save plot
        """
        print("Generating confusion matrix plots...")

        n_methods = len(self.baseline_results) + 1
        fig, axes = plt.subplots(1, n_methods, figsize=(5 * n_methods, 4))

        if n_methods == 1:
            axes = [axes]

        idx = 0

        # Baseline confusion matrices
        for threshold, metrics in sorted(self.baseline_results.items()):
            cm = np.array(metrics['confusion_matrix'])
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[idx],
                       xticklabels=['Non-Bug', 'Bug'],
                       yticklabels=['Non-Bug', 'Bug'])
            axes[idx].set_title(f'Baseline (threshold={threshold})')
            axes[idx].set_ylabel('True Label')
            axes[idx].set_xlabel('Predicted Label')
            idx += 1

        # RL confusion matrix
        cm = np.array(self.rl_results['confusion_matrix'])
        sns.heatmap(cm, annot=True, fmt='d', cmap='Greens', ax=axes[idx],
                   xticklabels=['Non-Bug', 'Bug'],
                   yticklabels=['Non-Bug', 'Bug'])
        axes[idx].set_title('RL (Adaptive Thresholds)')
        axes[idx].set_ylabel('True Label')
        axes[idx].set_xlabel('Predicted Label')

        plt.tight_layout()

        # Save plot
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Confusion matrices saved to: {save_path}")

        plt.close()

    def generate_report(self, save_path: str = "rl/evaluation_report.txt"):
        """
        Generate comprehensive evaluation report.

        Args:
            save_path: Path to save report
        """
        print("\nGenerating evaluation report...")

        report = []
        report.append("=" * 80)
        report.append("ATHENA RL EVALUATION REPORT")
        report.append("=" * 80)
        report.append(f"\nGenerated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append(f"\nDataset: {self.commits_csv_path}")
        report.append(f"Samples: {self.rl_results['n_samples']}")
        report.append(f"  - Positive (bugs): {self.rl_results['n_positive']}")
        report.append(f"  - Negative (normal): {self.rl_results['n_negative']}")

        report.append("\n" + "=" * 80)
        report.append("BASELINE RESULTS")
        report.append("=" * 80)

        for threshold, metrics in sorted(self.baseline_results.items()):
            report.append(f"\nThreshold: {threshold}")
            report.append(f"  Mean Reward:     {metrics['mean_reward']:+.2f} ± {metrics['std_reward']:.2f}")
            report.append(f"  Accuracy:        {metrics['accuracy']:.2%}")
            report.append(f"  Precision:       {metrics['precision']:.2%}")
            report.append(f"  Recall:          {metrics['recall']:.2%}")
            report.append(f"  F1 Score:        {metrics['f1_score']:.2%}")
            report.append(f"  ROC-AUC:         {metrics['roc_auc']:.3f}")

        report.append("\n" + "=" * 80)
        report.append("RL AGENT RESULTS")
        report.append("=" * 80)

        report.append(f"\nMean Reward:     {self.rl_results['mean_reward']:+.2f} ± {self.rl_results['std_reward']:.2f}")
        report.append(f"Accuracy:        {self.rl_results['accuracy']:.2%}")
        report.append(f"Precision:       {self.rl_results['precision']:.2%}")
        report.append(f"Recall:          {self.rl_results['recall']:.2%}")
        report.append(f"F1 Score:        {self.rl_results['f1_score']:.2%}")
        report.append(f"ROC-AUC:         {self.rl_results['roc_auc']:.3f}")

        report.append(f"\nThreshold Statistics:")
        report.append(f"  Mean:  {self.rl_results['mean_threshold']:.3f}")
        report.append(f"  Std:   {self.rl_results['std_threshold']:.3f}")
        report.append(f"  Min:   {self.rl_results['min_threshold']:.3f}")
        report.append(f"  Max:   {self.rl_results['max_threshold']:.3f}")

        report.append("\n" + "=" * 80)
        report.append("IMPROVEMENT vs BEST BASELINE")
        report.append("=" * 80)

        # Find best baseline
        best_baseline_threshold = max(self.baseline_results.keys(),
                                      key=lambda t: self.baseline_results[t]['mean_reward'])
        best_baseline = self.baseline_results[best_baseline_threshold]

        improvements = {
            'Reward': ((self.rl_results['mean_reward'] - best_baseline['mean_reward']) / abs(best_baseline['mean_reward'])) * 100,
            'Accuracy': ((self.rl_results['accuracy'] - best_baseline['accuracy']) / best_baseline['accuracy']) * 100,
            'F1 Score': ((self.rl_results['f1_score'] - best_baseline['f1_score']) / max(best_baseline['f1_score'], 0.01)) * 100,
            'ROC-AUC': ((self.rl_results['roc_auc'] - best_baseline['roc_auc']) / best_baseline['roc_auc']) * 100
        }

        report.append(f"\nBest Baseline: Threshold = {best_baseline_threshold}")
        for metric, improvement in improvements.items():
            report.append(f"  {metric:15} {improvement:+.1f}%")

        report_text = "\n".join(report)

        # Save report
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        with open(save_path, 'w') as f:
            f.write(report_text)

        print(f"Evaluation report saved to: {save_path}")
        print("\n" + report_text)


def main():
    """Main evaluation pipeline."""
    print("\n" + "*" * 80)
    print("*" + " " * 78 + "*")
    print("*" + " " * 25 + "ATHENA RL EVALUATION" + " " * 34 + "*")
    print("*" + " " * 78 + "*")
    print("*" * 80)
    print()

    evaluator = RLEvaluator()
    evaluator.setup()

    # Compare methods
    comparison_df = evaluator.compare(baseline_thresholds=[0.3, 0.5, 0.7])

    # Generate plots
    evaluator.plot_comparison()
    evaluator.plot_confusion_matrices()

    # Generate report
    evaluator.generate_report()

    print("\n" + "=" * 80)
    print("EVALUATION COMPLETE!")
    print("=" * 80)
    print("\nResults saved to:")
    print("  - Comparison plot: rl/plots/comparison.png")
    print("  - Confusion matrices: rl/plots/confusion_matrices.png")
    print("  - Report: rl/evaluation_report.txt")
    print()


if __name__ == "__main__":
    main()
