"""
ATHENA RL Trainer - Training Loop for PPO Agent

Trains PPO agent using simulated feedback from ground truth labels.
Logs metrics, saves checkpoints, and generates learning curves.
"""
import os
import sys
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from rl.environment import create_environment
from rl.agent import create_agent, BugPredictionAgent
from rl.feedback_collector import FeedbackCollector

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)


class RLTrainer:
    """
    Trainer for ATHENA RL bug prediction optimization.

    Manages the complete training pipeline:
    - Environment setup
    - Agent training
    - Metrics logging
    - Visualization
    """

    def __init__(
        self,
        commits_csv_path: str = "commits_export.csv",
        save_dir: str = "rl/models",
        logs_dir: str = "rl/logs",
        plots_dir: str = "rl/plots"
    ):
        """
        Initialize RL trainer.

        Args:
            commits_csv_path: Path to commits CSV
            save_dir: Directory for saving models
            logs_dir: Directory for TensorBoard logs
            plots_dir: Directory for plots
        """
        self.commits_csv_path = commits_csv_path
        self.save_dir = Path(save_dir)
        self.logs_dir = Path(logs_dir)
        self.plots_dir = Path(plots_dir)

        # Create directories
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        self.plots_dir.mkdir(parents=True, exist_ok=True)

        # Initialize components
        self.env = None
        self.agent = None
        self.feedback_collector = FeedbackCollector()

        # Training history
        self.training_history = {
            'timesteps': [],
            'mean_reward': [],
            'mean_accuracy': [],
            'policy_loss': [],
            'value_loss': []
        }

    def setup(self):
        """Setup environment and agent."""
        print("=" * 80)
        print("SETTING UP RL TRAINING")
        print("=" * 80)

        # Create environment
        print("\nStep 1: Creating environment...")
        self.env = create_environment(self.commits_csv_path)

        # Create agent
        print("\nStep 2: Creating PPO agent...")
        self.agent = create_agent(self.env)

        # Setup feedback table (optional - skip if no DB access)
        print("\nStep 3: Setting up feedback collection...")
        try:
            self.feedback_collector.create_feedback_table()
            print("Feedback table ready")
        except Exception as e:
            print(f"Warning: Could not setup feedback table: {e}")
            print("Continuing without feedback storage (using simulation only)")

        print("\n" + "=" * 80)
        print("SETUP COMPLETE!")
        print("=" * 80)

    def train(
        self,
        total_timesteps: int = 10000,
        checkpoint_freq: int = 1000,
        eval_freq: int = 2000,
        n_eval_episodes: int = 10
    ) -> Dict[str, Any]:
        """
        Train RL agent with periodic evaluation.

        Args:
            total_timesteps: Total training timesteps
            checkpoint_freq: Checkpoint save frequency
            eval_freq: Evaluation frequency
            n_eval_episodes: Episodes per evaluation

        Returns:
            Training results dictionary
        """
        if self.env is None or self.agent is None:
            self.setup()

        print("\n" + "=" * 80)
        print("STARTING TRAINING")
        print("=" * 80)
        print(f"\nConfiguration:")
        print(f"  - Total timesteps: {total_timesteps:,}")
        print(f"  - Checkpoint frequency: {checkpoint_freq:,}")
        print(f"  - Evaluation frequency: {eval_freq:,}")
        print(f"  - Episodes per eval: {n_eval_episodes}")

        # Train agent
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        save_path = self.save_dir / f"ppo_bug_predictor_{timestamp}"

        training_stats = self.agent.train(
            total_timesteps=total_timesteps,
            checkpoint_freq=checkpoint_freq,
            save_path=str(save_path),
            tb_log_name=f"ppo_bug_predictor_{timestamp}"
        )

        # Final evaluation
        print("\n" + "=" * 80)
        print("FINAL EVALUATION")
        print("=" * 80)

        eval_metrics = self.agent.evaluate(n_episodes=n_eval_episodes)

        # Combine results
        results = {
            'timestamp': timestamp,
            'total_timesteps': total_timesteps,
            'training_stats': training_stats,
            'eval_metrics': eval_metrics,
            'model_path': str(save_path) + "_final.zip"
        }

        # Save results
        results_path = self.save_dir / f"training_results_{timestamp}.json"
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)

        print(f"\nResults saved to: {results_path}")

        return results

    def generate_learning_curves(
        self,
        tensorboard_log_dir: Optional[str] = None,
        window_size: int = 100
    ):
        """
        Generate learning curve plots from TensorBoard logs.

        Args:
            tensorboard_log_dir: Path to TensorBoard logs (auto-detect if None)
            window_size: Moving average window size
        """
        print("\n" + "=" * 80)
        print("GENERATING LEARNING CURVES")
        print("=" * 80)

        if tensorboard_log_dir is None:
            # Find most recent run
            log_dirs = sorted(self.logs_dir.glob("ppo_bug_predictor_*"))
            if not log_dirs:
                print("No TensorBoard logs found!")
                return
            tensorboard_log_dir = log_dirs[-1]

        print(f"\nReading logs from: {tensorboard_log_dir}")

        try:
            from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

            # Load TensorBoard events
            event_acc = EventAccumulator(str(tensorboard_log_dir))
            event_acc.Reload()

            # Extract scalars
            scalar_tags = event_acc.Tags()['scalars']
            print(f"Found {len(scalar_tags)} scalar metrics")

            # Create figure with subplots
            fig, axes = plt.subplots(2, 2, figsize=(15, 10))
            fig.suptitle('RL Training Progress - PPO Bug Predictor', fontsize=16, fontweight='bold')

            # Plot 1: Episode Reward
            if 'rollout/ep_rew_mean' in scalar_tags:
                rewards = event_acc.Scalars('rollout/ep_rew_mean')
                steps = [r.step for r in rewards]
                values = [r.value for r in rewards]

                axes[0, 0].plot(steps, values, alpha=0.6, label='Raw')
                if len(values) > window_size:
                    smoothed = pd.Series(values).rolling(window_size).mean()
                    axes[0, 0].plot(steps, smoothed, linewidth=2, label=f'{window_size}-step MA')
                axes[0, 0].set_xlabel('Timesteps')
                axes[0, 0].set_ylabel('Mean Episode Reward')
                axes[0, 0].set_title('Episode Reward Over Time')
                axes[0, 0].legend()
                axes[0, 0].grid(True, alpha=0.3)

            # Plot 2: Episode Length
            if 'rollout/ep_len_mean' in scalar_tags:
                ep_lens = event_acc.Scalars('rollout/ep_len_mean')
                steps = [r.step for r in ep_lens]
                values = [r.value for r in ep_lens]

                axes[0, 1].plot(steps, values, color='green', alpha=0.6)
                axes[0, 1].set_xlabel('Timesteps')
                axes[0, 1].set_ylabel('Mean Episode Length')
                axes[0, 1].set_title('Episode Length Over Time')
                axes[0, 1].grid(True, alpha=0.3)

            # Plot 3: Policy Loss
            if 'train/policy_gradient_loss' in scalar_tags:
                losses = event_acc.Scalars('train/policy_gradient_loss')
                steps = [r.step for r in losses]
                values = [r.value for r in losses]

                axes[1, 0].plot(steps, values, color='red', alpha=0.6, label='Raw')
                if len(values) > window_size:
                    smoothed = pd.Series(values).rolling(window_size).mean()
                    axes[1, 0].plot(steps, smoothed, color='darkred', linewidth=2, label=f'{window_size}-step MA')
                axes[1, 0].set_xlabel('Timesteps')
                axes[1, 0].set_ylabel('Policy Loss')
                axes[1, 0].set_title('Policy Gradient Loss')
                axes[1, 0].legend()
                axes[1, 0].grid(True, alpha=0.3)

            # Plot 4: Value Loss
            if 'train/value_loss' in scalar_tags:
                losses = event_acc.Scalars('train/value_loss')
                steps = [r.step for r in losses]
                values = [r.value for r in losses]

                axes[1, 1].plot(steps, values, color='orange', alpha=0.6, label='Raw')
                if len(values) > window_size:
                    smoothed = pd.Series(values).rolling(window_size).mean()
                    axes[1, 1].plot(steps, smoothed, color='darkorange', linewidth=2, label=f'{window_size}-step MA')
                axes[1, 1].set_xlabel('Timesteps')
                axes[1, 1].set_ylabel('Value Loss')
                axes[1, 1].set_title('Value Function Loss')
                axes[1, 1].legend()
                axes[1, 1].grid(True, alpha=0.3)

            plt.tight_layout()

            # Save plot
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            plot_path = self.plots_dir / f"learning_curves_{timestamp}.png"
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            print(f"\nLearning curves saved to: {plot_path}")

            plt.close()

        except ImportError:
            print("\nTensorBoard not installed. Install with: pip install tensorboard")
            print("Skipping learning curve generation.")
        except Exception as e:
            print(f"\nError generating learning curves: {e}")

    def evaluate_baseline_vs_rl(self, n_episodes: int = 50):
        """
        Compare baseline (fixed threshold) vs RL-optimized thresholds.

        Args:
            n_episodes: Number of episodes to evaluate
        """
        print("\n" + "=" * 80)
        print("BASELINE vs RL COMPARISON")
        print("=" * 80)

        if self.agent is None:
            print("No agent loaded! Train or load a model first.")
            return

        # Baseline: Fixed threshold at 0.5
        print("\nEvaluating baseline (fixed threshold = 0.5)...")
        baseline_rewards = []
        baseline_accuracies = []

        for ep in range(n_episodes):
            obs, info = self.env.reset()
            episode_reward = 0
            done = False

            while not done:
                # Fixed threshold action
                action = np.array([0.5, 1.0])  # threshold=0.5, priority=medium
                obs, reward, terminated, truncated, info = self.env.step(action)
                episode_reward += reward
                done = terminated or truncated

            baseline_rewards.append(episode_reward)
            baseline_accuracies.append(info.get('episode_accuracy', 0.0))

        # RL: Learned policy
        print("Evaluating RL agent (learned thresholds)...")
        rl_metrics = self.agent.evaluate(n_episodes=n_episodes, render=False)

        # Create comparison plot
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        fig.suptitle('Baseline vs RL Performance Comparison', fontsize=14, fontweight='bold')

        # Rewards comparison
        data_rewards = pd.DataFrame({
            'Method': ['Baseline'] * len(baseline_rewards) + ['RL'] * n_episodes,
            'Reward': baseline_rewards + [rl_metrics['mean_reward']] * n_episodes
        })

        axes[0].bar(['Baseline', 'RL'],
                   [np.mean(baseline_rewards), rl_metrics['mean_reward']],
                   color=['#3498db', '#2ecc71'],
                   yerr=[np.std(baseline_rewards), rl_metrics['std_reward']],
                   capsize=10)
        axes[0].set_ylabel('Mean Reward')
        axes[0].set_title('Episode Rewards')
        axes[0].grid(True, alpha=0.3)

        # Add value labels
        for i, (label, val) in enumerate([('Baseline', np.mean(baseline_rewards)),
                                          ('RL', rl_metrics['mean_reward'])]):
            axes[0].text(i, val + 0.5, f'{val:.1f}', ha='center', va='bottom', fontweight='bold')

        # Accuracy comparison
        axes[1].bar(['Baseline', 'RL'],
                   [np.mean(baseline_accuracies), rl_metrics['mean_accuracy']],
                   color=['#3498db', '#2ecc71'],
                   yerr=[np.std(baseline_accuracies), rl_metrics['std_accuracy']],
                   capsize=10)
        axes[1].set_ylabel('Accuracy')
        axes[1].set_title('Prediction Accuracy')
        axes[1].set_ylim(0, 1)
        axes[1].grid(True, alpha=0.3)

        # Add value labels
        for i, (label, val) in enumerate([('Baseline', np.mean(baseline_accuracies)),
                                          ('RL', rl_metrics['mean_accuracy'])]):
            axes[1].text(i, val + 0.02, f'{val:.1%}', ha='center', va='bottom', fontweight='bold')

        plt.tight_layout()

        # Save comparison plot
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        plot_path = self.plots_dir / f"baseline_vs_rl_{timestamp}.png"
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        print(f"\nComparison plot saved to: {plot_path}")

        # Print summary
        print("\n" + "=" * 80)
        print("COMPARISON RESULTS")
        print("=" * 80)
        print(f"\nBaseline (Fixed Threshold = 0.5):")
        print(f"  Mean Reward: {np.mean(baseline_rewards):+.2f} ± {np.std(baseline_rewards):.2f}")
        print(f"  Mean Accuracy: {np.mean(baseline_accuracies):.2%} ± {np.std(baseline_accuracies):.2%}")

        print(f"\nRL Agent (Learned Thresholds):")
        print(f"  Mean Reward: {rl_metrics['mean_reward']:+.2f} ± {rl_metrics['std_reward']:.2f}")
        print(f"  Mean Accuracy: {rl_metrics['mean_accuracy']:.2%} ± {rl_metrics['std_accuracy']:.2%}")

        improvement_reward = ((rl_metrics['mean_reward'] - np.mean(baseline_rewards)) / abs(np.mean(baseline_rewards))) * 100
        improvement_acc = ((rl_metrics['mean_accuracy'] - np.mean(baseline_accuracies)) / np.mean(baseline_accuracies)) * 100

        print(f"\nImprovement:")
        print(f"  Reward: {improvement_reward:+.1f}%")
        print(f"  Accuracy: {improvement_acc:+.1f}%")

        plt.close()


def main():
    """Main training pipeline."""
    print("\n" + "*" * 80)
    print("*" + " " * 78 + "*")
    print("*" + " " * 20 + "ATHENA RL TRAINER" + " " * 41 + "*")
    print("*" + " " * 15 + "Reinforcement Learning Bug Predictor" + " " * 27 + "*")
    print("*" + " " * 78 + "*")
    print("*" * 80)
    print()

    # Create trainer
    trainer = RLTrainer()

    # Setup
    trainer.setup()

    # Train
    print("\n" + "=" * 80)
    print("TRAINING PPO AGENT")
    print("=" * 80)

    results = trainer.train(
        total_timesteps=10000,
        checkpoint_freq=1000,
        eval_freq=2000,
        n_eval_episodes=10
    )

    # Generate learning curves
    trainer.generate_learning_curves()

    # Compare baseline vs RL
    trainer.evaluate_baseline_vs_rl(n_episodes=50)

    print("\n" + "=" * 80)
    print("TRAINING COMPLETE!")
    print("=" * 80)
    print(f"\nModel saved to: {results['model_path']}")
    print(f"\nFinal Performance:")
    print(f"  Mean Reward: {results['eval_metrics']['mean_reward']:+.2f}")
    print(f"  Mean Accuracy: {results['eval_metrics']['mean_accuracy']:.2%}")
    print("\nNext steps:")
    print("  1. View TensorBoard: tensorboard --logdir rl/logs")
    print("  2. Check plots in: rl/plots/")
    print("  3. Use RL agent in API: api/rl_predict.py")
    print()


if __name__ == "__main__":
    main()
