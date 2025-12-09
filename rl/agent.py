"""
ATHENA RL Agent - PPO Configuration for Bug Prediction Optimization

Uses Stable-Baselines3 PPO with MlpPolicy to learn optimal thresholds.
"""
import os
from pathlib import Path
from typing import Optional, Dict, Any
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback
from stable_baselines3.common.vec_env import DummyVecEnv
import torch


class TensorboardCallback(BaseCallback):
    """
    Custom callback for logging additional metrics to TensorBoard.
    """

    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.episode_rewards = []
        self.episode_accuracies = []

    def _on_step(self) -> bool:
        # Log episode metrics when available
        if 'episode' in self.locals.get('infos', [{}])[0]:
            info = self.locals['infos'][0]
            if 'episode_accuracy' in info:
                self.logger.record('train/episode_accuracy', info['episode_accuracy'])
                self.logger.record('train/total_accuracy', info['total_accuracy'])

        return True


class BugPredictionAgent:
    """
    PPO agent for learning optimal bug prediction thresholds.

    Architecture:
    - Policy: MlpPolicy with [256, 128] hidden layers
    - Learning rate: 3e-4
    - Batch size: 64
    - Epochs: 10
    """

    def __init__(
        self,
        env,
        learning_rate: float = 3e-4,
        n_steps: int = 2048,
        batch_size: int = 64,
        n_epochs: int = 10,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_range: float = 0.2,
        ent_coef: float = 0.01,
        vf_coef: float = 0.5,
        max_grad_norm: float = 0.5,
        device: str = "auto",
        verbose: int = 1
    ):
        """
        Initialize PPO agent.

        Args:
            env: Gym environment
            learning_rate: Learning rate for optimizer
            n_steps: Steps to collect before update
            batch_size: Minibatch size
            n_epochs: Training epochs per update
            gamma: Discount factor
            gae_lambda: GAE lambda parameter
            clip_range: PPO clipping parameter
            ent_coef: Entropy coefficient
            vf_coef: Value function coefficient
            max_grad_norm: Max gradient norm
            device: Device to use (cpu/cuda/auto)
            verbose: Verbosity level
        """
        self.env = env

        # Create PPO model with custom policy network
        self.model = PPO(
            policy="MlpPolicy",
            env=env,
            learning_rate=learning_rate,
            n_steps=n_steps,
            batch_size=batch_size,
            n_epochs=n_epochs,
            gamma=gamma,
            gae_lambda=gae_lambda,
            clip_range=clip_range,
            ent_coef=ent_coef,
            vf_coef=vf_coef,
            max_grad_norm=max_grad_norm,
            device=device,
            verbose=verbose,
            policy_kwargs={
                "net_arch": {
                    "pi": [256, 128],  # Policy network
                    "vf": [256, 128]   # Value network
                },
                "activation_fn": torch.nn.ReLU
            },
            tensorboard_log="rl/logs/"
        )

        self.training_stats = {
            'total_timesteps': 0,
            'episodes': 0,
            'mean_reward': 0.0,
            'mean_accuracy': 0.0
        }

    def train(
        self,
        total_timesteps: int = 10000,
        checkpoint_freq: int = 1000,
        save_path: str = "rl/models/ppo_bug_predictor",
        tb_log_name: str = "ppo_bug_predictor"
    ) -> Dict[str, Any]:
        """
        Train the PPO agent.

        Args:
            total_timesteps: Total timesteps to train
            checkpoint_freq: Frequency of checkpoint saves
            save_path: Path to save model checkpoints
            tb_log_name: TensorBoard log name

        Returns:
            Training statistics
        """
        print("=" * 80)
        print("TRAINING PPO AGENT")
        print("=" * 80)
        print(f"\nConfiguration:")
        print(f"  - Total timesteps: {total_timesteps:,}")
        print(f"  - Checkpoint frequency: {checkpoint_freq:,}")
        print(f"  - Learning rate: {self.model.learning_rate}")
        print(f"  - Policy architecture: [256, 128]")
        print(f"  - Device: {self.model.device}")

        # Setup callbacks
        checkpoint_callback = CheckpointCallback(
            save_freq=checkpoint_freq,
            save_path=str(Path(save_path).parent),
            name_prefix=Path(save_path).name
        )

        tensorboard_callback = TensorboardCallback()

        # Train
        print("\nStarting training...\n")
        self.model.learn(
            total_timesteps=total_timesteps,
            callback=[checkpoint_callback, tensorboard_callback],
            tb_log_name=tb_log_name,
            progress_bar=True
        )

        # Update stats
        self.training_stats['total_timesteps'] = total_timesteps

        print("\n" + "=" * 80)
        print("TRAINING COMPLETE!")
        print("=" * 80)

        # Save final model
        final_model_path = f"{save_path}_final.zip"
        self.model.save(final_model_path)
        print(f"\nFinal model saved to: {final_model_path}")

        return self.training_stats

    def predict(
        self,
        observation: np.ndarray,
        deterministic: bool = True
    ) -> tuple:
        """
        Predict action for given observation.

        Args:
            observation: Current state
            deterministic: Use deterministic policy (True for inference)

        Returns:
            action, state
        """
        action, state = self.model.predict(observation, deterministic=deterministic)
        return action, state

    def evaluate(
        self,
        n_episodes: int = 10,
        render: bool = False
    ) -> Dict[str, float]:
        """
        Evaluate agent performance.

        Args:
            n_episodes: Number of episodes to evaluate
            render: Whether to render environment

        Returns:
            Evaluation metrics
        """
        print("=" * 80)
        print(f"EVALUATING AGENT ({n_episodes} episodes)")
        print("=" * 80)

        episode_rewards = []
        episode_accuracies = []

        for ep in range(n_episodes):
            obs, info = self.env.reset()
            episode_reward = 0
            done = False

            while not done:
                action, _ = self.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, info = self.env.step(action)
                episode_reward += reward
                done = terminated or truncated

                if render:
                    self.env.render()

            episode_rewards.append(episode_reward)
            episode_accuracies.append(info.get('episode_accuracy', 0.0))

            print(f"Episode {ep + 1}: Reward = {episode_reward:+.1f}, Accuracy = {info.get('episode_accuracy', 0.0):.2%}")

        metrics = {
            'mean_reward': np.mean(episode_rewards),
            'std_reward': np.std(episode_rewards),
            'mean_accuracy': np.mean(episode_accuracies),
            'std_accuracy': np.std(episode_accuracies)
        }

        print("\n" + "=" * 80)
        print("EVALUATION RESULTS")
        print("=" * 80)
        print(f"Mean Reward: {metrics['mean_reward']:+.2f} ± {metrics['std_reward']:.2f}")
        print(f"Mean Accuracy: {metrics['mean_accuracy']:.2%} ± {metrics['std_accuracy']:.2%}")

        return metrics

    def save(self, path: str):
        """Save model to disk."""
        self.model.save(path)
        print(f"Model saved to: {path}")

    @classmethod
    def load(cls, path: str, env):
        """
        Load trained model from disk.

        Args:
            path: Path to saved model
            env: Environment instance

        Returns:
            BugPredictionAgent with loaded model
        """
        agent = cls(env)
        agent.model = PPO.load(path, env=env)
        print(f"Model loaded from: {path}")
        return agent


def create_agent(env, **kwargs) -> BugPredictionAgent:
    """
    Create PPO agent with default configuration.

    Args:
        env: Gym environment
        **kwargs: Additional arguments for agent

    Returns:
        BugPredictionAgent instance
    """
    print("=" * 80)
    print("CREATING PPO AGENT")
    print("=" * 80)

    agent = BugPredictionAgent(env, **kwargs)

    print("\nAgent created successfully!")
    print(f"  - Policy: MlpPolicy")
    print(f"  - Architecture: [256, 128]")
    print(f"  - Learning rate: {agent.model.learning_rate}")
    print(f"  - Device: {agent.model.device}")

    return agent


def load_latest_agent(env) -> Optional[BugPredictionAgent]:
    """
    Load most recently trained agent.

    Args:
        env: Environment instance

    Returns:
        BugPredictionAgent if found, None otherwise
    """
    models_dir = Path("rl/models")
    if not models_dir.exists():
        return None

    # Find latest final model
    model_files = list(models_dir.glob("ppo_bug_predictor_final.zip"))

    if not model_files:
        # Try checkpoint models
        model_files = list(models_dir.glob("ppo_bug_predictor_*_steps.zip"))

    if not model_files:
        return None

    latest_model = max(model_files, key=lambda p: p.stat().st_ctime)
    print(f"Loading latest model: {latest_model.name}")

    return BugPredictionAgent.load(str(latest_model), env)


if __name__ == "__main__":
    # Test agent creation
    from rl.environment import create_environment

    print("\nCreating environment...")
    env = create_environment()

    print("\nCreating agent...")
    agent = create_agent(env)

    print("\nTesting prediction...")
    obs, _ = env.reset()
    action, _ = agent.predict(obs)
    print(f"Sample action: [threshold={action[0]:.3f}, priority={action[1]:.1f}]")

    print("\nAgent ready for training!")
