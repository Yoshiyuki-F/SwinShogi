"""
Training module for SwinShogi reinforcement learning

This module implements the training pipeline that combines self-play data generation
with neural network training using policy gradient methods.
"""

import jax
import jax.numpy as jnp
import optax
import pickle
import numpy as np
import time
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional
from collections import deque

from src.utils.model_utils import PolicyGradientLoss
from src.utils.checkpoint import CheckpointManager, save_model_checkpoint, load_model_checkpoint
from src.model.shogi_model import create_swin_shogi_model
from src.model.actor_critic import ActorCritic
from src.rl.mcts import MCTS
from src.rl.self_play import SelfPlay
from src.rl.data_generator import (
    DataGenerationManager, TrainingExample, 
    create_self_play_data_source, create_game_record_data_source, create_ai_opponent_data_source
)
from config.default_config import get_model_config, get_training_config, get_mcts_config, TrainingConfig

# Logging configuration
logger = logging.getLogger(__name__)



class ReplayBuffer:
    """Replay buffer for storing training examples"""
    
    def __init__(self, max_size: int = 10000):
        """
        Initialize replay buffer
        
        Args:
            max_size: Maximum number of examples to store
        """
        self.max_size = max_size
        self.buffer = deque(maxlen=max_size)
        
    def add(self, example: TrainingExample):
        """Add training example to buffer"""
        self.buffer.append(example)
        
    def add_batch(self, examples: List[TrainingExample]):
        """Add batch of training examples"""
        for example in examples:
            self.add(example)
            
    def sample(self, batch_size: int) -> List[TrainingExample]:
        """Sample random batch from buffer"""
        if len(self.buffer) < batch_size:
            return list(self.buffer)
        
        indices = np.random.choice(len(self.buffer), batch_size, replace=False)
        return [self.buffer[i] for i in indices]
    
    def __len__(self) -> int:
        return len(self.buffer)
    
    def clear(self):
        """Clear all examples from buffer"""
        self.buffer.clear()


class TrainState:
    """Training state management"""
    
    def __init__(self, params, optimizer_state, step: int = 0):
        """
        Initialize training state
        
        Args:
            params: Model parameters
            optimizer_state: Optimizer state
            step: Current training step
        """
        self.params = params
        self.optimizer_state = optimizer_state
        self.step = step
        
    def update(self, params, optimizer_state):
        """Update training state"""
        self.params = params
        self.optimizer_state = optimizer_state
        self.step += 1
        return self


class Trainer:
    """Main trainer class for SwinShogi"""
    
    def __init__(self, 
                 model=None, 
                 params=None,
                 config: Optional[TrainingConfig] = None,
                 checkpoint_dir: str = "data/checkpoints",
                 resume_from_checkpoint: Optional[str] = None):
        """
        Initialize trainer
        
        Args:
            model: SwinShogi model (created if None)
            params: Model parameters (created if None)
            config: Training configuration
            checkpoint_dir: Directory to save/load checkpoints
            resume_from_checkpoint: Path to checkpoint to resume from
        """
        self.config = config or get_training_config()
        self.checkpoint_manager = CheckpointManager(checkpoint_dir)
        
        # Try to resume from checkpoint
        if resume_from_checkpoint or (model is None and params is None):
            try:
                checkpoint_path = resume_from_checkpoint
                if resume_from_checkpoint is None:
                    # Try to load latest checkpoint
                    checkpoints = self.checkpoint_manager.list_checkpoints()
                    if checkpoints:
                        checkpoint_path = checkpoints[-1]['path']  # Latest checkpoint
                        logger.info(f"Found existing checkpoint: {checkpoint_path}")
                
                if checkpoint_path:
                    params, optimizer_state, metadata = self.checkpoint_manager.load_checkpoint(
                        checkpoint_path, load_optimizer_state=True
                    )
                    
                    # Create model if not provided
                    if model is None:
                        rng = jax.random.PRNGKey(42)
                        model_config = get_model_config()
                        model, _ = create_swin_shogi_model(rng, model_config)
                    
                    logger.info(f"Resumed from checkpoint at step {metadata.get('step', 0)}")
                    self.start_step = metadata.get('step', 0)
                    self.resume_optimizer_state = optimizer_state
                else:
                    raise FileNotFoundError("No checkpoint found")
                    
            except FileNotFoundError:
                if resume_from_checkpoint:
                    raise FileNotFoundError(f"Checkpoint not found: {resume_from_checkpoint}")
                
                # Create new model
                logger.info("No existing checkpoint found, creating new model")
                rng = jax.random.PRNGKey(42)
                model_config = get_model_config()
                model, params = create_swin_shogi_model(rng, model_config)
                self.start_step = 0
                self.resume_optimizer_state = None
        else:
            # Use provided model and parameters
            self.start_step = 0
            self.resume_optimizer_state = None
            
        self.model = model
        self.actor_critic = ActorCritic(model, params)
        
        # Initialize optimizer
        self.optimizer = self._create_optimizer()
        
        # Use resume optimizer state if available
        if self.resume_optimizer_state is not None:
            optimizer_state = self.resume_optimizer_state
        else:
            optimizer_state = self.optimizer.init(params)
            
        self.train_state = TrainState(
            params=params,
            optimizer_state=optimizer_state
        )
        
        # Initialize replay buffer
        self.replay_buffer = ReplayBuffer(self.config.replay_buffer_size)
        
        # Initialize MCTS for self-play
        mcts_config = get_mcts_config()
        mcts_config.n_simulations = self.config.mcts_simulations
        self.mcts = MCTS(model, params, mcts_config)
        
        # Initialize self-play
        self.self_play = SelfPlay(
            model, 
            params, 
            mcts_config,
            max_moves=self.config.max_moves,
            temperature_threshold=self.config.temperature_threshold,
            temperature_init=self.config.temperature_init,
            temperature_final=self.config.temperature_final
        )
        
        # Initialize data generation manager
        self.data_manager = DataGenerationManager()
        
        # Add self-play data source by default
        self_play_source = create_self_play_data_source(model, params, 
                                                       n_simulations=self.config.mcts_simulations)
        self.data_manager.add_data_source(self_play_source)
        
        # Training statistics
        self.training_stats = {
            'iteration': 0,
            'total_games': 0,
            'avg_game_length': 0.0,
            'policy_loss': 0.0,
            'value_loss': 0.0,
            'entropy_loss': 0.0,
            'total_loss': 0.0,
            'learning_rate': self.config.learning_rate,
            'buffer_size': 0
        }
        
        # JIT compile training functions
        self._loss_fn = jax.jit(self._loss_fn_impl)
        self._train_step = jax.jit(self._train_step_impl)
        
        logger.info(f"Trainer initialized with config: {self.config}")
    
    def _create_optimizer(self):
        """Create optimizer with learning rate schedule"""
        # Calculate total training steps
        total_steps = max(1000, self.config.num_iterations * self.config.train_steps_per_iteration)
        
        # Learning rate schedule with warmup
        schedule = optax.warmup_cosine_decay_schedule(
            init_value=0.0,
            peak_value=self.config.learning_rate,
            warmup_steps=min(1000, total_steps // 10),
            decay_steps=total_steps,
            end_value=self.config.learning_rate * 0.01
        )
        
        # Optimizer with gradient clipping and weight decay
        optimizer = optax.chain(
            optax.clip_by_global_norm(self.config.grad_clip_norm),
            optax.adamw(
                learning_rate=schedule,
                weight_decay=self.config.weight_decay
            )
        )
        
        return optimizer
    
    def _loss_fn_impl(self, params, batch_board_states, batch_feature_vectors, 
                      batch_action_probs, batch_values):
        """
        Compute loss for a batch of training examples
        
        Args:
            params: Model parameters
            batch_board_states: Batch of board states (batch_size, 9, 9, 2)
            batch_feature_vectors: Batch of feature vectors (batch_size, 15)
            batch_action_probs: Batch of target action probabilities (batch_size, 2187)
            batch_values: Batch of target values (batch_size,)
            
        Returns:
            (total_loss, (policy_loss, value_loss, entropy_loss))
        """
        # Forward pass
        policy_logits, predicted_values = self.model.apply(
            params, 
            batch_board_states, 
            feature_vector=batch_feature_vectors,
            deterministic=True
        )
        
        # Reshape values if needed
        if len(predicted_values.shape) > 1:
            predicted_values = predicted_values.squeeze(-1)
        
        # Compute losses using PolicyGradientLoss
        total_loss, (policy_loss, value_loss, entropy_loss) = PolicyGradientLoss.compute_losses_from_model_outputs(
            policy_logits=policy_logits,
            values=predicted_values,
            action_onehot=batch_action_probs,
            advantages=jnp.zeros_like(batch_values),  # We don't use advantages in this setup
            target_values=batch_values,
            entropy_coeff=self.config.entropy_coef
        )
        
        # Apply loss coefficients
        weighted_total_loss = (
            self.config.policy_coef * policy_loss +
            self.config.value_coef * value_loss +
            self.config.entropy_coef * entropy_loss
        )
        
        return weighted_total_loss, (policy_loss, value_loss, entropy_loss)
    
    def _train_step_impl(self, train_state, batch_board_states, batch_feature_vectors,
                        batch_action_probs, batch_values):
        """
        Single training step
        
        Args:
            train_state: Current training state
            batch_board_states: Batch of board states
            batch_feature_vectors: Batch of feature vectors
            batch_action_probs: Batch of action probabilities
            batch_values: Batch of values
            
        Returns:
            (new_train_state, loss_info)
        """
        # Compute gradients
        (loss, loss_components), grads = jax.value_and_grad(
            self._loss_fn_impl, 
            has_aux=True
        )(train_state.params, batch_board_states, batch_feature_vectors,
          batch_action_probs, batch_values)
        
        # Apply gradients
        updates, new_optimizer_state = self.optimizer.update(
            grads, train_state.optimizer_state, train_state.params
        )
        new_params = optax.apply_updates(train_state.params, updates)
        
        # Update training state
        new_train_state = TrainState(
            params=new_params,
            optimizer_state=new_optimizer_state,
            step=train_state.step + 1
        )
        
        return new_train_state, (loss, loss_components)
    
    def generate_training_data(self, 
                              source_config: Dict[str, int] = None) -> List[TrainingExample]:
        """
        Generate training data from various sources
        
        Args:
            source_config: Dict mapping source types to number of games.
                          If None, uses self-play with default config.
            
        Returns:
            List of training examples
        """
        if source_config is None:
            source_config = {'SelfPlayDataSource': self.config.num_episodes}
        
        return self.data_manager.generate_mixed_data(source_config)
    
    def add_data_source(self, source_type: str, **kwargs):
        """
        Add a new data source to the trainer
        
        Args:
            source_type: Type of data source ('game_records', 'ai_opponent')
            **kwargs: Arguments for the data source
        """
        if source_type == 'game_records':
            source = create_game_record_data_source(self.model, self.train_state.params)
            self.data_manager.add_data_source(source)
        elif source_type == 'ai_opponent':
            opponent_path = kwargs.get('opponent_engine_path', '')
            source = create_ai_opponent_data_source(self.model, self.train_state.params, opponent_path)
            self.data_manager.add_data_source(source)
        else:
            raise ValueError(f"Unknown data source type: {source_type}")
        
        logger.info(f"Added {source_type} data source")
    
    def generate_from_game_records(self, game_records: List[List[str]]) -> List[TrainingExample]:
        """
        Generate training data from game records
        
        Args:
            game_records: List of games, each game is a list of USI moves
            
        Returns:
            List of training examples
        """
        return self.data_manager.generate_from_source(
            'GameRecordDataSource', 
            len(game_records), 
            game_records=game_records
        )
    
    def train_on_batch(self, examples: List[TrainingExample]) -> Dict[str, float]:
        """
        Train on a batch of examples
        
        Args:
            examples: List of training examples
            
        Returns:
            Dictionary of loss components
        """
        if len(examples) == 0:
            return {'policy_loss': 0.0, 'value_loss': 0.0, 'entropy_loss': 0.0, 'total_loss': 0.0}
        
        # Convert examples to batched arrays
        batch_board_states = jnp.stack([ex.board_state for ex in examples])
        batch_feature_vectors = jnp.stack([ex.feature_vector for ex in examples])
        batch_action_probs = jnp.stack([ex.action_probs for ex in examples])
        batch_values = jnp.array([ex.value for ex in examples])
        
        # Perform training step
        self.train_state, (total_loss, (policy_loss, value_loss, entropy_loss)) = self._train_step(
            self.train_state,
            batch_board_states,
            batch_feature_vectors,
            batch_action_probs,
            batch_values
        )
        
        # Update actor-critic with new parameters
        self.actor_critic.update_params(self.train_state.params)
        self.mcts.params = self.train_state.params
        self.self_play.update_model_params(self.train_state.params)
        
        return {
            'policy_loss': float(policy_loss),
            'value_loss': float(value_loss),
            'entropy_loss': float(entropy_loss),
            'total_loss': float(total_loss)
        }
    
    def train_iteration(self, iteration: int) -> Dict[str, Any]:
        """
        Run one training iteration
        
        Args:
            iteration: Current iteration number
            
        Returns:
            Training statistics
        """
        logger.info(f"Starting training iteration {iteration}")
        
        # Generate training data
        new_examples = self.generate_training_data()
        self.replay_buffer.add_batch(new_examples)
        
        # Train on data
        total_losses = {'policy_loss': [], 'value_loss': [], 'entropy_loss': [], 'total_loss': []}
        
        if len(self.replay_buffer) >= self.config.min_buffer_size:
            for step in range(self.config.train_steps_per_iteration):
                # Sample batch from replay buffer
                batch_examples = self.replay_buffer.sample(self.config.batch_size)
                
                # Train on batch
                losses = self.train_on_batch(batch_examples)
                
                # Accumulate losses
                for key, value in losses.items():
                    total_losses[key].append(value)
                
                if (step + 1) % 20 == 0:
                    logger.info(f"Training step {step + 1}/{self.config.train_steps_per_iteration}")
        
        # Calculate average losses
        avg_losses = {key: np.mean(values) if values else 0.0 for key, values in total_losses.items()}
        
        # Update training statistics
        self.training_stats.update({
            'iteration': iteration,
            'total_games': self.training_stats['total_games'] + len(new_examples),
            'buffer_size': len(self.replay_buffer),
            **avg_losses
        })
        
        logger.info(f"Iteration {iteration} completed. Losses: {avg_losses}")
        
        return self.training_stats
    
    def save_checkpoint(self, step: int):
        """
        Save training checkpoint using CheckpointManager
        
        Args:
            step: Current training step
        """
        metadata = {
            'training_stats': self.training_stats,
            'config': self.config.__dict__,
            'replay_buffer_size': len(self.replay_buffer)
        }
        
        checkpoint_path = self.checkpoint_manager.save_checkpoint(
            params=self.train_state.params,
            step=step,
            optimizer_state=self.train_state.optimizer_state,
            metadata=metadata
        )
        
        logger.info(f"Checkpoint saved at step {step}: {checkpoint_path}")
        
        # Clean up old checkpoints
        self.checkpoint_manager.delete_old_checkpoints(keep_n=5)
    
    def load_checkpoint(self, checkpoint_path: Optional[str] = None):
        """
        Load training checkpoint using CheckpointManager
        
        Args:
            checkpoint_path: Path to specific checkpoint (None for latest)
        """
        params, optimizer_state, metadata = self.checkpoint_manager.load_checkpoint(
            checkpoint_path, load_optimizer_state=True
        )
        
        # Update train state
        self.train_state = TrainState(
            params=params,
            optimizer_state=optimizer_state
        )
        
        # Restore training stats if available
        if 'training_stats' in metadata:
            self.training_stats = metadata['training_stats']
        
        # Update models with loaded parameters
        self.actor_critic.update_params(params)
        self.mcts.params = params
        self.self_play.update_model_params(params)
        
        step = metadata.get('step', 0)
        logger.info(f"Checkpoint loaded from step {step}")
        return step
    
    def train(self, num_iterations: Optional[int] = None):
        """
        Run full training loop
        
        Args:
            num_iterations: Number of iterations (uses config if None)
        """
        if num_iterations is None:
            num_iterations = self.config.num_iterations
        
        logger.info(f"Starting training for {num_iterations} iterations")
        
        for iteration in range(self.start_step + 1, self.start_step + num_iterations + 1):
            start_time = time.time()
            
            # Run training iteration
            stats = self.train_iteration(iteration)
            
            # Save checkpoint periodically
            if iteration % self.config.checkpoint_interval == 0:
                self.save_checkpoint(iteration)
            
            # Log progress
            elapsed_time = time.time() - start_time
            logger.info(f"Iteration {iteration}/{self.start_step + num_iterations} completed in {elapsed_time:.2f}s")
            logger.info(f"Stats: {stats}")
        
        # Save final checkpoint
        final_step = self.start_step + num_iterations
        self.save_checkpoint(final_step)
        logger.info("Training completed!")


# Convenience functions
def create_trainer(config: Optional[TrainingConfig] = None, 
                  checkpoint_dir: str = "data/checkpoints",
                  resume_from_checkpoint: Optional[str] = None) -> Trainer:
    """Create trainer with default configuration"""
    return Trainer(config=config, checkpoint_dir=checkpoint_dir, resume_from_checkpoint=resume_from_checkpoint)

def train_model(num_iterations: int = 100, 
               checkpoint_dir: str = "data/checkpoints",
               resume_from_checkpoint: Optional[str] = None) -> Trainer:
    """Train model with default configuration"""
    trainer = create_trainer(checkpoint_dir=checkpoint_dir, resume_from_checkpoint=resume_from_checkpoint)
    trainer.train(num_iterations)
    return trainer