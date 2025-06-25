"""
Training module for SwinShogi reinforcement learning

This module implements the training pipeline that combines self-play data generation
with neural network training using policy gradient methods.
"""

import jax
import jax.numpy as jnp
import optax
import numpy as np
import time
import logging
from typing import Dict, List, Any, Optional
from collections import deque

from src.utils.model_utils import PolicyGradientLoss
from src.utils.checkpoint import CheckpointManager
from src.model.shogi_model import create_swin_shogi_model
from src.model.actor_critic import ActorCritic
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
                 model: Any, 
                 params: Any,
                 config: Optional[TrainingConfig] = None,
                 checkpoint_dir: str = "data/checkpoints"):
        self.config = config or get_training_config()
        self.checkpoint_manager = CheckpointManager(checkpoint_dir)
        self.model = model

        # The ActorCritic is the single source of truth for the model and its parameters.
        self.actor_critic = ActorCritic(model, params)

        # The SelfPlay module gets the ActorCritic instance directly.
        self.self_play = SelfPlay(self.actor_critic, get_mcts_config())
        
        # Initialize data generation manager
        self.data_manager = DataGenerationManager()
        # Add default self-play data source
        self.data_manager.add_data_source(create_self_play_data_source(self.actor_critic))

        self.optimizer = self._create_optimizer()
        optimizer_state = self.optimizer.init(params)

        self.train_state = TrainState(params=params, optimizer_state=optimizer_state)
        self.replay_buffer = ReplayBuffer(self.config.replay_buffer_size)
        
        self.training_stats = {
            'iteration': 0,
            'total_games': 0,
            'buffer_size': 0,
            'policy_loss': 0.0,
            'value_loss': 0.0, 
            'entropy_loss': 0.0,
            'total_loss': 0.0
        }
        
        self.start_step = 0
        
        self._loss_fn = jax.jit(self._loss_fn_impl)
        self._train_step = jax.jit(self._train_step_impl)
    
    @classmethod
    def create(cls, 
               config: Optional[TrainingConfig] = None,
               checkpoint_dir: str = "data/checkpoints",
               resume_from: Optional[str] = None) -> 'Trainer':
        """
        Creates and initializes a Trainer instance, handling new creation and checkpoint resumption.
        
        Args:
            config: Training configuration (uses default if None)
            checkpoint_dir: Directory for checkpoints
            resume_from: Specific checkpoint path to resume from (auto-detects latest if None)
            
        Returns:
            Initialized Trainer instance
        """
        logger.info("Creating trainer...")
        config = config or get_training_config()
        checkpoint_manager = CheckpointManager(checkpoint_dir)
        
        # Determine the checkpoint to load from
        checkpoint_path = resume_from
        if checkpoint_path is None:
            latest_path = checkpoint_manager.checkpoint_dir / "latest"
            if latest_path.exists():
                logger.info(f"No specific checkpoint provided, resuming from latest")
                checkpoint_path = str(latest_path)

        if checkpoint_path:
            # --- Resume from checkpoint ---
            logger.info(f"Loading checkpoint from: {checkpoint_path}")
            try:
                # Load params and optimizer state
                params, opt_state, metadata = checkpoint_manager.load_checkpoint(
                    checkpoint_path, load_optimizer_state=True
                )
                
                # Create a dummy model to get the structure, then use loaded params
                rng_key = jax.random.PRNGKey(42)
                model, _ = create_swin_shogi_model(rng_key, get_model_config())
                
                trainer = cls(model=model, params=params, config=config, checkpoint_dir=checkpoint_dir)
                
                # Restore optimizer state and training step
                trainer.train_state.optimizer_state = opt_state
                trainer.train_state.step = metadata.get('step', 0)
                trainer.start_step = trainer.train_state.step
                
                # Restore training stats if available
                if 'training_stats' in metadata:
                    trainer.training_stats = metadata['training_stats']
                
                logger.info(f"Successfully resumed from step {trainer.start_step}")
                
            except FileNotFoundError:
                logger.error(f"Checkpoint not found at {checkpoint_path}. Aborting.")
                raise
        else:
            # --- Create from scratch ---
            logger.info("No checkpoint found. Creating new model and trainer.")
            rng_key = jax.random.PRNGKey(42)
            model, params = create_swin_shogi_model(rng_key, get_model_config())
            trainer = cls(model=model, params=params, config=config, checkpoint_dir=checkpoint_dir)

        return trainer
    
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
        
        # Compute losses using AlphaZero-style loss (no advantages)
        total_loss, (policy_loss, value_loss, entropy_loss) = PolicyGradientLoss.compute_alphazero_losses(
            policy_logits=policy_logits,
            values=predicted_values,
            action_probs=batch_action_probs,
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
            source = create_game_record_data_source(self.actor_critic)
            self.data_manager.add_data_source(source)
        elif source_type == 'ai_opponent':
            opponent_path = kwargs.get('opponent_engine_path', '')
            source = create_ai_opponent_data_source(self.actor_critic, opponent_path)
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
        """Trains the model on a single batch of examples."""
        if not examples:
            return {k: 0.0 for k in ['policy_loss', 'value_loss', 'entropy_loss', 'total_loss']}

        # Convert list of examples to batched JAX arrays
        board_states = jnp.stack([ex.board_state for ex in examples])
        feature_vectors = jnp.stack([ex.feature_vector for ex in examples])
        target_policies = jnp.stack([ex.action_probs for ex in examples])
        target_values = jnp.array([ex.value for ex in examples])

        # Perform a training step
        self.train_state, (loss, loss_components) = self._train_step(
            self.train_state, board_states, feature_vectors, target_policies, target_values
        )

        # CRITICAL: Update the ActorCritic with the new parameters.
        # This is the single point of update.
        self.actor_critic.update_params(self.train_state.params)

        policy_loss, value_loss, entropy_loss = loss_components
        return {
            'policy_loss': float(policy_loss),
            'value_loss': float(value_loss),
            'entropy_loss': float(entropy_loss),
            'total_loss': float(loss)
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
                  resume_from: Optional[str] = None) -> Trainer:
    """
    Create trainer with default configuration using factory method
    
    Args:
        config: Training configuration
        checkpoint_dir: Directory for checkpoints  
        resume_from: Specific checkpoint path to resume from
        
    Returns:
        Initialized Trainer instance
    """
    return Trainer.create(config=config, checkpoint_dir=checkpoint_dir, resume_from=resume_from)

def train_model(num_iterations: int = 100, 
               checkpoint_dir: str = "data/checkpoints",
               resume_from: Optional[str] = None) -> Trainer:
    """
    Train model with default configuration
    
    Args:
        num_iterations: Number of training iterations
        checkpoint_dir: Directory for checkpoints
        resume_from: Specific checkpoint path to resume from
        
    Returns:
        Trained Trainer instance
    """
    trainer = create_trainer(checkpoint_dir=checkpoint_dir, resume_from=resume_from)
    trainer.train(num_iterations)
    return trainer