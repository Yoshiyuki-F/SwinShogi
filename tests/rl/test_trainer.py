"""
Test cases for SwinShogi Trainer
"""

import unittest
import jax
import jax.numpy as jnp
import tempfile
import shutil
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from src.rl.trainer import Trainer, TrainingExample, ReplayBuffer, TrainState
from src.model.shogi_model import create_swin_shogi_model
from config.default_config import get_model_config, get_training_config


class TestTrainingExample(unittest.TestCase):
    """Test TrainingExample data structure"""
    
    def test_training_example_creation(self):
        """Test TrainingExample initialization"""
        board_state = jnp.zeros((9, 9, 2))
        feature_vector = jnp.zeros(15)
        action_probs = jnp.zeros(2187)
        value = 0.5
        
        example = TrainingExample(
            board_state=board_state,
            feature_vector=feature_vector,
            action_probs=action_probs,
            value=value
        )
        
        self.assertEqual(example.board_state.shape, (9, 9, 2))
        self.assertEqual(example.feature_vector.shape, (15,))
        self.assertEqual(example.action_probs.shape, (2187,))
        self.assertEqual(example.value, 0.5)


class TestReplayBuffer(unittest.TestCase):
    """Test ReplayBuffer functionality"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.buffer = ReplayBuffer(max_size=5)
        
    def create_example(self, value: float = 0.0) -> TrainingExample:
        """Create test training example"""
        return TrainingExample(
            board_state=jnp.zeros((9, 9, 2)),
            feature_vector=jnp.zeros(15),
            action_probs=jnp.zeros(2187),
            value=value
        )
    
    def test_buffer_initialization(self):
        """Test buffer initialization"""
        self.assertEqual(len(self.buffer), 0)
        self.assertEqual(self.buffer.max_size, 5)
    
    def test_add_example(self):
        """Test adding single example"""
        example = self.create_example()
        self.buffer.add(example)
        
        self.assertEqual(len(self.buffer), 1)
    
    def test_add_batch(self):
        """Test adding batch of examples"""
        examples = [self.create_example(i) for i in range(3)]
        self.buffer.add_batch(examples)
        
        self.assertEqual(len(self.buffer), 3)
    
    def test_buffer_overflow(self):
        """Test buffer size limit"""
        examples = [self.create_example(i) for i in range(10)]
        self.buffer.add_batch(examples)
        
        self.assertEqual(len(self.buffer), 5)  # Max size
    
    def test_sample(self):
        """Test sampling from buffer"""
        examples = [self.create_example(i) for i in range(3)]
        self.buffer.add_batch(examples)
        
        sample = self.buffer.sample(2)
        self.assertEqual(len(sample), 2)
        
        # Sample more than available
        large_sample = self.buffer.sample(10)
        self.assertEqual(len(large_sample), 3)
    
    def test_clear(self):
        """Test clearing buffer"""
        examples = [self.create_example(i) for i in range(3)]
        self.buffer.add_batch(examples)
        
        self.buffer.clear()
        self.assertEqual(len(self.buffer), 0)


class TestTrainState(unittest.TestCase):
    """Test TrainState functionality"""
    
    def test_train_state_creation(self):
        """Test TrainState creation"""
        rng = jax.random.PRNGKey(42)
        model_config = get_model_config()
        model, params = create_swin_shogi_model(rng, model_config)
        
        # Create dummy optimizer state
        optimizer_state = {}
        
        train_state = TrainState(params, optimizer_state, step=10)
        
        self.assertIsNotNone(train_state.params)
        self.assertEqual(train_state.step, 10)
    
    def test_train_state_update(self):
        """Test TrainState update"""
        rng = jax.random.PRNGKey(42)
        model_config = get_model_config()
        model, params = create_swin_shogi_model(rng, model_config)
        
        optimizer_state = {}
        train_state = TrainState(params, optimizer_state, step=5)
        
        new_params = params  # Same params for test
        new_optimizer_state = {}
        
        updated_state = train_state.update(new_params, new_optimizer_state)
        
        self.assertEqual(updated_state.step, 6)


class TestTrainer(unittest.TestCase):
    """Test Trainer functionality"""
    
    @classmethod
    def setUpClass(cls):
        """Set up test fixtures"""
        cls.temp_dir = tempfile.mkdtemp()
        cls.trainer_config = get_training_config()
        # Use minimal settings for testing
        cls.trainer_config.num_iterations = 1
        cls.trainer_config.num_episodes = 1
        cls.trainer_config.train_steps_per_iteration = 1
        cls.trainer_config.mcts_simulations = 5
        cls.trainer_config.batch_size = 1
        cls.trainer_config.min_buffer_size = 1
    
    @classmethod
    def tearDownClass(cls):
        """Clean up test fixtures"""
        shutil.rmtree(cls.temp_dir)
    
    def test_trainer_initialization(self):
        """Test trainer initialization"""
        trainer = Trainer(config=self.trainer_config, save_dir=self.temp_dir)
        
        self.assertIsNotNone(trainer.model)
        self.assertIsNotNone(trainer.actor_critic)
        self.assertIsNotNone(trainer.mcts)
        self.assertIsInstance(trainer.replay_buffer, ReplayBuffer)
        self.assertIsInstance(trainer.train_state, TrainState)
    
    def test_optimizer_creation(self):
        """Test optimizer creation"""
        trainer = Trainer(config=self.trainer_config, save_dir=self.temp_dir)
        optimizer = trainer._create_optimizer()
        
        self.assertIsNotNone(optimizer)
    
    def test_train_on_batch_empty(self):
        """Test training on empty batch"""
        trainer = Trainer(config=self.trainer_config, save_dir=self.temp_dir)
        
        losses = trainer.train_on_batch([])
        
        self.assertEqual(losses['total_loss'], 0.0)
        self.assertEqual(losses['policy_loss'], 0.0)
        self.assertEqual(losses['value_loss'], 0.0)
        self.assertEqual(losses['entropy_loss'], 0.0)
    
    def test_train_on_batch_with_data(self):
        """Test training on batch with data"""
        trainer = Trainer(config=self.trainer_config, save_dir=self.temp_dir)
        
        # Create test examples
        examples = [
            TrainingExample(
                board_state=jnp.zeros((9, 9, 2)),
                feature_vector=jnp.zeros(15),
                action_probs=jnp.ones(2187) / 2187,  # Uniform distribution
                value=0.5
            )
        ]
        
        losses = trainer.train_on_batch(examples)
        
        self.assertIsInstance(losses['total_loss'], float)
        self.assertIsInstance(losses['policy_loss'], float)
        self.assertIsInstance(losses['value_loss'], float)
        self.assertIsInstance(losses['entropy_loss'], float)
    
    def test_checkpoint_save_load(self):
        """Test checkpoint saving and loading"""
        trainer = Trainer(config=self.trainer_config, save_dir=self.temp_dir)
        
        # Save checkpoint
        checkpoint_path = os.path.join(self.temp_dir, "test_checkpoint.pkl")
        trainer.save_checkpoint(1, checkpoint_path)
        
        self.assertTrue(os.path.exists(checkpoint_path))
        
        # Create new trainer and load checkpoint
        trainer2 = Trainer(config=self.trainer_config, save_dir=self.temp_dir)
        trainer2.load_checkpoint(checkpoint_path)
        
        # Check that stats are loaded
        self.assertEqual(trainer2.training_stats['iteration'], trainer.training_stats['iteration'])
    
    def test_training_stats_update(self):
        """Test training statistics update"""
        trainer = Trainer(config=self.trainer_config, save_dir=self.temp_dir)
        
        initial_iteration = trainer.training_stats['iteration']
        
        # Simulate training iteration
        stats = trainer.train_iteration(1)
        
        self.assertEqual(stats['iteration'], initial_iteration + 1)
        self.assertIn('total_games', stats)
        self.assertIn('buffer_size', stats)
    
    def test_convenience_functions(self):
        """Test convenience functions"""
        from src.rl.trainer import create_trainer
        
        trainer = create_trainer(config=self.trainer_config, save_dir=self.temp_dir)
        
        self.assertIsInstance(trainer, Trainer)
        self.assertEqual(trainer.config, self.trainer_config)


if __name__ == '__main__':
    unittest.main(verbosity=2)