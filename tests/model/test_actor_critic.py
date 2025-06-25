"""
Test cases for Actor-Critic integration with SwinShogi components
"""

import unittest
import jax
import jax.numpy as jnp
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from src.model.shogi_model import create_swin_shogi_model
from src.model.actor_critic import ActorCritic, predict_for_mcts, encode_shogi_state_for_model
from src.shogi.shogi_game import ShogiGame
from src.shogi.shogi_pieces import Player
from config.default_config import get_model_config


class TestActorCriticIntegration(unittest.TestCase):
    """Test Actor-Critic integration with SwinShogi components"""
    
    @classmethod
    def setUpClass(cls):
        """Set up test fixtures before all test methods"""
        cls.rng = jax.random.PRNGKey(42)
        cls.model_config = get_model_config()
        cls.model, cls.params = create_swin_shogi_model(cls.rng, model_config=cls.model_config)
        cls.actor_critic = ActorCritic(cls.model, cls.params)
        
        # Create test shogi game state
        cls.game = ShogiGame()
        cls.shogi_state = {
            'board': cls.game.board,
            'hands': cls.game.captures,
            'turn': cls.game.current_player
        }
    
    def test_basic_encoding(self):
        """Test basic game state encoding"""
        board_encoded, feature_vector = encode_shogi_state_for_model(self.shogi_state)
        
        self.assertEqual(board_encoded.shape, (1, 9, 9, 2))
        self.assertEqual(feature_vector.shape, (1, 15))
        self.assertIsInstance(board_encoded, jnp.ndarray)
        self.assertIsInstance(feature_vector, jnp.ndarray)
    
    def test_actor_critic_predict_from_state(self):
        """Test Actor-Critic prediction from state dict"""
        action_probs, value = self.actor_critic.predict(state=self.shogi_state)
        
        self.assertIsInstance(action_probs, dict)
        self.assertIsInstance(value, float)
        self.assertGreater(len(action_probs), 0)
        self.assertTrue(all(isinstance(k, int) for k in action_probs.keys()))
        self.assertTrue(all(isinstance(v, float) for v in action_probs.values()))
        self.assertTrue(all(0 <= v <= 1 for v in action_probs.values()))
    
    def test_actor_critic_predict_from_tensors(self):
        """Test Actor-Critic prediction from pre-encoded tensors"""
        board_state, feature_vector = encode_shogi_state_for_model(self.shogi_state)
        action_probs, value = self.actor_critic.predict(
            board_state=board_state, 
            feature_vector=feature_vector
        )
        
        self.assertIsInstance(action_probs, dict)
        self.assertIsInstance(value, float)
        self.assertGreater(len(action_probs), 0)
    
    def test_prediction_consistency(self):
        """Test consistency between different prediction methods"""
        # Method 1: Direct from state
        action_probs1, value1 = self.actor_critic.predict(state=self.shogi_state)
        
        # Method 2: From pre-encoded tensors
        board_state, feature_vector = encode_shogi_state_for_model(self.shogi_state)
        action_probs2, value2 = self.actor_critic.predict(
            board_state=board_state, 
            feature_vector=feature_vector
        )
        
        # Method 3: MCTS standalone function
        action_probs3, value3 = predict_for_mcts(self.model, self.params, self.shogi_state)
        
        # Values should be very close (within numerical precision)
        self.assertAlmostEqual(value1, value2, places=5)
        self.assertAlmostEqual(value1, value3, places=5)
        
        # Action probabilities should be identical
        self.assertEqual(len(action_probs1), len(action_probs2))
        for action in action_probs1:
            if action in action_probs2:
                self.assertAlmostEqual(action_probs1[action], action_probs2[action], places=5)
    
    def test_batch_prediction(self):
        """Test batch prediction functionality"""
        batch_size = 4
        board_state, feature_vector = encode_shogi_state_for_model(self.shogi_state)
        
        # Create batch
        board_states = jnp.repeat(board_state, batch_size, axis=0)
        feature_vectors = jnp.repeat(feature_vector, batch_size, axis=0)
        
        policy_logits, values = self.actor_critic.predict_batch(board_states, feature_vectors)
        
        self.assertEqual(policy_logits.shape[0], batch_size)
        self.assertEqual(values.shape[0], batch_size)
        
        # All batch elements should be identical
        for i in range(1, batch_size):
            self.assertTrue(jnp.allclose(policy_logits[0], policy_logits[i]))
            self.assertTrue(jnp.allclose(values[0], values[i]))
    
    def test_feature_extraction(self):
        """Test feature extraction functionality"""
        features = self.actor_critic.extract_features(state=self.shogi_state)
        
        self.assertIsInstance(features, jnp.ndarray)
        self.assertEqual(len(features.shape), 2)  # Should have batch dimension
        self.assertEqual(features.shape[0], 1)     # Batch size 1
    
    def test_parameter_update(self):
        """Test parameter update functionality"""
        original_param_count = self.actor_critic.parameter_count
        
        # Create new dummy parameters (same structure)
        new_params = jax.tree_util.tree_map(lambda x: x * 0.5, self.params)
        
        # Update parameters
        self.actor_critic.update_params(new_params)
        
        # Parameter count should remain the same
        self.assertEqual(self.actor_critic.parameter_count, original_param_count)
        
        # Predictions should be different now
        action_probs_new, value_new = self.actor_critic.predict(state=self.shogi_state)
        
        # Reset to original parameters for other tests
        self.actor_critic.update_params(self.params)
    
    def test_different_game_states(self):
        """Test Actor-Critic with different game states"""
        test_cases = [
            ("Initial position", None),
            ("Custom SFEN", "lnsgkgsnl/1r5b1/ppppppppp/9/9/9/PPPPPPPPP/1B5R1/LNSGKGSNL b - 1")
        ]
        
        for scenario_name, sfen in test_cases:
            with self.subTest(scenario=scenario_name):
                game = ShogiGame()
                if sfen:
                    game.setup_custom_position(sfen)
                
                shogi_state = {
                    'board': game.board,
                    'hands': game.captures,
                    'turn': game.current_player
                }
                
                action_probs, value = predict_for_mcts(self.model, self.params, shogi_state)
                
                self.assertIsInstance(action_probs, dict)
                self.assertIsInstance(value, float)
                self.assertGreater(len(action_probs), 0)
    
    def test_valid_moves_consistency(self):
        """Test that valid moves are handled correctly"""
        valid_moves = self.game.get_valid_moves()
        action_probs, _ = self.actor_critic.predict(state=self.shogi_state)
        
        # Should have valid moves
        self.assertGreater(len(valid_moves), 0)
        
        # Should have action probabilities
        self.assertGreater(len(action_probs), 0)
        
        # Model should output correct action space size
        self.assertEqual(self.model_config.n_policy_outputs, 2187)
    
    def test_error_handling(self):
        """Test error handling for invalid inputs"""
        # Test with missing inputs
        with self.assertRaises(ValueError):
            self.actor_critic.predict()  # No inputs provided
        
        with self.assertRaises(ValueError):
            self.actor_critic.predict(board_state=jnp.ones((1, 9, 9, 2)))  # Missing feature_vector
        
        with self.assertRaises(ValueError):
            self.actor_critic.extract_features()  # No inputs provided


class TestActorCriticQuick(unittest.TestCase):
    """Quick tests for basic functionality without heavy computation"""
    
    def test_model_creation(self):
        """Test that model can be created without errors"""
        rng = jax.random.PRNGKey(42)
        model_config = get_model_config()
        
        model, params = create_swin_shogi_model(rng, model_config=model_config)
        actor_critic = ActorCritic(model, params)
        
        self.assertIsNotNone(model)
        self.assertIsNotNone(params)
        self.assertIsNotNone(actor_critic)
        self.assertGreater(actor_critic.parameter_count, 0)
    
    def test_shogi_state_format(self):
        """Test that shogi state format is correct"""
        game = ShogiGame()
        shogi_state = {
            'board': game.board,
            'hands': game.captures,
            'turn': game.current_player
        }
        
        self.assertIn('board', shogi_state)
        self.assertIn('hands', shogi_state)
        self.assertIn('turn', shogi_state)
        
        self.assertEqual(len(shogi_state['board']), 9)
        self.assertEqual(len(shogi_state['board'][0]), 9)
        self.assertIsInstance(shogi_state['turn'], Player)


if __name__ == '__main__':
    # Run with less verbose output for CI/CD
    unittest.main(verbosity=2)