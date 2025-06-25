"""
Test cases for SwinShogi Self-Play functionality
"""

import unittest
import jax
import jax.numpy as jnp
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from src.rl.self_play import SelfPlay, SelfPlayExample, SelfPlayResult, create_self_play, generate_training_data
from src.model.shogi_model import create_swin_shogi_model
from config.default_config import get_model_config, get_mcts_config


class TestSelfPlayExample(unittest.TestCase):
    """Test SelfPlayExample data structure"""
    
    def test_self_play_example_creation(self):
        """Test SelfPlayExample initialization"""
        board_state = jnp.zeros((9, 9, 2))
        feature_vector = jnp.zeros(15)
        action_probs = jnp.zeros(2187)
        player = 0
        
        example = SelfPlayExample(
            board_state=board_state,
            feature_vector=feature_vector,
            action_probs=action_probs,
            player=player
        )
        
        self.assertEqual(example.board_state.shape, (9, 9, 2))
        self.assertEqual(example.feature_vector.shape, (15,))
        self.assertEqual(example.action_probs.shape, (2187,))
        self.assertEqual(example.player, 0)


class TestSelfPlayResult(unittest.TestCase):
    """Test SelfPlayResult data structure"""
    
    def test_self_play_result_creation(self):
        """Test SelfPlayResult initialization"""
        examples = [
            SelfPlayExample(
                board_state=jnp.zeros((9, 9, 2)),
                feature_vector=jnp.zeros(15),
                action_probs=jnp.zeros(2187),
                player=0
            )
        ]
        
        result = SelfPlayResult(
            examples=examples,
            winner=1,
            game_length=50,
            final_score={0: -1.0, 1: 1.0}
        )
        
        self.assertEqual(len(result.examples), 1)
        self.assertEqual(result.winner, 1)
        self.assertEqual(result.game_length, 50)
        self.assertEqual(result.final_score[0], -1.0)
        self.assertEqual(result.final_score[1], 1.0)


class TestSelfPlay(unittest.TestCase):
    """Test SelfPlay functionality"""
    
    @classmethod
    def setUpClass(cls):
        """Set up test fixtures"""
        cls.rng = jax.random.PRNGKey(42)
        cls.model_config = get_model_config()
        cls.model, cls.params = create_swin_shogi_model(cls.rng, model_config=cls.model_config)
        cls.mcts_config = get_mcts_config()
        # Use minimal settings for testing
        cls.mcts_config.n_simulations = 3
    
    def test_self_play_initialization(self):
        """Test SelfPlay initialization"""
        self_play = SelfPlay(
            self.model, 
            self.params, 
            config=self.mcts_config,
            max_moves=10,
            temperature_threshold=5
        )
        
        self.assertIsNotNone(self_play.model)
        self.assertIsNotNone(self_play.params)
        self.assertIsNotNone(self_play.mcts)
        self.assertEqual(self_play.max_moves, 10)
        self.assertEqual(self_play.temperature_threshold, 5)
        self.assertEqual(self_play.games_played, 0)
    
    def test_action_to_index_conversion(self):
        """Test action to index conversion"""
        self_play = SelfPlay(self.model, self.params, config=self.mcts_config)
        
        action = "7g7f"
        index = self_play._action_to_index(action)
        
        self.assertIsInstance(index, int)
        self.assertGreaterEqual(index, 0)
        self.assertLess(index, 2187)
    
    def test_convert_to_full_action_vector(self):
        """Test converting action probabilities to full vector"""
        self_play = SelfPlay(self.model, self.params, config=self.mcts_config)
        
        action_probs = {
            "7g7f": 0.6,
            "2g2f": 0.4
        }
        
        full_vector = self_play._convert_to_full_action_vector(action_probs)
        
        self.assertEqual(full_vector.shape, (2187,))
        self.assertAlmostEqual(float(jnp.sum(full_vector)), 1.0, places=5)
    
    def test_calculate_final_scores(self):
        """Test final score calculation"""
        self_play = SelfPlay(self.model, self.params, config=self.mcts_config)
        
        # Test player 0 win
        scores_0_wins = self_play._calculate_final_scores(0)
        self.assertEqual(scores_0_wins[0], 1.0)
        self.assertEqual(scores_0_wins[1], -1.0)
        
        # Test player 1 win
        scores_1_wins = self_play._calculate_final_scores(1)
        self.assertEqual(scores_1_wins[0], -1.0)
        self.assertEqual(scores_1_wins[1], 1.0)
        
        # Test draw
        scores_draw = self_play._calculate_final_scores(None)
        self.assertEqual(scores_draw[0], 0.0)
        self.assertEqual(scores_draw[1], 0.0)
    
    def test_statistics_tracking(self):
        """Test statistics tracking"""
        self_play = SelfPlay(self.model, self.params, config=self.mcts_config)
        
        # Initial state
        stats = self_play.get_statistics()
        self.assertEqual(stats['games_played'], 0)
        
        # Reset statistics
        self_play.reset_statistics()
        self.assertEqual(self_play.games_played, 0)
        self.assertEqual(self_play.total_moves, 0)
    
    def test_model_params_update(self):
        """Test updating model parameters"""
        self_play = SelfPlay(self.model, self.params, config=self.mcts_config)
        
        # Create new parameters (same structure for test)
        rng = jax.random.PRNGKey(123)
        _, new_params = create_swin_shogi_model(rng, model_config=self.model_config)
        
        # Update parameters
        self_play.update_model_params(new_params)
        
        # Check that parameters are updated
        self.assertIs(self_play.params, new_params)
        self.assertIs(self_play.mcts.params, new_params)
    
    def test_results_to_training_examples(self):
        """Test converting results to training examples"""
        self_play = SelfPlay(self.model, self.params, config=self.mcts_config)
        
        # Create mock results
        examples = [
            SelfPlayExample(
                board_state=jnp.zeros((9, 9, 2)),
                feature_vector=jnp.zeros(15),
                action_probs=jnp.zeros(2187),
                player=0
            ),
            SelfPlayExample(
                board_state=jnp.zeros((9, 9, 2)),
                feature_vector=jnp.zeros(15),
                action_probs=jnp.zeros(2187),
                player=1
            )
        ]
        
        result = SelfPlayResult(
            examples=examples,
            winner=0,  # Player 0 wins
            game_length=2,
            final_score={0: 1.0, 1: -1.0}
        )
        
        training_examples = self_play.results_to_training_examples([result])
        
        self.assertEqual(len(training_examples), 2)
        
        # Check values assignment
        self.assertEqual(training_examples[0]['value'], 1.0)   # Player 0 wins
        self.assertEqual(training_examples[1]['value'], -1.0)  # Player 1 loses
        
        # Check structure
        for example in training_examples:
            self.assertIn('board_state', example)
            self.assertIn('feature_vector', example)
            self.assertIn('action_probs', example)
            self.assertIn('value', example)
            self.assertIn('player', example)


class TestSelfPlayUtilities(unittest.TestCase):
    """Test SelfPlay utility functions"""
    
    @classmethod
    def setUpClass(cls):
        """Set up test fixtures"""
        cls.rng = jax.random.PRNGKey(42)
        cls.model_config = get_model_config()
        cls.model, cls.params = create_swin_shogi_model(cls.rng, model_config=cls.model_config)
    
    def test_create_self_play_function(self):
        """Test create_self_play utility function"""
        self_play = create_self_play(
            self.model, 
            self.params, 
            n_simulations=5,
            max_moves=20,
            temperature_init=0.8
        )
        
        self.assertIsInstance(self_play, SelfPlay)
        self.assertEqual(self_play.mcts_config.n_simulations, 5)
        self.assertEqual(self_play.max_moves, 20)
        self.assertEqual(self_play.temperature_init, 0.8)
    
    def test_generate_training_data_function(self):
        """Test generate_training_data utility function"""
        # Use minimal configuration for quick test
        training_examples, statistics = generate_training_data(
            self.model,
            self.params,
            num_games=1,
            n_simulations=3,
            max_moves=5
        )
        
        self.assertIsInstance(training_examples, list)
        self.assertIsInstance(statistics, dict)
        self.assertIn('games_played', statistics)
        self.assertEqual(statistics['games_played'], 1)


class TestSelfPlayIntegration(unittest.TestCase):
    """Test SelfPlay integration with other components"""
    
    @classmethod
    def setUpClass(cls):
        """Set up test fixtures"""
        cls.rng = jax.random.PRNGKey(42)
        cls.model_config = get_model_config()
        cls.model, cls.params = create_swin_shogi_model(cls.rng, model_config=cls.model_config)
        
        # Minimal configuration for fast testing
        cls.mcts_config = get_mcts_config()
        cls.mcts_config.n_simulations = 2
    
    def test_single_game_generation(self):
        """Test playing a single game (with very short limits)"""
        self_play = SelfPlay(
            self.model, 
            self.params, 
            config=self.mcts_config,
            max_moves=3  # Very short game for testing
        )
        
        result = self_play.play_game(verbose=False)
        
        self.assertIsInstance(result, SelfPlayResult)
        self.assertIsInstance(result.examples, list)
        self.assertIsInstance(result.game_length, int)
        self.assertIn(result.winner, [0, 1, None])
        self.assertGreaterEqual(result.game_length, 0)
        self.assertLessEqual(result.game_length, 3)  # Should respect max_moves
    
    def test_multiple_games_generation(self):
        """Test generating multiple games"""
        self_play = SelfPlay(
            self.model, 
            self.params, 
            config=self.mcts_config,
            max_moves=2  # Very short games
        )
        
        results = self_play.generate_games(2, verbose=False)
        
        self.assertEqual(len(results), 2)
        self.assertEqual(self_play.games_played, 2)
        
        # Check statistics
        stats = self_play.get_statistics()
        self.assertEqual(stats['games_played'], 2)
        self.assertGreaterEqual(stats['total_moves'], 0)


if __name__ == '__main__':
    unittest.main(verbosity=2)