"""
Test cases for MCTS integration with SwinShogi components
"""

import unittest
import jax
import jax.numpy as jnp
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from src.rl.mcts import MCTS, MCTSNode, MCTSConfig, create_mcts, mcts_search
from src.model.shogi_model import create_swin_shogi_model
from src.shogi.shogi_game import ShogiGame
from config.default_config import get_model_config


class TestMCTSNode(unittest.TestCase):
    """Test MCTSNode functionality"""
    
    def test_node_initialization(self):
        """Test MCTSNode initialization"""
        node = MCTSNode(prior=0.5)
        
        self.assertEqual(node.visit_count, 0)
        self.assertEqual(node.value_sum, 0.0)
        self.assertEqual(node.prior, 0.5)
        self.assertIsNone(node.parent)
        self.assertEqual(len(node.children), 0)
        self.assertFalse(node.expanded)
        self.assertEqual(node.value, 0.0)
    
    def test_node_expansion(self):
        """Test node expansion"""
        node = MCTSNode()
        action_priors = {
            'move1': 0.3,
            'move2': 0.5,
            'move3': 0.2
        }
        
        node.expand(action_priors)
        
        self.assertTrue(node.is_expanded())
        self.assertEqual(len(node.children), 3)
        self.assertIn('move1', node.children)
        self.assertEqual(node.children['move1'].prior, 0.3)
        self.assertEqual(node.children['move1'].parent, node)
    
    def test_node_update(self):
        """Test node value updates"""
        node = MCTSNode()
        
        node.update(0.5)
        self.assertEqual(node.visit_count, 1)
        self.assertEqual(node.value_sum, 0.5)
        self.assertEqual(node.value, 0.5)
        
        node.update(-0.2)
        self.assertEqual(node.visit_count, 2)
        self.assertEqual(node.value_sum, 0.3)
        self.assertEqual(node.value, 0.15)
    
    def test_child_selection(self):
        """Test PUCT child selection"""
        node = MCTSNode()
        node.visit_count = 10
        
        action_priors = {
            'move1': 0.6,
            'move2': 0.4
        }
        node.expand(action_priors)
        
        # Update children with different values
        node.children['move1'].update(0.3)
        node.children['move2'].update(-0.1)
        
        action, child = node.select_child()
        
        self.assertIn(action, ['move1', 'move2'])
        self.assertIs(child, node.children[action])
    
    def test_exploration_noise(self):
        """Test Dirichlet noise addition"""
        node = MCTSNode()
        action_priors = {
            'move1': 0.5,
            'move2': 0.5
        }
        node.expand(action_priors)
        
        original_priors = {action: child.prior for action, child in node.children.items()}
        
        node.add_exploration_noise(noise_alpha=0.3, noise_epsilon=0.25)
        
        # Priors should be different after noise
        new_priors = {action: child.prior for action, child in node.children.items()}
        self.assertNotEqual(original_priors, new_priors)
        
        # Sum should still be close to 1
        total_prior = sum(new_priors.values())
        self.assertAlmostEqual(total_prior, 1.0, places=3)


class TestMCTSConfig(unittest.TestCase):
    """Test MCTS configuration"""
    
    def test_default_config(self):
        """Test default configuration values"""
        config = MCTSConfig()
        
        self.assertIsInstance(config.n_simulations, int)
        self.assertIsInstance(config.c_puct, float)
        self.assertIsInstance(config.max_depth, int)
        self.assertGreater(config.n_simulations, 0)
        self.assertGreater(config.max_depth, 0)
    
    def test_custom_config(self):
        """Test custom configuration"""
        config = MCTSConfig(
            n_simulations=50,
            c_puct=2.0,
            max_depth=100
        )
        
        self.assertEqual(config.n_simulations, 50)
        self.assertEqual(config.c_puct, 2.0)
        self.assertEqual(config.max_depth, 100)


class TestMCTSIntegration(unittest.TestCase):
    """Test MCTS integration with SwinShogi components"""
    
    @classmethod
    def setUpClass(cls):
        """Set up test fixtures"""
        cls.rng = jax.random.PRNGKey(42)
        cls.model_config = get_model_config()
        cls.model, cls.params = create_swin_shogi_model(cls.rng, model_config=cls.model_config)
        
        # Create test game state
        cls.game = ShogiGame()
        cls.game_state = {
            'board': cls.game.board,
            'hands': cls.game.captures,
            'turn': cls.game.current_player
        }
    
    def test_mcts_creation(self):
        """Test MCTS instance creation"""
        mcts = MCTS(self.model, self.params)
        
        self.assertIsNotNone(mcts.model)
        self.assertIsNotNone(mcts.params)
        self.assertIsInstance(mcts.config, MCTSConfig)
    
    def test_mcts_search_basic(self):
        """Test basic MCTS search functionality"""
        # Use minimal simulations for quick test
        config = MCTSConfig(n_simulations=5, max_depth=10)
        mcts = MCTS(self.model, self.params, config)
        
        action_probs, root_node = mcts.search(self.game_state)
        
        self.assertIsInstance(action_probs, dict)
        self.assertIsInstance(root_node, MCTSNode)
        self.assertTrue(root_node.is_expanded())
        self.assertGreater(len(action_probs), 0)
        
        # Check that probabilities are valid
        for action, prob in action_probs.items():
            self.assertIsInstance(action, str)
            self.assertIsInstance(prob, float)
            self.assertGreaterEqual(prob, 0.0)
            self.assertLessEqual(prob, 1.0)
    
    def test_mcts_with_valid_moves(self):
        """Test that MCTS only considers valid moves"""
        config = MCTSConfig(n_simulations=3, max_depth=5)
        mcts = MCTS(self.model, self.params, config)
        
        # Get valid moves from game
        valid_moves = self.game.get_valid_moves()
        
        action_probs, root_node = mcts.search(self.game_state)
        
        # All actions in search result should be valid moves
        for action in action_probs.keys():
            self.assertIn(action, valid_moves)
    
    def test_mcts_search_info(self):
        """Test search information reporting"""
        config = MCTSConfig(n_simulations=5, max_depth=10)
        mcts = MCTS(self.model, self.params, config)
        
        mcts.search(self.game_state)
        search_info = mcts.get_search_info()
        
        self.assertIn('nodes_created', search_info)
        self.assertIn('leaf_evaluations', search_info)
        self.assertIn('max_depth_reached', search_info)
        self.assertIn('simulations_run', search_info)
        
        self.assertGreaterEqual(search_info['nodes_created'], 0)
        self.assertGreaterEqual(search_info['leaf_evaluations'], 0)
        self.assertGreaterEqual(search_info['max_depth_reached'], 0)
        self.assertEqual(search_info['simulations_run'], 5)
    
    def test_action_selection(self):
        """Test action selection from probabilities"""
        config = MCTSConfig(n_simulations=3)
        mcts = MCTS(self.model, self.params, config)
        
        action_probs, _ = mcts.search(self.game_state)
        
        # Test deterministic selection
        action_det = mcts.select_action(action_probs, temperature=0.0)
        self.assertIn(action_det, action_probs.keys())
        
        # Test stochastic selection
        action_stoch = mcts.select_action(action_probs, temperature=1.0)
        self.assertIn(action_stoch, action_probs.keys())
    
    def test_node_reuse(self):
        """Test reusing search tree between searches"""
        config = MCTSConfig(n_simulations=3)
        mcts = MCTS(self.model, self.params, config)
        
        # First search
        action_probs1, root_node1 = mcts.search(self.game_state)
        visit_count1 = root_node1.visit_count
        
        # Second search with same root
        action_probs2, root_node2 = mcts.search(self.game_state, root_node1)
        visit_count2 = root_node2.visit_count
        
        # Root should be the same object
        self.assertIs(root_node1, root_node2)
        # Visit count should increase
        self.assertGreater(visit_count2, visit_count1)


class TestMCTSUtilities(unittest.TestCase):
    """Test MCTS utility functions"""
    
    @classmethod
    def setUpClass(cls):
        """Set up test fixtures"""
        cls.rng = jax.random.PRNGKey(42)
        cls.model_config = get_model_config()
        cls.model, cls.params = create_swin_shogi_model(cls.rng, model_config=cls.model_config)
        
        cls.game = ShogiGame()
        cls.game_state = {
            'board': cls.game.board,
            'hands': cls.game.captures,
            'turn': cls.game.current_player
        }
    
    def test_create_mcts_function(self):
        """Test create_mcts utility function"""
        mcts = create_mcts(self.model, self.params, n_simulations=10, max_depth=20)
        
        self.assertIsInstance(mcts, MCTS)
        self.assertEqual(mcts.config.n_simulations, 10)
        self.assertEqual(mcts.config.max_depth, 20)
    
    def test_mcts_search_function(self):
        """Test mcts_search utility function"""
        action_probs, search_info = mcts_search(
            self.model, 
            self.params, 
            self.game_state, 
            n_simulations=3
        )
        
        self.assertIsInstance(action_probs, dict)
        self.assertIsInstance(search_info, dict)
        self.assertGreater(len(action_probs), 0)
        self.assertEqual(search_info['simulations_run'], 3)


class TestMCTSGameStateIntegration(unittest.TestCase):
    """Test MCTS integration with game state management"""
    
    @classmethod
    def setUpClass(cls):
        """Set up test fixtures"""
        cls.rng = jax.random.PRNGKey(42)
        cls.model_config = get_model_config()
        cls.model, cls.params = create_swin_shogi_model(cls.rng, model_config=cls.model_config)
    
    def test_game_state_cloning(self):
        """Test game state cloning"""
        game = ShogiGame()
        game_state = {
            'board': game.board,
            'hands': game.captures,
            'turn': game.current_player
        }
        
        mcts = MCTS(self.model, self.params)
        cloned_state = mcts._clone_game_state(game_state)
        
        # Should be different objects
        self.assertIsNot(cloned_state, game_state)
        self.assertIsNot(cloned_state['board'], game_state['board'])
        
        # But same content
        self.assertEqual(cloned_state['turn'], game_state['turn'])
    
    def test_move_application(self):
        """Test applying moves to game state"""
        game = ShogiGame()
        game_state = {
            'board': game.board,
            'hands': game.captures,
            'turn': game.current_player
        }
        
        valid_moves = game.get_valid_moves()
        if valid_moves:
            move = valid_moves[0]
            
            mcts = MCTS(self.model, self.params)
            new_state = mcts._apply_action(game_state, move)
            
            # State should change
            self.assertNotEqual(new_state['turn'], game_state['turn'])
    
    def test_terminal_state_detection(self):
        """Test terminal state detection"""
        game = ShogiGame()
        game_state = {
            'board': game.board,
            'hands': game.captures,
            'turn': game.current_player
        }
        
        mcts = MCTS(self.model, self.params)
        
        # Initial position should not be terminal
        self.assertFalse(game_state['is_terminal'])


if __name__ == '__main__':
    unittest.main(verbosity=2)