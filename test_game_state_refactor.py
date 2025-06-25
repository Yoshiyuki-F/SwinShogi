#!/usr/bin/env python3
"""
Test refactored game state management
"""

import sys
import os
import jax
import logging

# Add project root to path
sys.path.insert(0, os.path.abspath('.'))

from src.shogi.shogi_game import ShogiGame
from src.utils.action_utils import action_to_index, convert_to_full_action_vector, filter_valid_actions
from src.rl.mcts import MCTS
from src.model.shogi_model import create_swin_shogi_model
from config.default_config import get_model_config, get_mcts_config

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_game_state_properties():
    """Test ShogiGame state properties"""
    logger.info("Testing ShogiGame state properties...")
    
    game = ShogiGame()
    
    # Test game_state property
    state = game.game_state
    logger.info(f"Game state keys: {list(state.keys())}")
    
    required_keys = ['board', 'hands', 'turn', 'is_terminal', 'is_checkmate']
    for key in required_keys:
        assert key in state, f"Missing key: {key}"
    
    # Test is_terminal property
    assert not game.is_terminal, "Initial position should not be terminal"
    
    # Test clone_state
    cloned_state = game.clone_state()
    assert cloned_state == state, "Cloned state should match original"
    
    # Test that modifications to cloned state don't affect original
    cloned_state['board'][0][0] = None
    assert game.board[0][0] is not None, "Original board should be unchanged"
    
    logger.info("✅ Game state properties test passed!")

def test_action_utils():
    """Test action utilities"""
    logger.info("Testing action utilities...")
    
    # Test action_to_index
    moves = ["7g7f", "3c3d", "8h2b+"]
    for move in moves:
        idx = action_to_index(move)
        assert 0 <= idx < 2187, f"Invalid action index: {idx}"
    
    # Test convert_to_full_action_vector
    action_probs = {"7g7f": 0.5, "3c3d": 0.3, "8h2b+": 0.2}
    full_vector = convert_to_full_action_vector(action_probs)
    assert full_vector.shape == (2187,), f"Wrong vector shape: {full_vector.shape}"
    assert abs(full_vector.sum() - 1.0) < 1e-5, f"Vector doesn't sum to 1: {full_vector.sum()}"
    
    # Test filter_valid_actions
    model_probs = {i: 0.1 for i in range(10)}  # Fake model output
    valid_moves = ["7g7f", "3c3d"]
    filtered = filter_valid_actions(model_probs, valid_moves)
    
    assert len(filtered) == len(valid_moves), "Filtered probs should match valid moves"
    assert abs(sum(filtered.values()) - 1.0) < 1e-5, "Filtered probs should sum to 1"
    
    logger.info("✅ Action utilities test passed!")

def test_mcts_integration():
    """Test MCTS with refactored game state"""
    logger.info("Testing MCTS integration...")
    
    # Create model
    rng = jax.random.PRNGKey(42)
    model_config = get_model_config()
    model, params = create_swin_shogi_model(rng, model_config)
    
    # Create MCTS with minimal config
    mcts_config = get_mcts_config()
    mcts_config.n_simulations = 2  # Minimal for testing
    mcts = MCTS(model, params, mcts_config)
    
    # Test with game state
    game = ShogiGame()
    game_state = game.game_state
    
    logger.info("Running MCTS search...")
    action_probs, root_node = mcts.search(game_state)
    
    assert len(action_probs) > 0, "MCTS should return some action probabilities"
    assert abs(sum(action_probs.values()) - 1.0) < 1e-5, "Action probs should sum to 1"
    
    logger.info("✅ MCTS integration test passed!")

def test_apply_action():
    """Test action application"""
    logger.info("Testing action application...")
    
    game = ShogiGame()
    initial_state = game.game_state
    
    # Get valid moves
    valid_moves = game.get_valid_moves()
    assert len(valid_moves) > 0, "Should have valid moves in initial position"
    
    # Apply first valid move
    first_move = valid_moves[0]
    new_state = game.apply_action_to_state(first_move, initial_state)
    
    # Verify state changed
    assert new_state != initial_state, "State should change after move"
    assert new_state['turn'] != initial_state['turn'], "Turn should change"
    
    # Verify original game unchanged
    assert game.game_state == initial_state, "Original game should be unchanged"
    
    logger.info("✅ Action application test passed!")

if __name__ == "__main__":
    try:
        test_game_state_properties()
        test_action_utils()
        test_apply_action()
        test_mcts_integration()
        
        logger.info("🎉 All game state refactor tests passed!")
        
    except Exception as e:
        logger.error(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)