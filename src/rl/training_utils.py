"""
Training utility functions for converting game results to training data
"""

import numpy as np
import logging
from typing import List, Dict, Any

logger = logging.getLogger(__name__)


def game_results_to_training_examples(results: List[Any]) -> List[Dict[str, Any]]:
    """
    Convert game results to training examples using pre-computed evaluations
    
    Args:
        results: List of game result objects with 'examples' attribute
        
    Returns:
        List of training examples compatible with trainer
    """
    training_examples = []
    
    for result in results:
        for example in result.examples:
            # Use pre-computed SwinTransformer evaluation from position_value
            training_example = {
                'board_state': example.board_state,
                'feature_vector': example.feature_vector,
                'action_probs': example.action_probs,
                'value': example.position_value,  # Use SwinTransformer evaluation
                'player': example.player
            }
            training_examples.append(training_example)
    
    return training_examples


def log_game_generation_summary(results: List[Any], total_time: float, 
                               generation_type: str = "game"):
    """
    Log summary of game generation
    
    Args:
        results: List of game result objects
        total_time: Total time taken for generation
        generation_type: Type of generation (e.g., "self-play", "ai-opponent")
    """
    total_games = len(results)
    total_examples = sum(len(result.examples) for result in results)
    avg_game_length = np.mean([result.game_length for result in results])
    
    # Count outcomes
    wins_0 = sum(1 for r in results if r.winner == 0)
    wins_1 = sum(1 for r in results if r.winner == 1)
    draws = sum(1 for r in results if r.winner is None)
    
    logger.info(f"{generation_type.title()} generation completed:")
    logger.info(f"  Games: {total_games}")
    logger.info(f"  Training examples: {total_examples}")
    logger.info(f"  Average game length: {avg_game_length:.1f} moves")
    logger.info(f"  Outcomes: P0={wins_0}, P1={wins_1}, Draws={draws}")
    logger.info(f"  Total time: {total_time:.1f}s ({total_time/total_games:.1f}s/game)")


def create_training_example_from_state(board_state, feature_vector, action_probs, 
                                     position_value: float, player: int) -> Dict[str, Any]:
    """
    Create a standardized training example
    
    Args:
        board_state: Encoded board state
        feature_vector: Game feature vector
        action_probs: Action probability distribution
        position_value: Position evaluation value
        player: Player ID (0 or 1)
        
    Returns:
        Training example dictionary
    """
    return {
        'board_state': board_state,
        'feature_vector': feature_vector,
        'action_probs': action_probs,
        'value': position_value,
        'player': player
    }


def validate_training_examples(examples: List[Dict[str, Any]]) -> bool:
    """
    Validate that training examples have the expected format
    
    Args:
        examples: List of training examples
        
    Returns:
        True if all examples are valid, False otherwise
    """
    required_keys = {'board_state', 'feature_vector', 'action_probs', 'value', 'player'}
    
    for i, example in enumerate(examples):
        if not isinstance(example, dict):
            logger.error(f"Example {i} is not a dictionary")
            return False
        
        if not required_keys.issubset(example.keys()):
            missing_keys = required_keys - set(example.keys())
            logger.error(f"Example {i} missing keys: {missing_keys}")
            return False
        
        # Check data types and shapes
        try:
            board_state = example['board_state']
            if board_state.shape != (9, 9, 2):
                logger.error(f"Example {i} board_state has wrong shape: {board_state.shape}")
                return False
            
            feature_vector = example['feature_vector']
            if feature_vector.shape != (15,):
                logger.error(f"Example {i} feature_vector has wrong shape: {feature_vector.shape}")
                return False
            
            action_probs = example['action_probs']
            if action_probs.shape != (2187,):
                logger.error(f"Example {i} action_probs has wrong shape: {action_probs.shape}")
                return False
            
            if not isinstance(example['value'], (int, float)):
                logger.error(f"Example {i} value is not numeric: {type(example['value'])}")
                return False
            
            if example['player'] not in [0, 1]:
                logger.error(f"Example {i} player is not 0 or 1: {example['player']}")
                return False
                
        except Exception as e:
            logger.error(f"Example {i} validation error: {e}")
            return False
    
    return True


def batch_training_examples(examples: List[Dict[str, Any]], 
                          batch_size: int) -> List[List[Dict[str, Any]]]:
    """
    Batch training examples for efficient processing
    
    Args:
        examples: List of training examples
        batch_size: Size of each batch
        
    Returns:
        List of batches
    """
    batches = []
    for i in range(0, len(examples), batch_size):
        batch = examples[i:i + batch_size]
        batches.append(batch)
    
    return batches


def merge_training_examples(example_lists: List[List[Dict[str, Any]]]) -> List[Dict[str, Any]]:
    """
    Merge multiple lists of training examples
    
    Args:
        example_lists: List of training example lists
        
    Returns:
        Combined list of training examples
    """
    merged = []
    for example_list in example_lists:
        merged.extend(example_list)
    
    return merged