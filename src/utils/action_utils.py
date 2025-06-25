"""
Action encoding/decoding utilities for SwinShogi
"""

import jax.numpy as jnp
from typing import Dict, List
from src.shogi.board_encoder import encode_move


def action_to_index(action: str) -> int:
    """
    Convert USI action string to action index
    
    Args:
        action: USI format move string
        
    Returns:
        Action index (0-2186)
    """
    try:
        # Use proper board encoder if available
        _, _, action_idx = encode_move(action)
        return action_idx
    except:
        # Fallback to hash if encoding fails
        return abs(hash(action)) % 2187


def convert_to_full_action_vector(action_probs: Dict[str, float]) -> jnp.ndarray:
    """
    Convert action probabilities dict to full action vector
    
    Args:
        action_probs: Dictionary mapping actions to probabilities
        
    Returns:
        Full action probability vector of size 2187
    """
    full_probs = jnp.zeros(2187)
    
    for action_str, prob in action_probs.items():
        # Convert action string to index
        action_idx = action_to_index(action_str)
        full_probs = full_probs.at[action_idx].set(prob)
    
    return full_probs


def index_to_action(action_idx: int) -> str:
    """
    Convert action index to USI action string
    
    Args:
        action_idx: Action index (0-2186)
        
    Returns:
        USI format move string
    """
    # This should implement proper decoding
    # For now, this is a placeholder
    # TODO: Implement proper index to USI conversion
    return f"action_{action_idx}"


def validate_action_vector(action_vector: jnp.ndarray) -> bool:
    """
    Validate action probability vector
    
    Args:
        action_vector: Action probabilities of size 2187
        
    Returns:
        True if valid (sums to ~1.0, all non-negative)
    """
    if action_vector.shape[0] != 2187:
        return False
    
    if jnp.any(action_vector < 0):
        return False
        
    total_prob = jnp.sum(action_vector)
    if not jnp.isclose(total_prob, 1.0, rtol=1e-5):
        return False
    
    return True


def filter_valid_actions(action_probs: Dict[str, float], valid_moves: List[str]) -> Dict[str, float]:
    """
    Filter action probabilities to only include valid moves
    
    Args:
        action_probs: Action probabilities from model
        valid_moves: List of valid moves in current position
        
    Returns:
        Filtered and normalized action probabilities
    """
    filtered_probs = {}
    
    for move in valid_moves:
        action_idx = action_to_index(move)
        if action_idx in action_probs:
            filtered_probs[move] = action_probs[action_idx]
        else:
            filtered_probs[move] = 1e-8  # Small prior for unseen moves
    
    # Normalize probabilities
    total_prob = sum(filtered_probs.values())
    if total_prob > 0:
        filtered_probs = {move: prob / total_prob for move, prob in filtered_probs.items()}
    else:
        # Uniform distribution if no valid priors
        uniform_prob = 1.0 / len(valid_moves)
        filtered_probs = {move: uniform_prob for move in valid_moves}
    
    return filtered_probs