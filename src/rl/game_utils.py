"""
Game utility functions that can be shared across different AI players and game modes
"""

import numpy as np
import logging
from typing import Optional, Dict, Any

logger = logging.getLogger(__name__)


def determine_game_outcome(game) -> Optional[int]:
    """
    Determine winner from game state
    
    Args:
        game: ShogiGame instance
        
    Returns:
        Winner player ID (0 or 1) or None for draw
    """
    if game.is_terminal:
        if game.is_checkmate():
            # Current player is in checkmate, so the other player wins
            return 1 - game.current_player.value
        else:
            # Draw (max moves reached or no valid moves)
            return None
    else:
        # Max moves reached
        return None


def calculate_game_scores(winner: Optional[int]) -> Dict[int, float]:
    """
    Calculate final scores for players
    
    Args:
        winner: Winner player ID or None for draw
        
    Returns:
        Dictionary mapping player ID to score
    """
    if winner is None:
        return {0: 0.0, 1: 0.0}  # Draw
    elif winner == 0:
        return {0: 1.0, 1: -1.0}  # Player 0 wins
    else:
        return {0: -1.0, 1: 1.0}  # Player 1 wins


def validate_and_apply_move(game, selected_action: str, verbose: bool = False) -> bool:
    """
    Validate and apply move with fallback to random valid move
    
    Args:
        game: ShogiGame instance
        selected_action: Proposed move in USI format
        verbose: Whether to log warnings
        
    Returns:
        True if move was applied successfully, False otherwise
    """
    # Try the selected action first
    success = game.move(selected_action)
    if not success:
        # If move fails, try a random valid move
        valid_moves = game.get_valid_moves()
        if valid_moves:
            random_move = np.random.choice(valid_moves)
            success = game.move(random_move)
            if verbose:
                logger.warning(f"Invalid move {selected_action}, used random move {random_move}")
        
        if not success:
            if verbose:
                logger.error(f"No valid moves available")
            return False
    
    return True


def get_temperature_for_move(move_count: int, threshold: int, 
                           init_temp: float, final_temp: float) -> float:
    """
    Calculate temperature based on move count
    
    Args:
        move_count: Current move number
        threshold: Move number to switch temperature
        init_temp: Initial temperature for early moves
        final_temp: Final temperature for late moves
        
    Returns:
        Temperature value for move selection
    """
    return init_temp if move_count < threshold else final_temp


def update_game_statistics(stats: Dict[str, Any], winner: Optional[int], move_count: int):
    """
    Update game statistics with game result
    
    Args:
        stats: Statistics dictionary to update
        winner: Winner player ID or None for draw
        move_count: Number of moves in the game
    """
    stats['games_played'] += 1
    stats['total_moves'] += move_count
    
    if winner is not None:
        if winner == 0:
            stats['game_outcomes']['player_0_wins'] += 1
        else:
            stats['game_outcomes']['player_1_wins'] += 1
    else:
        stats['game_outcomes']['draws'] += 1


def get_statistics_summary(stats: Dict[str, Any]) -> Dict[str, Any]:
    """
    Get summary statistics for game player
    
    Args:
        stats: Raw statistics dictionary
        
    Returns:
        Processed statistics with rates and averages
    """
    total_games = stats['games_played']
    if total_games == 0:
        return {'games_played': 0}
    
    return {
        'games_played': total_games,
        'total_moves': stats['total_moves'],
        'avg_game_length': stats['total_moves'] / total_games,
        'player_0_win_rate': stats['game_outcomes']['player_0_wins'] / total_games,
        'player_1_win_rate': stats['game_outcomes']['player_1_wins'] / total_games,
        'draw_rate': stats['game_outcomes']['draws'] / total_games,
        'outcomes': stats['game_outcomes'].copy()
    }


def reset_statistics(stats: Dict[str, Any]):
    """
    Reset game statistics to initial state
    
    Args:
        stats: Statistics dictionary to reset
    """
    stats['games_played'] = 0
    stats['total_moves'] = 0
    stats['game_outcomes'] = {'player_0_wins': 0, 'player_1_wins': 0, 'draws': 0}