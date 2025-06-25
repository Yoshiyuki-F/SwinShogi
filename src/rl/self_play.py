"""
Self-play data generation for SwinShogi reinforcement learning

This module implements self-play functionality that generates training data
by having the neural network play against itself using MCTS for move selection.
"""

import jax.numpy as jnp
import numpy as np
import time
import logging
from typing import List, Dict, Tuple, Optional, Any
from dataclasses import dataclass

from src.rl.mcts import MCTS
from src.shogi.shogi_game import ShogiGame
from src.shogi.board_encoder import encode_board_state, get_feature_vector
from src.shogi.board_visualizer import BoardVisualizer
from config.default_config import get_mcts_config, MCTSConfig

# Logger setup
logger = logging.getLogger(__name__)


@dataclass
class SelfPlayExample:
    """Single example from self-play game"""
    board_state: jnp.ndarray  # Encoded board state (9, 9, 2)
    feature_vector: jnp.ndarray  # Feature vector (15,)
    action_probs: jnp.ndarray  # MCTS action probabilities (2187,)
    player: int  # Player who made this move (0 or 1)
    position_value: float  # SwinTransformer evaluation of this position


@dataclass
class SelfPlayResult:
    """Result of a self-play game"""
    examples: List[SelfPlayExample]
    winner: Optional[int]  # 0, 1, or None for draw
    game_length: int
    final_score: Dict[int, float]  # Final scores for each player


class SelfPlay:
    """
    Self-play data generator for SwinShogi
    
    Generates training data by having the neural network play against itself
    using MCTS for move selection and policy improvement.
    """
    
    def __init__(self, 
                 model, 
                 params, 
                 config: Optional[MCTSConfig] = None,
                 max_moves: int = 300,
                 temperature_threshold: int = 30,
                 temperature_init: float = 1.0,
                 temperature_final: float = 0.1):
        """
        Initialize self-play generator
        
        Args:
            model: Neural network model
            params: Model parameters
            config: MCTS configuration
            max_moves: Maximum moves per game
            temperature_threshold: Move number to switch temperature
            temperature_init: Initial temperature for move selection
            temperature_final: Final temperature for move selection
        """
        self.model = model
        self.params = params
        self.mcts_config = config or get_mcts_config()
        self.max_moves = max_moves
        self.temperature_threshold = temperature_threshold
        self.temperature_init = temperature_init
        self.temperature_final = temperature_final
        
        # Initialize MCTS
        self.mcts = MCTS(model, params, self.mcts_config)
        
        # Statistics
        self.games_played = 0
        self.total_moves = 0
        self.game_outcomes = {'player_0_wins': 0, 'player_1_wins': 0, 'draws': 0}
    
    def play_game(self, verbose: bool = False) -> SelfPlayResult:
        """
        Play a single self-play game
        
        Args:
            verbose: Whether to log game progress
            
        Returns:
            SelfPlayResult containing game data and outcome
        """
        # Initialize game
        game = ShogiGame()
        examples = []
        move_count = 0
        
        if verbose:
            logger.info(f"Starting self-play game {self.games_played + 1}")
        
        while not game.is_terminal and move_count < self.max_moves:
            # Get current game state using ShogiGame property
            game_state = game.game_state
            
            # Determine temperature based on move count
            from src.rl.game_utils import get_temperature_for_move
            temperature = get_temperature_for_move(
                move_count, self.temperature_threshold, 
                self.temperature_init, self.temperature_final
            )
            
            # Display board if verbose (before MCTS search)
            if verbose:
                print(f"\n=== Move {move_count} ===")
                print(f"Current player: {'先手' if game.current_player.value == 0 else '後手'}")
                print(f"Temperature: {temperature:.2f}, MCTS simulations: {self.mcts_config.n_simulations}")
                print("Thinking...")
            
            # Run MCTS search - this will compute position evaluation as part of the process
            action_probs, root_node = self.mcts.search(game_state)
            
            # Extract position evaluation from MCTS root node evaluation
            # The root node gets evaluated during MCTS search
            nn_value = root_node.value
            checkmate_bonus = game._calculate_checkmate_bonus(game.current_player)
            position_value = nn_value + checkmate_bonus
            
            # Add evaluation to game_state for display
            game_state['evaluation'] = position_value
            
            # Display board with evaluation after MCTS search
            if verbose:
                BoardVisualizer.visualize_board(game_state, self.model, self.params)
            
            if not action_probs:
                if verbose:
                    logger.warning(f"No valid moves found at move {move_count}")
                break
            
            # Convert action probabilities to full action vector
            full_action_probs = self._convert_to_full_action_vector(action_probs)
            
            # Encode game state for training
            board_encoded = encode_board_state(game_state)
            feature_vector = get_feature_vector(game_state)
            
            # Use the pre-computed evaluation from game_state
            position_value = game_state['evaluation']
            
            # Create training example
            example = SelfPlayExample(
                board_state=jnp.array(board_encoded),
                feature_vector=jnp.array(feature_vector),
                action_probs=full_action_probs,
                player=game.current_player.value,
                position_value=position_value
            )
            examples.append(example)
            
            # Select action using temperature
            selected_action = self.mcts.select_action(action_probs, temperature)
            
            if verbose:
                print(f"Selected move: {selected_action}")
                if len(action_probs) > 1:
                    # Show top 3 candidate moves
                    sorted_actions = sorted(action_probs.items(), key=lambda x: x[1], reverse=True)[:3]
                    print("Top candidates:")
                    for i, (action, prob) in enumerate(sorted_actions, 1):
                        print(f"  {i}. {action}: {prob:.3f}")
                print()
            
            # Apply move using utility function
            from src.rl.game_utils import validate_and_apply_move
            success = validate_and_apply_move(game, selected_action, verbose)
            if not success:
                break
            
            move_count += 1
            
            if verbose and move_count % 20 == 0:
                logger.info(f"Move {move_count}, current player: {game.current_player}")
        
        # Determine game outcome using utility function
        from src.rl.game_utils import determine_game_outcome, calculate_game_scores
        winner = determine_game_outcome(game)
        final_score = calculate_game_scores(winner)
        
        # Update statistics using utility function
        from src.rl.game_utils import update_game_statistics
        stats = {
            'games_played': self.games_played,
            'total_moves': self.total_moves,
            'game_outcomes': self.game_outcomes
        }
        update_game_statistics(stats, winner, move_count)
        self.games_played = stats['games_played']
        self.total_moves = stats['total_moves']
        self.game_outcomes = stats['game_outcomes']
        
        if verbose:
            outcome_str = f"Player {winner} wins" if winner is not None else "Draw"
            logger.info(f"Game completed: {outcome_str}, {move_count} moves")
            
            # Display final board state
            final_evaluation = game.evaluate_with_model(self.model, self.params)
            final_game_state = game.game_state
            final_game_state['evaluation'] = final_evaluation
            print(f"\n=== Final Position (Game Over) ===")
            BoardVisualizer.visualize_board(final_game_state, self.model, self.params)
            print(f"Result: {outcome_str}")
        
        return SelfPlayResult(
            examples=examples,
            winner=winner,
            game_length=move_count,
            final_score=final_score
        )
    
    def generate_games(self, 
                      num_games: int, 
                      verbose: bool = True) -> List[SelfPlayResult]:
        """
        Generate multiple self-play games
        
        Args:
            num_games: Number of games to generate
            verbose: Whether to log progress
            
        Returns:
            List of SelfPlayResult objects
        """
        results = []
        start_time = time.time()
        
        if verbose:
            logger.info(f"Generating {num_games} self-play games...")
        
        for game_idx in range(num_games):
            game_start_time = time.time()
            
            # Play game
            result = self.play_game(verbose=False)
            results.append(result)
            
            # Log progress
            if verbose and (game_idx + 1) % max(1, num_games // 10) == 0:
                game_time = time.time() - game_start_time
                elapsed = time.time() - start_time
                eta = elapsed * (num_games / (game_idx + 1) - 1)
                
                logger.info(
                    f"Game {game_idx + 1}/{num_games} completed "
                    f"({game_time:.1f}s, {result.game_length} moves) "
                    f"- ETA: {eta:.1f}s"
                )
        
        total_time = time.time() - start_time
        if verbose:
            self._log_generation_summary(results, total_time)
        
        return results
    
    def results_to_training_examples(self, 
                                   results: List[SelfPlayResult]) -> List[Dict[str, Any]]:
        """
        Convert self-play results to training examples using pre-computed evaluations
        
        Args:
            results: List of SelfPlayResult objects
            
        Returns:
            List of training examples compatible with trainer
        """
        from src.rl.training_utils import game_results_to_training_examples
        return game_results_to_training_examples(results)
    
    def update_model_params(self, new_params):
        """Update model parameters for MCTS"""
        self.params = new_params
        self.mcts.params = new_params
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get self-play statistics"""
        from src.rl.game_utils import get_statistics_summary
        stats = {
            'games_played': self.games_played,
            'total_moves': self.total_moves,
            'game_outcomes': self.game_outcomes
        }
        return get_statistics_summary(stats)
    
    def reset_statistics(self):
        """Reset self-play statistics"""
        from src.rl.game_utils import reset_statistics
        stats = {
            'games_played': self.games_played,
            'total_moves': self.total_moves,
            'game_outcomes': self.game_outcomes
        }
        reset_statistics(stats)
        self.games_played = stats['games_played']
        self.total_moves = stats['total_moves']
        self.game_outcomes = stats['game_outcomes']
    
    # Helper methods
    def _convert_to_full_action_vector(self, action_probs: Dict[str, float]) -> jnp.ndarray:
        """
        Convert action probabilities dict to full action vector
        
        Args:
            action_probs: Dictionary mapping actions to probabilities
            
        Returns:
            Full action probability vector of size 2187
        """
        from src.utils.action_utils import convert_to_full_action_vector
        return convert_to_full_action_vector(action_probs)
    
    
    def _log_generation_summary(self, results: List[SelfPlayResult], total_time: float):
        """Log summary of game generation"""
        from src.rl.training_utils import log_game_generation_summary
        log_game_generation_summary(results, total_time, "self-play")


# Utility functions
def create_self_play(model, params, **config_overrides) -> SelfPlay:
    """
    Create SelfPlay instance with custom configuration
    
    Args:
        model: Neural network model
        params: Model parameters
        **config_overrides: Override default configuration
        
    Returns:
        Configured SelfPlay instance
    """
    mcts_config = get_mcts_config()
    
    # Apply MCTS config overrides
    mcts_overrides = {k: v for k, v in config_overrides.items() 
                     if k in ['n_simulations', 'c_puct', 'max_depth']}
    if mcts_overrides:
        for key, value in mcts_overrides.items():
            setattr(mcts_config, key, value)
    
    # Apply SelfPlay config overrides
    selfplay_kwargs = {k: v for k, v in config_overrides.items() 
                      if k in ['max_moves', 'temperature_threshold', 
                              'temperature_init', 'temperature_final']}
    
    return SelfPlay(model, params, mcts_config, **selfplay_kwargs)


def generate_training_data(model, 
                         params, 
                         num_games: int,
                         **config_overrides) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """
    Convenient function to generate training data
    
    Args:
        model: Neural network model
        params: Model parameters
        num_games: Number of games to generate
        **config_overrides: Configuration overrides
        
    Returns:
        Tuple of (training_examples, statistics)
    """
    self_play = create_self_play(model, params, **config_overrides)
    results = self_play.generate_games(num_games)
    training_examples = self_play.results_to_training_examples(results)
    statistics = self_play.get_statistics()
    
    return training_examples, statistics