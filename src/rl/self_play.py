"""
Self-play data generation for SwinShogi reinforcement learning.

This module orchestrates the self-play process, where the agent plays against
itself using a game-agnostic MCTS engine to generate training data.
"""

import jax.numpy as jnp
import numpy as np
import logging
from typing import List, Dict, Optional, Any
from dataclasses import dataclass

from src.rl.mcts import MCTS, MCTSConfig
from src.shogi.shogi_env import ShogiEnv
from src.model.actor_critic import ActorCritic
from src.utils.action_utils import convert_to_full_action_vector
from config.default_config import get_mcts_config

# Logger setup
logger = logging.getLogger(__name__)


@dataclass
class SelfPlayExample:
    """Single example from a self-play game."""
    observation: jnp.ndarray
    action_probs: jnp.ndarray
    outcome: float


@dataclass
class SelfPlayResult:
    """Result of a self-play game."""
    examples: List[SelfPlayExample]
    winner: Optional[int]
    game_length: int


class SelfPlay:
    """
    Orchestrates self-play games to generate training data.
    It uses a generic MCTS engine and a game-specific environment (ShogiEnv).
    """

    def __init__(self, 
                 actor_critic: 'ActorCritic', 
                 config: Optional[MCTSConfig] = None,
                 max_moves: int = 300,
                 temperature_threshold: int = 30,
                 temperature_init: float = 1.0,
                 temperature_final: float = 0.0):
        self.actor_critic = actor_critic
        self.mcts_config = config or get_mcts_config()
        self.max_moves = max_moves
        self.temperature_threshold = temperature_threshold
        self.temperature_init = temperature_init
        self.temperature_final = temperature_final

        # Inject the actor_critic's predict method as the evaluator for MCTS.
        self.mcts = MCTS(self.actor_critic.predict, self.mcts_config)

        self.games_played = 0
        self.total_moves = 0
        self.game_outcomes = {'player_0_wins': 0, 'player_1_wins': 0, 'draws': 0}

    def play_game(self, verbose: bool = False) -> SelfPlayResult:
        """Plays a single self-play game."""
        env = ShogiEnv()
        history = []
        move_count = 0

        while not env.is_terminal() and move_count < self.max_moves:
            temperature = self.temperature_init if move_count < self.temperature_threshold else self.temperature_final

            action_probs = self.mcts.search(env, temperature=temperature)
            
            observation = env.get_observation()
            full_action_probs = convert_to_full_action_vector(action_probs)
            history.append((observation, full_action_probs, env.current_player))

            if not action_probs:
                if verbose:
                    logger.warning(f"No valid moves found at move {move_count}")
                break

            # Select action based on the returned probabilities
            actions = list(action_probs.keys())
            probabilities = list(action_probs.values())
            selected_action = np.random.choice(actions, p=probabilities)

            env.apply_move(selected_action)
            move_count += 1

        outcome = env.get_outcome()
        winner = None
        if outcome > 0:
            winner = env.current_player
        elif outcome < 0:
            winner = 1 - env.current_player

        # Create training examples from the game history
        examples = []
        for i, (obs, probs, player) in enumerate(history):
            # The reward is from the perspective of the player who made the move
            reward = outcome if player == env.current_player else -outcome
            examples.append(SelfPlayExample(observation=obs, action_probs=probs, outcome=reward))

        self._update_stats(winner, move_count)

        return SelfPlayResult(examples=examples, winner=winner, game_length=move_count)

    def _update_stats(self, winner: Optional[int], move_count: int):
        self.games_played += 1
        self.total_moves += move_count
        if winner == 0:
            self.game_outcomes['player_0_wins'] += 1
        elif winner == 1:
            self.game_outcomes['player_1_wins'] += 1
        else:
            self.game_outcomes['draws'] += 1

    def get_statistics(self) -> Dict[str, Any]:
        """Returns the current self-play statistics."""
        return {
            'games_played': self.games_played,
            'total_moves': self.total_moves,
            'average_moves': self.total_moves / self.games_played if self.games_played > 0 else 0,
            **self.game_outcomes
        }
