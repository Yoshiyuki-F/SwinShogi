
import jax.numpy as jnp
from typing import Optional

from src.shogi.shogi_game import ShogiGame, GameState
from src.shogi.board_encoder import encode_board_state

class ShogiEnv:
    """
    A wrapper for the ShogiGame to provide a generic environment interface.
    This decouples the MCTS algorithm from the specifics of the game logic.
    """
    def __init__(self, initial_state: Optional[GameState] = None):
        if initial_state:
            self.game_state = initial_state
        else:
            self.game_state = ShogiGame.get_initial_state()

    @property
    def current_player(self) -> int:
        """Returns the current player (0 for black, 1 for white)."""
        return self.game_state.current_player

    def get_legal_moves(self) -> list[int]:
        """Returns a list of legal moves for the current state."""
        return ShogiGame.get_valid_moves(self.game_state)

    def apply_move(self, move: int):
        """Applies a move to the current state."""
        self.game_state = ShogiGame.apply_action(self.game_state, move)

    def is_terminal(self) -> bool:
        """Checks if the current state is terminal."""
        is_terminal, _ = ShogiGame.is_terminal(self.game_state)
        return is_terminal

    def get_outcome(self) -> float:
        """
        Returns the game outcome from the perspective of the current player.
        1 for a win, -1 for a loss, 0 for a draw.
        """
        is_terminal, winner = ShogiGame.is_terminal(self.game_state)
        if not is_terminal:
            return 0.0

        if winner == -1:  # Draw
            return 0.0
        
        # If it's the winner's turn, they won, return 1.0
        # If it's the loser's turn, they lost, return -1.0
        return 1.0 if winner == self.current_player else -1.0

    def get_observation(self) -> jnp.ndarray:
        """
        Returns the neural network input for the current board state.
        """
        return encode_board_state(self.game_state)

    def copy(self) -> 'ShogiEnv':
        """Creates a deep copy of the environment."""
        # GameState is a dataclass, so a simple copy is sufficient if it holds value types.
        # If it holds mutable objects, a deepcopy would be safer.
        # For now, assuming GameState is copy-safe.
        return ShogiEnv(initial_state=self.game_state)

    def __str__(self) -> str:
        """Returns a string representation of the board."""
        # This is a placeholder. A proper visualizer should be used.
        return f"Player: {self.current_player}, Moves: {len(self.get_legal_moves())}"

