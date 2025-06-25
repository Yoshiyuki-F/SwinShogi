"""
Actor-Critic network for predicting policy and value in SwinShogi
"""
import jax
import jax.numpy as jnp
import logging
from typing import Tuple, Dict, Optional, Any

from src.shogi.shogi_env import ShogiEnv

# Logging configuration
logger = logging.getLogger(__name__)

class ActorCritic:
    """
    Wraps the model and parameters to provide a clean interface for prediction and updates.
    This class is the single source of truth for the agent's "brain".
    """
    
    def __init__(self, model: Any, params: Any):
        self.model = model
        self.params = params
        self._predict_jit = jax.jit(self.model.apply)

    def predict(self, env: ShogiEnv) -> Tuple[Dict[int, float], float]:
        """
        Predicts policy and value for a given game environment.
        This method is the designated evaluator for the MCTS.
        """
        observation = env.get_observation()
        # The model expects a batch dimension
        observation = jnp.expand_dims(observation, axis=0)

        policy_logits, value = self._predict_jit(self.params, observation, deterministic=True)

        # Remove batch dimension and convert to standard Python types
        policy_probs = jax.nn.softmax(policy_logits[0])
        value = float(value[0])

        # Create a dictionary for MCTS, filtering for significant probabilities
        action_probs = {i: float(prob) for i, prob in enumerate(policy_probs) if prob > 1e-3}
        
        return action_probs, value

    def update_params(self, new_params: Any):
        """Updates the model parameters."""
        self.params = new_params
        logger.info("ActorCritic parameters have been updated.")
