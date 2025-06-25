"""
Actor-Critic network for predicting policy and value in SwinShogi
"""
import jax
import jax.numpy as jnp
import logging
from typing import Tuple, Dict, Optional
from src.shogi.board_encoder import encode_board_state, get_feature_vector

# Logging configuration
logger = logging.getLogger(__name__)


class ActorCritic:
    """
    Actor-Critic network for SwinShogi
    
    This class wraps a SwinTransformer model to predict both policy (action probabilities)
    and value (state evaluation) for MCTS integration. It handles the interface between
    the shogi game state and the neural network.
    """
    
    def __init__(self, model, params):
        """
        Initialize ActorCritic
        
        Args:
            model: Flax model (SwinShogiModel)
            params: Model parameters
        """
        self.model = model
        self.params = params
        
        # Cache JIT-compiled functions for performance
        self._predict_jit = jax.jit(self._predict_impl)
        self._extract_features_jit = jax.jit(self._extract_features_impl)
    
    def _predict_impl(self, params, board_state, feature_vector):
        """Internal JIT-compiled prediction implementation"""
        return self.model.apply(params, board_state, feature_vector=feature_vector, deterministic=True)
    
    def _extract_features_impl(self, params, board_state, feature_vector):
        """Internal JIT-compiled feature extraction implementation"""
        return self.model.apply(
            params, 
            board_state, 
            feature_vector=feature_vector,
            method=self.model.extract_features,
            deterministic=True
        )
    
    def predict(self, state: Optional[Dict] = None, 
                board_state: Optional[jnp.ndarray] = None,
                feature_vector: Optional[jnp.ndarray] = None) -> Tuple[Dict[int, float], float]:
        """
        Predict policy and value from shogi game state
        
        This is the main interface function for MCTS integration. It can accept either:
        1. A shogi game state dict (will be encoded automatically)
        2. Pre-encoded board_state and feature_vector tensors
        
        Args:
            state: Shogi game state dict with 'board', 'hands', 'turn' keys
            board_state: Pre-encoded board state tensor (1, 9, 9, 2)
            feature_vector: Pre-encoded feature vector tensor (1, 15)
            
        Returns:
            Tuple of (action_probabilities_dict, state_value)
            - action_probabilities_dict: {action_idx: probability} for actions with prob > 0.001
            - state_value: Float value representing state evaluation
        """
        # Encode state if raw state dict is provided
        if state is not None:
            board_encoded = encode_board_state(state)
            board_state = jnp.array(board_encoded).reshape(1, 9, 9, 2)
            
            feature_vec = get_feature_vector(state)
            feature_vector = jnp.array(feature_vec).reshape(1, -1)
        
        # Validate inputs
        if board_state is None or feature_vector is None:
            raise ValueError("Either 'state' or both 'board_state' and 'feature_vector' must be provided")
        
        # Ensure correct batch dimensions
        if len(board_state.shape) == 3:
            board_state = jnp.expand_dims(board_state, axis=0)
        if len(feature_vector.shape) == 1:
            feature_vector = jnp.expand_dims(feature_vector, axis=0)
        
        # Model inference
        policy_logits, value = self._predict_jit(self.params, board_state, feature_vector)
        
        # Remove batch dimension and process outputs
        if policy_logits.shape[0] == 1:
            policy_logits = policy_logits[0]
            value = value[0, 0] if len(value.shape) > 1 else value[0]
        
        # Convert logits to probabilities
        policy_probs = jax.nn.softmax(policy_logits)
        
        # Convert to dict format for MCTS (only actions with significant probability)
        action_probs = {i: float(prob) for i, prob in enumerate(policy_probs) if float(prob) > 0.001}
        
        return action_probs, float(value)
    
    def predict_batch(self, board_states: jnp.ndarray, 
                     feature_vectors: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """
        Batch prediction for training efficiency
        
        Args:
            board_states: Batch of encoded board states (batch_size, 9, 9, 2)
            feature_vectors: Batch of feature vectors (batch_size, 15)
            
        Returns:
            Tuple of (policy_logits, values) with batch dimension preserved
        """
        return self._predict_jit(self.params, board_states, feature_vectors)
    
    def extract_features(self, state: Optional[Dict] = None,
                        board_state: Optional[jnp.ndarray] = None,
                        feature_vector: Optional[jnp.ndarray] = None) -> jnp.ndarray:
        """
        Extract intermediate features from SwinTransformer
        
        This corresponds to the Transformer->AC step in the sequence diagram,
        where the transformer extracts features that the Actor-Critic then processes.
        
        Args:
            state: Shogi game state dict (optional)
            board_state: Pre-encoded board state (optional)
            feature_vector: Pre-encoded feature vector (optional)
            
        Returns:
            Feature representation from the transformer
        """
        # Encode state if needed
        if state is not None:
            board_encoded = encode_board_state(state)
            board_state = jnp.array(board_encoded).reshape(1, 9, 9, 2)
            
            feature_vec = get_feature_vector(state)
            feature_vector = jnp.array(feature_vec).reshape(1, -1)
        
        # Validate inputs
        if board_state is None or feature_vector is None:
            raise ValueError("Either 'state' or both 'board_state' and 'feature_vector' must be provided")
        
        # Ensure batch dimensions
        if len(board_state.shape) == 3:
            board_state = jnp.expand_dims(board_state, axis=0)
        if len(feature_vector.shape) == 1:
            feature_vector = jnp.expand_dims(feature_vector, axis=0)
        
        # Extract features using the model's feature extraction method
        features = self._extract_features_jit(self.params, board_state, feature_vector)
        
        return features
    
    def update_params(self, new_params):
        """
        Update model parameters
        
        Args:
            new_params: New model parameters
        """
        self.params = new_params
        logger.debug("ActorCritic parameters updated")
    
    @property
    def parameter_count(self) -> int:
        """Get total number of model parameters"""
        return sum(param.size for param in jax.tree_util.tree_leaves(self.params))


# Standalone utility functions for MCTS integration
def predict_for_mcts(model, params, shogi_state: Dict) -> Tuple[Dict[int, float], float]:
    """
    Standalone prediction function optimized for MCTS usage
    
    This function provides a simple interface for MCTS without requiring 
    an ActorCritic instance. It's designed to be the primary interface
    between MCTS and the neural network.
    
    Args:
        model: SwinShogi model
        params: Model parameters  
        shogi_state: Shogi game state dict with 'board', 'hands', 'turn'
        
    Returns:
        Tuple of (action_probabilities_dict, state_value)
    """
    # Encode the shogi state
    board_encoded = encode_board_state(shogi_state)
    board_state = jnp.array(board_encoded).reshape(1, 9, 9, 2)
    
    feature_vec = get_feature_vector(shogi_state)
    feature_vector = jnp.array(feature_vec).reshape(1, -1)
    
    # JIT-compiled prediction
    @jax.jit
    def _predict(params, board_state, feature_vector):
        return model.apply(params, board_state, feature_vector=feature_vector, deterministic=True)
    
    # Get predictions
    policy_logits, value = _predict(params, board_state, feature_vector)
    
    # Process outputs
    policy_logits = policy_logits[0]  # Remove batch dimension
    value = value[0, 0] if len(value.shape) > 1 else value[0]
    
    # Convert to probabilities and filter
    policy_probs = jax.nn.softmax(policy_logits)
    action_probs = {i: float(prob) for i, prob in enumerate(policy_probs) if float(prob) > 0.001}
    
    return action_probs, float(value)


def encode_shogi_state_for_model(shogi_state: Dict) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Encode shogi state into model input format
    
    Utility function to convert shogi game state into the tensor format
    expected by the model.
    
    Args:
        shogi_state: Shogi game state dict
        
    Returns:
        Tuple of (board_state_tensor, feature_vector_tensor)
    """
    board_encoded = encode_board_state(shogi_state)
    board_state = jnp.array(board_encoded).reshape(1, 9, 9, 2)
    
    feature_vec = get_feature_vector(shogi_state)
    feature_vector = jnp.array(feature_vec).reshape(1, -1)
    
    return board_state, feature_vector