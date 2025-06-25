"""
Model utility functions for SwinShogi
"""

import jax
import jax.numpy as jnp
from typing import Tuple, Optional


class PolicyGradientLoss:
    """
    Policy gradient loss functions for reinforcement learning
    
    This class aggregates loss functions used in policy gradient methods (REINFORCE, A2C, etc.):
    
    - policy_loss: Policy-related loss (with optional advantage weighting)
    - value_loss: Value estimation loss (squared error)
    - entropy_loss: Policy entropy loss (promotes exploration)
    
    Also provides comprehensive functions for combining multiple losses.
    """
    
    @staticmethod
    def policy_loss(logits: jnp.ndarray, action_probs: jnp.ndarray, advantages: Optional[jnp.ndarray] = None) -> jnp.ndarray:
        """
        Calculate policy loss
        
        Args:
            logits: Model output logits
            action_probs: Target action probabilities
            advantages: Advantage values (if None, computes standard cross-entropy loss)
            
        Returns:
            Policy loss value
        """
        log_probs = jax.nn.log_softmax(logits)
        
        if advantages is None:
            # Standard cross-entropy loss
            return -jnp.mean(jnp.sum(action_probs * log_probs, axis=-1))
        
        # Advantage-weighted cross-entropy loss
        return -jnp.mean(jnp.sum(action_probs * log_probs, axis=-1) * advantages)
    
    @staticmethod
    def value_loss(values: jnp.ndarray, target_values: jnp.ndarray) -> jnp.ndarray:
        """
        Calculate value loss
        
        Args:
            values: Model predicted values
            target_values: Target values
            
        Returns:
            Value loss value
        """
        # Mean squared error
        return jnp.mean(jnp.square(values - target_values))
    
    @staticmethod
    def entropy_loss(logits: jnp.ndarray) -> jnp.ndarray:
        """
        Calculate entropy loss
        
        Higher entropy leads to more uniform policy distribution, promoting exploration.
        Returns negative value to maximize entropy (minimize loss) during optimization.
        
        Args:
            logits: Model output logits
            
        Returns:
            Entropy loss value
        """
        # Efficient entropy calculation using log_softmax
        log_probs = jax.nn.log_softmax(logits)
        probs = jnp.exp(log_probs)
        entropy = -jnp.sum(probs * log_probs, axis=-1)
        return -jnp.mean(entropy)  # Negative for maximization
    
    @staticmethod
    @jax.jit
    def compute_losses_from_model_outputs(policy_logits: jnp.ndarray, values: jnp.ndarray, 
                                        action_onehot: jnp.ndarray, advantages: jnp.ndarray,
                                        target_values: jnp.ndarray, 
                                        entropy_coeff: float = 0.01) -> Tuple[jnp.ndarray, Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]]:
        """
        Comprehensive function to calculate losses from model outputs
        
        Calculates individual losses and combines them with weighting for total loss.
        
        Args:
            policy_logits: Model policy output
            values: Model value output
            action_onehot: Target action one-hot encoding
            advantages: Advantage values
            target_values: Target values
            entropy_coeff: Entropy coefficient (balances exploration vs exploitation)
            
        Returns:
            (total_loss, (policy_loss, value_loss, entropy_loss)): Tuple of loss values
        """
        # Calculate individual losses
        policy_loss = PolicyGradientLoss.policy_loss(policy_logits, action_onehot, advantages)
        value_loss = PolicyGradientLoss.value_loss(values, target_values)
        entropy_loss = PolicyGradientLoss.entropy_loss(policy_logits)
        
        # Calculate total loss
        total_loss = policy_loss + value_loss + entropy_coeff * entropy_loss
            
        return total_loss, (policy_loss, value_loss, entropy_loss)