"""
Monte Carlo Tree Search (MCTS) implementation for SwinShogi

This module implements the MCTS algorithm that efficiently explores the shogi game state space.
MCTS is used in conjunction with neural networks to guide search towards promising positions.
"""

import math
import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Tuple, Any, Optional

from config.default_config import MCTS_CONFIG
from src.model.actor_critic import predict_for_mcts


@dataclass
class MCTSConfig:
    """MCTS hyperparameter configuration"""
    n_simulations: int = MCTS_CONFIG['n_simulations']
    c_puct: float = MCTS_CONFIG['uct_c']
    max_depth: int = MCTS_CONFIG['max_depth']
    min_visits_to_expand: int = MCTS_CONFIG['min_visits_to_expand']
    
    # Exploration noise parameters
    use_dirichlet_noise: bool = MCTS_CONFIG['use_dirichlet_noise']
    exploration_noise_alpha: float = MCTS_CONFIG['exploration_noise_alpha']
    exploration_noise_epsilon: float = MCTS_CONFIG['exploration_noise_epsilon']
    
    # Advanced PUCT parameters
    c_puct_init: float = MCTS_CONFIG['c_puct_init']
    c_puct_base: float = MCTS_CONFIG['c_puct_base']
    fpu_reduction: float = MCTS_CONFIG['fpu_reduction']


class MCTSNode:
    """Node in the Monte Carlo Tree Search tree"""
    
    def __init__(self, prior: float = 0.0, parent: Optional['MCTSNode'] = None):
        """
        Initialize MCTS node
        
        Args:
            prior: Prior probability from neural network policy
            parent: Parent node (None for root)
        """
        self.visit_count = 0
        self.value_sum = 0.0
        self.prior = prior
        self.parent = parent
        self.children: Dict[str, 'MCTSNode'] = {}
        self.expanded = False
        
    @property
    def value(self) -> float:
        """Average value of this node"""
        if self.visit_count == 0:
            return 0.0
        return self.value_sum / self.visit_count
    
    def is_expanded(self) -> bool:
        """Check if node has been expanded"""
        return self.expanded
    
    def expand(self, action_priors: Dict[str, float]):
        """
        Expand node with child nodes for valid actions
        
        Args:
            action_priors: Dictionary mapping actions to their prior probabilities
        """
        self.expanded = True
        for action, prior in action_priors.items():
            self.children[action] = MCTSNode(prior=prior, parent=self)
    
    def select_child(self) -> Tuple[str, 'MCTSNode']:
        """
        Select child using PUCT algorithm
        
        Returns:
            Tuple of (action, child_node) with highest PUCT value
        """
        best_score = -float('inf')
        best_action = None
        best_child = None
        
        # Calculate dynamic c_puct
        c_puct = math.log((self.visit_count + MCTS_CONFIG['c_puct_base'] + 1) / MCTS_CONFIG['c_puct_base']) + MCTS_CONFIG['c_puct_init']
        
        for action, child in self.children.items():
            # PUCT score calculation
            q_value = -child.value  # Flip sign due to alternating players
            
            # First Play Urgency (FPU) - reduce score for unvisited nodes
            if child.visit_count == 0:
                q_value -= MCTS_CONFIG['fpu_reduction']
            
            # Upper confidence bound
            exploration_score = c_puct * child.prior * math.sqrt(self.visit_count) / (1 + child.visit_count)
            
            puct_score = q_value + exploration_score
            
            if puct_score > best_score:
                best_score = puct_score
                best_action = action
                best_child = child
        
        return best_action, best_child
    
    def update(self, value: float):
        """
        Update node statistics with simulation result
        
        Args:
            value: Value to backup
        """
        self.visit_count += 1
        self.value_sum += value
    
    def add_exploration_noise(self, noise_alpha: float = 0.3, noise_epsilon: float = 0.25):
        """
        Add Dirichlet noise to root node to encourage exploration
        
        Args:
            noise_alpha: Alpha parameter for Dirichlet distribution
            noise_epsilon: Fraction of noise to add to priors
        """
        if not self.children:
            return
        
        actions = list(self.children.keys())
        noise = np.random.dirichlet([noise_alpha] * len(actions))
        
        for action, noise_value in zip(actions, noise):
            child = self.children[action]
            child.prior = (1 - noise_epsilon) * child.prior + noise_epsilon * noise_value


class MCTS:
    """Monte Carlo Tree Search for SwinShogi"""
    
    def __init__(self, model, params, config: Optional[MCTSConfig] = None):
        """
        Initialize MCTS
        
        Args:
            model: Neural network model for position evaluation
            params: Model parameters
            config: MCTS configuration parameters
        """
        self.model = model
        self.params = params
        self.config = config or MCTSConfig()
        
        # Search statistics
        self.search_stats = {
            'nodes_created': 0,
            'leaf_evaluations': 0,
            'max_depth_reached': 0
        }
    
    def search(self, game_state: Dict, root_node: Optional[MCTSNode] = None) -> Tuple[Dict[str, float], MCTSNode]:
        """
        Run MCTS search from given game state
        
        Args:
            game_state: Current shogi game state dict
            root_node: Existing root node to continue search from (optional)
            
        Returns:
            Tuple of (action_probabilities, final_root_node)
        """
        # Initialize or reuse root node
        if root_node is None:
            root_node = MCTSNode()
            
        # Reset search statistics
        self.search_stats = {'nodes_created': 0, 'leaf_evaluations': 0, 'max_depth_reached': 0}
        
        # Expand root node if not already expanded
        if not root_node.is_expanded():
            self._expand_node(root_node, game_state)
            
        # Add exploration noise to root
        if self.config.use_dirichlet_noise:
            root_node.add_exploration_noise(
                self.config.exploration_noise_alpha,
                self.config.exploration_noise_epsilon
            )
        
        # Run simulations
        for _ in range(self.config.n_simulations):
            # Clone game state for simulation
            sim_game_state = self._clone_game_state(game_state)
            self._simulate(sim_game_state, root_node, depth=0)
        
        # Calculate action probabilities based on visit counts
        action_probs = self._get_action_probabilities(root_node)
        
        return action_probs, root_node
    
    def _simulate(self, game_state: Dict, node: MCTSNode, depth: int) -> float:
        """
        Run single MCTS simulation
        
        Args:
            game_state: Current game state
            node: Current node in tree
            depth: Current search depth
            
        Returns:
            Value estimate for this position
        """
        # Check depth limit
        if depth >= self.config.max_depth:
            return self._evaluate_position(game_state)
        
        # Check for terminal state
        if game_state['is_terminal']:
            return self._get_terminal_value(game_state)
        
        # Expand node if not expanded and has enough visits
        if not node.is_expanded() and node.visit_count >= self.config.min_visits_to_expand:
            value = self._expand_node(node, game_state)
            node.update(value)
            return value
        
        # If node is not expanded (leaf node), evaluate with neural network
        if not node.is_expanded():
            value = self._evaluate_position(game_state)
            node.update(value)
            return value
        
        # Select best child using PUCT
        action, child_node = node.select_child()
        
        # Apply action to game state
        new_game_state = self._apply_action(game_state, action)
        
        # Recursively simulate
        value = self._simulate(new_game_state, child_node, depth + 1)
        
        # Update statistics
        self.search_stats['max_depth_reached'] = max(self.search_stats['max_depth_reached'], depth)
        
        # Backup value (flip sign for alternating players)
        node.update(-value)
        
        return -value
    
    def _expand_node(self, node: MCTSNode, game_state: Dict) -> float:
        """
        Expand node by evaluating position with neural network
        
        Args:
            node: Node to expand
            game_state: Current game state
            
        Returns:
            Position value from neural network
        """
        # Get neural network evaluation
        action_priors, value = predict_for_mcts(self.model, self.params, game_state)
        
        # Filter for valid moves only
        from src.shogi.shogi_game import ShogiGame
        valid_moves = ShogiGame.get_valid_moves_from_state(game_state)
        from src.utils.action_utils import filter_valid_actions
        valid_action_priors = filter_valid_actions(action_priors, valid_moves)
        
        # Expand node
        node.expand(valid_action_priors)
        
        # Update statistics
        self.search_stats['nodes_created'] += len(valid_action_priors)
        self.search_stats['leaf_evaluations'] += 1
        
        return value
    
    def _evaluate_position(self, game_state: Dict) -> float:
        """
        Evaluate position using neural network
        
        Args:
            game_state: Game state to evaluate
            
        Returns:
            Position value estimate
        """
        _, value = predict_for_mcts(self.model, self.params, game_state)
        self.search_stats['leaf_evaluations'] += 1
        return value
    
    def _get_action_probabilities(self, root_node: MCTSNode, temperature: float = 1.0) -> Dict[str, float]:
        """
        Calculate action probabilities from visit counts
        
        Args:
            root_node: Root node of search tree
            temperature: Temperature for softmax (0 = deterministic)
            
        Returns:
            Dictionary mapping actions to probabilities
        """
        if not root_node.children:
            return {}
        
        # Get visit counts
        visits = {action: child.visit_count for action, child in root_node.children.items()}
        
        if temperature == 0:
            # Deterministic selection
            best_action = max(visits.items(), key=lambda x: x[1])[0]
            return {action: 1.0 if action == best_action else 0.0 for action in visits}
        else:
            # Temperature-scaled probabilities
            visit_counts = np.array(list(visits.values()))
            if temperature != 1.0:
                visit_counts = visit_counts ** (1.0 / temperature)
            
            probabilities = visit_counts / np.sum(visit_counts)
            return {action: prob for action, prob in zip(visits.keys(), probabilities)}
    
    def select_action(self, action_probs: Dict[str, float], temperature: float = 0.0) -> str:
        """
        Select action based on probabilities
        
        Args:
            action_probs: Action probabilities from search
            temperature: Temperature for selection randomness
            
        Returns:
            Selected action
        """
        if not action_probs:
            raise ValueError("No actions available")
        
        if temperature == 0:
            # Deterministic selection
            return max(action_probs.items(), key=lambda x: x[1])[0]
        else:
            # Stochastic selection
            actions = list(action_probs.keys())
            probs = list(action_probs.values())
            
            # Apply temperature
            if temperature != 1.0:
                probs = np.array(probs) ** (1.0 / temperature)
                probs = probs / np.sum(probs)
            
            return np.random.choice(actions, p=probs)
    
    # Helper methods for game state management
    def _clone_game_state(self, game_state: Dict) -> Dict:
        """Create deep copy of game state"""
        import copy
        return copy.deepcopy(game_state)

    
    def _apply_action(self, game_state: Dict, action: str) -> Dict:
        """Apply action to game state and return new state"""
        from src.shogi.shogi_game import ShogiGame
        return ShogiGame.apply_action_to_state(action, game_state)


    def _get_terminal_value(self, game_state: Dict) -> float:
        """Get value for terminal game state using SwinTransformer evaluation"""
        from src.model.actor_critic import predict_for_mcts
        
        # Use SwinTransformer evaluation directly on game_state
        # This gives more nuanced values than just -1/0/1
        _, value = predict_for_mcts(self.model, self.params, game_state)
        return float(value)

    def _move_to_action_index(self, move: str) -> int:
        """Convert USI move string to action index"""
        from src.utils.action_utils import action_to_index
        return action_to_index(move)
    
    def get_search_info(self) -> Dict[str, Any]:
        """Get information about the last search"""
        return {
            'nodes_created': self.search_stats['nodes_created'],
            'leaf_evaluations': self.search_stats['leaf_evaluations'],
            'max_depth_reached': self.search_stats['max_depth_reached'],
            'simulations_run': self.config.n_simulations
        }


# Utility functions for easier integration
def create_mcts(model, params, **config_overrides) -> MCTS:
    """
    Create MCTS instance with custom configuration
    
    Args:
        model: Neural network model
        params: Model parameters
        **config_overrides: Override default MCTS configuration
        
    Returns:
        Configured MCTS instance
    """
    config_dict = MCTS_CONFIG.copy()
    config_dict.update(config_overrides)
    
    # Create config object with overrides
    config = MCTSConfig(**{k: v for k, v in config_dict.items() if k in MCTSConfig.__dataclass_fields__})
    
    return MCTS(model, params, config)


def mcts_search(model, params, game_state: Dict, n_simulations: int = None) -> Tuple[Dict[str, float], Dict[str, Any]]:
    """
    Convenient function to run MCTS search
    
    Args:
        model: Neural network model
        params: Model parameters
        game_state: Current game state
        n_simulations: Number of simulations (uses config default if None)
        
    Returns:
        Tuple of (action_probabilities, search_info)
    """
    config_overrides = {}
    if n_simulations is not None:
        config_overrides['n_simulations'] = n_simulations
    
    mcts = create_mcts(model, params, **config_overrides)
    action_probs, _ = mcts.search(game_state)
    
    return action_probs, mcts.get_search_info()