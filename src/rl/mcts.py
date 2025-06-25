"""
Monte Carlo Tree Search (MCTS) implementation.

This module provides a game-agnostic MCTS algorithm based on the AlphaZero-style
iterative Select, Expand, and Backup loop.
"""

import math
import numpy as np
from dataclasses import dataclass
from typing import Dict, Tuple, Any, Optional, Callable

from config.default_config import MCTS_CONFIG

Evaluator = Callable[[Any], Tuple[Dict[int, float], float]]

@dataclass
class MCTSConfig:
    n_simulations: int = MCTS_CONFIG['n_simulations']
    c_puct_init: float = MCTS_CONFIG['c_puct_init']
    c_puct_base: float = MCTS_CONFIG['c_puct_base']
    fpu_reduction: float = MCTS_CONFIG['fpu_reduction']
    exploration_noise_alpha: float = MCTS_CONFIG['exploration_noise_alpha']
    exploration_noise_epsilon: float = MCTS_CONFIG['exploration_noise_epsilon']

class MCTSNode:
    def __init__(self, prior: float = 0.0, parent: Optional['MCTSNode'] = None):
        self.visit_count = 0
        self.value_sum = 0.0
        self.prior = prior
        self.parent = parent
        self.children: Dict[int, 'MCTSNode'] = {}

    @property
    def value(self) -> float:
        if self.visit_count == 0:
            return 0.0
        return self.value_sum / self.visit_count

    def is_expanded(self) -> bool:
        return len(self.children) > 0

    def expand(self, action_priors: Dict[int, float]):
        for action, prior in action_priors.items():
            self.children[action] = MCTSNode(prior=prior, parent=self)

    def select_child(self, config: MCTSConfig) -> Tuple[int, 'MCTSNode']:
        best_score = -float('inf')
        best_action = -1
        best_child = None

        c_puct = math.log((self.visit_count + config.c_puct_base + 1) / config.c_puct_base) + config.c_puct_init

        for action, child in self.children.items():
            q_value = -child.value if child.visit_count > 0 else -config.fpu_reduction
            exploration_score = c_puct * child.prior * math.sqrt(self.visit_count) / (1 + child.visit_count)
            puct_score = q_value + exploration_score

            if puct_score > best_score:
                best_score = puct_score
                best_action = action
                best_child = child
        
        return best_action, best_child

    def add_exploration_noise(self, config: MCTSConfig):
        if not self.children:
            return
        actions = list(self.children.keys())
        noise = np.random.dirichlet([config.exploration_noise_alpha] * len(actions))
        for action, noise_val in zip(actions, noise):
            self.children[action].prior = self.children[action].prior * (1 - config.exploration_noise_epsilon) + noise_val * config.exploration_noise_epsilon

class MCTS:
    def __init__(self, evaluator: Evaluator, config: Optional[MCTSConfig] = None):
        self.evaluator = evaluator
        self.config = config or MCTSConfig()

    def search(self, root_env: Any, temperature: float = 1.0) -> Dict[int, float]:
        root_node = MCTSNode()

        # Initial expansion of the root
        action_priors, _ = self._evaluate_and_expand(root_node, root_env)
        root_node.expand(action_priors)
        root_node.add_exploration_noise(self.config)

        for _ in range(self.config.n_simulations):
            node = root_node
            sim_env = root_env.copy()
            path = [node]

            # 1. Select
            while node.is_expanded():
                action, node = node.select_child(self.config)
                sim_env.apply_move(action)
                path.append(node)

            # 2. Expand & Evaluate
            if not sim_env.is_terminal():
                action_priors, value = self._evaluate_and_expand(node, sim_env)
                node.expand(action_priors)
            else:
                value = sim_env.get_outcome()

            # 3. Backup
            for node_in_path in reversed(path):
                node_in_path.value_sum += value
                node_in_path.visit_count += 1
                value = -value  # Flip value for the parent

        return self._get_action_probabilities(root_node, temperature)

    def _get_action_probabilities(self, root_node: MCTSNode, temperature: float) -> Dict[int, float]:
        if not root_node.children:
            return {}

        visits = {action: child.visit_count for action, child in root_node.children.items()}
        if not visits:
            return {}

        if temperature == 0:
            best_action = max(visits, key=visits.get)
            return {action: 1.0 if action == best_action else 0.0 for action in visits}
        
        visit_counts = np.array(list(visits.values()), dtype=np.float32)
        visit_counts_temp = visit_counts ** (1.0 / temperature)
        probabilities = visit_counts_temp / np.sum(visit_counts_temp)
        
        return {action: prob for action, prob in zip(visits.keys(), probabilities)}

    def _evaluate_and_expand(self, node: MCTSNode, env: Any) -> Tuple[Dict[int, float], float]:
        action_priors, value = self.evaluator(env)
        legal_moves = env.get_legal_moves()
        legal_action_priors = {move: action_priors.get(move, 0) for move in legal_moves}
        sum_priors = sum(legal_action_priors.values())
        if sum_priors > 0:
            legal_action_priors = {k: v / sum_priors for k, v in legal_action_priors.items()}
        return legal_action_priors, value
