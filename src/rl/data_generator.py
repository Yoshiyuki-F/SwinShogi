"""
Training data generation for SwinShogi

This module provides flexible data generation from various sources:
- Self-play games
- Games against other AIs  
- Human games
- Loaded game records
"""

import logging
from typing import List, Dict, Any
from dataclasses import dataclass
from abc import ABC, abstractmethod

import jax.numpy as jnp

from src.rl.self_play import SelfPlay
from src.shogi.shogi_game import ShogiGame
from src.shogi.board_encoder import encode_board_state, get_feature_vector

logger = logging.getLogger(__name__)


@dataclass
class TrainingExample:
    """Training example for neural network"""
    board_state: jnp.ndarray  # Encoded board state
    feature_vector: jnp.ndarray  # Game features
    action_probs: jnp.ndarray  # Target policy
    value: float  # Target value
    player: int  # Player perspective (0 or 1)


class DataSource(ABC):
    """Abstract base class for training data sources"""
    
    @abstractmethod
    def generate_data(self, num_games: int) -> List[TrainingExample]:
        """Generate training examples"""
        pass
    
    @abstractmethod
    def get_statistics(self) -> Dict[str, Any]:
        """Get generation statistics"""
        pass


from src.model.actor_critic import ActorCritic

class SelfPlayDataSource(DataSource):
    """Self-play data source"""
    
    def __init__(self, actor_critic: ActorCritic, mcts_config: dict):
        self.self_play = SelfPlay(actor_critic, mcts_config)
    
    def generate_data(self, num_games: int) -> List[TrainingExample]:
        """Generate data from self-play games"""
        logger.info(f"Generating {num_games} self-play games...")
        
        results = [self.self_play.play_game() for _ in range(num_games)]
        
        examples = []
        for result in results:
            for ex in result.examples:
                examples.append(TrainingExample(
                    board_state=ex.observation,
                    feature_vector=jnp.zeros(15), # Placeholder, as observation now contains all info
                    action_probs=ex.action_probs,
                    value=ex.outcome,
                    player=0 # Placeholder, player info is implicitly handled by outcome
                ))
        
        logger.info(f"Generated {len(examples)} training examples from self-play")
        return examples
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get self-play statistics"""
        return self.self_play.get_statistics()


class GameRecordDataSource(DataSource):
    """Data source from recorded games (USI format, PGN, etc.)"""
    
    def __init__(self, actor_critic: ActorCritic):
        self.actor_critic = actor_critic
        self.games_processed = 0
    
    def generate_data(self, game_records: List[List[str]]) -> List[TrainingExample]:
        """
        Generate training data from game records
        
        Args:
            game_records: List of games, each game is a list of USI moves
            
        Returns:
            List of training examples
        """
        examples = []
        
        for game_moves in game_records:
            game_examples = self._process_game_record(game_moves)
            examples.extend(game_examples)
            self.games_processed += 1
        
        logger.info(f"Generated {len(examples)} examples from {len(game_records)} game records")
        return examples
    
    def _process_game_record(self, moves: List[str]) -> List[TrainingExample]:
        """Process a single game record into training examples"""
        examples = []
        game = ShogiGame()
        
        for move_idx, move in enumerate(moves):
            # Get current state
            game_state = game.game_state
            
            # Encode state
            board_encoded = encode_board_state(game_state)
            feature_vector = get_feature_vector(game_state)
            
            # Get model evaluation for this position
            position_value = game.evaluate_with_model(self.actor_critic.model, self.actor_critic.params)
            
            # Create dummy action probabilities (uniform over valid moves)
            # In practice, you might want to use a stronger policy or search
            valid_moves = game.get_valid_moves()
            action_probs = jnp.zeros(2187)
            if valid_moves:
                from src.utils.action_utils import action_to_index
                prob_per_move = 1.0 / len(valid_moves)
                for valid_move in valid_moves:
                    idx = action_to_index(valid_move)
                    action_probs = action_probs.at[idx].set(prob_per_move)
            
            example = TrainingExample(
                board_state=jnp.array(board_encoded),
                feature_vector=jnp.array(feature_vector),
                action_probs=action_probs,
                value=position_value,
                player=game.current_player.value
            )
            examples.append(example)
            
            # Apply move
            if not game.move(move):
                logger.warning(f"Invalid move {move} at position {move_idx}")
                break
            
            # Check if game ended
            if game.is_terminal:
                break
        
        return examples
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get processing statistics"""
        return {
            'games_processed': self.games_processed,
            'source_type': 'game_records'
        }


class AIOpponentDataSource(DataSource):
    """Data source from games against other AI engines"""
    
    def __init__(self, actor_critic: ActorCritic, opponent_engine_path: str):
        self.actor_critic = actor_critic
        self.opponent_engine_path = opponent_engine_path
        self.games_played = 0
    
    def generate_data(self, num_games: int) -> List[TrainingExample]:
        """
        Generate training data by playing against another AI
        
        Args:
            num_games: Number of games to play
            
        Returns:
            List of training examples
        """
        # This would require implementing USI engine communication
        # For now, this is a placeholder implementation
        logger.info(f"Playing {num_games} games against AI opponent: {self.opponent_engine_path}")
        
        examples = []
        
        # TODO: Implement actual AI vs AI gameplay
        # This would involve:
        # 1. Starting the opponent engine process
        # 2. Communicating via USI protocol
        # 3. Recording moves and positions
        # 4. Converting to training examples
        
        logger.warning("AI opponent data source not yet implemented")
        self.games_played = num_games
        
        return examples
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get AI opponent statistics"""
        return {
            'games_played': self.games_played,
            'opponent_engine': self.opponent_engine_path,
            'source_type': 'ai_opponent'
        }


class DataGenerationManager:
    """Manages multiple data sources for training"""
    
    def __init__(self):
        self.data_sources: List[DataSource] = []
        self.generation_history = []
    
    def add_data_source(self, source: DataSource):
        """Add a data source"""
        self.data_sources.append(source)
        logger.info(f"Added data source: {type(source).__name__}")
    
    def generate_mixed_data(self, 
                           num_games_per_source: Dict[str, int]) -> List[TrainingExample]:
        """
        Generate training data from multiple sources
        
        Args:
            num_games_per_source: Dict mapping source type to number of games
            
        Returns:
            Combined training examples
        """
        all_examples = []
        source_stats = {}
        
        for source in self.data_sources:
            source_name = type(source).__name__
            
            if source_name in num_games_per_source:
                num_games = num_games_per_source[source_name]
                logger.info(f"Generating {num_games} games from {source_name}")
                
                if isinstance(source, GameRecordDataSource):
                    # Special handling for game records
                    logger.info("GameRecordDataSource requires game_records parameter")
                    continue
                
                examples = source.generate_data(num_games)
                all_examples.extend(examples)
                
                # Collect statistics
                stats = source.get_statistics()
                source_stats[source_name] = stats
        
        # Log combined statistics
        self._log_generation_summary(all_examples, source_stats)
        
        # Save generation history
        import datetime
        self.generation_history.append({
            'total_examples': len(all_examples),
            'source_stats': source_stats,
            'timestamp': datetime.datetime.now().isoformat()
        })
        
        return all_examples
    
    def generate_from_source(self, 
                           source_type: str, 
                           num_games: int,
                           **kwargs) -> List[TrainingExample]:
        """
        Generate data from a specific source type
        
        Args:
            source_type: Type of source ('SelfPlayDataSource', etc.)
            num_games: Number of games
            **kwargs: Additional arguments for specific sources
            
        Returns:
            Training examples
        """
        for source in self.data_sources:
            if type(source).__name__ == source_type:
                if isinstance(source, GameRecordDataSource) and 'game_records' in kwargs:
                    return source.generate_data(kwargs['game_records'])
                else:
                    return source.generate_data(num_games)
        
        raise ValueError(f"Data source not found: {source_type}")
    
    def get_all_statistics(self) -> Dict[str, Any]:
        """Get statistics from all data sources"""
        stats = {}
        for source in self.data_sources:
            source_name = type(source).__name__
            stats[source_name] = source.get_statistics()
        
        stats['generation_history'] = self.generation_history
        return stats
    
    def _log_generation_summary(self, 
                               examples: List[TrainingExample], 
                               source_stats: Dict[str, Any]):
        """Log summary of data generation"""
        logger.info("=== Data Generation Summary ===")
        logger.info(f"Total examples generated: {len(examples)}")
        
        for source_name, stats in source_stats.items():
            logger.info(f"{source_name}:")
            for key, value in stats.items():
                logger.info(f"  {key}: {value}")
        
        logger.info("==============================")


# Convenience functions
def create_self_play_data_source(actor_critic: ActorCritic, **config_overrides) -> SelfPlayDataSource:
    """Create self-play data source"""
    mcts_config = config_overrides.get('mcts_config', {})
    return SelfPlayDataSource(actor_critic, mcts_config)


def create_game_record_data_source(actor_critic: ActorCritic) -> GameRecordDataSource:
    """Create game record data source"""
    return GameRecordDataSource(actor_critic)


def create_ai_opponent_data_source(actor_critic: ActorCritic, opponent_engine_path: str) -> AIOpponentDataSource:
    """Create AI opponent data source"""
    return AIOpponentDataSource(actor_critic, opponent_engine_path)