#!/usr/bin/env python3
"""
Minimal test script to run a very short self-play game
"""

import sys
import os
import jax
import logging

# Add project root to path
sys.path.insert(0, os.path.abspath('..'))


from config.default_config import get_model_config

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def main():
    """Run a minimal self-play game test"""
    logger.info("Starting minimal self-play test...")
    
    # Initialize model
    rng = jax.random.PRNGKey(42)
    model_config = get_model_config()
    
    logger.info("Creating SwinShogi model...")
    model, params = create_swin_shogi_model(rng, model_config)
    logger.info("Model created successfully")
    
    # Create self-play with very minimal settings
    logger.info("Creating SelfPlay instance...")
    self_play = create_self_play(
        model, 
        params,
        n_simulations=10,      # Minimal simulations
        max_moves=300,
        temperature_init=1.0,
        temperature_final=0.1,
        temperature_threshold=2
    )
    logger.info("SelfPlay instance created")
    
    # Play a single game with board display
    logger.info("Starting minimal self-play game...")
    try:
        print("Starting self-play game...\n")
        
        result = self_play.play_game(verbose=True)
        
        logger.info("Game completed!")
        logger.info(f"  Winner: {'Player ' + str(result.winner) if result.winner is not None else 'Draw'}")
        logger.info(f"  Game length: {result.game_length} moves")
        logger.info(f"  Training examples: {len(result.examples)}")
        
        # Convert to training examples
        training_data = self_play.results_to_training_examples([result])
        logger.info(f"Generated {len(training_data)} training examples")
        
        # Show first example details
        if training_data:
            example = training_data[0]
            logger.info("First training example:")
            logger.info(f"  board_state shape: {example['board_state'].shape}")
            logger.info(f"  feature_vector shape: {example['feature_vector'].shape}")
            logger.info(f"  action_probs shape: {example['action_probs'].shape}")
            logger.info(f"  value: {example['value']}")
            logger.info(f"  player: {example['player']}")
        
        # Show statistics
        stats = self_play.get_statistics()
        logger.info("Self-play statistics:")
        for key, value in stats.items():
            if isinstance(value, float):
                logger.info(f"  {key}: {value:.3f}")
            else:
                logger.info(f"  {key}: {value}")
        
        logger.info("✅ Self-play test completed successfully!")
        return True
        
    except Exception as e:
        logger.error(f"❌ Error during self-play: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)