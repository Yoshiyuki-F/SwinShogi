#!/usr/bin/env python3
"""
Test checkpoint functionality
"""

import sys
import os
import jax
import logging

# Add project root to path
sys.path.insert(0, os.path.abspath('.'))

from src.utils.checkpoint import CheckpointManager, save_model_checkpoint, load_model_checkpoint
from src.model.shogi_model import create_swin_shogi_model
from config.default_config import get_model_config

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_checkpoint_basic():
    """Test basic checkpoint save/load functionality"""
    logger.info("Testing basic checkpoint functionality...")
    
    # Create model and parameters
    rng = jax.random.PRNGKey(42)
    model_config = get_model_config()
    model, params = create_swin_shogi_model(rng, model_config)
    
    # Test checkpoint manager
    manager = CheckpointManager("data/test_checkpoints")
    
    # Save checkpoint
    step = 100
    metadata = {'test': True, 'loss': 0.5}
    checkpoint_path = manager.save_checkpoint(params, step, metadata=metadata)
    logger.info(f"Saved checkpoint: {checkpoint_path}")
    
    # Load checkpoint
    loaded_params, _, loaded_metadata = manager.load_checkpoint()
    logger.info(f"Loaded checkpoint metadata: {loaded_metadata}")
    
    # Verify parameters match
    def compare_params(p1, p2, path=""):
        if isinstance(p1, dict) and isinstance(p2, dict):
            for key in p1:
                if key in p2:
                    compare_params(p1[key], p2[key], f"{path}.{key}")
                else:
                    raise ValueError(f"Key {path}.{key} missing in loaded params")
        else:
            # Compare arrays
            import jax.numpy as jnp
            if not jnp.allclose(p1, p2, rtol=1e-6):
                raise ValueError(f"Parameters don't match at {path}")
    
    compare_params(params, loaded_params)
    logger.info("✅ Parameters match!")
    
    # List checkpoints
    checkpoints = manager.list_checkpoints()
    logger.info(f"Available checkpoints: {len(checkpoints)}")
    for cp in checkpoints:
        logger.info(f"  - {cp['name']} (step {cp['step']})")
    
    logger.info("✅ Basic checkpoint test passed!")

def test_convenience_functions():
    """Test convenience save/load functions"""
    logger.info("Testing convenience functions...")
    
    # Create model
    rng = jax.random.PRNGKey(123)
    model_config = get_model_config()
    model, params = create_swin_shogi_model(rng, model_config)
    
    # Save using convenience function
    checkpoint_path = save_model_checkpoint(
        model, params, step=200, 
        checkpoint_dir="data/test_checkpoints",
        metadata={'convenience_test': True}
    )
    logger.info(f"Saved using convenience function: {checkpoint_path}")
    
    # Load using convenience function
    loaded_params, _, loaded_metadata = load_model_checkpoint(
        checkpoint_dir="data/test_checkpoints"
    )
    logger.info(f"Loaded metadata: {loaded_metadata.get('convenience_test', False)}")
    
    logger.info("✅ Convenience functions test passed!")

if __name__ == "__main__":
    try:
        # Create test directory
        os.makedirs("data/test_checkpoints", exist_ok=True)
        
        test_checkpoint_basic()
        test_convenience_functions()
        
        logger.info("🎉 All checkpoint tests passed!")
        
    except Exception as e:
        logger.error(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)