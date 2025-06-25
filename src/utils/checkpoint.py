"""
Checkpoint utilities for saving and loading model parameters and training state
"""

import os
import pickle
import json
from datetime import datetime
from typing import Dict, Any, Optional, Tuple
import jax
import jax.numpy as jnp
from pathlib import Path

class CheckpointManager:
    """Manages saving and loading of model checkpoints"""
    
    def __init__(self, checkpoint_dir: str = "data/checkpoints"):
        """
        Initialize checkpoint manager
        
        Args:
            checkpoint_dir: Directory to save checkpoints
        """
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
    def save_checkpoint(self, 
                       params: Dict, 
                       step: int,
                       optimizer_state: Optional[Dict] = None,
                       metadata: Optional[Dict] = None) -> str:
        """
        Save model parameters and training state
        
        Args:
            params: Model parameters
            step: Training step number
            optimizer_state: Optimizer state (optional)
            metadata: Additional metadata (optional)
            
        Returns:
            Path to saved checkpoint
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        checkpoint_name = f"checkpoint_step_{step}_{timestamp}"
        checkpoint_path = self.checkpoint_dir / checkpoint_name
        checkpoint_path.mkdir(exist_ok=True)
        
        # Save parameters
        params_path = checkpoint_path / "params.pkl"
        with open(params_path, 'wb') as f:
            pickle.dump(params, f)
            
        # Save optimizer state if provided
        if optimizer_state is not None:
            optimizer_path = checkpoint_path / "optimizer_state.pkl"
            with open(optimizer_path, 'wb') as f:
                pickle.dump(optimizer_state, f)
                
        # Save metadata
        checkpoint_metadata = {
            'step': step,
            'timestamp': timestamp,
            'jax_version': jax.__version__,
            'params_shape': self._get_params_info(params),
            'has_optimizer_state': optimizer_state is not None
        }
        
        if metadata:
            checkpoint_metadata.update(metadata)
            
        metadata_path = checkpoint_path / "metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(checkpoint_metadata, f, indent=2)
            
        # Create symlink to latest checkpoint
        latest_path = self.checkpoint_dir / "latest"
        if latest_path.exists() or latest_path.is_symlink():
            latest_path.unlink()
        latest_path.symlink_to(checkpoint_name)
        
        print(f"Checkpoint saved: {checkpoint_path}")
        return str(checkpoint_path)
    
    def load_checkpoint(self, 
                       checkpoint_path: Optional[str] = None,
                       load_optimizer_state: bool = True) -> Tuple[Dict, Optional[Dict], Dict]:
        """
        Load model parameters and training state
        
        Args:
            checkpoint_path: Path to checkpoint (if None, loads latest)
            load_optimizer_state: Whether to load optimizer state
            
        Returns:
            Tuple of (params, optimizer_state, metadata)
        """
        if checkpoint_path is None:
            checkpoint_path = self.checkpoint_dir / "latest"
            if not checkpoint_path.exists():
                raise FileNotFoundError("No checkpoints found")
            checkpoint_path = checkpoint_path.readlink()
            checkpoint_path = self.checkpoint_dir / checkpoint_path
        else:
            checkpoint_path = Path(checkpoint_path)
            
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
            
        # Load parameters
        params_path = checkpoint_path / "params.pkl"
        with open(params_path, 'rb') as f:
            params = pickle.load(f)
            
        # Load optimizer state
        optimizer_state = None
        if load_optimizer_state:
            optimizer_path = checkpoint_path / "optimizer_state.pkl"
            if optimizer_path.exists():
                with open(optimizer_path, 'rb') as f:
                    optimizer_state = pickle.load(f)
                    
        # Load metadata
        metadata_path = checkpoint_path / "metadata.json"
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
            
        print(f"Checkpoint loaded: {checkpoint_path}")
        return params, optimizer_state, metadata
    
    def list_checkpoints(self) -> list:
        """List all available checkpoints"""
        checkpoints = []
        for path in self.checkpoint_dir.iterdir():
            if path.is_dir() and path.name.startswith("checkpoint_"):
                metadata_path = path / "metadata.json"
                if metadata_path.exists():
                    with open(metadata_path, 'r') as f:
                        metadata = json.load(f)
                    checkpoints.append({
                        'path': str(path),
                        'name': path.name,
                        'step': metadata.get('step', 0),
                        'timestamp': metadata.get('timestamp', ''),
                        'has_optimizer_state': metadata.get('has_optimizer_state', False)
                    })
        
        # Sort by step number
        checkpoints.sort(key=lambda x: x['step'])
        return checkpoints
    
    def delete_old_checkpoints(self, keep_n: int = 5):
        """
        Delete old checkpoints, keeping only the latest N
        
        Args:
            keep_n: Number of checkpoints to keep
        """
        checkpoints = self.list_checkpoints()
        if len(checkpoints) <= keep_n:
            return
            
        # Sort by step (oldest first)
        checkpoints.sort(key=lambda x: x['step'])
        to_delete = checkpoints[:-keep_n]
        
        for checkpoint in to_delete:
            checkpoint_path = Path(checkpoint['path'])
            if checkpoint_path.exists():
                import shutil
                shutil.rmtree(checkpoint_path)
                print(f"Deleted old checkpoint: {checkpoint_path.name}")
    
    def _get_params_info(self, params: Dict) -> Dict:
        """Get information about parameter structure"""
        def get_shape_info(obj, prefix=""):
            info = {}
            if isinstance(obj, dict):
                for key, value in obj.items():
                    new_prefix = f"{prefix}.{key}" if prefix else key
                    info.update(get_shape_info(value, new_prefix))
            elif hasattr(obj, 'shape'):
                info[prefix] = {
                    'shape': list(obj.shape),
                    'dtype': str(obj.dtype)
                }
            return info
        
        return get_shape_info(params)


def save_model_checkpoint(model, params, step: int, 
                         optimizer_state=None, 
                         checkpoint_dir: str = "data/checkpoints",
                         metadata: Dict = None) -> str:
    """
    Convenience function to save model checkpoint
    
    Args:
        model: The model instance
        params: Model parameters
        step: Training step
        optimizer_state: Optimizer state
        checkpoint_dir: Directory for checkpoints
        metadata: Additional metadata
        
    Returns:
        Path to saved checkpoint
    """
    manager = CheckpointManager(checkpoint_dir)
    
    # Add model info to metadata
    if metadata is None:
        metadata = {}
    
    metadata.update({
        'model_type': model.__class__.__name__,
        'model_config': getattr(model, 'config', {}).__dict__ if hasattr(getattr(model, 'config', {}), '__dict__') else {}
    })
    
    return manager.save_checkpoint(params, step, optimizer_state, metadata)


def load_model_checkpoint(checkpoint_path: Optional[str] = None,
                         checkpoint_dir: str = "data/checkpoints",
                         load_optimizer_state: bool = True) -> Tuple[Dict, Optional[Dict], Dict]:
    """
    Convenience function to load model checkpoint
    
    Args:
        checkpoint_path: Specific checkpoint path (if None, loads latest)
        checkpoint_dir: Directory for checkpoints
        load_optimizer_state: Whether to load optimizer state
        
    Returns:
        Tuple of (params, optimizer_state, metadata)
    """
    if checkpoint_path is None:
        manager = CheckpointManager(checkpoint_dir)
        return manager.load_checkpoint(load_optimizer_state=load_optimizer_state)
    else:
        manager = CheckpointManager(checkpoint_dir)
        return manager.load_checkpoint(checkpoint_path, load_optimizer_state)