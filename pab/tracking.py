"""
Checkpoint tracking utilities for Process-Aware Benchmarking (PAB).

This module provides functionality for saving and loading model checkpoints
along with their associated metrics.
"""

import os
import json
import pickle
import torch
import logging
from typing import Dict, List, Optional, Tuple, Union, Any, Iterator

logger = logging.getLogger(__name__)

class Checkpoint:
    """
    Represents a model checkpoint at a specific training epoch.
    """
    
    def __init__(
        self,
        model_state: Dict[str, torch.Tensor],
        epoch: int,
        metrics: Dict[str, Any],
        timestamp: Optional[float] = None
    ):
        """
        Initialize a checkpoint.
        
        Args:
            model_state: Model state dictionary
            epoch: Training epoch number
            metrics: Dictionary of metrics at this checkpoint
            timestamp: Optional timestamp when checkpoint was created
        """
        self.model_state = model_state
        self.epoch = epoch
        self.metrics = metrics
        self.timestamp = timestamp if timestamp is not None else time.time()
    
    def save(self, path: str):
        """
        Save checkpoint to disk.
        
        Args:
            path: Path to save the checkpoint
        """
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        # Save model state
        torch.save(self.model_state, f"{path}.pt")
        
        # Save metadata and metrics
        metadata = {
            'epoch': self.epoch,
            'timestamp': self.timestamp,
            'metrics': self.metrics
        }
        
        with open(f"{path}.json", 'w') as f:
            json.dump(metadata, f, indent=2)
    
    @classmethod
    def load(cls, path: str) -> 'Checkpoint':
        """
        Load checkpoint from disk.
        
        Args:
            path: Path to load the checkpoint from
            
        Returns:
            Loaded checkpoint object
        """
        # Load model state
        model_state = torch.load(f"{path}.pt", map_location='cpu')
        
        # Load metadata and metrics
        with open(f"{path}.json", 'r') as f:
            metadata = json.load(f)
        
        return cls(
            model_state=model_state,
            epoch=metadata['epoch'],
            metrics=metadata['metrics'],
            timestamp=metadata['timestamp']
        )


class CheckpointManager:
    """
    Manages saving and loading of model checkpoints.
    """
    
    def __init__(self, checkpoint_dir: str):
        """
        Initialize a CheckpointManager.
        
        Args:
            checkpoint_dir: Directory to store checkpoints
        """
        self.checkpoint_dir = checkpoint_dir
        os.makedirs(checkpoint_dir, exist_ok=True)
    
    def save_checkpoint(
        self,
        model: torch.nn.Module,
        epoch: int,
        metrics: Dict[str, Any]
    ) -> str:
        """
        Save a model checkpoint.
        
        Args:
            model: Model to save
            epoch: Current epoch number
            metrics: Dictionary of metrics to save with the checkpoint
            
        Returns:
            Path to the saved checkpoint
        """
        # Create checkpoint name
        checkpoint_name = f"checkpoint_epoch_{epoch:04d}"
        checkpoint_path = os.path.join(self.checkpoint_dir, checkpoint_name)
        
        # Create checkpoint object
        checkpoint = Checkpoint(
            model_state=model.state_dict(),
            epoch=epoch,
            metrics=metrics
        )
        
        # Save checkpoint
        checkpoint.save(checkpoint_path)
        
        return checkpoint_path
    
    def load_checkpoint(
        self,
        checkpoint_name: str
    ) -> Tuple[Dict[str, torch.Tensor], Dict[str, Any]]:
        """
        Load a model checkpoint.
        
        Args:
            checkpoint_name: Name of the checkpoint to load
            
        Returns:
            Tuple of (model_state_dict, metrics)
        """
        # Construct checkpoint path
        checkpoint_path = os.path.join(self.checkpoint_dir, checkpoint_name)
        
        # Load checkpoint
        try:
            checkpoint = Checkpoint.load(checkpoint_path)
            return checkpoint.model_state, checkpoint.metrics
        except FileNotFoundError:
            logger.error(f"Checkpoint {checkpoint_name} not found")
            raise
    
    def list_checkpoints(self) -> List[str]:
        """
        List all available checkpoints.
        
        Returns:
            List of checkpoint names
        """
        # List all JSON files (metadata files)
        try:
            json_files = [f[:-5] for f in os.listdir(self.checkpoint_dir) 
                          if f.endswith('.json')]
            
            # Verify that corresponding model files exist
            valid_checkpoints = []
            for checkpoint in json_files:
                if os.path.exists(os.path.join(self.checkpoint_dir, f"{checkpoint}.pt")):
                    valid_checkpoints.append(checkpoint)
            
            return sorted(valid_checkpoints)
        except FileNotFoundError:
            logger.warning(f"Checkpoint directory {self.checkpoint_dir} not found")
            return []
    
    def iter_checkpoints(
        self,
        every_n: int = 1,
        start_epoch: Optional[int] = None,
        end_epoch: Optional[int] = None
    ) -> Iterator[Tuple[int, Dict[str, torch.Tensor], Dict[str, Any]]]:
        """
        Iterate through checkpoints in epoch order.
        
        Args:
            every_n: Only yield every n-th checkpoint
            start_epoch: Optional starting epoch (inclusive)
            end_epoch: Optional ending epoch (inclusive)
            
        Yields:
            Tuples of (epoch, model_state_dict, metrics)
        """
        checkpoints = self.list_checkpoints()
        
        # Extract epochs from checkpoint names
        checkpoint_epochs = {}
        for checkpoint in checkpoints:
            try:
                # Assuming format "checkpoint_epoch_XXXX"
                epoch = int(checkpoint.split('_')[-1])
                checkpoint_epochs[checkpoint] = epoch
            except (ValueError, IndexError):
                logger.warning(f"Could not extract epoch from checkpoint name: {checkpoint}")
        
        # Sort checkpoints by epoch
        sorted_checkpoints = sorted(checkpoint_epochs.items(), key=lambda x: x[1])
        
        # Apply filters
        filtered_checkpoints = []
        for i, (checkpoint, epoch) in enumerate(sorted_checkpoints):
            if start_epoch is not None and epoch < start_epoch:
                continue
            if end_epoch is not None and epoch > end_epoch:
                continue
            if i % every_n == 0:
                filtered_checkpoints.append((checkpoint, epoch))
        
        # Yield checkpoints
        for checkpoint, epoch in filtered_checkpoints:
            model_state, metrics = self.load_checkpoint(checkpoint)
            yield epoch, model_state, metrics
    
    def get_latest_checkpoint(self) -> Optional[Tuple[Dict[str, torch.Tensor], Dict[str, Any]]]:
        """
        Get the latest checkpoint.
        
        Returns:
            Tuple of (model_state_dict, metrics) or None if no checkpoints exist
        """
        checkpoints = self.list_checkpoints()
        if not checkpoints:
            return None
        
        # Extract epochs from checkpoint names
        checkpoint_epochs = {}
        for checkpoint in checkpoints:
            try:
                # Assuming format "checkpoint_epoch_XXXX"
                epoch = int(checkpoint.split('_')[-1])
                checkpoint_epochs[checkpoint] = epoch
            except (ValueError, IndexError):
                logger.warning(f"Could not extract epoch from checkpoint name: {checkpoint}")
        
        # Find checkpoint with highest epoch
        latest_checkpoint = max(checkpoint_epochs.items(), key=lambda x: x[1])[0]
        
        return self.load_checkpoint(latest_checkpoint)
    
    def prune_checkpoints(
        self,
        keep_last_n: int = 5,
        keep_every_n: int = 10
    ) -> int:
        """
        Prune checkpoints to save disk space.
        
        Args:
            keep_last_n: Number of most recent checkpoints to keep
            keep_every_n: Keep every n-th checkpoint for earlier epochs
            
        Returns:
            Number of checkpoints deleted
        """
        checkpoints = self.list_checkpoints()
        if not checkpoints:
            return 0
        
        # Extract epochs from checkpoint names
        checkpoint_epochs = {}
        for checkpoint in checkpoints:
            try:
                # Assuming format "checkpoint_epoch_XXXX"
                epoch = int(checkpoint.split('_')[-1])
                checkpoint_epochs[checkpoint] = epoch
            except (ValueError, IndexError):
                logger.warning(f"Could not extract epoch from checkpoint name: {checkpoint}")
        
        # Sort checkpoints by epoch
        sorted_checkpoints = sorted(checkpoint_epochs.items(), key=lambda x: x[1])
        
        # Always keep the most recent checkpoints
        to_keep = sorted_checkpoints[-keep_last_n:]
        
        # For earlier checkpoints, keep every keep_every_n-th
        for i, (checkpoint, epoch) in enumerate(sorted_checkpoints[:-keep_last_n]):
            if i % keep_every_n == 0:
                to_keep.append((checkpoint, epoch))
        
        # Delete checkpoints not in to_keep
        to_keep_names = [c[0] for c in to_keep]
        deleted_count = 0
        
        for checkpoint in checkpoints:
            if checkpoint not in to_keep_names:
                # Delete model file
                model_path = os.path.join(self.checkpoint_dir, f"{checkpoint}.pt")
                if os.path.exists(model_path):
                    os.remove(model_path)
                
                # Delete metadata file
                meta_path = os.path.join(self.checkpoint_dir, f"{checkpoint}.json")
                if os.path.exists(meta_path):
                    os.remove(meta_path)
                
                deleted_count += 1
        
        return deleted_count
