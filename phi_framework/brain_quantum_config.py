"""
Configuration management for brain quantum specialization.
"""

import json
import os
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional
import numpy as np

@dataclass
class QuantumRegionConfig:
    """Configuration for a brain region's quantum processing."""
    activation_threshold: float
    learning_rate: float
    entanglement_weight: float
    quantum_channels: int
    processing_priority: int
    state_vector_size: int
    coherence_time: float
    error_threshold: float

class BrainQuantumConfig:
    """Configuration manager for brain quantum processing."""
    
    DEFAULT_CONFIG = {
        'cerebrum': {
            'activation_threshold': 0.7,
            'learning_rate': 0.1,
            'entanglement_weight': 0.8,
            'quantum_channels': 16,
            'processing_priority': 1,
            'state_vector_size': 8,
            'coherence_time': 1000.0,
            'error_threshold': 0.01
        },
        'limbic_system': {
            'activation_threshold': 0.6,
            'learning_rate': 0.15,
            'entanglement_weight': 0.7,
            'quantum_channels': 12,
            'processing_priority': 2,
            'state_vector_size': 6,
            'coherence_time': 800.0,
            'error_threshold': 0.02
        },
        'brainstem': {
            'activation_threshold': 0.8,
            'learning_rate': 0.05,
            'entanglement_weight': 0.9,
            'quantum_channels': 8,
            'processing_priority': 3,
            'state_vector_size': 4,
            'coherence_time': 1200.0,
            'error_threshold': 0.005
        },
        'deep_layer': {
            'activation_threshold': 0.5,
            'learning_rate': 0.2,
            'entanglement_weight': 0.6,
            'quantum_channels': 24,
            'processing_priority': 4,
            'state_vector_size': 12,
            'coherence_time': 600.0,
            'error_threshold': 0.03
        },
        'dream_processor': {
            'activation_threshold': 0.4,
            'learning_rate': 0.25,
            'entanglement_weight': 0.5,
            'quantum_channels': 32,
            'processing_priority': 5,
            'state_vector_size': 16,
            'coherence_time': 400.0,
            'error_threshold': 0.04
        }
    }
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize brain quantum configuration.
        
        Args:
            config_path: Optional path to configuration file
        """
        self.config: Dict[str, QuantumRegionConfig] = {}
        self._load_config(config_path)
        
    def _load_config(self, config_path: Optional[str]) -> None:
        """Load configuration from file or use defaults."""
        if config_path and os.path.exists(config_path):
            with open(config_path, 'r') as f:
                config_data = json.load(f)
        else:
            config_data = self.DEFAULT_CONFIG
            
        for region, params in config_data.items():
            self.config[region] = QuantumRegionConfig(**params)
            
    def save_config(self, config_path: str) -> None:
        """
        Save current configuration to file.
        
        Args:
            config_path: Path to save configuration
        """
        config_data = {
            region: asdict(params)
            for region, params in self.config.items()
        }
        
        os.makedirs(os.path.dirname(config_path), exist_ok=True)
        with open(config_path, 'w') as f:
            json.dump(config_data, f, indent=4)
            
    def get_region_config(self, region: str) -> QuantumRegionConfig:
        """
        Get configuration for a specific brain region.
        
        Args:
            region: Name of brain region
            
        Returns:
            Configuration for the region
        """
        return self.config[region]
    
    def update_region_config(self, region: str, **kwargs) -> None:
        """
        Update configuration parameters for a brain region.
        
        Args:
            region: Name of brain region
            **kwargs: Configuration parameters to update
        """
        config = asdict(self.config[region])
        config.update(kwargs)
        self.config[region] = QuantumRegionConfig(**config)
        
    def optimize_quantum_resources(self) -> None:
        """Optimize quantum resource allocation across regions."""
        total_channels = sum(
            config.quantum_channels for config in self.config.values()
        )
        total_priority = sum(
            config.processing_priority for config in self.config.values()
        )
        
        # Redistribute channels based on priority
        for region, config in self.config.items():
            priority_ratio = config.processing_priority / total_priority
            optimal_channels = int(total_channels * priority_ratio)
            self.update_region_config(
                region, 
                quantum_channels=optimal_channels
            )
            
    def adjust_coherence_thresholds(self) -> None:
        """Adjust coherence thresholds based on quantum channels."""
        for region, config in self.config.items():
            channel_factor = config.quantum_channels / 32  # Normalize to max channels
            coherence_scale = np.exp(-0.5 * (1 - channel_factor))
            
            self.update_region_config(
                region,
                coherence_time=config.coherence_time * coherence_scale,
                error_threshold=config.error_threshold / coherence_scale
            )

if __name__ == '__main__':
    # Example usage
    config = BrainQuantumConfig()
    
    # Get configuration for cerebrum
    cerebrum_config = config.get_region_config('cerebrum')
    print(f"Cerebrum configuration:")
    print(f"Quantum channels: {cerebrum_config.quantum_channels}")
    print(f"Coherence time: {cerebrum_config.coherence_time}ms")
    
    # Optimize resources
    config.optimize_quantum_resources()
    config.adjust_coherence_thresholds()
    
    # Save updated configuration
    config.save_config('brain_quantum_config.json')