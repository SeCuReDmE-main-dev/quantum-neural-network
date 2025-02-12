"""
φ-framework initialization and integration module.
"""

from .core import PhiFramework
from .cohomology import PhiCyclicCohomology, CyclicCocycle
from .heisenberg import PhiHeisenbergCalculus, HeisenbergSymbol
from .zeta import PhiZetaFunctions, ZetaResult
from .operators import PhiDifferentialOperators
from .brain_quantum_service import BrainQuantumService, BrainRegionQuantumState, QuantumProcessingResult
from .brain_quantum_config import BrainQuantumConfig, QuantumRegionConfig

__version__ = '1.0.0'

def initialize_brain_quantum_framework(config_path: str = None) -> BrainQuantumService:
    """
    Initialize the complete brain quantum processing framework.
    
    Args:
        config_path: Optional path to configuration file
        
    Returns:
        Initialized BrainQuantumService
    """
    # Initialize configuration
    config = BrainQuantumConfig(config_path)
    
    # Optimize quantum resources
    config.optimize_quantum_resources()
    config.adjust_coherence_thresholds()
    
    # Initialize quantum service
    service = BrainQuantumService()
    
    # Apply configuration to service
    for region_name, region_config in config.config.items():
        if region_name in service.regions:
            service.regions[region_name].active_threshold = region_config.activation_threshold
            
    # Initial synchronization
    service.synchronize_quantum_states()
    
    return service

__all__ = [
    'PhiFramework',
    'PhiCyclicCohomology',
    'CyclicCocycle',
    'PhiHeisenbergCalculus',
    'HeisenbergSymbol',
    'PhiZetaFunctions',
    'ZetaResult',
    'PhiDifferentialOperators',
    'BrainQuantumService',
    'BrainRegionQuantumState',
    'QuantumProcessingResult',
    'BrainQuantumConfig',
    'QuantumRegionConfig',
    'initialize_brain_quantum_framework'
]