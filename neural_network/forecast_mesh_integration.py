import torch
import numpy as np
from typing import Dict, Any, List, Optional
from dataclasses import dataclass
from .neural_forecast import NeuralForecast, NeuralForecastConfig
from ..middleware.h2o_mesh_network import H2OMeshNetwork, MeshConfig
from ..middleware.datadrop_manager import DatadropLevel

@dataclass
class ForecastIntegrationConfig:
    """Configuration for forecast mesh integration"""
    datadrop_levels: List[DatadropLevel]
    forecast_configs: Dict[str, NeuralForecastConfig]
    mesh_config: Optional[MeshConfig] = None
    enable_quantum_features: bool = True
    update_interval: float = 0.1

class ForecastMeshIntegration:
    """Integrates neural forecasts with H2O mesh network"""
    
    def __init__(self, config: ForecastIntegrationConfig):
        self.config = config
        
        # Initialize mesh network
        self.mesh = H2OMeshNetwork(config.mesh_config)
        
        # Initialize forecasts for each level
        self.forecasts = {}
        for level in config.datadrop_levels:
            self.forecasts[level] = {}
            for brain_part in self.mesh.datadrop.datadrops[level].keys():
                if brain_part in config.forecast_configs:
                    self.forecasts[level][brain_part] = NeuralForecast(
                        config.forecast_configs[brain_part]
                    )
                    
    async def update_forecasts(self, current_state: Dict[str, Any]) -> Dict[str, Any]:
        """Update forecasts with current system state"""
        results = {}
        
        for level in self.config.datadrop_levels:
            level_results = {}
            
            for brain_part, forecast in self.forecasts[level].items():
                try:
                    # Get input data for this brain part
                    input_data = self._prepare_forecast_input(
                        current_state,
                        level,
                        brain_part
                    )
                    
                    # Generate forecast
                    forecast_output, metrics = forecast.forecast(
                        torch.tensor(input_data)
                    )
                    
                    # Update mesh with forecast
                    await self.mesh.update_node(
                        level,
                        brain_part,
                        {'forecast': forecast_output.numpy()}
                    )
                    
                    level_results[brain_part] = {
                        'forecast': forecast_output.numpy().tolist(),
                        'metrics': metrics
                    }
                    
                except Exception as e:
                    print(f"Error updating forecast for {brain_part}: {e}")
                    level_results[brain_part] = {'error': str(e)}
                    
            results[level.name] = level_results
            
        # Propagate updates through mesh
        await self.mesh.propagate_mesh_updates()
        
        return results
        
    def _prepare_forecast_input(self,
                              current_state: Dict[str, Any],
                              level: DatadropLevel,
                              brain_part: str) -> np.ndarray:
        """Prepare input data for forecast"""
        # Get node data from mesh
        node_data = self.mesh.node_stats[brain_part].get('data', {})
        
        # Get relevant features for this brain part
        if brain_part in self.config.forecast_configs:
            config = self.config.forecast_configs[brain_part]
            input_size = config.input_size
            
            # Extract features from state and node data
            features = []
            
            # Add brain part specific state
            if brain_part in current_state:
                part_state = current_state[brain_part]
                features.extend(self._extract_features(part_state, input_size))
                
            # Add node data
            if node_data:
                node_features = self._extract_features(node_data, input_size)
                features.extend(node_features)
                
            # Pad if needed
            while len(features) < input_size:
                features.append(0.0)
                
            return np.array(features[:input_size])
            
        return np.zeros(self.config.forecast_configs.get(
            brain_part, NeuralForecastConfig(4, 8, 10)
        ).input_size)
        
    def _extract_features(self, data: Dict[str, Any], size: int) -> List[float]:
        """Extract numerical features from data"""
        features = []
        
        def process_value(value):
            if isinstance(value, (int, float)):
                return float(value)
            elif isinstance(value, (list, np.ndarray)):
                return float(np.mean(value))
            elif isinstance(value, dict):
                return float(np.mean([
                    process_value(v) for v in value.values()
                    if isinstance(v, (int, float, list, np.ndarray))
                ]))
            return 0.0
            
        # Process all values
        for value in data.values():
            if len(features) >= size:
                break
            features.append(process_value(value))
            
        return features
        
    async def train_forecasts(self, 
                            training_data: Dict[str, Any],
                            num_epochs: int = 10) -> Dict[str, Any]:
        """Train forecasts with provided data"""
        results = {}
        
        for level in self.config.datadrop_levels:
            level_results = {}
            
            for brain_part, forecast in self.forecasts[level].items():
                try:
                    # Prepare training data
                    input_data = self._prepare_forecast_input(
                        training_data,
                        level,
                        brain_part
                    )
                    
                    # Create target from next timestep if available
                    if f"{brain_part}_next" in training_data:
                        target = self._prepare_forecast_input(
                            training_data[f"{brain_part}_next"],
                            level,
                            brain_part
                        )
                    else:
                        target = input_data  # Autoencoder style if no next state
                        
                    # Train forecast
                    optimizer = torch.optim.Adam(forecast.network.parameters())
                    losses = []
                    
                    for epoch in range(num_epochs):
                        loss = forecast.train_step(
                            torch.tensor(input_data),
                            torch.tensor(target),
                            optimizer
                        )
                        losses.append(loss)
                        
                    level_results[brain_part] = {
                        'final_loss': losses[-1],
                        'loss_history': losses,
                        'state': forecast.get_state()
                    }
                    
                except Exception as e:
                    print(f"Error training forecast for {brain_part}: {e}")
                    level_results[brain_part] = {'error': str(e)}
                    
            results[level.name] = level_results
            
        return results
        
    def get_forecast_status(self) -> Dict[str, Any]:
        """Get status of all forecasts"""
        status = {}
        
        for level in self.config.datadrop_levels:
            level_status = {}
            
            for brain_part, forecast in self.forecasts[level].items():
                level_status[brain_part] = forecast.get_state()
                
            status[level.name] = level_status
            
        return {
            'forecasts': status,
            'mesh_metrics': self.mesh.get_mesh_metrics()
        }