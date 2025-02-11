import torch
import numpy as np
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass
from .phi_framework import PhiConfig

@dataclass
class NeuralForecastConfig:
    """Configuration for neural forecast"""
    input_size: int
    hidden_size: int
    forecast_horizon: int
    phi_scaling: bool = True
    quantum_features: bool = True
    memory_window: int = 100
    update_interval: float = 0.1

class NeuralForecast:
    """Neural forecast implementation with quantum integration"""
    
    def __init__(self, 
                 config: NeuralForecastConfig,
                 phi_config: Optional[PhiConfig] = None):
        self.config = config
        self.phi_config = phi_config or PhiConfig()
        
        # Initialize network
        self.network = torch.nn.Sequential(
            torch.nn.Linear(config.input_size, config.hidden_size),
            torch.nn.ReLU(),
            torch.nn.Linear(config.hidden_size, config.forecast_horizon)
        )
        
        # Initialize memory buffer
        self.memory_buffer = []
        
        # Track quantum states
        self.quantum_states = []
        
    def update_memory(self, state_vector: torch.Tensor):
        """Update memory buffer with new state"""
        self.memory_buffer.append(state_vector)
        
        # Apply phi-scaling if enabled
        if self.config.phi_scaling:
            state_vector *= self.phi_config.phi
            
        # Keep fixed window size
        if len(self.memory_buffer) > self.config.memory_window:
            self.memory_buffer.pop(0)
            
    def get_memory_context(self) -> torch.Tensor:
        """Get context from memory buffer"""
        if not self.memory_buffer:
            return torch.zeros(self.config.input_size)
            
        # Average recent states
        recent_states = torch.stack(self.memory_buffer[-10:])
        return torch.mean(recent_states, dim=0)
        
    def prepare_quantum_features(self, input_tensor: torch.Tensor) -> torch.Tensor:
        """Extract quantum features from input"""
        if not self.config.quantum_features:
            return input_tensor
            
        # Create quantum state
        quantum_state = self._create_quantum_state(input_tensor)
        self.quantum_states.append(quantum_state)
        
        # Extract features
        features = self._extract_quantum_features(quantum_state)
        
        # Apply phi-scaling
        if self.config.phi_scaling:
            features *= self.phi_config.phi
            
        return features
        
    def _create_quantum_state(self, classical_data: torch.Tensor) -> np.ndarray:
        """Create quantum state from classical data"""
        # Normalize data
        normalized = classical_data / torch.norm(classical_data)
        
        # Create quantum state matrix
        size = self.config.input_size
        state = np.zeros((size, size), dtype=np.complex128)
        
        # Fill with data
        for i in range(size):
            for j in range(size):
                if i < len(normalized) and j < len(normalized):
                    state[i,j] = normalized[i] * normalized[j].conj()
                    
        return state
        
    def _extract_quantum_features(self, quantum_state: np.ndarray) -> torch.Tensor:
        """Extract features from quantum state"""
        # Get eigenvalues
        eigenvals = np.linalg.eigvals(quantum_state)
        
        # Calculate quantum metrics
        entropy = -np.sum(np.abs(eigenvals) * np.log2(np.abs(eigenvals) + 1e-10))
        purity = np.trace(quantum_state @ quantum_state.conj().T).real
        coherence = np.sum(np.abs(quantum_state - np.diag(np.diag(quantum_state))))
        
        # Combine features
        features = torch.tensor([
            entropy,
            purity, 
            coherence,
            np.abs(eigenvals).mean()
        ])
        
        return features
        
    def forecast(self, 
                input_data: torch.Tensor,
                memory_context: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """Generate forecast with quantum feature integration"""
        # Get memory context if not provided
        if memory_context is None:
            memory_context = self.get_memory_context()
            
        # Prepare quantum features
        if self.config.quantum_features:
            quantum_features = self.prepare_quantum_features(input_data)
            
            # Combine with input
            input_data = torch.cat([
                input_data,
                quantum_features,
                memory_context
            ])
            
        # Generate forecast
        with torch.no_grad():
            forecast = self.network(input_data)
            
        # Update memory
        self.update_memory(input_data)
        
        # Get quantum metrics
        metrics = self._calculate_quantum_metrics()
        
        return forecast, metrics
        
    def _calculate_quantum_metrics(self) -> Dict[str, float]:
        """Calculate quantum metrics for recent states"""
        if not self.quantum_states:
            return {}
            
        recent_states = self.quantum_states[-10:]
        
        # Calculate average metrics
        avg_entropy = np.mean([
            -np.sum(np.abs(np.linalg.eigvals(state)) * 
                   np.log2(np.abs(np.linalg.eigvals(state)) + 1e-10))
            for state in recent_states
        ])
        
        avg_purity = np.mean([
            np.trace(state @ state.conj().T).real
            for state in recent_states
        ])
        
        avg_coherence = np.mean([
            np.sum(np.abs(state - np.diag(np.diag(state))))
            for state in recent_states
        ])
        
        return {
            'entropy': float(avg_entropy),
            'purity': float(avg_purity),
            'coherence': float(avg_coherence),
            'phi_scaling': float(self.phi_config.phi)
        }
        
    def train_step(self, 
                  input_data: torch.Tensor,
                  target: torch.Tensor,
                  optimizer: torch.optim.Optimizer) -> float:
        """Perform single training step"""
        # Prepare features
        if self.config.quantum_features:
            quantum_features = self.prepare_quantum_features(input_data)
            memory_context = self.get_memory_context()
            
            input_data = torch.cat([
                input_data,
                quantum_features,
                memory_context
            ])
            
        # Forward pass
        output = self.network(input_data)
        loss = torch.nn.functional.mse_loss(output, target)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        return loss.item()
        
    def get_state(self) -> Dict[str, Any]:
        """Get current state of forecast model"""
        return {
            'memory_size': len(self.memory_buffer),
            'quantum_states': len(self.quantum_states),
            'config': {
                'input_size': self.config.input_size,
                'hidden_size': self.config.hidden_size,
                'forecast_horizon': self.config.forecast_horizon,
                'phi_scaling': self.config.phi_scaling,
                'quantum_features': self.config.quantum_features
            },
            'quantum_metrics': self._calculate_quantum_metrics()
        }