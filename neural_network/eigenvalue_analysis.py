import numpy as np
import scipy.linalg as la
from typing import Tuple, List, Optional, Dict
from dataclasses import dataclass
from .phi_framework import PhiFramework, PhiConfig

@dataclass
class EigenState:
    """Represents a quantum eigenstate with ϕ-scaled properties"""
    eigenvalue: complex
    eigenvector: np.ndarray
    stability_metric: float
    phi_scaling: float

class EigenvalueAnalysis:
    """Analyzes stability of quantum systems using eigenvalue analysis with ϕ-framework"""
    
    def __init__(self, phi_config: Optional[PhiConfig] = None):
        self.phi_framework = PhiFramework(phi_config or PhiConfig())
        self.phi = self.phi_framework.phi
        self.previous_states: List[EigenState] = []
    
    def compute_hamiltonian(self, quantum_state: np.ndarray) -> np.ndarray:
        """Compute Hamiltonian operator for quantum state"""
        # Apply ϕ-scaling to kinetic and potential terms
        kinetic_term = -0.5 * self.phi * np.gradient(np.gradient(quantum_state))
        potential_term = self.phi * np.abs(quantum_state)**2
        
        return kinetic_term + potential_term
    
    def analyze_stability(self, quantum_state: np.ndarray, 
                         tolerance: float = 1e-6) -> Dict[str, any]:
        """Analyze stability of quantum state through eigenvalue decomposition"""
        # Compute Hamiltonian with ϕ-scaling
        hamiltonian = self.compute_hamiltonian(quantum_state)
        
        # Perform eigenvalue decomposition
        eigenvalues, eigenvectors = la.eigh(hamiltonian)
        
        # Scale eigenvalues by ϕ for energy level analysis
        scaled_eigenvalues = eigenvalues * self.phi
        
        # Calculate stability metrics
        energy_gaps = np.diff(scaled_eigenvalues)
        min_gap = np.min(np.abs(energy_gaps))
        avg_gap = np.mean(np.abs(energy_gaps))
        
        # Analyze state coherence
        coherence = self.analyze_coherence(eigenvectors)
        
        # Create eigenstates
        eigenstates = [
            EigenState(
                eigenvalue=ev,
                eigenvector=evec,
                stability_metric=self.compute_stability_metric(ev, evec),
                phi_scaling=self.phi
            )
            for ev, evec in zip(scaled_eigenvalues, eigenvectors.T)
        ]
        
        # Store states for temporal analysis
        self.previous_states.append(eigenstates[0])  # Store ground state
        if len(self.previous_states) > 100:  # Keep last 100 states
            self.previous_states.pop(0)
        
        return {
            'eigenstates': eigenstates,
            'stability_summary': {
                'min_energy_gap': min_gap,
                'average_energy_gap': avg_gap,
                'ground_state_energy': scaled_eigenvalues[0],
                'excited_states_count': len(eigenvalues[eigenvalues > tolerance]),
                'coherence_metric': coherence,
                'temporal_stability': self.analyze_temporal_stability()
            },
            'phi_scaling': self.phi
        }
    
    def compute_stability_metric(self, eigenvalue: complex, 
                               eigenvector: np.ndarray) -> float:
        """Compute stability metric for eigenstate using ϕ-framework"""
        # Consider both eigenvalue magnitude and eigenvector structure
        eigenvalue_contribution = np.abs(eigenvalue) * self.phi
        vector_norm = np.linalg.norm(eigenvector)
        vector_uniformity = np.std(np.abs(eigenvector)) / np.mean(np.abs(eigenvector))
        
        # Combine metrics with ϕ-scaling
        stability = (eigenvalue_contribution / (1 + vector_uniformity)) * \
                   (vector_norm / self.phi)
        
        return float(stability)
    
    def analyze_coherence(self, eigenvectors: np.ndarray) -> float:
        """Analyze quantum state coherence using ϕ-framework"""
        # Calculate density matrix
        density_matrix = np.dot(eigenvectors, eigenvectors.conj().T)
        
        # Apply ϕ-scaling to coherence calculation
        off_diagonal = np.sum(np.abs(density_matrix - np.diag(np.diag(density_matrix))))
        coherence = off_diagonal * self.phi / (len(eigenvectors) - 1)
        
        return float(coherence)
    
    def analyze_temporal_stability(self) -> float:
        """Analyze stability across time using stored states"""
        if len(self.previous_states) < 2:
            return 1.0
        
        # Calculate stability based on eigenvalue and eigenvector changes
        stability_metrics = []
        for i in range(1, len(self.previous_states)):
            prev_state = self.previous_states[i-1]
            curr_state = self.previous_states[i]
            
            # Calculate changes scaled by ϕ
            eigenvalue_change = abs(curr_state.eigenvalue - prev_state.eigenvalue)
            vector_change = np.linalg.norm(curr_state.eigenvector - prev_state.eigenvector)
            
            # Combine changes with ϕ-scaling
            total_change = (eigenvalue_change + self.phi * vector_change) / 2
            stability_metrics.append(np.exp(-total_change))
        
        return float(np.mean(stability_metrics))
    
    def predict_stability_threshold(self, quantum_state: np.ndarray,
                                  time_steps: int = 100) -> float:
        """Predict stability threshold for quantum state"""
        current_analysis = self.analyze_stability(quantum_state)
        ground_state_energy = current_analysis['stability_summary']['ground_state_energy']
        
        # Calculate stability threshold using ϕ-scaling
        threshold = abs(ground_state_energy) * self.phi / np.sqrt(time_steps)
        
        return float(threshold)
    
    def analyze_perturbation_effects(self, quantum_state: np.ndarray,
                                   perturbation_strength: float = 0.1) -> Dict[str, float]:
        """Analyze effects of perturbations on quantum state stability"""
        # Apply small perturbation scaled by ϕ
        perturbed_state = quantum_state + \
                         perturbation_strength * self.phi * \
                         np.random.randn(*quantum_state.shape)
        
        # Analyze both original and perturbed states
        original_analysis = self.analyze_stability(quantum_state)
        perturbed_analysis = self.analyze_stability(perturbed_state)
        
        # Calculate stability changes
        energy_shift = abs(perturbed_analysis['stability_summary']['ground_state_energy'] -
                         original_analysis['stability_summary']['ground_state_energy'])
        
        coherence_change = abs(perturbed_analysis['stability_summary']['coherence_metric'] -
                             original_analysis['stability_summary']['coherence_metric'])
        
        return {
            'energy_shift': float(energy_shift),
            'coherence_change': float(coherence_change),
            'relative_stability': float(perturbed_analysis['stability_summary']['temporal_stability'] /
                                     original_analysis['stability_summary']['temporal_stability']),
            'phi_scaling': float(self.phi)
        }

# Example usage
if __name__ == "__main__":
    analyzer = EigenvalueAnalysis()
    
    # Create example quantum state
    quantum_state = np.random.randn(100) + 1j * np.random.randn(100)
    quantum_state /= np.linalg.norm(quantum_state)
    
    # Analyze stability
    stability_analysis = analyzer.analyze_stability(quantum_state)
    
    # Predict stability threshold
    threshold = analyzer.predict_stability_threshold(quantum_state)
    
    # Analyze perturbation effects
    perturbation_analysis = analyzer.analyze_perturbation_effects(quantum_state)
