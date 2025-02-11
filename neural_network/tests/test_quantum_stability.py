import unittest
import numpy as np
import torch
from ..eigenvalue_analysis import EigenvalueAnalysis, EigenState
from ..phi_framework import PhiConfig
from ..quantum_neural_bridge import QuantumNeuralBridge

class TestQuantumStabilityMetrics(unittest.TestCase):
    """Test suite for quantum stability metrics and ϕ-framework integration"""
    
    def setUp(self):
        self.phi_config = PhiConfig()
        self.analyzer = EigenvalueAnalysis(self.phi_config)
        self.bridge = QuantumNeuralBridge(self.phi_config)
        
        # Create test quantum system
        self.n_qubits = 4
        self.test_state = self.create_test_quantum_state()
        
    def create_test_quantum_state(self) -> np.ndarray:
        """Create a normalized test quantum state"""
        state = np.random.randn(2**self.n_qubits) + 1j * np.random.randn(2**self.n_qubits)
        return state / np.linalg.norm(state)
    
    def test_quantum_stability_under_perturbation(self):
        """Test stability of quantum states under small perturbations"""
        # Initial stability analysis
        initial_analysis = self.analyzer.analyze_stability(self.test_state)
        initial_energy = initial_analysis['stability_summary']['ground_state_energy']
        
        # Apply small perturbation scaled by ϕ
        perturbation = self.phi_config.phi * 0.01 * (np.random.randn(len(self.test_state)) + 
                                                    1j * np.random.randn(len(self.test_state)))
        perturbed_state = self.test_state + perturbation
        perturbed_state /= np.linalg.norm(perturbed_state)
        
        # Analyze perturbed state
        perturbed_analysis = self.analyzer.analyze_stability(perturbed_state)
        perturbed_energy = perturbed_analysis['stability_summary']['ground_state_energy']
        
        # Energy difference should be proportional to perturbation size
        energy_diff = abs(perturbed_energy - initial_energy)
        self.assertLess(energy_diff, self.phi_config.phi * 0.1,
            f"Energy difference {energy_diff} too large for small perturbation")
    
    def test_entanglement_stability(self):
        """Test stability metrics for entangled states"""
        # Create maximally entangled state (Bell state)
        bell_state = np.zeros(4)
        bell_state[0] = 1 / np.sqrt(2)  # |00⟩
        bell_state[3] = 1 / np.sqrt(2)  # |11⟩
        
        # Create separable state
        separable_state = np.zeros(4)
        separable_state[0] = 1  # |00⟩
        
        # Analyze both states
        entangled_analysis = self.analyzer.analyze_stability(bell_state)
        separable_analysis = self.analyzer.analyze_stability(separable_state)
        
        # Entangled state should have higher coherence
        self.assertGreater(
            entangled_analysis['stability_summary']['coherence_metric'],
            separable_analysis['stability_summary']['coherence_metric'],
            "Entangled state should have higher coherence"
        )
    
    def test_quantum_neural_stability(self):
        """Test stability of quantum states after neural network processing"""
        # Convert quantum state to neural network input
        input_tensor = self.bridge.quantum_to_neural_mapping(self.test_state)
        
        # Process through neural network and back to quantum state
        particles = self.bridge.neural_to_quantum_mapping(input_tensor)
        
        # Analyze stability of processed state
        reconstructed_state = np.array([p.mass for p in particles])
        reconstructed_state = reconstructed_state / np.linalg.norm(reconstructed_state)
        
        processed_analysis = self.analyzer.analyze_stability(reconstructed_state)
        
        # Verify stability properties are preserved
        self.assertGreater(processed_analysis['stability_summary']['temporal_stability'], 0.5,
            "Quantum state should maintain stability after neural processing")
    
    def test_phi_scaled_decoherence(self):
        """Test decoherence effects with ϕ-scaling"""
        # Simulate decoherence by mixing with random environment
        environment = np.random.randn(len(self.test_state)) + 1j * np.random.randn(len(self.test_state))
        environment = environment / np.linalg.norm(environment)
        
        mixing_angles = np.linspace(0, np.pi/2, 10)
        coherence_values = []
        
        for theta in mixing_angles:
            # Mix system with environment
            mixed_state = np.cos(theta) * self.test_state + np.sin(theta) * environment
            mixed_state = mixed_state / np.linalg.norm(mixed_state)
            
            # Analyze mixed state
            analysis = self.analyzer.analyze_stability(mixed_state)
            coherence_values.append(analysis['stability_summary']['coherence_metric'])
        
        # Verify monotonic decrease in coherence
        for i in range(1, len(coherence_values)):
            self.assertLessEqual(coherence_values[i], coherence_values[i-1],
                "Coherence should decrease monotonically under decoherence")
    
    def test_eigenvalue_spacing_statistics(self):
        """Test eigenvalue spacing statistics with ϕ-scaling"""
        # Generate random Hermitian matrix
        n = 50
        matrix = np.random.randn(n, n) + 1j * np.random.randn(n, n)
        matrix = matrix + matrix.conj().T
        
        # Calculate eigenvalues
        eigenvalues = np.sort(np.real(np.linalg.eigvals(matrix)))
        spacings = np.diff(eigenvalues)
        
        # Calculate mean level spacing
        mean_spacing = np.mean(spacings)
        
        # Apply ϕ-scaling
        phi_spacings = spacings * self.phi_config.phi
        phi_mean_spacing = np.mean(phi_spacings)
        
        # Verify scaling relationship
        self.assertAlmostEqual(
            phi_mean_spacing / mean_spacing,
            self.phi_config.phi,
            places=7,
            msg="Level spacing statistics should follow ϕ-scaling"
        )
    
    def test_stability_under_unitary_evolution(self):
        """Test stability metrics under unitary time evolution"""
        # Create random Hermitian Hamiltonian
        hamiltonian = np.random.randn(len(self.test_state), len(self.test_state))
        hamiltonian = hamiltonian + hamiltonian.conj().T
        
        # Time evolution operator
        dt = 0.1
        U = np.exp(-1j * hamiltonian * dt * self.phi_config.phi)
        
        # Evolve state
        evolved_state = np.dot(U, self.test_state)
        
        # Analyze stability before and after evolution
        initial_analysis = self.analyzer.analyze_stability(self.test_state)
        evolved_analysis = self.analyzer.analyze_stability(evolved_state)
        
        # Verify stability metrics are preserved under unitary evolution
        self.assertAlmostEqual(
            initial_analysis['stability_summary']['temporal_stability'],
            evolved_analysis['stability_summary']['temporal_stability'],
            places=7,
            msg="Stability should be preserved under unitary evolution"
        )

if __name__ == '__main__':
    unittest.main()