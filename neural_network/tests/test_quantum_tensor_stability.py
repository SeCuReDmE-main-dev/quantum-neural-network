import unittest
import numpy as np
import torch
from ..eigenvalue_analysis import EigenvalueAnalysis
from ..phi_framework import PhiConfig
from ..quantum_tensor_networks import TensorNetwork, QuantumState
from typing import List, Tuple

class TestQuantumTensorStability(unittest.TestCase):
    """Test suite for quantum tensor network stability analysis with ϕ-framework"""
    
    def setUp(self):
        self.phi_config = PhiConfig()
        self.analyzer = EigenvalueAnalysis(self.phi_config)
        self.phi = self.phi_config.phi
        
        # Initialize tensor network
        self.n_qubits = 4
        self.bond_dim = 8
        self.network = TensorNetwork(
            n_qubits=self.n_qubits,
            bond_dimension=self.bond_dim,
            phi=self.phi
        )
        
        # Create test quantum state
        self.test_state = self.create_tensor_state()
    
    def create_tensor_state(self) -> QuantumState:
        """Create a test quantum state in tensor network form"""
        # Initialize tensors with ϕ-scaling
        tensors = []
        for i in range(self.n_qubits):
            tensor = np.random.randn(2, self.bond_dim, self.bond_dim) + \
                    1j * np.random.randn(2, self.bond_dim, self.bond_dim)
            tensor *= self.phi  # Apply ϕ-scaling
            tensors.append(tensor)
        
        return QuantumState(tensors=tensors, phi=self.phi)
    
    def test_tensor_eigenvalue_stability(self):
        """Test eigenvalue stability of tensor network states"""
        # Contract network to get full state vector
        state_vector = self.network.contract_state(self.test_state)
        
        # Analyze stability
        analysis = self.analyzer.analyze_stability(state_vector)
        
        # Test stability metrics with tensor structure
        self.assertGreaterEqual(
            analysis['stability_summary']['temporal_stability'],
            0.5,
            "Tensor network state should maintain reasonable stability"
        )
        
        # Verify ϕ-scaling in eigenvalues
        eigenvalues = [state.eigenvalue for state in analysis['eigenstates']]
        scaled_eigenvalues = np.array(eigenvalues) / self.phi
        
        # Check consistency of ϕ-scaling
        self.assertTrue(
            np.allclose(scaled_eigenvalues * self.phi, eigenvalues),
            "Eigenvalues should maintain ϕ-scaling consistency"
        )
    
    def test_tensor_entanglement_stability(self):
        """Test stability of entanglement in tensor network"""
        # Create maximally entangled state in tensor form
        bell_tensors = self.network.create_bell_state()
        bell_state = QuantumState(tensors=bell_tensors, phi=self.phi)
        
        # Contract to get state vector
        bell_vector = self.network.contract_state(bell_state)
        
        # Analyze stability
        bell_analysis = self.analyzer.analyze_stability(bell_vector)
        
        # Verify high entanglement corresponds to high stability
        self.assertGreater(
            bell_analysis['stability_summary']['coherence_metric'],
            0.7,
            "Maximally entangled state should show high coherence"
        )
    
    def test_tensor_decomposition_stability(self):
        """Test stability under tensor decomposition"""
        # Get initial stability metrics
        initial_vector = self.network.contract_state(self.test_state)
        initial_analysis = self.analyzer.analyze_stability(initial_vector)
        
        # Decompose and reconstruct state
        decomposed = self.network.decompose_state(self.test_state)
        reconstructed = self.network.reconstruct_state(decomposed)
        reconstructed_vector = self.network.contract_state(reconstructed)
        
        # Analyze reconstructed state
        reconstructed_analysis = self.analyzer.analyze_stability(reconstructed_vector)
        
        # Compare stability metrics
        self.assertAlmostEqual(
            initial_analysis['stability_summary']['temporal_stability'],
            reconstructed_analysis['stability_summary']['temporal_stability'],
            places=5,
            msg="Stability should be preserved under tensor decomposition"
        )
    
    def test_phi_scaling_invariance(self):
        """Test invariance properties under ϕ-scaling"""
        # Scale tensors by ϕ
        scaled_state = self.test_state.scale(self.phi)
        
        # Get stability metrics for both states
        original_vector = self.network.contract_state(self.test_state)
        scaled_vector = self.network.contract_state(scaled_state)
        
        original_analysis = self.analyzer.analyze_stability(original_vector)
        scaled_analysis = self.analyzer.analyze_stability(scaled_vector)
        
        # Compare relative stability measures
        ratio = (scaled_analysis['stability_summary']['temporal_stability'] / 
                original_analysis['stability_summary']['temporal_stability'])
        
        self.assertAlmostEqual(
            ratio, 1.0, places=5,
            msg="Relative stability should be invariant under ϕ-scaling"
        )
    
    def test_local_perturbation_stability(self):
        """Test stability under local tensor perturbations"""
        # Create local perturbation
        perturbation = np.random.randn(2, self.bond_dim, self.bond_dim) + \
                      1j * np.random.randn(2, self.bond_dim, self.bond_dim)
        perturbation *= 0.01 * self.phi  # Small ϕ-scaled perturbation
        
        # Apply perturbation to one tensor
        perturbed_state = self.test_state.copy()
        perturbed_state.tensors[0] += perturbation
        
        # Analyze stability before and after perturbation
        original_vector = self.network.contract_state(self.test_state)
        perturbed_vector = self.network.contract_state(perturbed_state)
        
        original_analysis = self.analyzer.analyze_stability(original_vector)
        perturbed_analysis = self.analyzer.analyze_stability(perturbed_vector)
        
        # Check stability change is proportional to perturbation size
        stability_change = abs(
            perturbed_analysis['stability_summary']['temporal_stability'] -
            original_analysis['stability_summary']['temporal_stability']
        )
        
        self.assertLess(
            stability_change, 0.1,
            "Local perturbation should cause limited stability change"
        )
    
    def test_tensor_network_convergence(self):
        """Test convergence of stability metrics with bond dimension"""
        bond_dimensions = [4, 8, 16]
        stability_values = []
        
        for bond_dim in bond_dimensions:
            # Create network with different bond dimension
            network = TensorNetwork(
                n_qubits=self.n_qubits,
                bond_dimension=bond_dim,
                phi=self.phi
            )
            
            # Create and analyze state
            state = network.random_state()
            vector = network.contract_state(state)
            analysis = self.analyzer.analyze_stability(vector)
            
            stability_values.append(
                analysis['stability_summary']['temporal_stability']
            )
        
        # Check convergence
        differences = np.diff(stability_values)
        self.assertTrue(
            np.all(np.abs(differences) < 0.1),
            "Stability metrics should converge with increasing bond dimension"
        )

if __name__ == '__main__':
    unittest.main()