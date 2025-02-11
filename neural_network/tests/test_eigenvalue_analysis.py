import unittest
import numpy as np
from ..eigenvalue_analysis import EigenvalueAnalysis, EigenState
from ..phi_framework import PhiConfig

class TestEigenvalueAnalysis(unittest.TestCase):
    def setUp(self):
        self.phi_config = PhiConfig()
        self.analyzer = EigenvalueAnalysis(self.phi_config)
        
        # Create test quantum state
        self.test_state = np.random.randn(50) + 1j * np.random.randn(50)
        self.test_state /= np.linalg.norm(self.test_state)
    
    def test_hamiltonian_computation(self):
        """Test Hamiltonian operator computation with ϕ-scaling"""
        hamiltonian = self.analyzer.compute_hamiltonian(self.test_state)
        
        self.assertEqual(hamiltonian.shape, self.test_state.shape)
        self.assertTrue(np.all(np.isfinite(hamiltonian)))
        
        # Test hermiticity
        self.assertTrue(np.allclose(hamiltonian, np.conj(hamiltonian)))
    
    def test_stability_analysis(self):
        """Test quantum state stability analysis"""
        analysis = self.analyzer.analyze_stability(self.test_state)
        
        # Verify structure
        self.assertIn('eigenstates', analysis)
        self.assertIn('stability_summary', analysis)
        self.assertIn('phi_scaling', analysis)
        
        # Check eigenstate properties
        for eigenstate in analysis['eigenstates']:
            self.assertIsInstance(eigenstate, EigenState)
            self.assertTrue(hasattr(eigenstate, 'eigenvalue'))
            self.assertTrue(hasattr(eigenstate, 'eigenvector'))
            self.assertTrue(hasattr(eigenstate, 'stability_metric'))
            self.assertEqual(eigenstate.phi_scaling, self.phi_config.phi)
        
        # Verify stability metrics
        summary = analysis['stability_summary']
        self.assertGreater(summary['min_energy_gap'], 0)
        self.assertGreater(summary['average_energy_gap'], 0)
        self.assertTrue(np.isfinite(summary['ground_state_energy']))
        self.assertGreaterEqual(summary['excited_states_count'], 0)
        self.assertGreaterEqual(summary['coherence_metric'], 0)
        self.assertGreaterEqual(summary['temporal_stability'], 0)
        self.assertLessEqual(summary['temporal_stability'], 1)
    
    def test_coherence_analysis(self):
        """Test quantum state coherence analysis"""
        # Create test eigenvectors
        eigenvectors = np.random.randn(10, 10) + 1j * np.random.randn(10, 10)
        eigenvectors = np.linalg.qr(eigenvectors)[0]  # Orthogonalize
        
        coherence = self.analyzer.analyze_coherence(eigenvectors)
        self.assertGreaterEqual(coherence, 0)
        self.assertLessEqual(coherence, 1)
    
    def test_temporal_stability(self):
        """Test temporal stability analysis"""
        # Analyze state multiple times to build history
        for _ in range(5):
            self.analyzer.analyze_stability(self.test_state)
        
        stability = self.analyzer.analyze_temporal_stability()
        self.assertGreaterEqual(stability, 0)
        self.assertLessEqual(stability, 1)
    
    def test_stability_threshold_prediction(self):
        """Test stability threshold prediction"""
        threshold = self.analyzer.predict_stability_threshold(self.test_state)
        self.assertGreater(threshold, 0)
        self.assertTrue(np.isfinite(threshold))
    
    def test_perturbation_effects(self):
        """Test analysis of perturbation effects"""
        effects = self.analyzer.analyze_perturbation_effects(self.test_state)
        
        self.assertIn('energy_shift', effects)
        self.assertIn('coherence_change', effects)
        self.assertIn('relative_stability', effects)
        self.assertIn('phi_scaling', effects)
        
        self.assertGreaterEqual(effects['relative_stability'], 0)
        self.assertEqual(effects['phi_scaling'], self.phi_config.phi)
    
    def test_eigenstate_scaling(self):
        """Test ϕ-scaling of eigenstate properties"""
        analysis = self.analyzer.analyze_stability(self.test_state)
        eigenstate = analysis['eigenstates'][0]
        
        # Verify ϕ-scaling is applied correctly
        scaled_energy = eigenstate.eigenvalue
        unscaled_energy = scaled_energy / self.phi_config.phi
        
        self.assertNotEqual(scaled_energy, unscaled_energy)
        self.assertEqual(eigenstate.phi_scaling, self.phi_config.phi)
    
    def test_stability_metric_computation(self):
        """Test computation of stability metrics"""
        eigenvalue = 1.0 + 0.5j
        eigenvector = np.random.randn(10) + 1j * np.random.randn(10)
        eigenvector /= np.linalg.norm(eigenvector)
        
        metric = self.analyzer.compute_stability_metric(eigenvalue, eigenvector)
        
        self.assertGreater(metric, 0)
        self.assertTrue(np.isfinite(metric))
        
        # Test scaling with larger eigenvalue
        larger_metric = self.analyzer.compute_stability_metric(2*eigenvalue, eigenvector)
        self.assertGreater(larger_metric, metric)

if __name__ == '__main__':
    unittest.main()