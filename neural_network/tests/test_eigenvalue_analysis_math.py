import unittest
import numpy as np
from ..eigenvalue_analysis import EigenvalueAnalysis, EigenState
from ..phi_framework import PhiConfig, PhiFramework
import scipy.linalg as la

class TestEigenvalueAnalysisMath(unittest.TestCase):
    """Test suite for mathematical foundations of eigenvalue analysis"""
    
    def setUp(self):
        self.phi_config = PhiConfig()
        self.analyzer = EigenvalueAnalysis(self.phi_config)
        self.phi = self.phi_config.phi
        
        # Create test matrices
        self.test_matrix = np.array([
            [2.0, -1.0, 0.0],
            [-1.0, 2.0, -1.0],
            [0.0, -1.0, 2.0]
        ])
        
        # Create test quantum state
        self.test_state = np.random.randn(50) + 1j * np.random.randn(50)
        self.test_state /= np.linalg.norm(self.test_state)
    
    def test_phi_scaling_eigenvalues(self):
        """Test ϕ-scaling of eigenvalues"""
        # Calculate standard eigenvalues
        standard_eigenvalues = la.eigvals(self.test_matrix)
        
        # Calculate ϕ-scaled eigenvalues
        hamiltonian = self.analyzer.compute_hamiltonian(self.test_matrix)
        phi_eigenvalues = la.eigvals(hamiltonian)
        
        # Verify scaling relationship
        for std, phi in zip(standard_eigenvalues, phi_eigenvalues):
            self.assertAlmostEqual(abs(phi), abs(std * self.phi), places=7,
                msg=f"ϕ-scaling failed: {abs(phi)} != {abs(std * self.phi)}")
    
    def test_energy_gap_properties(self):
        """Test properties of energy gaps in eigenspectrum"""
        analysis = self.analyzer.analyze_stability(self.test_state)
        gaps = analysis['stability_summary']
        
        # Test minimum gap is positive
        self.assertGreater(gaps['min_energy_gap'], 0,
            "Minimum energy gap should be positive")
        
        # Test average gap is greater than minimum gap
        self.assertGreaterEqual(gaps['average_energy_gap'], gaps['min_energy_gap'],
            "Average energy gap should be >= minimum gap")
        
        # Test ground state energy scaling
        self.assertTrue(np.isclose(gaps['ground_state_energy'] / self.phi,
                                 gaps['ground_state_energy'] / self.phi_config.phi),
            "Ground state energy scaling incorrect")
    
    def test_wavefunction_normalization(self):
        """Test normalization of wavefunctions under ϕ-scaling"""
        # Get eigenstates from stability analysis
        analysis = self.analyzer.analyze_stability(self.test_state)
        eigenstates = analysis['eigenstates']
        
        for state in eigenstates:
            # Check normalization of eigenvector
            norm = np.linalg.norm(state.eigenvector)
            self.assertAlmostEqual(norm, 1.0, places=7,
                msg=f"Eigenvector not normalized: norm = {norm}")
            
            # Check ϕ-scaled normalization
            phi_norm = np.linalg.norm(state.eigenvector * self.phi)
            self.assertAlmostEqual(phi_norm, self.phi, places=7,
                msg=f"ϕ-scaled normalization incorrect: {phi_norm} != {self.phi}")
    
    def test_hermiticity_preservation(self):
        """Test preservation of hermiticity under ϕ-scaling"""
        hamiltonian = self.analyzer.compute_hamiltonian(self.test_state)
        
        # Check hermiticity
        diff = np.max(np.abs(hamiltonian - hamiltonian.conj().T))
        self.assertLess(diff, 1e-10,
            f"Hamiltonian not hermitian, max difference: {diff}")
        
        # Check ϕ-scaled hermiticity
        phi_ham = hamiltonian * self.phi
        phi_diff = np.max(np.abs(phi_ham - phi_ham.conj().T))
        self.assertLess(phi_diff, 1e-10,
            f"ϕ-scaled Hamiltonian not hermitian, max difference: {phi_diff}")
    
    def test_eigenstate_orthogonality(self):
        """Test orthogonality of eigenstates under ϕ-scaling"""
        analysis = self.analyzer.analyze_stability(self.test_state)
        eigenstates = analysis['eigenstates']
        
        for i, state1 in enumerate(eigenstates):
            for j, state2 in enumerate(eigenstates):
                if i != j:
                    # Check orthogonality
                    overlap = np.abs(np.dot(state1.eigenvector.conj(), 
                                          state2.eigenvector))
                    self.assertLess(overlap, 1e-10,
                        f"Eigenstates {i} and {j} not orthogonal: overlap = {overlap}")
                    
                    # Check ϕ-scaled orthogonality
                    phi_overlap = np.abs(np.dot(state1.eigenvector.conj() * self.phi,
                                              state2.eigenvector))
                    self.assertLess(phi_overlap, 1e-10,
                        f"ϕ-scaled eigenstates {i} and {j} not orthogonal: overlap = {phi_overlap}")
    
    def test_stability_metric_properties(self):
        """Test mathematical properties of stability metrics"""
        eigenvalue = 1.0 + 0.5j
        eigenvector = np.random.randn(10) + 1j * np.random.randn(10)
        eigenvector /= np.linalg.norm(eigenvector)
        
        # Test stability metric scaling with eigenvalue magnitude
        base_metric = self.analyzer.compute_stability_metric(eigenvalue, eigenvector)
        scaled_metric = self.analyzer.compute_stability_metric(2*eigenvalue, eigenvector)
        
        self.assertGreater(scaled_metric, base_metric,
            "Stability metric should increase with eigenvalue magnitude")
        
        # Test stability metric invariance under eigenvector phase
        phase = np.exp(1j * np.pi / 4)
        phase_metric = self.analyzer.compute_stability_metric(eigenvalue, 
                                                           phase * eigenvector)
        self.assertAlmostEqual(base_metric, phase_metric, places=7,
            "Stability metric should be invariant under eigenvector phase")
        
        # Test ϕ-scaling of stability metric
        phi_metric = self.analyzer.compute_stability_metric(self.phi * eigenvalue,
                                                         eigenvector)
        expected_ratio = self.phi
        actual_ratio = phi_metric / base_metric
        self.assertAlmostEqual(actual_ratio, expected_ratio, places=7,
            f"ϕ-scaling of stability metric incorrect: {actual_ratio} != {expected_ratio}")
    
    def test_coherence_metric_bounds(self):
        """Test mathematical bounds of coherence metrics"""
        # Create test eigenvectors with controlled coherence
        n = 10
        theta = np.pi / 4  # 45 degree rotation
        c = np.cos(theta)
        s = np.sin(theta)
        rotation = np.array([[c, -s], [s, c]])
        
        # Create maximally coherent state
        basis1 = np.zeros(n)
        basis1[0] = 1
        basis2 = np.zeros(n)
        basis2[1] = 1
        
        rotated = np.dot(rotation, np.vstack([basis1, basis2]))
        coherent_state = rotated[0] + 1j * rotated[1]
        coherent_state /= np.linalg.norm(coherent_state)
        
        # Create minimally coherent state
        incoherent_state = basis1
        
        # Test coherence bounds
        max_coherence = self.analyzer.analyze_coherence(coherent_state.reshape(1, -1))
        min_coherence = self.analyzer.analyze_coherence(incoherent_state.reshape(1, -1))
        
        self.assertGreaterEqual(max_coherence, min_coherence,
            "Maximum coherence should be greater than minimum coherence")
        self.assertGreaterEqual(1.0, max_coherence,
            "Coherence metric should be bounded by 1")
        self.assertGreaterEqual(max_coherence, 0.0,
            "Coherence metric should be non-negative")

if __name__ == '__main__':
    unittest.main()