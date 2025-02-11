import unittest
import numpy as np
import torch
from ..eigenvalue_analysis import EigenvalueAnalysis, EigenState
from ..phi_framework import PhiConfig, PhiFramework, DifferentialGeometry, NoncommutativeGeometry

class TestPhiDifferentialGeometry(unittest.TestCase):
    """Test suite for ϕ-based differential geometry and noncommutative aspects"""
    
    def setUp(self):
        self.phi_config = PhiConfig()
        self.phi_framework = PhiFramework(self.phi_config)
        self.diff_geom = DifferentialGeometry(self.phi_framework)
        self.noncomm_geom = NoncommutativeGeometry(self.phi_framework)
        self.analyzer = EigenvalueAnalysis(self.phi_config)
        
        # Create test manifold data
        self.test_vector_field = np.random.randn(10, 3)  # 10 points in 3D
        self.test_connection = np.random.randn(10, 3, 3)  # Connection 1-form
        self.test_path = np.random.randn(5, 3)  # Path with 5 points
    
    def test_connection_form_properties(self):
        """Test properties of ϕ-modified connection forms"""
        connection = self.diff_geom.connection_form(self.test_vector_field)
        
        # Test scaling by ϕ
        manual_connection = self.phi_framework.phi * np.gradient(self.test_vector_field)
        np.testing.assert_array_almost_equal(
            connection, manual_connection,
            err_msg="Connection form should scale correctly with ϕ"
        )
        
        # Test linearity
        scalar = 2.0
        scaled_connection = self.diff_geom.connection_form(scalar * self.test_vector_field)
        np.testing.assert_array_almost_equal(
            scaled_connection, scalar * connection,
            err_msg="Connection form should be linear"
        )
    
    def test_curvature_properties(self):
        """Test properties of ϕ-modified curvature"""
        curvature = self.diff_geom.curvature(self.test_connection)
        
        # Test antisymmetry
        for i in range(curvature.shape[1]):
            for j in range(curvature.shape[2]):
                self.assertAlmostEqual(
                    curvature[0,i,j], -curvature[0,j,i],
                    msg=f"Curvature should be antisymmetric at indices {i},{j}"
                )
        
        # Test ϕ-scaling
        scaled_curvature = self.diff_geom.curvature(self.test_connection * self.phi_framework.phi)
        expected_scaling = self.phi_framework.phi ** 2
        actual_ratio = np.max(np.abs(scaled_curvature)) / np.max(np.abs(curvature))
        self.assertAlmostEqual(
            actual_ratio, expected_scaling,
            places=7,
            msg="Curvature should scale quadratically with ϕ"
        )
    
    def test_parallel_transport(self):
        """Test ϕ-modified parallel transport"""
        # Create test vector
        vector = np.array([1.0, 0.0, 0.0])
        
        # Transport along path
        transported = self.diff_geom.parallel_transport(vector, self.test_path)
        
        # Test length preservation
        initial_length = np.linalg.norm(vector)
        final_length = np.linalg.norm(transported)
        self.assertAlmostEqual(
            initial_length, final_length,
            places=7,
            msg="Parallel transport should preserve vector length"
        )
        
        # Test ϕ-scaling behavior
        scaled_vector = vector * self.phi_framework.phi
        scaled_transported = self.diff_geom.parallel_transport(scaled_vector, self.test_path)
        self.assertAlmostEqual(
            np.linalg.norm(scaled_transported) / np.linalg.norm(transported),
            self.phi_framework.phi,
            places=7,
            msg="Parallel transport should preserve ϕ-scaling"
        )
    
    def test_quantum_metric_properties(self):
        """Test properties of ϕ-modified quantum metric"""
        # Create test states
        states = torch.randn(5, 10, dtype=torch.complex128)  # 5 states in 10D Hilbert space
        states = states / torch.linalg.norm(states, dim=1).unsqueeze(1)
        
        # Calculate quantum metric
        metric = self.noncomm_geom.quantum_metric(states)
        
        # Test positivity
        eigenvalues = torch.linalg.eigvals(metric)
        self.assertTrue(
            torch.all(torch.real(eigenvalues) > -1e-10),
            "Quantum metric should be positive semi-definite"
        )
        
        # Test hermiticity
        diff = torch.max(torch.abs(metric - metric.conj().transpose(-2, -1)))
        self.assertLess(diff, 1e-10, "Quantum metric should be Hermitian")
        
        # Test ϕ-scaling
        scaled_states = states * self.phi_framework.phi
        scaled_metric = self.noncomm_geom.quantum_metric(scaled_states)
        expected_ratio = self.phi_framework.phi ** 2
        actual_ratio = torch.max(torch.abs(scaled_metric)) / torch.max(torch.abs(metric))
        self.assertAlmostEqual(
            actual_ratio.item(), expected_ratio,
            places=7,
            msg="Quantum metric should scale quadratically with ϕ"
        )
    
    def test_spectral_action_properties(self):
        """Test properties of ϕ-modified spectral action"""
        # Create test Dirac operator
        n = 10
        dirac = torch.randn(n, n, dtype=torch.complex128)
        dirac = dirac + dirac.conj().transpose(-2, -1)  # Make Hermitian
        
        # Calculate spectral action
        cutoff = 1.0
        action = self.noncomm_geom.spectral_action(dirac, cutoff)
        
        # Test positivity
        self.assertGreaterEqual(action.item(), 0, "Spectral action should be non-negative")
        
        # Test ϕ-scaling
        scaled_dirac = dirac * self.phi_framework.phi
        scaled_action = self.noncomm_geom.spectral_action(scaled_dirac, cutoff)
        
        # The spectral action should scale exponentially with ϕ due to the exponential cutoff
        log_ratio = torch.log(scaled_action / action)
        expected_scaling = -self.phi_framework.phi
        self.assertAlmostEqual(
            log_ratio.item() / expected_scaling,
            1.0,
            places=1,  # Less precise due to exponential behavior
            msg="Spectral action should scale correctly with ϕ"
        )
    
    def test_bianchi_identity(self):
        """Test Bianchi identity for ϕ-modified curvature"""
        # Calculate curvature
        curvature = self.diff_geom.curvature(self.test_connection)
        
        # Calculate cyclic sum (Bianchi identity)
        bianchi_sum = np.zeros_like(curvature[0])
        for i in range(3):
            j = (i + 1) % 3
            k = (i + 2) % 3
            bianchi_sum += np.gradient(curvature[0,:,:,i], axis=j)[...,k]
        
        # Test if Bianchi identity is satisfied up to ϕ-scaling
        max_violation = np.max(np.abs(bianchi_sum))
        self.assertLess(
            max_violation, self.phi_framework.phi * 1e-10,
            msg="Bianchi identity should be satisfied up to ϕ-scaling"
        )

if __name__ == '__main__':
    unittest.main()