import numpy as np
import torch
from dataclasses import dataclass
from typing import Optional, Tuple, List
from .phi_framework import PhiFramework, PhiConfig

@dataclass
class CubicParticle:
    """Represents a particle with cubic properties"""
    mass: float
    momentum: np.ndarray  # 3D vector
    charge: float
    spin: float
    cubic_dimension: float
    position: np.ndarray  # 3D vector
    corners: List[np.ndarray]  # List of 8 3D vectors representing corners

    def __post_init__(self):
        if len(self.corners) != 8:
            # Initialize corners based on cubic_dimension
            self.corners = self._generate_corners()
    
    def _generate_corners(self) -> List[np.ndarray]:
        """Generate corner positions based on cubic dimension"""
        l = self.cubic_dimension / 2
        corners = []
        for x in [-l, l]:
            for y in [-l, l]:
                for z in [-l, l]:
                    corners.append(self.position + np.array([x, y, z]))
        return corners

class CubicFramework:
    """Implementation of the Cubic Framework quantum theory"""
    
    def __init__(self, phi_framework: PhiFramework):
        self.phi = phi_framework
        self.particles: List[CubicParticle] = []
    
    def create_particle(self, mass: float, momentum: np.ndarray, 
                       charge: float, spin: float, 
                       cubic_dimension: float, 
                       position: np.ndarray) -> CubicParticle:
        """Create a new cubic particle"""
        particle = CubicParticle(
            mass=mass,
            momentum=momentum,
            charge=charge,
            spin=spin,
            cubic_dimension=cubic_dimension,
            position=position,
            corners=[]
        )
        self.particles.append(particle)
        return particle
    
    def calculate_hypotenuse(self, p1: CubicParticle, p2: CubicParticle) -> float:
        """Calculate hypotenuse length between two particles"""
        distance = np.linalg.norm(p1.position - p2.position)
        return np.sqrt(p1.cubic_dimension**2 + distance**2)
    
    def check_corner_interaction(self, p1: CubicParticle, p2: CubicParticle, 
                               threshold: float = 1e-6) -> List[Tuple[int, int]]:
        """Check for interacting corners between particles"""
        interactions = []
        for i, c1 in enumerate(p1.corners):
            for j, c2 in enumerate(p2.corners):
                if np.linalg.norm(c1 - c2) < threshold:
                    interactions.append((i, j))
        return interactions
    
    def entangle_particles(self, p1: CubicParticle, p2: CubicParticle) -> None:
        """Entangle two particles through corner interactions"""
        interactions = self.check_corner_interaction(p1, p2)
        if interactions:
            # Modify properties based on entanglement
            self._modify_entangled_properties(p1, p2, interactions)
    
    def _modify_entangled_properties(self, p1: CubicParticle, p2: CubicParticle, 
                                   interactions: List[Tuple[int, int]]) -> None:
        """Modify particle properties based on entanglement"""
        # Conservation of mass
        total_mass = p1.mass + p2.mass
        p1.mass = p2.mass = total_mass / 2
        
        # Conservation of momentum
        total_momentum = p1.momentum + p2.momentum
        p1.momentum = p2.momentum = total_momentum / 2
        
        # Charge interaction
        charge_transfer = self.phi.phi * min(abs(p1.charge), abs(p2.charge))
        if p1.charge > p2.charge:
            p1.charge -= charge_transfer
            p2.charge += charge_transfer
        else:
            p1.charge += charge_transfer
            p2.charge -= charge_transfer
    
    def calculate_wavefunction(self, particle: CubicParticle, 
                             grid_points: int = 100) -> np.ndarray:
        """Calculate quantum wavefunction for a cubic particle"""
        # Create 3D grid
        x = y = z = np.linspace(-2*particle.cubic_dimension, 
                               2*particle.cubic_dimension, 
                               grid_points)
        X, Y, Z = np.meshgrid(x, y, z)
        
        # Calculate wavefunction based on cubic geometry
        psi = np.zeros((grid_points, grid_points, grid_points), dtype=complex)
        
        for i in range(grid_points):
            for j in range(grid_points):
                for k in range(grid_points):
                    r = np.sqrt((X[i,j,k] - particle.position[0])**2 +
                              (Y[i,j,k] - particle.position[1])**2 +
                              (Z[i,j,k] - particle.position[2])**2)
                    # Wavefunction decays with distance, modulated by φ
                    psi[i,j,k] = np.exp(-self.phi.phi * r) * \
                                np.exp(1j * np.dot(particle.momentum, 
                                                 [X[i,j,k], Y[i,j,k], Z[i,j,k]]))
        
        return psi
    
    def simulate_collision(self, p1: CubicParticle, p2: CubicParticle, 
                         dt: float = 0.01, steps: int = 100) -> List[Tuple[np.ndarray, np.ndarray]]:
        """Simulate collision between two cubic particles"""
        trajectories = []
        for _ in range(steps):
            # Update positions based on momentum
            p1.position += p1.momentum * dt / p1.mass
            p2.position += p2.momentum * dt / p2.mass
            
            # Check for interactions
            if self.check_corner_interaction(p1, p2):
                self.entangle_particles(p1, p2)
            
            trajectories.append((p1.position.copy(), p2.position.copy()))
            
            # Update corners after position change
            p1.corners = p1._generate_corners()
            p2.corners = p2._generate_corners()
        
        return trajectories

# Example usage
if __name__ == "__main__":
    config = PhiConfig()
    phi_framework = PhiFramework(config)
    cubic_framework = CubicFramework(phi_framework)
    
    # Create two particles
    p1 = cubic_framework.create_particle(
        mass=1.0,
        momentum=np.array([0.1, 0, 0]),
        charge=1.0,
        spin=0.5,
        cubic_dimension=1.0,
        position=np.array([-2.0, 0, 0])
    )
    
    p2 = cubic_framework.create_particle(
        mass=1.0,
        momentum=np.array([-0.1, 0, 0]),
        charge=-1.0,
        spin=-0.5,
        cubic_dimension=1.0,
        position=np.array([2.0, 0, 0])
    )
    
    # Simulate collision
    trajectories = cubic_framework.simulate_collision(p1, p2)
    
    # Calculate wavefunctions
    psi1 = cubic_framework.calculate_wavefunction(p1)
    psi2 = cubic_framework.calculate_wavefunction(p2)