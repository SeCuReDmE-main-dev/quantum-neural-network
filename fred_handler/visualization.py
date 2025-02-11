import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
from typing import Dict, List, Optional
import torch

class QuantumVisualizer:
    def __init__(self):
        self.color_map = plt.cm.viridis
        plt.style.use('dark_background')
        
    def plot_quantum_state(self, state: np.ndarray, title: str = "Quantum State"):
        """Plot quantum state amplitudes"""
        plt.figure(figsize=(12, 6))
        amplitudes = np.abs(state)
        phases = np.angle(state)
        
        plt.subplot(1, 2, 1)
        plt.bar(range(len(amplitudes)), amplitudes)
        plt.title("State Amplitudes")
        plt.xlabel("State Index")
        plt.ylabel("Amplitude")
        
        plt.subplot(1, 2, 2)
        plt.bar(range(len(phases)), phases)
        plt.title("State Phases")
        plt.xlabel("State Index")
        plt.ylabel("Phase")
        
        plt.suptitle(title)
        plt.tight_layout()
        plt.show()
        
    def plot_brain_connectivity(self, connectivity_matrix: np.ndarray, 
                              region_names: Optional[List[str]] = None):
        """Plot brain region connectivity matrix"""
        plt.figure(figsize=(12, 10))
        sns.heatmap(connectivity_matrix, 
                    xticklabels=region_names,
                    yticklabels=region_names,
                    cmap='plasma',
                    annot=True)
        plt.title("Brain Region Connectivity")
        plt.tight_layout()
        plt.show()
        
    def plot_energy_landscape(self, eigenvalues: np.ndarray, eigenvectors: np.ndarray):
        """Plot quantum energy landscape"""
        fig = plt.figure(figsize=(12, 8))
        ax = fig.add_subplot(111, projection='3d')
        
        x = np.arange(len(eigenvalues))
        y = np.arange(eigenvectors.shape[1])
        X, Y = np.meshgrid(x, y)
        Z = np.abs(eigenvectors)**2
        
        surf = ax.plot_surface(X, Y, Z, cmap=self.color_map)
        plt.colorbar(surf)
        
        ax.set_xlabel('Eigenvalue Index')
        ax.set_ylabel('Vector Component')
        ax.set_zlabel('Probability Density')
        plt.title("Quantum Energy Landscape")
        plt.show()
        
    def plot_stability_evolution(self, stability_metrics: List[Dict[str, float]]):
        """Plot evolution of stability metrics over time"""
        metrics = ['spectral_radius', 'stability_index', 'eigenvalue_spread']
        plt.figure(figsize=(12, 6))
        
        for metric in metrics:
            values = [m[metric] for m in stability_metrics]
            plt.plot(values, label=metric)
            
        plt.xlabel('Time Step')
        plt.ylabel('Metric Value')
        plt.title('Stability Metrics Evolution')
        plt.legend()
        plt.grid(True)
        plt.show()
        
    def plot_agent_network(self, agents: List[Dict], connections: Dict[str, List[str]]):
        """Plot quantum agent network"""
        plt.figure(figsize=(12, 12))
        
        # Create positions for agents
        n_agents = len(agents)
        angles = np.linspace(0, 2*np.pi, n_agents, endpoint=False)
        pos_x = np.cos(angles)
        pos_y = np.sin(angles)
        
        # Plot connections
        for agent_id, connected_ids in connections.items():
            agent_idx = next(i for i, a in enumerate(agents) if a['id'] == agent_id)
            for connected_id in connected_ids:
                connected_idx = next(i for i, a in enumerate(agents) if a['id'] == connected_id)
                plt.plot([pos_x[agent_idx], pos_x[connected_idx]],
                        [pos_y[agent_idx], pos_y[connected_idx]],
                        'w-', alpha=0.3)
        
        # Plot agents
        energies = np.array([agent['energy'] for agent in agents])
        normalized_energies = (energies - np.min(energies)) / (np.max(energies) - np.min(energies))
        
        plt.scatter(pos_x, pos_y, c=normalized_energies, 
                   cmap=self.color_map, s=100)
        
        plt.title('Quantum Agent Network')
        plt.axis('equal')
        plt.grid(True)
        plt.show()
        
    def plot_quantum_evolution(self, states_history: List[np.ndarray]):
        """Plot quantum state evolution over time"""
        plt.figure(figsize=(12, 8))
        
        times = np.arange(len(states_history))
        state_matrix = np.array(states_history)
        
        plt.imshow(np.abs(state_matrix.T), 
                  aspect='auto', 
                  cmap=self.color_map,
                  extent=[0, len(times)-1, 0, state_matrix.shape[1]-1])
        
        plt.colorbar(label='Amplitude')
        plt.xlabel('Time Step')
        plt.ylabel('State Component')
        plt.title('Quantum State Evolution')
        plt.show()

if __name__ == "__main__":
    visualizer = QuantumVisualizer()
    
    # Test with random quantum state
    test_state = np.random.rand(16) + 1j * np.random.rand(16)
    test_state = test_state / np.linalg.norm(test_state)
    
    # Plot quantum state
    visualizer.plot_quantum_state(test_state, "Test Quantum State")
    
    # Test with random connectivity matrix
    test_connectivity = np.random.rand(5, 5)
    region_names = ['A', 'B', 'C', 'D', 'E']
    visualizer.plot_brain_connectivity(test_connectivity, region_names)
