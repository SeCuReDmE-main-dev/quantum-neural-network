import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
from typing import Dict, List, Tuple, Optional
import sys
import os

# Add the project root to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from quantum_neural.neural_network.phi_framework import PhiFramework, PhiConfig
from quantum_neural.neural_network.agent_based_modeling import AgentBasedModeling
from quantum_neural.neural_network.brain_structure import BrainStructureAnalysis
import torch

class QuantumDataVisualizer:
    def __init__(self, data=None, phi_config: Optional[PhiConfig] = None):
        self.data = data
        self.phi_framework = PhiFramework(phi_config or PhiConfig())
        plt.style.use('dark_background')
        
    def visualize_quantum_state(self, quantum_state: np.ndarray, title: str = '') -> plt.Figure:
        """Visualize quantum state with ϕ-based scaling"""
        fig = plt.figure(figsize=(12, 8))
        ax = fig.add_subplot(111)
        
        # Apply ϕ-scaling to visualization
        scaled_state = np.abs(quantum_state) * self.phi_framework.phi
        im = ax.imshow(scaled_state, cmap='viridis')
        plt.colorbar(im, ax=ax, label='ϕ-scaled amplitude')
        
        ax.set_title(f'Quantum State: {title}')
        ax.set_xlabel('State dimension')
        ax.set_ylabel('State dimension')
        return fig
    
    def plot_brain_waves(self, wave_pattern: np.ndarray, region: str) -> plt.Figure:
        """Plot brain wave patterns with ϕ-based analysis"""
        fig = plt.figure(figsize=(12, 6))
        ax = fig.add_subplot(111)
        
        wave_types = ['Alpha', 'Beta', 'Theta', 'Delta', 'Gamma']
        x = np.arange(len(wave_types))
        
        # Scale wave amplitudes by ϕ
        scaled_pattern = wave_pattern * self.phi_framework.phi
        
        bars = ax.bar(x, scaled_pattern)
        ax.set_xticks(x)
        ax.set_xticklabels(wave_types)
        ax.set_ylabel('ϕ-scaled amplitude')
        ax.set_title(f'Brain Wave Pattern: {region}')
        
        # Add ϕ-ratio annotations
        for i, bar in enumerate(bars):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'ϕ={height/self.phi_framework.phi:.2f}',
                   ha='center', va='bottom')
        
        return fig
    
    def visualize_particle_trajectories(self, 
                                      simulation_history: List[Dict],
                                      agents: List) -> plt.Figure:
        """Visualize quantum particle trajectories in 3D"""
        fig = plt.figure(figsize=(15, 10))
        ax = fig.add_subplot(111, projection='3d')
        
        colors = plt.cm.rainbow(np.linspace(0, 1, len(agents)))
        
        for i, agent in enumerate(agents):
            positions = np.array([state['agent_positions'][i] 
                                for state in simulation_history])
            
            # Scale trajectory by ϕ for visualization
            scaled_positions = positions * self.phi_framework.phi
            
            ax.plot3D(scaled_positions[:, 0], 
                     scaled_positions[:, 1],
                     scaled_positions[:, 2],
                     color=colors[i], 
                     label=f'Agent {i} ({agent.brain_region})')
            
            # Plot current position
            current_pos = scaled_positions[-1]
            ax.scatter(current_pos[0], current_pos[1], current_pos[2],
                      color=colors[i], s=100, marker='o')
        
        ax.set_xlabel('X Position (ϕ-scaled)')
        ax.set_ylabel('Y Position (ϕ-scaled)')
        ax.set_zlabel('Z Position (ϕ-scaled)')
        ax.set_title('Quantum Particle Trajectories')
        ax.legend()
        
        return fig
    
    def plot_emergence_analysis(self, emergence_results: Dict) -> plt.Figure:
        """Visualize emergence analysis results"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Plot trajectory complexity
        complexities = emergence_results['trajectory_complexity']
        agents = list(complexities.keys())
        values = list(complexities.values())
        
        ax1.bar(agents, values)
        ax1.set_title('Trajectory Complexity')
        ax1.set_xlabel('Agent ID')
        ax1.set_ylabel('ϕ-scaled Fractal Dimension')
        
        # Plot quantum evolution
        evolutions = emergence_results['quantum_evolution']
        regions = list(evolutions.keys())
        values = list(evolutions.values())
        
        ax2.bar(regions, values)
        ax2.set_title('Quantum State Evolution')
        ax2.set_xlabel('Brain Region')
        ax2.set_ylabel('ϕ-scaled State Change')
        plt.xticks(rotation=45)
        
        plt.tight_layout()
        return fig
    
    def plot_brain_region_interaction(self, 
                                    interaction_results: Dict,
                                    region1: str,
                                    region2: str) -> plt.Figure:
        """Visualize interaction between brain regions"""
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='polar')
        
        # Plot interaction metrics on a radar chart
        metrics = ['coherence', 'phase_sync', 'connectivity']
        values = [interaction_results[m] for m in metrics]
        
        # Scale values by ϕ for visualization
        scaled_values = np.array(values) * self.phi_framework.phi
        
        angles = np.linspace(0, 2*np.pi, len(metrics), endpoint=False)
        values = np.concatenate((scaled_values, [scaled_values[0]]))  # close the polygon
        angles = np.concatenate((angles, [angles[0]]))  # close the polygon
        
        ax.plot(angles, values)
        ax.fill(angles, values, alpha=0.25)
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(metrics)
        ax.set_title(f'Interaction: {region1} ↔ {region2}')
        
        return fig

    def visualize_qubit_relationships(self):
        """Visualize relationships between qubits"""
        if self.data is None:
            return
        
        plt.figure(figsize=(10, 8))
        matrix = np.corrcoef(self.data.T)
        sns.heatmap(matrix, cmap='viridis', annot=True)
        plt.title('Qubit Relationship Matrix')
        plt.show()
        
    def visualize_data_packet_flow(self):
        """Visualize quantum data packet flow"""
        if self.data is None:
            return
            
        plt.figure(figsize=(12, 6))
        timesteps = range(len(self.data))
        plt.plot(timesteps, self.data, alpha=0.5)
        plt.title('Quantum Data Packet Flow')
        plt.xlabel('Time Step')
        plt.ylabel('Packet Value')
        plt.grid(True)
        plt.show()
        
    def visualize_quantum_state_similarity(self):
        """Visualize similarity between quantum states"""
        if self.data is None:
            return
            
        from sklearn.manifold import MDS
        
        # Create distance matrix
        distances = np.zeros((self.data.shape[0], self.data.shape[0]))
        for i in range(self.data.shape[0]):
            for j in range(self.data.shape[0]):
                distances[i,j] = np.linalg.norm(self.data[i] - self.data[j])
                
        # Apply MDS
        mds = MDS(n_components=2, dissimilarity='precomputed')
        transformed = mds.fit_transform(distances)
        
        plt.figure(figsize=(10, 8))
        plt.scatter(transformed[:, 0], transformed[:, 1])
        plt.title('Quantum State Similarity Map')
        plt.xlabel('Component 1')
        plt.ylabel('Component 2')
        for i in range(len(transformed)):
            plt.annotate(f'State {i}', (transformed[i, 0], transformed[i, 1]))
        plt.grid(True)
        plt.show()
        
    def plot_eigenspectrum(self, eigenvalues):
        """Plot eigenvalue spectrum"""
        plt.figure(figsize=(10, 6))
        real_parts = np.real(eigenvalues)
        imag_parts = np.imag(eigenvalues)
        
        plt.scatter(real_parts, imag_parts)
        plt.axhline(y=0, color='r', linestyle='-', alpha=0.3)
        plt.axvline(x=0, color='r', linestyle='-', alpha=0.3)
        plt.title('Eigenvalue Spectrum')
        plt.xlabel('Real Part')
        plt.ylabel('Imaginary Part')
        plt.grid(True)
        plt.show()
        
    def plot_stability_metric(self, stability_history):
        """Plot stability metric over time"""
        plt.figure(figsize=(10, 6))
        plt.plot(stability_history, '-o')
        plt.title('System Stability Over Time')
        plt.xlabel('Time Step')
        plt.ylabel('Stability Metric')
        plt.grid(True)
        plt.show()

# Example usage
if __name__ == "__main__":
    # Initialize components
    visualizer = QuantumDataVisualizer()
    modeling = AgentBasedModeling()
    brain_analyzer = BrainStructureAnalysis()
    
    # Generate example data
    eeg_data = np.random.randn(1000)
    activity_data = np.random.randn(1000)
    
    # Register brain regions
    cerebrum = brain_analyzer.register_brain_region(
        "Cerebrum", 1400, 86_000_000_000, activity_data, eeg_data
    )
    
    # Create agents and run simulation
    agent = modeling.create_agent("Cerebrum")
    simulation_results = modeling.simulate_agent_interaction()
    emergence_results = modeling.analyze_emergence(simulation_results)
    
    # Generate visualizations
    quantum_state_fig = visualizer.visualize_quantum_state(
        simulation_results['final_state']['brain_regions']['Cerebrum'].wave_pattern,
        "Cerebrum"
    )
    
    trajectory_fig = visualizer.visualize_particle_trajectories(
        simulation_results['history'],
        [agent]
    )
    
    emergence_fig = visualizer.plot_emergence_analysis(emergence_results)
    
    plt.show()

    # Test visualization module
    test_data = np.random.rand(10, 5)
    visualizer = QuantumDataVisualizer(test_data)
    
    print("Testing quantum visualizations...")
    visualizer.visualize_qubit_relationships()
    visualizer.visualize_data_packet_flow()
    visualizer.visualize_quantum_state_similarity()
    
    # Test eigenspectrum visualization
    test_matrix = np.random.rand(5, 5)
    eigenvalues = np.linalg.eigvals(test_matrix)
    visualizer.plot_eigenspectrum(eigenvalues)
    
    # Test stability visualization
    stability_history = np.random.rand(20)
    visualizer.plot_stability_metric(stability_history)
