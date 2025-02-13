import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
from typing import Dict, List, Tuple, Optional
import os

# Create output directory for plots
PLOT_DIR = "quantum_plots"
if not os.path.exists(PLOT_DIR):
    os.makedirs(PLOT_DIR)

class QuantumDataVisualizer:
    def __init__(self, data=None):
        self.data = data
        plt.style.use('dark_background')
        
    def visualize_qubit_relationships(self):
        """Visualize relationships between qubits"""
        if self.data is None:
            return
        
        plt.figure(figsize=(10, 8))
        matrix = np.corrcoef(self.data.T)
        sns.heatmap(matrix, cmap='viridis', annot=True)
        plt.title('Qubit Relationship Matrix')
        plt.savefig(os.path.join(PLOT_DIR, 'qubit_relationships.png'))
        plt.close()
        
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
        plt.savefig(os.path.join(PLOT_DIR, 'data_packet_flow.png'))
        plt.close()
        
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
        plt.savefig(os.path.join(PLOT_DIR, 'quantum_state_similarity.png'))
        plt.close()
        
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
        plt.savefig(os.path.join(PLOT_DIR, 'eigenspectrum.png'))
        plt.close()
        
    def plot_stability_metric(self, stability_history):
        """Plot stability metric over time"""
        plt.figure(figsize=(10, 6))
        plt.plot(stability_history, '-o')
        plt.title('System Stability Over Time')
        plt.xlabel('Time Step')
        plt.ylabel('Stability Metric')
        plt.grid(True)
        plt.savefig(os.path.join(PLOT_DIR, 'stability_metric.png'))
        plt.close()

if __name__ == "__main__":
    # Test visualization module
    print("Testing quantum visualizations...")
    
    # Generate test data
    test_data = np.random.rand(10, 5)  # 10 samples with 5 features
    visualizer = QuantumDataVisualizer(test_data)
    
    print("\nGenerating visualizations in 'quantum_plots' directory...")
    
    # Test basic visualizations
    visualizer.visualize_qubit_relationships()
    print("Generated qubit_relationships.png")
    
    visualizer.visualize_data_packet_flow()
    print("Generated data_packet_flow.png")
    
    visualizer.visualize_quantum_state_similarity()
    print("Generated quantum_state_similarity.png")
    
    # Test eigenspectrum visualization
    test_matrix = np.random.rand(5, 5)
    eigenvalues = np.linalg.eigvals(test_matrix)
    visualizer.plot_eigenspectrum(eigenvalues)
    print("Generated eigenspectrum.png")
    
    # Test stability visualization
    stability_history = np.random.rand(20)
    visualizer.plot_stability_metric(stability_history)
    print("Generated stability_metric.png")
    
    print("\nVisualization tests complete. Check the 'quantum_plots' directory for the generated plots.")
