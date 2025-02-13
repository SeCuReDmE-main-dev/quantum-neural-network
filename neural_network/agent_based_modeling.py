import numpy as np
from typing import Dict, List, Optional, Any
import sys
import os

# Add project root to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from quantum_neural.neural_network.phi_framework import PhiFramework, PhiConfig

class Agent:
    def __init__(self, brain_region: str, state: Optional[np.ndarray] = None):
        self.brain_region = brain_region
        self.state = state if state is not None else np.random.randn(4)
        self.history = [self.state.copy()]

class AgentBasedModeling:
    """Agent-based modeling for quantum neural simulation"""
    
    def __init__(self, phi_config: Optional[PhiConfig] = None):
        self.phi_framework = PhiFramework(phi_config or PhiConfig())
        self.agents: List[Agent] = []
        self.time_step = 0
        
    def create_agent(self, brain_region: str, initial_state: Optional[np.ndarray] = None) -> Agent:
        """Create a new quantum agent"""
        agent = Agent(brain_region, initial_state)
        self.agents.append(agent)
        return agent
        
    def simulate_agent_interaction(self, steps: int = 100) -> Dict[str, Any]:
        """Simulate quantum interactions between agents"""
        history = []
        
        for _ in range(steps):
            # Update time step
            self.time_step += 1
            
            # Store current state
            current_state = {
                'time_step': self.time_step,
                'agent_positions': [agent.state for agent in self.agents],
                'brain_regions': {
                    agent.brain_region: {
                        'wave_pattern': self._compute_wave_pattern(agent.state)
                    }
                    for agent in self.agents
                }
            }
            history.append(current_state)
            
            # Update agent states
            self._update_agents()
            
        return {
            'history': history,
            'final_state': history[-1]
        }
        
    def _update_agents(self):
        """Update agent states based on quantum interactions"""
        for agent in self.agents:
            # Apply φ-scaled quantum evolution
            evolved_state = self.phi_framework.scale_quantum_state(agent.state)
            
            # Add quantum noise
            noise = np.random.randn(*agent.state.shape) * 0.1
            evolved_state += self.phi_framework.apply_phi_scaling(noise)
            
            # Update agent state
            agent.state = evolved_state
            agent.history.append(agent.state.copy())
            
    def _compute_wave_pattern(self, state: np.ndarray) -> np.ndarray:
        """Compute quantum wave pattern from agent state"""
        fourier = np.fft.fft(state)
        amplitudes = np.abs(fourier)
        return self.phi_framework.apply_phi_scaling(amplitudes)

    def analyze_emergence(self) -> Dict[str, Any]:
        """Analyze emergent behavior from simulation"""
        if not self.agents:
            return {"error": "No agents to analyze"}
            
        trajectory_complexity = self._analyze_trajectory_complexity()
        state_evolution = self._analyze_state_evolution()
        
        return {
            'trajectory_complexity': trajectory_complexity,
            'state_evolution': state_evolution
        }
        
    def _analyze_trajectory_complexity(self) -> Dict[str, float]:
        """Analyze complexity of agent trajectories"""
        complexity = {}
        for i, agent in enumerate(self.agents):
            # Calculate fractal dimension of trajectory
            trajectory = np.array(agent.history)
            # Apply φ-scaling to complexity metric
            complexity[f'Agent_{i}'] = self._calculate_fractal_dimension(trajectory)
        return complexity
        
    def _analyze_state_evolution(self) -> Dict[str, float]:
        """Analyze quantum state evolution"""
        evolution = {}
        for i, agent in enumerate(self.agents):
            history = np.array(agent.history)
            initial_state = history[0]
            final_state = history[-1]
            
            # Calculate φ-scaled state change
            evolution[f'Agent_{i}'] = float(
                self.phi_framework.compute_phi_resonance(initial_state, final_state)
            )
            
        return evolution
        
    def _calculate_fractal_dimension(self, trajectory: np.ndarray) -> float:
        """Calculate fractal dimension of a trajectory"""
        # Box counting dimension calculation
        scales = np.logspace(-2, 1, 20)
        counts = []
        
        for scale in scales:
            scaled_trajectory = trajectory / scale
            unique_boxes = set(map(tuple, np.floor(scaled_trajectory)))
            counts.append(len(unique_boxes))
            
        coeffs = np.polyfit(np.log(scales), np.log(counts), 1)
        dimension = -coeffs[0]  # Negative slope gives dimension
        
        return float(self.phi_framework.apply_phi_scaling(np.array([dimension])))

if __name__ == "__main__":
    # Test the agent-based modeling system
    print("Initializing Agent-Based Modeling system...")
    
    # Create modeling system
    abm = AgentBasedModeling()
    
    # Create test agents
    print("\nCreating test agents...")
    regions = ["Cerebrum", "Cerebellum", "Brainstem"]
    for region in regions:
        agent = abm.create_agent(region)
        print(f"Created agent for {region} with initial state shape: {agent.state.shape}")
    
    # Run simulation
    print("\nRunning quantum simulation...")
    steps = 50
    results = abm.simulate_agent_interaction(steps)
    print(f"Completed {steps} simulation steps")
    
    # Analyze results
    print("\nAnalyzing emergence patterns...")
    analysis = abm.analyze_emergence()
    
    print("\nTrajectory Complexity:")
    for agent, complexity in analysis['trajectory_complexity'].items():
        print(f"{agent}: {complexity:.4f}")
    
    print("\nState Evolution:")
    for agent, evolution in analysis['state_evolution'].items():
        print(f"{agent}: {evolution:.4f}")
    
    print("\nAgent-Based Modeling simulation complete.")
