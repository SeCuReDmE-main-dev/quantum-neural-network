import numpy as np
import random
import torch
from typing import Optional, Dict, Any
import json
import os

class RandomSeedManager:
    def __init__(self, base_seed: int = 42):
        self.base_seed = base_seed
        self.current_seed = base_seed
        self.seed_history = []
        self.experiment_seeds: Dict[str, int] = {}
        
    def set_seed(self, seed: Optional[int] = None) -> int:
        """Set random seed for all random number generators"""
        if seed is None:
            seed = self.current_seed
            
        # Set seeds for different libraries
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            
        self.current_seed = seed
        self.seed_history.append(seed)
        return seed
        
    def get_next_seed(self) -> int:
        """Generate next seed using golden ratio"""
        golden_ratio = (1 + 5 ** 0.5) / 2
        next_seed = int(self.current_seed * golden_ratio) % (2**32)
        return next_seed
        
    def register_experiment(self, experiment_name: str, seed: Optional[int] = None) -> int:
        """Register a new experiment with its seed"""
        if seed is None:
            seed = self.get_next_seed()
        self.experiment_seeds[experiment_name] = seed
        self.set_seed(seed)
        return seed
        
    def save_state(self, filepath: str):
        """Save seed manager state to file"""
        state = {
            'base_seed': self.base_seed,
            'current_seed': self.current_seed,
            'seed_history': self.seed_history,
            'experiment_seeds': self.experiment_seeds
        }
        
        with open(filepath, 'w') as f:
            json.dump(state, f, indent=4)
            
    def load_state(self, filepath: str) -> bool:
        """Load seed manager state from file"""
        try:
            with open(filepath, 'r') as f:
                state = json.load(f)
                
            self.base_seed = state['base_seed']
            self.current_seed = state['current_seed']
            self.seed_history = state['seed_history']
            self.experiment_seeds = state['experiment_seeds']
            return True
        except Exception as e:
            print(f"Error loading state: {e}")
            return False
            
    def get_experiment_seed(self, experiment_name: str) -> Optional[int]:
        """Get seed for a specific experiment"""
        return self.experiment_seeds.get(experiment_name)
        
    def reset(self):
        """Reset seed manager to initial state"""
        self.current_seed = self.base_seed
        self.seed_history = []
        self.experiment_seeds = {}
        self.set_seed(self.base_seed)
        
    def get_state_summary(self) -> Dict[str, Any]:
        """Get summary of current state"""
        return {
            'base_seed': self.base_seed,
            'current_seed': self.current_seed,
            'num_seeds_used': len(self.seed_history),
            'num_experiments': len(self.experiment_seeds),
            'last_5_seeds': self.seed_history[-5:] if self.seed_history else []
        }

if __name__ == "__main__":
    # Test the seed manager
    seed_manager = RandomSeedManager()
    
    # Register some experiments
    seed_manager.register_experiment("quantum_evolution_test")
    seed_manager.register_experiment("neural_network_training")
    
    # Save state
    seed_manager.save_state("seed_manager_state.json")
    
    # Print summary
    print("Seed Manager State:", seed_manager.get_state_summary())
