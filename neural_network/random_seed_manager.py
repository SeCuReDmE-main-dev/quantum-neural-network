import numpy as np
import random
import secrets
import torch
from typing import Optional, Dict, Any, List, Sequence, Union
import json
import os

class RandomSeedManager:
    def __init__(self, base_seed: int = 42):
        self.base_seed = base_seed
        self.current_seed: int = base_seed
        self.seed_history: List[int] = []
        self.experiment_seeds: Dict[str, int] = {}
        self.rng = np.random.default_rng(base_seed)
        self.random = random.Random(base_seed)
        self.set_seed(base_seed)
        
    def set_seed(self, seed: Optional[int] = None) -> int:
        """Set random seed for all random number generators"""
        if seed is None:
            seed = self.current_seed
        if seed is None:  # Double check since self.current_seed could be None
            seed = self.base_seed
            
        # Set seeds for different libraries
        random.seed(seed)
        np.random.seed(seed)
        self.rng = np.random.default_rng(seed)
        self.random.seed(seed)
        
        # Set PyTorch seeds
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
        if self.current_seed is None:
            self.current_seed = self.base_seed
        next_seed = int(self.current_seed * golden_ratio) % (2**32)
        return next_seed

    def generate_seed(self) -> int:
        """Generate a new random seed"""
        new_seed = self.rng.integers(0, 2**32 - 1)
        self.set_seed(new_seed)
        return new_seed

    def get_seed(self) -> int:
        """Get current seed"""
        if self.current_seed is None:
            self.current_seed = self.base_seed
        return self.current_seed

    def random_bytes(self, length: int) -> bytes:
        """Generate random bytes"""
        return self.rng.bytes(length)

    def random_integers(self, low: int, high: int, size: Optional[int] = None) -> np.ndarray:
        """Generate random integers in range [low, high)"""
        return self.rng.integers(low, high, size)

    def random_floats(self, size: Optional[int] = None) -> np.ndarray:
        """Generate random floats in range [0.0, 1.0)"""
        result = self.rng.random(size)
        if isinstance(result, float):
            return np.array([result])
        return result

    def random_choice(self, seq: Sequence) -> Any:
        """Choose a random element from a sequence"""
        return self.random.choice(seq)

    def random_sample(self, population: Sequence, k: int) -> List:
        """Choose k unique random elements from a population"""
        return self.random.sample(population, k)

    def secure_random_bytes(self, length: int) -> bytes:
        """Generate cryptographically secure random bytes"""
        return secrets.token_bytes(length)

    def secure_random_integers(self, low: int, high: int) -> int:
        """Generate cryptographically secure random integer"""
        return secrets.randbelow(high - low) + low

    def register_experiment(self, experiment_name: str, seed: Optional[int] = None) -> int:
        """Register a new experiment with its seed"""
        if seed is None:
            seed = self.get_next_seed()
        self.experiment_seeds[experiment_name] = seed
        self.set_seed(seed)
        return seed

    def get_experiment_seed(self, experiment_name: str) -> Optional[int]:
        """Get seed for a specific experiment"""
        return self.experiment_seeds.get(experiment_name)

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
            self.set_seed(self.current_seed)
            return True
        except Exception as e:
            print(f"Error loading state: {e}")
            return False

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
    print("Initializing Random Seed Manager...")
    seed_manager = RandomSeedManager()
    
    # Test basic random generation
    print("\nTesting random generation:")
    print(f"Current seed: {seed_manager.get_seed()}")
    print(f"Random integers: {seed_manager.random_integers(0, 100, 5)}")
    print(f"Random floats: {seed_manager.random_floats(3)}")
    
    # Test experiment registration
    print("\nTesting experiment registration:")
    exp1_seed = seed_manager.register_experiment("quantum_evolution_test")
    exp2_seed = seed_manager.register_experiment("neural_network_training")
    print(f"Registered experiments with seeds: {exp1_seed, {exp2_seed}")
    
    # Test secure random generation
    print("\nTesting secure random generation:")
    secure_int = seed_manager.secure_random_integers(0, 100)
    print(f"Secure random integer: {secure_int}")
    
    # Save and load state
    state_file = "seed_manager_state.json"
    print(f"\nSaving state to {state_file}")
    seed_manager.save_state(state_file)
    
    # Print final summary
    print("\nFinal state summary:")
    print(json.dumps(seed_manager.get_state_summary(), indent=2))
