import numpy as np
from mahout import Mahout

class RandomSeedManager:
    def __init__(self, seed=None):
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        self.mahout = Mahout()

    def generate_seed(self):
        self.seed = self.rng.integers(0, 2**32 - 1)
        self.rng = np.random.default_rng(self.seed)
        return self.seed

    def get_seed(self):
        return self.seed

    def set_seed(self, seed):
        self.seed = seed
        self.rng = np.random.default_rng(seed)

    def random_bytes(self, length):
        return self.rng.bytes(length)

    def random_integers(self, low, high, size=None):
        return self.rng.integers(low, high, size)

    def random_floats(self, size=None):
        return self.rng.random(size)
