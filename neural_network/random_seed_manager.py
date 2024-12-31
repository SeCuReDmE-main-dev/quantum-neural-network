import numpy as np
import random
import secrets

class RandomSeedManager:
    def __init__(self, seed=None):
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        self.random = random.Random(seed)
        self.secrets = secrets.SystemRandom()

    def generate_seed(self):
        self.seed = self.rng.integers(0, 2**32 - 1)
        self.rng = np.random.default_rng(self.seed)
        self.random.seed(self.seed)
        return self.seed

    def get_seed(self):
        return self.seed

    def set_seed(self, seed):
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        self.random.seed(seed)

    def random_bytes(self, length):
        return self.rng.bytes(length)

    def random_integers(self, low, high, size=None):
        return self.rng.integers(low, high, size)

    def random_floats(self, size=None):
        return self.rng.random(size)

    def random_choice(self, seq):
        return self.random.choice(seq)

    def random_sample(self, population, k):
        return self.random.sample(population, k)

    def secure_random_bytes(self, length):
        return self.secrets.token_bytes(length)

    def secure_random_integers(self, low, high):
        return self.secrets.randint(low, high)
