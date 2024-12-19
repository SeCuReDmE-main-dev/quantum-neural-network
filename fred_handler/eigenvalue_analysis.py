import numpy as np

class EigenvalueAnalysis:
    def __init__(self, matrix):
        self.matrix = matrix

    def compute_eigenvalues(self):
        eigenvalues, _ = np.linalg.eig(self.matrix)
        return eigenvalues

    def is_stable(self):
        eigenvalues = self.compute_eigenvalues()
        return np.all(np.real(eigenvalues) < 0)

    def optimize_parameters(self, params, learning_rate=0.01, max_iter=100):
        for _ in range(max_iter):
            eigenvalues = self.compute_eigenvalues()
            if self.is_stable():
                break
            params -= learning_rate * np.real(eigenvalues)
        return params
