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

    def advanced_stability_analysis(self):
        eigenvalues = self.compute_eigenvalues()
        stability_metrics = {
            'max_real_part': np.max(np.real(eigenvalues)),
            'min_real_part': np.min(np.real(eigenvalues)),
            'max_imaginary_part': np.max(np.imag(eigenvalues)),
            'min_imaginary_part': np.min(np.imag(eigenvalues))
        }
        return stability_metrics

    def optimize_parameters_with_constraints(self, params, constraints, learning_rate=0.01, max_iter=100):
        for _ in range(max_iter):
            eigenvalues = self.compute_eigenvalues()
            if self.is_stable():
                break
            for i, param in enumerate(params):
                if constraints[i][0] <= param <= constraints[i][1]:
                    params[i] -= learning_rate * np.real(eigenvalues[i])
        return params

    def compute_eigenvalues_for_large_matrices(self):
        if self.matrix.shape[0] > 1000:
            eigenvalues = np.linalg.eigvals(self.matrix)
        else:
            eigenvalues, _ = np.linalg.eig(self.matrix)
        return eigenvalues
