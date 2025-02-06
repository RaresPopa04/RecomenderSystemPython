import numpy as np




def train_nmf(X, n_components, num_iterations, tol):
    W = np.random.rand(X.shape[0], n_components)
    H = np.random.rand(n_components, X.shape[1])

    for i in range(num_iterations):
        H = H * (W.T @ X) / (W.T @ W @ H + 1e-9)
        W = W * (X @ H.T) / (W @ H @ H.T + 1e-9)

        frobenius_norm = np.linalg.norm(X - W @ H, 'fro')

        if frobenius_norm < tol:
            break

    return W, H

