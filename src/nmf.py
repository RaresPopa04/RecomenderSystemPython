import numpy as np


def nmf(X, n_components, test_indices, max_iter=1, tol=1e-6):
    """
        Perform non-negative matrix factorization (NMF) on a given data matrix X.
        
        Parameters:
            X (numpy.ndarray): The data matrix to be factorized, where missing values are represented as NaN.
            n_components (int): The number of latent features.
            test_indices (list of tuples): Indices in X where the data is missing and should be tested.
            max_iter (int): Maximum number of iterations to perform.
            tol (float): Tolerance for the stopping condition based on the Frobenius norm of the difference matrix.
        
        Returns:
            W (numpy.ndarray): Basis matrix where each column is a component.
            H (numpy.ndarray): Coefficient matrix that represents the contribution of each component to the original matrix.
            error (float): Frobenius norm of the difference between the original and reconstructed matrix at the end of factorization.
        """

    # Validate that all test indices refer to NaN entries in the original matrix
    for position in test_indices:
        assert np.isnan(X[position[0], position[1]])

    # Initialize the factor matrices W and H with random non-negative values
    known_mask = ~np.isnan(X)
    W = np.random.rand(X.shape[0], n_components)
    H = np.random.rand(n_components, X.shape[1])

    error = np.inf
    X_non_nan = np.nan_to_num(X, nan=0)
    # Iterate until maximum iterations or until error falls below the tolerance level
    for i in range(max_iter):
        X_hat = W @ H
        H = H * (W.T @ (known_mask * X_non_nan) / (W.T @ (known_mask * X_hat) + 1e-9))
        X_hat = W @ H
        W = W * ((known_mask * X_non_nan) @ H.T / ((known_mask * X_hat) @ H.T + 1e-9))

        frobenius_norm = np.linalg.norm(known_mask * (X_non_nan - X_hat), 'fro')
        error = frobenius_norm
        if frobenius_norm < tol:
            break

    return W, H, error


def cross_validation(V, nmf, n_components, k=5):
    """
    Perform k-fold cross-validation on the given data matrix V to evaluate the performance of NMF.
    
    Parameters:
        V (numpy.ndarray): The data matrix with potential NaNs indicating missing values.
        nmf (function): The NMF function to be validated.
        n_components (int): The number of latent features for NMF.
        k (int): The number of folds in the cross-validation.
    
    Returns:
        float: The average error across all folds.
    """

    # Find indices of non-NaN values in V
    non_nan_indices = np.argwhere(~np.isnan(V))

    # Shuffle these indices to randomize the folds
    np.random.shuffle(non_nan_indices)

    # Calculate the size of each fold
    fold_size = len(non_nan_indices) // k
    fold_errors = []

    for i in range(k):
        start = i * fold_size
        end = (i + 1) * fold_size
        test_indices = non_nan_indices[start:end]

        # Create a training version of V by setting the test indices to NaN
        V_train = np.copy(V)
        for position in test_indices:
            V_train[position[0], position[1]] = np.nan

        # Apply the NMF function to the training data
        W, H, _ = nmf(V_train, n_components, test_indices)

        V_hat = W @ H

        error_curent_fold = 0
        # Calculate the error for the current fold using the test indices
        for (row, col) in test_indices:
            error_curent_fold += (V[row, col] - V_hat[row, col]) ** 2

        # Average the error by the number of test points in the fold
        fold_errors.append(error_curent_fold / len(test_indices))

    return np.mean(fold_errors)


def calculate_prediction_matrix(data_matrix):
    """
    Calculate the predicted ratings matrix using the factorized matrices W and H.
    
    Parameters:
        data_matrix (numpy.ndarray): The original data matrix.
    
    Returns:
        numpy.ndarray: The predicted ratings matrix.
    """
    # List of different numbers of latent components to evaluate
    list_components = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]

    # Initialize lists to store the results of the cross-validation
    errors = []

    # Set a random seed for reproducibility of random operations such as shuffling
    np.random.seed(42)

    for n_components in list_components:
        # Perform cross-validation on the dataset with the current number of components
        error = cross_validation(data_matrix.values, nmf, n_components, 5)
        print(f"Error for {n_components} components: {error}")
        errors.append(error)

    # Find the optimal number of components using the elbow method
    optimal_components = list_components[np.argmin(errors)]
    for i in range(1, len(errors)):
        if errors[i] < errors[i - 1] * 0.95:  # If the error reduction is less than 5%
            optimal_components = list_components[i - 1]
            break
    print(f"Optimal number of components: {optimal_components}")
    n_components = optimal_components

    # Perform NMF on the full dataset with the optimal number of components
    W, H, _ = nmf(data_matrix.values, n_components, [])
    predicted_ratings = W @ H
    return predicted_ratings
