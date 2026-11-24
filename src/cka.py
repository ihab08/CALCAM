import numpy as np


def linear_cka(X, Y, eps=1e-12):
    """
    Compute Linear Centered Kernel Alignment (CKA)
    between two aligned representation matrices.

    Parameters
    ----------
    X : np.ndarray, shape (n_samples, d_x)
    Y : np.ndarray, shape (n_samples, d_y)

    Returns
    -------
    float
        Linear CKA similarity score.
    """

    X = np.asarray(X, dtype=np.float64)
    Y = np.asarray(Y, dtype=np.float64)

    if X.ndim != 2 or Y.ndim != 2:
        raise ValueError("X and Y must be 2D matrices.")

    if X.shape[0] != Y.shape[0]:
        raise ValueError(
            "X and Y must contain the same number of aligned samples."
        )

    # Center representations
    X = X - X.mean(axis=0, keepdims=True)
    Y = Y - Y.mean(axis=0, keepdims=True)

    numerator = np.linalg.norm(X.T @ Y, ord="fro") ** 2

    denominator = (
        np.linalg.norm(X.T @ X, ord="fro")
        * np.linalg.norm(Y.T @ Y, ord="fro")
    )

    return float(numerator / (denominator + eps))
