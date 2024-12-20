import numpy as np

def analyze_distance_matrices(matrix1, matrix2, frobenius_norm=1.1933):
    """
    Analyze two distance matrices to contextualize their Frobenius norm difference.
    
    Parameters:
    matrix1, matrix2: numpy arrays of shape (n,n) containing pairwise distances
    frobenius_norm: float, pre-calculated Frobenius norm of their difference
    
    Returns:
    dict containing mean pairwise distance and relative difference metrics
    """
    # Validate inputs
    if matrix1.shape != matrix2.shape:
        raise ValueError("Matrices must have the same shape")
    if not np.allclose(matrix1, matrix1.T) or not np.allclose(matrix2, matrix2.T):
        raise ValueError("Matrices must be symmetric")
    
    # Calculate mean pairwise distance for each matrix
    # Only use upper triangle since matrices are symmetric
    def mean_pairwise(mat):
        upper_tri = mat[np.triu_indices_from(mat, k=1)]
        return np.mean(upper_tri)
    
    mean1 = mean_pairwise(matrix1)
    mean2 = mean_pairwise(matrix2)
    overall_mean = (mean1 + mean2) / 2
    
    # Calculate relative Frobenius norm as percentage
    relative_frob = (frobenius_norm / overall_mean) * 100
    
    return {
        'mean_distance_matrix1': mean1,
        'mean_distance_matrix2': mean2,
        'overall_mean_distance': overall_mean,
        'relative_frobenius_percent': relative_frob
    }

results = analyze_distance_matrices(mafft_distance_matrix, clustal_distance_matrix, 1.1933)
print(f"Mean pairwise distance: {results['overall_mean_distance']:.4f}")
print(f"Relative Frobenius norm: {results['relative_frobenius_percent']:.2f}%")    
results = analyze_distance_matrices(mafft_distance_matrix, euclidean_dist.detach().numpy(), 28.15013835277865)
print(f"Mean pairwise distance: {results['overall_mean_distance']:.4f}")
print(f"Relative Frobenius norm: {results['relative_frobenius_percent']:.2f}%")
results = analyze_distance_matrices(clustal_distance_matrix, euclidean_dist.detach().numpy(), 27.161297429347748)
print(f"Mean pairwise distance: {results['overall_mean_distance']:.4f}")
print(f"Relative Frobenius norm: {results['relative_frobenius_percent']:.2f}%")
