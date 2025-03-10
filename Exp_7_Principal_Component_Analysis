
#Algorithm
#S-1 : Standardize the dataset to have zero mean and unit variance.
#S-2 : Compute the covariance matrix of the standardized data.
#S-3 : Calculate eigenvalues and eigenvectors of the covariance matrix.
#S-4 : Sort the eigenvalues and select the top principal components.
#S-5 : Project the original data onto the selected principal components.

import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

def pca(X, num_components):
    scaler = StandardScaler()
    X_std = scaler.fit_transform(X)
    covariance_matrix = np.cov(X_std, rowvar=False)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance_matrix)
    sorted_indices = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[sorted_indices]
    eigenvectors = eigenvectors[:, sorted_indices]
    principal_components = eigenvectors[:, :num_components]
    X_pca = np.dot(X_std, principal_components)
    return X_pca, eigenvalues[:num_components], principal_components

def main():
    np.random.seed(25)
    X = np.random.rand(35, 2)  
    num_components = 2
    X_pca, eigenvalues, principal_components = pca(X, num_components)
    plt.scatter(X_pca[:, 0], X_pca[:, 1], alpha=0.7, edgecolors='k')
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.title('PCA Projection')
    plt.show()
    print("Eigenvalues:", eigenvalues)
    print("Principal Components:\n", principal_components)

if __name__ == "__main__":
    main()
