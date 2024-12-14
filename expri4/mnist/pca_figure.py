import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA as SklearnPCA
from sklearn.datasets import fetch_openml


def pca(data, n_components):
    """
    Perform Principal Component Analysis (PCA) for dimensionality reduction.

    Parameters:
    data (numpy.ndarray): The dataset, a 2D array where rows are samples and columns are features.
    n_components (int): The number of principal components to keep.

    Returns:
    numpy.ndarray: The dataset transformed into the reduced dimension space.
    """
    mean = np.mean(data, axis=0)
    centered_data = data - mean

    covariance_matrix = np.cov(centered_data, rowvar=False)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance_matrix)
    sorted_indices = np.argsort(eigenvalues)[::-1]
    eigenvectors = eigenvectors[:, sorted_indices]
    principal_components = eigenvectors[:, :n_components]
    reduced_data = np.dot(centered_data, principal_components)

    return reduced_data


mnist = fetch_openml('mnist_784', version=1)
X = mnist.data / 255.0
y = mnist.target.astype(int)

os.makedirs("pca", exist_ok=True)

pca_dims = [392, 196, 98, 49, 24, 12, 6, 5, 4, 3, 2]

useLib = True

for digit in [7, 9]:
    idx = np.where(y == digit)[0][0]
    image = X.iloc[idx].to_numpy().reshape(28, 28)

    fig, axes = plt.subplots(4, 3, figsize=(10, 12))
    axes = axes.flatten()

    axes[0].imshow(image, cmap='gray')
    axes[0].set_title(f"Original Image of Number {digit}, 784")
    axes[0].axis('off')

    for i, dims in enumerate(pca_dims):
        if useLib:
            sklearn_pca = SklearnPCA(n_components=dims)
            reduced_data = sklearn_pca.fit_transform(X)

            pca_image = sklearn_pca.inverse_transform(reduced_data[idx]).reshape(28, 28)

        else:
            reduced_data = pca(X, dims)
            pca_image = reduced_data[idx].reshape(28, 28)

        axes[i + 1].imshow(pca_image, cmap='gray')
        axes[i + 1].set_title(f"PCA: {dims}")
        axes[i + 1].axis('off')

    plt.tight_layout()
    plt.savefig(f"pca/{digit}_all_pca.png")
    plt.clf()

print("所有 PCA 降维图像已保存到 'pca' 文件夹。")
