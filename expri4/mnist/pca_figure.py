import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA as SklearnPCA
from sklearn.datasets import fetch_openml


# 手写PCA
def pca(data, n_components):
    """
    Perform Principal Component Analysis (PCA) for dimensionality reduction.

    Parameters:
    data (numpy.ndarray): The dataset, a 2D array where rows are samples and columns are features.
    n_components (int): The number of principal components to keep.

    Returns:
    numpy.ndarray: The dataset transformed into the reduced dimension space.
    """
    # Center the data
    mean = np.mean(data, axis=0)
    centered_data = data - mean

    # Compute covariance matrix
    covariance_matrix = np.cov(centered_data, rowvar=False)

    # Compute eigenvalues and eigenvectors
    eigenvalues, eigenvectors = np.linalg.eigh(covariance_matrix)

    # Sort eigenvectors by eigenvalues in descending order
    sorted_indices = np.argsort(eigenvalues)[::-1]
    eigenvectors = eigenvectors[:, sorted_indices]

    # Select the top n_components eigenvectors
    principal_components = eigenvectors[:, :n_components]

    # Project the data onto the principal components
    reduced_data = np.dot(centered_data, principal_components)

    return reduced_data


# 加载 MNIST 数据集
mnist = fetch_openml('mnist_784', version=1)
X = mnist.data / 255.0  # 数据归一化
y = mnist.target.astype(int)

# 创建保存 PCA 图像的文件夹
os.makedirs("pca2", exist_ok=True)

# 定义 PCA 降维的维度
pca_dims = [392, 196, 98, 49, 24, 12, 6, 5, 4, 3, 2]

# 使用库函数的 PCA 或者手写的 PCA
useLib = True  # 设置为 True 使用库函数，设置为 False 使用手写 PCA

# 选择每个数字（0-9）的一张图片
for digit in range(10):
    # 找到该数字的第一张图片
    idx = np.where(y == digit)[0][0]
    image = X.iloc[idx].to_numpy().reshape(28, 28)

    # 创建一个 4x3 的子图（4 行 3 列）
    fig, axes = plt.subplots(4, 3, figsize=(10, 12))
    axes = axes.flatten()  # 将子图展平，便于索引

    # 保存原图在第一位置
    axes[0].imshow(image, cmap='gray')
    axes[0].set_title(f"Original Image of Number {digit}, 784")
    axes[0].axis('off')

    # 使用循环处理每个降维后的 PCA 图像
    for i, dims in enumerate(pca_dims):
        if useLib:  # 使用库函数的 PCA
            sklearn_pca = SklearnPCA(n_components=dims)
            reduced_data = sklearn_pca.fit_transform(X)  # 使用整个数据集进行降维

            # 重构图像
            pca_image = sklearn_pca.inverse_transform(reduced_data[idx]).reshape(28, 28)

        else:  # 使用手写的 PCA
            reduced_data = pca(X, dims)  # 调用自定义的 pca 函数

            # 重构图像
            pca_image = reduced_data[idx].reshape(28, 28)

        # 保存降维后的图像
        axes[i + 1].imshow(pca_image, cmap='gray')
        axes[i + 1].set_title(f"PCA: {dims}")
        axes[i + 1].axis('off')

    # 保存图像到文件
    plt.tight_layout()
    plt.savefig(f"pca/{digit}_all_pca.png")
    plt.clf()

print("所有 PCA 降维图像已保存到 'pca' 文件夹。")
