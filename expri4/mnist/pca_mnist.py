from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
from scipy.stats import mode


def k_means_clustering(data, k, max_iterations=100, tol=1e-4, random_state=42):
    """
    Perform K-means clustering on the given data.

    Parameters:
    data (numpy.ndarray): The dataset, a 2D array where rows are samples and columns are features.
    k (int): The number of clusters.
    max_iterations (int): The maximum number of iterations to run the algorithm.
    tol (float): The tolerance for convergence based on centroids' movement.

    Returns:
    tuple: A tuple containing the cluster assignments and the final centroids.
    """
    # Randomly initialize k centroids from the dataset
    n_samples, n_features = data.shape
    np.random.seed(random_state)
    centroids = data[np.random.choice(n_samples, k, replace=False)]

    for iteration in range(max_iterations):
        # Assign each sample to the nearest centroid
        distances = np.linalg.norm(data[:, np.newaxis] - centroids, axis=2)
        labels = np.argmin(distances, axis=1)

        # Update centroids
        new_centroids = np.array([data[labels == i].mean(axis=0) for i in range(k)])

        # Check for convergence
        if np.all(np.linalg.norm(new_centroids - centroids, axis=1) < tol):
            break

        centroids = new_centroids

    return labels, centroids


def compute_accuracy(true_labels, predicted_labels, k):
    """
    Compute the accuracy of the clustering by finding the best label mapping.
    """
    # Find the best label mapping
    label_mapping = np.zeros_like(predicted_labels)
    for cluster in range(k):
        mask = (predicted_labels == cluster)
        label_mapping[mask] = mode(true_labels[mask])[0]

    # Compute accuracy
    return accuracy_score(true_labels, label_mapping)


def visualize_clusters(data, true_labels, predicted_labels, centroids):
    """
    Visualize clustering results for each pair of features below the diagonal.
    On the diagonal, draw histograms of the feature values and vertical lines for centroids.
    Include axis labels on the outermost rows and columns.
    """
    num_features = data.shape[1]
    feature_names = [f"Feature {i+1}" for i in range(num_features)]
    fig, axes = plt.subplots(num_features, num_features, figsize=(12, 12))
    for i in range(num_features):
        for j in range(i):
            ax = axes[i, j]
            # Scatter plot for true points
            ax.scatter(data[:, j], data[:, i], c=true_labels, cmap="viridis", s=30, alpha=0.6, label="True Points")
            # Scatter plot for centroids
            ax.scatter(centroids[:, j], centroids[:, i], c="red", s=200, alpha=0.75, marker="X", label="Centroids")
            if i == num_features - 1 and j == 0:
                ax.legend()
        for j in range(i, num_features):
            if i == j:
                ax = axes[i, j]
                # Histogram of feature values
                ax.hist(data[:, i], bins=20, color="blue", alpha=0.7, label="Feature Distribution")
                # Vertical lines for centroids
                for centroid in centroids[:, i]:
                    ax.axvline(x=centroid, color="red", linestyle="--", label="Centroid")
                if i == num_features - 1:
                    ax.legend()
            else:
                axes[i, j].axis("off")

    # Add feature names to the outermost rows and columns
    for i in range(num_features):
        axes[i, 0].set_ylabel(feature_names[i], fontsize=12)
        axes[-1, i].set_xlabel(feature_names[i], fontsize=12)

    plt.suptitle("Clustering Visualization (Each Pair of Features Below Diagonal)", fontsize=16)
    plt.tight_layout()
    plt.show()


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


# Load the MNIST dataset
mnist = fetch_openml('mnist_784', version=1)
data = mnist.data / 255.0  # Normalize data
target = mnist.target.astype(int)

useLib = (True, True)

# for pca2 in [784 // 2**x for x in range(9)]:
for pca2 in [5, 4, 2]:
    if useLib[0]:
        libpca = PCA(n_components=pca2)
        reduced_data = libpca.fit_transform(data)
    else:
        reduced_data = pca(data, n_components=pca2)

    # Split the data into training and testing sets
    reduced_train, reduced_test, labels_train, labels_test = train_test_split(
        reduced_data, target, test_size=0.3, random_state=42)

    # Perform K-means clustering on the training set
    k = 10
    if useLib[1]:
        kmeans = KMeans(n_clusters=k, random_state=42)
        kmeans.fit(reduced_train)
        predicted_train_labels = kmeans.labels_
        centroids = kmeans.cluster_centers_
    else:
        predicted_train_labels, centroids = k_means_clustering(reduced_train, k)

    # Predict on the test set
    test_distances = np.linalg.norm(reduced_test[:, np.newaxis] - centroids, axis=2)
    predicted_test_labels = np.argmin(test_distances, axis=1)

    # Compute accuracy
    train_accuracy = compute_accuracy(labels_train, predicted_train_labels, k)
    test_accuracy = compute_accuracy(labels_test, predicted_test_labels, k)

    print(f"K-means accuracy on training set after PCA to {pca2}: {train_accuracy * 100:.2f}%")
    print(f"K-means accuracy on testing  set after PCA to {pca2}: {test_accuracy * 100:.2f}%")

    # Use SVC to evaluate classification performance
    svc = SVC()
    svc.fit(reduced_train, labels_train)
    svc_train_accuracy = svc.score(reduced_train, labels_train)
    svc_test_accuracy = svc.score(reduced_test, labels_test)

    print(f"SVC     accuracy on training set after PCA to {pca2}: {svc_train_accuracy * 100:.2f}%")
    print(f"SVC     accuracy on testing  set after PCA to {pca2}: {svc_test_accuracy * 100:.2f}%")
    print()
