import tensorflow as tf
import numpy as np

# 加载CIFAR-10数据集 -> np.ndarray
(X_train, y_train), (X_test, y_test) = tf.keras.datasets.cifar10.load_data()

np.random.seed(43)

# 打乱训练集
shuffle_indices = np.random.permutation(len(X_train))
X_train = X_train[shuffle_indices]
y_train = y_train[shuffle_indices]

# 打乱测试集
shuffle_indices_test = np.random.permutation(len(X_test))
X_test = X_test[shuffle_indices_test]
y_test = y_test[shuffle_indices_test]


X_train = X_train.reshape(X_train.shape[0], -1)[:1000, :]
y_train = y_train[:1000]
X_test = X_test.reshape(X_test.shape[0], -1)
X_test = X_test.reshape(X_test.shape[0], -1)[:100, :]
y_test = y_test[:100]
