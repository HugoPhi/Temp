import tensorflow as tf
import numpy as np

# 加载CIFAR-10数据集 -> np.ndarray
(X_train, y_train), (X_test, y_test) = tf.keras.datasets.cifar10.load_data()

sed = np.random.randint(0, 1000)
print(f'seed is {sed}')
np.random.seed(sed)

# 打乱训练集
shuffle_indices = np.random.permutation(len(X_train))
X_train = X_train[shuffle_indices] / 255.0
y_train = y_train[shuffle_indices]

# 打乱测试集
shuffle_indices_test = np.random.permutation(len(X_test))
X_test = X_test[shuffle_indices_test] / 255.0
y_test = y_test[shuffle_indices_test]


X_train = X_train.reshape(X_train.shape[0], -1)[:, :]
y_train = y_train[:]
X_test = X_test.reshape(X_test.shape[0], -1)
X_test = X_test.reshape(X_test.shape[0], -1)[:100, :]
y_test = y_test[:100]
