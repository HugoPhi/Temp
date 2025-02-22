from plugins.clfs import Clfs
import time
import numpy as np


class KNNClf(Clfs):
    def __init__(self, k=1, p='euclid'):
        super(KNNClf, self).__init__()
        self.k = k
        self.p = p

        if p == 'euclid':
            self.distance = self.__euclid_distance
        elif p == 'manhattan':
            self.distance = self.__manhattan_distance
        elif p == 'cosine':
            self.distance = self.__cosine_distances
        elif p == 'chebyshev':
            self.distance = self.__chebyshev_distances
        else:
            print('p should be euclid, manhattan, cosine or chebyshev !')
            print('use default p: euclid')
            self.distance = self.__manhattan_distance

    def __euclid_distance(self, x, y):
        '''
        计算两个数据集的欧几里德距离，就是两两向量之间的欧几里德距离。

        Parameters
        ----------
        x, y: np.ndarray
            两个数据集，形状为(Nx, D)和(Ny, D)，Nx, Ny是数据集大小；D是向量的维度。

        Returns
        -------
        distance : np.ndarray
            两个数据集之间的距离，形状为(Ny, Nx)
        '''
        dists = np.sqrt(-2 * np.dot(y, x.T) + np.sum(x**2, axis=1) + np.sum(y**2, axis=1)[:, np.newaxis])
        return dists

    def __manhattan_distance(self, x, y):
        '''
        计算两个数据集的曼哈顿距离，就是两两向量之间的曼哈顿距离。

        Parameters
        ----------
        x, y: np.ndarray
            两个数据集，形状为(Nx, D)和(Ny, D），Nx, Ny是数据集大小；D是向量的维度。

        Returns
        -------
        distance : np.ndarray
            两个数据集之间的距离，形状为(Ny, Nx)
        '''

        dists = np.sum(np.abs(y[:, np.newaxis] - x), axis=2)
        return dists

    def __chebyshev_distances(X_train, X_test):
        '''
        计算两个数据集的切比雪夫距离，就是两两向量之间的切比雪夫距离。

        Parameters
        ----------
        x, y: np.ndarray
            两个数据集，形状为(Nx, D)和(Ny, D），Nx, Ny是数据集大小；D是向量的维度。

        Returns
        -------
        distance : np.ndarray
            两个数据集之间的距离，形状为(Ny, Nx)
        '''

        dists = np.max(np.abs(X_test[:, np.newaxis] - X_train), axis=2)
        return dists

    def __cosine_distances(X_train, X_test):
        '''
        计算两个数据集的余弦距离，就是两两向量之间的余弦距离。

        Parameters
        ----------
        x, y: np.ndarray
            两个数据集，形状为(Nx, D)和(Ny, D），Nx, Ny是数据集大小；D是向量的维度。

        Returns
        -------
        distance : np.ndarray
            两个数据集之间的距离，形状为(Ny, Nx)
        '''

        # 归一化向量以简化余弦距离的计算
        X_train_normalized = X_train / np.linalg.norm(X_train, axis=1, keepdims=True)
        X_test_normalized = X_test / np.linalg.norm(X_test, axis=1, keepdims=True)

        # 计算余弦相似度矩阵，然后转换为余弦距离
        similarity = np.dot(X_test_normalized, X_train_normalized.T)
        dists = 1 - similarity
        return dists

    def fit(self, X_train, y_train):
        start = time.time()

        self.x = X_train
        self.y = y_train.reshape(-1)
        classes, self.static = np.unique(self.y, return_counts=True)  # 统计每种类别

        if (classes != np.arange(classes.shape[0])).any():
            raise ValueError('Make sure y is start form 0 !')  # 确保类别是形如：0, 1, 2 ...

        self.static = self.static / self.static.sum()  # 每种类别可能的概率，用于抵抗数据不均衡

        self.training_time = time.time() - start

    def predict(self, x_test):
        proba = self.predict_proba(x_test)
        diff = proba - self.static

        return np.argmax(diff, axis=1)  # 可能性最大

    def predict_proba(self, x_test):
        start = time.time()

        proba = np.zeros((x_test.shape[0], self.static.shape[0]))  # 每种类别的可能性，（N_test, C）

        n_k_neighbors = np.argsort(self.distance(self.x, x_test), axis=1)[:, :self.k]  # K个邻居

        proba[np.arange(x_test.shape[0])[:, None], self.y[n_k_neighbors]] += 1  # 统计可能性

        proba = proba / np.sum(proba, axis=1, keepdims=True)  # 计算概率

        self.testing_time = time.time() - start

        return proba

    def get_params(self):
        return {}, {'k': self.k, 'p': self.p}

    def get_testing_time(self):
        return self.testing_time

    def get_training_time(self):
        return self.training_time
