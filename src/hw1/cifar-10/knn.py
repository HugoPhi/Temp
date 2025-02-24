from plugins.clfs import Clfs
import time
import numpy as np


class KNNClf(Clfs):
    def __init__(self, k=1, d='euclid', batch_size=128):
        super(KNNClf, self).__init__()
        self.k = k
        self.d = d
        self.batch_size = batch_size

        if d == 'euclid':
            self.distance = self.__euclid_distance
        elif d == 'manhattan':
            self.distance = self.__manhattan_distance
        elif d == 'cosine':
            self.distance = self.__cosine_distances
        elif d == 'chebyshev':
            self.distance = self.__chebyshev_distances
        else:
            print('d should be euclid, manhattan, cosine or chebyshev !')
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
        dists = np.sqrt(
            np.sum(y**2, axis=1)[:, np.newaxis] + np.sum(x**2, axis=1) - 2 * np.dot(y, x.T)
        )
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

    def __chebyshev_distances(self, x, y):
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

        dists = np.max(np.abs(y[:, np.newaxis] - x), axis=2)
        return dists

    def __cosine_distances(self, x, y):
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
        x_normalized = x / np.linalg.norm(x, axis=1, keepdims=True)
        y_normalized = y / np.linalg.norm(y, axis=1, keepdims=True)

        # 计算余弦相似度矩阵，然后转换为余弦距离
        similarity = np.dot(y_normalized, x_normalized.T)
        dists = 1 - similarity
        return dists

    def fit(self, X_train, y_train):
        start = time.time()

        self.x = X_train
        self.y = y_train.reshape(-1)
        classes, self.static = np.unique(self.y, return_counts=True)  # 统计每种类别

        if classes[0] != 0:
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

        for i in range(0, x_test.shape[0], self.batch_size):
            n_k_neighbors = np.argsort(self.distance(self.x, x_test[i:i + self.batch_size]), axis=1)[:, :self.k]  # K个邻居

            proba[i + np.arange(x_test[i:i + self.batch_size].shape[0])[:, None], self.y[n_k_neighbors]] += 1  # 统计可能性

            proba[i:i + self.batch_size] = proba[i:i + self.batch_size] / np.sum(proba[i:i + self.batch_size], axis=1, keepdims=True)  # 计算概率

        self.testing_time = time.time() - start

        return proba

    def get_params(self):
        return {'batch_size': self.batch_size}, {'k': self.k, 'd': self.d}

    def get_testing_time(self):
        return self.testing_time

    def get_training_time(self):
        return self.training_time
