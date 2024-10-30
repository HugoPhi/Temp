import numpy as np
from itertools import combinations
from collections import Counter

class CustomSVC:
    def __init__(self, C=1.0, kernel='rbf', gamma='scale', max_iter=1000, tol=1e-3):
        self.C = C  # 惩罚参数
        self.kernel = kernel
        self.gamma = gamma
        self.max_iter = max_iter
        self.tol = tol  # 停止条件
        self.models = {}  # 存储所有类别对的分类器

    def _rbf_kernel(self, X, Y):
        """计算高斯核矩阵"""
        if self.gamma == 'scale':
            gamma = 1 / (X.shape[1] * X.var())
        else:
            gamma = self.gamma
        K = np.exp(-gamma * np.sum((X[:, np.newaxis] - Y) ** 2, axis=2))
        return K

    def _compute_kernel(self, X, Y):
        """计算核函数矩阵"""
        if self.kernel == 'rbf':
            return self._rbf_kernel(X, Y)
        else:
            raise ValueError("仅支持 RBF 核")

    def fit(self, X, y):
        # 生成所有类别对的组合，使用 One-vs-One 策略
        self.classes_ = np.unique(y)
        for (class_1, class_2) in combinations(self.classes_, 2):
            # 筛选出这两个类别的样本
            idx = np.where((y == class_1) | (y == class_2))[0]
            X_pair = X[idx]
            y_pair = np.where(y[idx] == class_1, 1, -1)  # 将类别转换为 +1 和 -1

            # 训练一个二分类模型
            alpha = np.zeros(X_pair.shape[0])
            b = 0
            K = self._compute_kernel(X_pair, X_pair)

            # 梯度下降优化
            for _ in range(self.max_iter):
                alpha_prev = np.copy(alpha)
                for i in range(X_pair.shape[0]):
                    condition = y_pair[i] * (np.dot((alpha * y_pair), K[i]) - b) < 1
                    if condition:
                        alpha[i] += self.C * (1 - y_pair[i] * np.dot((alpha * y_pair), K[i]))
                        b += self.C * (y_pair[i] - y_pair[i] * np.dot((alpha * y_pair), K[i]))

                # 检查收敛
                if np.linalg.norm(alpha - alpha_prev) < self.tol:
                    break

            # 保存模型参数
            self.models[(class_1, class_2)] = (alpha, b, X_pair, y_pair)

    def decision_function(self, X, alpha, b, X_train, y_train):
        """计算决策函数值"""
        K = self._compute_kernel(X, X_train)
        decision = np.dot(K, alpha * y_train) - b
        return decision

    def predict(self, X):
        """多类别预测，通过投票机制确定类别"""
        votes = np.zeros((X.shape[0], len(self.classes_)))
        for (class_1, class_2), (alpha, b, X_train, y_train) in self.models.items():
            decision = self.decision_function(X, alpha, b, X_train, y_train)
            predictions = np.where(decision >= 0, class_1, class_2)
            for i, pred in enumerate(predictions):
                votes[i, np.where(self.classes_ == pred)[0][0]] += 1
        return self.classes_[np.argmax(votes, axis=1)]

# 替换代码示例中的 SVC
import hym.LogisticRegression as lr
from sklearn import datasets
from sklearn.model_selection import train_test_split

# 加载鸢尾花数据集
iris = datasets.load_iris()
X = iris.data
y = iris.target

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 使用自定义的 SVM 模型，带有高斯核 (RBF kernel)
svm = CustomSVC(C=1.0, kernel='rbf', gamma='scale')
svm.fit(X_train, y_train)

# 预测
y_pred = svm.predict(X_test)

# 评估分类效果
mt = lr.Metrics(y_test, y_pred, 3)
print(mt)
