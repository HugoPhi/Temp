import hym.LogisticRegression as lr
from sklearn import datasets
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import numpy as np

# 加载鸢尾花数据集
iris = datasets.load_iris()
X = iris.data
y = iris.target

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=42)

# 创建 SVM 模型，使用高斯核 (RBF kernel)
svm = SVC(kernel='linear', C=1, gamma='scale')
svm.fit(X_train, y_train)

# 预测
y_pred = svm.predict(X_test)

# 评估分类效果
# accuracy = accuracy_score(y_test, y_pred)
# print("分类准确率:", accuracy)
# print("\n分类报告:\n", classification_report(y_test, y_pred))

mt = lr.Metrics(y_test, y_pred, 3)
print(mt)

