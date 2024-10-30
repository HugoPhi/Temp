import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler, LabelEncoder

# 加载数据集
file_path = './winequality.xlsx'
wine_data = pd.read_excel(file_path)

# 分离特征和目标标签
X = wine_data.drop('quality label', axis=1)
y = wine_data['quality label']

# 编码目标标签
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

# 标准化特征
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# sed = np.random.randint(0, 1000)
# print(f'sed is: {sed}')

# 定义SVM模型并指定参数
svm = SVC(degree=3, C=1.5, gamma='scale', kernel='rbf', class_weight='balanced', random_state=42)
# 训练模型
svm.fit(X_train, y_train)

# 预测测试集
y_pred = svm.predict(X_test)

# 输出分类报告和模型评估
from hym.LogisticRegression import Metrics  # 假设 Metrics 类在此模块中定义
mt = Metrics(y_test, y_pred, 3)
print(mt)
