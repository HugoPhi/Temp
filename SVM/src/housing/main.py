import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, r2_score

# 加载数据
data = pd.read_excel('./housing.xlsx')

# 查看数据的前几行，确保加载成功
print(data.head())

# 假设数据的最后一列为目标变量（房价），其余列为特征
X = data.iloc[:, :-1].values  # 特征
y = data.iloc[:, -1].values   # 目标变量（房价）

# 数据标准化（特征缩放）
scaler_X = StandardScaler()
X = scaler_X.fit_transform(X)

scaler_y = StandardScaler()
y = scaler_y.fit_transform(y.reshape(-1, 1)).flatten()  # 转换为 1D 数组

# 划分数据集（80% 用于训练，20% 用于测试）
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 创建支持向量回归模型，使用 RBF 核
svr = SVR(kernel='rbf', C=0.5, epsilon=0.01)
svr.fit(X_train, y_train)

# 预测
y_pred = svr.predict(X_test)

# 将预测结果和测试集目标值反转换为原始房价单位
y_test_orig = scaler_y.inverse_transform(y_test.reshape(-1, 1)).flatten()
y_pred_orig = scaler_y.inverse_transform(y_pred.reshape(-1, 1)).flatten()

# 评估模型
mse = mean_squared_error(y_test_orig, y_pred_orig)
r2 = r2_score(y_test_orig, y_pred_orig)
print("均方误差 (MSE):", mse)
print("R² 分数:", r2)
