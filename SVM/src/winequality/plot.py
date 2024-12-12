import hym.DecisionTree as hdt
import pandas as pd
import hym.LogisticRegression as lr
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt



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
X_train = X_train.to_numpy()

names = list(wine_data.columns[:-1])
for i in range(X_train.shape[1]):
    for j in range(X_train.shape[1]):
        if i != j:
            plt.figure(figsize=(20, 20))
            plt.scatter(X_train[:, i], X_train[:, j], c=y_train, cmap='viridis')
            plt.title(f'{names[i]} vs {names[j]}')
            plt.savefig(f'./{names[i]}_{names[j]}.png', dpi=500)
            plt.close()
