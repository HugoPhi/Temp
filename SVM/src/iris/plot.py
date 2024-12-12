from sklearn import datasets
import matplotlib.pyplot as plt

iris = datasets.load_iris()
X = iris.data
y = iris.target
feature_names = iris.feature_names

print(X.shape)
fig, ax = plt.subplots(4, 3, figsize=(20, 20), dpi=500)

for i in range(4):
    ix = 0
    for j in range(4):
        if i != j:
            for k in range(len(iris.target_names)):
                ax[i, ix].scatter(X[y == k, i], X[y == k, j], label=iris.target_names[k])
            ax[i, ix].set_xlabel(feature_names[i])
            ax[i, ix].set_ylabel(feature_names[j])
            ax[i, ix].legend()
            ix += 1

plt.tight_layout()
plt.savefig('./iris.png')