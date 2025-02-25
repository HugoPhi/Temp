import numpy as np
from knn import KNNClf
from utils import display_images
from data_process import X_train, X_test, y_test, y_train, class2name


n_train = 5000
n_test = 100
X_train = X_train[:n_train]
y_train = y_train[:n_train]
X_test = X_test[:n_test]
y_test = y_test[:n_test]


n_samples = 5  # 看几个样本
k = 5  # k个邻居

clf = KNNClf(k=k, d='manhattan', backend='torch', batch_size=(256, 2048))
clf.fit(X_train, y_train)  # 训练
print(f'pre proba: {clf.get_pre_proba()}')  # 打印各个类别占测试集的比例


imgs = []
labels = []
pred = []
for _ in range(5):
    ix = np.random.randint(0, X_test.shape[0])
    # kn = kkn[ix]
    pred.append(class2name[clf.predict(X_test[ix:ix + 1])[0]])
    kn = clf.get_k_neighbors()[0]

    def get_img_from_train(ix):
        img = X_train[ix].reshape(3, 32, 32).transpose((1, 2, 0))
        img = img * 0.5 + 0.5
        img = (img * 255).astype('uint8')

        return img

    def get_img_from_test(ix):
        img = X_test[ix].reshape(3, 32, 32).transpose((1, 2, 0))
        img = img * 0.5 + 0.5
        img = (img * 255).astype('uint8')

        return img

    imgs.append(get_img_from_test(ix))
    labels.append(class2name[y_test[ix]])

    for k_ix in kn:  # 邻居要从训练集里面选
        img = get_img_from_train(k_ix)
        imgs.append(img)
        labels.append(class2name[y_train[k_ix]])


print(f'pred: {pred}')
display_images(imgs, labels, n_samples, k + 1)
