from plugins.executer import Excuter

from knn import KNNClf
from data_process import X_train, X_test, y_test, y_train

n_train = 50000
n_test = 10000
X_train = X_train[:n_train]
y_train = y_train[:n_train]
X_test = X_test[:n_test]
y_test = y_test[:n_test]


clf_dict = {}
k = 5

clf_dict[f'knn_{k}_12_numpy'] = KNNClf(k=k, d='manhattan', batch_size=(512, 2048), backend='numpy')
clf_dict[f'knn_{k}_x2_numpy'] = KNNClf(k=k, d='manhattan', batch_size=(1, 50000), backend='numpy')
clf_dict[f'knn_{k}_1x_numpy'] = KNNClf(k=k, d='manhattan', batch_size=(10000, 1), backend='numpy')
clf_dict[f'knn_{k}_xx_numpy'] = KNNClf(k=k, d='manhattan', batch_size=(1, 1), backend='numpy')

clf_dict[f'knn_{k}_12_torch'] = KNNClf(k=k, d='manhattan', batch_size=(512, 2048), backend='torch')
clf_dict[f'knn_{k}_x2_torch'] = KNNClf(k=k, d='manhattan', batch_size=(1, 50000), backend='torch')
clf_dict[f'knn_{k}_1x_torch'] = KNNClf(k=k, d='manhattan', batch_size=(10000, 1), backend='torch')
clf_dict[f'knn_{k}_xx_torch'] = KNNClf(k=k, d='manhattan', batch_size=(1, 1), backend='torch')

exc = Excuter(
    X_train, y_train, X_test, y_test,
    metric_list=['accuracy', 'avg_recall'],
    clf_dict=clf_dict,
    log=True,
    log_dir='./task1/'
)

exc.run_all()
