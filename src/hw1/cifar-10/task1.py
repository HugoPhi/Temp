from plugins.executer import Excuter

from knn import KNNClf
from data_process import X_train, X_test, y_test, y_train

n_train = 5000
n_test = 100
X_train = X_train[:n_train]
y_train = y_train[:n_train]
X_test = X_test[:n_test]
y_test = y_test[:n_test]


clf_dict = {}
for k in range(1, 20):
    clf_dict[f'knn_{k}'] = KNNClf(k=k, d='manhattan', batch_size=(512, 2048), backend='torch')

exc = Excuter(
    X_train, y_train, X_test, y_test,
    metric_list=['accuracy', 'avg_recall'],
    clf_dict=clf_dict,
    log=False,
    log_dir='./task1/'
)

exc.run_all()
