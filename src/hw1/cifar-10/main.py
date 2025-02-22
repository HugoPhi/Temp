from data_process import X_train, X_test, y_test, y_train
from plugins.excuter import Excuter

from knn import KNNClf

clf_dict = {}
for k in range(1, 20):
    clf_dict[f'knn_{k}'] = KNNClf(k=k, p='euclid')

exc = Excuter(
    X_train=X_train,
    X_test=X_test,
    y_train=y_train,
    y_test=y_test,
    metric_list=['accuracy', 'avg_recall'],
    clf_dict=clf_dict,
    log=False
)

exc.run_all(sort_by='accuracy', ascending=False)
