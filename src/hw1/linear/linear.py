from plugins.clfs import Clfs


class LinearClf(Clfs):
    def __init__(self):
        super(LinearClf, self).__init__()

    def fit(self, X_train, y_train):
        pass

    def predict_proba(self, x_test):
        pass

    def predict(self, x_test):
        pass

    def get_testing_time(self):
        pass

    def get_training_time(self):
        pass
