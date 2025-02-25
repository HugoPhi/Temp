import atexit
import toml
from datetime import datetime
import os
import traceback
import pandas as pd
import numpy as np
from .metric import Metrics


class Excuter:
    '''
    Clf的执行器
    ==========
      - 快捷管理训练，测试，日志全过程，并灵活调试Classifier数组里面的各个模型。
      - 开启Log，支持中途运行出错，结果不丢失。
      - 在使用的时候根据需要重写excute(self)方法。
      - 这个类没有Valid。

    Parameters
    ----------
    X_train : np.ndarray
        训练集的X。
    y_train : np.ndarray
        训练集的y。
    X_test : np.ndarray
        测试集的X。
    y_test : np.ndarray
        测试集的y。
    clf_dict : dict
        Clf字典。包含多个实验的{name : Clf}
    metric_list : list
        测评指标列表。在两端分别加上name和time之后，作为结果表格的表头。
    log : bool
        是否开启日志。开启之后会将过程参数写入到对应文件夹的hyper.toml，将结果写入到同一文件夹的result.csv。
    log_dir : str
        存放日志的文件夹。日志会被放到一个日期为名字的子文件夹里面。
    '''

    def __init__(self, X_train, y_train, X_test, y_test,
                 clf_dict: dict,
                 metric_list=['accuracy', 'macro_f1', 'micro_f1', 'avg_recall'],
                 log=False,
                 log_dir='./log/'):
        '''
        初始化。

        Parameters
        ----------
        X_train : np.ndarray
            训练集的X。
        y_train : np.ndarray
            训练集的y。
        X_test : np.ndarray
            测试集的X。
        y_test : np.ndarray
            测试集的y。
        clf_dict : dict
            Clf字典。包含多个实验的{name : Clf}
        metric_list : list
            测评指标列表。在两端分别加上name和time之后，作为结果表格的表头。
        log : bool
            是否开启日志。开启之后会将过程参数写入到对应文件夹的hyper.toml，将结果写入到同一文件夹的result.csv。
        log_dir : str
            存放日志的文件夹。日志会被放到一个日期为名字的子文件夹里面。
        '''

        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test
        self.clf_dict = clf_dict
        self.metric_list = metric_list
        self.log = log

        self.test = pd.DataFrame(columns=['model'] + self.metric_list + ['training time'] + ['testing time'])

        # log
        if log:
            self.log_dir = log_dir
            if not os.path.exists(self.log_dir):
                os.makedirs(self.log_dir)

            self.log_path = os.path.join(self.log_dir, f'{datetime.now().strftime("%Y_%m_%d_%H-%M-%S")}/')
            os.mkdir(self.log_path)

            hyper_config = dict()
            for name, clf in clf_dict.items():
                hyper_config[name] = clf.get_params()

            toml.dump(hyper_config, open(os.path.join(self.log_path, 'hyper.toml'), 'w'))  # 保存超参数和模型参数

            atexit.register(self.save_df)  # 保证退出的时候能保存已经生成的df

    def save_df(self):
        '''
        保存df到日志
        '''
        self.test.to_csv(os.path.join(self.log_path, 'test.csv'), index=False)

    def excute(self, name, clf):
        '''
        执行实验。

        Notes
        -----
          - 这里必须返回一个测试器和一个训练好的分类器，因为写入日志要用。

        Parameters
        ----------
        name : str
            实验的名字。
        clf : Clfs
            实验获取的分类器，继承自接口Clfs。

        Returns
        -------
        clf : Clfs
            训练好的分类器。
        metric: Metrics
            有记录的Metric实例。

        Examples
        --------

        可以这么重写：
        ```python
        class MyExcuter(Excuter):
            def excute(self, name, clf):
                print(f'>> {name}')

                clf.fit(self.X_train, self.y_train)
                print(f'Train {name} Cost: {clf.get_training_time():.4f} s')

                y_pred = clf.predict(self.X_test)

                mtc = Metrics(self.y_test, y_pred)

                return mtc, clf
        ```
        '''
        print(f'>> {name}')

        clf.fit(self.X_train, self.y_train)  # 训练分类器
        print(f'Train {name} Cost: {clf.get_training_time():.4f} s')

        y_pred = clf.predict(self.X_test)

        mtc = Metrics(self.y_test, y_pred, proba=False)  # 构建测试器
        print(f'Testing {name} Cost: {clf.get_testing_time():.4f} s')

        return mtc, clf  # 返回测试器和分类器

    def logline(self, name, mtc, clf):
        '''
        将某次实验的结果写入日志df。
        '''

        func_list = []
        for metric in self.metric_list:
            func = getattr(mtc, metric, None)
            if callable(func):
                func_list.append(func)
            else:
                raise ValueError(f'{metric} is not in Metric.')

        self.test.loc[len(self.test)] = [name] + [func() for func in func_list] + [clf.get_training_time(), clf.get_testing_time()]

    def run(self, key):
        '''
        运行单个实验。不会消耗clf_dict。

        Parameters
        ----------
        key : str
            实验的名字。

        Returns
        -------
        mtc : Metric
            本次实验的测试器。
        clf : any
            本次实验的分类器。
        '''
        if key in self.clf_dict.keys():
            mtc, clf = self.excute(key, self.clf_dict[key])

            self.logline(key, mtc, clf)

            return mtc, clf
        else:
            raise KeyError(f'{key} is not in clf_dict')

    def step(self):
        '''
        迭代运行实验。采用迭代器模式。会逐个消耗实验，直到clf_dict为空。过程中会返回对应的名字和Clf对象，如果是最后一个，返回None。

        Returns
        -------
        name : str
            实验的名字。
        clf : Clfs
            实验获取的分类器，继承自接口Clfs。
        '''
        if len(self.clf_dict) == 0:
            return None

        try:
            name, clf = self.clf_dict.popitem()

            mtc, clf = self.excute(name, clf)

            self.logline(name, mtc, clf)

            return name, clf
        except Exception as e:
            print(f'Error: {e}')
            traceback.print_exc()

    def run_all(self, sort_by=None, ascending=False):
        '''
        运行所有实验。

        Parameters
        ----------
        sort_by : str
            按照哪个指标进行排序。
        ascending : bool
            是否升序。
        '''

        for name, clf in self.clf_dict.items():
            mtc, clf = self.excute(name, clf)

            self.logline(name, mtc, clf)

        if sort_by is not None:
            print(self.test.sort_values(sort_by, ascending=ascending))
        else:
            print(self.test)

    def get_result(self):
        '''
        返回实验结果对应的表格。

        Returns
        -------
        self.pd : pd.DataFrame
        '''
        return self.test


class KFlodExcuter(Excuter):
    def __init__(self, X_train, y_train, X_test, y_test,
                 clf_dict: dict,
                 metric_list=['accuracy', 'macro_f1', 'micro_f1', 'avg_recall'],
                 k=10,
                 log=False,
                 log_dir='./log/'):
        super().__init__(X_train, y_train, X_test, y_test,
                         clf_dict, metric_list, log, log_dir)

        self.k = k
        metrics = self.metric_list + ['training time', 'testing time']
        self.valid = pd.DataFrame(columns=['model'] + [f'{x}_{suffix}' for x in metrics for suffix in ['mean', 'std']])

    def excute(self, name, clf):
        '''
        执行实验。

        Notes
        -----
          - 这里必须返回一个测试器和一个训练好的分类器，因为写入日志要用。

        Parameters
        ----------
        name : str
            实验的名字。
        clf : Clfs
            实验获取的分类器，继承自接口Clfs。

        Returns
        -------
        clf : Clfs
            训练好的分类器。
        metric: Metrics
            有记录的Metric实例。

        Examples
        --------

        可以这么重写：
        ```python
        class MyExcuter(Excuter):
            def excute(self, name, clf):
                print(f'>> {name}')

                clf.fit(self.X_train, self.y_train)
                print(f'Train {name} Cost: {clf.get_training_time():.4f} s')

                y_pred = clf.predict(self.X_test)

                mtc = Metrics(self.y_test, y_pred)

                return mtc, clf
        ```
        '''
        print(f'>> {name}')

        # k折交叉验证
        k_fold_x_train = np.array_split(self.X_train, self.k)
        k_fold_y_train = np.array_split(self.y_train, self.k)
        mtcs = []
        for i in range(self.k):
            x_train = np.concatenate(k_fold_x_train[:i] + k_fold_x_train[i + 1:])
            y_train = np.concatenate(k_fold_y_train[:i] + k_fold_y_train[i + 1:])
            x_test = k_fold_x_train[i]
            y_test = k_fold_y_train[i]
            clf.fit(x_train, y_train)

            y_pred = clf.predict(x_test)
            mtc = Metrics(y_test, y_pred, proba=False)
            mtcs.append(mtc)

        # real train & test
        clf.fit(self.X_train, self.y_train)  # 训练分类器
        print(f'Train {name} Cost: {clf.get_training_time():.4f} s')

        y_pred = clf.predict(self.X_test)

        mtc = Metrics(self.y_test, y_pred, proba=False)  # 构建测试器
        mtcs.append(mtc)
        print(f'Testing {name} Cost: {clf.get_testing_time():.4f} s')

        return mtcs, clf  # 返回所有测试器和分类器

    def logline(self, name, mtcs: list, clf):
        '''
        将某次实验的结果写入日志df。
        '''

        test_mtc = mtcs.pop()

        def getline(mtc):
            func_list = []
            for metric in self.metric_list:
                func = getattr(mtc, metric, None)
                if callable(func):
                    func_list.append(func)
                else:
                    raise ValueError(f'{metric} is not in Metric.')

            return [func() for func in func_list] + [clf.get_training_time(), clf.get_testing_time()]

        self.test.loc[len(self.test)] = [name] + getline(test_mtc)  # 获取测试的结果

        valid_rows = [getline(mtc) for mtc in mtcs]
        valids_array = np.array(valid_rows)

        mean_vals = np.mean(valids_array, axis=0).tolist()
        std_vals = np.std(valids_array, axis=0).tolist()

        valid_result = []
        for mean, std in zip(mean_vals, std_vals):
            valid_result.append(mean)
            valid_result.append(std)

        self.valid.loc[len(self.valid)] = [name] + valid_result

    def save_df(self):
        '''
        保存df到日志
        '''
        self.test.to_csv(os.path.join(self.log_path, 'test.csv'), index=False)
        self.valid.to_csv(os.path.join(self.log_path, 'valid.csv'), index=False)

    def run_all(self, sort_by=['accuracy', 'accuracy_mean'], ascending=False):
        '''
        运行所有实验。

        Parameters
        ----------
        sort_by : str
            按照哪个指标进行排序。
        ascending : bool
            是否升序。
        '''

        for name, clf in self.clf_dict.items():
            mtc, clf = self.excute(name, clf)

            self.logline(name, mtc, clf)

        if sort_by is not None:
            print(self.test.sort_values(sort_by[0], ascending=ascending))
            print(self.valid.sort_values(sort_by[1], ascending=ascending))
        else:
            print(self.test)
            print(self.valid)
