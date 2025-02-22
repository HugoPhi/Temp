import atexit
import toml
from datetime import datetime
import os
import traceback
import pandas as pd
from .metric import Metrics


class Excuter:
    '''
    Clf的执行器
    ==========
      - 快捷管理训练，测试，日志全过程，并灵活调试Classifier数组里面的各个模型。
      - 开启Log，支持中途运行出错，结果不丢失。
      - 在使用的时候根据需要重写excute(self)方法。

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
                 metric_list=['acc', 'macro_f1', 'micro_f1', 'avg_recall'],
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

        self.df = pd.DataFrame(columns=['model'] + self.metric_list + ['training time'] + ['testing time'])

        # log
        if log:
            self.log_dir = log_dir
            if not os.path.exists(self.log_dir):
                os.makedirs(self.log_dir)

            self.log_path = os.path.join(self.log_dir, f'{datetime.now().strftime("%Y_%m_%d_%H-%M-%S")}/')
            os.mkdir(self.log_path)

            hyper_config = dict()
            for name, clf in clf_dict.items():
                hyper_config[name] = dict()
                hyper_config[name]['hyper'], hyper_config[name]['model'] = clf.get_params()

            toml.dump(hyper_config, open(os.path.join(self.log_path, 'hyper.toml'), 'w'))  # 保存超参数和模型参数

            atexit.register(self.save_df)  # 保证退出的时候能保存已经生成的df

    def save_df(self):
        '''
        保存df到日志
        '''
        self.df.to_csv(os.path.join(self.log_path, 'result.csv'), index=False)

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

                y_pred = clf.predict_proba(self.X_test)

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

        self.df.loc[len(self.df)] = [name] + [func() for func in func_list] + [clf.get_training_time(), clf.get_testing_time()]

    def run(self, key):
        '''
        运行单个实验。不会消耗clf_dict。

        Parameters
        ----------
        key : str
            实验的名字。
        '''
        if key in self.clf_dict.keys():
            mtc, clf = self.excute(key, self.clf_dict[key])

            self.logline(key, mtc, clf)
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
            print(self.df.sort_values(sort_by, ascending=ascending))
        else:
            print(self.df)

    def get_result(self):
        '''
        返回实验结果对应的表格。

        Returns
        -------
        self.pd : pd.DataFrame
        '''
        return self.df
