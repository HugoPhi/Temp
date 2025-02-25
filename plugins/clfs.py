from abc import ABC, abstractmethod
import inspect
from typing import Tuple, Dict
from numpy.typing import NDArray


class Clfs(ABC):
    '''
    Classifier 接口
    ===============
    - fit(self, X_train, y_train, load=Flase): 训练分类器
    - predict(self, x_test): 返回x_test的预测值
    - predict_proba(self, x_test): 返回x_test的预测概率矩阵
    - get_params(self): 返回超参数字典，用于复刻实验
    - get_training_time(self): 获取训练时间，如果load=False
    - get_testing_time(self): 获取测试时间

    Notes
    -----
    1. 这样的接口设计可以使得学习器的训练和学习过程有一个同一的API，便于不同框架下模型的对比
    2. 这套接口让'预测器'和将要使用的数据集分割开；让预测器和模型的具体架构分割开
    '''

    def __new__(cls, *args, **kwargs):
        '''
        这个方法会在子类初始化的时候被调用，可以把子类的所有参数放入到params字典里面。
        '''
        instance = super().__new__(cls)
        original_init = cls.__init__
        sig = inspect.signature(original_init)
        bound_args = sig.bind(instance, *args, **kwargs)
        bound_args.apply_defaults()
        bound_args.arguments.pop('self', None)

        instance.params = {}

        def merge_params(params_dict, arguments):
            for key, value in arguments.items():
                if isinstance(value, dict):
                    merge_params(params_dict, value)
                else:
                    params_dict[key] = value

        merge_params(instance.params, bound_args.arguments)

        return instance

    @abstractmethod
    def __init__(self):
        '''
        初始化。
        '''

        self.training_time = -1
        self.testing_time = -1

    @abstractmethod
    def predict(self, x_test) -> NDArray:
        '''
        根据模型计算结果。直接返回预测结果。

        Notes
        -----
          - 返回y_train同样形式的结果向量。
          - 在这里要统计预测时间。

        Parameters
        ----------
        x_test : numpy.ndarray
            测试集
        '''

        pass

    @abstractmethod
    def predict_proba(self, x_test) -> NDArray:
        '''
        根据模型计算结果。返回预测的概率。

        Notes
        -----
          - 这里返回经过softmax之后的概率矩阵。
          - 在这里要统计预测时间。

        Parameters
        ----------
        x_test : numpy.ndarray
            测试集
        '''

        pass

    @abstractmethod
    def fit(self, X_train, y_train, load=False):
        '''
        训练模型。

        Notes
        -----
          - 在这里要统计测试时间。
        '''

        pass

    def get_params(self) -> Tuple[Dict]:
        '''
        获取超参数。

        Notes
        -----
          - 返回self.params，就是__init__()所需要的参数。用于编写日志复刻模型。

        Returns
        -------
        self.params : dict
            当前类的__init__的所有参数
        '''

        return self.params

    @abstractmethod
    def get_training_time(self):
        '''
        返回最近一次模型训练的时间。

        Notes
        -----
          - 返回训练模型的时间，通常返回self.training_time
        '''

        pass

    @abstractmethod
    def get_testing_time(self):
        '''
        返回最近一次模型测试的时间。

        Notes
        -----
          - 返回最近一次模型测试的时间，通常返回self.testing_time
        '''

        pass
