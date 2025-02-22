import numpy as np


class Metrics:
    '''
    分类模型的评价指标。

    Examples
    --------
    ```python
    y_true = np.array([0, 1, 2])         # 真实标签
    y_pred = np.array([[0.7, 0.1, 0.2],  # 对应样本的预测概率，一行为一个样本
                       [0.3, 0.3, 0.4],
                       [0.2, 0.1, 0.7]])
    ```
    尤其是二分类，一定要做成两类：
    ```python
    y_true = np.array([0, 1])       # 真实标签
    y_pred = np.array([[0.9, 0.1],  # 对应样本的预测概率，一行为一个样本
                       [0.3, 0.7],
                       [0.2, 0.8]]
    ```
    '''

    def __init__(self, y, y_pred, proba=True):
        '''
        初始化。

        Parameters
        ----------
        y : np.ndarray
            真实标签。
        y_pred : np.ndarray
            预测标签。
        proba : bool
            输入是否为概率向量。
        '''

        self.proba = proba
        self.y = y
        self.y_pred = y_pred
        uni = np.unique(self.y)
        if uni[0] != 0:
            raise ValueError('y must start from 0')

        self.classes = uni.shape[0]  # get classes num

        self.matrix = np.zeros((self.classes, self.classes))  # get confusion matrix

        if self.proba:
            for i, j in zip(y, np.argmax(y_pred, axis=1)):
                self.matrix[i, j] += 1
        else:
            for i, j in zip(y, y_pred):
                self.matrix[i, j] += 1

    def precision(self):
        '''
        Compute the precision for each class.

        Precision is the ratio of true positives to the total predicted positives.

        Returns
        -------
        numpy.ndarray
            The precision of each class.
        '''
        return np.diag(self.matrix) / self.matrix.sum(axis=0)

    def recall(self):
        '''
        Compute the recall for each class.

        Recall is the ratio of true positives to the total actual positives.

        Returns
        -------
        numpy.ndarray
            The recall of each class.
        '''

        return np.diag(self.matrix) / self.matrix.sum(axis=1)

    def f1(self):
        '''
        Compute the F1 score for each class.

        The F1 score is the harmonic mean of precision and recall.

        Returns
        -------
        numpy.ndarray
            The F1 score of each class.
        '''

        return 2 * self.precision() * self.recall() / (self.precision() + self.recall())

    def accuracy(self):
        '''
        Compute the overall accuracy.

        Accuracy is the ratio of correctly predicted instances to the total instances.

        Returns
        -------
        float
            The accuracy of the model.
        '''

        return np.diag(self.matrix).sum() / self.matrix.sum()

    def roc(self):
        '''
        Compute the ROC curve for each class. Only callable when 'proba == Ture'

        Uses the one-vs-rest ('ovr') approach and returns the AUC for each class.

        Returns
        -------
        list of tuple
            A list where each element is a tuple containing true positive rates (TPR)
            and false positive rates (FPR) for each class.
        '''

        if not self.proba:
            raise ValueError('roc() can only be called when proba == True')

        def calculate_tpr_fpr(y_true, y_pred):
            tp = np.sum((y_pred == 1) & (y_true == 1))
            tn = np.sum((y_pred == 0) & (y_true == 0))
            fp = np.sum((y_pred == 1) & (y_true == 0))
            fn = np.sum((y_pred == 0) & (y_true == 1))

            tpr = tp / (tp + fn) if (tp + fn) > 0 else 0
            fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
            return tpr, fpr

        rocs = []
        for class_idx in range(self.classes):
            tprs = []
            fprs = []
            thresholds = self.y_pred[:, class_idx].reshape(-1)
            for threshold in np.sort(thresholds)[::-1]:
                idx_pred = (self.y_pred[:, class_idx] >= threshold).astype(int)  # '=' here is important
                idx_true = (self.y == class_idx).astype(int).reshape(-1)
                tpr, fpr = calculate_tpr_fpr(idx_true, idx_pred)
                tprs.append(tpr)
                fprs.append(fpr)

            rocs.append((tprs, fprs))

        return rocs

    def auc(self):
        '''
        Compute the AUC for each class. Only callable when 'proba == Ture'

        Uses the one-vs-rest ('ovr') approach to calculate the AUC for each class.

        Returns
        -------
        numpy.ndarray
            The AUC of each class.
        '''

        if not self.proba:
            raise ValueError('auc() can only be called when proba == True')

        rocs = self.roc()
        aucs = []
        for (tprs, fprs) in rocs:
            auc = 0
            for i in range(1, len(fprs)):
                auc += (fprs[i] - fprs[i - 1]) * (tprs[i] + tprs[i - 1]) / 2
            aucs.append(auc)

        return np.array(aucs)

    def ap(self):
        '''
        Compute the Average Precision (AP) for each class. Only callable when 'proba == Ture'

        Uses the one-vs-rest ('ovr') approach to calculate the AP for each class.

        Returns
        -------
        numpy.ndarray
            The average precision (AP) of each class.
        '''

        if not self.proba:
            raise ValueError('ap() can only be called when proba == True')

        def calculate_prec_rec(y_true, y_pred):
            tp = np.sum((y_pred == 1) & (y_true == 1))
            fp = np.sum((y_pred == 1) & (y_true == 0))
            fn = np.sum((y_pred == 0) & (y_true == 1))

            prec = tp / (tp + fp) if (tp + fp) > 0 else 0
            rec = tp / (tp + fn) if (fp + fn) > 0 else 0
            return prec, rec

        aps = []
        for class_idx in range(self.classes):
            precs = []
            recs = []
            thresholds = self.y_pred[:, class_idx].reshape(-1)
            for threshold in np.sort(thresholds)[::-1]:
                idx_pred = (self.y_pred[:, class_idx] >= threshold).astype(int)  # '=' here is important
                idx_true = (self.y == class_idx).astype(int).reshape(-1)
                prec, rec = calculate_prec_rec(idx_true, idx_pred)
                precs.append(prec)
                recs.append(rec)

            ap = 0
            for i in range(1, len(recs)):
                ap += (recs[i] - recs[i - 1]) * (precs[i] + precs[i - 1]) / 2
            aps.append(ap)

        return aps

    def avg_ap(self):
        '''
        Compute the average of average precision (AP) scores. Only callable when 'proba == Ture'

        Returns
        -------
        float
            The mean average precision score across all classes.
        '''

        if not self.proba:
            raise ValueError('avg_ap() can only be called when proba == True')

        return self.ap().mean()

    def avg_pre(self):
        '''
        Compute the average precision score.

        Returns
        -------
        float
            The mean precision score across all classes.
        '''

        return self.precision().mean()

    def avg_recall(self):
        '''
        Compute the average recall score.

        Returns
        -------
        float
            The mean recall score across all classes.
        '''

        return self.recall().mean()

    def avg_auc(self):
        '''
        Compute the average AUC score.

        Returns
        -------
        float
            The mean AUC score across all classes.
        '''

        return self.auc().mean()

    def macro_f1(self):
        '''
        Compute the macro F1 score.

        The macro F1 score is the average F1 score across all classes.

        Returns
        -------
        float
            The macro F1 score.
        '''

        return self.f1().mean()

    def micro_f1(self):
        '''
        Compute the micro F1 score.

        The micro F1 score is computed using the global counts of true positives,
        false positives, and false negatives across all classes.

        Returns
        -------
        float
            The micro F1 score.
        '''

        return 2 * self.precision().mean() * self.recall().mean() / (self.precision().mean() + self.recall().mean())

    def confusion_matrix(self):
        '''
        Get the confusion matrix.

        Returns
        -------
        numpy.ndarray
            The confusion matrix.
        '''

        return self.matrix

    def __repr__(self) -> str:
        '''
        Provide a string representation of the performance metrics.

        Returns
        -------
        str
            A formatted string showing precision, recall, F1, accuracy,
            macro average, and micro average.
        '''

        table = ' ' * 6
        print(f'        {table}Precision{table}Recall{table}  F1')
        for i in range(len(self.precision())):
            print(f'Class {i} {table}{self.precision()[i]:.6f} {table}{self.recall()[i]:.6f}{table}{self.f1()[i]:.6f}')
        print()
        print(f'Accuracy      {self.accuracy():.6f}')
        print(f'Macro F1      {self.macro_f1():.6f}')
        print(f'Micro F1      {self.micro_f1():.6f}')

        return ''
