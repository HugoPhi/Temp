''' 作业操作说明

一、基本内容：
请同学们按照代码中给的TODO提示填写关键部分(pass部分)的代码，要求

k_nearest_neighbor.py 
linear_classifier.py
softmax.py
linear_svm.py
TwoLayerNet.py
cross-validation.py  

按照以上顺序填写，确保最终提交一份为以 .py 为扩展名的 Python 文件，命名'作业一_学号_姓名.py'。（所有类按顺序整合到一份文件中）
其中cross-validation.py 文件未定义特定类，要求复制整份代码，根据TODO题目提示，填写关键部分代码。

二、进阶内容：
可在Mindspore平台或者本地搭建python环境，配置环境所需（pytorch等），根据cross-validation.py给出的代码提示，设计5/10折交叉验证实验，验证KNN中最佳K值（自选数值范围）。
另外提交一份word文件，文件内格式不限，内容包含实验结果和实验源代码等，文件命名形式'作业一补充实验_学号_姓名'。

其中基本内容示例如下：
'''

# 1.k_nearest_neighbor.py：

class KNearestNeighbor(object):
    ...

# 2.linear_classifier.py

class LinearClassifier(object):
    ...

# 3.softmax.py

def softmax_loss_naive(W, X, y, reg):

    """
    Softmax loss function, naive implementation (with loops)

    Inputs have dimension D, there are C classes, and we operate on minibatches
    of N examples.

    Inputs:
    - W: A numpy array of shape (D, C) containing weights.
    - X: A numpy array of shape (N, D) containing a minibatch of data.
    - y: A numpy array of shape (N,) containing training labels; y[i] = c means
      that X[i] has label c, where 0 <= c < C.
    - reg: (float) regularization strength

    Returns a tuple of:
    - loss as single float
    - gradient with respect to weights W; an array of same shape as W
    """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)

    #############################################################################
    # TODO: Compute the softmax loss and its gradient using explicit loops.     #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    loss += reg * np.sum(W * W)
    dW += 2 * reg * W     #此处填写答案

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW
