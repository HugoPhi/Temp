# 机器学习实验

[English Version](README.md)

本仓库包含基于 numpy，pandas，matplotlib 的机器学习模型从零实现，旨在帮助学习者理解各类机器学习算法的内部工作原理。如有错误请多指教，可以在issue里提出你的疑问或者提出Pull Request如果你可以对错误提出修改。

## 功能
- 手动实现机器学习算法，并封装成python包以便测试
- 较为完备的基础测试
- 方便，自由地创建测试和运行测试

## 依赖要求
通过requirements.txt配置即可，推荐使用[pload](https://github.com/HugoPhi/python_venv_loader)作为虚拟环境管理工具，这是一个轻量级的python虚拟环境管理工具，不同于某些繁杂而庞大的工具，它只专注与python环境的管理，功能也许不太完善希望大家提出issue。具体操作如下：
```bash
pload new -m 'MLLabs' -v 
```

## 使用方法
这里介绍一些基本的用途，譬如如何运行基准测试（指已经创建好的用来检验模型的正确性的性能）、自定义模型、创建运行测试：

### 1. 运行基本测试 
先为项目创建一个特定的虚拟环境，并进入到虚拟环境，使用pload创建的方法在：[使用pload创建虚拟环境](https://github.com/HugoPhi/python_venv_loader)。
1. 克隆仓库：
    ```bash
    git clone --branch main --single-branch https://github.com/HugoPhi/MachineLearningLabs.git
    ```
2. 编译本地库：
    先进入到`MachineLearningLabs/`即本项目的工程目录下：
    ```bash
    cd MachineLearningLabs/
    ```
    然后编译本地库：
    ```bash
    pip install .
    ```
    可以通过检查`pip list`命令列出来的库检查是否包含`hym`来判断是否已经安装成功。
3. 运行你希望运行测试的目录下的main.py：
    比如运行：`test\DecisionTree\watermelon2.0\`实验，可以在工程目录下运行：
    ```bash
    python ./test/DecisionTree/watermelon2.0/main.py
    ```
    即可得到实验的结果。

### 2. 自定义模型
你可以修改或者完全实现你自己的机器学习模型，这只是一个不太完美的参考。你只需要在修改`src\`文件夹下面相应的算法的文件夹下面的实现，级可以实现你自己的算法。然后回到工程目录下重新编译就可以正常使用自定义的模型。在此之前，希望你了解一下本项目的构造以方便你作出更高效、正确的修改。
本项目的工程结构主要构成就是两部分：src和test，build为构建自动生成的文件。src用于存放机器学习算法的源码包括：算法的实现，数据集的加载，辅助函数。而test则负责存放各种算法的基础测试以及自定义测试。两个文件夹内部都有以下的规范。

#### 2.1. src
这是一个用来存放源文件的文件夹，这里的整体的结构如下：
```
\src\   
|   \hym\   
|   |   __init__.py    
|   |   DecisionTree\    
|   |   |   __init__.py    
|   |   |   DecisionTree.py    
|   |   |   ...    
|   |   LinearRegression\    
|   |   ...    
```
我们从最外面开始讲。src里面只有一个文件夹就是hym，它是我们要制作的算法包的顶层模块。然后在下面的是以一类机器学习算法命名的文件夹，代表一类算法，通常这下面需要实现一个基类包含最基本的算法流程，然后再继承它实现其它派生类。如果你想新增一类机器学习算法，譬如支持向量机，那么你就可以在hym下面创建一个新的文件夹叫`SupportVectorMachine\`。同时，要注意的是你需要在hym下的__init__.py文件里面新添加导入信息：
```python
from . import DecisionTree
from . import LinearRegression
from . import SupportVectorMachine  # 加入你想实现的算法模块

__all__ = [
    'DecisionTree',
    'LinearRegression',
    'SupportVectorMachine'] # 在最后加入新导入的模块
```
这样你的算法模块才可以在构建的时候被识别到。
对于一个算法模块内，这里用决策树算法作为例子，有一些文件的使用也是有规范的：
1. 包含算法类的文件：大驼峰命名，用来实现整个算法的类。比如：BasicDecisionTree.py，Variants.py。里面实现的都是具体决策树算法的类。
2. 包含辅助类的文件：小写下划线命名，用来实现辅助算法类的实现。比如node.py，里面有Node和Leaf两个类，是决策树的组成部分。
3. 辅助函数文件：utils.py，用来实现一些辅助算法的库，比如数据集加载，预处理，以及算法中用到的一些数学函数式等与主算法无关的函数。
4. 包初始化文件：\_\_init\_\_.py，用来标记一个包，让外界知道这是包或者包的子模块。通常要将你想向外传输的内容加入到__all__列表以隔绝某些内部实现。

#### 2.2. test
这是一个用来存放测试的文件夹，整体结构与src十分相似，在功能上也与src具有对应关系：
```txt
\test\    
|   DecisionTree\
|   |   iris\
|   |   |   iris.xlsx
|   |   |   main.py
|   |   watermelon2.0\
|   |   |   watermelon2.0.xlsx
|   |   |   main.py
|   |   ...
|   LinearRegression\    
|   ... 
```
test下面先分类别创建各种不同算法类的文件，这里必须和src里面各种机器学习算法类的名称对应。然后在对应的算法下面创建对某个算法的测试。这里已经实现了一下比较基础数据集的测试，读者也可以自行创建一些实验，以便更好地测试自定义算法。

#### 2.3. 其它重要文件
1. setup.py
这里面是关于包的构建信息的文件，包括版本、依赖库、作者等信息。其中版本信息的更新是遵循以下规范：
版本的格式是：v\[x\].\[y\].\[z\]，其中各个参量的意义如下：
- x：当版本有重大更新（比如项目结构和使用方法发生巨大改变时）以至于旧的API不再被新的API所兼容的时候需要更换这个版本号。
- y：当版本添加较大新的功能的时候更新。比如实现了新的机器学习算法类。
- z：阶段性更新，通常在修改Bugs，添加小功能，或者较小的改动时使用。

2. README.md
这里会记录项目的使用方法以及最新的一些更改之类的。建议定期阅读，因为我会定期更新（至少在项目的上升期是这样的）。

## 参考文献














## 许可
本项目遵循 MIT 许可协议，详情请参阅 LICENSE 文件。
