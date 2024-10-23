# 机器学习实验

[English Version](README.md)

本仓库包含基于 numpy，pandas，matplotlib 的机器学习模型从零实现，旨在帮助学习者理解各类机器学习算法的内部工作原理。如有错误请多指教，可以在issue里提出你的疑问或者提出Pull Request如果你可以对错误提出修改。

## 功能
- 手动实现机器学习算法，并封装成python包以便测试
- 较为完备的基础测试
- 方便，自由地创建测试和运行测试

## 依赖要求
通过requirements.txt配置即可，推荐使用[pload](https://github.com/HugoPhi/python_venv_loader)作为虚拟环境管理工具，这是一个轻量级的python虚拟环境管理工具，不同于某些繁杂而庞大的工具，它只专注与python环境的管理，功能也许不太完善希望大家提出issue。

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


## 许可
本项目遵循 MIT 许可协议，详情请参阅 LICENSE 文件。
