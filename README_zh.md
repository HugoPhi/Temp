# 机器学习实验

[![GitHub stars](https://img.shields.io/github/stars/HugoPhi/MachineLearningLabs.svg?style=social)](https://github.com/HugoPhi/MachineLearningLabs/stargazers)
[![GitHub forks](https://img.shields.io/github/forks/HugoPhi/MachineLearningLabs.svg?style=social)](https://github.com/HugoPhi/MachineLearningLabs/network/members)
[![GitHub license](https://img.shields.io/github/license/HugoPhi/MachineLearningLabs.svg)](https://github.com/HugoPhi/MachineLearningLabs/blob/main/LICENSE)
[![GitHub issues](https://img.shields.io/github/issues/HugoPhi/MachineLearningLabs.svg)](https://github.com/HugoPhi/MachineLearningLabs/issues)

[English](README.md)

---

本仓库包含基于 `numpy`、`pandas`、`matplotlib` 从零实现的机器学习模型，旨在帮助学习者理解各类机器学习算法的内部工作原理。如果您在使用过程中发现任何错误，欢迎在 [Issues](https://github.com/HugoPhi/MachineLearningLabs/issues) 中提出，或者通过提交 Pull Request 来改进本项目。

## 功能

- 手动实现机器学习算法，并封装成 Python 包以便测试。
- 提供完备的基础测试。
- 方便自由地创建和运行自定义测试。

## 安装依赖

请使用 `requirements.txt` 文件来配置依赖环境。推荐使用 [pload](https://github.com/HugoPhi/python_venv_loader) 作为虚拟环境管理工具，这是一个轻量级的 Python 虚拟环境管理工具，专注于 Python 环境的管理。具体操作如下：

```bash
pload new -m 'MLLabs' -v 
```

## 使用方法

以下介绍如何运行基准测试、自定义模型和创建运行测试。

### 1. 运行基准测试

首先，为项目创建一个特定的虚拟环境并进入该环境。有关使用 pload 创建虚拟环境的方法，请参阅：[使用 pload 创建虚拟环境](https://github.com/HugoPhi/python_venv_loader)。

1. **克隆仓库：**

    ```bash
    git clone --branch main --single-branch https://github.com/HugoPhi/MachineLearningLabs.git
    ```

2. **安装本地库：**

    进入项目目录：

    ```bash
    cd MachineLearningLabs/
    ```

    编译库：

    ```bash
    pip install .
    ```

    您可以通过运行 `pip list` 命令，检查是否包含 `hym` 库来验证是否安装成功。

3. **运行测试：**

    例如，要运行 `test/DecisionTree/watermelon2.0` 实验，可在项目目录下执行：

    ```bash
    python ./test/DecisionTree/watermelon2.0/main.py
    ```

    即可获得实验结果。

### 2. 自定义模型

您可以修改或实现自己的机器学习模型。本项目的结构主要包括两个部分：`src` 和 `test`。`src` 用于存放机器学习算法的源码，包括算法实现、数据集加载、辅助函数等；`test` 用于存放各算法的基础测试和自定义测试。了解项目结构有助于您更高效、正确地进行修改。

#### 2.1. `src` 目录

`src` 目录用于存放源代码，其结构如下：

```
src/   
├── hym/   
│   ├── __init__.py    
│   ├── DecisionTree/    
│   │   ├── __init__.py    
│   │   ├── DecisionTree.py    
│   │   └── ...    
│   ├── LinearRegression/    
│   └── ...    
```

- **`hym/`**：顶层模块，包含各类机器学习算法的实现。
- 如果要新增算法类别，如支持向量机，请在 `hym/` 下创建 `SupportVectorMachine/` 目录，并在 `hym/__init__.py` 中添加：

    ```python
    from . import DecisionTree
    from . import LinearRegression
    from . import SupportVectorMachine  # 新增算法模块

    __all__ = [
        'DecisionTree',
        'LinearRegression',
        'SupportVectorMachine'  # 添加新模块
    ]
    ```

- **文件命名规范：**

    1. **算法类文件**：使用大驼峰命名法，例如 `BasicDecisionTree.py`、`Variants.py`，用于实现算法类。
    2. **辅助类文件**：使用小写加下划线命名法，例如 `node.py`，用于实现辅助类。
    3. **辅助函数文件**：`utils.py`，用于实现辅助函数库，如数据加载、预处理、数学函数等。
    4. **包初始化文件**：`__init__.py`，用于标记包和子模块，将需要导出的内容添加到 `__all__` 列表中。

#### 2.2. `test` 目录

`test` 目录用于存放测试代码，结构与 `src` 类似：

```
test/    
├── DecisionTree/
│   ├── iris/
│   │   ├── iris.xlsx
│   │   └── main.py
│   ├── watermelon2.0/
│   │   ├── watermelon2.0.xlsx
│   │   └── main.py
│   └── ...
├── LinearRegression/    
└── ... 
```

- 在 `test/` 下按照算法类别创建目录，名称需与 `src/` 中的算法类别对应。
- 在相应的算法目录下创建测试案例。已实现了一些基础数据集的测试，您也可以自行添加实验。

#### 2.3. 其他重要文件

1. **setup.py**

    包含包的构建信息，如版本、依赖库、作者等。版本号遵循以下格式：`v[x].[y].[z]`，其中：

    - `x`：重大更新，API 发生重大变化，不兼容旧版本。
    - `y`：新增较大功能，如实现新算法类。
    - `z`：小更新，修复 Bug，添加小功能或进行小改动。

2. **README.md**

    记录项目的使用方法和最新更新。建议定期查看，以获取最新信息。

## 进度

<details>
<summary>算法库</summary>

- [ ] **监督学习**
  - [ ] 线性回归
  - [x] 逻辑回归
  - [x] 决策树
    - [x] ID3
    - [x] C4.5
    - [ ] CART 
  - [ ] 支持向量机
  - [ ] 神经网络
- [ ] **无监督学习**
  - [ ] K 均值聚类
  - [ ] 主成分分析
     
</details>

<details>
<summary>测试</summary>

- [ ] **监督学习**
  - [ ] 线性回归
  - [x] 逻辑回归
    - [x] iris 
  - [x] 决策树
    - [x] watermelon2.0
    - [ ] iris
    - [ ] ice-cream
    - [ ] wine quality
    - [ ] house price
  - [ ] 支持向量机
  - [ ] 神经网络
- [ ] **无监督学习**
  - [ ] K 均值聚类
  - [ ] 主成分分析

</details>

## 参考文献

请在此处添加您的参考文献。

## 许可证

本项目采用 [MIT 许可证](LICENSE)，详情请参阅 LICENSE 文件。

---

如果您觉得本项目对您有帮助，欢迎 [⭐️ Star](https://github.com/HugoPhi/MachineLearningLabs) 支持我们！

[![GitHub stars](https://img.shields.io/github/stars/HugoPhi/MachineLearningLabs.svg?style=social&label=Star)](https://github.com/HugoPhi/MachineLearningLabs)

---
