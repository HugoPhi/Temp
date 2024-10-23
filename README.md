# Machine Learning Experiments

[![GitHub stars](https://img.shields.io/github/stars/HugoPhi/MachineLearningLabs.svg?style=social)](https://github.com/HugoPhi/MachineLearningLabs/stargazers)
[![GitHub forks](https://img.shields.io/github/forks/HugoPhi/MachineLearningLabs.svg?style=social)](https://github.com/HugoPhi/MachineLearningLabs/network/members)
[![GitHub license](https://img.shields.io/github/license/HugoPhi/MachineLearningLabs.svg)](https://github.com/HugoPhi/MachineLearningLabs/blob/main/LICENSE)
[![GitHub issues](https://img.shields.io/github/issues/HugoPhi/MachineLearningLabs.svg)](https://github.com/HugoPhi/MachineLearningLabs/issues)

[中文](README_zh.md)

---

This repository contains machine learning models implemented from scratch using `numpy`, `pandas`, and `matplotlib`, aiming to help learners understand the internal workings of various machine learning algorithms. If you find any errors during use, please raise them in the [Issues](https://github.com/HugoPhi/MachineLearningLabs/issues) section or contribute via a Pull Request to improve this project.

## Features

- Manually implement machine learning algorithms and package them into Python modules for testing.
- Provide comprehensive basic tests.
- Easily create and run custom tests.

## Installation Dependencies

Please use the `requirements.txt` file to configure the dependency environment. It is recommended to use [pload](https://github.com/HugoPhi/python_venv_loader) as a virtual environment management tool. This is a lightweight Python virtual environment management tool that focuses on Python environment management. The specific operation is as follows:

```bash
pload new -m 'MLLabs' -v 
```

## Usage

The following introduces how to run benchmark tests, customize models, and create and run tests.

### 1. Run Benchmark Tests

First, create a specific virtual environment for the project and enter that environment. For information on how to create a virtual environment using pload, please refer to: [Using pload to Create a Virtual Environment](https://github.com/HugoPhi/python_venv_loader).

1. **Clone the repository:**

    ```bash
    git clone --branch main --single-branch https://github.com/HugoPhi/MachineLearningLabs.git
    ```

2. **Install the local library:**

    Enter the project directory:

    ```bash
    cd MachineLearningLabs/
    ```

    Build the library:

    ```bash
    pip install .
    ```

    You can verify the installation by running `pip list` to check if the `hym` library is included.

3. **Run the test:**

    For example, to run the `test/DecisionTree/watermelon2.0` experiment, execute the following in the project directory:

    ```bash
    python ./test/DecisionTree/watermelon2.0/main.py
    ```

    You will obtain the experimental results.

### 2. Customize Models

You can modify or implement your own machine learning models. The project structure mainly includes two parts: `src` and `test`. `src` is used to store the source code of machine learning algorithms, including algorithm implementation, data loading, auxiliary functions, etc.; `test` is used to store basic tests and custom tests for each algorithm. Understanding the project structure will help you make more efficient and accurate modifications.

#### 2.1. `src` Directory

The `src` directory is used to store source code, and its structure is as follows:

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

- **`hym/`**: Top-level module containing implementations of various machine learning algorithms.
- If you want to add a new algorithm category, such as Support Vector Machine, create a `SupportVectorMachine/` directory under `hym/` and add the following in `hym/__init__.py`:

    ```python
    from . import DecisionTree
    from . import LinearRegression
    from . import SupportVectorMachine  # Add new algorithm module

    __all__ = [
        'DecisionTree',
        'LinearRegression',
        'SupportVectorMachine'  # Add new module
    ]
    ```

- **File Naming Conventions:**

    1. **Algorithm Class Files**: Use PascalCase naming, e.g., `BasicDecisionTree.py`, `Variants.py`, used to implement algorithm classes.
    2. **Auxiliary Class Files**: Use lowercase with underscores, e.g., `node.py`, used to implement auxiliary classes.
    3. **Auxiliary Function Files**: `utils.py`, used to implement auxiliary function libraries such as data loading, preprocessing, and mathematical functions.
    4. **Package Initialization Files**: `__init__.py`, used to mark packages and submodules. Add the content to be exported to the `__all__` list.

#### 2.2. `test` Directory

The `test` directory is used to store test code, and its structure is similar to `src`:

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

- Under `test/`, create directories according to algorithm categories; the names should correspond to the algorithm categories in `src/`.
- Create test cases in the corresponding algorithm directories. Some basic dataset tests have been implemented; you can also add your own experiments.

#### 2.3. Other Important Files

1. **setup.py**

    Contains package build information such as version, dependencies, and author. The version number follows the format: `v[x].[y].[z]`, where:

    - `x`: Major updates with significant API changes that are not backward compatible.
    - `y`: Addition of significant new features, such as implementing new algorithm classes.
    - `z`: Minor updates, bug fixes, small feature additions, or minor changes.

2. **README.md**

    Records the usage of the project and the latest updates. It is recommended to check regularly for the latest information.

## Progress

<details>
<summary>Algorithm Library</summary>

- [ ] **Supervised Learning**
  - [ ] Linear Regression
  - [x] Logistic Regression
  - [x] Decision Tree
    - [x] ID3
    - [x] C4.5
    - [ ] CART
  - [ ] Support Vector Machine
  - [ ] Neural Network
- [ ] **Unsupervised Learning**
  - [ ] K-Means Clustering
  - [ ] Principal Component Analysis

</details>

<details>
<summary>Tests</summary>

- [ ] **Supervised Learning**
  - [ ] Linear Regression
  - [x] Logistic Regression
    - [x] iris 
  - [x] Decision Tree
    - [x] watermelon2.0
    - [ ] iris
    - [ ] ice-cream
    - [ ] wine quality
    - [ ] house price
  - [ ] Support Vector Machine
  - [ ] Neural Network
- [ ] **Unsupervised Learning**
  - [ ] K-Means Clustering
  - [ ] Principal Component Analysis

</details>

## References

Please add your references here.

## License

This project is licensed under the [MIT License](LICENSE). For details, please refer to the LICENSE file.

---

If you find this project helpful, please [⭐️ Star](https://github.com/HugoPhi/MachineLearningLabs) to support us!

[![GitHub stars](https://img.shields.io/github/stars/HugoPhi/MachineLearningLabs.svg?style=social&label=Star)](https://github.com/HugoPhi/MachineLearningLabs)

---
