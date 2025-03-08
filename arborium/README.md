# Arborium

[![PyPI version](https://img.shields.io/pypi/v/arborium.svg)](https://pypi.org/project/arborium/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Jupyter Compatible](https://img.shields.io/badge/Jupyter-Compatible-orange.svg)](https://jupyter.org)

Interactive visualization for tree-based models in Python, with a focus on XGBoost models. **Designed for use in Jupyter notebooks and similar interactive environments.**

## Table of Contents

- [Introduction](#introduction)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Features](#features)
- [Example Notebooks](#example-notebooks)
- [Usage Examples](#usage-examples)
  - [Basic Tree Visualization](#basic-tree-visualization)
  - [Working with Multi-Class Models](#working-with-multi-class-models)
  - [Simplified Tree Representations](#simplified-tree-representations)
  - [Feature Importance Visualization](#feature-importance-visualization)
- [API Reference](#api-reference)
- [Contributing](#contributing)
- [License](#license)

## Introduction

Arborium is a Python package designed to make tree-based models more interpretable through advanced visualization techniques. While tree-based models like XGBoost are powerful predictive tools, understanding how they make decisions can be challenging due to their complexity. Arborium addresses this by providing interactive, intuitive visualizations of tree structures, making it easier for data scientists and machine learning practitioners to gain insights into model behavior.

The package currently focuses on XGBoost models but plans to expand support for other tree-based algorithms in future releases.

> **Note:** Arborium is specifically designed for use in Jupyter notebooks or similar interactive environments (JupyterLab, Google Colab, etc.) where HTML visualizations can be rendered inline.

## Installation

### Basic Installation

```bash
pip install arborium
```

### With XGBoost Support

```bash
pip install arborium[xgboost]
```

### Jupyter Notebook Support

Arborium requires an environment that can render HTML and JavaScript. To get the full interactive experience:

```bash
# If you don't already have Jupyter installed
pip install jupyter

# Then launch Jupyter Notebook
jupyter notebook
```

### Development Installation

```bash
git clone https://github.com/yourusername/arborium.git
cd arborium
pip install -e ".[dev]"
```

## Quick Start

```python
import xgboost as xgb
from arborium import XGBTreeVisualizer
import numpy as np
from sklearn.datasets import load_breast_cancer

# Load a dataset
data = load_breast_cancer()
X, y = data.data, data.target
feature_names = data.feature_names

# Train a simple XGBoost model
model = xgb.XGBClassifier(n_estimators=10, max_depth=3)
model.fit(X, y)

# Visualize the trees
visualizer = XGBTreeVisualizer(model, X, y, feature_names=feature_names)
visualizer.show_tree()
```

## Features

Arborium offers the following key features:

- **Interactive Tree Visualization**: Explore tree structures with an intuitive, interactive interface
- **Split Point Analysis**: Visualize feature distributions at split points with histograms
- **Multi-Tree Navigation**: Easily navigate between trees in ensemble models
- **Simplified Tree Creation**: Generate simplified decision trees that approximate complex models
- **Classification & Regression Support**: Works with both classification and regression models
- **Customizable Visualizations**: Control depth, components, and styling of visualizations
- **Jupyter Integration**: Seamless display in Jupyter notebooks and lab environments
- **Model Insights**: Gain interpretability without sacrificing model performance

## Example Notebooks

For interactive examples, explore our Jupyter notebooks:

- [Basic Usage](https://github.com/rishabhmandayam/xgboost/blob/main/arborium/notebooks/01_basic_usage.ipynb) - Introduction to tree visualization
- [Multiclass Classification](https://github.com/rishabhmandayam/xgboost/blob/main/arborium/notebooks/02_multiclass_classification.ipynb) - Visualizing trees in multiclass models
- [Simplified Trees](https://github.com/rishabhmandayam/xgboost/blob/main/arborium/notebooks/03_simplified_trees.ipynb) - Creating and using simplified tree representations

You can run these notebooks locally after installing arborium:

```bash
git clone https://github.com/rishabhmandayam/xgboost.git
cd xgboost/arborium
pip install -e .
jupyter notebook notebooks/
```

Or open directly in Google Colab:

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/rishabhmandayam/xgboost/blob/main/arborium/notebooks/01_basic_usage.ipynb)

## Usage Examples

### Multiclass Classification 

```python
from arborium import XGBTreeVisualizer
from sklearn.datasets import load_iris
import xgboost as xgb

# Load regression dataset
iris = load_iris()
X, y = iris.data, iris.target

# Create DMatrix for XGBoost
dtrain = xgb.DMatrix(X, label=y)

# Set parameters for XGBoost
params = {
    'objective': 'multi:softmax',  # multiclass classification
    'num_class': 3,  # iris has 3 classes
    'max_depth': None,
    'learning_rate': 0.1,
    'eval_metric': 'mlogloss'
}

# Train XGBoost model
num_rounds = 100
model = xgb.train(params, dtrain, num_rounds)

# Create a visualizer
visualizer = XGBTreeVisualizer(model, X, y, feature_names=iris.feature_names, target_names=iris.target_names)

# Show the trees
visualizer.show_tree()
```

### Regression

```python
from arborium import XGBTreeVisualizer
import numpy as np
import xgboost as xgb
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# Load a regression dataset (California Housing)
housing = fetch_california_housing()
X, y = housing.data, housing.target

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create DMatrix for XGBoost
dtrain_reg = xgb.DMatrix(X_train, label=y_train)
dtest_reg = xgb.DMatrix(X_test, label=y_test)

# Set parameters for regression
params_reg = {
    'objective': 'reg:squarederror',
    'max_depth': 4,
    'learning_rate': 0.1,
    'eval_metric': 'rmse'
}

# Train the regression model
num_rounds = 50
reg_model = xgb.train(params_reg, dtrain_reg, num_rounds)

# Evaluate the model
y_pred = reg_model.predict(dtest_reg)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
print(f"Regression model RMSE: {rmse:.4f}")

# Create a visualizer for the regression model
reg_vizualizer = XGBTreeVisualizer(reg_model, X_train, y_train, feature_names=housing.feature_names)

visualizer.show_tree()
```

### Simplified Tree Representations

```python
from arborium import XGBTreeVisualizer
from sklearn.datasets import load_iris
import xgboost as xgb

# Load regression dataset
iris = load_iris()
X, y = iris.data, iris.target

# Create DMatrix for XGBoost
dtrain = xgb.DMatrix(X, label=y)

# Set parameters for XGBoost
params = {
    'objective': 'multi:softmax',  # multiclass classification
    'num_class': 3,  # iris has 3 classes
    'max_depth': None,
    'learning_rate': 0.1,
    'eval_metric': 'mlogloss'
}

# Train XGBoost model
num_rounds = 100
model = xgb.train(params, dtrain, num_rounds)

visualizer = XGBTreeVisualizer(model, X, y, feature_names=iris.feature_names, target_names=iris.target_names)

simple_model = visualizer.show_simplified_tree(max_depth=3)

simple_predictions = simple_model.predict(X_test)
```

### Feature Importance Visualization

Coming in a future release.

## API Reference

### XGBTreeVisualizer

The main class for visualizing XGBoost models.

```python
XGBTreeVisualizer(model, X, y, feature_names=None, target_names=None)
```

**Parameters:**

- `model`: A trained XGBoost model (booster or sklearn API)
- `X`: Input features used during training (array-like or DataFrame)
- `y`: Target values (array-like or Series)
- `feature_names`: List of feature names (optional)
- `target_names`: List of target class names (optional)

**Methods:**

- `show_tree()`: Display an interactive visualization of the tree
- `show_simplified_tree(max_depth=3, n_components=None, n_samples=10000)`: Create and display a simplified decision tree that approximates the full model
- `get_simplified_model()`: Get the simplified decision tree model object
- `predict_with_simplified_tree(X)`: Make predictions using the simplified model

## Contributing

We welcome contributions to Arborium! If you'd like to contribute, please:

1. Fork the repository
2. Create a feature branch
3. Add your changes
4. Run the tests
5. Submit a pull request

For major changes, please open an issue first to discuss the proposed changes.

## License

Arborium is released under the MIT License. See [LICENSE](LICENSE) for details. 