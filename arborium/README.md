# Arborium

[![PyPI version](https://badge.fury.io/py/arborium.svg)](https://badge.fury.io/py/arborium)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Interactive visualization for tree-based models in Python, with a focus on XGBoost models.

## Table of Contents

- [Introduction](#introduction)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Features](#features)
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

## Installation

### Basic Installation

```bash
pip install arborium
```

### With XGBoost Support

```bash
pip install arborium[xgboost]
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

## Usage Examples

### Basic Tree Visualization

```python
from arborium import XGBTreeVisualizer
import xgboost as xgb
from sklearn.datasets import load_boston
import numpy as np

# Load regression dataset
boston = load_boston()
X, y = boston.data, boston.target
feature_names = boston.feature_names

# Train a regression model
model = xgb.XGBRegressor(n_estimators=100, max_depth=4)
model.fit(X, y)

# Create a visualizer
visualizer = XGBTreeVisualizer(model, X, y, feature_names=feature_names)

# Show a specific tree
visualizer.show_tree()
```

### Working with Multi-Class Models

```python
from arborium import XGBTreeVisualizer
import xgboost as xgb
from sklearn.datasets import load_iris

# Load a multi-class dataset
iris = load_iris()
X, y = iris.data, iris.target
feature_names = iris.feature_names
target_names = iris.target_names

# Train a multi-class model
model = xgb.XGBClassifier(n_estimators=30, max_depth=3)
model.fit(X, y)

# Create a visualizer with target names
visualizer = XGBTreeVisualizer(model, X, y, 
                              feature_names=feature_names,
                              target_names=target_names)


visualizer.show_tree()
```

### Simplified Tree Representations

```python
from arborium import XGBTreeVisualizer
import xgboost as xgb
from sklearn.datasets import fetch_california_housing

# Load a large dataset
housing = fetch_california_housing()
X, y = housing.data, housing.target
feature_names = housing.feature_names

# Train a complex model
model = xgb.XGBRegressor(n_estimators=200, max_depth=8)
model.fit(X, y)

# Create a visualizer
visualizer = XGBTreeVisualizer(model, X, y, feature_names=feature_names)

# Show a simplified representation of the entire model
simplified_tree = visualizer.show_simplified_tree(
    max_depth=3,              # Control the depth of the simplified tree
    n_components=None,        # Use all features (no dimensionality reduction)
    n_samples=5000            # Use 5000 samples to build the simplified model
)

# Use the simplified model for predictions
test_sample = X[0:5]
predictions = visualizer.predict_with_simplified_tree(test_sample)
print(f"Simplified tree predictions: {predictions}")
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