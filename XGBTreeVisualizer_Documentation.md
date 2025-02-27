# XGBTreeVisualizer Documentation

This document provides a detailed explanation of the `XGBTreeVisualizer` class implemented in `Visualizer.py`, which creates interactive visualizations of XGBoost decision trees.

## Table of Contents

1. [Introduction](#introduction)
2. [Class Initialization](#class-initialization)
3. [Data Preparation Methods](#data-preparation-methods)
4. [HTML Generation Methods](#html-generation-methods)
5. [Tree Rendering Methods](#tree-rendering-methods)
6. [Simplified Tree Methods](#simplified-tree-methods)
7. [Helper Methods](#helper-methods)
8. [Data Structures](#data-structures)

## Introduction

`XGBTreeVisualizer` is a Python class that creates interactive HTML visualizations of XGBoost decision trees, allowing users to explore and understand the structure of their trained models. It supports:

- Visualizing individual trees in an XGBoost ensemble
- Switching between trees using a selector
- Panning and zooming to explore large trees
- Fitting simplified decision trees to the ensemble's predictions
- Showing feature histograms at each split

## Class Initialization

### `__init__` Method

```python
def __init__(self, model, X, y, feature_names=None, target_names=None):
    """
    Initialize the visualizer.
    
    Parameters:
        model : Trained XGBoost booster (e.g., from xgb.train or XGBClassifier.get_booster())
        X     : Input features used during training (for context)
        y     : Target values (for context)
        feature_names : List of feature names. If provided, will be used to label splits.
        target_names  : List of target names (for classification).
    """
```

The initializer sets up the visualizer with a trained XGBoost model and the training data. It performs the following steps:

1. Stores model, feature data, and metadata.
2. Checks that the model is a valid XGBoost model with the required methods.
3. Extracts the tree information from the model using `get_dump(dump_format='json')`.
4. Creates feature histograms from the input data.
5. Detects the task type (regression, binary classification, or multiclass classification).
6. Determines the number of classes for multiclass problems.

**Example usage:**
```python
import xgboost as xgb
from sklearn.datasets import load_boston
from Visualizer import XGBTreeVisualizer

# Load data and train a model
data = load_boston()
X, y = data.data, data.target
feature_names = data.feature_names

# Train a model
model = xgb.XGBRegressor(n_estimators=10)
model.fit(X, y)

# Create visualizer
viz = XGBTreeVisualizer(model, X, y, feature_names=feature_names)

# Show the first tree
viz.show_tree(0)
```

## Data Preparation Methods

### `_ensure_numpy_array` Method

```python
def _ensure_numpy_array(self, X):
    """
    Convert input features to numpy array if they're not already.
    Handles pandas DataFrames and other array-like objects.
    """
```

This helper method converts various input data types (including pandas DataFrames) to numpy arrays, which are used for the internal computations.

### `parse_tree_json` Method

```python
def parse_tree_json(self, json_str):
    """
    Parse a JSON string representing one tree into a dictionary.
    """
```

Converts the JSON string representation of a tree (obtained from XGBoost's `get_dump`) into a Python dictionary that can be processed by the visualizer.

**Example input:**
```json
{
  "nodeid": 0,
  "split": "f2",
  "split_condition": 2.5,
  "yes": 1,
  "no": 2,
  "missing": 1,
  "children": [
    {
      "nodeid": 1,
      "leaf": 0.0344
    },
    {
      "nodeid": 2,
      "split": "f0",
      "split_condition": 1.5,
      "yes": 3,
      "no": 4,
      "missing": 3,
      "children": [
        {
          "nodeid": 3,
          "leaf": -0.0229
        },
        {
          "nodeid": 4,
          "leaf": 0.125
        }
      ]
    }
  ]
}
```

**Example output (parsed dictionary):**
```python
{
  'nodeid': 0,
  'split': 'f2',
  'split_condition': 2.5,
  'yes': 1,
  'no': 2,
  'missing': 1,
  'children': [
    {'nodeid': 1, 'leaf': 0.0344},
    {
      'nodeid': 2,
      'split': 'f0',
      'split_condition': 1.5,
      'yes': 3,
      'no': 4,
      'missing': 3,
      'children': [
        {'nodeid': 3, 'leaf': -0.0229},
        {'nodeid': 4, 'leaf': 0.125}
      ]
    }
  ]
}
```

### `_generate_svg_for_node` Method

```python
def _generate_svg_for_node(self, feature, split_condition):
    """
    Create an inline SVG showing a mini histogram of the feature's distribution.
    A vertical line is drawn at the split threshold.
    """
```

This method generates an SVG histogram visualization for a feature at a split node. The histogram shows the distribution of the feature values in the training data, with a vertical line indicating the split threshold.

## HTML Generation Methods

### `_tree_to_html` Method

```python
def _tree_to_html(self, node, is_yes_branch=None):
    """
    Recursively convert a tree node (in dict form) into HTML.
    Internal nodes display the feature split and mini histogram.
    Leaf nodes display the leaf value.
    
    Args:
        node: The tree node to convert
        is_yes_branch: Whether this node is from a 'yes' branch (True), 'no' branch (False), or root (None)
    """
```

This is a recursive method that converts a node in the tree dictionary into HTML. It handles:

- Leaf nodes (displaying the predicted value)
- Split nodes (displaying the feature, split condition, and histogram)
- Branches (indicating "Yes" and "No" paths in the tree)

The method includes error handling to gracefully display issues in the tree structure.

### `_generate_tree_html` Method

```python
def _generate_tree_html(self, tree_dict):
    """
    Generate the full HTML representation for a tree given its dictionary.
    """
```

This method wraps the tree HTML generated by `_tree_to_html` in the appropriate container tags.

### `_generate_tree_selector` Method

```python
def _generate_tree_selector(self):
    """
    Generate HTML for the tree selector input field.
    """
```

Creates the HTML for the tree selection UI element, which allows users to select different trees in the ensemble to visualize.

### `_generate_all_trees_html` Method

```python
def _generate_all_trees_html(self):
    """
    Generate HTML for all trees, but initially hide all except the first one.
    """
```

This method generates HTML for all trees in the ensemble. Each tree is wrapped in a container div with a unique ID, and all trees except the first one are initially hidden. This enables the tree switching functionality.

## Tree Rendering Methods

### `show_tree` Method

```python
def show_tree(self, tree=0):
    """
    Render the nth tree as interactive HTML with pan and zoom support.
    Also includes a selector to switch between trees.
    """
```

This is the main method users call to visualize a tree from the ensemble. It:

1. Validates the tree index
2. Generates the tree selector
3. Generates HTML for all trees
4. Constructs the final HTML template using `templates.get_tree_html_template`
5. Displays the HTML using IPython's `display(HTML(...))`

The resulting visualization is interactive, allowing users to:
- Pan and zoom to explore the tree
- Switch between different trees using the selector
- Use arrow keys for navigation

## Simplified Tree Methods

### `show_simplified_tree` Method

```python
def show_simplified_tree(self, max_depth=3):
    """
    Fit a simplified sklearn decision tree to the XGBoost model's predictions,
    then visualize it using the same format as the XGBoost trees.
    
    Parameters:
        max_depth (int): Maximum depth of the simplified tree (default: 3)
    """
```

This method fits a simplified decision tree (using scikit-learn) to the predictions of the XGBoost model, then visualizes it. This can be helpful for understanding the overall behavior of the ensemble with a simpler model.

The method:
1. Gets predictions from the XGBoost model for the input data
2. Creates an appropriate sklearn tree model (regressor or classifier)
3. Fits the simplified model to the XGBoost predictions
4. Converts the sklearn tree to the visualization format
5. Generates and displays the HTML visualization

### `_sklearn_tree_to_dict` Method

```python
def _sklearn_tree_to_dict(self, sklearn_tree):
    """
    Convert a scikit-learn decision tree to a dictionary format
    compatible with our tree visualization.
    
    Parameters:
        sklearn_tree: A trained sklearn.tree.DecisionTreeClassifier or DecisionTreeRegressor
        
    Returns:
        dict: Tree structure in a format compatible with our visualization
    """
```

This method converts a scikit-learn decision tree structure into the dictionary format needed for visualization with XGBTreeVisualizer. The conversion process involves:

1. **Accessing the underlying tree structure**: It first accesses the underlying tree structure from the sklearn model through `sklearn_tree.tree_`, which contains the arrays representing the decision tree.

2. **Recursive Tree Building**: It defines an inner function `build_tree(node_id)` that recursively traverses the sklearn tree structure starting from the root node (node 0).

3. **Extracting Tree Components**: For each node in the tree, it extracts:
   - **For leaf nodes**: 
     - For classification trees: The predicted class, class probability, and class name
     - For regression trees: The predicted value
   - **For decision nodes**:
     - Feature index used for the split
     - Threshold value for the decision
     - Links to left child (samples meeting the condition) and right child (samples not meeting the condition)

4. **Handling Feature Names**: If feature names were provided during initialization, they are used to label the splits in the tree. Otherwise, generic feature labels like "f0", "f1", etc. are used.

5. **Class Name Resolution**: For classification trees, it attempts to resolve class names using:
   - Target names provided during initialization
   - Classes from the sklearn model
   - Fallback to generic class labels

The sklearn tree structure is accessed through several attributes:
- `tree.children_left[node_id]`: Index of the left child for node_id
- `tree.children_right[node_id]`: Index of the right child for node_id
- `tree.feature[node_id]`: Feature index used for splitting at node_id
- `tree.threshold[node_id]`: Threshold value for the split at node_id
- `tree.value[node_id]`: Contains the prediction value or class distribution at node_id

The method checks for leaf nodes by comparing with `_tree.TREE_LEAF` constant from sklearn.

**Example input (sklearn tree structure):**
```
sklearn_tree.tree_:
- children_left: [1, -1, 3, -1, -1]  # -1 indicates leaf node
- children_right: [2, -1, 4, -1, -1]
- feature: [0, -2, 2, -2, -2]  # -2 indicates no feature (at leaf)
- threshold: [0.5, -2.0, 1.5, -2.0, -2.0]
- value: [[[5.0]], [[3.0]], [[7.0]], [[6.0]], [[8.0]]]
```

**Example output (converted dictionary):**
```python
{
  'split': 'f0',  # Feature index 0
  'split_condition': 0.5,  # Threshold
  'children': [
    {'leaf': 3.0},  # Left child (yes branch)
    {
      'split': 'f2',  # Feature index 2
      'split_condition': 1.5,
      'children': [
        {'leaf': 6.0},  # Left child of inner node
        {'leaf': 8.0}   # Right child of inner node
      ]
    }
  ]
}
```

This converted dictionary structure maintains the hierarchical nature of the decision tree while formatting it to be compatible with the XGBTreeVisualizer's rendering code.

## Helper Methods

### `_get_num_classes` Method

```python
def _get_num_classes(self):
    """
    Get the number of classes for multiclass classification models.
    """
```

Determines the number of classes for multiclass classification models. It attempts to get this information from:
1. Model parameters ('num_class')
2. Target names
3. Inferring from the number of trees

### `get_tree_class` Method

```python
def get_tree_class(self, tree_idx):
    """
    For multiclass models, determine which class a tree contributes to.
    
    Args:
        tree_idx (int): Index of the tree
        
    Returns:
        tuple: (class_index, class_name, num_classes)
    """
```

For multiclass models, this method determines which class a specific tree contributes to. In XGBoost's multiclass implementation, trees are grouped by class.

### `_get_model_params` Method

```python
def _get_model_params(self):
    """
    Get parameters from the XGBoost model, handling different model types.
    """
```

Extracts parameters from the XGBoost model, handling different model interfaces (sklearn wrapper vs. native XGBoost).

### `_detect_task_type` Method

```python
def _detect_task_type(self):
    """
    Detect if the XGBoost model is for regression, binary classification, or multiclass classification.
    Assumes native XGBoost (xgb.Booster or similar).
    
    Returns:
        str: 'regression', 'binary_classification', or 'multiclass_classification'
    """
```

Detects the task type (regression, binary classification, or multiclass classification) based on the model parameters and structure.

### `_transform_leaf_value` Method

```python
def _transform_leaf_value(self, leaf_value):
    """
    Transform raw leaf value based on the model's task type.
    
    Args:
        leaf_value (float): The raw leaf value from the tree
        
    Returns:
        tuple: (display_value, tooltip_text)
    """
```

Transforms the raw leaf value from the tree into a human-readable format, with appropriate context based on the task type. For example, in binary classification, leaf values are log-odds contributions.

## Data Structures

### XGBoost Tree JSON Structure

The XGBoost model provides trees in JSON format, which includes:

- Internal nodes with:
  - `nodeid`: Unique identifier for the node
  - `split`: Feature used for splitting (e.g., "f2" for feature index 2)
  - `split_condition`: Threshold value for the split
  - `yes`: Index of the child node for samples meeting the condition
  - `no`: Index of the child node for samples not meeting the condition
  - `missing`: Index of the child node for samples with missing values
  - `children`: Array of child nodes

- Leaf nodes with:
  - `nodeid`: Unique identifier for the node
  - `leaf`: The predicted value or contribution

Example of a small tree in XGBoost JSON format:

```json
{
  "nodeid": 0,
  "split": "f2",
  "split_condition": 2.5,
  "yes": 1,
  "no": 2,
  "missing": 1,
  "children": [
    {
      "nodeid": 1,
      "leaf": 0.0344
    },
    {
      "nodeid": 2,
      "split": "f0",
      "split_condition": 1.5,
      "yes": 3,
      "no": 4,
      "missing": 3,
      "children": [
        {
          "nodeid": 3,
          "leaf": -0.0229
        },
        {
          "nodeid": 4,
          "leaf": 0.125
        }
      ]
    }
  ]
}
```

### Simplified Tree Dictionary Structure

The simplified tree format (converted from sklearn) follows a similar structure but with fewer attributes:

- Internal nodes with:
  - `split`: Feature used for splitting 
  - `split_condition`: Threshold value for the split
  - `children`: Array of child nodes

- Leaf nodes with:
  - `leaf`: The predicted value (or class with probability for classification)

Example of a simplified tree structure:

```python
{
  'split': 'f5',
  'split_condition': 6.941,
  'children': [
    {
      'split': 'f12',
      'split_condition': 9.725,
      'children': [
        {'leaf': 24.767},
        {'leaf': 50.353}
      ]
    },
    {
      'split': 'f12',
      'split_condition': 5.149,
      'children': [
        {'leaf': 45.223},
        {'leaf': 22.905}
      ]
    }
  ]
}
```

### Feature Histograms

The feature histograms are stored as dictionaries with:
- `bins`: Array of bin edges
- `counts`: Array of counts for each bin

These are used to generate the SVG histograms displayed at each split node.

For each feature, the visualizer also stores the min/max range in `feature_ranges`. 