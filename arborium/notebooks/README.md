# Arborium Example Notebooks

This directory contains Jupyter notebooks demonstrating how to use Arborium for visualizing and interpreting XGBoost models.

## Notebooks

1. **[01_basic_usage.ipynb](01_basic_usage.ipynb)**: Introduction to the basic usage of Arborium with a binary classification example
2. **[02_multiclass_classification.ipynb](02_multiclass_classification.ipynb)**: Demonstrates how to visualize trees in multi-class models
3. **[03_simplified_trees.ipynb](03_simplified_trees.ipynb)**: Shows how to create simplified tree representations of complex models

## Running the Notebooks

### Local Setup

To run these notebooks locally:

1. Make sure you have installed Jupyter:
   ```bash
   pip install jupyter
   ```

2. Install Arborium with XGBoost support:
   ```bash
   pip install arborium[xgboost]
   ```

3. Start Jupyter:
   ```bash
   jupyter notebook
   ```

### Google Colab

You can also run these notebooks in Google Colab by clicking the "Open in Colab" badge in each notebook or by using the following links:

- [01_basic_usage.ipynb in Colab](https://colab.research.google.com/github/rishabhmandayam/xgboost/blob/main/arborium/notebooks/01_basic_usage.ipynb)
- [02_multiclass_classification.ipynb in Colab](https://colab.research.google.com/github/rishabhmandayam/xgboost/blob/main/arborium/notebooks/02_multiclass_classification.ipynb)
- [03_simplified_trees.ipynb in Colab](https://colab.research.google.com/github/rishabhmandayam/xgboost/blob/main/arborium/notebooks/03_simplified_trees.ipynb)

When running in Colab, make sure to uncomment and run the installation cell at the beginning of each notebook:

```python
# !pip install arborium[xgboost]
```

## Notes

- These examples assume you're using a Jupyter environment where HTML output can be rendered.
- The visualizations are interactive within the notebook environment.
- You may need to adjust the paths if you're running these notebooks outside the Arborium repository. 