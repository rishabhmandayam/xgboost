# Import necessary libraries
import numpy as np
import xgboost as xgb
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from Visualizer import XGBTreeVisualizer

# Load a regression dataset (California Housing)
housing = fetch_california_housing()
X, y = housing.data, housing.target
print(f"Dataset shape: {X.shape}, Features: {housing.feature_names}")

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create DMatrix for XGBoost
dtrain_reg = xgb.DMatrix(X_train, label=y_train)
dtest_reg = xgb.DMatrix(X_test, label=y_test)

# Set parameters for regression
params_reg = {
    'objective': 'reg:squarederror',
    'max_depth': 3,
    'learning_rate': 0.1,
    'eval_metric': 'rmse',
    'seed': 42
}

# Train the regression model
print("Training XGBoost regression model...")
num_rounds = 50
reg_model = xgb.train(params_reg, dtrain_reg, num_rounds)

# Evaluate the model
y_pred = reg_model.predict(dtest_reg)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
print(f"Regression model RMSE: {rmse:.4f}")

# Create a visualizer for the regression model
print("Creating visualization for the regression model...")
reg_viz = XGBTreeVisualizer(reg_model, X_train, y_train, feature_names=housing.feature_names)

# Show the first tree
print("Displaying the first tree...")
reg_viz.show_tree(0)  # Display first tree

# Optionally show a simplified version of the decision tree
print("Displaying a simplified tree representation...")
reg_viz.show_simplified_tree(max_depth=3)

print("Visualization completed!") 