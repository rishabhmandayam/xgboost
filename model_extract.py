import numpy as np
import xgboost as xgb
from sklearn.datasets import load_iris
from sklearn.mixture import GaussianMixture
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

# -----------------------------
# 1. Train the XGBoost Blackbox
# -----------------------------
iris = load_iris()
X, y = iris.data, iris.target

# Create DMatrix for XGBoost and train the model
dtrain = xgb.DMatrix(X, label=y)
params = {
    'objective': 'multi:softmax',  # multiclass classification
    'num_class': 3,
    'max_depth': 3,  # modest depth for illustration
    'learning_rate': 0.1,
    'eval_metric': 'mlogloss'
}
num_rounds = 100
xgb_model = xgb.train(params, dtrain, num_rounds)

def blackbox_predict(x):
    dmatrix = xgb.DMatrix(x)
    return xgb_model.predict(dmatrix)

# -----------------------------
# 2. Fit a Gaussian Mixture Model
# -----------------------------
# For the Iris dataset, we choose 3 components (one per class) as a heuristic.
gmm = GaussianMixture(n_components=3, covariance_type='diag', random_state=42)
gmm.fit(X)

# -----------------------------
# 3. Sample an Augmented Dataset
# -----------------------------
# Sample a large number of points from the GMM.
n_samples = 10000
X_augmented, _ = gmm.sample(n_samples)

# Query the blackbox model on these sampled points.
y_augmented = blackbox_predict(X_augmented)

# -----------------------------
# 4. Train a Sklearn Decision Tree on the Augmented Data
# -----------------------------
clf = DecisionTreeClassifier(max_depth=5, random_state=42)
clf.fit(X_augmented, y_augmented)

# -----------------------------
# 5. Evaluate the Fidelity of the Extracted Tree
# -----------------------------
# Compare predictions on the original training set.
y_xgb = blackbox_predict(X)
y_tree = clf.predict(X)
acc = accuracy_score(y_xgb, y_tree)
print("Fidelity (accuracy) of decision tree surrogate:", acc)