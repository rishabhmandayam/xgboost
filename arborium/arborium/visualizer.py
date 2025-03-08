import json
import numpy as np
from IPython.display import display, HTML
# Add sklearn imports
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor, _tree
# Import HTML templates
from . import visualizer_templates as templates

class XGBTreeVisualizer:
    _instance_counter = 0  # Class variable to track instances
    
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
        # Assign a unique ID to this instance
        XGBTreeVisualizer._instance_counter += 1
        self.instance_id = f"xgb_viz_{XGBTreeVisualizer._instance_counter}"
        
        self.model = model
        self.X = X
        self.y = y
        self.feature_names = feature_names
        self.target_names = target_names
        self.simplified_model = None  # To store the simplified decision tree model
        
        if not hasattr(model, 'get_dump'):
            raise ValueError("Model does not have 'get_dump' method. Please provide a valid XGBoost model.")
            
        try:
            self.trees_json = model.get_dump(dump_format='json')
            self.num_trees = len(self.trees_json)
        except Exception as e:
            raise ValueError(f"Failed to get tree dump from model: {str(e)}. Please ensure you're using a valid XGBoost model.")
        
        #Create Histograms
        self.feature_ranges = {}
        self.feature_histograms = {}
        if X is not None:
            X_array = self._ensure_numpy_array(X)
            for i in range(X_array.shape[1]):
                feature_key = f"f{i}"
                self.feature_ranges[feature_key] = (np.min(X_array[:, i]), np.max(X_array[:, i]))
                counts, bins = np.histogram(X_array[:, i], bins=20)
                self.feature_histograms[feature_key] = {"bins": bins, "counts": counts}
        self.task_type = self._detect_task_type()
        self.num_classes = self._get_num_classes()

    def _ensure_numpy_array(self, X):
        """
        Convert input features to numpy array if they're not already.
        Handles pandas DataFrames and other array-like objects.
        """
        if X is None:
            return None
            
        # Check if it's a pandas DataFrame
        if hasattr(X, 'values'):
            return X.values
            
        # Convert to numpy array if it's not already
        return np.array(X)
    
    def parse_tree_json(self, json_str):
        """
        Parse a JSON string representing one tree into a dictionary.
        """
        return json.loads(json_str)
    
    def _generate_svg_for_node(self, feature, split_condition):
        """
        Create an inline SVG showing a mini histogram of the feature's distribution.
        A vertical line is drawn at the split threshold.
        """
        if feature not in self.feature_ranges or feature not in self.feature_histograms:
            return ""
            
        # Skip if X was not provided or feature data is invalid
        if not self.feature_ranges or not self.feature_histograms:
            return ""
            
        min_val, max_val = self.feature_ranges[feature]
        # Skip if min and max are the same (no variation in feature)
        if min_val == max_val:
            return ""
            
        try:
            threshold = float(split_condition)
            norm_threshold = (threshold - min_val) / (max_val - min_val) if max_val != min_val else 0.5
            norm_threshold = max(0, min(1, norm_threshold))
            # Retrieve histogram data computed from X.
            hist_data = self.feature_histograms[feature]
            counts = hist_data["counts"]
            num_bins = len(counts)
            
            # Skip if histogram is empty or invalid
            if num_bins == 0 or np.sum(counts) == 0:
                return ""
                
            # Define SVG dimensions
            svg_width = 100
            svg_height = 30  # Total height of the SVG
            chart_height = 20  # Height available for the histogram bars
            bar_width = svg_width / num_bins
            max_count = np.max(counts) if np.max(counts) > 0 else 1
            
            svg_parts = []
            svg_parts.append(f"<svg width='{svg_width}' height='{svg_height}' style='margin-left:10px;'>")
            # Draw histogram bars
            for i, count in enumerate(counts):
                # Scale bar height relative to chart_height
                bar_height = (count / max_count) * chart_height
                x = i * bar_width
                # Align bars to the bottom of the SVG
                y = svg_height - bar_height
                svg_parts.append(
                    f"<rect x='{x:.1f}' y='{y:.1f}' width='{bar_width:.1f}' height='{bar_height:.1f}' "
                    f"fill='#b3cde0' stroke='#fff'/>"
                )
            # Draw the vertical threshold line across the full height of the SVG
            x_line = norm_threshold * svg_width
            svg_parts.append(
                f"<line x1='{x_line:.1f}' y1='0' x2='{x_line:.1f}' y2='{svg_height}' "
                f"stroke='#ff5722' stroke-width='2'/>"
            )
            svg_parts.append("</svg>")
            return "".join(svg_parts)
        except Exception as e:
            # Silently fail and return empty string on error
            return ""
    
    def _tree_to_html(self, node, is_yes_branch=None):
        """
        Recursively convert a tree node (in dict form) into HTML.
        Internal nodes display the feature split and mini histogram.
        Leaf nodes display the leaf value.
        
        Args:
            node: The tree node to convert
            is_yes_branch: Whether this node is from a 'yes' branch (True), 'no' branch (False), or root (None)
        """
        if node is None:
            # Handle null node case
            return "<li class='node leaf'><div class='node-content'>Empty Node</div></li>"
        
        branch_indicator = ""
        branch_class = ""
        if is_yes_branch is not None:
            branch_class = "yes-branch-node" if is_yes_branch else "no-branch-node"
            branch_label = "Yes (≤)" if is_yes_branch else "No (>)"
            indicator_class = "yes-indicator" if is_yes_branch else "no-indicator"
            branch_indicator = f"<span class='branch-indicator {indicator_class}'>{branch_label}</span>"
            
        if 'leaf' in node:
            leaf_value = node.get('leaf', 'N/A')
            
            # Check if leaf_value is already a string (from simplified tree)
            if isinstance(leaf_value, str):
                display_val = leaf_value
                tooltip = "Prediction from simplified tree model."
            else:
                # For numeric leaf values from XGBoost trees
                try:
                    display_val, tooltip = self._transform_leaf_value(leaf_value)
                except Exception as e:
                    display_val = f"Error: {str(e)}"
                    tooltip = "Error processing leaf value"
            
            return (
                f"<li class='node leaf {branch_class}'>"
                f"<div class='node-content' title='{tooltip}'>"
                f"{branch_indicator}"
                f"<span class='node-type'>Leaf</span>: {display_val} "
                f"</div></li>"
            )
        else:
            try:
                feature = node.get('split', 'unknown')
                if self.feature_names:
                    try:
                        idx = int(feature.replace("f", ""))
                        feature_label = self.feature_names[idx] if idx < len(self.feature_names) else feature
                    except Exception:
                        feature_label = feature
                else:
                    feature_label = feature

                split_condition = node.get('split_condition', '')
                svg_chart = self._generate_svg_for_node(feature, split_condition)
                node_label = (
                    f"<div class='node-content'>"
                    f"{branch_indicator}"
                    f"<span class='node-type'>Split:</span> <span class='feature'>{feature_label}</span> "
                    f"&le; <span class='condition'>{split_condition}</span> {svg_chart} "
                    f"</div>"
                )
                
                # Get the children, ensuring we have both yes and no branches
                children = node.get('children', [])
                yes_child = None
                no_child = None
                
                # In XGBoost tree format, the first child is the 'yes' branch (≤ condition)
                # and the second child is the 'no' branch (> condition)
                if len(children) > 0:
                    yes_child = children[0]
                if len(children) > 1:
                    no_child = children[1]
                    
                # Generate HTML for children
                children_html = ""
                if yes_child:
                    children_html += self._tree_to_html(yes_child, True)
                if no_child:
                    children_html += self._tree_to_html(no_child, False)
                    
                return f"<li class='node internal {branch_class}'>{node_label}<ul>{children_html}</ul></li>"
            except Exception as e:
                # If we have an error generating the node, display it
                error_html = (
                    f"<li class='node error'>"
                    f"<div class='node-content' style='background-color: #ffebee;'>"
                    f"<span class='node-type' style='color: #d32f2f;'>Error:</span> {str(e)}"
                    f"</div></li>"
                )
                return error_html
    
    def _generate_tree_html(self, tree_dict):
        """
        Generate the full HTML representation for a tree given its dictionary.
        """
        html_content = "<ul class='tree'>" + self._tree_to_html(tree_dict) + "</ul>"
        return html_content
    
    def _generate_tree_selector(self):
        """
        Generate HTML for the tree selector input field.
        """
        return templates.get_tree_selector_html(self.num_trees, self.instance_id)
    
    def _generate_all_trees_html(self):
        """
        Generate HTML for all trees, but initially hide all except the first one.
        """
        all_trees_html = ""
        for i in range(self.num_trees):
            try:
                tree_json_str = self.trees_json[i]
                #print(i)
                #print(tree_json_str)
                tree_dict = self.parse_tree_json(tree_json_str)
                tree_html = self._generate_tree_html(tree_dict)

                #print(tree_html)
                
                # Add class information header for multiclass
                class_header = ""
                if self.task_type == 'multiclass_classification':
                    class_idx, class_name, num_classes = self.get_tree_class(i)
                    if class_name:
                        round_num = i // num_classes if num_classes > 0 else 0
                        class_header = templates.get_tree_class_header_html(i, class_name, round_num)
                
                # Add tree info for all tree types
                tree_info = ""
                if self.task_type == 'multiclass_classification':
                    if class_header:
                        tree_info = class_header
                    else:
                        tree_info = templates.get_tree_info_html(i)
                elif self.task_type == 'regression':
                    # For regression, ensure we have a clear header for each tree
                    tree_info = f"""<div class="tree-class-header"><h3>Regression Tree {i}</h3><p style="color: #333333; font-weight: 500;">This tree contributes directly to the predicted value. Final prediction is the sum of all tree outputs.</p></div>"""
                else:
                    tree_info = templates.get_tree_info_html(i)
                
                # Set display based on tree index
                display_style = "block" if i == 0 else "none"

                
                # Create tree container with proper ID and class that includes the instance ID
                all_trees_html += f'''
                <div id="{self.instance_id}_tree-{i}" class="tree-container" style="display: {display_style}; position: relative;">
                    {tree_info}
                    <div class="tree-content">
                        {tree_html}
                    </div>
                </div>
                '''
            except Exception as e:
                # Add error information with instance-specific ID
                display_style = "block" if i == 0 else "none"
                error_html = f'''
                <div id="{self.instance_id}_tree-{i}" class="tree-container" style="display: {display_style}; position: relative;">
                    <div class="tree-info" style="background-color: #ffebee; border-left-color: #f44336;">
                        <h3>Error Loading Tree {i}</h3>
                        <p style="color: #d32f2f;">Failed to generate HTML for this tree: {str(e)}</p>
                    </div>
                </div>
                '''
                all_trees_html += error_html
                print(f"Error generating tree {i}: {str(e)}")  # Output to console for debugging
        
        return all_trees_html
    
    def show_tree(self, tree=0):
        """
        Render the nth tree as interactive HTML with pan and zoom support.
        Also includes a selector to switch between trees.
        """
        if tree < 0 or tree >= self.num_trees:
            raise ValueError("Invalid tree index.")
        
        tree_selector = self._generate_tree_selector()
        all_trees_html = self._generate_all_trees_html()

        # Store variables that will be used in JavaScript
        num_trees = self.num_trees
        max_tree_index = self.num_trees - 1
        instance_id = self.instance_id
        
        # Use template from visualizer_templates.py with instance ID
        html_template = templates.get_tree_html_template(tree_selector, all_trees_html, num_trees, max_tree_index, instance_id)
        display(HTML(html_template))

    def _get_num_classes(self):
        """
        Get the number of classes for multiclass classification models.
        """
        if self.task_type != 'multiclass_classification':
            return 0
            
        # Try to get from model parameters
        try:
            params = self._get_model_params()
            if 'num_class' in params:
                return int(params['num_class'])
        except:
            pass
            
        # Try to infer from target_names
        if self.target_names is not None:
            return len(self.target_names)
            
        # Try to infer from number of trees
        # In multiclass XGBoost, trees are grouped by class
        # If we can't determine, default to 0
        return 0
        
    def get_tree_class(self, tree_idx):
        """
        For multiclass models, determine which class a tree contributes to.
        
        Args:
            tree_idx (int): Index of the tree
            
        Returns:
            tuple: (class_index, class_name, num_classes)
        """
        if self.task_type != 'multiclass_classification' or self.num_classes <= 0:
            return (0, None, 0)
            
        # In multiclass XGBoost, trees are grouped by class
        # Tree i % num_classes corresponds to class i // num_classes
        class_idx = tree_idx % self.num_classes
        
        # Get class name if available
        class_name = None
        if self.target_names is not None and class_idx < len(self.target_names):
            class_name = self.target_names[class_idx]
        else:
            class_name = f"Class {class_idx}"
            
        return (class_idx, class_name, self.num_classes)

    def _get_model_params(self):
        """
        Get parameters from the XGBoost model, handling different model types.
        """
        # Try different methods to get parameters based on model type
        try:
            # For sklearn wrapper
            return self.model.get_xgb_params()
        except:
            try:
                # For base XGBoost model
                return self.model.get_params()
            except:
                try:
                    # For base XGBoost model (alternative)
                    return self.model.attributes()
                except:
                    # If all methods fail, return empty dict
                    return {}

    def _detect_task_type(self):
        """
        Detect if the XGBoost model is for regression, binary classification, or multiclass classification.
        Assumes native XGBoost (xgb.Booster or similar).
        
        Returns:
            str: 'regression', 'binary_classification', or 'multiclass_classification'
        """
        # Get parameters from native XGBoost model
        try:
            params = self._get_model_params()
        except:
            params = {}
        
        # Check objective parameter which specifies the learning task
        objective = params.get('objective', '')
        
        # Determine task type based on objective
        if any(obj in objective for obj in ['binary:logistic', 'binary:logitraw', 'binary:hinge']):
            return 'binary_classification'
        elif any(obj in objective for obj in ['multi:softmax', 'multi:softprob']):
            return 'multiclass_classification'
        elif any(obj in objective for obj in ['reg:linear', 'reg:squarederror', 'reg:logistic', 'reg:pseudohubererror',
                                             'reg:squaredlogerror', 'reg:absoluteerror', 'reg:gamma', 'reg:tweedie']):
            return 'regression'
        
        # If objective doesn't provide enough information, check num_class
        if 'num_class' in params and int(params['num_class']) > 1:
            return 'multiclass_classification'
        
        # If we have target_names, use that as a hint
        if self.target_names is not None:
            if len(self.target_names) > 2:
                return 'multiclass_classification'
            elif len(self.target_names) == 2:
                return 'binary_classification'
        
        # Default to regression if we can't determine
        return 'regression'

    def _transform_leaf_value(self, leaf_value):
        """
        Transform raw leaf value based on the model's task type.
        
        Args:
            leaf_value (float): The raw leaf value from the tree
            
        Returns:
            tuple: (display_value, tooltip_text)
        """
        if self.task_type == 'binary_classification':
            # For binary classification, raw values are log-odds
            display_val = f"Contribution: {leaf_value:.4f}"
            tooltip = "Partial contribution to log-odds score. Final prediction requires summing all tree contributions and applying sigmoid."
            
        elif self.task_type == 'multiclass_classification':
            display_val = f"Contribution: {leaf_value:.4f}"
            tooltip = "Partial contribution to class score. Final prediction requires summing all tree contributions and applying softmax."
            
        else:  # regression
            display_val = f"{leaf_value:.4f}"
            tooltip = "Partial contribution to prediction. Final value requires summing contributions from all trees."
        
        return display_val, tooltip

    def show_simplified_tree(self, max_depth=3, n_components=None, n_samples=10000):
        """
        Fit a simplified sklearn decision tree to the XGBoost model's predictions,
        then visualize it using the same format as the XGBoost trees.
        
        Uses GMM-based sampling to create a synthetic dataset that better represents
        the feature space, then queries the XGBoost model for predictions on this
        synthetic data.
        
        Parameters:
            max_depth (int): Maximum depth of the simplified tree (default: 3)
            n_components (int, optional): Number of components for the GMM.
                Defaults to number of classes for classification or 3 for regression.
            n_samples (int): Number of samples to generate from the GMM (default: 10000)
                
        Returns:
            The simplified decision tree model that can be used for predictions
        """
        if self.X is None or not isinstance(self.X, (np.ndarray, list)) or len(self.X) == 0:
            raise ValueError("Input data X is required to fit a simplified tree")
        
        X_array = self._ensure_numpy_array(self.X)
        
        # Determine number of GMM components if not specified
        if n_components is None:
            if self.task_type == 'multiclass_classification' and self.num_classes > 0:
                n_components = self.num_classes
            elif self.task_type == 'binary_classification':
                n_components = 2
            else:  # regression or fallback
                n_components = 3
        
        # 1. Fit a Gaussian Mixture Model to the input data
        try:
            from sklearn.mixture import GaussianMixture
            gmm = GaussianMixture(n_components=n_components, 
                                 covariance_type='full', 
                                 random_state=42)
            gmm.fit(X_array)
            
            # 2. Sample from the GMM to create synthetic dataset
            X_synthetic, _ = gmm.sample(n_samples)
            
            # 3. Query the XGBoost model on these sampled points
            # Get XGBoost model predictions on the synthetic data
            try:
                is_sklearn_api = False
                
                if hasattr(self.model, 'predict') and not hasattr(self.model, 'get_booster'):
                    try:
                        # Try sklearn API first with proper error handling
                        y_synthetic = self.model.predict(X_synthetic)
                        is_sklearn_api = True
                    except (TypeError, ValueError):
                        # If it fails, we'll try native XGBoost approach below
                        pass
                        
                if not is_sklearn_api:
                    # Try to import XGBoost for native API
                    try:
                        import xgboost as xgb
                    
                        if hasattr(self.model, 'get_booster'):
                            xgb_model = self.model.get_booster()
                        else:
                            xgb_model = self.model
                            
                        dmatrix = xgb.DMatrix(X_synthetic, missing=np.nan)
                        
                        y_synthetic = xgb_model.predict(dmatrix)
                        
                    except ImportError:
                        raise ValueError("XGBoost not found. Please install it to use this feature.")
                    except Exception as e:
                        raise ValueError(f"Failed to get predictions from XGBoost model: {str(e)}")
            except Exception as e:
                raise ValueError(f"Failed to get predictions from XGBoost model: {str(e)}")
            
            # 4. Train a simplified decision tree on the synthetic data
            from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
            
            if self.task_type == 'regression':
                simplified_model = DecisionTreeRegressor(max_depth=max_depth)
                simplified_model.fit(X_synthetic, y_synthetic)
            elif self.task_type == 'binary_classification':
                simplified_model = DecisionTreeClassifier(max_depth=max_depth)
                # For binary classification, y_pred might be probabilities
                if len(y_synthetic.shape) > 1 and y_synthetic.shape[1] > 1:
                    # If we have probability outputs, use the probability of class 1
                    simplified_model.fit(X_synthetic, y_synthetic[:, 1] > 0.5)
                else:
                    simplified_model.fit(X_synthetic, y_synthetic)
            else:  # multiclass_classification
                simplified_model = DecisionTreeClassifier(max_depth=max_depth)
                # For multiclass, y_pred might be probabilities or class indices
                if len(y_synthetic.shape) > 1 and y_synthetic.shape[1] > 1:
                    # If we have probability outputs, use the class with highest probability
                    simplified_model.fit(X_synthetic, np.argmax(y_synthetic, axis=1))
                else:
                    simplified_model.fit(X_synthetic, y_synthetic)
            
            # Store the simplified model for later use
            self.simplified_model = simplified_model
            
            # Convert the sklearn tree to our visualization format
            tree_dict = self._sklearn_tree_to_dict(simplified_model)
            
            # Generate HTML for the simplified tree
            tree_html = self._generate_tree_html(tree_dict)
            
            # Create a header for the simplified tree with GMM information
            tree_header = templates.get_simplified_tree_header_html(
                max_depth=max_depth,
                n_components=n_components,
                n_samples=n_samples
            )
            
            # Use template from visualizer_templates.py
            html_template = templates.get_simplified_tree_html_template(tree_header, tree_html)
            display(HTML(html_template))
            
            # Return the simplified model for direct use
            return self.simplified_model
            
        except ImportError:
            print("Error: scikit-learn is required for Gaussian Mixture Model (GMM) sampling.")
            print("Please install it with: pip install scikit-learn")
            
            # Fall back to the original implementation without GMM
            return self._show_simplified_tree_original(max_depth)
    
    def _show_simplified_tree_original(self, max_depth=3):
        """
        Original implementation of show_simplified_tree without GMM sampling.
        Used as a fallback if sklearn is not available.
        """
        if self.X is None or not isinstance(self.X, (np.ndarray, list)) or len(self.X) == 0:
            raise ValueError("Input data X is required to fit a simplified tree")
        
        X_array = self._ensure_numpy_array(self.X)
        
        # Get XGBoost model predictions
        try:
            is_sklearn_api = False
            
            if hasattr(self.model, 'predict') and not hasattr(self.model, 'get_booster'):
                try:
                    # Try sklearn API first with proper error handling
                    y_pred = self.model.predict(self.X)
                    is_sklearn_api = True
                except (TypeError, ValueError):
                    # If it fails, we'll try native XGBoost approach below
                    pass
                    
            if not is_sklearn_api:
                # Try to import XGBoost for native API
                try:
                    import xgboost as xgb
                
                    if hasattr(self.model, 'get_booster'):
                        xgb_model = self.model.get_booster()
                    else:
                        xgb_model = self.model
                        
                    dmatrix = xgb.DMatrix(X_array, missing=np.nan)
                    
                    y_pred = xgb_model.predict(dmatrix)
                    
                except ImportError:
                    raise ValueError("XGBoost not found. Please install it to use this feature.")
                except Exception as e:
                    raise ValueError(f"Failed to get predictions from XGBoost model: {str(e)}")
        except Exception as e:
            raise ValueError(f"Failed to get predictions from XGBoost model: {str(e)}")
        
        # Choose the appropriate sklearn tree model based on task type
        if self.task_type == 'regression':
            simplified_model = DecisionTreeRegressor(max_depth=max_depth)
            simplified_model.fit(X_array, y_pred)
        elif self.task_type == 'binary_classification':
            simplified_model = DecisionTreeClassifier(max_depth=max_depth)
            # For binary classification, y_pred might be probabilities
            if len(y_pred.shape) > 1 and y_pred.shape[1] > 1:
                # If we have probability outputs, use the probability of class 1
                simplified_model.fit(X_array, y_pred[:, 1] > 0.5)
            else:
                simplified_model.fit(X_array, y_pred)
        else:  # multiclass_classification
            simplified_model = DecisionTreeClassifier(max_depth=max_depth)
            # For multiclass, y_pred might be probabilities or class indices
            if len(y_pred.shape) > 1 and y_pred.shape[1] > 1:
                # If we have probability outputs, use the class with highest probability
                simplified_model.fit(X_array, np.argmax(y_pred, axis=1))
            else:
                simplified_model.fit(X_array, y_pred)
        
        # Store the simplified model for later use
        self.simplified_model = simplified_model
        
        # Convert the sklearn tree to our visualization format
        tree_dict = self._sklearn_tree_to_dict(simplified_model)
        
        # Generate HTML for the simplified tree
        tree_html = self._generate_tree_html(tree_dict)
        
        # Create a header for the simplified tree
        tree_header = templates.get_simplified_tree_header_html(max_depth)
        
        # Use template from visualizer_templates.py
        html_template = templates.get_simplified_tree_html_template(tree_header, tree_html)
        display(HTML(html_template))
        
        # Return the simplified model for direct use
        return self.simplified_model

    def _sklearn_tree_to_dict(self, sklearn_tree):
        """
        Convert a scikit-learn decision tree to a dictionary format
        compatible with our tree visualization.
        
        Parameters:
            sklearn_tree: A trained sklearn.tree.DecisionTreeClassifier or DecisionTreeRegressor
            
        Returns:
            dict: Tree structure in a format compatible with our visualization
        """
        tree = sklearn_tree.tree_
        
        # Function to recursively build the tree
        def build_tree(node_id):
            if tree.children_left[node_id] == _tree.TREE_LEAF:
                # Leaf node
                if hasattr(sklearn_tree, 'classes_'):
                    # For classification trees, show predicted class
                    # Get the majority class
                    value = tree.value[node_id][0]
                    class_idx = np.argmax(value)
                    
                    # Try to get class name from target_names if available
                    if self.target_names is not None and class_idx < len(self.target_names):
                        class_name = self.target_names[class_idx]
                    # Fallback to sklearn's class labels if available
                    elif hasattr(sklearn_tree, 'classes_') and class_idx < len(sklearn_tree.classes_):
                        # If the class is numeric, try to map it to target_names by position
                        if np.issubdtype(type(sklearn_tree.classes_[class_idx]), np.number) and self.target_names:
                            actual_class = int(sklearn_tree.classes_[class_idx])
                            if actual_class < len(self.target_names):
                                class_name = self.target_names[actual_class]
                            else:
                                class_name = f"Class {sklearn_tree.classes_[class_idx]}"
                        else:
                            class_name = f"{sklearn_tree.classes_[class_idx]}"
                    else:
                        class_name = f"Class {class_idx}"
                    
                    # Calculate probability or confidence
                    total = np.sum(value)
                    prob = value[class_idx] / total if total > 0 else 0
                    
                    # Format leaf value as class name with probability
                    leaf_value = f"{class_name} ({prob:.3f})"
                else:
                    # For regression trees, show predicted value
                    leaf_value = float(tree.value[node_id][0][0])
                
                return {'leaf': leaf_value}
            else:
                # Decision node
                feature_idx = tree.feature[node_id]
                threshold = tree.threshold[node_id]
                
                # Get feature name if available
                if self.feature_names and feature_idx < len(self.feature_names):
                    feature_label = self.feature_names[feature_idx]
                    feature = f"f{feature_idx}"
                else:
                    feature = f"f{feature_idx}"
                    feature_label = f"feature_{feature_idx}"
                
                # Create the node dictionary
                node = {
                    'split': feature,
                    'split_condition': float(threshold),
                    'children': [
                        build_tree(tree.children_left[node_id]),   # Yes branch
                        build_tree(tree.children_right[node_id])   # No branch
                    ]
                }
                return node
        
        # Start building the tree from the root (node 0)
        return build_tree(0)

    def get_simplified_model(self):
        """
        Get the simplified decision tree model created by show_simplified_tree.
        
        Returns:
            The simplified decision tree model (sklearn.tree.DecisionTreeClassifier or DecisionTreeRegressor)
            or None if show_simplified_tree has not been called yet.
        """
        if self.simplified_model is None:
            print("No simplified model available. Call show_simplified_tree() first to create one.")
        return self.simplified_model
    
    def predict_with_simplified_tree(self, X):
        """
        Make predictions using the simplified decision tree model.
        
        Parameters:
            X: Features to make predictions on, in the same format as the original data.
               Can be a numpy array, list, or compatible data structure.
        
        Returns:
            Predictions from the simplified model. For classification tasks, can
            return class labels or probabilities depending on the method called.
        
        Raises:
            ValueError: If no simplified model is available or if input format is incorrect.
        """
        if self.simplified_model is None:
            raise ValueError("No simplified model available. Call show_simplified_tree() first to create one.")
        
        X_array = self._ensure_numpy_array(X)
        
        # Make sure the features have the right shape
        n_features = self.simplified_model.n_features_in_
        if X_array.shape[1] != n_features:
            raise ValueError(f"Input has {X_array.shape[1]} features, but the simplified model expects {n_features} features.")
        
        # Return predictions based on the task type
        try:
            if hasattr(self.simplified_model, 'predict_proba') and isinstance(self.simplified_model, DecisionTreeClassifier):
                # For classification tasks, we provide both class predictions and probabilities
                class_predictions = self.simplified_model.predict(X_array)
                class_probs = self.simplified_model.predict_proba(X_array)
                return {
                    'class_predictions': class_predictions,
                    'class_probabilities': class_probs
                }
            else:
                # For regression tasks, just return the predictions
                return self.simplified_model.predict(X_array)
        except Exception as e:
            raise ValueError(f"Error making predictions with the simplified model: {str(e)}")