import json
import numpy as np
from IPython.display import display, HTML

class XGBTreeVisualizer:
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
        self.model = model
        self.X = X
        self.y = y
        self.feature_names = feature_names
        self.target_names = target_names
        
        # Check if model has the required methods
        if not hasattr(model, 'get_dump'):
            raise ValueError("Model does not have 'get_dump' method. Please provide a valid XGBoost model.")
            
        try:
            self.trees_json = model.get_dump(dump_format='json')
            self.num_trees = len(self.trees_json)
        except Exception as e:
            raise ValueError(f"Failed to get tree dump from model: {str(e)}. Please ensure you're using a valid XGBoost model.")
            
        self.feature_ranges = {}
        self.feature_histograms = {}
        if X is not None:
            # Convert X to numpy array if it's not already
            X_array = self._ensure_numpy_array(X)
            # Compute min/max ranges and histograms for each feature
            for i in range(X_array.shape[1]):
                feature_key = f"f{i}"
                self.feature_ranges[feature_key] = (np.min(X_array[:, i]), np.max(X_array[:, i]))
                # Compute a histogram with 20 bins for the feature distribution
                counts, bins = np.histogram(X_array[:, i], bins=20)
                self.feature_histograms[feature_key] = {"bins": bins, "counts": counts}
        self.task_type = self._detect_task_type()
        # Store number of classes for multiclass models
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
    
    def _tree_to_html(self, node):
        """
        Recursively convert a tree node (in dict form) into HTML.
        Internal nodes display the feature split and mini histogram.
        Leaf nodes display the leaf value.
        """
        if 'leaf' in node:
            leaf_value = node['leaf']
            display_val, tooltip = self._transform_leaf_value(leaf_value)
            
            return (
                f"<li class='node leaf'>"
                f"<div class='node-content' title='{tooltip}'>"
                f"<span class='node-type'>Leaf</span>: {display_val} "
                f"</div></li>"
            )
        else:
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
                f"<span class='node-type'>Split:</span> <span class='feature'>{feature_label}</span> "
                f"&le; <span class='condition'>{split_condition}</span> {svg_chart} "
                f"</div>"
            )
            children_html = "".join(self._tree_to_html(child) for child in node.get('children', []))
            return f"<li class='node internal'>{node_label}<ul>{children_html}</ul></li>"
    
    def _generate_tree_html(self, tree_dict):
        """
        Generate the full HTML representation for a tree given its dictionary.
        """
        html_content = "<ul class='tree'>" + self._tree_to_html(tree_dict) + "</ul>"
        return html_content
    
    def _generate_tree_selector(self):
        """
        Generate HTML for the tree selector dropdown.
        """
        options = []
        for i in range(self.num_trees):
            if self.task_type == 'multiclass_classification':
                class_idx, class_name, _ = self.get_tree_class(i)
                class_info = f" ({class_name})" if class_name else ""
                options.append(f'<option value="{i}">Tree {i}{class_info}</option>')
            else:
                options.append(f'<option value="{i}">Tree {i}</option>')
                
        options_html = "".join(options)
        selector_html = f"""
        <div id="tree-selector-container">
            <label for="tree-selector" style="font-weight: 600; color: #333;">Select Tree: </label>
            <input list="tree-options" id="tree-selector" placeholder="Enter tree number..." 
                   min="0" max="{self.num_trees-1}" type="number"
                   onchange="changeTree(this.value)" onkeyup="if(event.key==='Enter')changeTree(this.value)">
            <datalist id="tree-options">
                {options_html}
            </datalist>
            <div id="tree-class-info" style="margin-top: 8px; font-weight: 500;"></div>
        </div>
        """
        return selector_html
    
    def _generate_all_trees_html(self):
        """
        Generate HTML for all trees, but initially hide all except the first one.
        """
        all_trees_html = ""
        for i in range(self.num_trees):
            tree_json_str = self.trees_json[i]
            tree_dict = self.parse_tree_json(tree_json_str)
            tree_html = self._generate_tree_html(tree_dict)
            
            # Add class information header for multiclass
            class_header = ""
            if self.task_type == 'multiclass_classification':
                class_idx, class_name, num_classes = self.get_tree_class(i)
                if class_name:
                    round_num = i // num_classes if num_classes > 0 else 0
                    class_header = f"""
                    <div class="tree-class-header">
                        <h3>Tree {i}: Contributing to class "{class_name}" (Round {round_num})</h3>
                        <p>This tree contributes to the score for class "{class_name}". 
                           The final prediction is determined by summing contributions across all trees for each class.</p>
                    </div>
                    """
            
            display_style = "block" if i == 0 else "none"
            all_trees_html += f'''
            <div id="tree-{i}" class="tree-container" style="display: {display_style};">
                {class_header}
                {tree_html}
            </div>
            '''
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
        
        # Build an HTML fragment without full document wrappers,
        # so that the notebook's own background is not overwritten.
        html_template = f"""
        <style>
            /* Container styling for the tree */
            #visualizer-container {{
                width: 100%;
                height: 650px;
                display: flex;
                flex-direction: column;
                border: 1px solid #ddd;
                background: #fff;
                box-shadow: 0 2px 8px rgba(0,0,0,0.1);
                border-radius: 5px;
            }}
            #tree-selector-container {{
                padding: 15px;
                text-align: center;
                border-bottom: 1px solid #eee;
                background: #f8f9fa;
                border-radius: 5px 5px 0 0;
            }}
            #tree-selector {{
                padding: 8px 12px;
                border-radius: 4px;
                border: 1px solid #ccc;
                background: #fff;
                font-size: 14px;
                cursor: pointer;
                min-width: 150px;
                color: #333;
                font-weight: 500;
            }}
            .tree-container {{
                flex: 1;
                overflow: auto;
                padding: 20px;
                cursor: grab;
            }}
            ul.tree {{
                list-style-type: none;
                margin: 0;
                padding-left: 20px;
                position: relative;
            }}
            ul.tree ul {{
                margin-left: 20px;
                border-left: 1px dashed #ccc;
                padding-left: 15px;
            }}
            li.node {{
                margin: 10px 0;
                position: relative;
            }}
            .node-content {{
                background: #d0d0d0;
                padding: 8px 12px;
                border-radius: 4px;
                box-shadow: 0 1px 3px rgba(0,0,0,0.1);
                display: inline-block;
                color: #000;
                font-weight: 500;
            }}
            .node-type {{
                font-weight: bold;
                color: #005a9c;
            }}
            .feature {{
                font-weight: bold;
                color: #000;
            }}
            .condition {{
                color: #b71c1c;
                font-weight: 600;
            }}
            .node-stats {{
                font-size: 12px;
                color: #222;
                margin-top: 4px;
                font-weight: 500;
            }}
            .node-cover {{
                color: #222;
                font-weight: 500;
            }}
            li.leaf .node-content {{
                background: #e8f5e9;
                color: #000;
                font-weight: 500;
            }}
            .tree-class-header {{
                background-color: #e3f2fd;
                padding: 10px 15px;
                margin-bottom: 15px;
                border-radius: 5px;
                border-left: 4px solid #2196f3;
            }}
            .tree-class-header h3 {{
                margin: 0 0 8px 0;
                color: #0d47a1;
            }}
            .tree-class-header p {{
                margin: 0;
                color: #555;
                font-size: 14px;
            }}
        </style>
        <div id="visualizer-container">
            {tree_selector}
            <div id="trees-wrapper">
                {all_trees_html}
            </div>
        </div>
        <script src="https://unpkg.com/@panzoom/panzoom/dist/panzoom.min.js"></script>
        <script>
            // Initialize panzoom for the first tree
            const initPanzoom = (treeId) => {{
                const elem = document.getElementById(treeId);
                if (elem) {{
                    panzoom(elem, {{
                        zoomSpeed: 0.065,
                        maxZoom: 5,
                        minZoom: 0.3,
                        bounds: true,
                        boundsPadding: 0.1
                    }});
                    // Update cursor on mouse events for smoother interaction.
                    elem.addEventListener('mousedown', () => {{
                        elem.style.cursor = 'grabbing';
                    }});
                    elem.addEventListener('mouseup', () => {{
                        elem.style.cursor = 'grab';
                    }});
                }}
            }};
            
            // Initialize panzoom for the first tree
            initPanzoom('tree-0');
            
            // Function to change the displayed tree
            function changeTree(treeIndex) {{
                // Validate input
                treeIndex = parseInt(treeIndex);
                if (isNaN(treeIndex) || treeIndex < 0 || treeIndex >= {self.num_trees}) {{
                    alert(`Please enter a valid tree number between 0 and {self.num_trees-1}`);
                    return;
                }}
                
                // Hide all trees
                const trees = document.querySelectorAll('.tree-container');
                trees.forEach(tree => {{
                    tree.style.display = 'none';
                }});
                
                // Show the selected tree
                const selectedTree = document.getElementById(`tree-${{treeIndex}}`);
                if (selectedTree) {{
                    selectedTree.style.display = 'block';
                    // Initialize panzoom for this tree if not already done
                    initPanzoom(`tree-${{treeIndex}}`);
                    // Update the input value to match the selected tree
                    document.getElementById('tree-selector').value = treeIndex;
                    
                    // Update the dropdown to show the current selection
                    const options = document.querySelectorAll('#tree-options option');
                    if (options.length > treeIndex) {{
                        const selectedOption = options[treeIndex];
                        const treeClassInfo = document.getElementById('tree-class-info');
                        if (treeClassInfo && selectedOption.textContent.includes('(')) {{
                            treeClassInfo.textContent = selectedOption.textContent;
                        }}
                    }}
                }}
            }}
        </script>
        """
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