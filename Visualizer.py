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
        self.trees_json = model.get_dump(dump_format='json')
        self.num_trees = len(self.trees_json)
        self.feature_ranges = {}
        self.feature_histograms = {}
        if X is not None:
            # Compute min/max ranges and histograms for each feature
            for i in range(X.shape[1]):
                feature_key = f"f{i}"
                self.feature_ranges[feature_key] = (np.min(X[:, i]), np.max(X[:, i]))
                # Compute a histogram with 20 bins for the feature distribution
                counts, bins = np.histogram(X[:, i], bins=20)
                self.feature_histograms[feature_key] = {"bins": bins, "counts": counts}

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
        min_val, max_val = self.feature_ranges[feature]
        try:
            threshold = float(split_condition)
            norm_threshold = (threshold - min_val) / (max_val - min_val) if max_val != min_val else 0.5
            norm_threshold = max(0, min(1, norm_threshold))
            # Retrieve histogram data computed from X.
            hist_data = self.feature_histograms[feature]
            counts = hist_data["counts"]
            num_bins = len(counts)
            
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
        except Exception:
            return ""
    
    def _tree_to_html(self, node):
        """
        Recursively convert a tree node (in dict form) into HTML.
        Internal nodes display the feature split and mini histogram.
        Leaf nodes display the leaf value.
        """
        if 'leaf' in node:
            leaf_value = node['leaf']
            if self.target_names is not None:
                try:
                    target_index = int(round(leaf_value))
                    target_label = self.target_names[target_index]
                    display_val = f"{target_label} (raw: {leaf_value:.3f})"
                except Exception:
                    display_val = f"{leaf_value:.3f}"
            else:
                display_val = f"{leaf_value:.3f}"
            return (
                f"<li class='node leaf'>"
                f"<div class='node-content'>"
                f"<span class='node-type'>Leaf</span>: {display_val} "
                f"<span class='node-cover'>[cover: {node.get('cover', '')}]</span>"
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
        options = "".join([f'<option value="{i}">Tree {i}</option>' for i in range(self.num_trees)])
        selector_html = f"""
        <div id="tree-selector-container">
            <label for="tree-selector" style="font-weight: 600; color: #333;">Select Tree: </label>
            <input list="tree-options" id="tree-selector" placeholder="Enter tree number..." 
                   min="0" max="{self.num_trees-1}" type="number"
                   onchange="changeTree(this.value)" onkeyup="if(event.key==='Enter')changeTree(this.value)">
            <datalist id="tree-options">
                {options}
            </datalist>
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
            display_style = "block" if i == 0 else "none"
            all_trees_html += f'<div id="tree-{i}" class="tree-container" style="display: {display_style};">{tree_html}</div>'
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
                }}
            }}
        </script>
        """
        display(HTML(html_template))