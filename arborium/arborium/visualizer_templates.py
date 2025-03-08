"""
HTML templates and styles for XGBTreeVisualizer.

This module contains all the HTML, CSS, and JavaScript code used by XGBTreeVisualizer
to render interactive tree visualizations.
"""

# CSS Styles for the tree visualization
TREE_CSS = """
    /* Container styling for the tree */
    #visualizer-container {
        width: 100%;
        height: 650px;
        display: flex;
        flex-direction: column;
        border: 0px solid #ddd;
        background: transparent; /* Make container background transparent */
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        border-radius: 0px;
        overflow: visible; /* Allow content to extend beyond container */
    }
    #tree-selector-container {
        padding: 15px;
        text-align: center;
        border-bottom: 1px solid #eee;
        background: #f8f9fa;
        border-radius: 5px 5px 0 0;
        z-index: 10;
    }
    #tree-selector {
        padding: 8px 12px;
        border-radius: 4px;
        border: 1px solid #ccc;
        background: #fff;
        font-size: 14px;
        cursor: pointer;
        color: #333;
        font-weight: 500;
    }
    .tree-container {
        flex: 1;
        overflow: visible; /* Allow tree to extend beyond container */
        padding: 20px;
        cursor: grab;
        background: transparent; /* Make tree container background transparent */
        width: 100%;
        height: 100%;
        position: absolute;
        top: 0;
        left: 0;
        box-sizing: border-box;
    }
    .tree-content {
        width: fit-content;
        height: fit-content;
        min-width: 100%;
        min-height: 100%;
    }
    #trees-wrapper {
        flex: 1;
        position: relative;
        overflow: visible /* Contain the absolutely positioned trees */
        background: transparent; /* Make wrapper background transparent */
    }
    ul.tree {
        list-style-type: none;
        margin: 0;
        padding-left: 20px;
        position: relative;
        background: transparent; /* Make tree background transparent */
    }
    ul.tree ul {
        margin-left: 20px;
        border-left: 1px dashed #ccc;
        padding-left: 15px;
        position: relative;
    }
    /* Add visual indicators for yes/no branches */
    ul.tree ul:before {
        content: '';
        position: absolute;
        top: -10px;
        left: -15px;
        width: 15px;
        height: 20px;
        border-bottom: 1px dashed #ccc;
    }
    li.node {
        margin: 10px 0;
        position: relative;
    }
    /* Branch styling */
    li.yes-branch-node > ul {
        border-left-color: rgba(76, 175, 80, 0.7);
        border-left-style: solid;
        border-left-width: 2px;
    }
    li.no-branch-node > ul {
        border-left-color: rgba(244, 67, 54, 0.7);
        border-left-style: solid;
        border-left-width: 2px;
    }
    .node-content {
        background: #d0d0d0;
        padding: 8px 12px;
        border-radius: 4px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1);
        display: inline-block;
        color: #000;
        font-weight: 500;
        position: relative;
    }
    .node-type {
        font-weight: bold;
        color: #005a9c;
    }
    .feature {
        font-weight: bold;
        color: #000;
    }
    .condition {
        color: #b71c1c;
        font-weight: 600;
    }
    .branch-indicator {
        display: inline-block;
        padding: 3px 6px;
        margin-right: 8px;
        border-radius: 3px;
        font-size: 12px;
        font-weight: bold;
        vertical-align: middle;
    }
    /* Always use consistent colors for yes/no indicators regardless of parent node */
    .yes-indicator {
        background-color: #4caf50;
        color: white;
    }
    .no-indicator {
        background-color: #f44336;
        color: white;
    }
    /* Add left border to nodes based on branch type */
    li.yes-branch-node > .node-content {
        border-left: 4px solid #4caf50;
    }
    li.no-branch-node > .node-content {
        border-left: 4px solid #f44336;
    }
    .node-stats {
        font-size: 12px;
        color: #222;
        margin-top: 4px;
        font-weight: 500;
    }
    .node-cover {
        color: #222;
        font-weight: 500;
    }
    li.leaf .node-content {
        background: #e8f5e9;
        color: #000;
        font-weight: 500;
    }
    .tree-class-header, .tree-info {
        background-color: #e3f2fd;
        padding: 10px 15px;
        margin-bottom: 15px;
        border-radius: 5px;
        border-left: 4px solid #2196f3;
    }
    .tree-class-header h3, .tree-info h3 {
        margin: 0 0 8px 0;
        color: #0d47a1;
    }
    .tree-class-header p {
        margin: 0;
        color: #555;
        font-size: 14px;
    }
"""

# CSS for simplified tree, identical to main tree CSS to ensure consistent styling
SIMPLIFIED_TREE_CSS = TREE_CSS

# JavaScript for tree visualization with panzoom functionality
TREE_JAVASCRIPT_TEMPLATE = """
    // Store tree count as a variable
    const numTrees = {num_trees};
    const maxTreeIndex = {max_tree_index};
    
    // Store panzoom instances to prevent reinitializing
    const panzoomInstances = {{}};
    
    // Initialize panzoom for a specific tree container
    const initPanzoom = (treeId) => {{
        const elem = document.getElementById(treeId);
        if (!elem) {{
            console.error(`Element with ID ${{treeId}} not found`);
            return null;
        }}
        
        try {{
            // Create panzoom instance for this tree
            const pz = panzoom(elem, {{
                zoomSpeed: 0.065,
                maxZoom: 5,
                minZoom: 0.3,
                bounds: false,
                boundsPadding: 0.1,
                autocenter: false
            }});
            
            // Update cursor on mouse events
            elem.addEventListener('mousedown', () => {{
                elem.style.cursor = 'grabbing';
            }});
            elem.addEventListener('mouseup', () => {{
                elem.style.cursor = 'grab';
            }});
            
            console.log(`Initialized panzoom for ${{treeId}}`);
            return pz;
        }} catch (e) {{
            console.error(`Failed to initialize panzoom: ${{e.message}}`);
            return null;
        }}
    }};
    
    // Function to change the displayed tree
    function changeTree(treeIndex) {{
        // Validate input
        treeIndex = parseInt(treeIndex);
        
        if (isNaN(treeIndex) || treeIndex < 0 || treeIndex >= numTrees) {{
            alert(`Please enter a valid tree number between 0 and ${{maxTreeIndex}}`);
            return;
        }}
        
        console.log(`Changing to tree ${{treeIndex}}`);
        
        // Hide all trees first
        for (let i = 0; i < numTrees; i++) {{
            const treeElem = document.getElementById(`tree-${{i}}`);
            if (treeElem) {{
                treeElem.style.display = 'none';
                
                // Destroy existing panzoom instance if it exists
                if (panzoomInstances[`tree-${{i}}`]) {{
                    try {{
                        panzoomInstances[`tree-${{i}}`].dispose();
                        delete panzoomInstances[`tree-${{i}}`];
                        console.log(`Disposed panzoom for tree-${{i}}`);
                    }} catch (e) {{
                        console.warn(`Failed to dispose panzoom for tree-${{i}}: ${{e.message}}`);
                    }}
                }}
            }}
        }}
        
        // Show the selected tree
        const selectedTree = document.getElementById(`tree-${{treeIndex}}`);
        console.log("REAAAACHED")
        console.log(selectedTree)
        if (selectedTree) {{
            // Make sure the tree is visible
            selectedTree.style.display = 'block';
            
            // Update tree selector value
            const treeSelector = document.getElementById('tree-selector');
            if (treeSelector && treeSelector.value != treeIndex) {{
                treeSelector.value = treeIndex;
            }}
            
            // Update tree info display
            updateTreeInfo(treeIndex, selectedTree);
            
            // Initialize panzoom after a small delay to ensure the tree is fully visible
            setTimeout(() => {{
                if (!panzoomInstances[`tree-${{treeIndex}}`]) {{
                    panzoomInstances[`tree-${{treeIndex}}`] = initPanzoom(`tree-${{treeIndex}}`);
                }}
            }}, 50);
        }} else {{
            console.error(`Tree container for tree-${{treeIndex}} not found`);
        }}
    }}
    
    // Update the tree info display
    function updateTreeInfo(treeIndex, treeElement) {{
        const treeClassInfo = document.getElementById('tree-class-info');
        if (treeClassInfo && treeElement) {{
            // Get class info from the tree header
            const treeHeader = treeElement.querySelector('.tree-class-header h3, .tree-info h3');
            if (treeHeader) {{
                treeClassInfo.textContent = treeHeader.textContent;
            }} else {{
                treeClassInfo.textContent = `Tree ${{treeIndex}}`;
            }}
        }}
    }}
    
    // Initialize only first tree on page load
    window.addEventListener('DOMContentLoaded', () => {{
        // Initialize panzoom for the first tree after a short delay
        setTimeout(() => {{
            panzoomInstances['tree-0'] = initPanzoom('tree-0');
        }}, 50);
        
        // Set up keyboard navigation
        document.addEventListener('keydown', (event) => {{
            // Only respond to arrow keys when not in an input field
            if (document.activeElement.tagName !== 'INPUT') {{
                const currentTree = parseInt(document.getElementById('tree-selector').value);
                
                // Left arrow key - previous tree
                if (event.key === 'ArrowLeft') {{
                    changeTree(Math.max(0, currentTree - 1));
                }}
                // Right arrow key - next tree
                else if (event.key === 'ArrowRight') {{
                    changeTree(Math.min(maxTreeIndex, currentTree + 1));
                }}
            }}
        }});
        
        // Update initial tree info
        const firstTree = document.getElementById('tree-0');
        if (firstTree) {{
            updateTreeInfo(0, firstTree);
        }}
        
        // Set up event handler for the number input
        const treeSelector = document.getElementById('tree-selector');
        if (treeSelector) {{
            treeSelector.addEventListener('change', function() {{
                changeTree(this.value);
            }});
            
            // Also handle input event for real-time response
            treeSelector.addEventListener('input', function() {{
                // Only change tree if the value is valid
                const value = parseInt(this.value);
                if (!isNaN(value) && value >= 0 && value <= maxTreeIndex) {{
                    changeTree(value);
                }}
            }});
        }}
    }});
"""

# JavaScript for simplified tree visualization (single tree)
SIMPLIFIED_TREE_JAVASCRIPT = """
    // Initialize panzoom for the simplified tree
    const elem = document.getElementById('simplified-tree-wrapper');
    if (elem) {{
        try {{
            // Initialize panzoom with configuration for visible overflow
            panzoom(elem, {{
                zoomSpeed: 0.065,
                maxZoom: 5,
                minZoom: 0.3,
                bounds: false,
                boundsPadding: 0.1,
                autocenter: false
            }});
            
            // Update cursor on mouse events for smoother interaction
            elem.addEventListener('mousedown', () => {{
                elem.style.cursor = 'grabbing';
            }});
            elem.addEventListener('mouseup', () => {{
                elem.style.cursor = 'grab';
            }});
            
            console.log('Simplified tree panzoom initialized successfully');
        }} catch (e) {{
            console.error(`Failed to initialize panzoom for simplified tree: ${{e.message}}`);
        }}
    }} else {{
        console.warn('Simplified tree element not found for panzoom initialization');
    }}
"""

# Function to generate the main tree visualization HTML
def get_tree_html_template(tree_selector, all_trees_html, num_trees, max_tree_index, instance_id):
    """
    Generate the HTML template for the main tree visualization.
    
    Args:
        tree_selector (str): HTML for the tree selector UI
        all_trees_html (str): HTML containing all tree structures
        num_trees (int): Total number of trees
        max_tree_index (int): Maximum tree index
        instance_id (str): Unique identifier for the visualizer instance
        
    Returns:
        str: Complete HTML template
    """
    # Format the JavaScript with the tree counts
    tree_javascript = TREE_JAVASCRIPT_TEMPLATE.format(
        num_trees=num_trees,
        max_tree_index=max_tree_index
    )
    
    return f"""
    <style>
    {TREE_CSS}
    </style>
    <div id="{instance_id}_container" class="xgb-tree-visualizer">
        {tree_selector}
        <div id="trees-wrapper">
            {all_trees_html}
        </div>
    </div>
    <script src="https://unpkg.com/@panzoom/panzoom/dist/panzoom.min.js"></script>
    <script>
    (function() {{
        // Using an IIFE to create a closure and prevent global namespace pollution
        const instance = "{instance_id}";
        
        // Get elements specific to this instance
        const treeSelector = document.getElementById(instance + "_tree-selector");
        
        // Function to show a specific tree
        function showTree(treeIndex) {{
            // Validate the tree index
            treeIndex = parseInt(treeIndex);
            if (isNaN(treeIndex) || treeIndex < 0 || treeIndex >= {num_trees}) {{
                console.error("Invalid tree index:", treeIndex);
                return;
            }}
            
            // Hide all trees
            for (let i = 0; i <= {max_tree_index}; i++) {{
                const treeElement = document.getElementById(instance + "_tree-" + i);
                if (treeElement) {{
                    treeElement.style.display = "none";
                }}
            }}
            
            // Show the selected tree
            const selectedTree = document.getElementById(instance + "_tree-" + treeIndex);
            if (selectedTree) {{
                selectedTree.style.display = "block";
            }}
        }}
        
        // Add event listeners to tree selector
        if (treeSelector) {{
            // Handle change event (when user presses enter or field loses focus)
            treeSelector.addEventListener("change", function() {{
                showTree(this.value);
            }});
            
            // Handle input event for real-time response when typing or using up/down buttons
            treeSelector.addEventListener("input", function() {{
                const value = parseInt(this.value);
                if (!isNaN(value) && value >= 0 && value < {num_trees}) {{
                    showTree(value);
                }}
            }});
        }}
    }})();
    </script>
    """

# Function to generate the simplified tree visualization HTML
def get_simplified_tree_html_template(tree_header, tree_html):
    """
    Generate the HTML template for the simplified tree visualization.
    
    Args:
        tree_header (str): HTML header for the simplified tree
        tree_html (str): HTML containing the simplified tree structure
        
    Returns:
        str: Complete HTML template
    """
    return f"""
    <style>
    {SIMPLIFIED_TREE_CSS}
    </style>
    <div id="simplified-tree-container">
        {tree_header}
        <div id="simplified-tree-wrapper">
            {tree_html}
        </div>
    </div>
    <script src="https://unpkg.com/@panzoom/panzoom/dist/panzoom.min.js"></script>
    <script>
    {SIMPLIFIED_TREE_JAVASCRIPT}
    </script>
    """

# HTML templates for different components

def get_tree_selector_html(num_trees, instance_id):
    """
    Generate HTML for the tree selector component.
    
    Args:
        num_trees (int): Total number of trees
        instance_id (str): Unique identifier for the visualizer instance
        
    Returns:
        str: HTML for tree selector
    """
    return f"""
    <div class="tree-controls">
        <label for="{instance_id}_tree-selector">Select Tree: </label>
        <input type="number" id="{instance_id}_tree-selector" class="tree-selector" min="0" max="{num_trees-1}" value="0" style="width: 80px; padding: 8px 12px; border-radius: 4px; border: 1px solid #ccc; font-size: 14px;">
    </div>
    """

def get_tree_class_header_html(i, class_name, round_num):
    """
    Generate HTML for a tree class header for multiclass models.
    
    Args:
        i (int): Tree index
        class_name (str): Class name
        round_num (int): Round number
        
    Returns:
        str: HTML for tree class header
    """
    return f"""
    <div class="tree-class-header">
        <h3>Tree {i}: Contributing to class "{class_name}" (Round {round_num})</h3>
        <p style="color: #333333; font-weight: 500;">This tree contributes to the score for class "{class_name}". 
           The final prediction is determined by summing contributions across all trees for each class.</p>
    </div>
    """

def get_tree_info_html(i):
    """
    Generate HTML for basic tree info header.
    
    Args:
        i (int): Tree index
        
    Returns:
        str: HTML for tree info
    """
    return f"""<div class="tree-class-header"><h3>Tree {i}</h3><p style="color: #333333; font-weight: 500;">This tree contributes directly to the predicted value. Final prediction is the sum of all tree outputs.</p></div>"""

def get_simplified_tree_header_html(max_depth, n_components=None, n_samples=None):
    """
    Generate HTML header for simplified tree.
    
    Args:
        max_depth (int): Maximum depth of the simplified tree
        n_components (int, optional): Number of GMM components used, if GMM sampling was performed
        n_samples (int, optional): Number of samples used, if GMM sampling was performed
        
    Returns:
        str: HTML for simplified tree header
    """
    if n_components is not None and n_samples is not None:
        # If GMM sampling was used
        return f"""
        <div class="tree-class-header">
            <h3>Simplified Decision Tree (max_depth={max_depth})</h3>
            <p style="color: #333333; font-weight: 500;">This is a simplified decision tree fitted to match the XGBoost model's predictions.</p>
            <p style="color: #333333; font-weight: 500;">
                Created using Gaussian Mixture Model (GMM) sampling with {n_components} components
                and {n_samples:,} synthetic samples.
            </p>
        </div>
        """
    else:
        # Original version (no GMM information)
        return f"""
        <div class="tree-class-header">
            <h3>Simplified Decision Tree (max_depth={max_depth})</h3>
            <p style="color: #333333; font-weight: 500;">This is a simplified decision tree fitted to match the XGBoost model's predictions.</p>
        </div>
        """ 