
import json
import re

def create_cell(source, cell_type="code"):
    return {
        "cell_type": cell_type,
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": source
    }

def read_file(filepath):
    with open(filepath, 'r') as f:
        return f.read()

def clean_source(source):
    # Ensure consistent newlines
    return source.replace('\r\n', '\n')

def generate_notebook():
    py_file = 'inventory_forecaster.py'
    out_file = 'inventory_forecaster.ipynb'
    
    content = read_file(py_file)
    lines = content.splitlines(keepends=True)
    
    cells = []
    
    # Cell 1: Imports
    # Grab all lines until the first class definition or function
    imports = []
    class_start_idx = 0
    for i, line in enumerate(lines):
        if line.strip().startswith('class InventoryForecaster'):
            class_start_idx = i
            break
        imports.append(line)
    
    cells.append(create_cell("".join(imports).strip()))
    
    # Cell 2: Class Definition
    # Capture the entire class block.
    # We assume 'def main():' marks the end of the class.
    class_lines = []
    main_start_idx = 0
    for i in range(class_start_idx, len(lines)):
        line = lines[i]
        if line.strip().startswith('def main():'):
            main_start_idx = i
            break
        class_lines.append(line)
        
    class_source = "".join(class_lines).rstrip()
    
    # SAFETY CHECK: Ensure indentation is consistent (4 spaces)
    # and check for suspicious 'self' usage at module level?
    # No, just dump it. But we can validate if 'tscv' is indented.
    
    cells.append(create_cell(class_source))
    
    # Execution Cells (Breaking down the main function logic)
    
    # 3. Initialize
    cells.append(create_cell("# Initialize Forecaster\n# We use 3 splits for Time Series Cross-Validation\nforecaster = InventoryForecaster(n_splits=3)"))
    
    # 4. Load Data
    cells.append(create_cell("# Load Data\n# Reads the CSV and parses dates\nforecaster.load_data('Silverstone.csv')"))
    
    # 5. Resample
    cells.append(create_cell("# Resample Data\n# Aggregates daily data into weekly buckets per brand\nforecaster.resample_by_brand()"))
    
    # 6. Brand Selection
    # Extracting logic from main() implies writing it out explicitly
    brand_logic = """# Brand Selection Logic
# Checks if BLACKHAWK exists, otherwise picks the brand with most data
available_brands = list(forecaster.brand_data.keys())
target_brand = None

# Case-insensitive search for Blackhawk
for brand in available_brands:
    if 'blackhawk' in brand.lower():
        target_brand = brand
        break

# Fallback
if target_brand is None:
    print(f"BLACKHAWK brand not found. Available brands: {available_brands[:10]}...")
    brand_sizes = {b: len(df) for b, df in forecaster.brand_data.items()}
    target_brand = max(brand_sizes, key=brand_sizes.get)
    print(f"Using brand with most data: {target_brand}")
else:
    print(f"Selected brand: {target_brand}")
"""
    cells.append(create_cell(brand_logic))
    
    # 7. Train
    cells.append(create_cell("""# Train Model
# Uses Tweedie Regression (Negative Binomial equivalent)
results = forecaster.train(target_brand)"""))
    
    # 8. Feature Importance
    cells.append(create_cell("""# Compute Feature Importance
# Uses permutation importance to determine which features drive predictions
importance = forecaster.compute_permutation_importance(results['X'], results['y'])"""))
    
    # 9. Plotting
    cells.append(create_cell("""# Plot Predictions
# Visual comparison of Predicted vs Actual for the holdout set
fig1 = forecaster.plot_predictions(
    results,
    fold=-1,  # Last fold
    title=f"{results['brand']} - Inventory Forecast (Tweedie Regressor)"
)
plt.show()"""))
    
    cells.append(create_cell("""# Plot Feature Importance
fig2 = forecaster.plot_feature_importance()
plt.show()"""))

    # Create Notebook structure
    notebook = {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3"
            },
            "language_info": {
                "codemirror_mode": {
                    "name": "ipython",
                    "version": 3
                },
                "file_extension": ".py",
                "mimetype": "text/x-python",
                "name": "python",
                "nbconvert_exporter": "python",
                "pygments_lexer": "ipython3",
                "version": "3.8.5"
            }
        },
        "nbformat": 4,
        "nbformat_minor": 4
    }
    
    with open(out_file, 'w') as f:
        json.dump(notebook, f, indent=4)
        
    print(f"Generated {out_file} with {len(cells)} cells.")

if __name__ == "__main__":
    generate_notebook()
