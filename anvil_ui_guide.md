# Anvil Dashboard UI Guide

Step-by-step guide to create the dashboard UI in Anvil Editor.

## Prerequisites

1. **Install anvil-uplink locally**:
   ```bash
   pip install anvil-uplink
   ```

2. **Create Anvil App**:
   - Go to [anvil.works](https://anvil.works) and sign up
   - Click "New App" → Select "Material Design 3"
   - Name it "Sales Analytics Dashboard"

3. **Enable Uplink**:
   - Click the `+` button in the sidebar
   - Select "Uplink..."
   - Click "Enable Server Uplink"
   - Copy the Uplink key

4. **Start Local Server**:
   ```bash
   cd "c:\Users\ADMIN\Desktop\silverstone 2 test"
   python anvil_server.py YOUR_UPLINK_KEY
   ```

---

## Color Scheme Reference

### Customer Segments
| Segment | Color | Hex | Meaning |
|---------|-------|-----|---------|
| Champions | 🟢 Green | `#2ecc71` | Top customers |
| Loyal Customers | 🔵 Blue | `#3498db` | Good customers |
| Potential Loyalists | 🟠 Orange | `#f39c12` | Growing |
| At Risk | 🔴 Red | `#e74c3c` | Needs attention |
| Hibernating | ⚪ Gray | `#95a5a6` | Inactive |

### Inventory Stock Status
| Status | Color | Hex | Meaning |
|--------|-------|-----|---------|
| In Stock | 🟢 Green | `#2ecc71` | Good levels |
| Low Stock | 🟠 Orange | `#f39c12` | Running low |
| Out of Stock | 🔴 Red | `#e74c3c` | Need to reorder |
| Overstocked | 🔵 Blue | `#3498db` | Excess inventory |

---

## Dashboard Pages

### Page 1: Overview Dashboard

**Components to add:**
1. **Header Label**: "Sales Analytics Dashboard"
2. **4 Metric Cards** (use Card components):
   - Total Customers
   - Total Revenue  
   - Champions Count (green badge)
   - At Risk Count (red badge)

**Code (Form1):**
```python
from anvil import *
import anvil.server

class Form1(Form1Template):
    def __init__(self, **properties):
        self.init_components(**properties)
        self.load_overview()
    
    def load_overview(self):
        data = anvil.server.call('get_dashboard_overview')
        self.lbl_customers.text = f"{data['total_customers']:,}"
        self.lbl_revenue.text = f"KES {data['total_revenue']:,.0f}"
        self.lbl_champions.text = str(data['champions'])
        self.lbl_at_risk.text = str(data['at_risk'])
```

---

### Page 2: Customer Segmentation

**Components:**
1. **Plot** (for pie chart): `self.plot_segments`
2. **DataGrid**: `self.grid_customers`
3. **DropDown**: `self.dd_segment` (segment selector)

**Code:**
```python
def load_segments(self):
    data = anvil.server.call('get_customer_segments')
    
    # Pie chart with segment colors
    self.plot_segments.data = [{
        'type': 'pie',
        'labels': data['distribution']['labels'],
        'values': data['distribution']['values'],
        'marker': {'colors': data['distribution']['colors']},
        'hole': 0.4
    }]
    
    self.plot_segments.layout = {
        'title': 'Customer Segments',
        'showlegend': True
    }

def dd_segment_change(self, **event_args):
    segment = self.dd_segment.selected_value
    data = anvil.server.call('get_customers_by_segment', segment)
    
    # Color the header based on segment
    self.card_header.background = data['segment_color']
    self.grid_customers.items = data['customers']
```

---

### Page 3: Inventory Forecast

**Components:**
1. **DropDown**: `self.dd_brand` (brand selector)
2. **Plot**: `self.plot_forecast` (line chart)
3. **Plot**: `self.plot_importance` (bar chart)
4. **Labels**: MAE and RMSE metrics

**Code:**
```python
def form_show(self, **event_args):
    # Load brands for dropdown
    brands = anvil.server.call('get_available_brands')
    self.dd_brand.items = brands
    
def dd_brand_change(self, **event_args):
    brand = self.dd_brand.selected_value
    data = anvil.server.call('get_inventory_forecast', brand)
    
    if not data['success']:
        alert(f"Error: {data['error']}")
        return
    
    predictions = data['predictions']
    
    # Line chart with color-coded points
    self.plot_forecast.data = [
        {
            'x': [p['date'] for p in predictions],
            'y': [p['actual'] for p in predictions],
            'name': 'Actual',
            'type': 'scatter',
            'line': {'color': '#3498db'}
        },
        {
            'x': [p['date'] for p in predictions],
            'y': [p['predicted'] for p in predictions],
            'name': 'Predicted', 
            'type': 'scatter',
            'line': {'color': '#e74c3c', 'dash': 'dash'},
            'marker': {
                'color': [p['status_color'] for p in predictions],
                'size': 10
            }
        }
    ]
    
    self.plot_forecast.layout = {
        'title': f'{brand} Inventory Forecast',
        'xaxis': {'title': 'Date'},
        'yaxis': {'title': 'Quantity'}
    }
    
    # Metrics
    self.lbl_mae.text = f"MAE: {data['metrics']['mae']}"
    self.lbl_rmse.text = f"RMSE: {data['metrics']['rmse']}"
```

---

## Stock Status Legend

Add a legend component showing the color meanings:

```python
def show_stock_legend(self):
    colors = anvil.server.call('get_stock_summary')['colors']
    
    # Create colored labels
    self.lbl_in_stock.background = colors['in_stock']
    self.lbl_low_stock.background = colors['low_stock']  
    self.lbl_out_stock.background = colors['out_of_stock']
    self.lbl_overstock.background = colors['overstocked']
```

---

## Available Server Functions

| Function | Returns | Use For |
|----------|---------|---------|
| `get_dashboard_overview()` | dict | Overview metrics |
| `get_customer_segments()` | dict | Segment pie chart |
| `get_customers_by_segment(segment, limit)` | dict | Customer list |
| `get_rfm_chart_data()` | dict | RFM scatter plot |
| `get_available_brands()` | list | Brand dropdown |
| `get_inventory_forecast(brand)` | dict | Forecast chart |
| `get_stock_summary()` | dict | Stock status donut |

---

## Quick Start

1. Start the local server:
   ```bash
   python anvil_server.py YOUR_UPLINK_KEY
   ```

2. In Anvil Editor, test the connection:
   ```python
   # In the Anvil Python console
   print(anvil.server.call('get_available_brands'))
   ```

3. Build your UI using the code snippets above!
