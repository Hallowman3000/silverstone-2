# 📊 Silverstone Streamlit Analytics Dashboard

A comprehensive sales analytics dashboard built with Streamlit for analyzing customer behavior, inventory forecasting, and profit optimization.

## 🎯 Overview

This dashboard provides multi-dimensional analytics for sales data, featuring customer segmentation using RFM analysis and K-Means clustering, brand-level inventory forecasting with XGBoost, and detailed profit analysis across categories and brands.

## 🛠️ Technology Stack

- **Frontend**: Streamlit
- **Data Processing**: pandas, numpy
- **Visualization**: Plotly (Express & Graph Objects)
- **Machine Learning**: scikit-learn, XGBoost
- **Clustering**: K-Means (6 segments)
- **Forecasting**: XGBoost Regressor with time-series features

## 📱 Page Functionality

### 1. Dashboard Overview
Executive summary providing high-level insights:
- **Key Metrics**: Whale customers, at-risk customers, total profit, average order value
- **Customer Distribution**: Pie chart showing segment breakdown
- **Revenue Analysis**: Bar chart of revenue by customer segment
- **Segment Legend**: Color-coded overview of all customer segments

### 2. Customer Segmentation
RFM (Recency, Frequency, Monetary) analysis with K-Means clustering:
- **6 Customer Segments**:
  - **Whales** 🐋 (Green): Highest spenders with excellent engagement
  - **Loyal Customers** 💙 (Blue): High value, frequent purchasers
  - **Potential Loyalists** 🟠 (Orange): Growing customers with potential
  - **Promising** 🟣 (Purple): Medium value customers
  - **At Risk** 🔴 (Red): Declining engagement, needs retention
  - **Hibernating** ⚪ (Gray): Lowest engagement, dormant

**Features**:
- Interactive RFM scatter plots (configurable X/Y axes)
- Segment statistics and averages
- Comprehensive segment summary table
- Cluster feature analysis with gradient visualization
- Feature lineage documentation

### 3. Inventory Forecast
Brand-level demand analysis and data reliability metrics:
- **Weekly Demand Visualization**: Time-series plots with moving averages
- **Data Reliability Metrics**:
  - **Data Density**: Percentage of weeks with sales activity
  - **Average Gap Length**: Mean consecutive weeks with zero sales
  - **Max Gap**: Longest period of zero sales
- **Brand Statistics**: Total weeks, average weekly demand, max demand
- **Comparative Analysis**: Most consistent vs. variable traffic brands

### 4. Profit Analysis
Profitability insights across multiple dimensions:
- **Profit Trends**: Monthly time-series with filled area chart
- **Category Analysis**: Top 10 categories by total profit
- **Brand Analysis**: Top 10 brands by profitability
- **Profit Margins Table**: Detailed breakdown with transaction counts and margin percentages

### 5. Customer Lookup
Search and browse customer information:
- **Customer Search**: Find customers by name with instant results
- **Color-Coded Cards**: Visual representation using segment colors
- **Segment Filtering**: Browse top 50 customers by segment
- **Data Export**: Download complete customer segmentation CSV

## 🔄 Data Transformations

### RFM Metrics Computation
1. **Recency**: Days since last purchase (lower is better)
   - Source: `postingDate` column
   - Calculation: Reference date - max(postingDate) per customer

2. **Frequency**: Number of transactions (higher is better)
   - Source: `entryNo` count per customer
   - Measures customer activity level

3. **Monetary**: Total sales amount (higher is better)
   - Source: Sum of `salesAmountActual` per customer
   - Measures customer lifetime value

### Data Quality Enhancements
- **Outlier Handling**: RFM values clipped to 5th-95th percentiles to reduce influence of extreme values
- **Log Transformation**: Applied to RFM features before scaling for better clustering performance
- **Standard Scaling**: Features normalized using StandardScaler for K-Means clustering

### Clustering Process
- **Algorithm**: K-Means with 6 clusters
- **Initialization**: 10 random initializations (n_init=10) for stability
- **Segment Assignment**: Clusters ranked by mean monetary value and labeled hierarchically

### Inventory Forecasting
- **Aggregation**: Weekly resampling (Monday start) of `invoicedQuantity` by brand
- **Feature Engineering**:
  - Lag features: 1, 4, 12 weeks
  - Rolling averages: 4, 12 weeks
  - Cyclical month encoding: sin/cos transformation
- **Model**: XGBoost Regressor with time-series cross-validation
- **Reliability Calculation**: Gap analysis for zero-sales periods

### Profit Analysis
- **Monthly Aggregation**: Profit summed by month for trend analysis
- **Category/Brand Grouping**: Aggregated profit, sales, and transaction counts
- **Margin Calculation**: (Total Profit / Total Sales) × 100

## 🚀 Setup Instructions

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/Hallowman3000/Silverstone_Streamlit_Dashboard.git
   cd Silverstone_Streamlit_Dashboard
   ```

2. **Install dependencies**
   ```bash
   pip install streamlit pandas numpy plotly scikit-learn xgboost
   ```

3. **Prepare your data file**
   
   You need a CSV file named `Silverstone.csv` with the following required columns:
   - `postingDate` (format: DD-MMM-YY, e.g., "15-Jan-24")
   - `customerName`
   - `entryNo`
   - `salesAmountActual`
   - `profit`
   - `invoicedQuantity`
   - `brand`
   - `itemCategory`
   - `description`

   Place this file in the root directory of the project.

4. **Run the application**
   ```bash
   streamlit run streamlit_app.py
   ```

5. **Access the dashboard**
   
   Open your browser and navigate to `http://localhost:8501`

## 📁 Project Structure

```
Silverstone_Streamlit_Dashboard/
├── streamlit_app.py          # Main Streamlit application with all 5 pages
├── inventory_forecaster.py   # XGBoost-based inventory forecasting module
├── profit_predictor.py       # Profit prediction module with feature engineering
├── anvil_server.py          # Anvil server integration (optional)
├── test_xgb.py              # XGBoost testing utilities
├── regenerate_notebook.py   # Notebook regeneration helper
├── anvil_ui_guide.md        # Anvil UI documentation
├── README.md                # This file
├── .gitignore               # Git ignore rules
└── Silverstone.csv          # Your data file (not included in repo)
```

## 🔒 Security & Privacy

**Important**: This repository is public and does NOT include any sensitive data files:
- All CSV files are excluded via `.gitignore`
- Jupyter notebooks are excluded (may contain embedded data)
- Only source code and documentation are version-controlled

**You must provide your own data file** (`Silverstone.csv`) following the schema outlined in the Setup Instructions.

## 🎨 Features & Highlights

- **Interactive Visualizations**: All charts are interactive Plotly graphs with hover details
- **Color-Coded Segments**: Consistent color scheme across all pages for easy recognition
- **Responsive Design**: Wide layout optimized for desktop viewing
- **Real-time Metrics**: Cached data loading for fast performance
- **Export Functionality**: Download customer segmentation data as CSV
- **Data Quality Insights**: Reliability metrics for inventory forecasting

## 📊 Sample Insights

The dashboard helps answer questions like:
- Who are my most valuable customers, and how can I retain them?
- Which customer segments are at risk of churning?
- What is the demand pattern for each brand?
- Which categories and brands are most profitable?
- How reliable is my sales data for forecasting?

## 🤝 Contributing

This is a personal analytics project. If you have suggestions or improvements, feel free to fork and submit pull requests.

## 📄 License

This project is open source and available under the MIT License.

## 👤 Author

**Hallowman3000**
- GitHub: [@Hallowman3000](https://github.com/Hallowman3000)

---

**Note**: This dashboard was built for business analytics and decision support. Ensure your data file complies with your organization's data privacy policies before use.
