"""
Anvil Server for Dashboard

Connects existing inventory forecaster and customer segmentation models
to Anvil via Uplink with color-coded outputs for easy interpretability.
"""

import os
import sys
import pandas as pd
import numpy as np
from datetime import datetime
import anvil.server

from inventory_forecaster import InventoryForecaster

# Your Anvil Uplink Key
UPLINK_KEY = "server_HM4FTOSI3W7LEVJP2NZTSF27-GPJZNK2ATFX2WPVE"


# =============================================================================
# COLOR SCHEMES
# =============================================================================

# Customer Segment Colors (Green = Best, Red = Needs Attention)
SEGMENT_COLORS = {
    'Champions': '#2ecc71',           # Green - Top customers
    'Loyal Customers': '#3498db',     # Blue - Good customers
    'Potential Loyalists': '#f39c12', # Orange - Growing
    'At Risk': '#e74c3c',             # Red - Needs attention
    'Hibernating': '#95a5a6'          # Gray - Inactive
}

# Stock Status Colors
STOCK_COLORS = {
    'in_stock': '#2ecc71',      # Green - Good stock levels
    'low_stock': '#f39c12',     # Orange - Running low
    'out_of_stock': '#e74c3c',  # Red - No stock
    'overstocked': '#3498db'    # Blue - Too much stock
}


# =============================================================================
# GLOBAL STATE
# =============================================================================

_forecaster = None
_customer_data = None
_rfm_data = None
_data_loaded = False


def initialize_models(data_path: str = 'Silverstone.csv'):
    """Initialize models on first call."""
    global _forecaster, _customer_data, _rfm_data, _data_loaded
    
    if _data_loaded:
        return
    
    print("\n" + "=" * 60)
    print("INITIALIZING ANVIL SERVER")
    print("=" * 60)
    
    # Load raw data for customer segmentation
    print("\n[1/2] Loading customer data...")
    df = pd.read_csv(data_path, encoding='latin-1')
    df['postingDate'] = pd.to_datetime(df['postingDate'], format='%d-%b-%y', errors='coerce')
    df = df.dropna(subset=['postingDate', 'customerName'])
    _customer_data = df
    
    # Compute RFM
    _rfm_data = compute_rfm(df)
    print(f"✓ Loaded {len(_rfm_data)} customers with segments")
    
    # Initialize Inventory Forecaster
    print("\n[2/2] Loading Inventory Forecaster...")
    _forecaster = InventoryForecaster(n_splits=3)
    _forecaster.load_data(data_path)
    _forecaster.resample_by_brand()
    print(f"✓ Loaded {len(_forecaster.brand_data)} brands")
    
    _data_loaded = True
    print("\n" + "=" * 60)
    print("SERVER READY")
    print("=" * 60 + "\n")


def compute_rfm(df):
    """Compute RFM metrics and segment customers."""
    from sklearn.preprocessing import StandardScaler
    from sklearn.cluster import KMeans
    
    reference_date = df['postingDate'].max() + pd.Timedelta(days=1)
    
    # Aggregate by customer
    rfm = df.groupby('customerName').agg({
        'postingDate': 'max',
        'entryNo': 'count',
        'salesAmountActual': 'sum',
        'profit': 'sum'
    }).reset_index()
    
    rfm.columns = ['customerName', 'lastPurchaseDate', 'frequency', 'monetary', 'profit']
    rfm['recency'] = (reference_date - rfm['lastPurchaseDate']).dt.days
    
    # Handle NaN and infinite values
    rfm = rfm.dropna(subset=['recency', 'frequency', 'monetary'])
    rfm = rfm[rfm['monetary'] > 0]  # Only customers with positive monetary value
    
    # K-Means clustering
    features = rfm[['recency', 'frequency', 'monetary']].copy()
    features_log = np.log1p(features)
    
    # Fill any remaining NaN with 0
    features_log = features_log.fillna(0)
    
    scaler = StandardScaler()
    scaled = scaler.fit_transform(features_log)
    
    kmeans = KMeans(n_clusters=5, random_state=42, n_init=10)
    rfm['cluster'] = kmeans.fit_predict(scaled)
    
    # Assign labels based on monetary
    cluster_ranks = rfm.groupby('cluster')['monetary'].mean().sort_values(ascending=False)
    labels = ['Champions', 'Loyal Customers', 'Potential Loyalists', 'At Risk', 'Hibernating']
    mapping = {cluster: labels[i] for i, cluster in enumerate(cluster_ranks.index)}
    
    rfm['segment'] = rfm['cluster'].map(mapping)
    rfm['segment_color'] = rfm['segment'].map(SEGMENT_COLORS)
    
    return rfm


# =============================================================================
# CUSTOMER SEGMENTATION ENDPOINTS
# =============================================================================

@anvil.server.callable
def get_customer_segments():
    """Get customer segments with color coding."""
    initialize_models()
    
    # Segment distribution
    distribution = _rfm_data['segment'].value_counts()
    
    # Summary stats per segment
    summary = _rfm_data.groupby('segment').agg({
        'recency': 'mean',
        'frequency': 'mean',
        'monetary': ['mean', 'sum'],
        'profit': 'sum',
        'customerName': 'count'
    }).round(2)
    summary.columns = ['Avg Recency', 'Avg Frequency', 'Avg Monetary', 
                       'Total Revenue', 'Total Profit', 'Count']
    
    return {
        'distribution': {
            'labels': distribution.index.tolist(),
            'values': distribution.values.tolist(),
            'colors': [SEGMENT_COLORS[s] for s in distribution.index]
        },
        'summary': summary.reset_index().to_dict('records'),
        'colors': SEGMENT_COLORS,
        'total_customers': len(_rfm_data)
    }


@anvil.server.callable
def get_customers_by_segment(segment: str, limit: int = 100):
    """Get customers in a segment with their color."""
    initialize_models()
    
    df = _rfm_data[_rfm_data['segment'] == segment].copy()
    df = df.sort_values('monetary', ascending=False).head(limit)
    
    return {
        'customers': df[['customerName', 'recency', 'frequency', 'monetary', 
                        'profit', 'segment', 'segment_color']].to_dict('records'),
        'segment_color': SEGMENT_COLORS.get(segment, '#666666'),
        'count': len(df)
    }


@anvil.server.callable
def get_rfm_chart_data():
    """Get RFM data for scatter plot with segment colors."""
    initialize_models()
    
    # Sample if too large
    df = _rfm_data if len(_rfm_data) <= 500 else _rfm_data.sample(500, random_state=42)
    
    return {
        'data': [{
            'customerName': row['customerName'],
            'recency': row['recency'],
            'frequency': row['frequency'],
            'monetary': row['monetary'],
            'segment': row['segment'],
            'color': row['segment_color']
        } for _, row in df.iterrows()],
        'legend': [{'segment': s, 'color': c} for s, c in SEGMENT_COLORS.items()]
    }


# =============================================================================
# INVENTORY FORECAST ENDPOINTS
# =============================================================================

@anvil.server.callable
def get_available_brands():
    """Get list of brands for dropdown."""
    initialize_models()
    return sorted(list(_forecaster.brand_data.keys()))


@anvil.server.callable
def get_inventory_forecast(brand: str):
    """Get forecast with color-coded stock status."""
    initialize_models()
    
    try:
        results = _forecaster.train(brand)
        fold = results['fold_results'][-1]
        
        # Calculate stock status for each prediction
        predictions = []
        avg_demand = np.mean(fold['y_val'])
        
        for i, (date, actual, pred) in enumerate(zip(
            fold['dates'], fold['y_val'], fold['y_pred']
        )):
            # Determine stock status based on predicted vs actual
            if pred < avg_demand * 0.2:
                status = 'out_of_stock'
            elif pred < avg_demand * 0.5:
                status = 'low_stock'
            elif pred > avg_demand * 1.5:
                status = 'overstocked'
            else:
                status = 'in_stock'
            
            predictions.append({
                'date': date.isoformat() if hasattr(date, 'isoformat') else str(date),
                'actual': float(actual),
                'predicted': float(pred),
                'status': status,
                'status_color': STOCK_COLORS[status]
            })
        
        # Feature importance
        importance = _forecaster.compute_permutation_importance(results['X'], results['y'])
        
        return {
            'success': True,
            'brand': brand,
            'predictions': predictions,
            'metrics': {
                'mae': round(results['mean_mae'], 2),
                'rmse': round(results['mean_rmse'], 2)
            },
            'feature_importance': importance.to_dict('records'),
            'stock_colors': STOCK_COLORS
        }
        
    except Exception as e:
        return {'success': False, 'error': str(e), 'brand': brand}


@anvil.server.callable
def get_stock_summary():
    """Get overall stock status summary with colors."""
    initialize_models()
    
    # Sample brands for summary
    brands = list(_forecaster.brand_data.keys())[:20]
    
    summary = {'in_stock': 0, 'low_stock': 0, 'out_of_stock': 0, 'overstocked': 0}
    
    for brand in brands:
        df = _forecaster.brand_data[brand]
        recent_qty = df['quantity'].tail(4).mean()
        avg_qty = df['quantity'].mean()
        
        if recent_qty < avg_qty * 0.2:
            summary['out_of_stock'] += 1
        elif recent_qty < avg_qty * 0.5:
            summary['low_stock'] += 1
        elif recent_qty > avg_qty * 1.5:
            summary['overstocked'] += 1
        else:
            summary['in_stock'] += 1
    
    return {
        'summary': [
            {'status': 'In Stock', 'count': summary['in_stock'], 'color': STOCK_COLORS['in_stock']},
            {'status': 'Low Stock', 'count': summary['low_stock'], 'color': STOCK_COLORS['low_stock']},
            {'status': 'Out of Stock', 'count': summary['out_of_stock'], 'color': STOCK_COLORS['out_of_stock']},
            {'status': 'Overstocked', 'count': summary['overstocked'], 'color': STOCK_COLORS['overstocked']}
        ],
        'colors': STOCK_COLORS
    }


# =============================================================================
# DASHBOARD OVERVIEW
# =============================================================================

@anvil.server.callable
def get_dashboard_overview():
    """Get overview stats for main dashboard."""
    initialize_models()
    
    segment_counts = _rfm_data['segment'].value_counts()
    
    return {
        'total_customers': len(_rfm_data),
        'total_revenue': float(_rfm_data['monetary'].sum()),
        'total_profit': float(_rfm_data['profit'].sum()),
        'total_brands': len(_forecaster.brand_data),
        'champions': int(segment_counts.get('Champions', 0)),
        'at_risk': int(segment_counts.get('At Risk', 0)),
        'segment_colors': SEGMENT_COLORS,
        'stock_colors': STOCK_COLORS
    }


# =============================================================================
# MAIN
# =============================================================================

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Anvil Dashboard Server')
    parser.add_argument('uplink_key', nargs='?', help='Anvil Uplink key')
    parser.add_argument('--test', action='store_true', help='Test mode without Anvil')
    parser.add_argument('--data', default='Silverstone.csv', help='Data file path')
    
    args = parser.parse_args()
    
    if args.test:
        print("=" * 60)
        print("TEST MODE")
        print("=" * 60)
        initialize_models(args.data)
        
        print("\n[Testing Customer Segments]")
        distribution = _rfm_data['segment'].value_counts()
        print(f"  Found {len(distribution)} segments:")
        for seg, count in distribution.items():
            color = SEGMENT_COLORS.get(seg, '#666')
            print(f"    {seg}: {count} customers ({color})")
        
        print("\n[Testing Stock Summary]")
        brands = list(_forecaster.brand_data.keys())[:10]
        print(f"  Loaded {len(_forecaster.brand_data)} brands")
        print(f"  Sample: {brands[:5]}")
        
        print("\n" + "=" * 60)
        print("✓ All tests passed!")
        print("=" * 60)
        return
    
    # Use the hardcoded key or command line argument
    uplink_key = args.uplink_key or UPLINK_KEY
    
    initialize_models(args.data)
    
    print(f"\nConnecting to Anvil...")
    anvil.server.connect(uplink_key)
    print("✓ Connected! Server running...")
    print("\nPress Ctrl+C to stop the server.\n")
    
    try:
        anvil.server.wait_forever()
    except KeyboardInterrupt:
        print("\nShutting down...")


if __name__ == '__main__':
    main()

