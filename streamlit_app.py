import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from datetime import timedelta
import sys

# Add current directory to path for imports
sys.path.insert(0, '.')

# Page config
st.set_page_config(
    page_title="Sales Analytics Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =============================================================================
# COLOR SCHEMES
# =============================================================================

SEGMENT_COLORS = {
    'Whales': '#2ecc71',              # Green - Highest spenders
    'Loyal Customers': '#3498db',     # Blue - High value, frequent
    'Potential Loyalists': '#f39c12', # Orange - Growing customers
    'Promising': '#9b59b6',           # Purple - Medium value
    'At Risk': '#e74c3c',             # Red - Lower engagement
    'Hibernating': '#95a5a6'          # Gray - Lowest engagement
}



# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 1rem;
    }
    .segment-card {
        border-radius: 10px;
        padding: 1rem;
        margin: 0.5rem 0;
    }
    .champion { background: linear-gradient(135deg, #2ecc71 0%, #27ae60 100%); }
    .loyal { background: linear-gradient(135deg, #3498db 0%, #2980b9 100%); }
    .potential { background: linear-gradient(135deg, #f39c12 0%, #e67e22 100%); }
    .atrisk { background: linear-gradient(135deg, #e74c3c 0%, #c0392b 100%); }
    .hibernating { background: linear-gradient(135deg, #95a5a6 0%, #7f8c8d 100%); }
</style>
""", unsafe_allow_html=True)


# =============================================================================
# DATA LOADING
# =============================================================================

@st.cache_data
def load_data():
    """Load Silverstone data"""
    df = pd.read_csv('Silverstone.csv', encoding='latin-1')
    df['postingDate'] = pd.to_datetime(df['postingDate'], format='%d-%b-%y', errors='coerce')
    df = df.dropna(subset=['postingDate', 'customerName'])
    return df


@st.cache_data
def compute_rfm(df):
    """Compute RFM metrics and segment customers"""
    reference_date = df['postingDate'].max() + pd.Timedelta(days=1)
    
    rfm = df.groupby('customerName').agg({
        'postingDate': 'max',
        'entryNo': 'count',
        'salesAmountActual': 'sum',
        'profit': 'sum'
    }).reset_index()
    
    rfm.columns = ['customerName', 'lastPurchaseDate', 'frequency', 'monetary', 'profit']
    rfm['recency'] = (reference_date - rfm['lastPurchaseDate']).dt.days
    
    # Filter valid data
    rfm = rfm.dropna(subset=['recency', 'frequency', 'monetary'])
    rfm = rfm[rfm['monetary'] > 0]
    
    # Outlier Handling: Clip RFM values to 5th and 95th percentiles
    for col in ['recency', 'frequency', 'monetary']:
        lower_limit = rfm[col].quantile(0.05)
        upper_limit = rfm[col].quantile(0.95)
        rfm[col] = rfm[col].clip(lower=lower_limit, upper=upper_limit)
    
    # K-Means clustering
    features = rfm[['recency', 'frequency', 'monetary']].copy()
    features_log = np.log1p(features).fillna(0)
    
    scaler = StandardScaler()
    scaled = scaler.fit_transform(features_log)
    
    kmeans = KMeans(n_clusters=6, random_state=42, n_init=10)
    rfm['cluster'] = kmeans.fit_predict(scaled)
    
    # Assign labels based on monetary value ranking
    cluster_ranks = rfm.groupby('cluster')['monetary'].mean().sort_values(ascending=False)
    labels = ['Whales', 'Loyal Customers', 'Potential Loyalists', 'Promising', 'At Risk', 'Hibernating']
    mapping = {cluster: labels[i] for i, cluster in enumerate(cluster_ranks.index)}
    
    rfm['segment'] = rfm['cluster'].map(mapping)
    rfm['segment_color'] = rfm['segment'].map(SEGMENT_COLORS)
    
    return rfm


@st.cache_data
def get_inventory_data(df):
    """Get inventory data by brand"""
    brand_data = {}
    for brand in df['brand'].dropna().unique():
        brand_df = df[df['brand'] == brand].copy()
        brand_df = brand_df.set_index('postingDate')
        weekly = brand_df['invoicedQuantity'].resample('W-MON').sum()
        
        if len(weekly) > 20:  # Only brands with enough data
            brand_data[brand] = weekly.to_frame(name='quantity')
    
    return brand_data


@st.cache_data
def get_profit_by_category(df):
    """Calculate profit by category"""
    stats = df.groupby('itemCategory').agg({
        'profit': ['sum', 'mean', 'count'],
        'salesAmountActual': 'sum'
    }).round(2)
    stats.columns = ['Total Profit', 'Avg Profit', 'Transactions', 'Total Sales']
    stats['Profit Margin %'] = (stats['Total Profit'] / stats['Total Sales'] * 100).round(2)
    return stats.sort_values('Total Profit', ascending=False)


@st.cache_data
def get_profit_by_brand(df, top_n=10):
    """Calculate profit by brand"""
    stats = df.groupby('brand').agg({
        'profit': ['sum', 'mean', 'count'],
        'salesAmountActual': 'sum'
    }).round(2)
    stats.columns = ['Total Profit', 'Avg Profit', 'Transactions', 'Total Sales']
    stats['Profit Margin %'] = (stats['Total Profit'] / stats['Total Sales'] * 100).round(2)
    return stats.sort_values('Total Profit', ascending=False).head(top_n)


# =============================================================================
# MAIN APP
# =============================================================================

def main():
    st.markdown('<h1 class="main-header">📊 Sales Analytics Dashboard</h1>', unsafe_allow_html=True)
    
    # Sidebar Navigation
    st.sidebar.title("Navigation")
    page = st.sidebar.radio(
        "Select Page",
        options=[
            "Dashboard Overview",
            "Customer Segmentation",
            "Inventory Forecast",
            "Profit Analysis",
            "Customer Lookup"
        ]
    )
    
    st.sidebar.markdown("---")
    
    # Load data
    try:
        df = load_data()
        rfm = compute_rfm(df)
        brand_data = get_inventory_data(df)
    except Exception as e:
        st.error(f"Error loading data: {e}")
        st.info("Ensure 'Silverstone.csv' is in the current directory.")
        return
    
    # Sidebar metrics
    st.sidebar.markdown("### Quick Stats")
    st.sidebar.metric("Total Customers", f"{len(rfm):,}")
    st.sidebar.metric("Total Revenue", f"KES {rfm['monetary'].sum():,.0f}")
    st.sidebar.metric("Total Brands", f"{len(brand_data):,}")
    
    # ==========================================================================
    # PAGE: Dashboard Overview
    # ==========================================================================
    if page == "Dashboard Overview":
        st.header("Executive Overview")
        
        # Key metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            whales = len(rfm[rfm['segment'] == 'Whales'])
            st.metric("🐋 Whales", whales, 
                     help="Top customers - highest spenders with excellent engagement")
        
        with col2:
            at_risk = len(rfm[rfm['segment'] == 'At Risk'])
            st.metric("⚠️ At Risk", at_risk,
                     help="Customers showing declining engagement")
        
        with col3:
            total_profit = df['profit'].sum()
            st.metric("💰 Total Profit", f"KES {total_profit:,.0f}")
        
        with col4:
            avg_order = df['salesAmountActual'].mean()
            st.metric("📦 Avg Order Value", f"KES {avg_order:,.0f}")
        
        st.markdown("---")
        
        # Charts row
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Customer Segments")
            segment_counts = rfm['segment'].value_counts()
            fig = px.pie(
                values=segment_counts.values,
                names=segment_counts.index,
                color=segment_counts.index,
                color_discrete_map=SEGMENT_COLORS,
                hole=0.4
            )
            fig.update_layout(height=350)
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.subheader("Revenue by Segment")
            revenue_by_seg = rfm.groupby('segment')['monetary'].sum().sort_values(ascending=True)
            fig = px.bar(
                x=revenue_by_seg.values,
                y=revenue_by_seg.index,
                orientation='h',
                color=revenue_by_seg.index,
                color_discrete_map=SEGMENT_COLORS
            )
            fig.update_layout(height=350, showlegend=False,
                            xaxis_title="Revenue (KES)", yaxis_title="")
            st.plotly_chart(fig, use_container_width=True)
        
        # Segment legend
        st.subheader("Segment Legend")
        cols = st.columns(6)
        for i, (seg, color) in enumerate(SEGMENT_COLORS.items()):
            with cols[i]:
                count = len(rfm[rfm['segment'] == seg])
                st.markdown(f"""
                <div style="background-color: {color}; color: white; padding: 10px; 
                           border-radius: 8px; text-align: center;">
                    <strong>{seg}</strong><br>
                    {count} customers
                </div>
                """, unsafe_allow_html=True)
    
    # ==========================================================================
    # PAGE: Customer Segmentation
    # ==========================================================================
    elif page == "Customer Segmentation":
        st.header("Customer Segmentation Analysis")
        
        # Summary metrics
        col1, col2, col3, col4, col5, col6 = st.columns(6)
        segments = ['Whales', 'Loyal Customers', 'Potential Loyalists', 'Promising', 'At Risk', 'Hibernating']
        
        for i, seg in enumerate(segments):
            with [col1, col2, col3, col4, col5, col6][i]:
                count = len(rfm[rfm['segment'] == seg])
                color = SEGMENT_COLORS[seg]
                st.markdown(f"""
                <div style="background-color: {color}; color: white; padding: 15px; 
                           border-radius: 10px; text-align: center;">
                    <h3 style="margin: 0; color: white;">{count}</h3>
                    <small>{seg}</small>
                </div>
                """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        # Scatter plot
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.subheader("RFM Scatter Plot")
            x_axis = st.selectbox("X-Axis", ['recency', 'frequency', 'monetary'], index=0)
            y_axis = st.selectbox("Y-Axis", ['recency', 'frequency', 'monetary'], index=2)
            
            fig = px.scatter(
                rfm, x=x_axis, y=y_axis, color='segment',
                color_discrete_map=SEGMENT_COLORS,
                hover_data=['customerName', 'recency', 'frequency', 'monetary'],
                opacity=0.7
            )
            fig.update_layout(height=500)
            fig.update_traces(marker=dict(size=8))
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.subheader("Segment Stats")
            selected_segment = st.selectbox("Select Segment", segments)
            seg_data = rfm[rfm['segment'] == selected_segment]
            
            st.metric("Customer Count", len(seg_data))
            st.metric("Avg Recency", f"{seg_data['recency'].mean():.0f} days")
            st.metric("Avg Frequency", f"{seg_data['frequency'].mean():.1f} orders")
            st.metric("Avg Monetary", f"KES {seg_data['monetary'].mean():,.0f}")
            st.metric("Total Revenue", f"KES {seg_data['monetary'].sum():,.0f}")
        
        # Segment table
        st.subheader("Segment Summary Table")
        summary = rfm.groupby('segment').agg({
            'recency': 'mean',
            'frequency': 'mean',
            'monetary': ['mean', 'sum'],
            'profit': 'sum',
            'customerName': 'count'
        }).round(2)
        summary.columns = ['Avg Recency', 'Avg Frequency', 'Avg Monetary', 
                          'Total Revenue', 'Total Profit', 'Count']
        st.dataframe(summary, use_container_width=True)

        # Cluster Feature Analysis
        st.markdown("---")
        st.subheader("Cluster Feature Analysis")
        
        col1, col2 = st.columns([1.5, 1])
        
        with col1:
            st.markdown("##### Segment Profiles (Cluster Centroids)")
            st.write("Average feature values for each segment. Darker colors indicate higher values.")
            
            # Select features for the "confusion matrix" style view
            feature_view = summary[['Avg Recency', 'Avg Frequency', 'Avg Monetary']].copy()
            
            # Display with gradient
            st.dataframe(
                feature_view.style.background_gradient(cmap='Blues', axis=0),
                use_container_width=True
            )
            
        with col2:
            st.markdown("##### Feature Lineage")
            st.info("""
            **How features are derived from data:**
            
            1. **Recency** (Days)
               * *Source*: `postingDate`
               * *Impact*: Measures customer engagement. Lower is better (recent).
            
            2. **Frequency** (Count)
               * *Source*: `entryNo` (Transaction Count)
               * *Impact*: Measures activity level. Higher is better.
            
            3. **Monetary** (Value)
               * *Source*: `salesAmountActual`
               * *Impact*: Measures spending power. Higher is better.
            """)
    
    # ==========================================================================
    # PAGE: Inventory Forecast
    # ==========================================================================
    elif page == "Inventory Forecast":
        st.header("Inventory Forecast Analysis")
        
        # Brand selection
        brands = sorted(brand_data.keys())
        selected_brand = st.selectbox("Select Brand", brands)
        
        if selected_brand:
            brand_df = brand_data[selected_brand]
            
            col1, col2 = st.columns([2, 1])
            
            with col1:
                st.subheader(f"{selected_brand} - Weekly Demand")
                
                st.subheader(f"{selected_brand} - Weekly Demand")
                
                # Calculate Data Accuracy Metrics
                total_weeks = len(brand_df)
                active_weeks = len(brand_df[brand_df['quantity'] > 0])
                density = (active_weeks / total_weeks) * 100
                
                # Gap Analysis
                # Create a boolean series where True = Zero Sale
                is_zero = brand_df['quantity'] == 0
                # Group consecutive zeros
                gaps = is_zero.astype(int).groupby(is_zero.ne(is_zero.shift()).cumsum()).sum()
                # Filter only the groups that were actually zeros
                zero_gaps = gaps[is_zero.groupby(is_zero.ne(is_zero.shift()).cumsum()).first()]
                
                avg_gap = zero_gaps.mean() if not zero_gaps.empty else 0
                max_gap = zero_gaps.max() if not zero_gaps.empty else 0
                
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=brand_df.index,
                    y=brand_df['quantity'],
                    mode='lines+markers',
                    line=dict(color='#667eea', width=2),
                    marker=dict(
                        color='#764ba2',
                        size=8
                    ),
                    name='Quantity'
                ))
                
                # Add average line
                avg_qty = brand_df['quantity'].mean()
                fig.add_hline(y=avg_qty, line_dash="dash", line_color="gray",
                             annotation_text=f"Avg: {avg_qty:.0f}")
                
                fig.update_layout(
                    height=400,
                    xaxis_title="Week",
                    yaxis_title="Quantity Sold"
                )
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                st.subheader("Data Reliability")
                
                st.metric("Data Density", f"{density:.1f}%", 
                         help="Percentage of weeks with > 0 sales")
                
                st.metric("Avg Gap Length", f"{avg_gap:.1f} weeks",
                         help="Average consecutive weeks with 0 sales",
                         delta_color="inverse")
                         
                st.markdown("---")
                st.subheader("Brand Stats")
                st.metric("Total Weeks", total_weeks)
                st.metric("Avg Weekly Demand", f"{avg_qty:.0f}")
                st.metric("Max Demand", f"{brand_df['quantity'].max():.0f}")
        
        # Data Reliability Summary
        st.markdown("---")
        st.subheader("Overall Data Reliability")
        
        reliability_data = []
        
        for brand, bdf in brand_data.items():
            t_weeks = len(bdf)
            a_weeks = len(bdf[bdf['quantity'] > 0])
            dens = (a_weeks / t_weeks) * 100
            
            # Gap calc for table
            iz = bdf['quantity'] == 0
            gps = iz.astype(int).groupby(iz.ne(iz.shift()).cumsum()).sum()
            zgps = gps[iz.groupby(iz.ne(iz.shift()).cumsum()).first()]
            a_gap = zgps.mean() if not zgps.empty else 0
            
            reliability_data.append({
                'Brand': brand,
                'Data Density': dens,
                'Avg Gap (Weeks)': a_gap,
                'Total Weeks': t_weeks
            })
            
        rel_df = pd.DataFrame(reliability_data).sort_values('Data Density', ascending=False)
        
        col1, col2 = st.columns(2)
        
        with col1:
             st.markdown("##### Most Consistent Brands (High Density)")
             st.dataframe(
                 rel_df.head(10).style.format({
                     'Data Density': '{:.1f}%',
                     'Avg Gap (Weeks)': '{:.1f}'
                 }).background_gradient(subset=['Data Density'], cmap='Greens'),
                 use_container_width=True
             )
             
        with col2:
             st.markdown("##### Variable Traffic Brands (Low Density)")
             st.dataframe(
                 rel_df.tail(10).sort_values('Data Density', ascending=True).style.format({
                     'Data Density': '{:.1f}%',
                     'Avg Gap (Weeks)': '{:.1f}'
                 }).background_gradient(subset=['Data Density'], cmap='Reds_r'),
                 use_container_width=True
             )
    
    # ==========================================================================
    # PAGE: Profit Analysis
    # ==========================================================================
    elif page == "Profit Analysis":
        st.header("Profit Analysis")
        
        # Profit trends
        st.subheader("Profit Trends")
        
        df_profit = df.set_index('postingDate')
        monthly_profit = df_profit['profit'].resample('M').sum()
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=monthly_profit.index,
            y=monthly_profit.values,
            mode='lines+markers',
            fill='tozeroy',
            line=dict(color='#667eea', width=3),
            marker=dict(size=8)
        ))
        fig.update_layout(
            height=350,
            xaxis_title="Month",
            yaxis_title="Profit (KES)"
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Profit by category and brand
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Profit by Category")
            profit_cat = get_profit_by_category(df)
            fig = px.bar(
                x=profit_cat.index[:10],
                y=profit_cat['Total Profit'][:10],
                color=profit_cat['Total Profit'][:10],
                color_continuous_scale='Viridis'
            )
            fig.update_layout(height=400, xaxis_title="Category", yaxis_title="Profit (KES)")
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.subheader("Top 10 Brands by Profit")
            profit_brand = get_profit_by_brand(df, 10)
            fig = px.bar(
                x=profit_brand['Total Profit'],
                y=profit_brand.index,
                orientation='h',
                color=profit_brand['Total Profit'],
                color_continuous_scale='Viridis'
            )
            fig.update_layout(height=400, xaxis_title="Profit (KES)", yaxis_title="")
            st.plotly_chart(fig, use_container_width=True)
        
        # Profit margin table
        st.subheader("Profit Margins by Category")
        st.dataframe(get_profit_by_category(df), use_container_width=True)
    
    # ==========================================================================
    # PAGE: Customer Lookup
    # ==========================================================================
    elif page == "Customer Lookup":
        st.header("Customer Lookup")
        
        # Search
        search = st.text_input("Search Customer", placeholder="Enter customer name...")
        
        if search:
            results = rfm[rfm['customerName'].str.contains(search, case=False, na=False)]
            
            if len(results) > 0:
                st.success(f"Found {len(results)} customers")
                
                # Show results with color coding
                for _, row in results.iterrows():
                    color = SEGMENT_COLORS.get(row['segment'], '#666')
                    st.markdown(f"""
                    <div style="background-color: {color}; color: white; padding: 15px; 
                               border-radius: 10px; margin: 10px 0;">
                        <h4 style="margin: 0; color: white;">{row['customerName']}</h4>
                        <p style="margin: 5px 0;">
                            <strong>Segment:</strong> {row['segment']} | 
                            <strong>Recency:</strong> {row['recency']} days | 
                            <strong>Frequency:</strong> {row['frequency']} | 
                            <strong>Monetary:</strong> KES {row['monetary']:,.0f}
                        </p>
                    </div>
                    """, unsafe_allow_html=True)
            else:
                st.warning("No customers found.")
        
        # Filter by segment
        st.markdown("---")
        st.subheader("Browse by Segment")
        
        segment_filter = st.selectbox("Select Segment to View", list(SEGMENT_COLORS.keys()))
        segment_data = rfm[rfm['segment'] == segment_filter].sort_values('monetary', ascending=False)
        
        st.markdown(f"**Showing top 50 customers in {segment_filter}**")
        st.dataframe(
            segment_data[['customerName', 'recency', 'frequency', 'monetary', 'profit']].head(50),
            use_container_width=True,
            height=400
        )
        
        # Download
        st.download_button(
            "📥 Download All Customer Data",
            rfm.to_csv(index=False),
            "customer_segments.csv",
            "text/csv"
        )


if __name__ == "__main__":
    main()
