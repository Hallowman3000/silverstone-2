""" Aggregates sales data by brand at weekly frequency and predicts future demand. """

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import TimeSeriesSplit
from sklearn.inspection import permutation_importance
from xgboost import XGBRegressor
from typing import Tuple, Dict, Optional
import warnings

warnings.filterwarnings('ignore')


class InventoryForecaster:
    
    def __init__(self, n_splits: int = 3):

        self.n_splits = n_splits
        self.model: Optional[XGBRegressor] = None
        self.feature_names: list = []
        self.importance_results: Optional[Dict] = None
        self.data: Optional[pd.DataFrame] = None
        self.brand_data: Optional[Dict[str, pd.DataFrame]] = None
        
    def load_data(self, filepath: str, encoding: str = 'latin-1') -> pd.DataFrame:

        print(f"Loading data from {filepath}...")
        
        df = pd.read_csv(filepath, encoding=encoding)
        
        # Parse postingDate
        df['postingDate'] = pd.to_datetime(df['postingDate'], format='%d-%b-%y', errors='coerce')
        
        # Filter out rows with invalid dates
        initial_count = len(df)
        df = df.dropna(subset=['postingDate'])
        print(f"Loaded {len(df)} records (dropped {initial_count - len(df)} with invalid dates)")
        
        # Store data
        self.data = df
        
        return df
    
    def resample_by_brand(self, df: Optional[pd.DataFrame] = None) -> Dict[str, pd.DataFrame]:

        if df is None:
            df = self.data
            
        if df is None:
            raise ValueError("No data loaded. Call load_data() first.")
        
        print("Resampling data by brand (weekly aggregation)...")
        
        brand_data = {}
        
        for brand in df['brand'].unique():
            if pd.isna(brand) or brand == '':
                continue
                
            brand_df = df[df['brand'] == brand].copy()
            
            # Set postingDate as index for resampling
            brand_df = brand_df.set_index('postingDate')
            
            # Resample to weekly (Monday start) and sum quantities
            weekly = brand_df['invoicedQuantity'].resample('W-MON').sum()
            weekly = weekly.to_frame(name='quantity')
            
            # Fill missing weeks with 0
            date_range = pd.date_range(
                start=weekly.index.min(),
                end=weekly.index.max(),
                freq='W-MON'
            )
            weekly = weekly.reindex(date_range, fill_value=0)
            weekly.index.name = 'week'
            
            brand_data[brand] = weekly
            
        print(f"Processed {len(brand_data)} brands")
        self.brand_data = brand_data
        
        return brand_data
    
    def create_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Generate lag features, rolling means, and cyclical month features.
        
        Features created:
        - Lags: 1, 4, 12 weeks
        - Rolling means: 4, 12 weeks
        - Cyclical month: sin and cos encoding
        
        Args:
            df: DataFrame with weekly quantity data
            
        Returns:
            DataFrame with features added (NaN rows dropped)
        """
        df = df.copy()
        
        # Lag features
        df['lag_1'] = df['quantity'].shift(1)
        df['lag_4'] = df['quantity'].shift(4)

        
        # Rolling mean features
        df['rolling_mean_4'] = df['quantity'].shift(1).rolling(window=4).mean()

        
        # Cyclical month features
        df['month'] = df.index.month
        df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)

        
        # Week of year (cyclical)
        df['week_of_year'] = df.index.isocalendar().week.astype(int)
        df['week_sin'] = np.sin(2 * np.pi * df['week_of_year'] / 52)
        df['week_cos'] = np.cos(2 * np.pi * df['week_of_year'] / 52)
        
        # Drop helper columns
        df = df.drop(columns=['month', 'week_of_year'])
        
        # Drop rows with NaN (created by lags and rolling)
        initial_count = len(df)
        df = df.dropna()
        print(f"Features created. Dropped {initial_count - len(df)} rows with NaN values.")
        
        return df
    
    def prepare_data(self, brand: str) -> Tuple[np.ndarray, np.ndarray, pd.DataFrame]:
        """
        Prepare feature matrix and target vector for a specific brand.
        
        Args:
            brand: Brand name to prepare data for
            
        Returns:
            Tuple of (X, y, full_dataframe)
        """
        if self.brand_data is None:
            raise ValueError("No brand data. Call resample_by_brand() first.")
            
        if brand not in self.brand_data:
            raise ValueError(f"Brand '{brand}' not found. Available: {list(self.brand_data.keys())[:10]}...")
        
        df = self.brand_data[brand].copy()
        
        # Outlier Handling: Clip quantity to 5th and 95th percentiles
        q_lower = df['quantity'].quantile(0.05)
        q_upper = df['quantity'].quantile(0.95)
        df['quantity'] = df['quantity'].clip(q_lower, q_upper)
        print(f"  Clipped quantity for {brand}: [{q_lower:.2f}, {q_upper:.2f}]")
        
        df = self.create_features(df)
        
        # Define feature columns
        self.feature_names = [
            'lag_1', 'lag_4',
            'rolling_mean_4',
            'month_sin',
            'week_sin', 'week_cos'
        ]
        
        X = df[self.feature_names].values
        y = df['quantity'].values
        
        return X, y, df
    
    def train(self, brand: str, **xgb_params) -> Dict:

        print(f"\nTraining model for brand: {brand}")
        
        X, y, df = self.prepare_data(brand)
        
        # Default XGBoost parameters
        default_params = {
            'objective': 'reg:tweedie',
            'tweedie_variance_power': 1.5,
            'n_estimators': 100,
            'max_depth': 5,
            'learning_rate': 0.1,
            'random_state': 42,
            'n_jobs': -1
        }
        default_params.update(xgb_params)
        
        # TimeSeriesSplit cross-validation
        
        tscv = TimeSeriesSplit(n_splits=self.n_splits)
        
        cv_scores = []
        fold_results = []
        
        for fold, (train_idx, val_idx) in enumerate(tscv.split(X)):
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]
            
            model = XGBRegressor(**default_params)
            model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
            
            y_pred = model.predict(X_val)
            
            # Ensure non-negative predictions (Poisson outputs can be very small but positive)
            y_pred = np.maximum(y_pred, 0)
            
            # Calculate metrics
            mae = np.mean(np.abs(y_val - y_pred))
            rmse = np.sqrt(np.mean((y_val - y_pred) ** 2))
            
            cv_scores.append({'fold': fold + 1, 'mae': mae, 'rmse': rmse})
            fold_results.append({
                'train_idx': train_idx,
                'val_idx': val_idx,
                'y_val': y_val,
                'y_pred': y_pred,
                'dates': df.index[val_idx]
            })
            
            print(f"  Fold {fold + 1}: MAE={mae:.2f}, RMSE={rmse:.2f}")
        
        # Train final model on all data
        self.model = XGBRegressor(**default_params)
        self.model.fit(X, y, verbose=False)
        
        # Store results
        results = {
            'brand': brand,
            'cv_scores': cv_scores,
            'fold_results': fold_results,
            'mean_mae': np.mean([s['mae'] for s in cv_scores]),
            'mean_rmse': np.mean([s['rmse'] for s in cv_scores]),
            'X': X,
            'y': y,
            'df': df
        }
        
        print(f"\nMean CV MAE: {results['mean_mae']:.2f}")
        print(f"Mean CV RMSE: {results['mean_rmse']:.2f}")
        
        return results
    
    def compute_permutation_importance(
        self, 
        X: np.ndarray, 
        y: np.ndarray, 
        n_repeats: int = 10
    ) -> pd.DataFrame:
        """
        Compute permutation importance to explain feature contributions.
        
        Args:
            X: Feature matrix
            y: Target vector
            n_repeats: Number of permutation repeats
            
        Returns:
            DataFrame with feature importance scores
        """
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")
        
        print("\nComputing Permutation Importance...")
        
        result = permutation_importance(
            self.model, X, y,
            n_repeats=n_repeats,
            random_state=42,
            scoring='neg_mean_absolute_error'
        )
        
        importance_df = pd.DataFrame({
            'feature': self.feature_names,
            'importance_mean': result.importances_mean,
            'importance_std': result.importances_std
        }).sort_values('importance_mean', ascending=False)
        
        self.importance_results = importance_df
        
        print("\nFeature Importance (higher = more important):")
        print(importance_df.to_string(index=False))
        
        return importance_df
    
    def plot_predictions(
        self,
        results: Dict,
        fold: int = -1,
        title: Optional[str] = None,
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """
        Plot actual vs predicted values for a specific fold.
        
        Args:
            results: Training results dictionary
            fold: Fold index (-1 for last fold / holdout)
            title: Plot title (auto-generated if None)
            save_path: Path to save figure (optional)
            
        Returns:
            Matplotlib figure
        """
        fold_data = results['fold_results'][fold]
        
        y_true = fold_data['y_val']
        y_pred = fold_data['y_pred']
        dates = fold_data['dates']
        
        if title is None:
            title = f"{results['brand']} Brand - Actual vs Predicted (Holdout Set)"
        
        fig, ax = plt.subplots(figsize=(12, 6))
        
        ax.plot(dates, y_true, 'b-', linewidth=2, label='Actual ($y_{true}$)', marker='o', markersize=4)
        ax.plot(dates, y_pred, 'r--', linewidth=2, label='Predicted ($y_{pred}$)', marker='x', markersize=4)
        
        ax.fill_between(dates, y_true, y_pred, alpha=0.3, color='gray', label='Prediction Error')
        
        ax.set_xlabel('Week', fontsize=12)
        ax.set_ylabel('Invoiced Quantity', fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.legend(loc='upper right', fontsize=10)
        ax.grid(True, alpha=0.3)
        
        # Rotate x-axis labels
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Plot saved to: {save_path}")
        
        return fig
    
    def plot_feature_importance(self,save_path: Optional[str] = None ) -> plt.Figure:
      
        if self.importance_results is None:
            raise ValueError("Importance not computed. Call compute_permutation_importance() first.")
        
        df = self.importance_results.sort_values('importance_mean', ascending=True)
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(df)))
        
        bars = ax.barh(df['feature'], df['importance_mean'], xerr=df['importance_std'],
                       color=colors, edgecolor='black', linewidth=0.5)
        
        ax.set_xlabel('Permutation Importance (MAE increase)', fontsize=12)
        ax.set_ylabel('Feature', fontsize=12)
        ax.set_title('Feature Importance - Which Lags Drive the Forecast', fontsize=14, fontweight='bold')
        ax.grid(True, axis='x', alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Plot saved to: {save_path}")
        
        return fig


def main():
    """ Blackhawk brand. """
    # Initialize forecaster
    forecaster = InventoryForecaster(n_splits=3)
    
    # Load data
    forecaster.load_data('Silverstone.csv')
    
    # Resample by brand
    forecaster.resample_by_brand()
    
    # Check if BLACKHAWK exists (case sensitivity)
    available_brands = list(forecaster.brand_data.keys())
    blackhawk_brand = None
    
    for brand in available_brands:
        if 'blackhawk' in brand.lower():
            blackhawk_brand = brand
            break
    
    if blackhawk_brand is None:
        print(f"\nBLACKHAWK brand not found. Available brands: {available_brands[:10]}...")
        print("Using first available brand with sufficient data...")
        
        # Find brand with most data
        brand_sizes = {b: len(df) for b, df in forecaster.brand_data.items()}
        blackhawk_brand = max(brand_sizes, key=brand_sizes.get)
    
    print(f"\nUsing brand: {blackhawk_brand}")
    
    # Train model
    results = forecaster.train(blackhawk_brand)
    
    # Compute feature importance
    importance = forecaster.compute_permutation_importance(results['X'], results['y'])
    
    # Plot predictions
    forecaster.plot_predictions(
        results,
        fold=-1,  # Last fold (holdout)
        save_path='blackhawk_predictions.png'
    )
    
    # Plot feature importance
    forecaster.plot_feature_importance(save_path='feature_importance.png')
    
    plt.show()
    
    return forecaster, results


if __name__ == '__main__':
    forecaster, results = main()
