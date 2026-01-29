"""
Profit Predictor Module

XGBoost-based profit prediction with feature engineering.
Predicts profit based on product attributes and transaction details.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBRegressor
from typing import Tuple, Dict, Optional, List
import warnings

warnings.filterwarnings('ignore')


class ProfitPredictor:
    
    def __init__(self, n_splits: int = 3):
        """
        Initialize the ProfitPredictor.
        
        Args:
            n_splits: Number of splits for TimeSeriesSplit validation
        """
        self.n_splits = n_splits
        self.model: Optional[XGBRegressor] = None
        self.label_encoders: Dict[str, LabelEncoder] = {}
        self.feature_names: List[str] = []
        self.data: Optional[pd.DataFrame] = None
        self.feature_importance: Optional[pd.DataFrame] = None
        
    def load_data(self, filepath: str, encoding: str = 'latin-1') -> pd.DataFrame:

        print(f"Loading data from {filepath}...")
        
        df = pd.read_csv(filepath, encoding=encoding)
        
        # Parse postingDate
        df['postingDate'] = pd.to_datetime(df['postingDate'], format='%d-%b-%y', errors='coerce')
        
        # Filter out rows with invalid dates or missing key columns
        initial_count = len(df)
        df = df.dropna(subset=['postingDate', 'profit', 'brand', 'itemCategory'])
        print(f"Loaded {len(df)} records (dropped {initial_count - len(df)} with missing data)")
        
        # Sort by date
        df = df.sort_values('postingDate').reset_index(drop=True)
        
        self.data = df
        return df
    
    def prepare_features(self, df: Optional[pd.DataFrame] = None) -> Tuple[np.ndarray, np.ndarray, pd.DataFrame]:

        if df is None:
            df = self.data
            
        if df is None:
            raise ValueError("No data loaded. Call load_data() first.")
        
        df = df.copy()
        
        # Categorical features to encode
        categorical_cols = ['brand', 'itemCategory', 'itemSubcategory', 'salesPerson']
        
        # Encode categorical features
        for col in categorical_cols:
            if col not in self.label_encoders:
                self.label_encoders[col] = LabelEncoder()
                df[f'{col}_encoded'] = self.label_encoders[col].fit_transform(df[col].fillna('Unknown').astype(str))
            else:
                # Handle unseen categories
                df[col] = df[col].fillna('Unknown').astype(str)
                known_classes = set(self.label_encoders[col].classes_)
                df[col] = df[col].apply(lambda x: x if x in known_classes else 'Unknown')
                df[f'{col}_encoded'] = self.label_encoders[col].transform(df[col])
        
        # Define feature columns
        self.feature_names = [
            'brand_encoded',
            'itemCategory_encoded', 
            'itemSubcategory_encoded',
            'salesPerson_encoded',
            'invoicedQuantity',
            'unitPrice',
            'costPerUnit',
            'discountAmount'
        ]
        
        # Outlier Handling: Clip numerical features to 5th and 95th percentiles
        numerical_cols = ['invoicedQuantity', 'unitPrice', 'costPerUnit', 'discountAmount']
        for col in numerical_cols:
            lower = df[col].quantile(0.05)
            upper = df[col].quantile(0.95)
            df[col] = df[col].clip(lower, upper)
            print(f"  Clipped {col}: [{lower:.2f}, {upper:.2f}]")
        
        # Prepare X and y
        X = df[self.feature_names].fillna(0).values
        y = df['profit'].values
        
        return X, y, df
    
    def train(self, **xgb_params) -> Dict:

        print("\nTraining Profit Prediction Model...")
        
        X, y, df = self.prepare_features()
        
        # Default XGBoost parameters
        default_params = {
            'objective': 'reg:squarederror',
            'n_estimators': 100,
            'max_depth': 6,
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
            
            # Calculate metrics
            mae = np.mean(np.abs(y_val - y_pred))
            rmse = np.sqrt(np.mean((y_val - y_pred) ** 2))
            mape = np.mean(np.abs((y_val - y_pred) / (y_val + 1e-8))) * 100
            
            cv_scores.append({'fold': fold + 1, 'mae': mae, 'rmse': rmse, 'mape': mape})
            fold_results.append({
                'train_idx': train_idx,
                'val_idx': val_idx,
                'y_val': y_val,
                'y_pred': y_pred,
                'dates': df.iloc[val_idx]['postingDate'].values
            })
            
            print(f"  Fold {fold + 1}: MAE={mae:.2f}, RMSE={rmse:.2f}, MAPE={mape:.1f}%")
        
        # Train final model on all data
        self.model = XGBRegressor(**default_params)
        self.model.fit(X, y, verbose=False)
        
        # Calculate feature importance
        self.feature_importance = pd.DataFrame({
            'feature': self.feature_names,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        results = {
            'cv_scores': cv_scores,
            'fold_results': fold_results,
            'mean_mae': np.mean([s['mae'] for s in cv_scores]),
            'mean_rmse': np.mean([s['rmse'] for s in cv_scores]),
            'mean_mape': np.mean([s['mape'] for s in cv_scores]),
            'X': X,
            'y': y,
            'df': df
        }
        
        print(f"\nMean CV MAE: {results['mean_mae']:.2f}")
        print(f"Mean CV RMSE: {results['mean_rmse']:.2f}")
        print(f"Mean CV MAPE: {results['mean_mape']:.1f}%")
        
        return results
    
    def predict(self, data: Dict) -> float:

        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")
        
        # Prepare features
        features = []
        for col in ['brand', 'itemCategory', 'itemSubcategory', 'salesPerson']:
            val = data.get(col, 'Unknown')
            if col in self.label_encoders:
                known_classes = set(self.label_encoders[col].classes_)
                val = val if val in known_classes else 'Unknown'
                encoded = self.label_encoders[col].transform([val])[0]
            else:
                encoded = 0
            features.append(encoded)
        
        # Add numerical features
        features.extend([
            data.get('invoicedQuantity', 0),
            data.get('unitPrice', 0),
            data.get('costPerUnit', 0),
            data.get('discountAmount', 0)
        ])
        
        return float(self.model.predict([features])[0])
    
    def get_profit_drivers(self) -> List[Dict]:
  
        if self.feature_importance is None:
            raise ValueError("Model not trained. Call train() first.")
        
        return self.feature_importance.to_dict('records')
    
    def get_profit_by_category(self) -> pd.DataFrame:

        if self.data is None:
            raise ValueError("No data loaded. Call load_data() first.")
        
        stats = self.data.groupby('itemCategory').agg({
            'profit': ['sum', 'mean', 'count'],
            'salesAmountActual': 'sum'
        }).round(2)
        
        stats.columns = ['Total Profit', 'Avg Profit', 'Transactions', 'Total Sales']
        stats['Profit Margin %'] = (stats['Total Profit'] / stats['Total Sales'] * 100).round(2)
        
        return stats.sort_values('Total Profit', ascending=False)
    
    def get_profit_by_brand(self, top_n: int = 10) -> pd.DataFrame:

        if self.data is None:
            raise ValueError("No data loaded. Call load_data() first.")
        
        stats = self.data.groupby('brand').agg({
            'profit': ['sum', 'mean', 'count'],
            'salesAmountActual': 'sum'
        }).round(2)
        
        stats.columns = ['Total Profit', 'Avg Profit', 'Transactions', 'Total Sales']
        stats['Profit Margin %'] = (stats['Total Profit'] / stats['Total Sales'] * 100).round(2)
        
        return stats.sort_values('Total Profit', ascending=False).head(top_n)
    
    def get_profit_trends(self, freq: str = 'M') -> pd.DataFrame:

        if self.data is None:
            raise ValueError("No data loaded. Call load_data() first.")
        
        df = self.data.set_index('postingDate')
        
        trends = df['profit'].resample(freq).agg(['sum', 'mean', 'count'])
        trends.columns = ['Total Profit', 'Avg Profit', 'Transactions']
        
        # Calculate cumulative profit
        trends['Cumulative Profit'] = trends['Total Profit'].cumsum()
        
        return trends


def main():
 
    predictor = ProfitPredictor(n_splits=3)
    
    # Load data
    predictor.load_data('Silverstone.csv')
    
    # Train model
    results = predictor.train()
    
    # Display feature importance
    print("\n" + "=" * 60)
    print("PROFIT DRIVERS (Feature Importance)")
    print("=" * 60)
    for item in predictor.get_profit_drivers():
        print(f"  {item['feature']:30} | {item['importance']:.4f}")
    
    # Display profit by category
    print("\n" + "=" * 60)
    print("PROFIT BY CATEGORY")
    print("=" * 60)
    print(predictor.get_profit_by_category())
    
    # Display profit by brand
    print("\n" + "=" * 60)
    print("TOP 10 BRANDS BY PROFIT")
    print("=" * 60)
    print(predictor.get_profit_by_brand(10))
    
    return predictor, results


if __name__ == '__main__':
    predictor, results = main()
