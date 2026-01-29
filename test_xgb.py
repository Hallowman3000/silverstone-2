
import xgboost as xgb
import numpy as np
try:
    model = xgb.XGBRegressor(objective='count:negative_binomial')
    X = np.array([[1], [2], [3]])
    y = np.array([1, 2, 3])
    model.fit(X, y)
    print("Success")
except Exception as e:
    print(f"Error: {e}")
