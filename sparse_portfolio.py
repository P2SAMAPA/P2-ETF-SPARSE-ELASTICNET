import numpy as np
import pandas as pd
from sklearn.linear_model import ElasticNet
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

def bic_criterion(y_true, y_pred, n_features, alpha, l1_ratio):
    """Approximate BIC for elastic net."""
    n = len(y_true)
    resid = y_true - y_pred
    rss = np.sum(resid**2)
    df = n_features  # approximate degrees of freedom
    bic = n * np.log(rss/n) + df * np.log(n)
    return bic

def walk_forward_predictions(factor_exposures, forward_returns, train_window=252, forecast_horizon=21, alpha=0.5):
    """
    factor_exposures: DataFrame with MultiIndex (date, ETF) – columns are factors.
    forward_returns: Series with same MultiIndex (date, ETF) – target: future 21d return.
    Returns: DataFrame with predictions and actuals.
    """
    dates = factor_exposures.index.get_level_values('date').unique().sort_values()
    predictions = []
    for i in range(train_window, len(dates)-forecast_horizon):
        train_end = i
        test_start = i
        train_dates = dates[:train_end]
        test_dates = dates[test_start:test_start+forecast_horizon]
        # Train data
        X_train = factor_exposures.loc[pd.IndexSlice[train_dates, :]]
        y_train = forward_returns.loc[pd.IndexSlice[train_dates, :]]
        # Drop NaN
        valid = ~(X_train.isna().any(axis=1) | y_train.isna())
        X_train = X_train[valid]
        y_train = y_train[valid]
        if len(X_train) < 100:
            continue
        # Scale features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_train)
        # Elastic net with BIC-selected lambda
        best_lambda = None
        best_bic = np.inf
        best_model = None
        for lam in np.logspace(-4, 1, 20):
            model = ElasticNet(alpha=lam, l1_ratio=alpha, max_iter=10000)
            model.fit(X_scaled, y_train)
            y_pred = model.predict(X_scaled)
            bic = bic_criterion(y_train.values, y_pred, X_scaled.shape[1], lam, alpha)
            if bic < best_bic:
                best_bic = bic
                best_lambda = lam
                best_model = model
        # Test data
        X_test = factor_exposures.loc[pd.IndexSlice[test_dates, :]]
        y_test = forward_returns.loc[pd.IndexSlice[test_dates, :]]
        valid_test = ~(X_test.isna().any(axis=1) | y_test.isna())
        X_test = X_test[valid_test]
        y_test = y_test[valid_test]
        if len(X_test) == 0:
            continue
        X_test_scaled = scaler.transform(X_test)
        pred = best_model.predict(X_test_scaled)
        # Store predictions
        for idx, etf in enumerate(X_test.index.get_level_values('ETF')):
            predictions.append({
                'date': test_dates[0],  # we are predicting the next period's return
                'ETF': etf,
                'predicted_return': pred[idx],
                'actual_return': y_test.values[idx],
                'non_zero_coeffs': np.sum(np.abs(best_model.coef_) > 1e-6)
            })
    return pd.DataFrame(predictions)

def compute_forward_returns(etf_returns, horizon=21):
    """Compute forward returns (sum of log returns over horizon days)."""
    forward = etf_returns.rolling(horizon).apply(lambda x: np.sum(x), raw=False).shift(-horizon)
    return forward
