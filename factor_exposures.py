"""
Compute factor exposures for each ETF: market beta, size, value, momentum, volatility, macro sensitivities.
"""
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import config

def compute_factor_exposures(etf_returns, market_returns, smb_returns, hml_returns, macro_df, window=60):
    """
    etf_returns: DataFrame with dates as index, ETFs as columns.
    market_returns, smb_returns, hml_returns: Series.
    macro_df: DataFrame with macro levels (will be differenced inside).
    Returns: MultiIndex DataFrame (date, ETF) with factor exposure columns.
    """
    all_dates = etf_returns.index
    # Which macro columns are actually available?
    macro_cols = [c for c in config.MACRO_COLUMNS if c in macro_df.columns]
    factors = ['MKT', 'SMB', 'HML', 'MOM', 'VOL'] + macro_cols
    exposure_dict = {etf: pd.DataFrame(index=all_dates, columns=factors) for etf in etf_returns.columns}
    
    for etf in etf_returns.columns:
        ret = etf_returns[etf].fillna(0)
        # Momentum: 12‑month (252 days) minus 1‑month (21 days) return
        mom = ret.rolling(252).apply(lambda x: (1+x).prod() - 1, raw=False) - ret.rolling(21).apply(lambda x: (1+x).prod() - 1, raw=False)
        # Volatility: 21-day rolling std annualized
        vol = ret.rolling(21).std() * np.sqrt(252)
        # Rolling betas
        for i in range(window, len(all_dates)):
            idx = all_dates[i]
            train_range = slice(i-window, i)
            # Build X matrix
            X_data = {
                'MKT': market_returns.iloc[train_range],
                'SMB': smb_returns.iloc[train_range],
                'HML': hml_returns.iloc[train_range]
            }
            for mc in macro_cols:
                X_data[mc] = macro_df[mc].diff().iloc[train_range]
            X = pd.DataFrame(X_data).dropna()
            y = ret.iloc[train_range].loc[X.index]
            if len(X) < 30:
                continue
            lr = LinearRegression()
            lr.fit(X, y)
            coefs = lr.coef_
            # Store coefficients
            for j, factor in enumerate(factors[:len(coefs)]):  # only those that are present
                exposure_dict[etf].loc[idx, factor] = coefs[j]
        # Store momentum and volatility (same for all dates, but we fill only at the end of window)
        # We'll fill forward
        exposure_dict[etf]['MOM'] = mom
        exposure_dict[etf]['VOL'] = vol
    
    # Combine into one DataFrame with MultiIndex (date, ETF)
    combined = []
    for etf, df_exp in exposure_dict.items():
        df_exp['ETF'] = etf
        combined.append(df_exp.reset_index().rename(columns={'index': 'date'}))
    full = pd.concat(combined).set_index(['date', 'ETF']).sort_index()
    # Fill NaN forward (for momentum/volatility that started later)
    full = full.groupby(level='ETF').ffill()
    return full
