import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import config

def compute_factor_exposures(etf_returns, market_returns, macro_df, window=60):
    """
    For each ETF, compute:
    - momentum (21-day total return)
    - volatility (21-day std dev annualized)
    - market beta (rolling 60-day regression on market returns)
    - macro betas (rolling 60-day regression on macro changes, if macro_df not empty)
    Returns MultiIndex DataFrame (date, ETF) with columns: MOM, VOL, BETA, and macro betas.
    """
    all_dates = etf_returns.index
    # Macro columns that exist in macro_df
    macro_cols = [c for c in config.MACRO_COLUMNS if c in macro_df.columns]
    # Output columns
    out_cols = ['MOM', 'VOL', 'BETA'] + macro_cols
    # Prepare empty storage
    data = []
    
    for etf in etf_returns.columns:
        ret = etf_returns[etf].fillna(0)
        # Momentum: 21-day cumulative return
        mom = ret.rolling(21).apply(lambda x: (1+x).prod() - 1, raw=False)
        # Volatility: 21-day std annualized
        vol = ret.rolling(21).std() * np.sqrt(252)
        # Rolling betas to market
        beta = pd.Series(index=ret.index, dtype=float)
        # For each day with at least `window` observations, compute beta
        for i in range(window, len(ret)):
            y = ret.iloc[i-window:i]
            X = market_returns.iloc[i-window:i]
            valid = ~(y.isna() | X.isna())
            if valid.sum() < 30:
                continue
            lr = LinearRegression()
            lr.fit(X[valid].values.reshape(-1,1), y[valid].values)
            beta.iloc[i] = lr.coef_[0]
        # Macro betas
        macro_betas = {}
        for mc in macro_cols:
            macro_series = macro_df[mc].diff().fillna(0)
            mb = pd.Series(index=ret.index, dtype=float)
            for i in range(window, len(ret)):
                y = ret.iloc[i-window:i]
                X = macro_series.iloc[i-window:i]
                valid = ~(y.isna() | X.isna())
                if valid.sum() < 30:
                    continue
                lr = LinearRegression()
                lr.fit(X[valid].values.reshape(-1,1), y[valid].values)
                mb.iloc[i] = lr.coef_[0]
            macro_betas[mc] = mb
        # Combine into DataFrame for this ETF
        df_etf = pd.DataFrame(index=ret.index)
        df_etf['MOM'] = mom
        df_etf['VOL'] = vol
        df_etf['BETA'] = beta
        for mc in macro_cols:
            df_etf[mc] = macro_betas[mc]
        df_etf['ETF'] = etf
        data.append(df_etf.reset_index().rename(columns={'index':'date'}))
    
    full = pd.concat(data).set_index(['date', 'ETF']).sort_index()
    # Forward fill any NaN (e.g., first few rows)
    full = full.groupby(level='ETF').ffill()
    return full
