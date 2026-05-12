import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import config

def compute_factor_exposures(etf_returns, market_returns, smb_returns, hml_returns, macro_df, window=60):
    """
    For each ETF, compute rolling betas to market, SMB, HML, and macro factors.
    Also compute momentum (12‑1) and volatility as separate factors.
    Returns a DataFrame of factor exposures (each column = one factor, multi-index?).
    We'll output a 3D structure: for each date, for each ETF, factor values.
    Simpler: For each ETF, compute a DataFrame of factor exposures over time.
    """
    all_dates = etf_returns.index
    factors = ['MKT', 'SMB', 'HML', 'MOM', 'VOL', 'VIX', 'DXY', 'TERM', 'CREDIT']
    exposure_dict = {etf: pd.DataFrame(index=all_dates, columns=factors) for etf in etf_returns.columns}
    
    for etf in etf_returns.columns:
        ret = etf_returns[etf].fillna(0)
        # Momentum: 12‑month (252 days) minus 1‑month (21 days) return
        mom = ret.rolling(252).apply(lambda x: (1+x).prod() - 1, raw=False) - ret.rolling(21).apply(lambda x: (1+x).prod() - 1, raw=False)
        # Volatility: 21-day rolling std
        vol = ret.rolling(21).std() * np.sqrt(252)
        # Rolling betas
        for i in range(window, len(all_dates)):
            idx = all_dates[i]
            train_range = slice(i-window, i)
            X = pd.DataFrame({
                'MKT': market_returns.iloc[train_range],
                'SMB': smb_returns.iloc[train_range],
                'HML': hml_returns.iloc[train_range],
                'VIX': macro_df['VIX'].diff().iloc[train_range],
                'DXY': macro_df['DXY'].diff().iloc[train_range],
                'TERM': macro_df['T10Y2Y'].diff().iloc[train_range],
                'CREDIT': macro_df['HY_SPREAD'].diff().iloc[train_range]
            }).dropna()
            y = ret.iloc[train_range].loc[X.index]
            if len(X) < 30:
                continue
            lr = LinearRegression()
            lr.fit(X, y)
            coefs = lr.coef_
            exposure_dict[etf].loc[idx, ['MKT','SMB','HML','VIX','DXY','TERM','CREDIT']] = coefs
        exposure_dict[etf]['MOM'] = mom
        exposure_dict[etf]['VOL'] = vol
    # Combine into one DataFrame with MultiIndex (date, ETF)
    combined = []
    for etf, df_exp in exposure_dict.items():
        df_exp['ETF'] = etf
        combined.append(df_exp.reset_index().rename(columns={'index':'date'}))
    full = pd.concat(combined).set_index(['date', 'ETF']).sort_index()
    return full
