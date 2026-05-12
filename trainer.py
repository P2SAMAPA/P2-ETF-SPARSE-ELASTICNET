import pandas as pd
import numpy as np
from pathlib import Path
import json
from datetime import datetime
import config
import data_manager
from factor_exposures import compute_factor_exposures
from sparse_portfolio import compute_forward_returns, walk_forward_predictions

def main():
    if not config.HF_TOKEN:
        print("HF_TOKEN not set")
        return

    df = data_manager.load_master_data()
    all_results = {}
    today = datetime.now().strftime("%Y-%m-%d")

    for universe_name, tickers in config.UNIVERSES.items():
        print(f"\n=== Universe: {universe_name} ===")
        returns = data_manager.prepare_returns_matrix(df, tickers)
        if returns.empty or len(returns) < config.TRAIN_WINDOW + config.FORECAST_HORIZON + 10:
            print("  Insufficient data")
            all_results[universe_name] = {"top_etfs": []}
            continue

        # Create market, size, value proxies from available ETFs
        # Market: SPY if available, else first ETF
        market_proxy = 'SPY' if 'SPY' in tickers else tickers[0]
        market_returns = returns[market_proxy]
        # SMB: small minus big – use IWM vs SPY if both exist
        if 'IWM' in tickers and market_proxy != 'IWM':
            smb_returns = returns['IWM'] - market_returns
        else:
            smb_returns = pd.Series(0, index=returns.index)
        # HML: value minus growth – use IWD vs IWF if both exist
        if 'IWD' in tickers and 'IWF' in tickers:
            hml_returns = returns['IWD'] - returns['IWF']
        else:
            hml_returns = pd.Series(0, index=returns.index)

        # Macro data: use only columns that exist in the dataset
        available_macro = [c for c in config.MACRO_COLUMNS if c in df.columns]
        macro_df = df[available_macro] if available_macro else pd.DataFrame(index=df.index)

        # Compute factor exposures for all ETFs in this universe
        factor_exp = compute_factor_exposures(returns, market_returns, smb_returns, hml_returns, macro_df, window=60)
        # Compute forward returns (21d)
        forward_ret = compute_forward_returns(returns, horizon=config.FORECAST_HORIZON)
        # Align indices
        forward_ret_long = forward_ret.stack().rename('forward_return')
        forward_ret_long.index.names = ['date', 'ETF']
        # Merge
        merged = factor_exp.join(forward_ret_long, how='inner').dropna()
        if merged.empty:
            print("  No valid factor exposures")
            continue

        # Walk‑forward predictions
        predictions_df = walk_forward_predictions(
            merged.drop(columns='forward_return'), 
            merged['forward_return'],
            train_window=config.TRAIN_WINDOW,
            forecast_horizon=config.FORECAST_HORIZON,
            alpha=config.ELASTIC_NET_ALPHA
        )
        if predictions_df.empty:
            print("  No predictions")
            continue

        # For the most recent date, get predictions
        latest_date = predictions_df['date'].max()
        latest_preds = predictions_df[predictions_df['date'] == latest_date]
        latest_preds = latest_preds.sort_values('predicted_return', ascending=False)
        top3 = latest_preds.head(config.TOP_N)
        top_etfs = []
        for _, row in top3.iterrows():
            top_etfs.append({
                'ticker': row['ETF'],
                'pred_return': float(row['predicted_return']),
                'non_zero_coeffs': int(row['non_zero_coeffs'])
            })
        print(f"  Top 3 ETFs (pred return, non‑zero coefficients):")
        for etf in top_etfs:
            print(f"    {etf['ticker']}: pred={etf['pred_return']:.4f}, non-zero={etf['non_zero_coeffs']}")
        all_results[universe_name] = {
            "top_etfs": top_etfs,
            "run_date": today
        }

    # Save results
    Path("results").mkdir(exist_ok=True)
    local_path = Path(f"results/sparse_elasticnet_{today}.json")
    with open(local_path, "w") as f:
        json.dump({"run_date": today, "universes": all_results}, f, indent=2)

    import push_results
    push_results.push_daily_result(local_path)
    print("\n=== Sparse Portfolio via Elastic Net complete ===")

if __name__ == "__main__":
    main()
