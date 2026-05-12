# Sparse Portfolio via Elastic Net

Predicts 21‑day forward returns using factor exposures (market, size, value, momentum, volatility, macro). Elastic net with L1+L2 penalty induces sparsity: only ETFs with non‑zero coefficients survive.

- Walk‑forward: train 252 days, test 21 days
- Model: Elastic net (α=0.5), lambda selected via BIC
- Output: top 3 ETFs per universe by predicted return, plus count of non‑zero coefficients
- Runs daily via GitHub Actions

## Local execution
```bash
pip install -r requirements.txt
export HF_TOKEN=your_token
python trainer.py
streamlit run streamlit_app.py
