# 🧠 EvoRisk: Autonomously Discovered Regime-Adaptive Resilience-Aware Financial Metric

**EvoRisk** is an autonomously discovered, regime-adaptive, and resilience-aware performance metric that generalizes classical ratios such as **Sharpe** and **Calmar**.  It models **volatility asymmetry**, **tail risk**, and **drawdown persistence** under non-stationary market regimes — providing a robust signal for **asset selection** and **portfolio optimization**. Developed within an **AlphaEvolve-style AlphaSharpe framework**, EvoRisk was evolved autonomously through iterative generation, mutation, scoring, and selection — merging LLM creativity with empirical validation.

---

## 🚀 Features

- **Robust Financial Metrics**
  - `alpha_sharpe()`: volatility-aware Sharpe-like ratio using downside and forecasted volatility.
  - `robust_calmar()`: volatility-adaptive, jump-aware, entropy-regularized drawdown metric (EvoRisk core).

- **Portfolio Optimization**
  - Inverse-covariance Bayesian projection for signal-weighted allocations.
  - Combines signal strength with covariance orthogonalization.

- **Evaluation Framework**
  - Time-series cross-validation for 15-year equity dataset.
  - Out-of-sample evaluation of Sharpe, Calmar, and mean log-returns.
  - Dual evaluation modes: *selection* (ranking) and *optimization* (allocation).

- **Visualization**
  - Comparative Sharpe, Calmar, and mean-return curves across selection ratios.
