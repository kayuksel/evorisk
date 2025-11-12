import _pickle as cPickle
import numpy as np
import torch
import matplotlib.pyplot as plt
import math
import pandas as pd

def alpha_sharpe(
    log_returns: torch.Tensor, risk_free: 0.0, eps: 1.5e-5, dr: 2.0, fv: 1.33, window: int = 3):
    log_returns = log_returns.unsqueeze(0) if log_returns.ndim == 1 else log_returns
    # Calculate mean log excess return (expected log excess return) and standard deviation of log returns
    mean_log_excess_return = log_returns.mean(dim=-1) - risk_free
    std_log_returns = log_returns.std(dim=-1, unbiased=False)
    # Downside Risk (DR) calculation
    negative_returns = log_returns[log_returns < 0]
    downside = dr * (
        negative_returns.std(dim=-1, unbiased=False) +
        (negative_returns.numel() ** 0.5) * std_log_returns
    ) / (negative_returns.numel() + eps)
    # Forecasted Volatility (V) calculation
    volatility = fv * log_returns[:, -log_returns.shape[-1] // window:].std(dim=-1, unbiased=False).sqrt()
    return mean_log_excess_return.exp() / ((std_log_returns.pow(2) + eps).sqrt() + downside + volatility)

def evorisk_metric(log_returns: torch.Tensor, risk_free_rate: float = 0.0) -> torch.Tensor:
    device, dtype = log_returns.device, log_returns.dtype
    N, T = log_returns.shape
    lr = log_returns - risk_free_rate

    # dynamic window sizes based on last realized volatility
    realized_std_last = lr[:, -min(60, T):].std(1, unbiased=False) + 1e-12
    sw = torch.clamp((1 / realized_std_last.mean()).round().long(), min=5, max=15)
    lw = torch.clamp((1 / realized_std_last.mean()).round().long(), min=60, max=120)
    sw = min(sw.item(), T)
    lw = min(lw.item(), T)

    # winsorized volatility
    mid_sw = lr[:, -sw:].median(1, keepdim=True)[0]
    abs_diff = torch.abs(lr[:, -sw:] - mid_sw)
    low_q = torch.quantile(abs_diff, 0.01, dim=1, keepdim=True)
    high_q = torch.quantile(abs_diff, 0.99, dim=1, keepdim=True)
    winsorized = torch.clamp(abs_diff, low_q, high_q).mean(1)
    winsorized_vol = winsorized * 1.4826

    realized_std = lr[:, -lw:].std(1, unbiased=False)
    realized_var = lr[:, -lw:].var(1, unbiased=False)

    lr_lw = lr[:, -lw:]
    abs_lag = lr_lw[:, :-1].abs()
    abs_curr = lr_lw[:, 1:].abs()
    exp_decay = torch.exp(-torch.arange(lw - 1, device=device, dtype=dtype))
    bipower = (math.pi / 2) * (abs_lag * abs_curr * exp_decay).sum(1)
    continuous_var = bipower
    jump_variance = torch.clamp(realized_var - continuous_var, min=0.0)
    jump_frac = jump_variance / (realized_var + 1e-12)

    alpha_g, beta_g, gamma_g = 0.1, 0.85, 0.02
    omega = (1 - alpha_g - beta_g - gamma_g) * realized_std**2
    sigma2 = torch.empty(N, T, device=device, dtype=dtype)
    sigma2[:, 0] = lr[:, 0].abs() ** 2
    for t in range(1, T):
        sigma2[:, t] = torch.exp(
            alpha_g * lr[:, t-1].abs()**2
            + beta_g * torch.log(sigma2[:, t-1] + 1e-12)
            + gamma_g * torch.sign(lr[:, t-1]) * lr[:, t-1].abs()**2
            + torch.log(omega + 1e-12)
        )
    sigma_next = torch.exp(
        alpha_g * lr[:, -1].abs()**2
        + beta_g * torch.log(sigma2[:, -1] + 1e-12)
        + gamma_g * torch.sign(lr[:, -1]) * lr[:, -1].abs()**2
        + torch.log(omega + 1e-12)
    )

    reg_window = min(30, T)
    high_regime = realized_std > lr[:, -reg_window:].mean(1)
    prior_sigma2 = torch.where(
        high_regime,
        torch.full((N,), 0.04 ** 2, device=device, dtype=dtype),
        torch.full((N,), 0.02 ** 2, device=device, dtype=dtype),
    )
    prior_df = 4
    posterior_sigma2 = (lw * realized_std**2 + prior_df * prior_sigma2) / (lw + prior_df)
    posterior_sigma = torch.sqrt(posterior_sigma2 + 1e-12)

    vol_30_std = lr[:, -30:].std(1, unbiased=False)
    vol_30_mean = lr[:, -30:].mean(1) + 1e-12
    cv = vol_30_std / vol_30_mean
    blend_factor = 0.5 * (1 - cv) + 0.5 * cv
    blend_factor = blend_factor.clamp(0, 1)
    blended_std = blend_factor * winsorized_vol + (1 - blend_factor) * realized_std

    realized_std_prev = lr[:, -(lw + 5) : -5].std(1, unbiased=False)
    vol_change = posterior_sigma / (realized_std_prev + 1e-12)
    high_vol_flag = (vol_change > 1.2).float()
    sigma0_t = 0.02 * (1 + 0.5 * vol_change)
    lam = torch.where(posterior_sigma > sigma0_t, 0.02, 0.1)
    lam = lam * (0.5 + 0.5 * posterior_sigma / (posterior_sigma.max() + 1e-8))
    lam = lam * (1 + 0.5 * high_vol_flag)
    lam = lam * (1 + 0.5 * (sigma_next / posterior_sigma).clamp(max=1.5))

    lam_ema = 0.1 * (1 + 0.2 * vol_change.mean())
    idx = torch.arange(1, 31, device=device, dtype=dtype).flip(0)
    w_ema = torch.exp(-lam_ema * idx).flip(0)
    w_ema /= w_ema.sum()
    ema3 = (lr[:, -3:] * w_ema[:3]).sum(1)
    ema10 = (lr[:, -10:] * w_ema[:10]).sum(1)
    ema30 = (lr[:, -30:] * w_ema[:30]).sum(1)
    combined_mom = 0.6 * ema3 + 0.3 * ema10 + 0.1 * ema30
    lam *= (1 + 0.05 * torch.sign(combined_mom))

    cum_ret = torch.exp(lr.cumsum(1))
    running_max = cum_ret.cummax(1).values
    drawdowns = (cum_ret - running_max) / running_max
    max_drawdown = -drawdowns.min(1)[0]
    t = torch.arange(T, device=device, dtype=dtype)

    # dynamic beta decay based on skew
    win_skew = lr[:, :min(10, T)]
    mean_skew = win_skew.mean(1, keepdim=True)
    std_skew = win_skew.std(1, unbiased=False, keepdim=True) + 1e-8
    skew_retros = ((win_skew - mean_skew) ** 3).mean(1) / (std_skew.squeeze() ** 3)
    beta = (0.05 + 0.2 * skew_retros.abs()).clamp(0.05, 0.25)
    w_depth = torch.exp(-beta.unsqueeze(1) * t)
    weighted_dd = (-drawdowns) * w_depth
    weighted_penalty = weighted_dd.sum(1) / (w_depth.sum(1) + 1e-12)
    lam *= (1 + 0.5 * weighted_penalty / (weighted_penalty.max() + 1e-8))

    ema_len = min(10, T)
    ema = lr[:, -ema_len:].mean(1)
    lam *= (1 + 0.05 * torch.sign(ema))

    win_skew = lr[:, :min(10, T)]
    mean_skew = win_skew.mean(1, keepdim=True)
    std_skew = win_skew.std(1, unbiased=False, keepdim=True) + 1e-8
    skew_retros = ((win_skew - mean_skew) ** 3).mean(1) / (std_skew.squeeze() ** 3)

    loss = -lr.clamp(max=0)
    k_gpd = max(1, int(0.01 * T))
    top_losses_gpd = loss.topk(k_gpd, dim=1).values
    u = top_losses_gpd[:, -1]
    exceed = top_losses_gpd - u.unsqueeze(1)
    m1 = exceed.mean(1)
    m2 = (exceed**2).mean(1)
    xi = (m1**2) / (2 * m2 - m1**2 + 1e-12)
    xi = xi.clamp(min=-0.99, max=0.99)
    sigma = m1 * (1 - xi + 1e-12)
    es_forecast = u + sigma / (1 - xi + 1e-12)

    k_tail = max(1, int(0.05 * T))
    top_losses = loss.topk(k_tail, dim=1).values
    exp_decay_tail = torch.exp(-torch.arange(k_tail, device=device, dtype=dtype))
    es = (top_losses * exp_decay_tail).sum(1) / exp_decay_tail.sum()
    x_min = top_losses[:, -1] + 1e-12
    tail_log_ratios = torch.log(top_losses / x_min.unsqueeze(1))
    alpha_hat = k_tail / (tail_log_ratios.sum(1) + 1e-12)
    xi_tilde = 1.0 / (alpha_hat + 1e-12)
    xi_tilde = torch.clamp(xi_tilde, max=0.99)
    mean_tail = top_losses.mean(1)
    var_tail = top_losses.var(1, unbiased=False) + 1e-12
    shape_est = 0.5 * (mean_tail**2 / var_tail - 1.0)
    shape_est = torch.clamp(shape_est, min=-0.99, max=0.99)
    scale_est = mean_tail * (1.0 - shape_est)
    prior_shape = torch.full((N,), 0.05, device=device, dtype=dtype)
    prior_scale = torch.full((N,), 0.02, device=device, dtype=dtype)
    posterior_shape = (k_tail * shape_est + prior_shape) / (k_tail + 1.0)
    posterior_scale = (k_tail * scale_est + prior_scale) / (k_tail + 1.0)
    kurt_tail = ((top_losses - top_losses.mean(1, keepdim=True)) ** 4).mean(1) / ((top_losses.std(1, unbiased=False) + 1e-8) ** 4)
    es_forecast = es_forecast * (1 + 0.5 * torch.clamp(kurt_tail, min=0))
    neg = lr.clamp(max=0)
    neg_prev = neg[:, :-1]
    neg_curr = neg[:, 1:]
    mean_prev = neg_prev.mean(1, keepdim=True)
    mean_curr = neg_curr.mean(1, keepdim=True)
    cov = ((neg_prev - mean_prev) * (neg_curr - mean_curr)).mean(1)
    var_prev = ((neg_prev - mean_prev) ** 2).mean(1)
    var_curr = ((neg_curr - mean_curr) ** 2).mean(1)
    rho_neg = cov / torch.sqrt(var_prev * var_curr + 1e-12)
    rho_neg = torch.clamp(rho_neg, max=0.99)
    es_forecast = es_forecast / (1.0 - rho_neg + 1e-12)
    k_cvar = max(1, int(0.01 * T))
    top_cvar_losses, _ = loss.topk(k_cvar, dim=1)
    cvar99 = top_cvar_losses.mean(1)
    skew_decay = torch.exp(-torch.abs(skew_retros) * (cvar99 / sigma0_t))
    lam *= skew_decay

    dd95 = -torch.quantile(drawdowns, 0.05, dim=1)
    blended_dd = 0.7 * max_drawdown + 0.3 * dd95
    rank = torch.argsort(torch.argsort(blended_dd))
    rank_w = 1.0 / (rank + 1).float()
    rank_w = rank_w.clamp(max=0.5)
    lam *= (1 + rank_w)

    low_q = (0.05 + 0.05 * blended_std / sigma0_t).clamp(0.05, 0.10)
    high_q = (0.95 - 0.05 * blended_std / sigma0_t).clamp(0.90, 0.95)
    lr_sorted = lr.sort(1)[0]
    idx_low = (low_q.unsqueeze(1) * T).clamp(min=0, max=T - 1).floor().long()
    idx_high = (high_q.unsqueeze(1) * T).clamp(min=0, max=T - 1).floor().long()
    q_low = lr_sorted.gather(1, idx_low)
    q_high = lr_sorted.gather(1, idx_high)
    trim_w = torch.clamp((lr - q_low) / (q_high - q_low + 1e-8), 0, 1)
    trimmed = lr * trim_w
    stochastic_vol = realized_std

    k = max(1, min(int(0.1 * T), T - 1))
    top_tail, _ = trimmed.abs().topk(k, dim=1)
    probs = top_tail / (top_tail.sum(1, keepdim=True) + 1e-12)
    tail_entropy = -(probs * torch.log(torch.clamp(probs, 1e-8, 1.0))).sum(1)
    dd_dur_median = (drawdowns.lt(0).float().cumsum(1)).median(1)[0]
    dd_loss_weight = (-drawdowns * lr.clamp(max=0).abs()).sum(1) / T
    downside_var = lr.clamp(max=0).pow(2).mean(1)
    tail_penalty = 0.3 * torch.exp(-xi)
    frac_weights = (1 - 0.5 ** torch.arange(1, lw + 1, device=device, dtype=dtype)).float()
    vol_frac = torch.mean(lr[:, -lw:].abs() * frac_weights, dim=1)

    w1 = torch.exp(-0.1 * torch.arange(T, device=device, dtype=dtype))
    w5 = torch.exp(-0.02 * torch.arange(T, device=device, dtype=dtype))
    w20 = torch.exp(-0.005 * torch.arange(T, device=device, dtype=dtype))
    w1 /= w1.sum()
    w5 /= w5.sum()
    w20 /= w20.sum()
    sigma1 = torch.sqrt(((lr**2) * w1).sum(1) + 1e-12)
    sigma5 = torch.sqrt(((lr**2) * w5).sum(1) + 1e-12)
    sigma20 = torch.sqrt(((lr**2) * w20).sum(1) + 1e-12)

    vov = torch.abs(sigma20 - sigma5) / (sigma5 + 1e-12)
    blend_factor_vol = torch.sigmoid(5.0 * vov)
    blended_vol = blend_factor_vol * sigma5 + (1.0 - blend_factor_vol) * sigma20
    blended_vol = blended_vol + 0.1 * sigma1

    risk_measure = torch.norm(
        torch.stack([es_forecast, stochastic_vol, tail_entropy * (xi / (xi + 1.0))], dim=0),
        dim=0,
    )
    risk_measure = risk_measure * (vol_frac / (realized_std + 1e-12))
    risk_measure += 0.1 * weighted_penalty + 1e-8
    risk_measure = risk_measure * (1 + dd_loss_weight)
    risk_measure += tail_penalty + 0.05 * cvar99
    risk_measure = risk_measure * (1 + 0.05 / (1 + dd_dur_median + 1e-8))
    risk_measure = risk_measure * (1 + 0.1 * high_vol_flag)
    risk_measure += 0.05 * es_forecast
    risk_measure += 0.2 * jump_frac
    std30 = lr[:, -30:].std(1, unbiased=False) if T >= 30 else lr.std(1, unbiased=False)
    std_prev30 = lr[:, -(30 * 2) : -30].std(1, unbiased=False) if T >= 60 else lr[:, :-30].std(1, unbiased=False)
    vov_dyn = torch.abs(std30 - std_prev30) / (std_prev30 + 1e-12)
    risk_measure += 0.05 * vov_dyn
    lam *= (1 + 0.1 * vov_dyn)
    sigma_fw = 0.5 * sigma_next + 0.5 * realized_std
    risk_measure += 0.1 * sigma_fw / sigma0_t
    lam *= (1 + 0.05 * sigma_fw / sigma0_t)
    beta_entropy = 0.05
    risk_measure += beta_entropy * tail_entropy

    risk_measure = risk_measure * (blended_vol / sigma0_t + 1e-12)
    lam = lam * (blended_vol / sigma0_t + 1e-12)

    cum_log = lr.cumsum(1)
    idx_draw = drawdowns.argmin(1)
    cum_log_peak = cum_log.gather(1, idx_draw.unsqueeze(1)).squeeze(1)
    cum_log_last = cum_log[:, -1]
    log_ret_from_peak = cum_log_last - cum_log_peak
    penalty_factor = 1.0 - torch.clamp(log_ret_from_peak, min=0) / (cum_log_last.abs() + 1e-8)
    win_5 = lr.unfold(1, 5, 1)
    probs_win = torch.softmax(win_5, 2)
    entropy_win = -(probs_win * torch.log(torch.clamp(probs_win, 1e-8, 1.0))).sum(2).mean(1)
    penalty_factor *= entropy_win / (entropy_win.max() + 1e-8)
    risk_measure *= penalty_factor
    w = torch.exp(-lam.unsqueeze(1) * (T - 1 - t))
    td_mean = (trimmed * w).sum(1) / (w.sum(1) + 1e-8)
    td_mean *= (1 + torch.sigmoid((sigma0_t - posterior_sigma) / 0.02))
    mean_trim = trimmed.median(1)[0]
    geom_mean_ret = torch.exp(mean_trim) - 1
    centered = trimmed - geom_mean_ret.unsqueeze(1)
    downside_std = torch.sqrt(lr.clamp(max=0).pow(2).mean(1) + 1e-8)
    skew_feat = torch.mean(centered ** 3, 1) / (downside_std ** 3 + 1e-8)
    kurt_feat = (torch.mean(centered ** 4, 1) / (downside_std ** 4 + 1e-8) - 3).clamp(-3, 3)
    w_skew = torch.exp(-posterior_sigma / sigma0_t)
    w_kurt = torch.exp(-posterior_sigma / sigma0_t)
    risk_aversion = (drawdowns.mean(1) + 1e-8) / (downside_std + stochastic_vol + 0.1 * sigma0_t)
    risk_aversion *= (1 + (dd_dur_median + 1e-8) / (dd_dur_median.max() + 1e-8))
    risk_aversion *= (1 + 0.2 * torch.sign(skew_retros))
    risk_aversion += 0.2 * high_vol_flag
    risk_aversion *= (1 - 0.3 * jump_frac)

    feat = torch.stack([skew_feat, kurt_feat, entropy_win, stochastic_vol], dim=1)
    uncertainty = feat.var(1) / (feat.mean(1) + 1e-8)
    hurst = torch.log((drawdowns.max(1)[0] - drawdowns.min(1)[0] + 1e-12) / (lr_lw.std(1, unbiased=False) + 1e-12)) / torch.log(torch.tensor(lw, dtype=dtype, device=device) + 1e-12)

    vol_series = lr[:, -lw:].unfold(1, 30, 1)
    vol_std_series = vol_series.std(2, unbiased=False)
    vol_skew_series = ((vol_series - vol_series.mean(2, keepdim=True)) ** 3).mean(2) / (vol_std_series ** 3 + 1e-8)
    w_ewma = torch.exp(-0.1 * torch.arange(vol_series.shape[1]-1, -1, -1, device=device, dtype=dtype))
    w_ewma = w_ewma / w_ewma.sum()
    ewma_skew = (w_ewma * vol_skew_series).sum(1)
    hurst = hurst * (1 + 0.1 * ewma_skew / (ewma_skew.abs().mean() + 1e-12))
    risk_measure = risk_measure * (1 + 0.05 * ewma_skew.abs())
    tail_excess = blended_dd * (downside_std + stochastic_vol) * torch.median(torch.abs(drawdowns), dim=1)[0]
    tail_excess /= (blended_dd + downside_std + 1e-12)
    extreme_freq = (lr < -es.unsqueeze(1)).sum(1) / T
    tail_excess *= torch.exp(-3.0 * extreme_freq)

    lag = min(10, T)
    lag_mean = lr[:, -lag:].mean(1)
    lag_term = lag_mean / (downside_std + stochastic_vol + 1e-12)
    cum_ret_last20 = lr[:, -min(20, T) :].sum(1)
    score_cum_penalty = torch.sqrt(cum_ret_last20.clamp(min=0))
    w_skew = torch.exp(-posterior_sigma / sigma0_t)
    w_kurt = torch.exp(-posterior_sigma / sigma0_t)
    trend_factor = torch.sign(torch.exp(lr[:, -20:].mean(1)) - torch.exp(lr[:, -50:].mean(1)))
    trend_adjust = 0.4 * trend_factor
    score = td_mean * (1 + w_skew * skew_feat - w_kurt * kurt_feat) / risk_measure
    score += 1 + (lr[:, -1] - lr[:, -lag]).clamp(min=0) / (downside_std + stochastic_vol + 1e-12)
    score *= risk_aversion
    score -= uncertainty + hurst
    score += torch.std(trimmed, 1) / (torch.mean(trimmed, 1) + 1e-8)
    score -= tail_excess
    score -= 0.05 * (downside_std + stochastic_vol)
    score -= 0.5 * cvar99
    score -= 0.1 * ((drawdowns < (-blended_dd / 2).unsqueeze(1)).sum(1) / T)
    score += lag_term + score_cum_penalty + trend_adjust
    mean_dur_50 = (drawdowns.lt(-0.5).float().cumsum(1)).mean(1)
    score -= 0.02 * mean_dur_50 / (mean_dur_50.max() + 1e-12)
    mean_recovery = dd_dur_median / torch.maximum((drawdowns < 0).sum(1), torch.tensor(1.0, device=device))
    ratio_penalty = blended_dd / (mean_recovery + 1e-12)
    score -= 0.05 * ratio_penalty
    ir_window = min(30, T)
    skew_ir = (lr[:, -ir_window:].mean(1) - 0.5 * skew_retros * realized_std) / (realized_std + 1e-12)
    score += 0.3 * skew_ir
    score += 0.5 * (xi - 2.0)
    score -= 0.3 * torch.sqrt(downside_var)

    past_period = min(60, T)
    win_ret = cum_ret[:, -1] / cum_ret[:, -past_period] - 1
    cum_window = cum_ret[:, -past_period:]
    max_win = cum_window.cummax(1).values
    win_dd = (cum_window - max_win) / max_win
    max_dd_win = -win_dd.min(1)[0]
    past_calmar = win_ret / (max_dd_win + 1e-12)
    calmar_weight = 1.0 + 0.2 * (past_calmar - past_calmar.mean()) / (past_calmar.std() + 1e-8)
    decay = torch.exp(-0.1 * torch.arange(past_period, device=device, dtype=dtype))
    calmar_weight = calmar_weight * (decay.mean() + 1e-12)
    score *= calmar_weight

    win10 = lr[:, -10:]
    sharpe10 = win10.mean(1) / (win10.std(1, unbiased=False) + 1e-8)
    score += 0.15 * sharpe10
    forward_mean = lr[:, -5:].mean(1)
    score += 0.05 * forward_mean / (realized_std + 1e-8)
    score -= 0.02 * dd_dur_median / (dd_dur_median.max() + 1e-8)
    tail_multiplier = 1 + 0.15 * (1 - torch.exp(-xi))
    score *= tail_multiplier

    if T >= 90:
        cum_ret_window = cum_ret[:, -90:]
        running_max_window = cum_ret_window.cummax(1).values
        mdd90 = -(cum_ret_window - running_max_window) / running_max_window
        max_drawdown90 = -mdd90.min(1)[0]
        score -= 0.05 * (max_drawdown90 / (max_drawdown90.max() + 1e-8))

    gamma = 0.5
    f_reg = torch.exp(-gamma * high_vol_flag)
    score *= f_reg

    last20 = lr[:, -20:] if T >= 20 else lr
    mean_last20 = last20.mean(1)
    var_last20 = last20.var(1, unbiased=False)
    ts_sharpe = mean_last20 / (torch.sqrt(var_last20 + 0.5 * es) + 1e-12)
    score += 0.2 * ts_sharpe
    score *= (1 - 0.5 * jump_frac)
    score += 0.05 * hurst

    lag_win = min(30, T)
    lr_lag = lr[:, -lag_win:]
    x = lr_lag[:, :-1]
    y = lr_lag[:, 1:]
    mean_x = x.mean(1, keepdim=True)
    mean_y = y.mean(1, keepdim=True)
    cov_xy = ((x - mean_x) * (y - mean_y)).mean(1)
    var_x = ((x - mean_x) ** 2).mean(1)
    var_y = ((y - mean_y) ** 2).mean(1)
    rho = cov_xy / torch.sqrt(var_x * var_y + 1e-12)
    mask = torch.abs(rho) > 0.05
    penalty = torch.ones_like(rho)
    penalty[mask] = 1 - (torch.abs(rho[mask]) - 0.05) / (1 - 0.05)
    score = score * penalty

    win5 = min(5, T)
    win20 = min(20, T)
    win60 = min(60, T)
    ulcer5 = torch.sqrt((drawdowns[:, -win5:]**2).mean(1))
    ulcer20 = torch.sqrt((drawdowns[:, -win20:]**2).mean(1))
    ulcer60 = torch.sqrt((drawdowns[:, -win60:]**2).mean(1))
    ulcer = 0.5 * ulcer5 + 0.3 * ulcer20 + 0.2 * ulcer60
    score += 0.1 * ulcer

    mean_std = lr[:, -30:].std(1, unbiased=False)
    regime_prob = torch.sigmoid((sigma2[:, -1] - realized_var) / (realized_var + 1e-12))
    lam = lam * (0.8 + 0.4 * regime_prob)
    risk_measure = risk_measure * (1 + 0.5 * regime_prob)

    abs_dd_last60 = torch.abs(drawdowns[:, -60:]) + 1e-8
    prob_dd = abs_dd_last60 / abs_dd_last60.sum(1, keepdim=True)
    entropy_dd = -(prob_dd * torch.log(prob_dd + 1e-8)).sum(1)
    risk_measure = risk_measure + 0.02 * entropy_dd

    rank_w_sharpe = rank_w * (sharpe10 / (sharpe10 + 1))
    lam *= (1 + rank_w_sharpe)

    market = lr.mean(0)
    m_mean = market.mean()
    m_var = market.var(unbiased=False) + 1e-12
    asset_mean = lr.mean(1, keepdim=True)
    asset_var = lr.var(1, unbiased=False) + 1e-12
    cov = ((lr - asset_mean) * (market - m_mean)).mean(1)
    corr = cov / torch.sqrt(asset_var * m_var)
    corr = corr.clamp(-1, 1)
    corr_pos = torch.clamp(corr, min=0)
    score *= (1 - 0.2 * corr_pos)
    score *= torch.exp(-hurst)
    score += 0.02 * sigma_next / sigma0_t

    abs_ret = log_returns.abs()
    lags = torch.tensor([1, 5, 10], device=device, dtype=torch.long)
    persistence = torch.zeros(N, device=device, dtype=dtype)
    for lag in lags:
        if lag < T:
            x = abs_ret[:, :-lag]
            y = abs_ret[:, lag:]
            mean_x = x.mean(1, keepdim=True)
            mean_y = y.mean(1, keepdim=True)
            cov = ((x - mean_x) * (y - mean_y)).mean(1)
            std_x = x.std(1, unbiased=False)
            std_y = y.std(1, unbiased=False)
            corr = cov / (std_x * std_y + 1e-12)
            persistence += corr
    factor = 1 + 0.1 * persistence
    factor = torch.clamp(factor, 0.8, 1.2)
    score = score * factor

    asset_recent60 = lr[:, -60:] if T >= 60 else lr
    market_recent60 = lr.mean(0)[-60:] if T >= 60 else lr.mean(0)
    mean_asset60 = asset_recent60.mean(1, keepdim=True)
    mean_market60 = market_recent60.mean()
    cov60 = ((asset_recent60 - mean_asset60) * (market_recent60 - mean_market60)).mean(1)
    var_market60 = ((market_recent60 - mean_market60)**2).mean()
    beta60 = cov60 / var_market60
    beta_pos = torch.clamp(beta60, min=0)
    score *= (1 - 0.1 * beta_pos)

    asset_recent30 = lr[:, -30:] if T >= 30 else lr
    market_recent30 = lr.mean(0)[-30:] if T >= 30 else lr.mean(0)
    mean_asset30 = asset_recent30.mean(1, keepdim=True)
    mean_market30 = market_recent30.mean()
    cov30 = ((asset_recent30 - mean_asset30) * (market_recent30 - mean_market30)).mean(1)
    var_market30 = ((market_recent30 - mean_market30)**2).mean()
    var_asset30 = ((asset_recent30 - mean_asset30)**2).mean()
    corr30 = cov30 / torch.sqrt(var_asset30 * var_market30 + 1e-12)
    corr_pos30 = torch.clamp(corr30, min=0)
    score *= (1 - 0.2 * corr_pos30)

    drawdown_abs = (-drawdowns).clamp(min=0)
    probs_dd = drawdown_abs / (drawdown_abs.sum(1, keepdim=True) + 1e-12)
    entropy_dd = -(probs_dd * torch.log(torch.clamp(probs_dd, 1e-8, 1.0))).sum(1)
    score -= 0.05 * entropy_dd

    cs_rank = torch.argsort(torch.argsort(score))
    score = score / (cs_rank + 1).float()

    skew_window = min(10, T)
    cur_skew = ((lr[:, -skew_window:] - lr[:, -skew_window:].mean(1, keepdim=True)).pow(3).mean(1) / (lr[:, -skew_window:].std(1, unbiased=False)**3 + 1e-8))
    prev_skew = ((lr[:, -(skew_window+1):-1] - lr[:, -(skew_window+1):-1].mean(1, keepdim=True)).pow(3).mean(1) / (lr[:, -(skew_window+1):-1].std(1, unbiased=False)**3 + 1e-8))
    skew_change = cur_skew - prev_skew
    skew_change_neg = -skew_change
    skew_change_neg_clipped = torch.clamp(skew_change_neg, 0, 1)
    lam = lam * (1 + 0.15 * skew_change_neg_clipped)
    risk_measure = risk_measure * (1 + 0.05 * skew_change_neg_clipped)
    score = score * (1 + 0.1 * skew_change_neg_clipped)

    alpha_t = 1.0 / (xi + 1e-12)
    w_tail = torch.exp(-0.2 * alpha_t)
    score *= (1 + 0.3 * w_tail)

    if T >= 5:
        lead = lr[:, -1] - lr[:, -5]
    else:
        lead = torch.zeros(N, device=device, dtype=dtype)
    lead_std = lr[:, -5:].std(1) + 1e-8
    lead_score = lead / lead_std
    score = score * (1 + 0.1 * torch.tanh(lead_score))

    risk_measure = risk_measure * (1 + 0.1 * torch.clamp(lead_score, min=0))

    abs_lr_30 = lr[:, -30:].abs()
    sum_abs_30 = abs_lr_30.sum(1, keepdim=True) + 1e-12
    probs_30 = abs_lr_30 / sum_abs_30
    vol_entropy = -(probs_30 * torch.log(probs_30 + 1e-8)).sum(1)
    vol_entropy = vol_entropy / torch.log(torch.tensor(30.0, device=device, dtype=dtype) + 1e-12)
    blended_vol = blended_vol * (1 + 0.05 * vol_entropy)
    risk_measure = risk_measure * (1 + 0.05 * vol_entropy)

    market = lr.mean(0)
    asset60 = lr[:, -60:]
    market60 = market[-60:]
    mean_asset60 = asset60.mean(1, keepdim=True)
    mean_market60 = market60.mean()
    cov60 = ((asset60 - mean_asset60) * (market60 - mean_market60)).mean(1)
    std_asset60 = asset60.std(1, unbiased=False)
    std_market60 = market60.std(unbiased=False)
    corr60 = cov60 / (std_asset60 * std_market60 + 1e-12)

    asset30 = lr[:, -30:]
    market30 = market[-30:]
    mean_asset30 = asset30.mean(1, keepdim=True)
    mean_market30 = market30.mean()
    cov30 = ((asset30 - mean_asset30) * (market30 - mean_market30)).mean(1)
    std_asset30 = asset30.std(1, unbiased=False)
    std_market30 = market30.std(unbiased=False)
    corr30 = cov30 / (std_asset30 * std_market30 + 1e-12)

    decay = (corr60 - corr30) / (torch.abs(corr60) + 1e-12)
    decay = torch.clamp(decay, 0, 1)
    score = score * (1 + 0.2 * decay)

    return score

def optimize_portfolio(log_returns: torch.Tensor, evorisk_scores: torch.Tensor):
    """Risk-adjusted alpha projection via inverse covariance."""
    device, dtype = log_returns.device, log_returns.dtype
    N, T = log_returns.shape
    cov_matrix = torch.cov(log_returns) + 1e-6 * torch.eye(N, device=device, dtype=dtype)
    inv_cov_matrix = torch.linalg.pinv(cov_matrix)
    signal = torch.tanh(evorisk_scores / (evorisk_scores.std() + 1e-8)) * 0.05
    risk_adjusted = inv_cov_matrix @ signal
    risk_adjusted = torch.clamp(risk_adjusted, min=0.0)
    weights = risk_adjusted / risk_adjusted.sum()
    return weights

# ================================================================
# 1. Data Loading
# ================================================================
with open('Dataset.pkl', 'rb') as f:
    Dataset = cPickle.load(f)

data = np.array(Dataset).T.astype(np.float64)
cutoff_index = data.shape[1] // 5  # 20% test
train_np = data[:, :-cutoff_index]
test_np  = data[:, -cutoff_index:]
n_assets = train_np.shape[0]

# ================================================================
# 2. Convert to Torch
# ================================================================
device = 'cuda' if torch.cuda.is_available() else 'cpu'
train = torch.tensor(train_np, dtype=torch.float32, device=device)
test  = torch.tensor(test_np,  dtype=torch.float32, device=device)

# ================================================================
# 3. Compute Selector Scores
# ================================================================
scores_calmar = evorisk_metric(train, risk_free_rate=0.0)
scores_alpha  = alpha_sharpe(train, risk_free= 0.0, eps= 1.5e-5, dr= 2.0, fv= 1.33, window= 3)
rank_calmar   = torch.argsort(scores_calmar, descending=True)
rank_alpha    = torch.argsort(scores_alpha,  descending=True)

# ================================================================
# 4. Metric Helpers
# ================================================================
def calc_sharpe(log_returns_np, periods_per_year=252):
    mean_lr = np.mean(log_returns_np)
    std_lr  = np.std(log_returns_np, ddof=1)
    return (mean_lr / (std_lr + 1e-8)) * np.sqrt(periods_per_year)

def calc_calmar(log_returns_np, periods_per_year=252):
    cum = np.exp(np.cumsum(log_returns_np))
    run_max = np.maximum.accumulate(cum)
    dd = (run_max - cum) / (run_max + 1e-8)
    max_dd = np.max(dd)
    ann_ret = np.mean(log_returns_np) * periods_per_year
    return ann_ret / (max_dd + 1e-8)

# ================================================================
# 5. Evaluation Loop
# ================================================================
selection_ratios = np.linspace(0.1, 1.0, 20)

def evaluate_selector(ranking, scores):
    """Return arrays of Sharpe, Calmar, Mean for Equal & Optimized allocations."""
    results_eq, results_opt = [], []
    for ratio in selection_ratios:
        n_select = int(np.ceil(ratio * n_assets))
        sel = ranking[:n_select]
        tr, te = train[sel], test[sel]

        # Equal weights
        w_eq = torch.ones(n_select, device=device) / n_select
        r_eq = (w_eq @ te).detach().cpu().numpy()

        # Optimized weights
        w_opt = optimize_portfolio(tr, scores[:n_select])
        r_opt = (w_opt @ te).detach().cpu().numpy()

        # Metrics
        results_eq.append((
            calc_sharpe(r_eq),
            calc_calmar(r_eq),
            np.mean(r_eq)
        ))
        results_opt.append((
            calc_sharpe(r_opt),
            calc_calmar(r_opt),
            np.mean(r_opt)
        ))
    return np.array(results_eq), np.array(results_opt)

res_calmar_eq, res_calmar_opt = evaluate_selector(rank_calmar, scores_calmar)
#res_alpha_eq,  res_alpha_opt  = evaluate_selector(rank_alpha, scores_alpha)


rows = []
for i, ratio in enumerate(selection_ratios):
    rows.append({
        'Selection_Ratio': ratio,
        'EvoRisk_EQ_Sharpe':  res_calmar_eq[i,0],
        'EvoRisk_EQ_Calmar':  res_calmar_eq[i,1],
        'EvoRisk_EQ_Return':    res_calmar_eq[i,2],
        'EvoRisk_OPT_Sharpe': res_calmar_opt[i,0],
        'EvoRisk_OPT_Calmar': res_calmar_opt[i,1],
        'EvoRisk_OPT_Return':   res_calmar_opt[i,2],
    })

df_results = pd.DataFrame(rows)

# Round and show key metrics for compactness
pd.set_option('display.float_format', lambda x: f'{x:8.4f}')
print("\n===== Out-of-Sample Performance by Selection Ratio =====\n")
print(df_results.to_string(index=False))


# ================================================================
# 6. Visualization
# ================================================================
fig, axes = plt.subplots(3, 1, figsize=(12, 16), sharex=True)

labels = [
    ('EvoRisk (equal)', res_calmar_eq),
    ('EvoRisk (opt)',   res_calmar_opt),
    #('alpha_sharpe (equal)',  res_alpha_eq),
    #('alpha_sharpe (opt)',    res_alpha_opt)
]

# --- Sharpe ---
for name, res in labels:
    axes[0].plot(selection_ratios, res[:,0], marker='o', label=name)
axes[0].set_ylabel('Sharpe Ratio')
axes[0].set_title('Out-of-Sample Sharpe')
axes[0].legend(); axes[0].grid(True)

# --- Calmar ---
for name, res in labels:
    axes[1].plot(selection_ratios, res[:,1], marker='o', label=name)
axes[1].set_ylabel('Calmar Ratio')
axes[1].set_title('Out-of-Sample Calmar')
axes[1].legend(); axes[1].grid(True)

# --- Mean ---
for name, res in labels:
    axes[2].plot(selection_ratios, res[:,2], marker='o', label=name)
axes[2].set_xlabel('Selection Ratio (Fraction of Assets Selected)')
axes[2].set_ylabel('Mean Log Return')
axes[2].set_title('Out-of-Sample Mean Return')
axes[2].legend(); axes[2].grid(True)

plt.tight_layout()
plt.show()
