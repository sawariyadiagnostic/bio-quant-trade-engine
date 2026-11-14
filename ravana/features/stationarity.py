"""Stationarity and Mean-Reversion Estimators for Muscle 7.

Computes:
1. Rolling Augmented Dickey-Fuller (ADF) stationarity test statistic on tracking spread.
2. Rolling Ornstein-Uhlenbeck (OU) Mean-Reversion Half-Life (-ln(2) / beta).
"""

from typing import Optional
import numpy as np
import pandas as pd


class RollingADF:
    """Computes causal rolling Augmented Dickey-Fuller t-statistic."""

    def __init__(self, window: int = 60, max_lags: int = 1) -> None:
        self.window = max(10, window)
        self.max_lags = max(0, min(max_lags, 2))

    def compute(self, series: pd.Series) -> pd.Series:
        """Calculate rolling ADF t-statistic over specified lookback window using batched linear algebra."""
        from numpy.lib.stride_tricks import sliding_window_view

        values = series.to_numpy(dtype=np.float64)
        n = len(values)
        out = np.zeros(n, dtype=np.float64)

        if n < self.window:
            return pd.Series(out, index=series.index, name="ADF_STAT")

        windows = sliding_window_view(values, window_shape=self.window)  # shape: [M, W]
        dy = np.diff(windows, axis=1)                                    # shape: [M, W-1]
        y_lag = windows[:, :-1]                                          # shape: [M, W-1]
        m_wins, k = dy.shape

        if self.max_lags == 1 and k > 2:
            dy_lag = dy[:, :-1]
            y_vec = dy[:, 1:]
            y_prev = y_lag[:, 1:]
            ones = np.ones_like(y_prev)
            x_mat = np.stack([ones, y_prev, dy_lag], axis=-1)            # shape: [M, K', 3]
        else:
            ones = np.ones_like(y_lag)
            x_mat = np.stack([ones, y_lag], axis=-1)                     # shape: [M, K', 2]
            y_vec = dy

        n_obs, p = x_mat.shape[1], x_mat.shape[2]
        dof = max(1, n_obs - p)

        xtx = np.einsum("mki,mkj->mij", x_mat, x_mat)                    # shape: [M, P, P]
        traces = np.trace(xtx, axis1=1, axis2=2)
        ridges = 1e-8 * np.maximum(1.0, traces)[:, None, None] * np.eye(p)[None, :, :]
        xtx_reg = xtx + ridges
        xty = np.einsum("mki,mk->mi", x_mat, y_vec)                      # shape: [M, P]

        try:
            beta = np.linalg.solve(xtx_reg, xty[..., None]).squeeze(-1)  # shape: [M, P]
            y_hat = np.einsum("mki,mi->mk", x_mat, beta)
            residuals = y_vec - y_hat
            sigma2 = np.sum(residuals ** 2, axis=1) / dof
            inv_xtx = np.linalg.inv(xtx_reg)
            se_beta = np.sqrt(np.maximum(1e-12, sigma2 * inv_xtx[:, 1, 1]))
            t_stats = np.clip(beta[:, 1] / se_beta, -20.0, 20.0)
            out[self.window - 1 :] = np.nan_to_num(t_stats, nan=0.0)
        except (np.linalg.LinAlgError, ValueError):
            for t in range(self.window - 1, n):
                out[t] = self._estimate_adf_t_stat(values[t - self.window + 1 : t + 1])

        return pd.Series(out, index=series.index, name="ADF_STAT")

    def _estimate_adf_t_stat(self, sub: np.ndarray) -> float:
        """Estimate t-statistic of beta in Delta y_t = alpha + beta*y_{t-1} + gamma*Delta y_{t-1}."""
        dy = np.diff(sub)
        y_lag = sub[:-1]
        k = len(dy)

        if self.max_lags == 1 and k > 2:
            dy_lag = dy[:-1]
            dy_target = dy[1:]
            y_prev = y_lag[1:]
            ones = np.ones(len(dy_target))
            x_mat = np.column_stack([ones, y_prev, dy_lag])
            y_vec = dy_target
            target_idx = 1
        else:
            ones = np.ones(k)
            x_mat = np.column_stack([ones, y_lag])
            y_vec = dy
            target_idx = 1

        return self._solve_ols_t_stat(x_mat, y_vec, target_idx)

    def _solve_ols_t_stat(self, x_mat: np.ndarray, y_vec: np.ndarray, target_idx: int) -> float:
        """Solve OLS regression and extract t-statistic with numerical stabilization."""
        try:
            xtx = x_mat.T @ x_mat
            # Add trace-scaled Tikhonov ridge for collinear/constant series
            trace_val = float(np.trace(xtx))
            ridge = 1e-8 * max(1.0, trace_val)
            xtx_reg = xtx + np.eye(xtx.shape[0]) * ridge
            beta = np.linalg.solve(xtx_reg, x_mat.T @ y_vec)
            residuals = y_vec - x_mat @ beta
            n, p = x_mat.shape
            dof = max(1, n - p)
            sigma2 = np.sum(residuals**2) / dof
            cov_beta = sigma2 * np.linalg.pinv(xtx_reg)
            se_beta = np.sqrt(max(1e-12, cov_beta[target_idx, target_idx]))
            t_stat = beta[target_idx] / se_beta
            return float(np.clip(t_stat, -20.0, 20.0))
        except (np.linalg.LinAlgError, ValueError, ZeroDivisionError):
            return 0.0


class RollingOUHalfLife:
    """Computes causal rolling Ornstein-Uhlenbeck Mean-Reversion Half-Life."""

    def __init__(
        self,
        window: int = 60,
        min_half_life: float = 1.0,
        max_half_life: float = 252.0,
    ) -> None:
        self.window = max(10, window)
        self.min_half_life = min_half_life
        self.max_half_life = max_half_life

    def compute(self, series: pd.Series) -> pd.Series:
        """Compute rolling OU half-life: -ln(2) / beta from Delta y_t = alpha + beta*y_{t-1}."""
        s = series.astype(np.float64)
        y_lag = s.shift(1)
        dy = s - y_lag

        # Rolling covariance and variance using Pandas causal rolling
        cov = dy.rolling(self.window, min_periods=self.window // 2).cov(y_lag)
        var = y_lag.rolling(self.window, min_periods=self.window // 2).var()

        # Beta = Cov(dy, y_lag) / Var(y_lag)
        safe_var = var.replace(0.0, np.nan)
        beta = cov / safe_var

        # Mean reversion requires beta < 0 in Delta y_t = beta * y_{t-1}
        # Discrete AR(1) representation: y_t = (1 + beta) * y_{t-1}
        # Discrete half-life formula: -ln(2) / ln(1 + beta)
        hl = pd.Series(self.max_half_life, index=series.index, name="OU_HALF_LIFE")
        
        # 1. Standard discrete mean-reversion range: -1.0 < beta < -1e-6
        std_reverting = (beta > -0.9999) & (beta < -1e-6)
        hl[std_reverting] = -np.log(2.0) / np.log(1.0 + beta[std_reverting])

        # 2. Overshooting regime: beta <= -1.0 (mean crossed within <= 1 period)
        overshooting = beta <= -0.9999
        hl[overshooting] = self.min_half_life

        # Apply bounds clipping and strictly causal forward filling (zero bfill)
        hl = hl.clip(lower=self.min_half_life, upper=self.max_half_life)
        hl = hl.ffill().fillna(self.max_half_life)
        return hl
