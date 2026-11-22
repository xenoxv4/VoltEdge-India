"""
forecasting.py
AI/ML Forecasting Layer (Part 2):
- LightGBM (Quantile) for Demand
- XGBoost for Renewable Generation
- LightGBM + SARIMAX Ensemble for IEX Prices
- Physics-Informed GPR for Battery SOH
"""
import warnings
import numpy as np
import pandas as pd
import lightgbm as lgb
import xgboost as xgb
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel
from sklearn.metrics import mean_absolute_error, root_mean_squared_error
from statsmodels.tsa.statespace.sarimax import SARIMAX

warnings.filterwarnings("ignore")


def pinball_loss(y_true: np.ndarray, y_pred: np.ndarray, q: float) -> float:
    err = y_true - y_pred
    return float(np.mean(np.maximum(q * err, (q - 1.0) * err)))


class EnergyForecastingSuite:
    def __init__(self):
        self.demand_models = {}
        self.re_model = None
        self.iex_lgb = None
        self.iex_sarimax = None
        self.soh_gpr = None
        self.feature_cols = [
            "hour_sin", "hour_cos", "dow", "temp_c", "irradiance",
            "demand_kwh_lag24", "demand_kwh_lag168", "demand_kwh_roll24_mean",
            "iex_mcp_inr_lag24", "iex_mcp_inr_lag168", "iex_mcp_inr_roll24_mean",
            "re_gen_kwh_lag24", "re_gen_kwh_lag168", "re_gen_kwh_roll24_mean"
        ]

    def fit(self, train_df: pd.DataFrame):
        X = train_df[self.feature_cols]

        # 1. LightGBM Quantile Regression for Demand (q10, q50, q90)
        for q in [0.1, 0.5, 0.9]:
            model = lgb.LGBMRegressor(
                objective="quantile", alpha=q, n_estimators=100,
                learning_rate=0.05, verbose=-1, random_state=42
            )
            model.fit(X, train_df["demand_kwh"])
            self.demand_models[q] = model

        # 2. XGBoost for Renewable Generation
        self.re_model = xgb.XGBRegressor(
            n_estimators=100, max_depth=5, learning_rate=0.05, random_state=42
        )
        self.re_model.fit(X, train_df["re_gen_kwh"])

        # 3. Ensemble (LightGBM + SARIMAX) for IEX MCP
        self.iex_lgb = lgb.LGBMRegressor(
            n_estimators=100, learning_rate=0.05, verbose=-1, random_state=42
        )
        self.iex_lgb.fit(X, train_df["iex_mcp_inr"])

        # Lightweight SARIMAX on recent 14 days for linear seasonality
        recent_mcp = train_df["iex_mcp_inr"].tail(24 * 14).values
        self.iex_sarimax = SARIMAX(
            recent_mcp, order=(1, 0, 1), seasonal_order=(1, 0, 0, 24)
        ).fit(disp=False)

        # 4. Physics-Informed GPR for Battery SOH (subsampled for speed)
        sub = train_df.iloc[::24]
        X_soh = sub[["ah_throughput", "temp_c"]].values
        kernel = RBF(length_scale=[1e5, 10.0]) + WhiteKernel(noise_level=1e-5)
        self.soh_gpr = GaussianProcessRegressor(kernel=kernel, random_state=42)
        self.soh_gpr.fit(X_soh, sub["soh"].values)

    def predict_day_scenarios(self, test_day_df: pd.DataFrame) -> dict:
        """Generates 3 representative scenarios (Low, Median, High) for stochastic MILP."""
        X = test_day_df[self.feature_cols]
        d_q10 = np.clip(self.demand_models[0.1].predict(X), 5.0, None)
        d_q50 = np.clip(self.demand_models[0.5].predict(X), 5.0, None)
        d_q90 = np.clip(self.demand_models[0.9].predict(X), 5.0, None)

        re_pred = np.clip(self.re_model.predict(X), 0.0, None)

        mcp_lgb = self.iex_lgb.predict(X)
        mcp_sar = self.iex_sarimax.forecast(steps=len(X))
        mcp_ens = np.clip(0.7 * mcp_lgb + 0.3 * mcp_sar, 2.0, 10.0)

        return {
            "scenarios": [
                {"prob": 0.25, "demand": d_q10, "re": re_pred * 1.10, "mcp": np.clip(mcp_ens * 0.85, 2.0, 10.0)},
                {"prob": 0.50, "demand": d_q50, "re": re_pred,        "mcp": mcp_ens},
                {"prob": 0.25, "demand": d_q90, "re": re_pred * 0.85, "mcp": np.clip(mcp_ens * 1.15, 2.0, 10.0)},
            ],
            "metrics_pred": {"demand_q50": d_q50, "demand_q90": d_q90, "re": re_pred, "mcp": mcp_ens}
        }

    def evaluate(self, test_df: pd.DataFrame) -> dict:
        X = test_df[self.feature_cols]
        d_pred = self.demand_models[0.5].predict(X)
        d_q90 = self.demand_models[0.9].predict(X)
        return {
            "demand_mae_kwh": round(mean_absolute_error(test_df["demand_kwh"], d_pred), 2),
            "demand_rmse_kwh": round(root_mean_squared_error(test_df["demand_kwh"], d_pred), 2),
            "demand_pinball_q90": round(pinball_loss(test_df["demand_kwh"].values, d_q90, 0.9), 2),
        }