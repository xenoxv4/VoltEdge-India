"""
data_pipeline.py
Synthetic historical data generator (simulating the 5-year hourly dataset)
and feature engineering pipeline.
"""
import numpy as np
import pandas as pd


def generate_historical_dataset(days: int = 180, seed: int = 42) -> pd.DataFrame:
    """
    Generates an hourly dataset representing weather, cluster charging demand,
    solar RE generation, IEX DAM/RTM prices, and battery SOH.
    Default is 180 days for fast execution; set days=1825 for full 5-year run.
    """
    rng = np.random.default_rng(seed)
    n_hours = days * 24
    timestamps = pd.date_range(start="2025-01-01", periods=n_hours, freq="h")
    hours = timestamps.hour.values
    dow = timestamps.dayofweek.values

    # Ambient temperature (C) & Solar irradiance (W/m2)
    temp_c = 28.0 + 7.0 * np.sin(2 * np.pi * (hours - 9) / 24) + rng.normal(0, 1.5, n_hours)
    solar_curve = np.maximum(0, np.sin(np.pi * (hours - 6) / 12))
    cloud_factor = np.clip(rng.beta(5, 2, n_hours), 0.1, 1.0)
    irradiance = 900.0 * solar_curve * cloud_factor

    # Renewable generation (kWh)
    re_gen_kwh = np.clip(0.08 * irradiance + rng.normal(0, 2.0, n_hours), 0, 80.0)

    # EV Charging & Swap Demand (kWh) - morning & evening peaks
    morning_peak = 35.0 * np.exp(-0.5 * ((hours - 9) / 2.2) ** 2)
    evening_peak = 65.0 * np.exp(-0.5 * ((hours - 19) / 2.5) ** 2)
    weekend_boost = np.where(dow >= 5, 1.10, 1.0)
    demand_kwh = np.clip((35.0 + morning_peak + evening_peak) * weekend_boost + rng.normal(0, 5.0, n_hours), 10.0, 160.0)

    # IEX Market Clearing Price (INR/kWh) - capped at 10.00 INR/kWh
    iex_base = 4.2 + 2.8 * np.exp(-0.5 * ((hours - 19.5) / 2.5) ** 2) - 1.2 * solar_curve
    iex_mcp_inr = np.clip(iex_base + rng.normal(0, 0.65, n_hours), 2.0, 10.0)

    # Battery SOH degradation trajectory
    ah_throughput = np.cumsum(demand_kwh * 0.4)
    soh = np.clip(1.0 - 1.5e-6 * ah_throughput, 0.70, 1.0)

    df = pd.DataFrame({
        "timestamp": timestamps,
        "temp_c": temp_c,
        "irradiance": irradiance,
        "re_gen_kwh": re_gen_kwh,
        "demand_kwh": demand_kwh,
        "iex_mcp_inr": iex_mcp_inr,
        "ah_throughput": ah_throughput,
        "soh": soh,
    })
    return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Adds cyclical encodings, lags, and rolling statistics without leakage."""
    data = df.copy()
    data["hour"] = data["timestamp"].dt.hour
    data["dow"] = data["timestamp"].dt.dayofweek
    data["hour_sin"] = np.sin(2 * np.pi * data["hour"] / 24)
    data["hour_cos"] = np.cos(2 * np.pi * data["hour"] / 24)

    for col in ["demand_kwh", "iex_mcp_inr", "re_gen_kwh"]:
        data[f"{col}_lag24"] = data[col].shift(24)
        data[f"{col}_lag168"] = data[col].shift(168)
        data[f"{col}_roll24_mean"] = data[col].shift(24).rolling(24).mean()

    return data.dropna().reset_index(drop=True)