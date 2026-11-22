"""
main.py
End-to-end runner for the AI-Powered IEX Energy Allocation, BESS Optimization,
and Dynamic Pricing Framework.
"""
from config import ClusterConfig
from data_pipeline import generate_historical_dataset, engineer_features
from forecasting import EnergyForecastingSuite
from simulation import run_strategy_comparison


def main():
    print("=== 1. Initializing Cluster Configuration & Synthetic Data ===")
    cfg = ClusterConfig()
    raw_df = generate_historical_dataset(days=120, seed=42)
    feat_df = engineer_features(raw_df)

    # Time-series train/test split (last 24 hours as out-of-sample test day)
    train_df = feat_df.iloc[:-24].copy()
    test_day_df = feat_df.iloc[-24:].copy()

    print("=== 2. Training AI/ML Forecasting Suite (LightGBM, XGBoost, SARIMAX, GPR) ===")
    forecaster = EnergyForecastingSuite()
    forecaster.fit(train_df)
    metrics = forecaster.evaluate(test_day_df)
    print(f"Forecast Validation Metrics: {metrics}")

    print("=== 3. Running 6-Scenario Simulation (Strategies A vs. B vs. C) ===")
    results_df = run_strategy_comparison(cfg, forecaster, test_day_df)

    print("\n=== SIMULATION RESULTS SUMMARY ===")
    print(results_df.to_string(index=False))

    results_df.to_csv("simulation_results.csv", index=False)
    print("\nSaved detailed results to simulation_results.csv")


if __name__ == "__main__":
    main()