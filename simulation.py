"""
simulation.py
Runs the 6-Scenario Evaluation comparing Strategy A, B, and C (Part 7).
"""
import numpy as np
import pandas as pd
from config import ClusterConfig, SCENARIOS
from forecasting import EnergyForecastingSuite
from optimization import solve_stochastic_day_ahead
from mpc_controller import mpc_step_dispatch


def run_strategy_comparison(
    cfg: ClusterConfig,
    forecaster: EnergyForecastingSuite,
    test_day_df: pd.DataFrame
) -> pd.DataFrame:
    results = []

    for scen_name, mods in SCENARIOS.items():
        day = test_day_df.copy().reset_index(drop=True)
        day["iex_mcp_inr"] = np.clip(day["iex_mcp_inr"] * mods["mcp_mult"], 2.0, cfg.iex_price_cap_inr)
        day["re_gen_kwh"] = day["re_gen_kwh"] * mods["re_mult"]
        day["demand_kwh"] = day["demand_kwh"] * mods["demand_mult"]
        soh = mods["soh"]

        # Forecast scenarios for Strategy B & C
        fc = forecaster.predict_day_scenarios(day)
        # Adjust forecast scenarios by scenario modifiers to simulate stress regime
        for s in fc["scenarios"]:
            s["mcp"] = np.clip(s["mcp"] * mods["mcp_mult"], 2.0, cfg.iex_price_cap_inr)
            s["re"] = s["re"] * mods["re_mult"]
            s["demand"] = s["demand"] * mods["demand_mult"]

        # Solve Strategy C Day-Ahead Stochastic MILP
        soc_init = 0.50 * cfg.E_max_kwh * soh
        opt_c = solve_stochastic_day_ahead(cfg, fc["scenarios"], soc_init, soh, cfg.base_tariff_inr)

        for strat in ["A_Grid_Static", "B_Forecast_Rule", "C_AI_Stoch_MPC"]:
            soc = soc_init
            tot_rev = tot_ebitda = tot_ec = tot_deg = tot_re_used = tot_re_avail = tot_served = tot_unserved = 0.0
            comp_hours = 0

            for t in range(len(day)):
                d_act = day.loc[t, "demand_kwh"]
                re_act = day.loc[t, "re_gen_kwh"]
                mcp_act = day.loc[t, "iex_mcp_inr"]

                if strat == "A_Grid_Static":
                    # No BESS arbitrage, no dynamic pricing, grid-only
                    tariff = cfg.base_tariff_inr
                    landed = mcp_act + cfg.landed_addon_inr
                    rev = d_act * tariff
                    ec = d_act * landed
                    ebitda = rev - ec - cfg.fixed_opex_hr_inr
                    tot_rev += rev; tot_ebitda += ebitda; tot_ec += ec
                    tot_served += d_act; tot_re_avail += re_act; comp_hours += 1
                else:
                    if strat == "B_Forecast_Rule":
                        dam_bid = max(0.0, fc["scenarios"][1]["demand"][t] - fc["scenarios"][1]["re"][t])
                        tariff = cfg.base_tariff_inr
                    else:
                        dam_bid = opt_c["dam_bids"][t]
                        tariff = opt_c["tariffs"][t]

                    step = mpc_step_dispatch(
                        cfg, d_act, re_act, mcp_act, dam_bid, tariff,
                        cfg.base_tariff_inr, soc, soh
                    )
                    soc = step["soc_next"]
                    tot_rev += step["revenue"]; tot_ebitda += step["ebitda"]
                    tot_ec += step["energy_cost"]; tot_deg += step["deg_cost"]
                    tot_re_used += step["re_used"]; tot_re_avail += step["re_avail"]
                    tot_served += step["served_kwh"]; tot_unserved += step["unserved_kwh"]
                    comp_hours += int(step["reserve_compliant"])

            margin_pct = (tot_ebitda / tot_rev * 100.0) if tot_rev > 0 else 0.0
            re_util_pct = (tot_re_used / tot_re_avail * 100.0) if tot_re_avail > 0 else 0.0
            unserved_pct = (tot_unserved / (tot_served + tot_unserved) * 100.0) if (tot_served + tot_unserved) > 0 else 0.0

            results.append({
                "Scenario": scen_name,
                "Strategy": strat,
                "EBITDA_INR": round(tot_ebitda, 1),
                "EBITDA_Margin_%": round(margin_pct, 2),
                "Energy_Cost_INR_kWh": round(tot_ec / max(tot_served, 1.0), 2),
                "RE_Util_%": round(re_util_pct, 1),
                "Deg_Cost_INR": round(tot_deg, 1),
                "Unserved_%": round(unserved_pct, 2),
                "Reserve_Comp_%": round(comp_hours / len(day) * 100.0, 1),
            })

    return pd.DataFrame(results)