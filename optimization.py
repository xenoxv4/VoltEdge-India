"""
optimization.py
Two-Stage Stochastic MILP for Day-Ahead IEX Bidding, Dynamic Tariff Pricing,
and BESS Scheduling (Part 3 & Part 4).
"""
import numpy as np
import pulp
from config import ClusterConfig


def solve_stochastic_day_ahead(
    cfg: ClusterConfig,
    scenarios: list,
    soc_init_kwh: float,
    soh_current: float,
    prev_day_avg_tariff: float
) -> dict:
    T = len(scenarios[0]["demand"])
    W = len(scenarios)

    # Discrete tariff tiers within +/- 20% to linearize p_t * D_t(p_t) exactly
    tier_multipliers = [0.80, 0.90, 1.00, 1.10, 1.20]
    K = len(tier_multipliers)
    tier_prices = [prev_day_avg_tariff * m for m in tier_multipliers]

    E_usable_max = cfg.E_max_kwh * soh_current
    E_min_floor = (cfg.E_min_tech_frac + cfg.reserve_frac) * E_usable_max

    prob = pulp.LpProblem("Stochastic_IEX_BESS_MILP", pulp.LpMaximize)

    # First-stage decisions (Here-and-Now: Day-Ahead bids & Customer Tariffs)
    b_dam = pulp.LpVariable.dicts("b_DAM", range(T), lowBound=0, upBound=cfg.P_grid_max_kw)
    z_tier = pulp.LpVariable.dicts("z_tier", (range(T), range(K)), cat="Binary")

    # Second-stage recourse decisions (Wait-and-See per scenario w)
    b_rtm = pulp.LpVariable.dicts("b_RTM", (range(W), range(T)), lowBound=0, upBound=cfg.P_grid_max_kw)
    g_re = pulp.LpVariable.dicts("g_RE", (range(W), range(T)), lowBound=0)
    p_ch = pulp.LpVariable.dicts("P_ch", (range(W), range(T)), lowBound=0, upBound=cfg.P_ch_max_kw)
    p_dis = pulp.LpVariable.dicts("P_dis", (range(W), range(T)), lowBound=0, upBound=cfg.P_dis_max_kw)
    u_ch = pulp.LpVariable.dicts("u_ch", (range(W), range(T)), cat="Binary")
    soc = pulp.LpVariable.dicts("SOC", (range(W), range(T)), lowBound=E_min_floor, upBound=E_usable_max)
    unserved = pulp.LpVariable.dicts("unserved", (range(W), range(T)), lowBound=0)
    margin_slack = pulp.LpVariable("margin_slack", lowBound=0)

    # Objective: Maximize Expected EBITDA - soft margin violation penalty
    expected_ebitda_terms = []
    expected_rev_terms = []

    for w in range(W):
        pi_w = scenarios[w]["prob"]
        for t in range(T):
            d_hat = scenarios[w]["demand"][t]
            mcp = scenarios[w]["mcp"][t]
            landed_cost = mcp + cfg.landed_addon_inr

            # Linearized Revenue & Demand across tariff tiers
            rev_wt = pulp.lpSum([
                z_tier[t][k] * tier_prices[k] * d_hat * (1.0 + cfg.elasticity * (tier_multipliers[k] - 1.0))
                for k in range(K)
            ])
            dem_wt = pulp.lpSum([
                z_tier[t][k] * d_hat * (1.0 + cfg.elasticity * (tier_multipliers[k] - 1.0))
                for k in range(K)
            ])

            cost_wt = (
                landed_cost * (b_dam[t] + b_rtm[w][t])
                + cfg.re_cost_inr * g_re[w][t]
                + cfg.c_deg_inr * p_dis[w][t]
                + cfg.c_voll_inr * unserved[w][t]
                + cfg.fixed_opex_hr_inr
            )
            expected_rev_terms.append(pi_w * rev_wt)
            expected_ebitda_terms.append(pi_w * (rev_wt - cost_wt))

            # Constraints per (w, t)
            prob += g_re[w][t] <= scenarios[w]["re"][t]
            prob += b_dam[t] + b_rtm[w][t] <= cfg.P_grid_max_kw

            # Energy balance (Delta t = 1 hr)
            prob += b_dam[t] + b_rtm[w][t] + g_re[w][t] + p_dis[w][t] + unserved[w][t] == dem_wt + p_ch[w][t]

            # Non-simultaneous charge/discharge
            prob += p_ch[w][t] <= u_ch[w][t] * cfg.P_ch_max_kw
            prob += p_dis[w][t] <= (1 - u_ch[w][t]) * cfg.P_dis_max_kw

            # SOC dynamics
            prev_soc = soc_init_kwh if t == 0 else soc[w][t - 1]
            prob += soc[w][t] == prev_soc + cfg.eta_c * p_ch[w][t] - (1.0 / cfg.eta_d) * p_dis[w][t]

    # Exactly one tariff tier active per hour
    for t in range(T):
        prob += pulp.lpSum([z_tier[t][k] for k in range(K)]) == 1

    # Soft 12% EBITDA margin constraint
    total_exp_ebitda = pulp.lpSum(expected_ebitda_terms)
    total_exp_rev = pulp.lpSum(expected_rev_terms)
    prob += total_exp_ebitda + margin_slack >= cfg.ebitda_margin_target * total_exp_rev

    prob += total_exp_ebitda - 100.0 * margin_slack
    prob.solve(pulp.PULP_CBC_CMD(msg=False))

    # Extract First-Stage Decisions
    dam_bids = np.array([pulp.value(b_dam[t]) for t in range(T)])
    tariffs = np.array([
        sum(tier_prices[k] * pulp.value(z_tier[t][k]) for k in range(K)) for t in range(T)
    ])
    return {"dam_bids": dam_bids, "tariffs": tariffs, "status": pulp.LpStatus[prob.status]}