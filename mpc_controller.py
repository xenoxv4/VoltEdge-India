"""
mpc_controller.py
Real-Time MPC Controller (Part 3 & Part 5):
Re-optimizes BESS dispatch and RTM balancing against realized actuals
while strictly enforcing the 5% reserve obligation.
"""
from config import ClusterConfig


def mpc_step_dispatch(
    cfg: ClusterConfig,
    actual_demand_base: float,
    actual_re: float,
    actual_mcp: float,
    dam_bid_kwh: float,
    tariff_inr: float,
    prev_day_avg_tariff: float,
    soc_current: float,
    soh_current: float,
    avg_offpeak_cost: float = 6.25
) -> dict:
    # Realized elastic demand under chosen tariff
    rel_price_change = (tariff_inr - prev_day_avg_tariff) / prev_day_avg_tariff
    realized_demand = max(0.0, actual_demand_base * (1.0 + cfg.elasticity * rel_price_change))

    E_usable_max = cfg.E_max_kwh * soh_current
    E_min_floor = (cfg.E_min_tech_frac + cfg.reserve_frac) * E_usable_max

    # 1. Utilize available RE first
    re_used = min(actual_re, realized_demand + cfg.P_ch_max_kw)
    net_load = realized_demand - re_used - dam_bid_kwh

    p_dis = 0.0
    p_ch = 0.0
    rtm_buy = 0.0
    unserved = 0.0
    landed_rtm = actual_mcp + cfg.landed_addon_inr

    # 2. Arbitrage & Deficit Handling
    if net_load > 0:
        # Discharge BESS if economically superior to RTM (Arbitrage Threshold, Part 5)
        if landed_rtm > (avg_offpeak_cost + cfg.c_deg_inr):
            max_dis_energy = max(0.0, (soc_current - E_min_floor) * cfg.eta_d)
            p_dis = min(net_load, cfg.P_dis_max_kw, max_dis_energy)
            net_load -= p_dis

        # Buy remainder from RTM up to grid limit
        grid_headroom = max(0.0, cfg.P_grid_max_kw - dam_bid_kwh)
        rtm_buy = min(net_load, grid_headroom)
        net_load -= rtm_buy

        # Any remaining deficit is unserved (never breach 5% reserve)
        unserved = max(0.0, net_load)
    else:
        # Surplus energy (excess RE or DAM): charge BESS
        surplus = -net_load
        max_ch_energy = max(0.0, (E_usable_max - soc_current) / cfg.eta_c)
        p_ch = min(surplus, cfg.P_ch_max_kw, max_ch_energy)

    # Opportunistic low-price RTM charging
    if landed_rtm <= 6.50 and p_dis == 0.0:
        max_ch_energy = max(0.0, (E_usable_max - soc_current) / cfg.eta_c) - p_ch
        grid_headroom = max(0.0, cfg.P_grid_max_kw - dam_bid_kwh - rtm_buy)
        extra_ch = min(max(0.0, max_ch_energy), cfg.P_ch_max_kw - p_ch, grid_headroom)
        p_ch += extra_ch
        rtm_buy += extra_ch

    soc_next = soc_current + cfg.eta_c * p_ch - (1.0 / cfg.eta_d) * p_dis
    reserve_compliant = soc_next >= (E_min_floor - 1e-4)

    # Financials (INR)
    served_demand = realized_demand - unserved
    revenue = served_demand * tariff_inr
    energy_cost = (dam_bid_kwh + rtm_buy) * landed_rtm + re_used * cfg.re_cost_inr
    deg_cost = p_dis * cfg.c_deg_inr
    voll_cost = unserved * cfg.c_voll_inr
    ebitda = revenue - energy_cost - deg_cost - voll_cost - cfg.fixed_opex_hr_inr

    return {
        "revenue": revenue,
        "ebitda": ebitda,
        "energy_cost": energy_cost,
        "deg_cost": deg_cost,
        "re_used": re_used,
        "re_avail": actual_re,
        "served_kwh": served_demand,
        "unserved_kwh": unserved,
        "soc_next": soc_next,
        "reserve_compliant": reserve_compliant,
    }