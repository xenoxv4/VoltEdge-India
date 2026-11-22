"""
config.py
Baseline parameters (Part 6) and scenario definitions (Part 7)
All monetary values are in INR; power in kW; energy in kWh.
"""
import math
from dataclasses import dataclass


@dataclass
class ClusterConfig:
    cluster_id: str = "CLUSTER_01"
    # BESS & Swap Inventory
    E_max_kwh: float = 500.0           # Stationary BESS nominal capacity (kWh)
    soh_init: float = 1.00             # Initial State of Health (0 to 1)
    E_min_tech_frac: float = 0.10      # 10% technical minimum SOC
    reserve_frac: float = 0.05         # 5% uncommitted emergency reserve
    S_max_packs: int = 40              # Total swap-pack inventory (packs)
    P_ch_max_kw: float = 150.0         # Max BESS charge power (kW)
    P_dis_max_kw: float = 150.0        # Max BESS discharge power (kW)
    P_grid_max_kw: float = 300.0       # Grid connection limit (kW)
    eta_rt: float = 0.88               # AC-AC round-trip efficiency

    # Economic & Market Parameters (INR)
    base_tariff_inr: float = 18.00     # Prior-day average tariff (INR/kWh)
    tariff_band: float = 0.20          # +/- 20% dynamic pricing corridor
    elasticity: float = -0.30          # Short-run customer price elasticity
    landed_addon_inr: float = 3.00     # Transmission, wheeling, CSS/AS, losses
    re_cost_inr: float = 3.50          # Captive/GEOA solar PPA cost (INR/kWh)
    c_deg_inr: float = 2.00            # Marginal BESS degradation cost (INR/kWh)
    c_voll_inr: float = 50.00          # Value of Lost Load penalty (INR/kWh)
    fixed_opex_hr_inr: float = 600.0   # Allocated hourly fixed OpEx (INR/hr)
    ebitda_margin_target: float = 0.12 # 12% EBITDA margin floor
    iex_price_cap_inr: float = 10.00   # CERC DAM/RTM price cap (INR/kWh)

    @property
    def eta_c(self) -> float:
        return math.sqrt(self.eta_rt)

    @property
    def eta_d(self) -> float:
        return math.sqrt(self.eta_rt)

    @property
    def swap_reserve_packs(self) -> int:
        # Independent 5% swap-pack reserve (no double-counting with stationary BESS)
        return math.ceil(self.reserve_frac * self.S_max_packs)


SCENARIOS = {
    "1_Normal": {
        "mcp_mult": 1.0, "re_mult": 1.0, "demand_mult": 1.0, "soh": 1.00
    },
    "2_High_IEX_Prices": {
        "mcp_mult": 1.9, "re_mult": 1.0, "demand_mult": 1.0, "soh": 1.00
    },
    "3_Low_RE_Generation": {
        "mcp_mult": 1.1, "re_mult": 0.35, "demand_mult": 1.0, "soh": 1.00
    },
    "4_Demand_Spike": {
        "mcp_mult": 1.1, "re_mult": 1.0, "demand_mult": 1.40, "soh": 1.00
    },
    "5_Extreme_Price_Low_RE": {
        "mcp_mult": 2.0, "re_mult": 0.25, "demand_mult": 1.15, "soh": 1.00
    },
    "6_Degraded_BESS": {
        "mcp_mult": 1.0, "re_mult": 1.0, "demand_mult": 1.0, "soh": 0.75
    },
}