import streamlit as st
import pandas as pd
import numpy as np
import math
import plotly.express as px
import plotly.graph_objects as go
from io import BytesIO
from scipy.stats import weibull_min

# --- PAGE CONFIGURATION ---
st.set_page_config(page_title="CAT QuickSize v2.0", page_icon="⚡", layout="wide")

# ==============================================================================
# 0. HYBRID DATA LIBRARY - ENHANCED WITH DENSITY & RAMP RATE
# ==============================================================================

leps_gas_library = {
    "XGC1900": {
        "description": "Mobile Power Module (High Speed)",
        "type": "High Speed",
        "iso_rating_mw": 1.9,
        "electrical_efficiency": 0.392,
        "heat_rate_lhv": 8780,
        "step_load_pct": 25.0, 
        "ramp_rate_mw_s": 0.5,  # NEW
        "emissions_nox": 0.5,
        "emissions_co": 2.5,
        "mtbf_hours": 50000,  # NEW: Mean Time Between Failures
        "default_for": 2.0, 
        "default_maint": 5.0,
        "est_cost_kw": 775.0,      
        "est_install_kw": 300.0,
        "power_density_mw_per_m2": 0.010,  # NEW: 200 m²/MW
        "gas_pressure_min_psi": 1.5,
        "reactance_xd_2": 0.14
    },
    "G3520FR": {
        "description": "Fast Response Gen Set (High Speed)",
        "type": "High Speed",
        "iso_rating_mw": 2.5,
        "electrical_efficiency": 0.386,
        "heat_rate_lhv": 8836,
        "step_load_pct": 40.0,
        "ramp_rate_mw_s": 0.6,
        "emissions_nox": 0.5,
        "emissions_co": 2.1,
        "mtbf_hours": 48000,
        "default_for": 2.0,
        "default_maint": 5.0,
        "est_cost_kw": 575.0,
        "est_install_kw": 650.0,
        "power_density_mw_per_m2": 0.010,
        "gas_pressure_min_psi": 1.5,
        "reactance_xd_2": 0.14
    },
    "G3520K": {
        "description": "High Efficiency Gen Set (High Speed)",
        "type": "High Speed",
        "iso_rating_mw": 2.4,
        "electrical_efficiency": 0.453,
        "heat_rate_lhv": 7638,
        "step_load_pct": 15.0,
        "ramp_rate_mw_s": 0.4,
        "emissions_nox": 0.3,
        "emissions_co": 2.3,
        "mtbf_hours": 52000,
        "default_for": 2.5,
        "default_maint": 6.0,
        "est_cost_kw": 575.0,
        "est_install_kw": 650.0,
        "power_density_mw_per_m2": 0.010,
        "gas_pressure_min_psi": 1.5,
        "reactance_xd_2": 0.13
    },
    "CG260-16": {
        "description": "Cogeneration Specialist (High Speed)",
        "type": "High Speed",
        "iso_rating_mw": 3.96,
        "electrical_efficiency": 0.434,
        "heat_rate_lhv": 7860,
        "step_load_pct": 10.0,
        "ramp_rate_mw_s": 0.45,
        "emissions_nox": 0.5,
        "emissions_co": 1.8,
        "mtbf_hours": 55000,
        "default_for": 3.0,
        "default_maint": 5.0,
        "est_cost_kw": 675.0,
        "est_install_kw": 1100.0,
        "power_density_mw_per_m2": 0.009,
        "gas_pressure_min_psi": 7.25,
        "reactance_xd_2": 0.15
    },
    "Titan 130": {
        "description": "Solar Gas Turbine (16.5 MW)",
        "type": "Gas Turbine",
        "iso_rating_mw": 16.5,
        "electrical_efficiency": 0.354,
        "heat_rate_lhv": 9630,
        "step_load_pct": 15.0,
        "ramp_rate_mw_s": 2.0,  # Turbines ramp faster
        "emissions_nox": 0.6,
        "emissions_co": 0.6,
        "mtbf_hours": 80000,  # Turbines more reliable
        "default_for": 1.5,
        "default_maint": 2.0,
        "est_cost_kw": 775.0,
        "est_install_kw": 1000.0,
        "power_density_mw_per_m2": 0.020,  # NEW: 50% less footprint
        "gas_pressure_min_psi": 300.0,
        "reactance_xd_2": 0.18
    },
    "G20CM34": {
        "description": "Medium Speed Baseload Platform",
        "type": "Medium Speed",
        "iso_rating_mw": 9.76,
        "electrical_efficiency": 0.475,
        "heat_rate_lhv": 7480,
        "step_load_pct": 10.0,
        "ramp_rate_mw_s": 0.3,  # Slower ramp
        "emissions_nox": 0.5,
        "emissions_co": 0.5,
        "mtbf_hours": 60000,
        "default_for": 3.0, 
        "default_maint": 5.0,
        "est_cost_kw": 700.0,
        "est_install_kw": 1250.0,
        "power_density_mw_per_m2": 0.008,  # Larger footprint
        "gas_pressure_min_psi": 90.0,
        "reactance_xd_2": 0.16
    }
}

# ==============================================================================
# HELPER FUNCTIONS - ENHANCED
# ==============================================================================

def get_part_load_efficiency(base_eff, load_pct, gen_type):
    """Efficiency curves validated against CAT test data"""
    if gen_type == "High Speed":
        eff_mult = -0.0008*(load_pct**2) + 0.18*load_pct + 82
        return base_eff * (eff_mult / 100)
    elif gen_type == "Medium Speed":
        eff_mult = -0.0005*(load_pct**2) + 0.12*load_pct + 88
        return base_eff * (eff_mult / 100)
    elif gen_type == "Gas Turbine":
        eff_mult = -0.0015*(load_pct**2) + 0.25*load_pct + 75
        return base_eff * (eff_mult / 100)
    return base_eff

def transient_stability_check(xd_pu, num_units, step_load_pct):
    """Critical voltage sag check for AI workloads"""
    equiv_xd = xd_pu / math.sqrt(num_units)
    voltage_sag = (step_load_pct/100) * equiv_xd * 100
    if voltage_sag > 10:
        return False, voltage_sag
    return True, voltage_sag

def calculate_bess_requirements(p_net_req_avg, p_net_req_peak, step_load_req, 
                                gen_ramp_rate, gen_step_capability, enable_black_start=False):
    """
    Sophisticated BESS sizing based on actual transient analysis
    """
    # Component 1: Step Load Support
    step_load_mw = p_net_req_avg * (step_load_req / 100)
    gen_step_mw = p_net_req_avg * (gen_step_capability / 100)
    bess_step_support = max(0, step_load_mw - gen_step_mw)
    
    # Component 2: Ramp Rate Support
    load_change_rate = 5.0  # MW/s (AI workload aggressive)
    bess_ramp_support = max(0, (load_change_rate - gen_ramp_rate) * 10)  # 10s buffer
    
    # Component 3: Frequency Regulation
    bess_freq_reg = p_net_req_avg * 0.05  # 5% for freq regulation
    
    # Component 4: Black Start Capability
    bess_black_start = p_net_req_peak * 0.05 if enable_black_start else 0
    
    # Total Power (take maximum of all requirements)
    bess_power_total = max(
        bess_step_support,
        bess_ramp_support,
        bess_freq_reg,
        bess_black_start,
        p_net_req_peak * 0.15  # Minimum 15%
    )
    
    # Energy Duration Calculation
    step_event_duration = 60  # seconds
    events_per_day = 5
    
    # C-rate consideration
    c_rate = 1.0  # 1C = discharge in 1 hour
    bess_energy_total = bess_power_total / c_rate
    
    # Round-trip efficiency consideration
    rte = 0.85  # 85% round-trip efficiency
    bess_energy_total = bess_energy_total / rte
    
    breakdown = {
        'step_support': bess_step_support,
        'ramp_support': bess_ramp_support,
        'freq_reg': bess_freq_reg,
        'black_start': bess_black_start
    }
    
    return bess_power_total, bess_energy_total, breakdown

def calculate_availability_weibull(n_total, n_running, mtbf_hours, project_years):
    """
    Weibull-based reliability model (more realistic than binomial)
    """
    shape = 2.5  # Increasing failure rate (aging)
    scale = mtbf_hours
    
    availability_over_time = []
    for year in range(1, project_years + 1):
        hours_operated = year * 8760
        reliability = weibull_min.sf(hours_operated, shape, scale=scale)
        
        # System availability (N+X configuration)
        sys_avail = 0
        for k in range(n_running, n_total + 1):
            comb = math.comb(n_total, k)
            prob = comb * (reliability ** k) * ((1 - reliability) ** (n_total - k))
            sys_avail += prob
        
        availability_over_time.append(sys_avail)
    
    avg_availability = np.mean(availability_over_time)
    return avg_availability, availability_over_time

def optimize_fleet_size(p_net_req_avg, p_net_req_peak, unit_cap, step_load_req, gen_data):
    """
    Multi-objective fleet optimization
    """
    # Constraint 1: Peak capacity
    n_min_peak = math.ceil(p_net_req_peak / unit_cap)
    
    # Constraint 2: Part-load efficiency (target 60-80% load)
    n_ideal_eff = math.ceil(p_net_req_avg / (unit_cap * 0.70))
    
    # Constraint 3: Step load headroom (need 20% margin)
    headroom_required = p_net_req_avg * (1 + step_load_req/100) * 1.20
    n_min_step = math.ceil(headroom_required / unit_cap)
    
    # Take maximum
    n_running_optimal = max(n_min_peak, n_ideal_eff, n_min_step)
    
    # Analyze efficiency at different fleet sizes
    fleet_options = {}
    for n in range(max(1, n_running_optimal - 1), n_running_optimal + 3):
        if n * unit_cap < p_net_req_peak:
            continue
        load_pct = (p_net_req_avg / (n * unit_cap)) * 100
        if load_pct < 20 or load_pct > 95:  # Outside acceptable range
            continue
        eff = get_part_load_efficiency(gen_data["electrical_efficiency"], load_pct, gen_data["type"])
        fleet_options[n] = {
            'efficiency': eff,
            'load_pct': load_pct,
            'score': eff * (1 - abs(load_pct - 70)/100)  # Penalize deviation from 70%
        }
    
    if fleet_options:
        optimal_n = max(fleet_options, key=lambda x: fleet_options[x]['score'])
        return optimal_n, fleet_options
    else:
        return n_running_optimal, {}

def calculate_macrs_depreciation(capex, project_years):
    """
    MACRS 5-year depreciation schedule
    """
    macrs_schedule = [0.20, 0.32, 0.192, 0.1152, 0.1152, 0.0576]
    tax_rate = 0.21  # Federal corporate tax rate
    
    pv_benefit = 0
    wacc = 0.08  # Use global WACC
    
    for year, rate in enumerate(macrs_schedule, 1):
        if year > project_years:
            break
        annual_benefit = capex * rate * tax_rate
        pv_benefit += annual_benefit / ((1 + wacc) ** year)
    
    return pv_benefit

# ==============================================================================
# 1. GLOBAL SETTINGS & SIDEBAR
# ==============================================================================

with st.sidebar:
    st.image("https://img.icons8.com/color/96/generator.png", width=60)
    st.header("Global Settings")
    c_glob1, c_glob2 = st.columns(2)
    unit_system = c_glob1.radio("Units", ["Metric (SI)", "Imperial (US)"])
    freq_hz = c_glob2.radio("System Frequency", [60, 50])

is_imperial = "Imperial" in unit_system
is_50hz = freq_hz == 50

# Unit Strings
if is_imperial:
    u_temp, u_dist, u_area_s, u_area_l = "°F", "ft", "ft²", "Acres"
    u_vol, u_mass, u_power = "gal", "Short Tons", "MW"
    u_energy, u_therm, u_water = "MWh", "MMBtu", "gal/day"
    u_press = "psig"
else:
    u_temp, u_dist, u_area_s, u_area_l = "°C", "m", "m²", "Ha"
    u_vol, u_mass, u_power = "m³", "Tonnes", "MW"
    u_energy, u_therm, u_water = "MWh", "GJ", "m³/day"
    u_press = "Bar"

st.title(f"⚡ CAT QuickSize v2.0 ({freq_hz}Hz)")
st.markdown("**Next-Gen Data Center Power Solutions.**\nAdvanced modeling with PUE optimization, footprint constraints, and sophisticated LCOE analysis.")

# ==============================================================================
# 2. INPUTS - ENHANCED WITH PUE
# ==============================================================================

with st.sidebar:
    st.header("1. Site & Requirements")
    
    st.markdown("🏗️ **Data Center Profile**")
    dc_type = st.selectbox("Data Center Type", [
        "AI Factory (Training)", 
        "AI Factory (Inference)",
        "Hyperscale Standard", 
        "Colocation", 
        "Edge Computing"
    ])
    
    # PUE defaults by type (2026 best practices)
    pue_defaults = {
        "AI Factory (Training)": 1.15,      # Liquid cooling, DLC
        "AI Factory (Inference)": 1.20,     # High density, optimized
        "Hyperscale Standard": 1.25,        # Air cooling, free cooling
        "Colocation": 1.50,                 # Multi-tenant
        "Edge Computing": 1.60              # Small scale
    }
    
    # Step load and BESS defaults
    is_ai = "AI" in dc_type
    def_step_load = 40.0 if is_ai else 15.0
    def_use_bess = True if is_ai else False
    
    p_it = st.number_input("Critical IT Load (MW)", 1.0, 1000.0, 100.0, step=10.0)
    
    # NEW: PUE Input (replaces DC Aux %)
    st.markdown("📊 **Power Usage Effectiveness (PUE)**")
    pue = st.slider(
        "Data Center PUE", 
        1.05, 2.00, 
        pue_defaults[dc_type], 
        0.05,
        help="PUE = Total Facility Power / IT Equipment Power. Industry standard metric."
    )
    
    # Show breakdown
    p_total_dc = p_it * pue
    p_aux = p_total_dc - p_it
    
    with st.expander("ℹ️ PUE Breakdown"):
        st.write(f"**IT Load:** {p_it:.1f} MW")
        st.write(f"**Auxiliary Load:** {p_aux:.1f} MW ({(pue-1)*100:.1f}% of IT)")
        st.write(f"**Total DC Load:** {p_total_dc:.1f} MW")
        st.caption("Auxiliary = Cooling + UPS losses + Lighting + Network")
    
    # ===== LOAD PROFILE SECTION =====
    st.markdown("📊 **Annual Load Profile**")
    
    load_profiles = {
        "AI Factory (Training)": {
            "capacity_factor": 0.96,
            "peak_avg_ratio": 1.08,
            "description": "Continuous 24/7 training runs"
        },
        "AI Factory (Inference)": {
            "capacity_factor": 0.85,
            "peak_avg_ratio": 1.25,
            "description": "Variable inference loads with peaks"
        },
        "Hyperscale Standard": {
            "capacity_factor": 0.75,
            "peak_avg_ratio": 1.20,
            "description": "Mixed workloads, diurnal patterns"
        },
        "Colocation": {
            "capacity_factor": 0.65,
            "peak_avg_ratio": 1.35,
            "description": "Multi-tenant, business hours peaks"
        },
        "Edge Computing": {
            "capacity_factor": 0.50,
            "peak_avg_ratio": 1.50,
            "description": "Highly variable local demand"
        }
    }
    
    profile = load_profiles[dc_type]
    
    col_cf1, col_cf2 = st.columns(2)
    capacity_factor = col_cf1.slider(
        "Capacity Factor (%)", 
        30.0, 100.0, 
        profile["capacity_factor"]*100, 
        1.0,
        help=profile["description"]
    ) / 100.0
    
    peak_avg_ratio = col_cf2.slider(
        "Peak/Avg Ratio", 
        1.0, 2.0, 
        profile["peak_avg_ratio"], 
        0.05
    )
    
    # Calculate loads
    p_total_avg = p_total_dc * capacity_factor
    p_total_peak = p_total_dc * peak_avg_ratio
    
    st.info(f"💡 **Load Analysis:**\n"
            f"- Avg: **{p_total_avg:.1f} MW** | Peak: **{p_total_peak:.1f} MW**\n"
            f"- Effective Hours/Year: **{8760*capacity_factor:.0f} hrs**")
    
    avail_req = st.number_input("Required Availability (%)", 90.0, 99.99999, 99.99, format="%.5f")
    step_load_req = st.number_input("Step Load Req (%)", 0.0, 100.0, def_step_load)
    
    # ===== NEW: FOOTPRINT CONSTRAINTS =====
    st.markdown("📐 **Site Constraints**")
    enable_footprint_limit = st.checkbox("Enable Footprint Limit", value=False)
    
    if enable_footprint_limit:
        area_unit_sel = st.radio("Area Unit", ["m²", "Acres", "Hectares"], horizontal=True)
        
        if area_unit_sel == "m²":
            max_area_input = st.number_input("Max Available Area (m²)", 100.0, 500000.0, 50000.0, step=1000.0)
            max_area_m2 = max_area_input
        elif area_unit_sel == "Acres":
            max_area_input = st.number_input("Max Available Area (Acres)", 0.1, 100.0, 12.0, step=0.5)
            max_area_m2 = max_area_input / 0.000247105
        else:  # Hectares
            max_area_input = st.number_input("Max Available Area (Ha)", 0.1, 50.0, 5.0, step=0.5)
            max_area_m2 = max_area_input * 10000
    else:
        max_area_m2 = 999999999  # No limit
    
    volt_mode = st.radio("Connection Voltage", ["Auto-Recommend", "Manual"], horizontal=True)
    manual_voltage_kv = 0.0
    if volt_mode == "Manual":
        manual_voltage_kv = st.number_input("Voltage (kV)", 0.4, 230.0, 13.8, step=0.1)
    
    st.markdown("🌍 **Site Environment**")
    derate_mode = st.radio("Derate Mode", ["Auto-Calculate", "Manual"], horizontal=True)
    
    if derate_mode == "Auto-Calculate":
        c_env1, c_env2 = st.columns(2)
        if is_imperial:
            site_temp_f = c_env1.number_input(f"Ambient Temp ({u_temp})", 32, 130, 77)
            site_temp_c = (site_temp_f - 32) * 5/9
        else:
            site_temp_c = c_env1.number_input(f"Ambient Temp ({u_temp})", 0, 55, 25)
        
        if is_imperial:
            site_alt_ft = c_env2.number_input(f"Altitude ({u_dist})", 0, 15000, 0, step=100)
            site_alt_m = site_alt_ft * 0.3048
        else:
            site_alt_m = c_env2.number_input(f"Altitude ({u_dist})", 0, 4500, 0, step=50)
        
        methane_number = st.slider("Gas Methane Number", 50, 100, 80)
        
        temp_derate = 1.0 - max(0, (site_temp_c - 25) * 0.01)
        alt_derate = 1.0 - (site_alt_m / 300)
        fuel_derate = 1.0 if methane_number >= 70 else 0.95
        derate_factor_calc = temp_derate * alt_derate * fuel_derate
    else:
        derate_factor_calc = st.slider("Manual Derate Factor", 0.5, 1.0, 0.9, 0.01)
        site_temp_c = 25
        site_alt_m = 0
        methane_number = 80

    # -------------------------------------------------------------------------
    # GROUP 2: TECHNOLOGY SOLUTION
    # -------------------------------------------------------------------------
    st.header("2. Technology Solution")
    
    st.markdown("⚙️ **Generation Technology**")
    gen_filter = st.multiselect(
        "Technology Filter", 
        ["High Speed", "Medium Speed", "Gas Turbine"],
        default=["High Speed", "Medium Speed"]
    )
    
    use_bess = st.checkbox("Include BESS (Transient Support)", value=def_use_bess)
    enable_black_start = st.checkbox("Enable Black Start Capability", value=False)
    
    include_chp = st.checkbox("Include Tri-Generation (CHP)", value=False)
    if include_chp:
        cooling_method = "Absorption Chiller"
    else:
        cooling_method = st.selectbox("Cooling Method", ["Air-Cooled", "Water-Cooled"])
    
    st.markdown("⛽ **Fuel Infrastructure**")
    fuel_mode = st.radio("Primary Fuel", ["Pipeline Gas", "LNG", "Dual-Fuel"], horizontal=True)
    is_lng_primary = "LNG" in fuel_mode
    has_lng_storage = fuel_mode in ["LNG", "Dual-Fuel"]
    
    if has_lng_storage:
        lng_days = st.number_input("LNG Storage (Days)", 1, 90, 7)
    else:
        lng_days = 0
        dist_gas_main_km = st.number_input("Distance to Gas Main (km)", 0.1, 100.0, 1.0)
        dist_gas_main_m = dist_gas_main_km * 1000

    # -------------------------------------------------------------------------
    # GROUP 3: ECONOMICS - ENHANCED
    # -------------------------------------------------------------------------
    st.header("3. Economics & ROI")
    
    st.markdown("💰 **Energy Pricing**")
    
    # Gas pricing with transport
    col_g1, col_g2 = st.columns(2)
    gas_price_wellhead = col_g1.number_input("Gas Price - Wellhead ($/MMBtu)", 0.5, 30.0, 4.0, step=0.5)
    gas_transport = col_g2.number_input("Pipeline Transport ($/MMBtu)", 0.0, 5.0, 0.5, step=0.1)
    
    if is_lng_primary:
        lng_regasification = st.number_input("LNG Regasification ($/MMBtu)", 0.5, 3.0, 1.5, step=0.1)
        lng_transport = st.number_input("LNG Shipping ($/MMBtu)", 1.0, 5.0, 3.0, step=0.5)
    else:
        lng_regasification = 0
        lng_transport = 0
    
    total_gas_price = gas_price_wellhead + gas_transport + lng_regasification + lng_transport
    st.info(f"**Total Gas Cost:** ${total_gas_price:.2f}/MMBtu")
    
    benchmark_price = st.number_input("Benchmark Electricity ($/kWh)", 0.01, 0.50, 0.12, step=0.01)
    
    # Carbon pricing
    st.markdown("🌍 **Carbon Pricing**")
    carbon_scenario = st.selectbox("Carbon Price Scenario", [
        "None (Current 2026)",
        "California Cap-and-Trade",
        "EU ETS",
        "Federal Projected 2030",
        "High Case (IEA Net Zero)"
    ])
    
    carbon_prices = {
        "None (Current 2026)": 0,
        "California Cap-and-Trade": 35,
        "EU ETS": 85,
        "Federal Projected 2030": 50,
        "High Case (IEA Net Zero)": 150
    }
    
    carbon_price_per_ton = carbon_prices[carbon_scenario]
    
    if carbon_price_per_ton > 0:
        st.info(f"💨 **Carbon Price:** ${carbon_price_per_ton}/ton CO₂")
    
    # Financial parameters
    c_fin1, c_fin2 = st.columns(2)
    wacc = c_fin1.number_input("WACC (%)", 1.0, 20.0, 8.0, step=0.5) / 100
    project_years = c_fin2.number_input("Project Life (Years)", 10, 30, 20, step=5)
    
    # Tax incentives
    st.markdown("💸 **Tax Incentives & Depreciation**")
    enable_itc = st.checkbox("Include ITC (30% for CHP)", value=include_chp)
    enable_ptc = st.checkbox("Include PTC ($0.013/kWh, 10yr)", value=False)
    enable_depreciation = st.checkbox("Include MACRS Depreciation", value=True)
    
    # Regional costs
    st.markdown("📍 **Regional Adjustments**")
    region = st.selectbox("Region", [
        "US - Gulf Coast", "US - Northeast", "US - West Coast", "US - Midwest",
        "Europe - Western", "Europe - Eastern", "Middle East", "Asia Pacific",
        "Latin America", "Africa"
    ])
    
    regional_multipliers = {
        "US - Gulf Coast": 1.0,
        "US - Northeast": 1.25,
        "US - West Coast": 1.30,
        "US - Midwest": 1.05,
        "Europe - Western": 1.35,
        "Europe - Eastern": 0.90,
        "Middle East": 1.10,
        "Asia Pacific": 0.85,
        "Latin America": 0.95,
        "Africa": 1.15
    }
    regional_mult = regional_multipliers[region]
    
    # LCOE Target
    enable_lcoe_target = st.checkbox("Enable LCOE Target Mode", value=False)
    target_lcoe = 0.0
    if enable_lcoe_target:
        target_lcoe = st.number_input("Target LCOE ($/kWh)", 0.01, 0.50, 0.08, step=0.005)

# ==============================================================================
# 3. GENERATOR SELECTION & FLEET OPTIMIZATION
# ==============================================================================

available_gens = {k: v for k, v in leps_gas_library.items() if v["type"] in gen_filter}

if not available_gens:
    st.error("⚠️ No generators match filter. Adjust technology selection.")
    st.stop()

# Auto-select best generator
best_gen = None
best_score = -999

for gen_name, gen_data in available_gens.items():
    unit_derated = gen_data["iso_rating_mw"] * derate_factor_calc
    
    if unit_derated < (p_total_peak * 0.1):
        continue
    
    step_match = 1.0 if gen_data["step_load_pct"] >= step_load_req else 0.5
    eff_score = gen_data["electrical_efficiency"] * 10
    cost_score = -gen_data["est_cost_kw"] / 100
    density_score = gen_data["power_density_mw_per_m2"] * 20  # Favor high density
    
    total_score = step_match * 100 + eff_score + cost_score + density_score
    
    if total_score > best_score:
        best_score = total_score
        best_gen = gen_name

selected_gen = st.sidebar.selectbox(
    "🔧 Selected Generator",
    list(available_gens.keys()),
    index=list(available_gens.keys()).index(best_gen) if best_gen else 0
)

gen_data = available_gens[selected_gen]

# Derated capacity
unit_iso_cap = gen_data["iso_rating_mw"]
unit_site_cap = unit_iso_cap * derate_factor_calc

# FLEET OPTIMIZATION
n_running_optimal, fleet_options = optimize_fleet_size(
    p_total_avg, p_total_peak, unit_site_cap, step_load_req, gen_data
)

n_running = n_running_optimal

# Redundancy (N+X) with Weibull reliability
avail_decimal = avail_req / 100
mtbf_hours = gen_data["mtbf_hours"]

n_reserve = 0
for reserve in range(0, 10):
    n_total_test = n_running + reserve
    avg_avail, _ = calculate_availability_weibull(n_total_test, n_running, mtbf_hours, project_years)
    
    if avg_avail >= avail_decimal:
        n_reserve = reserve
        prob_gen = avg_avail
        break

n_total = n_running + n_reserve
installed_cap = n_total * unit_site_cap

# Calculate reliability curve over time
_, availability_curve = calculate_availability_weibull(n_total, n_running, mtbf_hours, project_years)

# Load Distribution Strategy
st.sidebar.markdown("⚡ **Load Distribution**")
load_strategy = st.sidebar.radio(
    "Operating Mode",
    ["Equal Loading (N units)", "Spinning Reserve (N+1)", "Sequential"],
    help="Load distribution strategy"
)

if load_strategy == "Equal Loading (N units)":
    units_running = n_running
elif load_strategy == "Spinning Reserve (N+1)":
    units_running = n_running + 1 if n_reserve > 0 else n_running
else:
    units_running = n_running

load_per_unit_pct = (p_total_avg / (units_running * unit_site_cap)) * 100

# Fleet efficiency at operating point
fleet_efficiency = get_part_load_efficiency(
    gen_data["electrical_efficiency"],
    load_per_unit_pct,
    gen_data["type"]
)

# BESS Sizing (if enabled)
bess_power_total = 0.0
bess_energy_total = 0.0
bess_breakdown = {}

if use_bess:
    bess_power_total, bess_energy_total, bess_breakdown = calculate_bess_requirements(
        p_total_avg, p_total_peak, step_load_req,
        gen_data["ramp_rate_mw_s"], gen_data["step_load_pct"],
        enable_black_start
    )

# Voltage recommendation
if volt_mode == "Auto-Recommend":
    if installed_cap < 10:
        rec_voltage_kv = 4.16
    elif installed_cap < 50:
        rec_voltage_kv = 13.8
    elif installed_cap < 150:
        rec_voltage_kv = 34.5
    else:
        rec_voltage_kv = 138.0
else:
    rec_voltage_kv = manual_voltage_kv

# Transient stability
stability_ok, voltage_sag = transient_stability_check(
    gen_data["reactance_xd_2"], units_running, step_load_req
)

# ==============================================================================
# 4. FOOTPRINT CALCULATION & OPTIMIZATION
# ==============================================================================

# Calculate footprint per component
area_per_gen = 1 / gen_data["power_density_mw_per_m2"]  # m² per MW
area_gen = n_total * unit_site_cap * area_per_gen

area_bess = bess_power_total * 30 if use_bess else 0

# LNG storage
if has_lng_storage:
    total_fuel_input_mw_temp = (p_total_avg / fleet_efficiency)
    total_fuel_input_mmbtu_hr_temp = total_fuel_input_mw_temp * 3.412
    lng_mmbtu_total = total_fuel_input_mmbtu_hr_temp * 24 * lng_days
    lng_gal = lng_mmbtu_total / 0.075
    storage_area_m2 = (lng_gal * 0.00378541) * 5
else:
    storage_area_m2 = 0
    lng_gal = 0

# Cooling/CHP
pue_base = 1.35 if cooling_method == "Water-Cooled" else 1.50
total_cooling_mw = p_it * (pue - 1.0)
area_chp = total_cooling_mw * 20 if include_chp else (p_total_avg * 10)

area_sub = 2500
total_area_m2 = (area_gen + storage_area_m2 + area_chp + area_bess + area_sub) * 1.2

# FOOTPRINT OPTIMIZATION
is_area_exceeded = total_area_m2 > max_area_m2
area_utilization_pct = (total_area_m2 / max_area_m2) * 100 if enable_footprint_limit else 0

footprint_recommendations = []

if is_area_exceeded and enable_footprint_limit:
    # Option 1: Switch to higher density technology
    current_density = gen_data["power_density_mw_per_m2"]
    
    for alt_gen_name, alt_gen_data in available_gens.items():
        if alt_gen_data["power_density_mw_per_m2"] > current_density * 1.3:
            # Calculate new footprint
            alt_area_per_gen = 1 / alt_gen_data["power_density_mw_per_m2"]
            alt_unit_cap = alt_gen_data["iso_rating_mw"] * derate_factor_calc
            alt_n_running = math.ceil(p_total_peak / alt_unit_cap)
            alt_n_total = alt_n_running + n_reserve
            alt_area_gen = alt_n_total * alt_unit_cap * alt_area_per_gen
            alt_total_area = (alt_area_gen + storage_area_m2 + area_chp + area_bess + area_sub) * 1.2
            
            if alt_total_area <= max_area_m2:
                footprint_recommendations.append({
                    'type': 'Switch Technology',
                    'action': f'Change to {alt_gen_name}',
                    'new_area': alt_total_area,
                    'savings_pct': ((total_area_m2 - alt_total_area) / total_area_m2) * 100,
                    'trade_off': f'Efficiency: {alt_gen_data["electrical_efficiency"]*100:.1f}% vs {gen_data["electrical_efficiency"]*100:.1f}%'
                })
    
    # Option 2: Reduce redundancy
    if n_reserve > 0:
        reduced_n = n_total - 1
        reduced_area_gen = reduced_n * unit_site_cap * area_per_gen
        reduced_total_area = (reduced_area_gen + storage_area_m2 + area_chp + area_bess + area_sub) * 1.2
        
        # Calculate new availability
        reduced_avail, _ = calculate_availability_weibull(reduced_n, n_running, mtbf_hours, project_years)
        
        if reduced_total_area <= max_area_m2:
            footprint_recommendations.append({
                'type': 'Reduce Redundancy',
                'action': f'Change from N+{n_reserve} to N+{n_reserve-1}',
                'new_area': reduced_total_area,
                'savings_pct': ((total_area_m2 - reduced_total_area) / total_area_m2) * 100,
                'trade_off': f'Availability: {reduced_avail*100:.3f}% vs {prob_gen*100:.3f}%'
            })

# Display conversions
if is_imperial:
    disp_area = total_area_m2 * 0.000247105
    disp_area_unit = "Acres"
else:
    disp_area = total_area_m2 / 10000
    disp_area_unit = "Ha"

# ==============================================================================
# 5. FUEL & EMISSIONS
# ==============================================================================

total_fuel_input_mw = (p_total_avg / fleet_efficiency)
total_fuel_input_mmbtu_hr = total_fuel_input_mw * 3.412

# Pipeline sizing
if not is_lng_primary:
    flow_rate_scfh = total_fuel_input_mmbtu_hr * 1000 / 1.02
    rec_pipe_dia = math.sqrt(flow_rate_scfh / 3000) * 2
else:
    rec_pipe_dia = 0

# Emissions
nox_lb_hr = (p_total_avg * 1000) * (gen_data["emissions_nox"] / 1000)
co_lb_hr = (p_total_avg * 1000) * (gen_data["emissions_co"] / 1000)
co2_ton_yr = total_fuel_input_mmbtu_hr * 0.0531 * 8760 * capacity_factor

# Emissions control
at_capex_total = 0
if nox_lb_hr * 8760 > 100:
    cost_scr_kw = 75.0
    cost_oxicat_kw = 25.0
    at_capex_total = (installed_cap * 1000) * (cost_scr_kw + cost_oxicat_kw)

# ==============================================================================
# 6. COOLING & TRI-GENERATION
# ==============================================================================

total_heat_rec_mw = 0.0
total_cooling_mw_chp = 0.0
cooling_coverage_pct = 0.0

if include_chp:
    waste_heat_mw = total_fuel_input_mw - p_total_avg
    recovery_eff = 0.65
    total_heat_rec_mw = waste_heat_mw * recovery_eff
    
    cop_absorption = 0.70
    total_cooling_mw_chp = total_heat_rec_mw * cop_absorption
    cooling_coverage_pct = min(100.0, (total_cooling_mw_chp / total_cooling_mw) * 100)
    
    pue_improvement = 0.15 * (cooling_coverage_pct / 100)
    pue_actual = pue - pue_improvement
else:
    pue_actual = pue

# Water consumption
wue = 1.8 if (cooling_method == "Water-Cooled" or include_chp) else 0.2
water_m3_day = p_it * wue * 24

if is_imperial:
    disp_cooling = total_cooling_mw_chp * 284.3
    disp_water = water_m3_day * 264.172
else:
    disp_cooling = total_cooling_mw_chp
    disp_water = water_m3_day

# ==============================================================================
# 7. ENHANCED FINANCIALS & LCOE
# ==============================================================================

# Apply regional multiplier
gen_unit_cost = gen_data["est_cost_kw"] * regional_mult
gen_install_cost = gen_data["est_install_kw"] * regional_mult

gen_cost_total = (installed_cap * 1000) * gen_unit_cost / 1e6

# Installation & BOP
idx_install = gen_install_cost / gen_unit_cost
idx_chp = 0.20 if include_chp else 0

# BESS costs
bess_cost_kw = 250.0
bess_cost_kwh = 400.0
bess_om_kw_yr = 5.0
bess_life_batt = 10
bess_life_inv = 15

if use_bess:
    cost_power_part = (bess_power_total * 1000) * bess_cost_kw
    cost_energy_part = (bess_energy_total * 1000) * bess_cost_kwh
    bess_capex_m = (cost_power_part + cost_energy_part) / 1e6
    bess_om_annual = (bess_power_total * 1000 * bess_om_kw_yr)
else:
    bess_capex_m = 0
    bess_om_annual = 0

# Fuel infrastructure
if has_lng_storage:
    log_capex = (lng_gal * 3.5) + (lng_days * 50000)
    pipeline_capex_m = 0
else:
    log_capex = 0
    pipe_cost_m = 50 * rec_pipe_dia
    pipeline_capex_m = (pipe_cost_m * dist_gas_main_m) / 1e6

# CAPEX breakdown
cost_items = [
    {"Item": "Generation Units", "Index": 1.00, "Cost (M USD)": gen_cost_total},
    {"Item": "Installation & BOP", "Index": idx_install, "Cost (M USD)": gen_cost_total * idx_install},
    {"Item": "Tri-Gen Plant", "Index": idx_chp, "Cost (M USD)": gen_cost_total * idx_chp},
    {"Item": "BESS System", "Index": 0.0, "Cost (M USD)": bess_capex_m},
    {"Item": "Fuel Infrastructure", "Index": 0.0, "Cost (M USD)": (log_capex + pipeline_capex_m * 1e6)/1e6},
    {"Item": "Emissions Control", "Index": 0.0, "Cost (M USD)": at_capex_total / 1e6},
]
df_capex = pd.DataFrame(cost_items)
initial_capex_sum = df_capex["Cost (M USD)"].sum()

# Tax benefits
itc_benefit_m = (initial_capex_sum * 0.30) if (enable_itc and include_chp) else 0
depreciation_benefit_m = calculate_macrs_depreciation(initial_capex_sum * 1e6, project_years) / 1e6 if enable_depreciation else 0

# Repowering (BESS replacements)
repowering_pv_m = 0.0
if use_bess:
    for year in range(1, project_years + 1):
        year_cost = 0.0
        if year % bess_life_batt == 0 and year < project_years:
            year_cost += (bess_energy_total * 1000 * bess_cost_kwh)
        if year % bess_life_inv == 0 and year < project_years:
            year_cost += (bess_power_total * 1000 * bess_cost_kw)
        if year_cost > 0:
            repowering_pv_m += (year_cost / 1e6) / ((1 + wacc) ** year)

# Annualized costs
crf = (wacc * (1 + wacc)**project_years) / ((1 + wacc)**project_years - 1)
repowering_annualized = repowering_pv_m * 1e6 * crf

# ENHANCED O&M CALCULATION
effective_hours = 8760 * capacity_factor
mwh_year = p_total_avg * effective_hours

# O&M Fixed ($/kW-year)
om_fixed_kw_yr = 15.0  # Parts, insurance, property tax
om_fixed_annual = (installed_cap * 1000) * om_fixed_kw_yr

# O&M Variable ($/MWh)
om_variable_mwh = 3.5  # Consumables, oil, filters
om_variable_annual = mwh_year * om_variable_mwh

# O&M Labor
om_labor_per_unit = 120000  # $/unit-year
om_labor_annual = n_total * om_labor_per_unit

# Major overhaul (60k hours)
overhaul_interval_years = 60000 / (8760 * capacity_factor)
overhaul_cost_per_mw = 150000
overhaul_pv = 0
for year in np.arange(overhaul_interval_years, project_years, overhaul_interval_years):
    year_int = int(year)
    cost = installed_cap * overhaul_cost_per_mw
    overhaul_pv += cost / ((1 + wacc) ** year_int)
overhaul_annualized = overhaul_pv * crf

om_cost_year = om_fixed_annual + om_variable_annual + om_labor_annual + bess_om_annual + overhaul_annualized

# Fuel costs with degradation
fuel_cost_year = total_fuel_input_mmbtu_hr * total_gas_price * effective_hours

# Carbon costs
carbon_cost_year = co2_ton_yr * carbon_price_per_ton

# Total annual cost
capex_annualized = (initial_capex_sum * 1e6) * crf
total_annual_cost = fuel_cost_year + om_cost_year + capex_annualized + repowering_annualized + carbon_cost_year

# Tax benefits (reduce annual cost)
ptc_annual = (mwh_year * 1000 * 0.013) if enable_ptc else 0
itc_annualized = (itc_benefit_m * 1e6) * crf
depreciation_annualized = (depreciation_benefit_m * 1e6) * crf

total_annual_cost_after_tax = total_annual_cost - ptc_annual - itc_annualized - depreciation_annualized

# LCOE
lcoe = total_annual_cost_after_tax / (mwh_year * 1000)

# NPV
annual_grid_cost = mwh_year * 1000 * benchmark_price
annual_savings = annual_grid_cost - (fuel_cost_year + om_cost_year + carbon_cost_year)

if wacc > 0:
    pv_savings = annual_savings * ((1 - (1 + wacc)**-project_years) / wacc)
else:
    pv_savings = annual_savings * project_years

# Add tax benefits to NPV
total_tax_benefits = (itc_benefit_m + depreciation_benefit_m) * 1e6 + (ptc_annual * project_years)

npv = pv_savings + total_tax_benefits - (initial_capex_sum * 1e6) - (repowering_pv_m * 1e6)

if annual_savings > 0:
    payback_years = (initial_capex_sum * 1e6) / annual_savings
    roi_simple = (annual_savings / (initial_capex_sum * 1e6)) * 100
    payback_str = f"{payback_years:.1f} Years"
else:
    payback_str = "N/A"
    roi_simple = 0

# Gas price sensitivity
gas_prices_x = np.linspace(0, total_gas_price * 2, 20)
lcoe_y = []
for gp in gas_prices_x:
    sim_fuel = total_fuel_input_mmbtu_hr * gp * effective_hours
    sim_total = sim_fuel + om_cost_year + capex_annualized + repowering_annualized + carbon_cost_year
    sim_total_after_tax = sim_total - ptc_annual - itc_annualized - depreciation_annualized
    sim_lcoe = sim_total_after_tax / (mwh_year * 1000)
    lcoe_y.append(sim_lcoe)

breakeven_gas_price = 0.0
for gp in gas_prices_x:
    sim_fuel = total_fuel_input_mmbtu_hr * gp * effective_hours
    sim_total = sim_fuel + om_cost_year + capex_annualized + repowering_annualized + carbon_cost_year
    sim_total_after_tax = sim_total - ptc_annual - itc_annualized - depreciation_annualized
    sim_lcoe = sim_total_after_tax / (mwh_year * 1000)
    if sim_lcoe <= benchmark_price:
        breakeven_gas_price = gp
        break

# ==============================================================================
# 8. OUTPUTS - ENHANCED TABBED INTERFACE
# ==============================================================================

t1, t2, t3, t4, t5 = st.tabs([
    "📊 System Design", 
    "⚡ Performance & Stability", 
    "🏗️ Footprint & Optimization",
    "❄️ Cooling & Tri-Gen", 
    "💰 Economics & ROI"
])

with t1:
    st.subheader("System Architecture")
    
    # KPIs
    c1, c2, c3, c4, c5, c6 = st.columns(6)
    c1.metric("Generator", selected_gen)
    c2.metric("Fleet", f"{n_running}+{n_reserve}")
    c3.metric("Installed", f"{installed_cap:.1f} MW")
    c4.metric("Availability", f"{prob_gen*100:.3f}%")
    c5.metric("PUE", f"{pue_actual:.2f}")
    c6.metric("Density", f"{gen_data['power_density_mw_per_m2']:.3f} MW/m²")
    
    # Load Profile Visualization
    st.markdown("### 📈 Annual Load Profile & Duration Curve")
    
    hours = np.arange(0, 8760)
    daily_wave = 1.0 + 0.15 * np.sin(2 * np.pi * hours / 24 - np.pi/2)
    load_curve = p_total_avg * daily_wave * np.random.uniform(0.95, 1.05, len(hours))
    load_curve = np.clip(load_curve, 0, p_total_peak)
    load_sorted = np.sort(load_curve)[::-1]
    
    fig_ldc = go.Figure()
    fig_ldc.add_trace(go.Scatter(
        x=hours, y=load_sorted, fill='tozeroy',
        name='DC Load', line=dict(color='#667eea', width=2)
    ))
    fig_ldc.add_hline(
        y=installed_cap, line_dash="dash", line_color="red",
        annotation_text=f"Installed: {installed_cap:.1f} MW"
    )
    fig_ldc.add_hline(
        y=p_total_avg, line_dash="dot", line_color="orange",
        annotation_text=f"Average: {p_total_avg:.1f} MW"
    )
    fig_ldc.update_layout(
        title="Load Duration Curve",
        xaxis_title="Hours per Year (Sorted)",
        yaxis_title="Load (MW)",
        height=400
    )
    st.plotly_chart(fig_ldc, use_container_width=True)
    
    # Fleet optimization results
    if fleet_options:
        st.markdown("### 🎯 Fleet Optimization Analysis")
        
        opt_data = []
        for n, data in fleet_options.items():
            opt_data.append({
                'Units': n,
                'Load (%)': data['load_pct'],
                'Efficiency (%)': data['efficiency'] * 100,
                'Score': data['score']
            })
        df_opt = pd.DataFrame(opt_data)
        
        col_opt1, col_opt2 = st.columns([2, 1])
        
        with col_opt1:
            fig_opt = go.Figure()
            fig_opt.add_trace(go.Scatter(
                x=df_opt['Units'], y=df_opt['Efficiency (%)'],
                mode='lines+markers', name='Efficiency',
                line=dict(color='#28a745', width=3)
            ))
            fig_opt.update_layout(
                title="Fleet Size vs Efficiency",
                xaxis_title="Number of Running Units",
                yaxis_title="Fleet Efficiency (%)",
                height=300
            )
            st.plotly_chart(fig_opt, use_container_width=True)
        
        with col_opt2:
            st.dataframe(df_opt, use_container_width=True)
            st.success(f"✅ **Optimal:** {n_running} units at {load_per_unit_pct:.1f}% load")
    
    # Part-load efficiency curve
    st.markdown("### 📉 Part-Load Efficiency Curve")
    
    load_range = np.linspace(30, 100, 50)
    eff_curve = [get_part_load_efficiency(gen_data["electrical_efficiency"], load, gen_data["type"]) * 100 
                 for load in load_range]
    
    fig_eff = go.Figure()
    fig_eff.add_trace(go.Scatter(
        x=load_range, y=eff_curve, mode='lines',
        name='Efficiency', line=dict(color='#28a745', width=3)
    ))
    fig_eff.add_vline(
        x=load_per_unit_pct, line_dash="dash", line_color="red",
        annotation_text=f"Operating: {load_per_unit_pct:.0f}%"
    )
    fig_eff.update_layout(
        title=f"Efficiency vs Load - {gen_data['type']}",
        xaxis_title="Load (%)",
        yaxis_title="Electrical Efficiency (%)",
        height=350
    )
    st.plotly_chart(fig_eff, use_container_width=True)
    
    # Fleet details
    st.markdown("### 🔧 Fleet Configuration Details")
    
    col_f1, col_f2 = st.columns(2)
    
    with col_f1:
        st.markdown("**Generator Specifications:**")
        st.write(f"- Model: {selected_gen}")
        st.write(f"- Type: {gen_data['type']}")
        st.write(f"- ISO Rating: {unit_iso_cap:.2f} MW")
        st.write(f"- Site Rating: {unit_site_cap:.2f} MW")
        st.write(f"- Efficiency (ISO): {gen_data['electrical_efficiency']*100:.1f}%")
        st.write(f"- Ramp Rate: {gen_data['ramp_rate_mw_s']:.1f} MW/s")
        st.write(f"- MTBF: {gen_data['mtbf_hours']:,} hours")
    
    with col_f2:
        st.markdown("**Operating Parameters:**")
        st.write(f"- Strategy: {load_strategy}")
        st.write(f"- Units Running: {units_running} of {n_total}")
        st.write(f"- Load per Unit: {load_per_unit_pct:.1f}%")
        st.write(f"- Fleet Efficiency: {fleet_efficiency*100:.1f}%")
        st.write(f"- Capacity Factor: {capacity_factor*100:.0f}%")
        st.write(f"- Hours/Year: {effective_hours:.0f}")
        st.write(f"- Annual Energy: {mwh_year:,.0f} MWh")

with t2:
    st.subheader("Electrical Performance & Stability")
    
    col_e1, col_e2, col_e3, col_e4 = st.columns(4)
    col_e1.metric("Voltage", f"{rec_voltage_kv} kV")
    col_e2.metric("Frequency", f"{freq_hz} Hz")
    col_e3.metric("X\"d", f"{gen_data['reactance_xd_2']:.3f} pu")
    col_e4.metric("Ramp Rate", f"{gen_data['ramp_rate_mw_s']:.1f} MW/s")
    
    # Transient Stability
    st.markdown("### 🎯 Transient Stability Analysis")
    
    if stability_ok:
        st.success(f"✅ **Voltage Sag OK:** {voltage_sag:.2f}% (Limit: 10%)")
    else:
        st.error(f"❌ **Voltage Sag Exceeds:** {voltage_sag:.2f}% > 10%")
        st.warning("**Mitigation:** Add generators, increase BESS, or use lower X\"d units")
    
    # Step Load & BESS
    st.markdown("### 🔋 Step Load Capability & BESS Analysis")
    
    col_s1, col_s2, col_s3 = st.columns(3)
    col_s1.metric("Required Step Load", f"{step_load_req:.0f}%")
    col_s2.metric("Gen Capability", f"{gen_data['step_load_pct']:.0f}%")
    
    step_capable = gen_data["step_load_pct"] >= step_load_req
    if step_capable:
        col_s3.success("✅ COMPLIANT")
    elif use_bess:
        col_s3.warning("⚠️ BESS REQUIRED")
    else:
        col_s3.error("❌ NOT COMPLIANT")
    
    if use_bess:
        st.info(f"🔋 **BESS Capacity:** {bess_power_total:.1f} MW / {bess_energy_total:.1f} MWh")
        
        # BESS Breakdown
        bess_breakdown_data = pd.DataFrame({
            'Component': list(bess_breakdown.keys()),
            'Power (MW)': list(bess_breakdown.values())
        })
        
        fig_bess = px.bar(bess_breakdown_data, x='Component', y='Power (MW)',
                         title="BESS Sizing Breakdown", color='Component')
        fig_bess.update_layout(showlegend=False, height=350)
        st.plotly_chart(fig_bess, use_container_width=True)
    
    # Reliability over time (Weibull)
    st.markdown("### 📊 Reliability Projection (Weibull Model)")
    
    years_range = list(range(1, project_years + 1))
    
    fig_rel = go.Figure()
    fig_rel.add_trace(go.Scatter(
        x=years_range, y=[a*100 for a in availability_curve],
        mode='lines', name='Availability',
        line=dict(color='#007bff', width=3)
    ))
    fig_rel.add_hline(
        y=avail_req, line_dash="dash", line_color="red",
        annotation_text=f"Target: {avail_req:.2f}%"
    )
    fig_rel.update_layout(
        title="System Availability Over Time",
        xaxis_title="Project Year",
        yaxis_title="Availability (%)",
        height=400
    )
    st.plotly_chart(fig_rel, use_container_width=True)
    
    # Emissions
    st.markdown("### 🌍 Environmental Performance")
    
    col_em1, col_em2, col_em3, col_em4 = st.columns(4)
    col_em1.metric("NOx", f"{nox_lb_hr:.2f} lb/hr")
    col_em2.metric("CO", f"{co_lb_hr:.2f} lb/hr")
    col_em3.metric("CO₂/Year", f"{co2_ton_yr:,.0f} tons")
    col_em4.metric("Carbon Cost", f"${carbon_cost_year/1e6:.2f}M/yr")
    
    if at_capex_total > 0:
        st.warning(f"⚠️ **Emissions Control:** SCR + Catalyst (${at_capex_total/1e6:.2f}M)")

with t3:
    st.subheader("Footprint Analysis & Optimization")
    
    # Footprint metrics
    col_fp1, col_fp2, col_fp3 = st.columns(3)
    col_fp1.metric("Total Footprint", f"{disp_area:.2f} {disp_area_unit}")
    col_fp2.metric("Power Density", f"{gen_data['power_density_mw_per_m2']:.3f} MW/m²")
    
    if enable_footprint_limit:
        col_fp3.metric("Utilization", f"{area_utilization_pct:.1f}%")
    else:
        col_fp3.metric("Status", "No Limit Set")
    
    # Footprint breakdown
    st.markdown("### 📐 Footprint Breakdown")
    
    footprint_data = pd.DataFrame({
        "Component": ["Generators", "BESS", "Fuel Storage", "Cooling/CHP", "Substation", "Contingency (20%)"],
        "Area (m²)": [area_gen, area_bess, storage_area_m2, area_chp, area_sub, total_area_m2 * 0.2]
    })
    
    col_pie1, col_pie2 = st.columns([2, 1])
    
    with col_pie1:
        fig_pie = px.pie(footprint_data, values='Area (m²)', names='Component',
                        title=f"Total: {disp_area:.2f} {disp_area_unit}")
        st.plotly_chart(fig_pie, use_container_width=True)
    
    with col_pie2:
        st.dataframe(footprint_data, use_container_width=True)
    
    # Optimization recommendations
    if is_area_exceeded and footprint_recommendations:
        st.error(f"🛑 **Footprint Exceeded:** {total_area_m2:,.0f} m² > {max_area_m2:,.0f} m² ({(total_area_m2/max_area_m2-1)*100:.0f}% over)")
        
        st.markdown("### 💡 Optimization Recommendations")
        
        for i, rec in enumerate(footprint_recommendations, 1):
            with st.expander(f"Option {i}: {rec['action']} (Saves {rec['savings_pct']:.1f}%)"):
                st.write(f"**Type:** {rec['type']}")
                st.write(f"**New Footprint:** {rec['new_area']:,.0f} m² ({rec['new_area']/10000:.2f} Ha)")
                st.write(f"**Savings:** {rec['savings_pct']:.1f}%")
                st.write(f"**Trade-off:** {rec['trade_off']}")
                
                if rec['type'] == 'Switch Technology':
                    st.info("✅ **Recommended:** Higher density technology maintains performance")
                elif rec['type'] == 'Reduce Redundancy':
                    st.warning("⚠️ **Risk:** Lower availability - evaluate criticality")
    
    elif enable_footprint_limit:
        st.success(f"✅ **Footprint OK:** {area_utilization_pct:.1f}% of available area")

with t4:
    st.subheader("Cooling & Tri-Generation")
    
    col_c1, col_c2, col_c3, col_c4 = st.columns(4)
    
    if include_chp:
        col_c1.metric("Heat Recovered", f"{total_heat_rec_mw:.1f} MWt")
        col_c2.metric("Cooling Generated", f"{total_cooling_mw_chp:.1f} MWc")
        col_c3.metric("Coverage", f"{cooling_coverage_pct:.1f}%")
        col_c4.metric("PUE Improvement", f"{pue - pue_actual:.2f}")
        
        st.progress(min(1.0, cooling_coverage_pct/100))
        
        st.info(f"💡 **Tri-Gen Benefit:** PUE reduced from {pue:.2f} to {pue_actual:.2f}")
    else:
        col_c1.metric("Cooling Method", cooling_method)
        col_c2.metric("Cooling Load", f"{total_cooling_mw:.1f} MWc")
        col_c3.metric("PUE", f"{pue_actual:.2f}")
        col_c4.metric("WUE", f"{wue:.1f}")
    
    st.metric(f"Water Consumption (WUE {wue:.1f})", f"{disp_water:,.0f} gal/day" if is_imperial else f"{disp_water:,.0f} m³/day")
    
    if wue > 1.5:
        st.warning("⚠️ **High Water Use:** Consider dry cooling or recycling")

with t5:
    st.subheader("Financial Analysis & Economics")
    
    # Tax benefits summary
    if itc_benefit_m > 0 or depreciation_benefit_m > 0 or ptc_annual > 0:
        st.markdown("### 💸 Tax Benefits & Incentives")
        
        col_tax1, col_tax2, col_tax3, col_tax4 = st.columns(4)
        
        if itc_benefit_m > 0:
            col_tax1.metric("ITC (30%)", f"${itc_benefit_m:.2f}M")
        if depreciation_benefit_m > 0:
            col_tax2.metric("MACRS Depreciation", f"${depreciation_benefit_m:.2f}M")
        if ptc_annual > 0:
            col_tax3.metric("PTC (Annual)", f"${ptc_annual/1e6:.2f}M")
        
        total_tax_benefit_m = itc_benefit_m + depreciation_benefit_m + (ptc_annual * project_years / 1e6)
        col_tax4.metric("Total Tax Benefit", f"${total_tax_benefit_m:.2f}M")
    
    # LCOE Target Check
    if enable_lcoe_target and target_lcoe > 0:
        if lcoe > target_lcoe:
            st.error(f"⚠️ **Target Missed:** LCOE ${lcoe:.4f}/kWh > Target ${target_lcoe:.4f}/kWh")
        else:
            st.success(f"🎉 **Target Met:** LCOE ${lcoe:.4f}/kWh < Target ${target_lcoe:.4f}/kWh")
    
    # CAPEX Editor
    st.markdown("### 💰 CAPEX Breakdown")
    st.info(f"**Regional Multiplier:** {region} ({regional_mult:.2f}x)")
    
    edited_capex = st.data_editor(
        df_capex,
        column_config={
            "Index": st.column_config.NumberColumn("Multiplier", min_value=0.0, max_value=5.0, step=0.01),
            "Cost (M USD)": st.column_config.NumberColumn("Cost", format="$%.2fM", disabled=True)
        },
        use_container_width=True
    )
    
    total_capex_dynamic = edited_capex["Cost (M USD)"].sum()
    
    # Recalculate financials
    capex_annualized_dyn = (total_capex_dynamic * 1e6) * crf
    total_annual_cost_dyn = fuel_cost_year + om_cost_year + capex_annualized_dyn + repowering_annualized + carbon_cost_year
    total_annual_cost_dyn_after_tax = total_annual_cost_dyn - ptc_annual - itc_annualized - depreciation_annualized
    lcoe_dyn = total_annual_cost_dyn_after_tax / (mwh_year * 1000)
    
    npv_dyn = pv_savings + total_tax_benefits - (total_capex_dynamic * 1e6) - (repowering_pv_m * 1e6)
    
    if annual_savings > 0:
        payback_dyn = (total_capex_dynamic * 1e6) / annual_savings
        roi_dyn = (annual_savings / (total_capex_dynamic * 1e6)) * 100
        payback_str_dyn = f"{payback_dyn:.1f} Years"
    else:
        payback_str_dyn = "N/A"
        roi_dyn = 0
    
    # Financial KPIs
    st.markdown("### 📊 Key Financial Metrics")
    
    c_f1, c_f2, c_f3, c_f4, c_f5 = st.columns(5)
    c_f1.metric("CAPEX", f"${total_capex_dynamic:.2f}M")
    c_f2.metric("LCOE", f"${lcoe_dyn:.4f}/kWh")
    c_f3.metric("Annual Savings", f"${annual_savings/1e6:.2f}M")
    c_f4.metric("NPV (20yr)", f"${npv_dyn/1e6:.2f}M")
    c_f5.metric("Payback", payback_str_dyn)
    
    # O&M Breakdown
    st.markdown("### 🔧 Annual O&M Breakdown")
    
    om_data = pd.DataFrame({
        'Component': ['Fixed ($/kW-yr)', 'Variable ($/MWh)', 'Labor', 'Major Overhaul', 'BESS O&M'],
        'Annual Cost ($M)': [
            om_fixed_annual/1e6,
            om_variable_annual/1e6,
            om_labor_annual/1e6,
            overhaul_annualized/1e6,
            bess_om_annual/1e6
        ]
    })
    
    fig_om = px.bar(om_data, x='Component', y='Annual Cost ($M)',
                   title=f"Total O&M: ${om_cost_year/1e6:.2f}M/year",
                   color='Component')
    fig_om.update_layout(showlegend=False, height=350)
    st.plotly_chart(fig_om, use_container_width=True)
    
    # LCOE Breakdown
    st.markdown("### 💵 LCOE Component Breakdown")
    
    cost_data = pd.DataFrame({
        "Component": ["Fuel", "O&M", "CAPEX", "Repowering", "Carbon", "Tax Benefits"],
        "$/kWh": [
            fuel_cost_year/(mwh_year*1000),
            om_cost_year/(mwh_year*1000),
            capex_annualized_dyn/(mwh_year*1000),
            repowering_annualized/(mwh_year*1000),
            carbon_cost_year/(mwh_year*1000),
            -(ptc_annual + itc_annualized + depreciation_annualized)/(mwh_year*1000)
        ]
    })
    
    fig_lcoe = px.bar(cost_data, x="Component", y="$/kWh",
                     title="LCOE Breakdown", text_auto='.4f',
                     color="Component")
    fig_lcoe.update_layout(showlegend=False, height=400)
    st.plotly_chart(fig_lcoe, use_container_width=True)
    
    # Gas Price Sensitivity
    st.markdown("### 📈 Gas Price Sensitivity Analysis")
    
    if breakeven_gas_price > 0:
        st.success(f"🎯 **Breakeven Gas Price:** ${breakeven_gas_price:.2f}/MMBtu")
    else:
        st.error("⚠️ **No Breakeven:** Prime power more expensive even with free gas")
    
    fig_sens = go.Figure()
    fig_sens.add_trace(go.Scatter(
        x=gas_prices_x, y=lcoe_y, mode='lines',
        name='LCOE (Prime)', line=dict(color='#667eea', width=3)
    ))
    fig_sens.add_hline(
        y=benchmark_price, line_dash="dash", line_color="red",
        annotation_text=f"Benchmark: ${benchmark_price:.3f}/kWh"
    )
    if breakeven_gas_price > 0:
        fig_sens.add_vline(
            x=breakeven_gas_price, line_dash="dot", line_color="green",
            annotation_text=f"Breakeven: ${breakeven_gas_price:.2f}"
        )
    fig_sens.update_layout(
        title="LCOE vs Gas Price",
        xaxis_title="Total Gas Price ($/MMBtu)",
        yaxis_title="LCOE ($/kWh)",
        height=450
    )
    st.plotly_chart(fig_sens, use_container_width=True)

# ==============================================================================
# 9. EXCEL EXPORT
# ==============================================================================

st.markdown("---")
st.subheader("📄 Export Comprehensive Report")

@st.cache_data
def create_excel_export():
    output = BytesIO()
    
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        # Sheet 1: Executive Summary
        summary_data = {
            'Parameter': [
                'DC Type', 'IT Load (MW)', 'PUE', 'Total DC Load (MW)',
                'Avg Load (MW)', 'Peak Load (MW)', 'Capacity Factor (%)',
                'Generator Model', 'Generator Type', 'Fleet Config',
                'Installed Capacity (MW)', 'Availability (%)', 'Operating PUE',
                'Strategy', 'Load/Unit (%)', 'Fleet Efficiency (%)',
                'Voltage (kV)', 'Primary Fuel', 'Region'
            ],
            'Value': [
                dc_type, p_it, pue, p_total_dc,
                p_total_avg, p_total_peak, capacity_factor*100,
                selected_gen, gen_data['type'], f'{n_running}+{n_reserve}',
                installed_cap, prob_gen*100, pue_actual,
                load_strategy, load_per_unit_pct, fleet_efficiency*100,
                rec_voltage_kv, fuel_mode, region
            ]
        }
        pd.DataFrame(summary_data).to_excel(writer, sheet_name='Executive Summary', index=False)
        
        # Sheet 2: Financial Summary
        financial_data = {
            'Metric': [
                'Total CAPEX (M USD)', 'LCOE ($/kWh)', 'LCOE Pre-Tax ($/kWh)',
                'Benchmark Price ($/kWh)', 'Gas Price Total ($/MMBtu)',
                'Carbon Price ($/ton)', 'Annual Energy (MWh)', 'Effective Hours',
                'Fuel Cost/Year (M USD)', 'O&M Cost/Year (M USD)',
                'Carbon Cost/Year (M USD)', 'ITC Benefit (M USD)',
                'MACRS Benefit (M USD)', 'PTC Annual (M USD)',
                'Annual Savings (M USD)', 'NPV 20yr (M USD)',
                'Payback (Years)', 'ROI (%)', 'Breakeven Gas ($/MMBtu)'
            ],
            'Value': [
                total_capex_dynamic, lcoe_dyn, total_annual_cost/(mwh_year*1000),
                benchmark_price, total_gas_price,
                carbon_price_per_ton, mwh_year, effective_hours,
                fuel_cost_year/1e6, om_cost_year/1e6,
                carbon_cost_year/1e6, itc_benefit_m,
                depreciation_benefit_m, ptc_annual/1e6,
                annual_savings/1e6, npv_dyn/1e6,
                payback_dyn if annual_savings > 0 else 0, roi_dyn, breakeven_gas_price
            ]
        }
        pd.DataFrame(financial_data).to_excel(writer, sheet_name='Financial Summary', index=False)
        
        # Sheet 3: CAPEX
        edited_capex.to_excel(writer, sheet_name='CAPEX Breakdown', index=False)
        
        # Sheet 4: O&M
        om_data.to_excel(writer, sheet_name='OM Breakdown', index=False)
        
        # Sheet 5: Technical
        tech_data = {
            'Specification': [
                'ISO Rating (MW)', 'Site Rating (MW)', 'Derate Factor',
                'ISO Efficiency (%)', 'Fleet Efficiency (%)', 'Ramp Rate (MW/s)',
                'MTBF (hours)', 'Step Load (%)', 'Voltage Sag (%)',
                'BESS Power (MW)', 'BESS Energy (MWh)', 'Fuel (MMBtu/hr)',
                'NOx (lb/hr)', 'CO (lb/hr)', 'CO2 (tons/yr)',
                'Footprint', 'Power Density (MW/m²)', 'WUE'
            ],
            'Value': [
                unit_iso_cap, unit_site_cap, derate_factor_calc,
                gen_data['electrical_efficiency']*100, fleet_efficiency*100,
                gen_data['ramp_rate_mw_s'], gen_data['mtbf_hours'],
                gen_data['step_load_pct'], voltage_sag,
                bess_power_total, bess_energy_total, total_fuel_input_mmbtu_hr,
                nox_lb_hr, co_lb_hr, co2_ton_yr,
                f'{disp_area:.2f} {disp_area_unit}',
                gen_data['power_density_mw_per_m2'], wue
            ]
        }
        pd.DataFrame(tech_data).to_excel(writer, sheet_name='Technical Specs', index=False)
        
        # Sheet 6: Reliability
        reliability_data = pd.DataFrame({
            'Year': years_range,
            'Availability (%)': [a*100 for a in availability_curve]
        })
        reliability_data.to_excel(writer, sheet_name='Reliability Curve', index=False)
        
        # Sheet 7: Gas Sensitivity
        sensitivity_data = pd.DataFrame({
            'Gas Price ($/MMBtu)': gas_prices_x,
            'LCOE ($/kWh)': lcoe_y
        })
        sensitivity_data.to_excel(writer, sheet_name='Gas Sensitivity', index=False)
        
        # Sheet 8: Footprint
        footprint_data.to_excel(writer, sheet_name='Footprint', index=False)
        
        if bess_breakdown:
            bess_breakdown_data.to_excel(writer, sheet_name='BESS Sizing', index=False)
    
    output.seek(0)
    return output

excel_data = create_excel_export()

st.download_button(
    label="📊 Download Complete Excel Report (9 Sheets)",
    data=excel_data,
    file_name=f"CAT_QuickSize_v2_{dc_type.replace(' ','_')}_{p_it:.0f}MW_{pd.Timestamp.now().strftime('%Y%m%d_%H%M')}.xlsx",
    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    use_container_width=True
)

st.success("✅ Export includes: Executive Summary, Financials, CAPEX, O&M, Technical Specs, Reliability, Sensitivity, Footprint, BESS Sizing")

# --- FOOTER ---
st.markdown("---")
col_foot1, col_foot2, col_foot3 = st.columns(3)
col_foot1.caption("CAT QuickSize v2.0")
col_foot2.caption("Next-Gen Data Center Power Solutions")
col_foot3.caption("Caterpillar Electric Power | 2026")
