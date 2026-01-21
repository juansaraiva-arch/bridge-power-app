import streamlit as st
import pandas as pd
import numpy as np
import math
import plotly.express as px
import plotly.graph_objects as go

# --- PAGE CONFIGURATION ---
st.set_page_config(page_title="CAT Primary Power Solutions", page_icon="⚡", layout="wide")

# ==============================================================================
# 0. HYBRID DATA LIBRARY
# ==============================================================================

leps_gas_library = {
    "XGC1900": {
        "description": "Mobile Power Module (High Speed)",
        "type": "High Speed",
        "iso_rating_mw": 1.9,
        "electrical_efficiency": 0.392,
        "heat_rate_lhv": 8780,
        "step_load_pct": 25.0, 
        "emissions_nox": 0.5,
        "emissions_co": 2.5,
        "default_for": 2.0, 
        "default_maint": 5.0,
        "est_cost_kw": 775.0,      
        "est_install_kw": 300.0,   
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
        "emissions_nox": 0.5,
        "emissions_co": 2.1,
        "default_for": 2.0,
        "default_maint": 5.0,
        "est_cost_kw": 575.0,
        "est_install_kw": 650.0,
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
        "emissions_nox": 0.3,
        "emissions_co": 2.3,
        "default_for": 2.5,
        "default_maint": 6.0,
        "est_cost_kw": 575.0,
        "est_install_kw": 650.0,
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
        "emissions_nox": 0.5,
        "emissions_co": 1.8,
        "default_for": 3.0,
        "default_maint": 5.0,
        "est_cost_kw": 675.0,
        "est_install_kw": 1100.0,
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
        "emissions_nox": 0.6,
        "emissions_co": 0.6,
        "default_for": 1.5,
        "default_maint": 2.0,
        "est_cost_kw": 775.0,
        "est_install_kw": 1000.0,
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
        "emissions_nox": 0.5,
        "emissions_co": 0.5,
        "default_for": 3.0, 
        "default_maint": 5.0,
        "est_cost_kw": 700.0,
        "est_install_kw": 1250.0,
        "gas_pressure_min_psi": 90.0,
        "reactance_xd_2": 0.16
    }
}

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

# Unit Strings & Conversions
if is_imperial:
    u_temp, u_dist, u_area_s, u_area_l = "°F", "ft", "ft²", "Acres"
    u_vol, u_mass, u_power = "gal", "Short Tons", "MW"
    u_energy, u_therm, u_water = "MWh", "MMBtu", "gal/day"
    u_press = "psig"
    u_hr = "Btu/kWh"
    hr_conv_factor = 1.0
else:
    u_temp, u_dist, u_area_s, u_area_l = "°C", "m", "m²", "Ha"
    u_vol, u_mass, u_power = "m³", "Tonnes", "MW"
    u_energy, u_therm, u_water = "MWh", "GJ", "m³/day"
    u_press = "Bar"
    u_hr = "kJ/kWh"
    hr_conv_factor = 1.055056 # Convert Btu to kJ

t = {
    "title": f"⚡ CAT Primary Power Solutions ({freq_hz}Hz)",
    "subtitle": "**Sovereign Energy Solutions.**\nAdvanced modeling for Off-Grid Microgrids, Tri-Generation, and Gas Infrastructure.",
    "sb_1": "1. Data Center Profile",
    "sb_2": "2. Generation Technology",
    "sb_3": "3. Site, Gas & Noise",
    "sb_4": "4. Strategy (BESS & LNG)",
    "sb_5": "5. Cooling & Tri-Gen",
    "sb_6": "6. Regulatory & Emissions",
    "sb_7": "7. Economics & ROI",
    "kpi_net": "Net Capacity",
    "kpi_pue": "Projected PUE"
}

st.title(t["title"])
st.markdown(t["subtitle"])

# ==============================================================================
# 2. INPUTS (SIDEBAR)
# ==============================================================================

with st.sidebar:
    # --- 1. DATA CENTER PROFILE ---
    st.header(t["sb_1"])
    dc_type = st.selectbox("Data Center Type", ["AI Factory (Training)", "Hyperscale Standard"])
    is_ai = "AI" in dc_type
    
    def_step_load = 40.0 if is_ai else 15.0
    def_use_bess = True if is_ai else False
    
    p_it = st.number_input("Critical IT Load (MW)", 1.0, 1000.0, 100.0, step=10.0)
    avail_req = st.number_input("Required Availability (%)", 90.0, 99.99999, 99.99, format="%.5f")
    step_load_req = st.number_input("Block Load / Step Req (%)", 0.0, 100.0, def_step_load, help="% of IT load that hits instantly")
    
    # Voltage Selection
    st.markdown("⚡ **Voltage Level**")
    volt_mode = st.radio("Connection Voltage Mode", ["Auto-Recommend", "Manual Selection"])
    manual_voltage_kv = 0.0
    if volt_mode == "Manual Selection":
        manual_voltage_kv = st.number_input("Connection Voltage (kV)", 0.4, 230.0, 13.8, step=0.1)
    
    dc_aux_pct = st.number_input("DC Building Auxiliaries (%)", 0.0, 20.0, 5.0) / 100.0
    dist_loss_pct = st.number_input("Distribution Losses (%)", 0.0, 10.0, 1.0) / 100.0

    st.divider()

    # --- 2. GENERATION TECHNOLOGY ---
    st.header(t["sb_2"])
    
    selected_model = st.selectbox("Select CAT/Solar Model", list(leps_gas_library.keys()))
    eng_data = leps_gas_library[selected_model]
    st.caption(f"**{eng_data['description']}**")
    
    # Efficiency & Rating
    eff_input_method = st.radio("Efficiency Input Mode", ["Efficiency (%)", f"Heat Rate LHV ({u_hr})"])
    
    def_mw = eng_data['iso_rating_mw']
    def_eff_pct = eng_data['electrical_efficiency'] * 100.0
    
    # Display default HR in correct unit
    def_hr_base = eng_data['heat_rate_lhv'] 
    def_hr_disp = def_hr_base * hr_conv_factor
    
    col_t1, col_t2 = st.columns(2)
    unit_size_iso = col_t1.number_input("Rating (ISO MW)", 0.1, 100.0, def_mw, format="%.2f")
    
    final_elec_eff = 0.0
    if eff_input_method == "Efficiency (%)":
        eff_user = col_t2.number_input("Eff (ISO %)", 20.0, 65.0, def_eff_pct, format="%.1f")
        final_elec_eff = eff_user / 100.0
    else:
        hr_user = col_t2.number_input(f"HR ({u_hr})", 5000.0, 15000.0, def_hr_disp, format="%.0f")
        hr_btu = hr_user / hr_conv_factor
        final_elec_eff = 3412.14 / hr_btu

    # ASSET VALUATION
    st.markdown("💰 **Asset Valuation & Costs**")
    col_c1, col_c2 = st.columns(2)
    gen_unit_cost = col_c1.number_input("Equip ($/kW)", 100.0, 3000.0, eng_data['est_cost_kw'], step=10.0)
    gen_install_cost = col_c2.number_input("Install ($/kW)", 50.0, 3000.0, eng_data['est_install_kw'], step=10.0)
    
    # Technical Params
    st.markdown("⚙️ **Technical Parameters**")
    col_p1, col_p2 = st.columns(2)
    step_load_cap = col_p1.number_input("Step Load Cap (%)", 0.0, 100.0, eng_data['step_load_pct'])
    xd_2_pu = col_p2.number_input('Xd" (pu)', 0.01000, 0.50000, eng_data.get('reactance_xd_2', 0.15), step=0.001, format="%.5f")

    # Reliability
    st.caption("Gen Set Availability (N+M+S)")
    c_r1, c_r2 = st.columns(2)
    maint_outage_pct = c_r1.number_input("Maint. Unavail (%)", 0.0, 20.0, float(eng_data.get('default_maint', 5.0))) / 100.0
    forced_outage_pct = c_r2.number_input("Forced Outage Rate (%)", 0.0, 20.0, float(eng_data.get('default_for', 2.0))) / 100.0
    
    gen_parasitic_pct = st.number_input("Gen. Parasitic Load (%)", 0.0, 10.0, 2.5, help="Fixed % of Nameplate") / 100.0

    st.divider()

    # --- 3. SITE, GAS & NOISE ---
    st.header(t["sb_3"])
    
    derate_mode = st.radio("Derate Method", ["Auto-Calculate", "Manual Entry"])
    derate_factor_calc = 1.0
    methane_number = 80
    
    if derate_mode == "Auto-Calculate":
        site_temp_c = 35 # default
        site_alt_m = 100 # default
        if is_imperial:
            site_temp_f = st.slider(f"Max Ambient Temp ({u_temp})", 32, 122, 95)
            site_alt_ft = st.number_input(f"Altitude ({u_dist})", 0, 13000, 328)
            site_temp_c = (site_temp_f - 32) * 5/9
            site_alt_m = site_alt_ft / 3.28084
        else:
            site_temp_c = st.slider(f"Max Ambient Temp ({u_temp})", 0, 50, 35)
            site_alt_m = st.number_input(f"Altitude ({u_dist})", 0, 4000, 100)
        
        methane_number = st.number_input("Gas Methane Number (MN)", 30, 100, 80)
        loss_temp = max(0, (site_temp_c - 25) * 0.01) 
        loss_alt = max(0, (site_alt_m - 100) * 0.0001)
        loss_mn = max(0, (75 - methane_number) * 0.005)
        derate_factor_calc = 1.0 - (loss_temp + loss_alt + loss_mn)
        st.info(f"Derate: {derate_factor_calc:.3f}")
    else:
        manual_derate_pct = st.number_input("Manual Derate (%)", 0.0, 50.0, 5.0)
        derate_factor_calc = 1.0 - (manual_derate_pct / 100.0)

    # GAS PIPELINE INPUTS
    st.markdown("⛽ **Gas Infrastructure**")
    gas_source = st.radio("Supply Method", ["Pipeline", "Virtual Pipeline (LNG)", "Virtual Pipeline (CNG)"])
    
    virtual_pipe_mode = "Pipeline"
    if "LNG" in gas_source: virtual_pipe_mode = "LNG"
    elif "CNG" in gas_source: virtual_pipe_mode = "CNG"

    # LOGISTICS INPUTS
    storage_days = 0
    tank_unit_cap = 1.0
    tank_mob_cost = 0.0
    tank_area_unit = 0.0
    
    if virtual_pipe_mode != "Pipeline":
        st.markdown(f"🔹 **{virtual_pipe_mode} Configuration**")
        storage_days = st.number_input("Storage Autonomy (Days)", 1, 60, 5)
        
        def_cap = 10000.0 if virtual_pipe_mode == "LNG" else 350000.0
        def_mob = 5000.0 if virtual_pipe_mode == "LNG" else 2500.0
        
        c_s1, c_s2 = st.columns(2)
        tank_unit_cap = c_s1.number_input(f"Tank Cap ({'scf' if virtual_pipe_mode=='CNG' else 'Gal'})", 1000.0, 1000000.0, def_cap)
        tank_mob_cost = c_s2.number_input("Mob Cost/Tank ($)", 0.0, 50000.0, def_mob)
        tank_area_unit = st.number_input("Area per Tank (m²)", 10.0, 200.0, 40.0)

    dist_gas_main_m = st.number_input("Distance to Gas Main (m)", 10.0, 20000.0, 1000.0, step=50.0)
    
    if is_imperial:
        supply_pressure_disp = st.number_input(f"Supply Pressure ({u_press})", 5.0, 1000.0, 60.0, step=5.0) 
        supply_pressure_psi = supply_pressure_disp
    else:
        supply_pressure_disp = st.number_input(f"Supply Pressure ({u_press})", 0.5, 100.0, 4.1, step=0.5) 
        supply_pressure_psi = supply_pressure_disp * 14.5038

    # ELECTRICAL
    st.markdown("🔌 **Grid Connection**")
    grid_connected = st.checkbox("Grid Connected (Parallel)", value=True)
    if grid_connected:
        grid_mva_sc = st.number_input("Grid Short Circuit Capacity (MVA)", 50.0, 5000.0, 500.0, step=50.0)
    else:
        grid_mva_sc = 0.0

    # NOISE
    st.markdown("🔊 **Noise**")
    dist_neighbor_m = st.number_input(f"Distance to Neighbor ({u_dist})", 10.0, 5000.0, 100.0)
    if is_imperial: dist_neighbor_m = dist_neighbor_m / 3.28084
    source_noise_dba = st.number_input("Source Noise @ 1m (dBA)", 60.0, 120.0, 85.0)
    noise_limit = 70.0 

    st.divider()

    # --- 4. STRATEGY (BESS & LNG) ---
    st.header(t["sb_4"])
    use_bess = st.checkbox("Include BESS (Synthetic Inertia)", value=def_use_bess)
    
    bess_maint_pct = 0.0
    bess_for_pct = 0.0
    bess_cost_kwh = 0.0
    bess_cost_kw = 0.0
    bess_life_batt = 10
    bess_life_inv = 15
    bess_om_kw_yr = 0.0
    
    if use_bess:
        st.markdown("🔋 **BESS Reliability & O&M**")
        c_b1, c_b2 = st.columns(2)
        bess_maint_pct = c_b1.number_input("BESS Maint. Unavail (%)", 0.0, 10.0, 1.0) / 100.0
        bess_for_pct = c_b2.number_input("BESS Forced Outage Rate (%)", 0.0, 10.0, 0.5) / 100.0
        
        st.markdown("💲 **BESS Economics & Lifecycle**")
        c_c1, c_c2 = st.columns(2)
        bess_cost_kwh = c_c1.number_input("Battery Cost ($/kWh)", 100.0, 1000.0, 280.0, help="Energy Block Cost")
        bess_cost_kw = c_c2.number_input("Inverter Cost ($/kW)", 50.0, 1000.0, 120.0, help="PCS/Power Block Cost")
        
        c_l1, c_l2 = st.columns(2)
        bess_life_batt = c_l1.number_input("Battery Useful Life (Yrs)", 5, 20, 10, help="Replacement cycle for cells")
        bess_life_inv = c_l2.number_input("Inverter Useful Life (Yrs)", 5, 25, 15)
        
        bess_om_kw_yr = st.number_input("BESS Fixed O&M ($/kW-yr)", 0.0, 100.0, 10.0)

    st.divider()

    # --- 5. COOLING & TRI-GEN ---
    st.header(t["sb_5"])
    include_chp = st.checkbox("Include Tri-Gen (CHP)", value=True)
    
    cooling_method = "Tri-Gen"
    if include_chp:
        cop_double = st.number_input("COP Double Effect", 0.5, 2.0, 1.2)
        cop_single = st.number_input("COP Single Effect", 0.4, 1.5, 0.7)
        pue_input = 0.0 
    else:
        cool_idx = 0 if is_ai else 1
        cooling_method = st.selectbox("Cooling Tech", ["Water Cooled", "Air Cooled"], index=cool_idx)
        def_pue = 1.25 if "Water" in cooling_method else 1.45
        pue_input = st.number_input("Expected PUE", 1.05, 2.0, def_pue)

    st.divider()
    
    # --- 6. REGULATORY & EMISSIONS ---
    st.header(t["sb_6"])
    reg_zone = st.selectbox("Regulatory Zone", ["USA - EPA Major", "EU Standard", "LatAm / No-Reg"])
    limit_nox_tpy = 250.0 if "EPA" in reg_zone else (150.0 if "EU" in reg_zone else 9999.0)
    urea_days = st.number_input("Urea Storage (Days)", 1, 30, 7)
    
    st.markdown("🛠️ **After-Treatment Costs (USD)**")
    cost_scr_kw = st.number_input("SCR System Cost (USD/kW)", 0.0, 200.0, 60.0)
    cost_oxicat_kw = st.number_input("Oxidation Cat Cost (USD/kW)", 0.0, 100.0, 15.0)
    force_oxicat = st.checkbox("Force Oxicat Inclusion", value=False)

    st.divider()

    # --- 7. ECONOMICS ---
    st.header(t["sb_7"])
    gas_price = st.number_input("Gas Price (USD/MMBtu)", 1.0, 20.0, 6.5)
    
    if virtual_pipe_mode in ["LNG", "CNG"]:
        vp_premium = st.number_input("Virtual Pipe Premium ($/MMBtu)", 0.0, 15.0, 4.0)
        gas_price += vp_premium

    om_var_price = st.number_input("Variable O&M (USD/MWh)", 1.0, 50.0, 12.0)
    grid_price = st.number_input("Grid Price (USD/kWh)", 0.05, 0.50, 0.15)
    project_years = st.number_input("Project Years", 5, 30, 20)
    wacc = st.number_input("WACC (%)", 0.0, 15.0, 8.0) / 100.0

    # Buyout Params
    st.caption("Post-Grid Strategy Options")
    buyout_pct = st.number_input("Buyout Residual Value (%)", 0.0, 100.0, 20.0)
    ref_new_capex = eng_data['est_cost_kw']
    vpp_arb_spread = st.number_input("VPP Arbitrage ($/MWh)", 0.0, 200.0, 40.0)
    vpp_cap_pay = st.number_input("VPP Capacity ($/MW-yr)", 0.0, 100000.0, 28000.0)

# ==============================================================================
# 2. CALCULATION ENGINE (PRIME PHYSICS ENGINE v2)
# ==============================================================================

# --- A. POWER BALANCE ---
if include_chp:
    p_cooling_elec_new = p_it * 0.03 
    p_net_req = p_it * (1 + dc_aux_pct) + p_cooling_elec_new
    pue_calc = p_net_req / p_it
    cooling_mode = "Thermal (Absorption)"
else:
    p_net_req = p_it * pue_input
    pue_calc = pue_input
    cooling_load_elec = p_net_req - p_it - (p_it * dc_aux_pct)
    cooling_mode = f"Elec ({cooling_method})"

p_dist_loss_mw = p_net_req * dist_loss_pct
p_gen_bus_req = p_net_req + p_dist_loss_mw

# Voltage Selection Logic
if volt_mode == "Manual Selection":
    op_voltage_kv = manual_voltage_kv
    rec_voltage = f"{manual_voltage_kv:.1f} kV (User)"
else:
    # Auto-Recommend based on ANSI/IEEE Amperage Constraints
    if is_50hz:
        rec_voltage = "11 kV" if p_gen_bus_req < 20 else ("33 kV" if p_gen_bus_req > 50 else "11 kV / 33 kV")
        op_voltage_kv = 11.0 if p_gen_bus_req < 35 else 33.0
    else:
        rec_voltage = "13.8 kV" if p_gen_bus_req < 25 else ("34.5 kV" if p_gen_bus_req > 60 else "13.8 kV / 34.5 kV")
        op_voltage_kv = 13.8 if p_gen_bus_req < 45 else 34.5

# --- B. FLEET SIZING (TRI-VECTOR ALGORITHM) ---
unit_site_cap = unit_size_iso * derate_factor_calc
step_mw_req = p_it * (step_load_req / 100.0)

driver_txt = "N/A"
n_steady = 0
n_transient = 0
n_headroom = 0

if use_bess:
    # BESS Optimized
    target_load_factor = 0.95 
    n_base_mw = p_gen_bus_req / (1 - gen_parasitic_pct) 
    n_running = math.ceil(n_base_mw / (unit_site_cap * target_load_factor))
    
    bess_power_req = max(step_mw_req, unit_site_cap) 
    
    driver_txt = "Steady State (BESS Optimized)"
else:
    # NO BESS - HARD CONSTRAINTS
    n_steady = math.ceil(p_gen_bus_req / (unit_site_cap * 0.90))
    
    # Transient Stiffness
    unit_step_mw_cap = unit_site_cap * (step_load_cap / 100.0)
    n_transient = math.ceil(step_mw_req / unit_step_mw_cap)
    
    # Headroom
    n_headroom = n_steady
    while True:
        total_cap = n_headroom * unit_site_cap
        total_parasitics = n_headroom * (unit_size_iso * gen_parasitic_pct)
        gross_needed = p_gen_bus_req + total_parasitics
        if (total_cap - gross_needed) >= step_mw_req:
            break
        n_headroom += 1
        
    n_running = max(n_steady, n_transient, n_headroom)
    
    if n_running == n_transient: driver_txt = f"Transient Stiffness (Step: {step_load_cap}%)"
    elif n_running == n_headroom: driver_txt = "Spinning Reserve (Headroom)"
    else: driver_txt = "Steady State Load"
    
    bess_power_req = 0

# --- C. RELIABILITY (PROBABILISTIC - GEN + BESS HYBRID LOOP) ---
n_maint = math.ceil(n_running * maint_outage_pct) 

prob_gen_unit = 1.0 - forced_outage_pct
prob_bess_unit = 1.0 - (bess_maint_pct + bess_for_pct)
target_reliability = avail_req / 100.0

n_reserve_gen = 0
n_redundant_bess = 0 

while True:
    # 1. Calc Generator Reliability
    n_pool_gen = n_running + n_reserve_gen
    prob_gen_sys = 0.0
    for k in range(n_running, n_pool_gen + 1):
        comb = math.comb(n_pool_gen, k)
        prob = comb * (prob_gen_unit ** k) * ((1 - prob_gen_unit) ** (n_pool_gen - k))
        prob_gen_sys += prob
        
    # 2. Calc BESS Reliability
    if use_bess:
        p_fail_bess_unit = 1.0 - prob_bess_unit
        prob_bess_sys = 1.0 - (p_fail_bess_unit ** (1 + n_redundant_bess))
    else:
        prob_bess_sys = 1.0
        
    # 3. Combined System Reliability
    system_reliability = prob_gen_sys * prob_bess_sys
    
    if system_reliability >= target_reliability:
        break
        
    if use_bess and (prob_bess_sys < prob_gen_sys):
        n_redundant_bess += 1
    else:
        n_reserve_gen += 1
        
    if n_reserve_gen > 25 or n_redundant_bess > 10: break

n_reserve = n_reserve_gen # ALIAS FIX
n_total = n_running + n_maint + n_reserve
installed_cap = n_total * unit_site_cap
system_reliability_pct = system_reliability * 100.0

reliability_bottleneck = "Generators" # Default init
if use_bess and (prob_bess_sys < prob_gen_sys):
    reliability_bottleneck = "BESS Availability"

# BESS Final Sizing
bess_multiplier = 1 + n_redundant_bess
bess_power_total = bess_power_req * bess_multiplier
bess_energy_total = bess_power_total * 2 

# --- D. THERMODYNAMICS (AGGRESSIVE CURVE) ---
total_parasitics_mw = n_running * (unit_size_iso * gen_parasitic_pct)
p_gross_total = p_gen_bus_req + total_parasitics_mw
real_load_factor = p_gross_total / (n_running * unit_site_cap)

base_eff = eng_data['electrical_efficiency']
type_tech = eng_data.get('type', 'High Speed')

if type_tech == "High Speed": 
    if real_load_factor >= 0.75: eff_factor = 1.0
    elif real_load_factor >= 0.50:
        eff_factor = 0.85 + (0.6 * (real_load_factor - 0.50)) 
    else:
        eff_factor = 0.65 + (1.0 * (real_load_factor - 0.30))
else: 
    eff_factor = 1.0 - (0.8 * (1.0 - real_load_factor))

eff_factor = max(eff_factor, 0.50)
gross_eff_site = base_eff * eff_factor
gross_hr_lhv = 3412.14 / gross_eff_site

# Fuel Calculation
total_fuel_input_mmbtu_hr = p_gross_total * (gross_hr_lhv / 1000) 

net_hr_lhv = (total_fuel_input_mmbtu_hr * 1e6) / p_net_req
net_hr_hhv = net_hr_lhv * 1.108

# Display Heat Rate logic (Rounding UP and removing decimals)
disp_hr_val = math.ceil(net_hr_lhv * hr_conv_factor)
disp_hr_unit = u_hr 

# --- E. SHORT CIRCUIT ---
gen_mva_total = installed_cap / 0.8
gen_sc_mva = gen_mva_total / xd_2_pu
total_sc_mva = gen_sc_mva + grid_mva_sc
isc_ka = total_sc_mva / (math.sqrt(3) * op_voltage_kv)
rec_breaker = 63
for b in [25, 31.5, 40, 50, 63]:
    if b > (isc_ka * 1.1): rec_breaker = b; break
switchgear_cost_factor = 1.2 if isc_ka > 40 else 1.0

# --- F. THERMAL (CHP) ---
heat_input_mw = total_fuel_input_mmbtu_hr / 3.41214
heat_exhaust_mw = heat_input_mw * 0.28
heat_jacket_mw = heat_input_mw * 0.18
total_heat_rec_mw = heat_exhaust_mw + heat_jacket_mw

if include_chp:
    total_cooling_mw = (heat_exhaust_mw * cop_double) + (heat_jacket_mw * cop_single)
    cooling_coverage_pct = min(100.0, (total_cooling_mw / (p_it * 1.05)) * 100)
    water_cons_m3_hr = total_cooling_mw * 1.8 
else:
    total_cooling_mw = 0
    cooling_coverage_pct = 0
    water_cons_m3_hr = (p_net_req - p_it) * 1.5 if "Water" in cooling_method else 0.0

water_cons_daily_m3 = water_cons_m3_hr * 24

# --- G. LOGISTICS ---
total_mmbtu_day = total_fuel_input_mmbtu_hr * 24
peak_scfh = total_fuel_input_mmbtu_hr * 1000 
req_pressure_min = eng_data.get('gas_pressure_min_psi', 0.5)
need_compressor = supply_pressure_psi < req_pressure_min
pressure_status = f"LOW PRESSURE (Compressor Req)" if need_compressor else "Pressure OK"

actual_flow_acfm = peak_scfh * (14.7 / (supply_pressure_psi + 14.7)) / 60 
target_area_ft2 = actual_flow_acfm / (65 * 60) 
rec_pipe_dia = max(4, math.ceil(math.sqrt(target_area_ft2 * 4 / math.pi) * 12))

num_tanks = 0; log_capex = 0; log_text = "Pipeline"; storage_area_m2 = 0 

if virtual_pipe_mode == "LNG":
    vol_day = total_mmbtu_day * 12.5
    num_tanks = math.ceil((vol_day * storage_days)/tank_unit_cap)
    log_capex = num_tanks * tank_mob_cost
    log_text = f"LNG: {vol_day:,.0f} gpd"
    storage_area_m2 = num_tanks * tank_area_unit
elif virtual_pipe_mode == "CNG":
    vol_day = total_mmbtu_day * 1000
    num_tanks = math.ceil((vol_day * storage_days)/tank_unit_cap)
    log_capex = num_tanks * tank_mob_cost
    log_text = f"CNG: {vol_day/1e6:.2f} MMscfd"
    storage_area_m2 = num_tanks * tank_area_unit
elif virtual_pipe_mode in ["Diesel", "Propane"]:
    conv = 7.3 if virtual_pipe_mode == "Diesel" else 11.0
    vol_day = total_mmbtu_day * conv
    num_tanks = math.ceil((vol_day * storage_days)/tank_unit_cap)
    log_capex = num_tanks * tank_mob_cost
    log_text = f"{virtual_pipe_mode}: {vol_day:,.0f} gpd"
    storage_area_m2 = num_tanks * tank_area_unit

# --- H. EMISSIONS ---
attenuation = 20 * math.log10(dist_neighbor_m)
noise_rec = source_noise_dba + (10 * math.log10(n_running)) - attenuation
raw_nox = eng_data['emissions_nox']
total_bhp = p_gross_total * 1341
nox_tpy = (raw_nox * total_bhp * 8760) / 907185
req_scr = nox_tpy > limit_nox_tpy
urea_vol_yr = p_gross_total * 1.5 * 8760 if req_scr else 0

at_capex_total = 0
if req_scr:
    at_capex_total += (installed_cap * 1000) * cost_scr_kw
if force_oxicat: 
    at_capex_total += (installed_cap * 1000) * cost_oxicat_kw

# --- I. FOOTPRINT ---
area_gen = n_total * 200 
area_chp = total_cooling_mw * 20 if include_chp else (p_net_req * 10) 
area_bess = bess_power_total * 30 
area_sub = 2500
total_area_m2 = (area_gen + storage_area_m2 + area_chp + area_bess + area_sub) * 1.2

# --- J. FINANCIALS & NPV (ENHANCED) ---
base_gen_cost_kw = gen_unit_cost 
gen_cost_total = (installed_cap * 1000) * base_gen_cost_kw / 1e6 

idx_install = (gen_install_cost / gen_unit_cost) * switchgear_cost_factor
idx_chp = 0.20 if include_chp else 0

# BESS DETAILED CAPEX
bess_capex_m = 0.0
bess_om_annual = 0.0
if use_bess:
    cost_power_part = (bess_power_total * 1000) * bess_cost_kw
    cost_energy_part = (bess_energy_total * 1000) * bess_cost_kwh
    bess_capex_m = (cost_power_part + cost_energy_part) / 1e6
    bess_om_annual = (bess_power_total * 1000 * bess_om_kw_yr) 

pipe_cost_m = 50 * rec_pipe_dia 
pipeline_capex_m = (pipe_cost_m * dist_gas_main_m) / 1e6 if virtual_pipe_mode == "Pipeline" else 0

cost_items = [
    {"Item": "Generation Units", "Default Index": 1.00, "Cost (M USD)": gen_cost_total},
    {"Item": "Installation & BOP", "Default Index": idx_install, "Cost (M USD)": gen_cost_total * idx_install},
    {"Item": "Tri-Gen Plant", "Default Index": idx_chp, "Cost (M USD)": gen_cost_total * idx_chp},
    {"Item": "BESS System", "Default Index": 0.0, "Cost (M USD)": bess_capex_m}, 
    {"Item": "Logistics/Fuel Infra", "Default Index": 0.0, "Cost (M USD)": (log_capex + pipeline_capex_m * 1e6)/1e6},
    {"Item": "Emissions Control", "Default Index": 0.0, "Cost (M USD)": at_capex_total / 1e6},
]
df_capex_base = pd.DataFrame(cost_items)

# REPOWERING CASH FLOW
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

# Annualize
crf = (wacc * (1 + wacc)**project_years) / ((1 + wacc)**project_years - 1)
repowering_annualized = repowering_pv_m * 1e6 * crf 

# LCOE Calculation
mwh_year = p_net_req * 8760
fuel_cost_year = total_fuel_input_mmbtu_hr * gas_price * 8760
om_cost_year = (mwh_year * om_var_price) + bess_om_annual 

# Initial Total CAPEX
initial_capex_sum = df_capex_base["Cost (M USD)"].sum()
capex_annualized = (initial_capex_sum * 1e6) * crf

total_annual_cost = fuel_cost_year + om_cost_year + capex_annualized + repowering_annualized
lcoe = total_annual_cost / (mwh_year * 1000)

# NPV Logic
annual_grid_cost = mwh_year * 1000 * grid_price
annual_prime_opex = fuel_cost_year + om_cost_year
annual_savings = annual_grid_cost - annual_prime_opex

if wacc > 0:
    pv_savings = annual_savings * ((1 - (1 + wacc)**-project_years) / wacc)
else:
    pv_savings = annual_savings * project_years

npv = pv_savings - (initial_capex_sum * 1e6) - (repowering_pv_m * 1e6)

# Payback
if annual_savings > 0:
    payback_years = (initial_capex_sum * 1e6) / annual_savings
    roi_simple = (annual_savings / (initial_capex_sum * 1e6)) * 100
    payback_str = f"{payback_years:.1f} Years"
else:
    payback_str = "N/A"; roi_simple = 0

# --- K. SENSITIVITY ANALYSIS (SWEET SPOT) ---
annual_grid_revenue = mwh_year * 1000 * grid_price
fixed_costs_annual = om_cost_year + capex_annualized + repowering_annualized
fuel_mmbtu_annual = total_fuel_input_mmbtu_hr * 8760

if fuel_mmbtu_annual > 0:
    breakeven_gas_price = (annual_grid_revenue - fixed_costs_annual) / fuel_mmbtu_annual
else:
    breakeven_gas_price = 0

# Generate Plot Data
gas_prices_x = np.linspace(1, 20, 50) # $1 to $20 range
lcoe_y = []
for g in gas_prices_x:
    fc = fuel_mmbtu_annual * g
    tot = fc + fixed_costs_annual
    lcoe_y.append(tot / (mwh_year * 1000))

# ==============================================================================
# 3. DASHBOARD OUTPUT
# ==============================================================================

if is_imperial:
    disp_cooling = total_cooling_mw * 284.345 
    disp_water = water_cons_daily_m3 * 264.172 
    disp_area = total_area_m2 * 10.764 
    disp_dist = dist_neighbor_m * 3.28
    footprint_large_val = total_area_m2 * 0.000247105 # Acres
    footprint_unit = "Acres"
else:
    disp_cooling = total_cooling_mw
    disp_water = water_cons_daily_m3
    disp_area = total_area_m2
    disp_dist = dist_neighbor_m
    footprint_large_val = total_area_m2 / 10000.0 # Hectares
    footprint_unit = "Ha"

# --- TOP KPIS ---
c1, c2, c3, c4 = st.columns(4)
c1.metric(t["kpi_net"], f"{p_net_req:.1f} MW", f"Gross: {p_gross_total:.1f} MW")
# Dynamic HR Unit (Rounded Up)
c2.metric(f"Net Heat Rate ({disp_hr_unit})", f"{disp_hr_val:,.0f}", f"Eff: {gross_eff_site*100:.1f}%")
c3.metric("Rec. Voltage", rec_voltage, f"Isc: {isc_ka:.1f} kA")
c4.metric(t["kpi_pue"], f"{pue_calc:.3f}", f"Cooling: {cooling_mode}")

st.divider()

# --- TABS ---
t1, t2, t3, t4 = st.tabs(["⚙️ Engineering", "🧪 Physics & Logistics", "❄️ Tri-Gen", "💰 Financials & Payback"])

with t1:
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Power Balance")
        df_bal = pd.DataFrame({
            "Component": ["Critical IT", "DC Auxiliaries", "CHP Pumps/Cooling", "Dist. Losses", "Gen Parasitics", "TOTAL GROSS"],
            "MW": [p_it, p_it*dc_aux_pct, cooling_load_elec if not include_chp else p_cooling_elec_new, p_dist_loss_mw, total_parasitics_mw, p_gross_total]
        })
        st.dataframe(df_bal.style.format({"MW": "{:.2f}"}), use_container_width=True)
        
        st.subheader("Electrical Sizing")
        st.write(f"**Grid Contribution:** {grid_mva_sc} MVA")
        st.write(f"**Gen Contribution:** {gen_sc_mva:.1f} MVA (Xd\" {xd_2_pu})")
        st.markdown(f"**Total Short Circuit:** :red[**{isc_ka:.1f} kA**]")
        st.success(f"✅ Recommended Switchgear Rating: **{rec_breaker} kA**")
            
    with col2:
        st.subheader("Fleet Strategy (Tri-Vector Sizing)")
        st.write(f"**Driver:** {driver_txt}")
        st.write(f"**Avg. Load Factor:** {real_load_factor*100:.1f}%")
        st.markdown("---")
        st.write(f"**N (Running):** {n_running}")
        st.write(f"**M (Maintenance):** {n_maint}")
        st.write(f"**S (Standby):** {n_reserve}")
        st.caption(f"Reserve: Probabilistic > {avail_req}% reliability.")
        
        # Reliability Coloring
        if system_reliability_pct >= avail_req:
            st.metric("Total Installed Fleet", f"{n_total} Units", f"Reliability: {system_reliability_pct:.4f}% (OK)")
        else:
            st.metric("Total Installed Fleet", f"{n_total} Units", f"Reliability: {system_reliability_pct:.4f}% (LOW)", delta_color="inverse")
            st.error(f"⚠️ Bottleneck: {reliability_bottleneck}. Consider redundant BESS or more Gens.")
        
        if use_bess:
            st.info(f"⚡ **BESS:** {bess_power_total:.1f} MW / {bess_energy_total:.1f} MWh")
            if n_redundant_bess > 0:
                st.warning(f"⚠️ **High Availability:** {n_redundant_bess} Redundant BESS Units Added.")
        
        if not use_bess and real_load_factor < 0.50:
            st.error("⛔ **RICE RISK:** LF < 50%. Wet Stacking Danger.")

with t2:
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Logistics: " + virtual_pipe_mode)
        if virtual_pipe_mode != "Pipeline":
            st.metric("Daily Volume", log_text)
            st.metric("Assets Req.", f"{num_tanks} Tanks")
            st.write(f"**Storage Area:** {storage_area_m2:.0f} m²")
            st.write(f"**Logistics CAPEX:** ${log_capex:,.0f}")
        else:
            st.success(f"Pipeline Connected. Dia: {rec_pipe_dia}\"")
            st.write(f"**Pipeline CAPEX:** ${pipeline_capex_m:.2f} M")
            
        st.subheader("Footprint Estimates")
        df_foot = pd.DataFrame({
            "Zone": ["Generation Hall", "Fuel/Logistics", "Cooling/CHP", "BESS Area", "Substation", "Total (+Roads)"],
            f"Area ({u_area_s})": [
                area_gen * (10.764 if is_imperial else 1),
                storage_area_m2 * (10.764 if is_imperial else 1),
                area_chp * (10.764 if is_imperial else 1),
                area_bess * (10.764 if is_imperial else 1),
                area_sub * (10.764 if is_imperial else 1),
                disp_area
            ]
        })
        st.dataframe(df_foot.style.format({f"Area ({u_area_s})": "{:,.0f}"}), use_container_width=True)

    with col2:
        st.subheader("Emissions & Urea")
        st.metric("NOx Emissions", f"{nox_tpy:,.0f} Ton/yr", f"Limit: {limit_nox_tpy}")
        if req_scr:
            st.info(f"SCR Required. Urea Consumption: {urea_vol_yr:,.0f} L/yr")
            tank_u = math.ceil((urea_vol_yr/365)*urea_days / 30000)
            st.write(f"**Urea Tanks ({urea_days} days):** {tank_u}x 30kL Tanks")
        else:
            st.success("No SCR Required.")
            
        st.subheader("Noise")
        if noise_rec > noise_limit:
            st.error(f"🛑 **Noise Fail:** {noise_rec:.1f} dBA (Limit {noise_limit})")
        else:
            st.success(f"✅ **Noise OK:** {noise_rec:.1f} dBA")

with t3:
    st.subheader("Cooling & Tri-Generation")
    if include_chp:
        c1, c2 = st.columns(2)
        c1.metric("Recoverable Heat", f"{total_heat_rec_mw:.1f} MWt")
        c2.metric("Cooling Generated", f"{total_cooling_mw:.1f} MWc", f"{disp_cooling:,.0f} Tons")
        st.metric("Cooling Coverage", f"{cooling_coverage_pct:.1f}%")
        st.progress(min(1.0, cooling_coverage_pct/100))
    else:
        st.info(f"Using **{cooling_method}** for cooling.")
    
    st.metric(f"Water Consumption (WUE)", f"{disp_water:,.0f} {u_water}")

with t4:
    st.subheader("Financial Feasibility & NPV Analysis")
    
    # 1. Cost Index Editor
    st.info(f"**Inst. Ratio Auto-Calc:** Installation Cost (${gen_install_cost:.0f}/kW) vs Equipment Cost (${gen_unit_cost:.0f}/kW)")
    
    st.markdown(f"👇 **Edit Indices to Adjust CAPEX:** (Base: **${gen_unit_cost:.0f}/kW**)")
    edited_capex = st.data_editor(
        df_capex_base, 
        column_config={
            "Default Index": st.column_config.NumberColumn("Cost Index", min_value=0.0, max_value=5.0, step=0.01),
            "Cost (M USD)": st.column_config.NumberColumn("Calculated Cost", format="$%.2fM", disabled=True)
        },
        use_container_width=True
    )
    
    # Recalculate Total CAPEX Dynamic
    final_capex_df = edited_capex.copy()
    total_capex_dynamic = 0
    for index, row in final_capex_df.iterrows():
        if row['Item'] in ["Logistics/Fuel Infra", "Gas Pipeline", "Emissions Control", "BESS System"]:
            total_capex_dynamic += row['Cost (M USD)']
        elif row['Item'] == "Generation Units":
            total_capex_dynamic += gen_cost_total
        else:
            total_capex_dynamic += gen_cost_total * row['Default Index']
    
    # Recalculate Financials based on edited CAPEX
    capex_annualized_dyn = (total_capex_dynamic * 1e6) * crf
    total_annual_cost_dyn = fuel_cost_year + om_cost_year + capex_annualized_dyn + repowering_annualized
    lcoe_dyn = total_annual_cost_dyn / (mwh_year * 1000)
    
    npv_dyn = pv_savings - (total_capex_dynamic * 1e6) - (repowering_pv_m * 1e6)
    if annual_savings > 0:
        payback_years_dyn = (total_capex_dynamic * 1e6) / annual_savings
        roi_dyn = (annual_savings / (total_capex_dynamic * 1e6)) * 100
        payback_str_dyn = f"{payback_years_dyn:.1f} Years"
    else:
        payback_str_dyn = "N/A"; roi_dyn = 0
    
    # Display Financials
    c_f1, c_f2, c_f3, c_f4, c_f5 = st.columns(5)
    c_f1.metric("Total CAPEX (USD)", f"${total_capex_dynamic:.2f} M")
    c_f2.metric("LCOE (Prime)", f"${lcoe_dyn:.4f} / kWh")
    c_f3.metric("Annual Savings", f"${annual_savings/1e6:.2f} M")
    c_f4.metric("NPV (20yr)", f"${npv_dyn/1e6:.2f} M")
    c_f5.metric("Payback", payback_str_dyn, f"ROI: {roi_dyn:.1f}%")
    
    # Sensitivity Chart (New in v45)
    st.divider()
    st.subheader("📊 Gas Price Sensitivity & Sweet Spot")
    
    if breakeven_gas_price > 0:
        st.success(f"🎯 **Gas Price to match Grid Energy Purchase = ${breakeven_gas_price:.2f}/MMBtu**")
    else:
        st.error("⚠️ **No Sweet Spot:** Prime Power is more expensive than Grid even with free gas (Fixed Costs too high).")
        
    fig_sens = go.Figure()
    fig_sens.add_trace(go.Scatter(x=gas_prices_x, y=lcoe_y, mode='lines', name='LCOE (Prime)'))
    fig_sens.add_hline(y=grid_price, line_dash="dash", line_color="red", annotation_text="Grid Price")
    fig_sens.update_layout(
        title="LCOE vs Gas Price",
        xaxis_title="Gas Price (USD/MMBtu)",
        yaxis_title="LCOE (USD/kWh)",
        height=400
    )
    st.plotly_chart(fig_sens, use_container_width=True)

    if use_bess:
        st.caption(f"ℹ️ **Repowering:** Includes NPV of battery replacement every {bess_life_batt} years.")

    # Chart
    cost_data = pd.DataFrame({
        "Component": ["Fuel", "O&M (OPEX)", "CAPEX (Amortized)", "Repowering (Future)"],
        "$/kWh": [
            fuel_cost_year/(mwh_year*1000), 
            om_cost_year/(mwh_year*1000), 
            capex_annualized_dyn/(mwh_year*1000),
            repowering_annualized/(mwh_year*1000)
        ]
    })
    
    fig_bar = px.bar(cost_data, x="Component", y="$/kWh", color="Component", 
                     title="LCOE Breakdown (USD/kWh)", text_auto='.4f')
    st.plotly_chart(fig_bar, use_container_width=True)

# --- FOOTER ---
st.markdown("---")
st.caption("CAT Primary Power Solutions | v2026.46 | Clean & Precise")
