import streamlit as st
import pandas as pd
import numpy as np
import math
import plotly.express as px
import plotly.graph_objects as go

# --- PAGE CONFIGURATION ---
st.set_page_config(page_title="CAT Primary Power Solutions v7.1", page_icon="⚡", layout="wide")

# ==============================================================================
# 0. HYBRID DATA LIBRARY
# ==============================================================================

# A. Generator Library
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

# B. Data Center Profiles
dc_profiles = {
    "AI Training (Steady)": {
        "lf": 92.0, "pue": 1.20, "step_req": 20.0, 
        "desc": "High utilization, liquid cooling, low transients."
    },
    "AI Inference (Dynamic)": {
        "lf": 60.0, "pue": 1.30, "step_req": 60.0, 
        "desc": "Variable traffic, requires fast transient response."
    },
    "Hyperscale Cloud": {
        "lf": 80.0, "pue": 1.15, "step_req": 40.0, 
        "desc": "Optimized efficiency, mixed workloads."
    },
    "Enterprise / Colo": {
        "lf": 50.0, "pue": 1.50, "step_req": 50.0, 
        "desc": "Lower utilization due to redundancy (2N), standard cooling."
    },
    "Crypto Mining": {
        "lf": 98.0, "pue": 1.05, "step_req": 5.0, 
        "desc": "Max utilization, minimal cooling, flat load."
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
    hr_conv_factor = 1.055056 

t = {
    "title": f"⚡ CAT Primary Power Solutions ({freq_hz}Hz)",
    "subtitle": "**Sovereign Energy Solutions.**\nAdvanced modeling for Off-Grid Microgrids, Tri-Generation, and Gas Infrastructure.",
    "sb_1": "1. Site & Requirements",
    "sb_2": "2. Technology Solution",
    "sb_3": "3. Economics & ROI",
    "kpi_net": "Net Capacity",
    "kpi_pue": "Projected PUE"
}

st.title(t["title"])
st.markdown(t["subtitle"])

# ==============================================================================
# 2. INPUTS (SIDEBAR)
# ==============================================================================

with st.sidebar:
    # -------------------------------------------------------------------------
    # GROUP 1: SITE & REQUIREMENTS
    # -------------------------------------------------------------------------
    st.header(t["sb_1"])
    
    st.markdown("🏗️ **Data Center Profile**")
    dc_type = st.selectbox("Data Center Type", list(dc_profiles.keys()))
    profile = dc_profiles[dc_type]
    st.caption(f"ℹ️ *{profile['desc']}*")
    
    # Defaults
    def_step_load = profile['step_req']
    def_load_factor = profile['lf']
    def_use_bess = True if "AI" in dc_type else False
    
    p_it = st.number_input("Critical IT Load (MW)", 1.0, 1000.0, 100.0, step=10.0)
    
    # Efficiency Input Logic
    st.markdown("**Efficiency Input**")
    aux_mode = st.radio("Mode", ["PUE Input", "Auxiliaries (%)"], horizontal=True, label_visibility="collapsed")
    
    if aux_mode == "PUE Input":
        pue_val = st.number_input("Design PUE", 1.0, 3.0, profile['pue'], 0.01)
        dc_aux_pct = (pue_val - 1.0) 
        aux_disp = dc_aux_pct * 100.0
    else:
        def_aux = (profile['pue'] - 1.0) * 100.0
        aux_disp = st.number_input("Auxiliaries (%)", 0.0, 100.0, def_aux, 0.5)
        dc_aux_pct = aux_disp / 100.0
        pue_val = 1.0 + dc_aux_pct
        
    st.caption(f"Calculated: PUE {pue_val:.2f} | Aux {aux_disp:.1f}%")
    
    # --- GROSS LOAD (DESIGN) ---
    design_gross_mw = p_it * pue_val
    st.markdown(f"**Gross Design Load:** `{design_gross_mw:.2f} MW`")
    
    avail_req = st.number_input("Required Availability (%)", 90.0, 99.99999, 99.99, format="%.5f")
    
    # Load Factor Input
    load_factor_pct = st.number_input("Annual Load Factor (%)", 10.0, 100.0, def_load_factor, 
                                      help="Average annual utilization. Affects LCOE (Energy / Efficiency) but NOT Fleet Sizing (Capacity).")
    
    step_load_req = st.number_input("Step Load Req (%)", 0.0, 100.0, def_step_load)
    
    volt_mode = st.radio("Connection Voltage", ["Auto-Recommend", "Manual Selection"], horizontal=True)
    manual_voltage_kv = 0.0
    if volt_mode == "Manual Selection":
        manual_voltage_kv = st.number_input("Voltage (kV)", 0.4, 230.0, 13.8, step=0.1)
    
    st.markdown("🌍 **Site Environment**")
    derate_mode = st.radio("Derate Mode", ["Auto-Calculate", "Manual"], horizontal=True)
    derate_factor_calc = 1.0
    
    if derate_mode == "Auto-Calculate":
        site_temp_c = 35 
        site_alt_m = 100 
        if is_imperial:
            site_temp_f = st.slider(f"Max Temp ({u_temp})", 32, 122, 95)
            site_alt_ft = st.number_input(f"Altitude ({u_dist})", 0, 13000, 328)
            site_temp_c = (site_temp_f - 32) * 5/9
            site_alt_m = site_alt_ft / 3.28084
        else:
            site_temp_c = st.slider(f"Max Temp ({u_temp})", 0, 50, 35)
            site_alt_m = st.number_input(f"Altitude ({u_dist})", 0, 4000, 100)
        
        methane_number = st.number_input("Methane Number (MN)", 30, 100, 80)
        loss_temp = max(0, (site_temp_c - 25) * 0.01) 
        loss_alt = max(0, (site_alt_m - 100) * 0.0001)
        loss_mn = max(0, (75 - methane_number) * 0.005)
        derate_factor_calc = 1.0 - (loss_temp + loss_alt + loss_mn)
        st.caption(f"Calculated Derate: {derate_factor_calc:.3f}")
    else:
        manual_derate_pct = st.number_input("Manual Derate (%)", 0.0, 50.0, 5.0)
        derate_factor_calc = 1.0 - (manual_derate_pct / 100.0)

    st.markdown("🚧 **Constraints**")
    enable_optimizer = st.checkbox("Area Constraint?", value=False)
    max_area_input = 0.0
    area_unit_sel = "m²"
    if enable_optimizer:
        c_a1, c_a2 = st.columns(2)
        area_unit_sel = c_a1.selectbox("Unit", ["m²", "Acres", "Hectares"])
        max_area_input = c_a2.number_input("Max Area", 0.0, 1000000.0, 0.0, step=100.0)

    gas_source = st.selectbox("Fuel Source Availability", ["Pipeline Network", "Pipeline + LNG Backup", "100% LNG Virtual Pipeline"])
    use_pipeline = "Pipeline" in gas_source
    has_lng_storage = "LNG" in gas_source
    is_lng_primary = "100%" in gas_source
    virtual_pipe_mode = "LNG" if has_lng_storage else "Pipeline"

    reg_zone = st.selectbox("Regulatory Zone", ["USA - EPA Major", "EU Standard", "LatAm / No-Reg"])
    limit_nox_tpy = 250.0 if "EPA" in reg_zone else (150.0 if "EU" in reg_zone else 9999.0)
    
    dist_neighbor_m = st.number_input(f"Dist. to Neighbor ({u_dist})", 10.0, 5000.0, 100.0)
    if is_imperial: dist_neighbor_m = dist_neighbor_m / 3.28084
    noise_limit = 70.0 

    st.divider()

    # -------------------------------------------------------------------------
    # GROUP 2: TECHNOLOGY SOLUTION
    # -------------------------------------------------------------------------
    st.header(t["sb_2"])

    st.markdown("⚙️ **Generators**")
    selected_model = st.selectbox("Select Model", list(leps_gas_library.keys()))
    eng_data = leps_gas_library[selected_model]
    
    # Tech Validation Warning
    if eng_data['step_load_pct'] < step_load_req:
        st.warning(f"⚠️ **Warning:** {selected_model} ({eng_data['step_load_pct']}%) may not meet Step Req ({step_load_req}%).")
    
    eff_input_method = st.radio("Efficiency Mode", ["Efficiency (%)", f"Heat Rate ({u_hr})"], horizontal=True)
    def_mw = eng_data['iso_rating_mw']
    def_eff_pct = eng_data['electrical_efficiency'] * 100.0
    def_hr_base = eng_data['heat_rate_lhv'] 
    def_hr_disp = def_hr_base * hr_conv_factor
    
    col_t1, col_t2 = st.columns(2)
    unit_size_iso = col_t1.number_input("Rating (MW)", 0.1, 100.0, def_mw, format="%.2f")
    
    if eff_input_method == "Efficiency (%)":
        eff_user = col_t2.number_input("Eff (ISO %)", 20.0, 65.0, def_eff_pct, format="%.1f")
        base_eff = eff_user / 100.0
    else:
        hr_user = col_t2.number_input(f"HR ({u_hr})", 5000.0, 15000.0, def_hr_disp, format="%.0f")
        hr_btu = hr_user / hr_conv_factor
        base_eff = 3412.14 / hr_btu

    c_aux1, c_aux2 = st.columns(2)
    dist_loss_pct = c_aux1.number_input("Dist Loss (%)", 0.0, 10.0, 1.0) / 100.0
    gen_parasitic_pct = c_aux2.number_input("Parasitics (%)", 0.0, 10.0, 2.5) / 100.0

    c_c1, c_c2 = st.columns(2)
    gen_unit_cost = c_c1.number_input("Equip ($/kW)", 100.0, 3000.0, eng_data['est_cost_kw'], step=10.0)
    gen_install_cost = c_c2.number_input("Install ($/kW)", 50.0, 3000.0, eng_data['est_install_kw'], step=10.0)
    
    c_p1, c_p2 = st.columns(2)
    step_load_cap = c_p1.number_input("Step Cap (%)", 0.0, 100.0, eng_data['step_load_pct'])
    xd_2_pu = c_p2.number_input('Xd" (pu)', 0.01, 0.50, eng_data.get('reactance_xd_2', 0.15), format="%.5f")

    with st.expander("Gen Reliability Stats"):
        c_r1, c_r2 = st.columns(2)
        maint_outage_pct = c_r1.number_input("Maint (%)", 0.0, 20.0, float(eng_data.get('default_maint', 5.0))) / 100.0
        forced_outage_pct = c_r2.number_input("FOR (%)", 0.0, 20.0, float(eng_data.get('default_for', 2.0))) / 100.0

    st.markdown("🔋 **BESS Strategy**")
    use_bess = st.checkbox("Enable BESS", value=def_use_bess)
    
    bess_maint_pct = 0.0; bess_for_pct = 0.0
    bess_cost_kwh = 0.0; bess_cost_kw = 0.0
    bess_life_batt = 10; bess_life_inv = 15; bess_om_kw_yr = 0.0
    
    if use_bess:
        with st.expander("BESS Details", expanded=True):
            c_b1, c_b2 = st.columns(2)
            bess_maint_pct = c_b1.number_input("Maint (%)", 0.0, 10.0, 1.0) / 100.0
            bess_for_pct = c_b2.number_input("FOR (%)", 0.0, 10.0, 0.5) / 100.0
            
            c_c1, c_c2 = st.columns(2)
            bess_cost_kwh = c_c1.number_input("Bat ($/kWh)", 100.0, 1000.0, 280.0)
            bess_cost_kw = c_c2.number_input("Inv ($/kW)", 50.0, 1000.0, 120.0)
            
            c_l1, c_l2 = st.columns(2)
            bess_life_batt = c_l1.number_input("Life Bat (Yr)", 5, 20, 10)
            bess_life_inv = c_l2.number_input("Life Inv (Yr)", 5, 25, 15)
            bess_om_kw_yr = st.number_input("O&M ($/kW-yr)", 0.0, 100.0, 10.0)

    st.markdown("🚚 **Logistics Infrastructure**")
    dist_gas_main_m = st.number_input("Pipeline Dist (m)", 10.0, 20000.0, 1000.0, step=50.0)
    
    if is_imperial:
        supply_pressure_disp = st.number_input(f"Supply Press ({u_press})", 5.0, 1000.0, 60.0) 
        supply_pressure_psi = supply_pressure_disp
    else:
        supply_pressure_disp = st.number_input(f"Supply Press ({u_press})", 0.5, 100.0, 4.1) 
        supply_pressure_psi = supply_pressure_disp * 14.5038

    storage_days = 0; tank_unit_cap = 10000.0 
    tank_mob_cost = 5000.0; tank_area_unit = 40.0
    
    if has_lng_storage:
        with st.expander(f"LNG Storage ({'Primary' if is_lng_primary else 'Backup'})", expanded=True):
            storage_days = st.number_input("Autonomy (Days)", 1, 60, 5)
            c_s1, c_s2 = st.columns(2)
            tank_unit_cap = c_s1.number_input("Tank (Gal)", 1000.0, 100000.0, 10000.0)
            tank_mob_cost = c_s2.number_input("Mob ($)", 0.0, 50000.0, 5000.0)
            tank_area_unit = st.number_input("Area/Tank (m²)", 10.0, 200.0, 40.0)

    st.markdown("❄️ **Cooling & Emissions**")
    include_chp = st.checkbox("Include Tri-Gen (CHP)", value=True)
    
    cooling_method = "Tri-Gen"
    if include_chp:
        with st.expander("CHP Specs"):
            cop_double = st.number_input("COP Double", 0.5, 2.0, 1.2)
            cop_single = st.number_input("COP Single", 0.4, 1.5, 0.7)
    else:
        cool_idx = 0 if is_ai else 1
        cooling_method = st.selectbox("Cooling Tech", ["Water Cooled", "Air Cooled"], index=cool_idx)

    with st.expander("Emission Hardware"):
        urea_days = st.number_input("Urea Days", 1, 30, 7)
        cost_scr_kw = st.number_input("SCR ($/kW)", 0.0, 200.0, 60.0)
        cost_oxicat_kw = st.number_input("Oxicat ($/kW)", 0.0, 100.0, 15.0)
        force_oxicat = st.checkbox("Force Oxicat", value=False)
        source_noise_dba = st.number_input("Source dBA @1m", 60.0, 120.0, 85.0)

    st.divider()

    # -------------------------------------------------------------------------
    # GROUP 3: ECONOMICS
    # -------------------------------------------------------------------------
    st.header(t["sb_3"])
    
    enable_lcoe_target = st.checkbox("Activate LCOE Optimization Loop")
    
    benchmark_price = 0.0
    if enable_lcoe_target:
        target_lcoe = st.number_input("Target LCOE ($/kWh)", 0.05, 0.50, 0.11, step=0.005) 
        benchmark_price = target_lcoe
        grid_price = 0.0 
    else:
        grid_price = st.number_input("Grid Price Benchmark ($/kWh)", 0.05, 0.50, 0.15)
        benchmark_price = grid_price
        target_lcoe = 0.0
    
    gas_price = st.number_input("Gas Price ($/MMBtu)", 1.0, 20.0, 6.5)
    
    if is_lng_primary:
        vp_premium = st.number_input("LNG Premium ($)", 0.0, 15.0, 4.0)
        gas_price += vp_premium

    om_var_price = st.number_input("Var O&M ($/MWh)", 1.0, 50.0, 12.0)
    
    c_e1, c_e2 = st.columns(2)
    project_years = c_e1.number_input("Years", 5, 30, 20)
    op_mode = st.selectbox("Operation Mode", ["Prime (24/7)", "Peaking (Hrs/Day)"])
    if op_mode == "Prime (24/7)":
        op_hours = 8760
    else:
        op_hours = st.slider("Operating Hours/Year", 500, 8760, 2000)
        
    wacc = c_e2.number_input("WACC (%)", 0.0, 15.0, 8.0) / 100.0

# ==============================================================================
# 2. CALCULATION ENGINE (PRIME PHYSICS ENGINE v2)
# ==============================================================================

# --- A. POWER BALANCE ---
p_net_req = design_gross_mw # Includes IT + Aux/PUE

if include_chp:
    cooling_mode = "Thermal (Absorption)"
else:
    cooling_mode = f"Elec ({cooling_method})"

p_dist_loss_mw = p_net_req * dist_loss_pct
p_gen_bus_req = p_net_req + p_dist_loss_mw

# Voltage Selection Logic
if volt_mode == "Manual Selection":
    op_voltage_kv = manual_voltage_kv
    rec_voltage = f"{manual_voltage_kv:.1f} kV (User)"
else:
    if is_50hz:
        rec_voltage = "11 kV" if p_gen_bus_req < 20 else ("33 kV" if p_gen_bus_req > 50 else "11 kV / 33 kV")
        op_voltage_kv = 11.0 if p_gen_bus_req < 35 else 33.0
    else:
        rec_voltage = "13.8 kV" if p_gen_bus_req < 25 else ("34.5 kV" if p_gen_bus_req > 60 else "13.8 kV / 34.5 kV")
        op_voltage_kv = 13.8 if p_gen_bus_req < 45 else 34.5

# --- B. FLEET SIZING (Tri-Vector) ---
unit_site_cap = unit_size_iso * derate_factor_calc
step_mw_req = p_it * (step_load_req / 100.0)

driver_txt = "N/A"
n_steady = 0
n_transient = 0
n_headroom = 0

if use_bess:
    target_load_factor = 0.95 
    n_base_mw = p_gen_bus_req / (1 - gen_parasitic_pct) 
    n_running = math.ceil(n_base_mw / (unit_site_cap * target_load_factor))
    bess_power_req = max(step_mw_req, unit_site_cap) 
    driver_txt = "Steady State (BESS Optimized)"
else:
    n_steady = math.ceil(p_gen_bus_req / (unit_site_cap * 0.90))
    unit_step_mw_cap = unit_site_cap * (step_load_cap / 100.0)
    n_transient = math.ceil(step_mw_req / unit_step_mw_cap)
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

# --- C. RELIABILITY LOOP ---
n_maint = math.ceil(n_running * maint_outage_pct) 

prob_gen_unit = 1.0 - forced_outage_pct
prob_bess_unit = 1.0 - (bess_maint_pct + bess_for_pct)
target_reliability = avail_req / 100.0

n_reserve_gen = 0
n_redundant_bess = 0 

while True:
    n_pool_gen = n_running + n_reserve_gen
    prob_gen_sys = 0.0
    for k in range(n_running, n_pool_gen + 1):
        comb = math.comb(n_pool_gen, k)
        prob = comb * (prob_gen_unit ** k) * ((1 - prob_gen_unit) ** (n_pool_gen - k))
        prob_gen_sys += prob
        
    if use_bess:
        p_fail_bess_unit = 1.0 - prob_bess_unit
        prob_bess_sys = 1.0 - (p_fail_bess_unit ** (1 + n_redundant_bess))
    else:
        prob_bess_sys = 1.0
        
    system_reliability = prob_gen_sys * prob_bess_sys
    if system_reliability >= target_reliability: break
    
    if use_bess and (prob_bess_sys < prob_gen_sys):
        n_redundant_bess += 1
    else:
        n_reserve_gen += 1
    if n_reserve_gen > 25 or n_redundant_bess > 10: break

n_reserve = n_reserve_gen 
n_total = n_running + n_maint + n_reserve
installed_cap = n_total * unit_site_cap
system_reliability_pct = system_reliability * 100.0

reliability_bottleneck = "Generators" 
if use_bess and (prob_bess_sys < prob_gen_sys):
    reliability_bottleneck = "BESS Availability"

bess_multiplier = 1 + n_redundant_bess
bess_power_total = bess_power_req * bess_multiplier
bess_energy_total = bess_power_total * 2 

# --- D. THERMODYNAMICS & ENERGY (UPDATED: LOAD FACTOR) ---
total_parasitics_mw = n_running * (unit_size_iso * gen_parasitic_pct)
p_gross_total = p_gen_bus_req + total_parasitics_mw

# Energy Calc using Load Factor
avg_operating_load_mw = p_gross_total * (load_factor_pct / 100.0)
mwh_year = avg_operating_load_mw * op_hours

# Eff Penalty based on Annual Load Factor
real_load_factor = avg_operating_load_mw / (n_running * unit_site_cap)
base_eff = eng_data['electrical_efficiency']
type_tech = eng_data.get('type', 'High Speed')

if type_tech == "High Speed": 
    if load_factor_pct >= 85: eff_factor = 1.0
    elif load_factor_pct >= 75: eff_factor = 0.99
    elif load_factor_pct >= 50: eff_factor = 0.97
    else: eff_factor = 0.92
else: 
    if load_factor_pct >= 85: eff_factor = 1.0
    else: eff_factor = 0.98

eff_factor = max(eff_factor, 0.50)
gross_eff_site = base_eff * eff_factor
gross_hr_lhv = 3412.14 / gross_eff_site

total_fuel_input_mmbtu_hr = avg_operating_load_mw * (gross_hr_lhv / 1000) 
net_hr_lhv = (total_fuel_input_mmbtu_hr * 1e6) / (p_net_req * (load_factor_pct/100.0) * 1000) 
net_hr_hhv = net_hr_lhv * 1.108

if is_imperial:
    hr_primary = math.ceil(net_hr_lhv)
    unit_primary = "Btu/kWh"
else:
    hr_primary = net_hr_lhv * 0.001055056
    unit_primary = "MJ/kWh"

# --- E. SHORT CIRCUIT ---
gen_mva_total = installed_cap / 0.8
gen_sc_mva = gen_mva_total / xd_2_pu
bess_sc_mva = bess_power_total * 1.5 if use_bess else 0.0

total_sc_mva = gen_sc_mva + bess_sc_mva
isc_ka = total_sc_mva / (math.sqrt(3) * op_voltage_kv)
switchgear_cost_factor = 1.2 if isc_ka > 40 else 1.0

# --- F. THERMAL (CHP) ---
heat_input_mw = total_fuel_input_mmbtu_hr / 3.41214
heat_exhaust_mw = heat_input_mw * 0.28
heat_jacket_mw = heat_input_mw * 0.18

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
if has_lng_storage:
    vol_day = total_mmbtu_day * 12.5 
    num_tanks = math.ceil((vol_day * storage_days)/tank_unit_cap)
    log_capex = num_tanks * tank_mob_cost
    log_text = f"LNG Storage: {vol_day:,.0f} gpd"
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

# --- I. FOOTPRINT (AREA OPTIMIZER) ---
area_gen = n_total * 200 
area_chp = total_cooling_mw * 20 if include_chp else (p_net_req * 10) 
area_bess = bess_power_total * 30 
area_sub = 2500
total_area_m2 = (area_gen + storage_area_m2 + area_chp + area_bess + area_sub) * 1.2

max_area_limit_m2 = 0
area_utilization_pct = 0
is_area_exceeded = False
savings_lng = 0; savings_chp = 0; savings_turb = 0

if enable_optimizer and max_area_input > 0:
    if area_unit_sel == "Acres": max_area_limit_m2 = max_area_input / 0.000247105
    elif area_unit_sel == "Hectares": max_area_limit_m2 = max_area_input * 10000
    else: max_area_limit_m2 = max_area_input

    area_utilization_pct = min(100.0, (total_area_m2 / max_area_limit_m2) * 100)
    is_area_exceeded = total_area_m2 > max_area_limit_m2
    
    savings_lng = storage_area_m2 * 1.2 
    savings_chp = (area_chp - (p_net_req * 10)) * 1.2 
    savings_turb = (area_gen * 0.60) * 1.2 

# --- J. FINANCIALS & LCOE ---
base_gen_cost_kw = gen_unit_cost 
gen_cost_total = (installed_cap * 1000) * base_gen_cost_kw / 1e6 

idx_install = (gen_install_cost / gen_unit_cost) * switchgear_cost_factor
idx_chp = 0.20 if include_chp else 0

bess_capex_m = 0.0
bess_om_annual = 0.0
if use_bess:
    cost_power_part = (bess_power_total * 1000) * bess_cost_kw
    cost_energy_part = (bess_energy_total * 1000) * bess_cost_kwh
    bess_capex_m = (cost_power_part + cost_energy_part) / 1e6
    bess_om_annual = (bess_power_total * 1000 * bess_om_kw_yr) 

pipe_cost_m = (dist_gas_main_m * 150) / 1e6 
civil_cost_m = gen_cost_total * 0.15 
install_cost_m = (gen_cost_total * idx_install) + (gen_cost_total * idx_chp) + (at_capex_total/1e6) + (log_capex/1e6) + pipe_cost_m

total_capex = (gen_cost_total + bess_capex_m + civil_cost_m + install_cost_m) * 1e6 

# OPEX (Fuel based on LOAD FACTOR)
fuel_cost_year = total_fuel_input_mmbtu_hr * op_hours * gas_price 
om_var_year = om_var_price * mwh_year 
om_fixed_year = (installed_cap * 1000 * 12 * 1.5) + bess_om_annual 
if include_chp: om_fixed_year += (total_cooling_mw * 1000 * 10) 

total_opex_year = fuel_cost_year + om_var_year + om_fixed_year

# LCOE
if wacc > 0:
    crf = (wacc * (1 + wacc)**project_years) / ((1 + wacc)**project_years - 1)
else:
    crf = 1.0 / project_years

annualized_capex = total_capex * crf
lcoe_usd_kwh = (annualized_capex + total_opex_year) / (mwh_year * 1000) if mwh_year > 0 else 0.0

# Sweet Spot (LCOE OPTIMIZER)
total_mmbtu_year = total_fuel_input_mmbtu_hr * op_hours
breakeven_gas_price = 0.0
if total_mmbtu_year > 0:
    breakeven_gas_price = ((mwh_year * 1000 * benchmark_price) - annualized_capex - om_var_year - om_fixed_year) / total_mmbtu_year

# ==============================================================================
# 3. DASHBOARD OUTPUT
# ==============================================================================

st.title(f"CAT Primary Power: {selected_model} Solution")
st.markdown(f"**Profile:** {dc_type} | **PUE:** {pue_val:.2f} | **Load Factor:** {load_factor_pct}% | **Reliability:** {system_reliability_pct:.5f}%")

# --- KPI CARDS ---
c1, c2, c3, c4 = st.columns(4)
c1.metric("LCOE", f"${lcoe_usd_kwh:.4f}/kWh", f"Fuel: ${fuel_cost_year/(mwh_year*1000) if mwh_year > 0 else 0:.4f}")
c2.metric("Total CAPEX", f"${total_capex/1e6:.1f} M")
c3.metric("Annual Fuel Cost", f"${fuel_cost_year/1e6:.1f} M", f"Eff Pen: {(eff_factor-1)*-100:.1f}%")
c4.metric("Annual Generation", f"{mwh_year/1e3:.1f} GWh")

st.markdown("---")

# --- TABS ---
tab1, tab2, tab3, tab4 = st.tabs(["📊 Dashboard", "💰 Financials & Optimization", "⚙️ Technical Specs", "🏗️ Area & Logistics"])

# --- TAB 1: DASHBOARD ---
with tab1:
    col_a, col_b = st.columns([1, 1])
    with col_a:
        st.subheader("Cost Structure")
        costs = {
            "CAPEX (Amortized)": annualized_capex,
            "Fuel": fuel_cost_year,
            "Fixed O&M": om_fixed_year,
            "Variable O&M": om_var_year
        }
        df_costs = pd.DataFrame(list(costs.items()), columns=["Category", "Annual Cost ($)"])
        fig_pie = px.pie(df_costs, values='Annual Cost ($)', names='Category', hole=0.4, 
                         color_discrete_sequence=px.colors.qualitative.Set2)
        st.plotly_chart(fig_pie, use_container_width=True)

    with col_b:
        st.subheader("Sensitivity: Utilization Impact")
        lf_range = range(30, 101, 10)
        lcoe_sens = []
        for lf in lf_range:
            e_mwh = design_gross_mw * (lf/100.0) * op_hours
            if type_tech == "High Speed": 
                if lf >= 85: ef = 1.0
                elif lf >= 75: ef = 0.99
                elif lf >= 50: ef = 0.97
                else: ef = 0.92
            else: 
                if lf >= 85: ef = 1.0
                else: ef = 0.98
            ef = max(ef, 0.50)
            hr_sens = (3412.14 / (base_eff * ef))
            fuel_mmbtu = (e_mwh * 1000 * hr_sens) / 1e6
            f_cost = fuel_mmbtu * gas_price
            om = (om_var_price * e_mwh) + om_fixed_year
            t_cost = annualized_capex + f_cost + om
            l = t_cost / (e_mwh * 1000) if e_mwh > 0 else 0
            lcoe_sens.append(l)
        
        fig_sens = px.line(x=lf_range, y=lcoe_sens, markers=True, labels={'x':'Load Factor (%)', 'y':'LCOE ($/kWh)'})
        fig_sens.add_vline(x=load_factor_pct, line_dash="dash", line_color="red", annotation_text="Current")
        st.plotly_chart(fig_sens, use_container_width=True)

# --- TAB 2: FINANCIALS & OPTIMIZATION ---
with tab2:
    col_f1, col_f2 = st.columns([2, 1])
    
    with col_f1:
        st.subheader("Gas Price Sensitivity & Sweet Spot")
        gas_prices_x = np.linspace(1, 15, 15)
        lcoe_y = []
        for gp in gas_prices_x:
            f_cost = total_fuel_input_mmbtu_hr * op_hours * gp
            t_cost = annualized_capex + f_cost + om_var_year + om_fixed_year
            lcoe_y.append(t_cost / (mwh_year * 1000) if mwh_year > 0 else 0)
            
        fig_gas = px.line(x=gas_prices_x, y=lcoe_y, labels={'x': 'Gas Price ($/MMBtu)', 'y': 'LCOE ($/kWh)'})
        fig_gas.add_vline(x=gas_price, line_dash="dash", line_color="red", annotation_text="Current")
        fig_gas.add_hline(y=grid_price, line_dash="dot", line_color="green", annotation_text=f"Benchmark ({grid_price:.2f})")
        st.plotly_chart(fig_gas, use_container_width=True)
        
        if enable_lcoe_target:
            if breakeven_gas_price > 0:
                st.success(f"🎯 **Sweet Spot:** To beat ${benchmark_price:.2f}/kWh, Gas must be < ${breakeven_gas_price:.2f}/MMBtu")
            else:
                st.error("⚠️ **No Sweet Spot:** Prime Power > Target even with free gas.")

    with col_f2:
        st.subheader("Detailed Cost Breakdown")
        fin_df = pd.DataFrame({
            "Item": ["Generation Equipment", "Installation & Civil", "Logistics/BESS/Emission", "Total CAPEX", "Annual Fuel", "Annual O&M"],
            "Value (M USD)": [
                gen_cost_total,
                civil_cost_m,
                install_cost_m - civil_cost_m,
                total_capex/1e6,
                fuel_cost_year/1e6,
                (om_var_year + om_fixed_year)/1e6
            ]
        })
        st.dataframe(fin_df.style.format({"Value (M USD)": "${:,.2f}"}), hide_index=True)

# --- TAB 3: TECH SPECS ---
with tab3:
    st.subheader("Fleet & Availability")
    c_t1, c_t2 = st.columns(2)
    with c_t1:
        st.info(f"**Sizing Driver:** {driver_txt}")
        st.markdown(f"""
        * **Steady State Req:** {n_steady} Units
        * **Transient Req:** {n_transient} Units
        * **Headroom Req:** {n_headroom} Units
        * **Running:** {n_running} | **Reserve:** {n_reserve} | **Maint:** {n_maint}
        * **Total Installed:** {n_total} Units (N+{n_reserve+n_maint})
        """)
        st.write(f"**Calculated Availability:** {system_reliability_pct:.5f}%")
        if use_bess: st.write(f"**BESS Support:** {bess_power_total:.1f} MW (N+{n_redundant_bess})")

    with c_t2:
        st.markdown("### Emissions & Environment")
        st.write(f"**NOx Emissions:** {nox_tpy:,.1f} Tons/Year")
        if req_scr:
            st.warning(f"⚠️ SCR Required! Urea: {urea_vol_yr:,.0f} Gal/Yr")
        else:
            st.success("✅ Emissions Compliant")
        st.write(f"**Noise:** {noise_rec:.1f} dBA @ {dist_neighbor_m:.0f}m")

# --- TAB 4: AREA & LOGISTICS ---
with tab4:
    c_l1, c_l2 = st.columns(2)
    with c_l1:
        st.subheader("Fuel Logistics")
        st.write(f"**Pipeline:** {rec_pipe_dia}\" dia ({pressure_status})")
        if has_lng_storage:
            st.write(f"**LNG Storage:** {num_tanks} Tanks ({tank_unit_cap:,.0f} gal each)")
            st.write(f"**Autonomy:** {storage_days} Days")
            
    with c_l2:
        if enable_optimizer:
            st.subheader("Area Constraint Analysis")
            st.metric("Area Utilization", f"{area_utilization_pct:.1f}%", f"{total_area_m2:,.0f} / {max_area_limit_m2:,.0f} {area_unit_sel}")
            if is_area_exceeded:
                st.error(f"❌ Limit Exceeded by {(total_area_m2 - max_area_limit_m2):,.0f} {area_unit_sel}")
                with st.expander("💡 Space Saving Suggestions"):
                    st.write(f"- Remove LNG: Save {savings_lng:,.0f}")
                    st.write(f"- Remove CHP: Save {savings_chp:,.0f}")
