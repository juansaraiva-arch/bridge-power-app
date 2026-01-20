import streamlit as st
import pandas as pd
import numpy as np
import math
import plotly.express as px
import plotly.graph_objects as go

# --- PAGE CONFIGURATION ---
st.set_page_config(page_title="CAT Bridge Solutions Designer v16", page_icon="🌉", layout="wide")

# ==============================================================================
# 0. HYBRID DATA LIBRARY (RENTAL FLEET: GAS, DIESEL & DUAL FUEL)
# Fuentes: Datasheets LEHX (Cat) & SMT (Solar Turbines)
# ==============================================================================

bridge_rental_library = {
    # --- GAS / DUAL FUEL UNITS ---
    "XGC1900": {
        "description": "Gas Rental Unit (G3516H) - High Efficiency",
        "fuels": ["Natural Gas"],
        "type": "High Speed",
        "iso_rating_mw": {60: 1.9, 50: 1.9}, 
        "electrical_efficiency": 0.392,
        "heat_rate_lhv": 8780,
        "step_load_pct": 40.0,
        "emissions_nox": 0.5,
        "emissions_co": 2.5,
        "default_for": 2.0, "default_maint": 5.0,
        "est_asset_value_kw": 850.0, 
        "est_mob_kw": 80.0,
        "gas_pressure_min_psi": 1.5,
        "reactance_xd_2": 0.14
    },
    "TM2500": {
        "description": "Mobile Gas Turbine (34 MW) - Aero",
        "fuels": ["Natural Gas", "Diesel"], 
        "type": "Gas Turbine",
        "iso_rating_mw": {60: 34.0, 50: 34.0}, 
        "electrical_efficiency": 0.370,
        "heat_rate_lhv": 9220,
        "step_load_pct": 20.0,
        "emissions_nox": 0.6,
        "emissions_co": 0.6,
        "default_for": 1.5, "default_maint": 3.0,
        "est_asset_value_kw": 900.0,
        "est_mob_kw": 120.0,
        "gas_pressure_min_psi": 450.0,
        "reactance_xd_2": 0.17
    },
    "SMT60": {
        "description": "Solar Mobile (Taurus 60) - Dual Fuel",
        "fuels": ["Natural Gas", "Diesel", "Propane"], 
        "type": "Gas Turbine",
        "iso_rating_mw": {60: 5.7, 50: 5.5}, 
        "electrical_efficiency": 0.315,
        "heat_rate_lhv": 10830,
        "step_load_pct": 20.0,
        "emissions_nox": 0.6,
        "emissions_co": 0.6,
        "default_for": 1.0, "default_maint": 2.0,
        "est_asset_value_kw": 950.0,
        "est_mob_kw": 60.0,
        "gas_pressure_min_psi": 250.0,
        "reactance_xd_2": 0.18
    },
    "SMT130": {
        "description": "Solar Mobile (Titan 130) - Dual Fuel",
        "fuels": ["Natural Gas", "Diesel", "Propane"],
        "type": "Gas Turbine",
        "iso_rating_mw": {60: 16.0, 50: 15.8}, 
        "electrical_efficiency": 0.354,
        "heat_rate_lhv": 9630,
        "step_load_pct": 20.0,
        "emissions_nox": 0.6,
        "emissions_co": 0.6,
        "default_for": 1.0, "default_maint": 2.0,
        "est_asset_value_kw": 900.0,
        "est_mob_kw": 70.0,
        "gas_pressure_min_psi": 300.0,
        "reactance_xd_2": 0.18
    },

    # --- DIESEL ONLY UNITS ---
    "XQ2280": {
        "description": "Diesel Power Module (3516C) - Tier 4 Final",
        "fuels": ["Diesel"],
        "type": "High Speed",
        "iso_rating_mw": {60: 1.825, 50: 1.6}, 
        "electrical_efficiency": 0.380, 
        "heat_rate_lhv": 9000,
        "step_load_pct": 80.0,
        "emissions_nox": 0.6,
        "emissions_co": 0.1,
        "default_for": 1.0, "default_maint": 4.0,
        "est_asset_value_kw": 600.0,
        "est_mob_kw": 50.0,
        "gas_pressure_min_psi": 0,
        "reactance_xd_2": 0.13
    },
    "XQC1600": {
        "description": "Diesel Power Module (3516C) - Switchable",
        "fuels": ["Diesel"],
        "type": "High Speed",
        "iso_rating_mw": {60: 1.705, 50: 1.515}, 
        "electrical_efficiency": 0.375,
        "heat_rate_lhv": 9100,
        "step_load_pct": 75.0,
        "emissions_nox": 4.0,
        "emissions_co": 1.0,
        "default_for": 1.5, "default_maint": 4.0,
        "est_asset_value_kw": 550.0,
        "est_mob_kw": 50.0,
        "gas_pressure_min_psi": 0,
        "reactance_xd_2": 0.14
    },
    "XQ1140": {
        "description": "Diesel Rental Set (C32) - Compact",
        "fuels": ["Diesel"],
        "type": "High Speed",
        "iso_rating_mw": {60: 0.91, 50: 0.8}, 
        "electrical_efficiency": 0.360,
        "heat_rate_lhv": 9480,
        "step_load_pct": 100.0,
        "emissions_nox": 4.0,
        "emissions_co": 1.0,
        "default_for": 1.0, "default_maint": 3.0,
        "est_asset_value_kw": 500.0,
        "est_mob_kw": 40.0,
        "gas_pressure_min_psi": 0,
        "reactance_xd_2": 0.12
    }
}

# ==============================================================================
# 1. GLOBAL SETTINGS
# ==============================================================================

with st.sidebar:
    st.image("https://img.icons8.com/color/96/bridge.png", width=60)
    st.header("Global Settings")
    c_glob1, c_glob2 = st.columns(2)
    unit_system = c_glob1.radio("Units", ["Metric (SI)", "Imperial (US)"])
    freq_hz = c_glob2.radio("Frequency", [60, 50])

is_imperial = "Imperial" in unit_system
is_50hz = freq_hz == 50

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

t = {
    "title": f"🌉 CAT Bridge Solutions Designer v16 ({freq_hz}Hz)",
    "subtitle": "**Time-to-Market Accelerator.**\nEngineering, Logistics & Financial Strategy for Bridge Power.",
    "sb_1": "1. Data Center Profile",
    "sb_2": "2. Technology & Fuel",
    "sb_3": "3. Site, Logistics & Noise",
    "sb_4": "4. Strategy (BESS & LNG)",
    "sb_5": "5. Cooling & Env",
    "sb_6": "6. Regulatory & Emissions",
    "sb_7": "7. Financials & Strategy"
}

st.title(t["title"])
st.markdown(t["subtitle"])

# ==============================================================================
# 2. INPUTS (SIDEBAR)
# ==============================================================================

with st.sidebar:
    # --- 1. PROFILE ---
    st.header(t["sb_1"])
    dc_type_sel = st.selectbox("Data Center Type", ["AI Factory (Training)", "Hyperscale Standard"])
    is_ai = "AI" in dc_type_sel
    
    def_step_load = 40.0 if is_ai else 10.0
    def_use_bess = True if is_ai else False
    
    p_it = st.number_input("Critical IT Load (MW)", 1.0, 1000.0, 50.0, step=10.0)
    pue_input = st.number_input("Design PUE", 1.0, 2.0, 1.35, step=0.01)
    
    avail_req = st.number_input("Availability Target (%)", 90.0, 99.99999, 99.99, format="%.5f")
    step_load_req = st.number_input("Expected Step Load (%)", 0.0, 100.0, def_step_load)
    dist_loss_pct = st.number_input("Distribution Losses (%)", 0.0, 10.0, 1.0) / 100.0

    st.divider()

    # --- 2. TECHNOLOGY & FUEL ---
    st.header(t["sb_2"])
    
    # 1. Fuel Selection
    fuel_type_sel = st.selectbox("Fuel Source", ["Natural Gas", "Diesel", "Propane"])
    
    # 2. Filter Models based on Freq & Fuel
    avail_models = []
    for k, v in bridge_rental_library.items():
        if fuel_type_sel in v['fuels']:
            avail_models.append(k)
            
    if not avail_models:
        st.error(f"No units found for {fuel_type_sel} at {freq_hz}Hz")
        st.stop()
        
    selected_model = st.selectbox("Select Bridge Unit", avail_models)
    eng_data = bridge_rental_library[selected_model]
    
    st.info(f"**{eng_data['description']}**")

    # 3. Fuel Logic & Properties
    is_gas = fuel_type_sel == "Natural Gas"
    is_diesel = fuel_type_sel == "Diesel"
    is_propane = fuel_type_sel == "Propane"
    
    virtual_pipe_mode = "None"
    methane_number = 80 # Default
    
    if is_gas:
        st.markdown("🔥 **Gas Properties**")
        methane_number = st.number_input("Methane Number (MN)", 30, 100, 80)
        gas_source = st.radio("Supply Method", ["Pipeline", "Virtual Pipeline (LNG)", "Virtual Pipeline (CNG)"])
        
        if "Pipeline" in gas_source:
            virtual_pipe_mode = "Pipeline"
        elif "LNG" in gas_source:
            virtual_pipe_mode = "LNG"
            storage_days = st.number_input("LNG Storage (Days)", 1, 30, 5)
        elif "CNG" in gas_source:
            virtual_pipe_mode = "CNG"
            storage_days = st.number_input("CNG Storage (Days)", 1, 10, 1) 
            
    elif is_diesel:
        st.markdown("🛢️ **Diesel Logistics**")
        storage_days = st.number_input("Diesel Storage (Days)", 1, 30, 3)
        virtual_pipe_mode = "Diesel"
        
    elif is_propane:
        st.markdown("⚪ **Propane Logistics**")
        storage_days = st.number_input("LPG Storage (Days)", 1, 30, 5)
        virtual_pipe_mode = "Propane"

    # 4. Tech Specs (Dynamic Rating)
    def_mw = eng_data['iso_rating_mw'][freq_hz]
    unit_size_iso = st.number_input("Unit Prime Rating (ISO MW)", 0.1, 100.0, def_mw, format="%.3f")
    
    eff_input_method = st.radio("Efficiency Input", ["Efficiency (%)", "Heat Rate LHV (Btu/kWh)"])
    def_eff_pct = eng_data['electrical_efficiency'] * 100.0
    
    final_elec_eff = 0.0
    if eff_input_method == "Efficiency (%)":
        eff_user = st.number_input("Electrical Efficiency (%)", 20.0, 65.0, def_eff_pct, format="%.1f")
        final_elec_eff = eff_user / 100.0
        hr_btu_kwh = 3412.14 / final_elec_eff
    else:
        hr_user = st.number_input("Heat Rate LHV (Btu/kWh)", 5000.0, 15000.0, eng_data['heat_rate_lhv'], format="%.0f")
        final_elec_eff = 3412.14 / hr_user
        hr_btu_kwh = hr_user

    step_load_cap = st.number_input("Unit Step Load Capability (%)", 0.0, 100.0, eng_data['step_load_pct'])
    
    # Electrical Params
    xd_2_pu = st.number_input('Subtransient Reactance (Xd" pu)', 0.05, 0.30, eng_data.get('reactance_xd_2', 0.15), step=0.01)

    st.caption("Availability Parameters (N+M+S)")
    c_r1, c_r2 = st.columns(2)
    maint_outage_pct = c_r1.number_input("Maint. Unavail (%)", 0.0, 20.0, float(eng_data.get('default_maint', 5.0))) / 100.0
    forced_outage_pct = c_r2.number_input("Forced Outage Rate (%)", 0.0, 20.0, float(eng_data.get('default_for', 2.0))) / 100.0
    
    gen_parasitic_pct = st.number_input("Gen. Parasitic Load (%)", 0.0, 10.0, 2.5) / 100.0

    st.divider()

    # --- 3. SITE ---
    st.header(t["sb_3"])
    
    derate_mode = st.radio("Derate Method", ["Auto-Calculate", "Manual Entry"])
    derate_factor_calc = 1.0
    
    if derate_mode == "Auto-Calculate":
        if is_imperial:
            site_temp_f = st.slider(f"Max Ambient Temp ({u_temp})", 32, 122, 95)
            site_alt_ft = st.number_input(f"Altitude ({u_dist})", 0, 13000, 328)
            site_temp_c = (site_temp_f - 32) * 5/9
            site_alt_m = site_alt_ft / 3.28084
        else:
            site_temp_c = st.slider(f"Max Ambient Temp ({u_temp})", 0, 50, 35)
            site_alt_m = st.number_input(f"Altitude ({u_dist})", 0, 4000, 100)
        
        # Derate Logic
        loss_temp = max(0, (site_temp_c - 25) * 0.01) 
        loss_alt = max(0, (site_alt_m - 100) * 0.0001)
        loss_mn = 0.0
        if is_gas:
            loss_mn = max(0, (75 - methane_number) * 0.005) # Penalty for low MN
            
        derate_factor_calc = 1.0 - (loss_temp + loss_alt + loss_mn)
        st.info(f"Calc Derate: {derate_factor_calc:.3f} (MN Loss: {loss_mn:.1%})")
    else:
        manual_derate_pct = st.number_input("Manual Derate (%)", 0.0, 50.0, 5.0)
        derate_factor_calc = 1.0 - (manual_derate_pct / 100.0)

    # Pipeline Inputs
    if virtual_pipe_mode == "Pipeline":
        st.markdown("⛽ **Pipeline Config**")
        dist_gas_main_m = st.number_input("Distance to Gas Main (m)", 10.0, 20000.0, 1000.0, step=50.0)
        
        if is_imperial:
            supply_pressure_disp = st.number_input(f"Supply Pressure ({u_press})", 5.0, 1000.0, 60.0, step=5.0)
            supply_pressure_psi = supply_pressure_disp
            supply_pressure_bar = supply_pressure_psi * 0.0689476
        else:
            supply_pressure_disp = st.number_input(f"Supply Pressure ({u_press})", 0.5, 100.0, 4.1, step=0.5)
            supply_pressure_bar = supply_pressure_disp
            supply_pressure_psi = supply_pressure_bar * 14.5038
    else:
        dist_gas_main_m = 0; supply_pressure_psi = 0 

    # Electrical Grid
    st.markdown("🔌 **Grid Connection**")
    grid_connected = st.checkbox("Grid Connected (Parallel)", value=True)
    if grid_connected:
        grid_mva_sc = st.number_input("Grid Short Circuit Capacity (MVA)", 50.0, 5000.0, 500.0, step=50.0)
    else:
        grid_mva_sc = 0.0

    # Noise
    st.markdown("🔊 **Noise**")
    dist_neighbor_m = st.number_input(f"Distance to Neighbor ({u_dist})", 10.0, 5000.0, 100.0)
    if is_imperial: dist_neighbor_m = dist_neighbor_m / 3.28084
    source_noise_dba = st.number_input("Source Noise @ 1m (dBA)", 60.0, 120.0, 85.0)
    noise_limit = 70.0 

    st.divider()

    # --- 4. STRATEGY ---
    st.header(t["sb_4"])
    use_bess = st.checkbox("Include BESS", value=def_use_bess)
    
    # LNG Logic
    include_lng = False
    if not is_diesel:
        include_lng = st.checkbox("Include LNG Plant", value=True)
        if include_lng:
            autonomy_days = st.number_input("LNG Autonomy (Days)", 1, 60, 7)

    st.divider()

    # --- 6. REGULATORY & EMISSIONS (PRIME LOGIC) ---
    st.header(t["sb_6"])
    reg_zone = st.selectbox("Regulatory Zone", ["USA - EPA Major", "EU Standard", "LatAm / No-Reg"])
    limit_nox_tpy = 250.0 if "EPA" in reg_zone else (150.0 if "EU" in reg_zone else 9999.0)
    urea_days = st.number_input("Urea Storage (Days)", 1, 30, 7)
    
    # After-Treatment Costs
    st.markdown("🛠️ **After-Treatment Costs (USD)**")
    cost_scr_kw = st.number_input("SCR System Cost (USD/kW)", 0.0, 200.0, 60.0)
    cost_oxicat_kw = st.number_input("Oxidation Cat Cost (USD/kW)", 0.0, 100.0, 15.0)
    force_oxicat = st.checkbox("Force Oxicat Inclusion", value=False)

    st.divider()

    # --- 7. FINANCIALS ---
    st.header(t["sb_7"])
    st.caption("Rental / PPA Structure")
    
    # Dynamic Fuel Price
    if is_gas:
        fuel_price_unit = st.number_input("Gas Price (USD/MMBtu)", 1.0, 20.0, 5.0)
        fuel_price_mmbtu = fuel_price_unit
    elif is_diesel:
        fuel_price_unit = st.number_input("Diesel Price (USD/Gal)", 1.0, 10.0, 3.50)
        fuel_price_mmbtu = fuel_price_unit / 0.138
    else: # Propane
        fuel_price_unit = st.number_input("Propane Price (USD/Gal)", 1.0, 10.0, 1.50)
        fuel_price_mmbtu = fuel_price_unit / 0.091 

    # Virtual Pipeline Premium
    if virtual_pipe_mode in ["LNG", "CNG"]:
        vp_premium = st.number_input("Virtual Pipe Premium ($/MMBtu)", 0.0, 15.0, 4.0, help="Logistics cost")
        fuel_price_mmbtu += vp_premium

    gen_mob_cost = st.number_input("Mobilization/Install (USD/kW)", 10.0, 1000.0, eng_data['est_mob_kw'])
    cap_charge = st.number_input("Capacity Charge (USD/MW-mo)", 5000.0, 100000.0, 28000.0, step=1000.0)
    var_om = st.number_input("Variable O&M (USD/MWh)", 0.0, 100.0, 21.50) 
    grid_rate_kwh = st.number_input("Utility Grid Rate (USD/kWh)", 0.01, 0.50, 0.12, format="%.3f") 
    
    st.markdown("⏱️ **Time-to-Market Analysis**")
    revenue_per_mw_mo = st.number_input("Revenue Loss (USD/MW/mo)", 10000.0, 1000000.0, 150000.0, step=10000.0, help="Revenue lost per month of delay")
    months_saved = st.number_input("Months Saved vs Utility", 1, 60, 18, help="Time bridge power is active before utility arrives")
    
    # Buyout & VPP
    st.caption("Future Value (Post-Grid)")
    buyout_pct = st.number_input("Buyout Residual Value (%)", 0.0, 100.0, 20.0)
    ref_new_capex = eng_data['est_asset_value_kw']
    
    st.caption("VPP Revenue")
    vpp_arb_spread = st.number_input("VPP Arbitrage ($/MWh)", 0.0, 200.0, 40.0)
    vpp_cap_pay = st.number_input("VPP Capacity ($/MW-yr)", 0.0, 100000.0, 28000.0)

# ==============================================================================
# 2. CALCULATION ENGINE
# ==============================================================================

# A. LOAD & POWER (PRIME LOGIC WITH PUE)
p_total_site_load = p_it * pue_input # PUE Based
p_dist_loss = p_total_site_load * dist_loss_pct
p_net_gen_req = p_total_site_load + p_dist_loss
p_gross_req = p_net_gen_req / (1 - gen_parasitic_pct)

# Voltage Logic
if is_50hz:
    rec_voltage_str = "11 kV" if p_gross_req < 20 else ("33 kV" if p_gross_req > 50 else "11 kV / 33 kV")
    op_voltage_kv = 11.0 if p_gross_req < 35 else 33.0
else:
    rec_voltage_str = "13.8 kV" if p_gross_req < 25 else ("34.5 kV" if p_gross_req > 60 else "13.8 kV / 34.5 kV")
    op_voltage_kv = 13.8 if p_gross_req < 45 else 34.5

# B. FLEET SIZING (N+M+S LOGIC from PRIME)
unit_site_cap = unit_size_iso * derate_factor_calc

if use_bess:
    target_load_factor = 0.95 
    n_running = math.ceil(p_gross_req / (unit_site_cap * target_load_factor))
    step_mw_req = p_it * (step_load_req / 100.0)
    bess_power = max(step_mw_req, unit_site_cap) 
    bess_energy = bess_power * 2 
else:
    target_load_factor = 0.85
    step_mw_req = p_it * (step_load_req / 100.0)
    n_calc = math.ceil(p_gross_req / unit_site_cap)
    while True:
        headroom_mw = (n_calc * unit_site_cap) * (step_load_cap/100.0)
        load_cap = n_calc * unit_site_cap
        if headroom_mw >= step_mw_req and load_cap >= p_gross_req:
            break
        n_calc += 1
    n_running = n_calc
    bess_power = 0
    bess_energy = 0

# Maintenance Units
n_maint = math.ceil(n_running * maint_outage_pct)

# Standby/Reserve Logic (Prime Style)
if avail_req < 99.0: n_red_tier = 1 
elif avail_req < 99.9: n_red_tier = 1 
elif avail_req < 99.99: n_red_tier = 2 
else: n_red_tier = 3 

n_forced_buffer = math.ceil(n_running * forced_outage_pct)
n_reserve = max(n_forced_buffer, n_red_tier)

n_total = n_running + n_maint + n_reserve
installed_cap_site = n_total * unit_site_cap

# C. FUEL & LOGISTICS (VIRTUAL PIPELINE LOGIC)
net_eff = final_elec_eff * (1 - (gen_parasitic_pct + dist_loss_pct))
hr_net_lhv_btu = 3412.14 / net_eff
total_mmbtu_day = (p_gross_req * 24 * hr_net_lhv_btu) / 1e6

logistics_info = []
storage_area_m2 = 0

if virtual_pipe_mode == "LNG":
    total_gal_day = total_mmbtu_day * 12.5 # ~12.5 Gal/MMBtu
    req_storage_gal = total_gal_day * storage_days
    num_iso_tanks = math.ceil(req_storage_gal / 10000)
    trucks_day = math.ceil(total_gal_day / 10000)
    logistics_info = [f"Daily: {total_gal_day:,.0f} Gal", f"Storage: {num_iso_tanks}x ISO Tanks", f"Traffic: {trucks_day} Trucks/Day"]
    storage_area_m2 = num_iso_tanks * 40 
    
elif virtual_pipe_mode == "CNG":
    total_scf_day = total_mmbtu_day * 1000 # ~1000 scf/MMBtu
    req_storage_scf = total_scf_day * storage_days
    num_tube_trailers = math.ceil(req_storage_scf / 350000)
    trucks_day = math.ceil(total_scf_day / 350000)
    logistics_info = [f"Daily: {total_scf_day/1e6:,.2f} MMscf", f"Storage: {num_tube_trailers}x Tube Trailers", f"Traffic: {trucks_day} Trucks/Day"]
    if trucks_day > 20: logistics_info.append("⚠️ HIGH TRAFFIC ALERT")
    storage_area_m2 = num_tube_trailers * 60 
    
elif virtual_pipe_mode == "Diesel":
    total_gal_day = total_mmbtu_day * 7.3 # ~7.3 Gal/MMBtu
    req_storage_gal = total_gal_day * storage_days
    num_frac_tanks = math.ceil(req_storage_gal / 20000)
    trucks_day = math.ceil(total_gal_day / 8000)
    logistics_info = [f"Daily: {total_gal_day:,.0f} Gal", f"Storage: {num_frac_tanks}x Frac Tanks", f"Traffic: {trucks_day} Trucks/Day"]
    storage_area_m2 = num_frac_tanks * 50

elif virtual_pipe_mode == "Propane":
    total_gal_day = total_mmbtu_day * 11.0 # ~11 Gal/MMBtu
    req_storage_gal = total_gal_day * storage_days
    num_lpg_tanks = math.ceil(req_storage_gal / 30000)
    trucks_day = math.ceil(total_gal_day / 9000)
    logistics_info = [f"Daily: {total_gal_day:,.0f} Gal", f"Storage: {num_lpg_tanks}x LPG Bullets", f"Traffic: {trucks_day} Trucks/Day"]
    storage_area_m2 = num_lpg_tanks * 80

# Pipeline Calculation
rec_pipe_dia = 0
if virtual_pipe_mode == "Pipeline":
    peak_scfh = (total_mmbtu_day / 24) * 1000
    actual_flow_acfm = peak_scfh * (14.7 / (supply_pressure_psi + 14.7)) / 60 
    target_area_ft2 = actual_flow_acfm / (65 * 60) 
    target_dia_in = math.sqrt(target_area_ft2 * 4 / math.pi) * 12
    rec_pipe_dia = max(4, math.ceil(target_dia_in))

# D. SHORT CIRCUIT (PRIME LOGIC)
gen_mva_total = installed_cap_site / 0.8
gen_sc_mva = gen_mva_total / xd_2_pu
total_sc_mva = gen_sc_mva + grid_mva_sc
isc_ka = total_sc_mva / (math.sqrt(3) * op_voltage_kv)
standard_breakers = [25, 31.5, 40, 50, 63]
rec_breaker = 63
for b in standard_breakers:
    if b > (isc_ka * 1.1): 
        rec_breaker = b
        break

# E. EMISSIONS (PRIME LOGIC)
attenuation = 20 * math.log10(dist_neighbor_m)
noise_rec = source_noise_dba + (10 * math.log10(n_running)) - attenuation
total_bhp = p_gross_req * 1341
nox_tpy = (eng_data['emissions_nox'] * total_bhp * 8760) / 907185
req_scr = nox_tpy > limit_nox_tpy
urea_vol_yr = p_gross_req * 1.5 * 8760 if req_scr else 0

at_capex_total = 0
if req_scr:
    at_capex_total += (installed_cap_site * 1000) * cost_scr_kw
if force_oxicat: 
    at_capex_total += (installed_cap_site * 1000) * cost_oxicat_kw

# F. FINANCIALS & TIME TO MARKET
# 1. LCOE Bridge
gen_mwh_yr = p_gross_req * 8760
fuel_cost_mwh = (hr_net_lhv_btu / 1e6) * fuel_price_mmbtu
rental_cost_yr = installed_cap_site * cap_charge * 12
rental_cost_mwh = rental_cost_yr / gen_mwh_yr
lcoe_bridge = fuel_cost_mwh + rental_cost_mwh + var_om
lcoe_utility = grid_rate_kwh * 1000

# 2. Opportunity Cost (Time to Market)
gross_revenue_gain = (p_it * revenue_per_mw_mo * months_saved)
bridge_premium_mwh = lcoe_bridge - lcoe_utility
total_energy_during_bridge = p_gross_req * 730 * months_saved # MWh approx
cost_of_bridge_premium = (bridge_premium_mwh * total_energy_during_bridge) / 1e6 
# Mobilization & Setup (Includes Emissions Capex)
capex_setup_m = ((installed_cap_site * 1000 * gen_mob_cost) + at_capex_total) / 1e6

net_benefit_m = (gross_revenue_gain / 1e6) - cost_of_bridge_premium - capex_setup_m

# 3. Buyout & Future Value
total_asset_value_m = (installed_cap_site * 1000 * ref_new_capex) / 1e6
buyout_price_m = total_asset_value_m * (buyout_pct / 100.0)
savings_vs_new = total_asset_value_m - buyout_price_m

rev_arb = installed_cap_site * vpp_arb_spread * 365 
rev_cap = installed_cap_site * vpp_cap_pay
total_vpp_yr_m = (rev_arb + rev_cap) / 1e6

# G. FOOTPRINT BREAKDOWN
area_gen_total = n_total * 150 
area_bess_total = bess_power * 30 
area_urea_total = (math.ceil((urea_vol_yr/365)*7/30000) * 50) if req_scr else 0
area_sub_total = 2500 # MV Yard
# storage_area_m2 comes from Fuel Logic

total_area_m2 = (area_gen_total + storage_area_m2 + area_bess_total + area_urea_total + area_sub_total) * 1.2 

# ==============================================================================
# 3. DASHBOARD
# ==============================================================================

if is_imperial:
    d_area_s = total_area_m2 * 10.764; u_as = "ft²"
    d_area_l = (total_area_m2 * 10.764) / 43560; u_al = "Acres"
else:
    d_area_s = total_area_m2; u_as = "m²"
    d_area_l = total_area_m2 / 10000; u_al = "Ha"

c1, c2, c3, c4 = st.columns(4)
c1.metric("Bridge Capacity", f"{p_net_gen_req:.1f} MW", f"IT Load: {p_it:.1f} MW")
c2.metric("Fleet Configuration", f"{n_total} Units", f"N+M+S: {n_running}+{n_maint}+{n_reserve}")
c3.metric("LCOE (Bridge)", f"${lcoe_bridge:.2f}/MWh", f"Grid: ${lcoe_utility:.2f}")
c4.metric("Net Benefit (TtM)", f"${net_benefit_m:.1f} M", f"{months_saved} Months Saved")

st.divider()

t1, t2, t3, t4 = st.tabs(["⚙️ Engineering", "🏗️ Site & Env", "💰 Business Case", "🔮 Future Value"])

with t1:
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Power Balance (PUE)")
        df_bal = pd.DataFrame({
            "Item": ["Critical IT", "Facility Aux (PUE)", "Dist. Losses", "Gen. Parasitics", "TOTAL GROSS"],
            "MW": [p_it, (p_total_site_load - p_it), p_dist_loss, (p_gross_req - p_net_gen_req), p_gross_req]
        })
        st.dataframe(df_bal.style.format({"MW": "{:.2f}"}), use_container_width=True)
        
        st.subheader("Electrical Sizing")
        st.write(f"**Operating Voltage:** {op_voltage_kv} kV")
        st.write(f"**Grid Contribution:** {grid_mva_sc} MVA")
        st.write(f"**Gen Contribution:** {gen_sc_mva:.1f} MVA (Xd\" {xd_2_pu})")
        st.markdown(f"**Total Short Circuit:** :red[**{isc_ka:.1f} kA**]")
        st.success(f"✅ Recommended Switchgear Rating: **{rec_breaker} kA**")
        
    with col2:
        st.subheader("Fleet Strategy (N+M+S)")
        st.write(f"**Target Avail:** {avail_req}% | **Model:** {selected_model}")
        st.write(f"**Site Capacity (Derated):** {unit_site_cap:.2f} MW")
        st.markdown("---")
        st.write(f"**N (Running):** {n_running}")
        st.write(f"**M (Maintenance @ {maint_outage_pct*100:.1f}%):** {n_maint}")
        st.write(f"**S (Standby/Reserve):** {n_reserve}")
        st.caption(f"Reserve logic: Max of FOR Buffer ({n_forced_buffer}) or Tier Requirement ({n_red_tier})")
        st.metric("Total Installed Fleet", f"{n_total} Units", f"{installed_cap_site:.1f} MW Total")

        if use_bess:
            st.divider()
            st.subheader("BESS Sizing")
            st.info(f"⚡ **System:** {bess_power:.1f} MW / {bess_energy:.1f} MWh")
            st.caption("Duration: 2 Hours (Standard Bridge Spec)")

with t2:
    c_e1, c_e2 = st.columns(2)
    with c_e1:
        st.subheader(f"Fuel Logistics: {virtual_pipe_mode}")
        if virtual_pipe_mode == "Pipeline":
            st.metric("Rec. Pipe Diameter", f"{rec_pipe_dia:.0f} inches")
        elif logistics_info:
            for item in logistics_info:
                st.write(f"• {item}")
            st.metric("Est. Storage Area", f"{storage_area_m2:.0f} m²")
            
        st.divider()
        st.subheader("Footprint Estimate")
        
        # Breakdown Table
        df_foot = pd.DataFrame({
            "Zone": ["Generation Hall", "Fuel Logistics", "BESS", "Emissions/Urea", "Substation", "Total (+Roads)"],
            f"Area ({u_as})": [
                area_gen_total * (10.764 if is_imperial else 1),
                storage_area_m2 * (10.764 if is_imperial else 1),
                area_bess_total * (10.764 if is_imperial else 1),
                area_urea_total * (10.764 if is_imperial else 1),
                area_sub_total * (10.764 if is_imperial else 1),
                d_area_s
            ]
        })
        st.dataframe(df_foot.style.format({f"Area ({u_as})": "{:,.0f}"}), use_container_width=True)
        st.metric("TOTAL LAND REQUIRED", f"{d_area_l:.2f} {u_al}")
        
    with c_e2:
        st.subheader("Emissions & Urea")
        st.write(f"NOx: {nox_tpy:.0f} Tons/yr")
        if req_scr:
            st.warning("SCR Required (Zone Limit Exceeded)")
            tank_u = math.ceil((urea_vol_yr/365)*urea_days/30000)
            st.write(f"Urea Tanks: {tank_u}x 30kL")
            st.write(f"Emissions Setup Cost: ${at_capex_total/1e6:.2f} M")
        else:
            st.success("No SCR Required")
            
        st.divider()
        st.subheader("Acoustics")
        st.write(f"Receiver Noise: **{noise_rec:.1f} dBA**")
        if noise_rec > noise_limit:
            st.error(f"Exceeds Limit ({noise_limit} dBA)")
        else:
            st.success("Compliant")

with t3:
    st.header("💰 Time-to-Market Analysis (Bridge Phase)")
    
    # 1. Waterfall Chart for Net Benefit (FIXED: NEGATIVE VALUES FOR COSTS)
    fig_water = go.Figure(go.Waterfall(
        name = "20", orientation = "v",
        measure = ["relative", "relative", "relative", "total"],
        x = ["Gross Revenue Gained", "Bridge Energy Premium", "Setup & Mob Cost", "NET BENEFIT"],
        textposition = "outside",
        text = [f"+{gross_revenue_gain/1e6:.1f}M", f"-{cost_of_bridge_premium:.1f}M", f"-{capex_setup_m:.1f}M", f"${net_benefit_m:.1f}M"],
        y = [gross_revenue_gain/1e6, -cost_of_bridge_premium, -capex_setup_m, net_benefit_m], # FIXED HERE
        connector = {"line":{"color":"rgb(63, 63, 63)"}},
    ))
    fig_water.update_layout(title = f"Value Created by Deploying {months_saved} Months Early", showlegend = False)
    st.plotly_chart(fig_water, use_container_width=True)
    
    # 2. LCOE
    st.subheader("Cost of Energy Comparison")
    lcoe_data = pd.DataFrame({
        "Cost Component": ["Fuel", "Rental (Capacity)", "Variable O&M", "Utility Tariff"],
        "$/MWh": [fuel_cost_mwh, rental_cost_mwh, var_om, lcoe_utility],
        "Type": ["Bridge", "Bridge", "Bridge", "Utility"]
    })
    fig_lcoe = px.bar(lcoe_data, x="Type", y="$/MWh", color="Cost Component", title="LCOE Composition", text_auto='.1f')
    st.plotly_chart(fig_lcoe, use_container_width=True)

with t4:
    st.header("🔮 Post-Grid Strategy (Future Value)")
    
    c_b1, c_b2 = st.columns(2)
    with c_b1:
        st.subheader("Asset Transfer (Buyout)")
        st.metric("Est. Buyout Price", f"${buyout_price_m:.1f} M", f"{buyout_pct}% Residual")
        st.metric("Value of New Plant", f"${total_asset_value_m:.1f} M")
        st.success(f"**Avoided CAPEX:** ${savings_vs_new:.1f} Million")
        
    with c_b2:
        st.subheader("Virtual Power Plant (VPP)")
        st.write("Revenue potential if assets are kept for Grid Services:")
        st.metric("Total VPP Revenue", f"${total_vpp_yr_m:.1f} M/year")
        st.write(f"• Arbitrage: ${rev_arb/1e6:.1f} M")
        st.write(f"• Capacity Payments: ${rev_cap/1e6:.1f} M")

# --- FOOTER ---
st.markdown("---")
st.caption("CAT Bridge Solutions Designer v16 | Powered by Prime Engineering Engine")
