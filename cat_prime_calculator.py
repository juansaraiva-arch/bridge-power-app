import streamlit as st
import pandas as pd
import numpy as np
import math
import plotly.express as px

# --- PAGE CONFIGURATION ---
st.set_page_config(page_title="CAT Prime Solution Designer", page_icon="⚡", layout="wide")

# ==============================================================================
# 0. DATA LIBRARY (FROM EXCEL "Libreria.xlsx")
# ==============================================================================

leps_gas_library = {
    "XGC1900": {
        "description": "Mobile Power Module (High Speed)",
        "type": "High Speed",
        "iso_rating_mw": 1.9,
        "electrical_efficiency": 0.392,
        "heat_rate_lhv": 8780, # 8.78 MMBtu/MWh
        "step_load_pct": 40.0,
        "emissions_nox": 0.5, # approx from 0.25-0.5 g/Nm3
        "emissions_co": 2.5,
        "default_for": 3.0, 
        "default_maint": 6.0, # 91% Avail
        "est_cost_kw": 775.0 
    },
    "G3520FR": {
        "description": "Fast Response Gen Set (High Speed)",
        "type": "High Speed",
        "iso_rating_mw": 2.5,
        "electrical_efficiency": 0.386,
        "heat_rate_lhv": 8836,
        "step_load_pct": 55.0, # Fast Response
        "emissions_nox": 0.5,
        "emissions_co": 2.1,
        "default_for": 3.0,
        "default_maint": 6.0, # 91% Avail
        "est_cost_kw": 575.0
    },
    "G3520K": {
        "description": "High Efficiency Gen Set (High Speed)",
        "type": "High Speed",
        "iso_rating_mw": 2.4, # Per Excel
        "electrical_efficiency": 0.453,
        "heat_rate_lhv": 7638,
        "step_load_pct": 35.0,
        "emissions_nox": 0.3, # 0.267 g/Nm3
        "emissions_co": 2.3,
        "default_for": 3.0,
        "default_maint": 6.0, # 91% Avail
        "est_cost_kw": 575.0
    },
    "CG260-16": {
        "description": "Cogeneration Specialist (High Speed)",
        "type": "High Speed",
        "iso_rating_mw": 3.96, # 3.957 MW
        "electrical_efficiency": 0.434,
        "heat_rate_lhv": 7860,
        "step_load_pct": 25.0,
        "emissions_nox": 0.5,
        "emissions_co": 1.8,
        "default_for": 3.0,
        "default_maint": 5.0, # 92% Avail
        "est_cost_kw": 675.0
    },
    "Titan 130": {
        "description": "Solar Gas Turbine (16.5 MW)",
        "type": "Gas Turbine",
        "iso_rating_mw": 16.5,
        "electrical_efficiency": 0.354,
        "heat_rate_lhv": 9630,
        "step_load_pct": 15.0,
        "emissions_nox": 0.6, # ~25 ppm
        "emissions_co": 0.6,
        "default_for": 1.0,
        "default_maint": 2.0, # 97% Avail
        "est_cost_kw": 775.0
    },
    "Titan 250": {
        "description": "Solar Gas Turbine (23.2 MW)",
        "type": "Gas Turbine",
        "iso_rating_mw": 23.2,
        "electrical_efficiency": 0.386,
        "heat_rate_lhv": 8670,
        "step_load_pct": 15.0,
        "emissions_nox": 0.6,
        "emissions_co": 0.6,
        "default_for": 1.0,
        "default_maint": 2.0, # 97% Avail
        "est_cost_kw": 775.0
    },
    "Titan 350": {
        "description": "Solar Gas Turbine (38.0 MW)",
        "type": "Gas Turbine",
        "iso_rating_mw": 38.0,
        "electrical_efficiency": 0.402,
        "heat_rate_lhv": 8495,
        "step_load_pct": 15.0,
        "emissions_nox": 0.6,
        "emissions_co": 0.6,
        "default_for": 1.0,
        "default_maint": 2.0, # 97% Avail
        "est_cost_kw": 775.0
    },
    "G20CM34": {
        "description": "Medium Speed Baseload Platform",
        "type": "Medium Speed",
        "iso_rating_mw": 9.76, # Per Excel
        "electrical_efficiency": 0.475,
        "heat_rate_lhv": 7480,
        "step_load_pct": 10.0,
        "emissions_nox": 0.5,
        "emissions_co": 0.5,
        "default_for": 3.0, 
        "default_maint": 5.0, # 92% Avail
        "est_cost_kw": 700.0
    }
}

# ==============================================================================
# 1. GLOBAL SETTINGS
# ==============================================================================

with st.sidebar:
    st.image("https://img.icons8.com/color/96/generator.png", width=60)
    st.header("Global Settings")
    c_glob1, c_glob2 = st.columns(2)
    unit_system = c_glob1.radio("Units", ["Metric (SI)", "Imperial (US)"])
    freq_hz = c_glob2.radio("System Frequency", [60, 50])

is_imperial = "Imperial" in unit_system
is_50hz = freq_hz == 50

# Unit Definitions
if is_imperial:
    u_temp, u_dist, u_area_s, u_area_l = "°F", "ft", "ft²", "Acres"
    u_vol, u_mass, u_power = "gal", "Short Tons", "MW"
    u_energy, u_therm, u_water = "MWh", "MMBtu", "gal/day"
else:
    u_temp, u_dist, u_area_s, u_area_l = "°C", "m", "m²", "Ha"
    u_vol, u_mass, u_power = "m³", "Tonnes", "MW"
    u_energy, u_therm, u_water = "MWh", "GJ", "m³/day"

# Dictionary
t = {
    "title": f"⚡ CAT Prime Solution Designer ({freq_hz}Hz)",
    "subtitle": "**Sovereign Energy Solutions.**\nAdvanced modeling for Off-Grid Microgrids, Tri-Generation, and ROI Analysis.",
    "sb_1": "1. Data Center Profile",
    "sb_2": "2. Generation Technology",
    "sb_3": "3. Site & Neighbors",
    "sb_4": "4. Strategy (BESS & LNG)",
    "sb_5": "5. Cooling & Tri-Gen",
    "sb_6": "6. Regulatory & Urea",
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
    step_load_req = st.number_input("Expected Step Load (%)", 0.0, 100.0, def_step_load)
    dc_aux_pct = st.number_input("DC Building Auxiliaries (%)", 0.0, 20.0, 5.0) / 100.0
    
    # Distribution Losses Variable
    dist_loss_pct = st.number_input("Distribution Losses (%)", 0.0, 10.0, 1.0) / 100.0

    st.divider()

    # --- 2. GENERATION TECH ---
    st.header(t["sb_2"])
    
    selected_model = st.selectbox("Select CAT/Solar Model", list(leps_gas_library.keys()))
    eng_data = leps_gas_library[selected_model]
    st.caption(f"**{eng_data['description']}**")
    
    eff_input_method = st.radio("Efficiency Input Mode", ["Efficiency (%)", "Heat Rate LHV (Btu/kWh)"])
    
    def_mw = eng_data['iso_rating_mw']
    def_eff_pct = eng_data['electrical_efficiency'] * 100.0
    def_hr_lhv = eng_data['heat_rate_lhv']
    
    unit_size_iso = st.number_input("Unit Prime Rating (ISO MW)", 0.1, 100.0, def_mw, format="%.2f")
    
    final_elec_eff = 0.0
    if eff_input_method == "Efficiency (%)":
        eff_user = st.number_input("Electrical Efficiency (ISO %)", 20.0, 65.0, def_eff_pct, format="%.1f")
        final_elec_eff = eff_user / 100.0
    else:
        hr_user = st.number_input("Heat Rate LHV (Btu/kWh)", 5000.0, 15000.0, def_hr_lhv, format="%.0f")
        final_elec_eff = 3412.14 / hr_user

    # UNIT COST INPUT
    st.markdown("💰 **Asset Valuation**")
    gen_unit_cost = st.number_input("Generator Unit Cost ($/kW)", 100.0, 3000.0, eng_data.get('est_cost_kw', 500.0), step=10.0)

    step_load_cap = st.number_input("Unit Step Load Capability (%)", 0.0, 100.0, eng_data['step_load_pct'])
    
    st.caption("Availability Parameters (N+M+S)")
    c_r1, c_r2 = st.columns(2)
    maint_outage_pct = c_r1.number_input("Maint. Unavail (%)", 0.0, 20.0, float(eng_data.get('default_maint', 5.0))) / 100.0
    forced_outage_pct = c_r2.number_input("Forced Outage Rate (%)", 0.0, 20.0, float(eng_data.get('default_for', 2.0))) / 100.0
    
    gen_parasitic_pct = st.number_input("Gen. Auxiliaries (%)", 0.0, 10.0, 2.5) / 100.0

    c_e1, c_e2 = st.columns(2)
    raw_nox = c_e1.number_input("Native NOx (g/bhp-hr)", 0.0, 10.0, eng_data['emissions_nox'])
    raw_co = c_e2.number_input("Native CO (g/bhp-hr)", 0.0, 10.0, eng_data['emissions_co'])

    st.divider()

    # --- 3. SITE & NEIGHBORS ---
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
        
        methane_number = st.number_input("Gas Methane Number (MN)", 30, 100, 80)
        
        loss_temp = max(0, (site_temp_c - 25) * 0.01) 
        loss_alt = max(0, (site_alt_m - 100) * 0.0001)
        # Simple MN penalty: 0.5% per point below 75
        loss_mn = max(0, (75 - methane_number) * 0.005)
        
        derate_factor_calc = 1.0 - (loss_temp + loss_alt + loss_mn)
        st.info(f"Calculated Derate: {derate_factor_calc:.3f} (MN Loss: {loss_mn:.1%})")
        
    else:
        manual_derate_pct = st.number_input("Manual Derate (%)", 0.0, 50.0, 5.0)
        derate_factor_calc = 1.0 - (manual_derate_pct / 100.0)

    dist_neighbor_m = st.number_input(f"Dist. to Neighbor ({u_dist})", 10.0, 5000.0, 100.0)
    if is_imperial: dist_neighbor_m = dist_neighbor_m / 3.28084
    
    noise_zone = st.selectbox("Zone Type", ["Industrial", "Residential"])
    noise_limit = 70.0 if noise_zone == "Industrial" else 55.0
    source_noise_dba = st.number_input("Source Noise @ 1m (dBA)", 60.0, 120.0, 85.0)

    st.divider()

    # --- 4. STRATEGY (BESS & LNG) ---
    st.header(t["sb_4"])
    use_bess = st.checkbox("Include BESS (Synthetic Inertia)", value=def_use_bess)
    include_lng = st.checkbox("Include LNG Plant (Sovereignty)", value=True)
    autonomy_days = st.number_input("LNG Autonomy (Days)", 1, 60, 15) if include_lng else 0

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
    
    # --- 6. REGULATORY ---
    st.header(t["sb_6"])
    reg_zone = st.selectbox("Regulatory Zone", ["USA - EPA Major", "EU Standard", "LatAm / No-Reg"])
    limit_nox_tpy = 250.0 if "EPA" in reg_zone else (150.0 if "EU" in reg_zone else 9999.0)
    urea_days = st.number_input("Urea Storage (Days)", 1, 30, 7)

    st.divider()

    # --- 7. ECONOMICS ---
    st.header(t["sb_7"])
    gas_price = st.number_input("Gas Price ($/MMBtu)", 1.0, 20.0, 6.5)
    om_var_price = st.number_input("Variable O&M ($/MWh)", 1.0, 50.0, 12.0)
    grid_price = st.number_input("Grid Price ($/kWh)", 0.05, 0.50, 0.15)
    project_years = st.number_input("Project Years", 5, 30, 20)
    wacc = st.number_input("WACC (%)", 0.0, 15.0, 8.0) / 100.0

# ==============================================================================
# 2. CALCULATION ENGINE
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

total_plant_loss_pct = gen_parasitic_pct + dist_loss_pct 
p_gross_req = p_net_req / (1 - total_plant_loss_pct)
p_parasitic_mw = p_gross_req * gen_parasitic_pct
p_loss_mw = p_gross_req * dist_loss_pct

if is_50hz:
    rec_voltage = "11 kV" if p_gross_req < 20 else ("33 kV" if p_gross_req > 50 else "11 kV / 33 kV")
else:
    rec_voltage = "13.8 kV" if p_gross_req < 25 else ("34.5 kV" if p_gross_req > 60 else "13.8 kV / 34.5 kV")

# --- B. FLEET & RELIABILITY (SENSITIVE ALGORITHM) ---
unit_site_cap = unit_size_iso * derate_factor_calc

# 1. N (Operation)
if use_bess:
    target_load_factor = 0.95 
    n_running = math.ceil(p_gross_req / (unit_site_cap * target_load_factor))
    step_mw_req = p_it * (step_load_req / 100.0)
    bess_power = max(step_mw_req, unit_site_cap) 
    bess_energy = bess_power * 2 
else:
    step_mw_req = p_it * (step_load_req / 100.0)
    target_load_factor = 0.80 
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

# 2. M (Maintenance)
n_maint = math.ceil(n_running * maint_outage_pct)

# 3. S (Standby) - SENSITIVE ALGORITHM
if avail_req < 99.0:
    n_red_tier = 1 
elif avail_req < 99.9:
    n_red_tier = 1 
elif avail_req < 99.99:
    n_red_tier = 2 
else: # >= 99.99
    n_red_tier = 3 

n_forced_buffer = math.ceil(n_running * forced_outage_pct)
n_reserve = max(n_forced_buffer, n_red_tier)

n_total = n_running + n_maint + n_reserve
installed_cap = n_total * unit_site_cap

# Heat Rates
hr_penalty = 1.05 if not use_bess else 1.0
real_elec_eff = final_elec_eff / hr_penalty
net_eff = real_elec_eff * (1 - total_plant_loss_pct)
hr_net_lhv_btu = 3412.14 / net_eff
hr_net_hhv_btu = hr_net_lhv_btu * 1.108

# --- C. THERMAL ---
heat_input_mw = p_gross_req / real_elec_eff
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

# --- D. LNG ---
total_gas_energy_day_mmbtu = (p_gross_req * 24 * 3412.14) / (real_elec_eff * 1e6)
gas_vol_day_m3 = total_gas_energy_day_mmbtu * 28.26
required_storage_m3 = (gas_vol_day_m3 / 600) * autonomy_days if include_lng else 0
num_tanks = math.ceil(required_storage_m3 / 3000) if include_lng else 0

# --- E. EMISSIONS ---
attenuation = 20 * math.log10(dist_neighbor_m)
noise_rec = source_noise_dba + (10 * math.log10(n_running)) - attenuation
total_bhp = p_gross_req * 1341
nox_tpy = (raw_nox * total_bhp * 8760) / 907185
req_scr = nox_tpy > limit_nox_tpy
urea_vol_yr = p_gross_req * 1.5 * 8760 if req_scr else 0

# --- F. FOOTPRINT ---
area_gen = n_total * 200 
area_lng = num_tanks * 600 
area_chp = total_cooling_mw * 20 if include_chp else (p_net_req * 10) 
area_bess = bess_power * 30 
area_sub = 2500
total_area_m2 = (area_gen + area_lng + area_chp + area_bess + area_sub) * 1.2

# --- G. FINANCIALS (UPDATED WITH UNIT COST) ---
base_gen_cost_kw = gen_unit_cost 
gen_cost_total = (installed_cap * 1000) * base_gen_cost_kw / 1e6 

idx_elec = 0.20
idx_civil = 0.15
idx_chp = 0.20 if include_chp else 0
idx_bess = 0.30 if use_bess else 0
idx_lng = 0.25 if include_lng else 0

cost_items = [
    {"Item": "Generation Units", "Default Index": 1.00, "Cost ($M)": gen_cost_total},
    {"Item": "Electrical (BOP)", "Default Index": idx_elec, "Cost ($M)": gen_cost_total * idx_elec},
    {"Item": "Civil & Engineering", "Default Index": idx_civil, "Cost ($M)": gen_cost_total * idx_civil},
    {"Item": "Tri-Gen Plant", "Default Index": idx_chp, "Cost ($M)": gen_cost_total * idx_chp},
    {"Item": "BESS System", "Default Index": idx_bess, "Cost ($M)": gen_cost_total * idx_bess},
    {"Item": "LNG Infra", "Default Index": idx_lng, "Cost ($M)": gen_cost_total * idx_lng},
]
df_capex_base = pd.DataFrame(cost_items)

# ==============================================================================
# 3. DASHBOARD OUTPUT
# ==============================================================================

if is_imperial:
    disp_lng_store = required_storage_m3 * 264.172 
    disp_cooling = total_cooling_mw * 284.345 
    disp_water = water_cons_daily_m3 * 264.172 
    disp_area = total_area_m2 * 10.764 
    disp_dist = dist_neighbor_m * 3.28
    footprint_large_val = total_area_m2 * 0.000247105 # Acres
    footprint_unit = "Acres"
else:
    disp_lng_store = required_storage_m3
    disp_cooling = total_cooling_mw
    disp_water = water_cons_daily_m3
    disp_area = total_area_m2
    disp_dist = dist_neighbor_m
    footprint_large_val = total_area_m2 / 10000.0 # Hectares
    footprint_unit = "Ha"

# --- TOP KPIS ---
c1, c2, c3, c4 = st.columns(4)
c1.metric(t["kpi_net"], f"{p_net_req:.1f} MW", f"Gross: {p_gross_req:.1f} MW")
c2.metric("Net Heat Rate (LHV)", f"{hr_net_lhv_btu:,.0f} Btu/kWh", f"HHV: {hr_net_hhv_btu:,.0f}")
c3.metric("Rec. Voltage", rec_voltage, f"Freq: {freq_hz} Hz")
c4.metric(t["kpi_pue"], f"{pue_calc:.3f}", f"Cooling: {cooling_mode}")

st.divider()

# --- TABS ---
t1, t2, t3, t4 = st.tabs(["⚙️ Engineering", "🧪 Physics & Env", "❄️ Tri-Gen & LNG", "💰 Financials & ROI"])

with t1:
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Power Balance")
        if include_chp:
            comps = ["Critical IT", "DC Auxiliaries", "CHP Pumps", "Gen Parasitics", "Dist. Losses", "TOTAL GROSS"]
            vals = [p_it, p_it*dc_aux_pct, p_cooling_elec_new, p_parasitic_mw, p_loss_mw, p_gross_req]
        else:
            comps = ["Critical IT", f"Cooling & Aux (PUE {pue_calc:.2f})", "Gen Parasitics", "Dist. Losses", "TOTAL GROSS"]
            vals = [p_it, p_net_req - p_it, p_parasitic_mw, p_loss_mw, p_gross_req]
            
        df_bal = pd.DataFrame({"Component": comps, "Power (MW)": vals})
        st.dataframe(df_bal.style.format({"Power (MW)": "{:.2f}"}), use_container_width=True)
        
        if use_bess:
            st.success("✅ **BESS Enabled:** Fleet runs at High Efficiency (Base Load).")
            st.metric("BESS Sizing", f"{bess_power:.1f} MW / {bess_energy:.1f} MWh")
        else:
            st.warning("⚠️ **No BESS:** Fleet runs De-Rated (Spinning Reserve) to cover Step Loads.")
            
    with col2:
        st.subheader("Fleet Strategy (N+M+S)")
        st.write(f"**Target Avail:** {avail_req}% | **Model:** {selected_model}")
        st.write(f"**Site Capacity (Derated):** {unit_site_cap:.2f} MW")
        st.markdown("---")
        st.write(f"**N (Running):** {n_running}")
        st.write(f"**M (Maintenance @ {maint_outage_pct*100:.1f}%):** {n_maint}")
        st.write(f"**S (Standby/Reserve):** {n_reserve}")
        st.caption(f"Reserve logic: Max of FOR Buffer ({n_forced_buffer}) or Tier Requirement ({n_red_tier})")
        st.metric("Total Installed Fleet", f"{n_total} Units", f"{installed_cap:.1f} MW Total")

with t2:
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Site Physics")
        st.write(f"**Distance to Neighbor:** {disp_dist:.0f} {u_dist}")
        if noise_rec > noise_limit:
            st.error(f"🛑 **Noise Fail:** {noise_rec:.1f} dBA (Limit {noise_limit})")
        else:
            st.success(f"✅ **Noise OK:** {noise_rec:.1f} dBA")
            
        st.subheader("Footprint Estimates")
        df_foot = pd.DataFrame({
            "Zone": ["Generation Hall", "LNG Plant", "Cooling/CHP", "BESS Area", "Substation", "Total (+Roads)"],
            f"Area ({u_area_s})": [
                area_gen * (10.764 if is_imperial else 1),
                area_lng * (10.764 if is_imperial else 1),
                area_chp * (10.764 if is_imperial else 1),
                area_bess * (10.764 if is_imperial else 1),
                area_sub * (10.764 if is_imperial else 1),
                disp_area
            ]
        })
        st.dataframe(df_foot.style.format({f"Area ({u_area_s})": "{:,.0f}"}), use_container_width=True)
        st.metric("TOTAL LAND REQUIRED", f"{footprint_large_val:.2f} {footprint_unit}")

    with col2:
        st.subheader("Emissions & Urea")
        st.metric("NOx Emissions", f"{nox_tpy:,.0f} Ton/yr", f"Limit: {limit_nox_tpy}")
        if req_scr:
            st.info(f"SCR Required. Urea Consumption: {urea_vol_yr:,.0f} L/yr")
            tank_u = math.ceil((urea_vol_yr/365)*urea_days / 30000)
            st.write(f"**Urea Tanks ({urea_days} days):** {tank_u}x 30kL Tanks")
        else:
            st.success("No SCR Required.")

with t3:
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Cooling & Water")
        if include_chp:
            st.write(f"🔥 **Recoverable Heat:** {total_heat_rec_mw:.1f} MWt")
            st.write(f"❄️ **Cooling Generated:** {total_cooling_mw:.1f} MWc ({disp_cooling:,.0f})")
            st.metric("Cooling Coverage", f"{cooling_coverage_pct:.1f}%")
        else:
            st.info(f"Using **{cooling_method}** for cooling.")
        
        st.metric(f"Water Consumption (WUE)", f"{disp_water:,.0f} {u_water}")

    with col2:
        st.subheader("LNG Sovereignty")
        if include_lng:
            st.metric("Required LNG Storage", f"{disp_lng_store:,.0f} {u_vol}")
            st.write(f"**Tanks Required:** {num_tanks}x Vertical Cryo")
        else:
            st.warning("LNG Plant Disabled")

with t4:
    st.subheader("Financial Feasibility & ROI")
    
    # 1. Cost Index Editor
    st.markdown(f"👇 **Edit Indices to Adjust CAPEX:** (Base: **${gen_unit_cost:.0f}/kW**)")
    edited_capex = st.data_editor(
        df_capex_base, 
        column_config={
            "Default Index": st.column_config.NumberColumn("Cost Index (Ratio)", min_value=0.0, max_value=5.0, step=0.01),
            "Cost ($M)": st.column_config.NumberColumn("Calculated Cost", format="$%.2fM", disabled=True)
        },
        use_container_width=True
    )
    
    # Recalculate Total CAPEX
    final_capex_df = edited_capex.copy()
    final_capex_df["Cost ($M)"] = gen_cost_total * final_capex_df["Default Index"]
    total_capex = final_capex_df["Cost ($M)"].sum()
    
    # 2. LCOE Calculation
    mwh_year = p_net_req * 8760
    fuel_cost_year = (heat_input_mw * 3.41214 * 8760) * gas_price
    om_cost_year = mwh_year * om_var_price 
    
    crf = (wacc * (1 + wacc)**project_years) / ((1 + wacc)**project_years - 1)
    capex_annualized = (total_capex * 1e6) * crf
    
    total_annual_cost = fuel_cost_year + om_cost_year + capex_annualized
    lcoe = total_annual_cost / (mwh_year * 1000)
    
    # 3. ROI Calculation
    annual_grid_cost = mwh_year * 1000 * grid_price
    annual_prime_opex = fuel_cost_year + om_cost_year
    annual_savings = annual_grid_cost - annual_prime_opex
    
    if total_capex > 0:
        roi_simple = (annual_savings / (total_capex * 1e6)) * 100
        payback_years = (total_capex * 1e6) / annual_savings if annual_savings > 0 else 999
    else:
        roi_simple = 0
        payback_years = 0
    
    # Display Financials
    c_f1, c_f2, c_f3, c_f4 = st.columns(4)
    c_f1.metric("Total CAPEX", f"${total_capex:.2f} M")
    c_f2.metric("LCOE (Prime)", f"${lcoe:.4f} / kWh")
    c_f3.metric("Annual Savings", f"${annual_savings/1e6:.2f} M")
    c_f4.metric("ROI", f"{roi_simple:.1f}%", f"Payback: {payback_years:.1f} yrs")

    # Chart
    cost_data = pd.DataFrame({
        "Component": ["Fuel", "O&M (OPEX)", "CAPEX (Amortized)"],
        "$/kWh": [fuel_cost_year/(mwh_year*1000), om_cost_year/(mwh_year*1000), capex_annualized/(mwh_year*1000)]
    })
    
    fig_bar = px.bar(cost_data, x="Component", y="$/kWh", color="Component", 
                     title="LCOE Breakdown", text_auto='.4f')
    st.plotly_chart(fig_bar, use_container_width=True)

# --- FOOTER ---
st.markdown("---")
st.caption("CAT Prime Solution Designer | v2026.9")
