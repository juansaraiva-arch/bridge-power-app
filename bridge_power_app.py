import streamlit as st
import pandas as pd
import numpy as np
import math

# --- PAGE CONFIGURATION ---
st.set_page_config(page_title="Bridge Power Design Engine V9.3 (EN)", page_icon="🏭", layout="wide")

# ==============================================================================
# 0. GLOBAL SETTINGS (UNITS ONLY)
# ==============================================================================

with st.sidebar:
    st.header("Global Settings")
    # Language selector removed. Defaulting to English.
    unit_system = st.radio("System of Units", ["Metric (SI)", "Imperial (US)"])

is_imperial = "Imperial" in unit_system

# Unit Definitions
if is_imperial:
    u_temp = "°F"
    u_dist = "ft"
    u_area_s = "ft²"
    u_area_l = "Acres"
    u_vol = "gal"
    u_mass = "Short Tons" 
    u_alt = "ft"
else:
    u_temp = "°C"
    u_dist = "m"
    u_area_s = "m²"
    u_area_l = "Ha"
    u_vol = "L"
    u_mass = "Tonnes"
    u_alt = "masl"

# Single English Dictionary
t = {
    "title": "🏭 Bridge Power Design Engine V9.3",
    "subtitle": "**Total Engineering Suite.**\nCalculates: Availability (N+M+S), BESS, Urea Logistics, Footprint, and Economics.",
    "sb_1_title": "1. Data Center Profile",
    "dc_type_label": "Data Center Type",
    "dc_opts": ["AI Factory (Training/Inference)", "Standard Hyperscale"],
    "p_max": "Critical IT Load (MW)",
    "step_load": "Expected Step Load (%)",
    "voltage_label": "Connection Voltage to Data Center (kV)",
    "dist_loss": "Distribution Losses (%)",
    "aux_load": "Campus Auxiliaries (%)",
    "sb_2_title": "2. Generation Technology",
    "tech_label": "Prime Mover Technology",
    "tech_opts": ["RICE (Reciprocating Engine)", "Gas Turbine (Aero)"],
    "unit_iso": "ISO Prime Rating (MW)",
    "eff_iso": "ISO Thermal Efficiency (%)",
    "parasitic": "Gen. Parasitic Load (%)",
    "step_cap": "Step Load Capability (%)",
    "emi_title": "Native Emissions",
    "sb_3_title": "3. Site Physics & Neighbors",
    "cond_atm": "Atmospheric Conditions",
    "derate_method": "Derate Method",
    "derate_opts": ["Automatic", "Manual"],
    "temp": "Avg Max Temp",
    "alt": "Altitude",
    "mn": "Methane Number (Gas)",
    "manual_derate": "Manual Derate (%)",
    "urban_int": "Urban Integration (Neighbors)",
    "bldg_h": "Tallest Nearby Building",
    "dist_n": "Distance to Nearest Neighbor",
    "n_type": "Neighbor Type",
    "n_opts": ["Industrial", "Residential/Sensitive"],
    "source_noise": "Source Noise @ 1m (dBA)",
    "sb_4_title": "4. Regulatory Framework & Urea",
    "reg_zone": "Regulatory Zone",
    "reg_opts": ["USA - EPA Major", "USA - Virginia", "EU Standard", "LatAm / Unregulated"],
    "urea_days": "Urea Storage Autonomy (Days)",
    "sb_5_title": "5. Reliability & BESS",
    "avail_target": "Availability Target (%)",
    "use_bess": "Include BESS (Synthetic Inertia)",
    "maint": "Maintenance Unavailability (%)",
    "sb_6_title": "6. Economics",
    "fuel_p": "Gas Price ($/MMBtu)",
    
    "kpi_cap": "IT Capacity",
    "kpi_hr": "Net Heat Rate",
    "kpi_eff": "Real Eff",
    "kpi_area": "Total Footprint",
    "kpi_compl": "Compliance",
    "kpi_gen_count": "Generators Required",
    
    "tab_tech": "⚙️ Engineering",
    "tab_area": "🏗️ Physical Footprint",
    "tab_env": "🧪 Physics & Env",
    "tab_fin": "💰 Costs",
    
    "area_breakdown": "Area Breakdown (Ref. Workshop Case 2)",
    "area_gen": "Generation Blocks",
    "area_bess": "BESS System",
    "area_sub": "AIS/MV Substation",
    "area_gas": "Gas Station (ERM)",
    "area_scr": "Urea/SCR Farm",
    "area_roads": "Roads & Logistics (+20%)",
    
    "acoustic_model": "🔊 Acoustic Model",
    "source_noise_lbl": "Source Noise",
    "level_rec": "Receiver Level",
    "limit": "Limit",
    "noise_violation": "🛑 ACOUSTIC VIOLATION",
    "noise_sol": "Requires Barriers or Hospital-Grade Silencers",
    
    "disp_model": "💨 Dispersion & Emissions",
    "min_stack": "Min Stack Height",
    "warn_scr": "🛑 SCR REQUIRED (Urea)",
    "cons_urea": "Urea Consumption",
    "store_urea": "Urea Storage",
    "tank_urea": "Tanks Required",
    "log_trucks": "Truck Logistics",
    
    "bess_sizing": "BESS Sizing",
    "bess_pow": "BESS Power",
    "bess_ene": "BESS Energy (2h)",
    
    "status_ok": "✅ OK",
    "status_fail": "🛑 FAIL",
    "cost_fuel": "Fuel Cost",
    "warn_fuel": "High cost due to Spinning Reserve inefficiency.",
    "aux_impact": "Auxiliary Impact:"
}

st.title(t["title"])
st.markdown(t["subtitle"])

# ==============================================================================
# 1. INPUTS (SIDEBAR)
# ==============================================================================

with st.sidebar:
    st.header(t["sb_1_title"])
    # 1.1 DC Type Selection
    dc_type_sel = st.selectbox(t["dc_type_label"], t["dc_opts"])
    is_ai = "AI" in dc_type_sel or "IA" in dc_type_sel
    
    # Intelligent Defaults
    def_step_load = 40.0 if is_ai else 10.0
    def_use_bess = True if is_ai else False
    
    p_max = st.number_input(t["p_max"], 10.0, 1000.0, 100.0, step=10.0)
    step_load_req = st.number_input(t["step_load"], 0.0, 100.0, def_step_load)
    voltage_kv = st.number_input(t["voltage_label"], 0.4, 500.0, 34.5, step=0.5)
    dist_loss_pct = st.number_input(t["dist_loss"], 0.0, 10.0, 3.0) / 100
    aux_load_pct = st.number_input(t["aux_load"], 0.0, 15.0, 2.5) / 100

    st.divider()

    st.header(t["sb_2_title"])
    tech_type_sel = st.selectbox(t["tech_label"], t["tech_opts"])
    is_rice = "RICE" in tech_type_sel
    
    if is_rice:
        def_mw = 2.5
        def_eff = 46.0
        def_par = 2.5 
        def_step_cap = 65.0 
        def_maint = 5.0
        def_noise_source = 85.0 
        def_nox = 1.0
        def_co = 2.0
    else: 
        def_mw = 35.0
        def_eff = 38.0
        def_par = 0.5 
        def_step_cap = 20.0 
        def_maint = 3.0
        def_noise_source = 90.0
        def_nox = 0.6 
        def_co = 0.5 

    unit_size_iso = st.number_input(t["unit_iso"], 1.0, 100.0, def_mw)
    eff_gen_base = st.number_input(t["eff_iso"], 20.0, 65.0, def_eff)
    parasitic_pct = st.number_input(t["parasitic"], 0.0, 10.0, def_par) / 100
    gen_step_cap = st.number_input(t["step_cap"], 0.0, 100.0, def_step_cap)
    
    st.subheader(t["emi_title"])
    c_e1, c_e2 = st.columns(2)
    raw_nox = c_e1.number_input("NOx (g/bhp-hr)", 0.0, 10.0, def_nox)
    raw_co = c_e2.number_input("CO (g/bhp-hr)", 0.0, 10.0, def_co)

    st.divider()

    st.header(t["sb_3_title"])
    derate_method_sel = st.radio(t["derate_method"], t["derate_opts"])
    is_auto_derate = "Auto" in derate_method_sel
    
    derate_factor_calc = 1.0
    
    # Metric/Imperial Logic
    site_alt_m = 0
    site_temp_c = 30
    
    if is_auto_derate:
        if is_imperial:
            site_temp_f = st.slider(f"{t['temp']} ({u_temp})", 32, 122, 86)
            site_alt_ft = st.number_input(f"{t['alt']} ({u_alt})", 0, 13000, 328)
            site_temp_c = (site_temp_f - 32) * 5/9
            site_alt_m = site_alt_ft / 3.28084
        else:
            site_temp_c = st.slider(f"{t['temp']} ({u_temp})", 0, 50, 30)
            site_alt_m = st.number_input(f"{t['alt']} ({u_alt})", 0, 4000, 100)
            
        methane_number = st.number_input(t["mn"], 30, 100, 80)
        
        loss_temp = max(0, (site_temp_c - 25) * 0.01) if is_rice else max(0, (site_temp_c - 15) * 0.007)
        loss_alt = max(0, (site_alt_m - 100) * 0.0001) 
        loss_mn = max(0, (75 - methane_number) * 0.02) if is_rice else 0
        total_loss = min(0.5, loss_temp + loss_alt + loss_mn)
        derate_factor_calc = 1.0 - total_loss
    else:
        manual_derate = st.number_input(t["manual_derate"], 0.0, 50.0, 0.0)
        derate_factor_calc = 1.0 - (manual_derate / 100.0)

    unit_size_site = unit_size_iso * derate_factor_calc

    st.subheader(t["urban_int"])
    if is_imperial:
        nearby_bldg_ft = st.number_input(f"{t['bldg_h']} ({u_dist})", 15.0, 350.0, 40.0)
        dist_neighbor_ft = st.number_input(f"{t['dist_n']} ({u_dist})", 30.0, 6500.0, 328.0)
        nearby_building_h_m = nearby_bldg_ft / 3.28084
        dist_neighbor_m = dist_neighbor_ft / 3.28084
    else:
        nearby_building_h_m = st.number_input(f"{t['bldg_h']} ({u_dist})", 5.0, 100.0, 12.0)
        dist_neighbor_m = st.number_input(f"{t['dist_n']} ({u_dist})", 10.0, 2000.0, 100.0)

    neighbor_type_sel = st.selectbox(t["n_type"], t["n_opts"])
    noise_limit = 70.0 if "Industrial" in neighbor_type_sel else 55.0
    source_noise_dba = st.number_input(t["source_noise"], 60.0, 120.0, def_noise_source)

    st.divider()

    st.header(t["sb_4_title"])
    reg_zone = st.selectbox(t["reg_zone"], t["reg_opts"])
    if "EPA Major" in reg_zone: limit_nox_tpy = 250.0
    elif "Virginia" in reg_zone: limit_nox_tpy = 100.0
    elif "EU" in reg_zone: limit_nox_tpy = 150.0
    else: limit_nox_tpy = 9999.0
    
    # Urea Days Input
    urea_days = st.number_input(t["urea_days"], 1, 30, 7)

    st.header(t["sb_5_title"])
    # Availability Input
    avail_target = st.number_input(t["avail_target"], 90.00, 99.99999, 99.999, format="%.5f")
    use_bess = st.checkbox(t["use_bess"], value=def_use_bess)
    maint_unav = st.number_input(t["maint"], 0.0, 20.0, def_maint) / 100
    
    st.header(t["sb_6_title"])
    fuel_price = st.number_input(t["fuel_p"], 1.0, 20.0, 4.0)

# ==============================================================================
# 2. CALCULATION ENGINE
# ==============================================================================

# A. LOAD
p_aux_mw = p_max * aux_load_pct 
p_dist_loss = (p_max + p_aux_mw) * dist_loss_pct
p_net_gen_req = p_max + p_aux_mw + p_dist_loss 
p_gross_gen_req = p_net_gen_req / (1 - parasitic_pct)

# B. FLEET & AVAILABILITY
n_base = math.ceil(p_gross_gen_req / unit_size_site)
req_step_mw = p_max * (step_load_req / 100.0)

# Minimum Redundancy Calculation
min_redundancy = 1
if avail_target >= 99.99:
    min_redundancy = 2

n_spin = 0
bess_mw = 0
bess_mwh = 0

if use_bess:
    # BESS handles transient
    bess_mw = req_step_mw + unit_size_site 
    bess_mwh = bess_mw * 2 
    # Spin covers only N+1/N+2 for availability
    n_spin = min_redundancy
else:
    # Spin handles transient AND availability
    step_cap_mw_per_unit = unit_size_site * (gen_step_cap / 100.0)
    
    n_spin_step = 0
    current_n = n_base
    while True:
        total_mw = current_n * unit_size_site
        total_step_cap = total_mw * (gen_step_cap / 100.0)
        if total_step_cap >= req_step_mw:
            break
        current_n += 1
        n_spin_step += 1
    
    n_spin = max(n_spin_step, min_redundancy)
    if ((n_base + n_spin - min_redundancy) * unit_size_site) < p_gross_gen_req:
        n_spin += 1

n_online = n_base + n_spin
n_maint = math.ceil(n_online * maint_unav)
n_total = n_online + n_maint
installed_cap_site = n_total * unit_size_site

# C. PHYSICS & ENVIRONMENTAL
attenuation_geo = 20 * math.log10(dist_neighbor_m / 1.0)
noise_at_receiver_raw = source_noise_dba - attenuation_geo
num_sources_running = n_online
multi_source_add = 10 * math.log10(num_sources_running)
noise_at_receiver_total = noise_at_receiver_raw + multi_source_add
req_attenuation = max(0, noise_at_receiver_total - noise_limit)

min_stack_height_m = nearby_building_h_m * 1.5

total_bhp_online = (p_gross_gen_req * 1341) 
hours_yr = 8760
nox_tpy_raw = (raw_nox * total_bhp_online * hours_yr) / 907185
req_scr = nox_tpy_raw > limit_nox_tpy

# Urea Logistics
urea_l_yr = 0
urea_storage_l = 0
num_tanks = 0
trucks_yr = 0

if req_scr:
    urea_l_hr = p_gross_gen_req * 1.5 
    urea_l_yr = urea_l_hr * 8760
    urea_storage_l = (urea_l_yr / 365) * urea_days
    tank_size_l = 30000 
    num_tanks = math.ceil(urea_storage_l / tank_size_l)
    trucks_yr = math.ceil(urea_l_yr / 25000)

# D. AREAS
area_factor_gen = 140.0 if is_rice else 200.0 
area_gen = n_total * area_factor_gen
area_bess = bess_mwh * 25.0
area_sub = 5500.0 if voltage_kv >= 115 else 2500.0
area_gas = 800.0
area_scr = (400.0 + (num_tanks * 50)) if req_scr else 0.0
area_subtotal = area_gen + area_bess + area_sub + area_gas + area_scr
area_roads = area_subtotal * 0.20 
area_total_m2 = area_subtotal + area_roads
area_total_ha = area_total_m2 / 10000.0

# E. ECONOMICS
load_factor = (p_gross_gen_req / (n_online * unit_size_site)) * 100.0
lf_dec = load_factor / 100.0
if is_rice:
    eff_curve_factor = 1.0 - (0.6 * (1.0 - lf_dec)**3) if lf_dec < 1.0 else 1.0
else:
    eff_curve_factor = 0.9 + (0.1 * lf_dec) - (0.4 * (1.0 - lf_dec)**2)
eff_curve_factor = max(0.5, eff_curve_factor) 
real_gen_eff = eff_gen_base * eff_curve_factor

fuel_mw_th = p_gross_gen_req / (real_gen_eff / 100.0)
fuel_btu_hr = fuel_mw_th * 3412142
useful_output_mw = p_max + p_aux_mw
net_system_heat_rate = fuel_btu_hr / (useful_output_mw * 1000)
fuel_cost_hr = (fuel_btu_hr / 1e6) * fuel_price
cost_kwh = fuel_cost_hr / (p_max * 1000)

# ==============================================================================
# 3. DASHBOARD (OUTPUT CONVERSION)
# ==============================================================================

if is_imperial:
    disp_area_l = area_total_ha * 2.47105 # Acres
    disp_area_s = area_total_m2 * 10.7639 # Sq Ft
    disp_area_gen = area_gen * 10.7639
    disp_area_bess = area_bess * 10.7639
    disp_area_sub = area_sub * 10.7639
    disp_area_gas = area_gas * 10.7639
    disp_area_scr = area_scr * 10.7639
    disp_area_rds = area_roads * 10.7639
    disp_area_tot = area_total_m2 * 10.7639
    
    disp_urea_yr = urea_l_yr * 0.264172 # Gallons
    disp_urea_store = urea_storage_l * 0.264172
    disp_stack = min_stack_height_m * 3.28084 # Feet
    disp_dist_n = dist_neighbor_m * 3.28084 # Feet
    
    disp_mass = nox_tpy_raw * 1.10231 # Short Tons
    disp_limit = limit_nox_tpy * 1.10231
else:
    disp_area_l = area_total_ha
    disp_area_s = area_total_m2
    disp_area_gen = area_gen
    disp_area_bess = area_bess
    disp_area_sub = area_sub
    disp_area_gas = area_gas
    disp_area_scr = area_scr
    disp_area_rds = area_roads
    disp_area_tot = area_total_m2
    
    disp_urea_yr = urea_l_yr
    disp_urea_store = urea_storage_l
    disp_stack = min_stack_height_m
    disp_dist_n = dist_neighbor_m
    
    disp_mass = nox_tpy_raw
    disp_limit = limit_nox_tpy

# Render
c1, c2, c3, c4 = st.columns(4)
c1.metric(t["kpi_cap"], f"{p_max} MW", f"Volt: {voltage_kv} kV")
c2.metric(t["kpi_gen_count"], f"{n_total} Units", f"N+M+S: {n_base}+{n_maint}+{n_spin}")
c3.metric(t["kpi_area"], f"{disp_area_l:.2f} {u_area_l}", f"{disp_area_s:,.0f} {u_area_s}")

status_txt = t["status_ok"]
if req_scr or req_attenuation > 0: status_txt = t["status_fail"]
c4.metric(t["kpi_compl"], status_txt, f"NOx: {disp_mass:.0f} {u_mass}")

st.divider()

t_tech, t_area, t_env, t_fin = st.tabs([t["tab_tech"], t["tab_area"], t["tab_env"], t["tab_fin"]])

with t_tech:
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Power Balance")
        df_load = pd.DataFrame({
            "Item": ["Critical IT", f"Auxiliaries ({aux_load_pct*100}%)", "Dist. Losses", "Gen. Parasitics", "TOTAL GROSS REQ."],
            "MW": [p_max, p_aux_mw, p_dist_loss, (p_gross_gen_req - p_net_gen_req), p_gross_gen_req]
        })
        st.dataframe(df_load.style.format({"MW": "{:.2f}"}), use_container_width=True)
    with col2:
        st.subheader("Fleet & Strategy")
        st.write(f"**Target Availability:** {avail_target}%")
        st.write(f"**Min Redundancy:** N+{min_redundancy}")
        st.markdown("---")
        st.write(f"Base Units (N): {n_base}")
        st.write(f"Reserve (S): {n_spin}")
        st.write(f"Maintenance (M): {n_maint}")
        st.metric("Total Units", n_total)
        
        if use_bess:
            st.markdown("### " + t["bess_sizing"])
            st.write(f"**{t['bess_pow']}:** {bess_mw:.2f} MW")
            st.write(f"**{t['bess_ene']}:** {bess_mwh:.2f} MWh")

with t_area:
    st.subheader(t["area_breakdown"])
    df_area = pd.DataFrame({
        "Component": [t["area_gen"], t["area_bess"], t["area_sub"], t["area_gas"], t["area_scr"], t["area_roads"], "TOTAL"],
        f"Area ({u_area_s})": [disp_area_gen, disp_area_bess, disp_area_sub, disp_area_gas, disp_area_scr, disp_area_rds, disp_area_tot]
    })
    st.dataframe(df_area.style.format({f"Area ({u_area_s})": "{:,.0f}"}), use_container_width=True)

with t_env:
    col1, col2 = st.columns(2)
    with col1:
        st.subheader(t["acoustic_model"])
        src_noise = source_noise_dba + multi_source_add
        st.write(f"**{t['source_noise_lbl']} (x{num_sources_running}):** {src_noise:.1f} dBA")
        st.write(f"**{t['level_rec']} ({disp_dist_n:.0f} {u_dist}):** {noise_at_receiver_total:.1f} dBA")
        st.write(f"**{t['limit']}:** {noise_limit} dBA")
        if req_attenuation > 0:
            st.error(f"{t['noise_violation']} (-{req_attenuation:.1f} dB)")
            st.info(t["noise_sol"])
        else:
            st.success(t["status_ok"])
    with col2:
        st.subheader(t["disp_model"])
        st.metric(f"NOx ({u_mass}/yr)", f"{disp_mass:,.0f}")
        st.metric("Zone Limit", f"{disp_limit:,.0f}")
        
        if req_scr: 
            st.warning(t["warn_scr"])
            st.write(f"**{t['cons_urea']}:** {disp_urea_yr:,.0f} {u_vol}/yr")
            st.markdown(f"**{t['store_urea']} ({urea_days} days):** {disp_urea_store:,.0f} {u_vol}")
            st.write(f"**{t['tank_urea']}:** {num_tanks}x 30kL Tanks")
            st.write(f"**{t['log_trucks']}:** {trucks_yr} Trucks/yr")
        
        st.markdown("---")
        st.metric(t["min_stack"], f"{disp_stack:.1f} {u_dist}")

with t_fin:
    st.metric(t["cost_fuel"], f"${cost_kwh:.4f} / kWh IT")
    st.metric(t["kpi_hr"], f"{net_system_heat_rate:,.0f} BTU/kWh")
    if not use_bess and is_ai: st.warning(t["warn_fuel"])

# --- FOOTER ---
st.markdown("---")
st.caption("Bridge Power Engine V9.3 (English Version)")
