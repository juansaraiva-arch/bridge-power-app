import streamlit as st
import pandas as pd
import numpy as np
import math
import plotly.express as px

# --- PAGE CONFIGURATION ---
st.set_page_config(page_title="CAT Prime Power Calculator", page_icon="⚡", layout="wide")

# ==============================================================================
# 0. GLOBAL SETTINGS
# ==============================================================================

with st.sidebar:
    st.image("https://img.icons8.com/color/96/generator.png", width=60)
    st.header("Global Settings")
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
    u_power = "MW"
    u_energy = "MWh"
    u_therm = "MMBtu"
    u_water = "gal/day"
else:
    u_temp = "°C"
    u_dist = "m"
    u_area_s = "m²"
    u_area_l = "Ha"
    u_vol = "m³"
    u_mass = "Tonnes"
    u_power = "MW"
    u_energy = "MWh"
    u_therm = "GJ"
    u_water = "m³/day"

# Dictionary
t = {
    "title": "⚡ CAT Prime Power Calculator",
    "subtitle": "**Sovereign Energy Solutions.**\nComprehensive modeling for Off-Grid Microgrids, Tri-Generation, and Financial Feasibility.",
    "sb_1": "1. Data Center Profile",
    "sb_2": "2. Generation Technology",
    "sb_3": "3. Site & Neighbors",
    "sb_4": "4. Strategy (BESS & LNG)",
    "sb_5": "5. Cooling & Tri-Gen",
    "sb_6": "6. Regulatory & Urea",
    "sb_7": "7. Economics",
    "kpi_net": "Net Capacity",
    "kpi_pue": "Projected PUE"
}

st.title(t["title"])
st.markdown(t["subtitle"])

# ==============================================================================
# 1. INPUTS (SIDEBAR)
# ==============================================================================

with st.sidebar:
    # --- 1. DATA CENTER PROFILE ---
    st.header(t["sb_1"])
    dc_type = st.selectbox("Data Center Type", ["AI Factory (Training)", "Hyperscale Standard"])
    is_ai = "AI" in dc_type
    
    # Defaults Dynamic
    def_step_load = 40.0 if is_ai else 15.0
    def_use_bess = True if is_ai else False
    
    p_it = st.number_input("Critical IT Load (MW)", 10.0, 500.0, 100.0, step=10.0)
    
    # NUEVO: Disponibilidad Requerida
    avail_req = st.number_input("Required Availability (%)", 90.0, 99.99999, 99.999, format="%.5f")
    
    step_load_req = st.number_input("Expected Step Load (%)", 0.0, 100.0, def_step_load)
    dc_aux_pct = st.number_input("DC Building Auxiliaries (%)", 0.0, 20.0, 5.0) / 100.0

    st.divider()

    # --- 2. GENERATION TECH ---
    st.header(t["sb_2"])
    tech_type = st.selectbox("Engine Technology", ["RICE Medium Speed (720 rpm)", "Gas Turbine"])
    
    # Defaults by Tech
    if "RICE" in tech_type:
        def_mw = 10.5
        def_eff = 48.5
        def_step_cap = 65.0
        def_maint_out = 5.0 
        def_gen_par = 2.5 
        def_nox = 1.0
    else:
        def_mw = 35.0
        def_eff = 38.0
        def_step_cap = 20.0
        def_maint_out = 3.0
        def_gen_par = 0.5
        def_nox = 0.6

    unit_size_iso = st.number_input("Unit Prime Rating (ISO MW)", 1.0, 100.0, def_mw)
    elec_eff_iso = st.number_input("Electrical Efficiency (ISO %)", 20.0, 60.0, def_eff) / 100.0
    step_load_cap = st.number_input("Unit Step Load Capability (%)", 0.0, 100.0, def_step_cap)
    
    c_g1, c_g2 = st.columns(2)
    maint_outage_pct = c_g1.number_input("Maint. Outage (%)", 0.0, 20.0, def_maint_out) / 100.0
    gen_parasitic_pct = c_g2.number_input("Gen. Auxiliaries (%)", 0.0, 10.0, def_gen_par) / 100.0

    c_e1, c_e2 = st.columns(2)
    raw_nox = c_e1.number_input("Native NOx (g/bhp-hr)", 0.0, 10.0, def_nox)
    raw_co = c_e2.number_input("Native CO (g/bhp-hr)", 0.0, 10.0, 0.5)

    st.divider()

    # --- 3. SITE & NEIGHBORS ---
    st.header(t["sb_3"])
    is_auto_derate = st.checkbox("Auto-Derate Calculation", value=True)
    derate_factor_calc = 1.0
    
    if is_auto_derate:
        if is_imperial:
            site_temp_f = st.slider(f"Max Ambient Temp ({u_temp})", 32, 122, 95)
            site_alt_ft = st.number_input(f"Altitude ({u_dist})", 0, 13000, 328)
            site_temp_c = (site_temp_f - 32) * 5/9
            site_alt_m = site_alt_ft / 3.28084
        else:
            site_temp_c = st.slider(f"Max Ambient Temp ({u_temp})", 0, 50, 35)
            site_alt_m = st.number_input(f"Altitude ({u_dist})", 0, 4000, 100)
            
        loss_temp = max(0, (site_temp_c - 25) * 0.01) if "RICE" in tech_type else max(0, (site_temp_c - 15) * 0.007)
        loss_alt = max(0, (site_alt_m - 100) * 0.0001) 
        derate_factor_calc = 1.0 - (loss_temp + loss_alt)
    else:
        manual_derate = st.number_input("Manual Derate (%)", 0.0, 50.0, 3.0)
        derate_factor_calc = 1.0 - (manual_derate / 100.0)

    dist_neighbor_m = st.number_input(f"Dist. to Neighbor ({u_dist})", 10.0, 5000.0, 100.0)
    if is_imperial: dist_neighbor_m = dist_neighbor_m / 3.28084
    
    noise_zone = st.selectbox("Zone Type", ["Industrial", "Residential"])
    noise_limit = 70.0 if noise_zone == "Industrial" else 55.0
    source_noise_dba = st.number_input("Source Noise @ 1m (dBA)", 60.0, 120.0, 85.0 if "RICE" in tech_type else 90.0)

    st.divider()

    # --- 4. STRATEGY (BESS & LNG) ---
    st.header(t["sb_4"])
    use_bess = st.checkbox("Include BESS (Synthetic Inertia)", value=def_use_bess)
    include_lng = st.checkbox("Include LNG Plant (Sovereignty)", value=True)
    
    autonomy_days = 0
    if include_lng:
        autonomy_days = st.number_input("LNG Autonomy Target (Days)", 1, 60, 30)

    st.divider()

    # --- 5. COOLING & TRI-GEN ---
    st.header(t["sb_5"])
    include_chp = st.checkbox("Include Tri-Generation (CHP)", value=True)
    
    cooling_method = "Tri-Gen"
    if include_chp:
        st.caption("Cooling via Waste Heat (Absorption).")
        cop_double = st.number_input("COP Double Effect (Exhaust)", 0.5, 2.0, 1.2)
        cop_single = st.number_input("COP Single Effect (Jacket)", 0.4, 1.5, 0.7)
        pue_input = 0.0 
    else:
        cool_idx = 0 if is_ai else 1
        cooling_method = st.selectbox("Cooling Technology", 
                                      ["Water Cooled Systems (Towers)", "Air Cooled Chillers"], 
                                      index=cool_idx)
        def_pue_val = 1.25 if "Water" in cooling_method else 1.45
        st.caption(f"Default PUE for {cooling_method}: {def_pue_val}")
        pue_input = st.number_input("Expected PUE", 1.05, 2.0, def_pue_val)

    st.divider()
    
    # --- 6. REGULATORY ---
    st.header(t["sb_6"])
    reg_zone = st.selectbox("Regulatory Zone", ["USA - EPA Major", "EU Standard", "LatAm / No-Reg"])
    limit_nox_tpy = 250.0 if "EPA" in reg_zone else (150.0 if "EU" in reg_zone else 9999.0)
    urea_days = st.number_input("Urea Storage (Days)", 1, 30, 7)

    st.divider()

    # --- 7. ECONOMICS ---
    st.header(t["sb_7"])
    gas_price = st.number_input("Delivered Gas Price ($/MMBtu)", 1.0, 20.0, 6.5)
    grid_price = st.number_input("Comparison Grid Price ($/kWh)", 0.05, 0.50, 0.15)
    project_years = st.number_input("Project Lifespan (Years)", 10, 30, 20)
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

loss_trans_pct = 0.012 
loss_dist_pct = 0.008  
total_plant_loss_pct = gen_parasitic_pct + loss_trans_pct + loss_dist_pct

p_gross_req = p_net_req / (1 - total_plant_loss_pct)
p_parasitic = p_gross_req - p_net_req

current_13_8 = (p_gross_req * 1000) / (math.sqrt(3) * 13.8 * 0.8)
rec_voltage = "34.5 kV" if current_13_8 > 3000 else "13.8 kV"

# --- B. FLEET & EFFICIENCY ---
unit_site_cap = unit_size_iso * derate_factor_calc

# Strategy Logic (Availability & BESS)
if use_bess:
    target_load_factor = 0.95 
    n_running = math.ceil(p_gross_req / (unit_site_cap * target_load_factor))
    
    # Redundancy based on Availability
    # If >= 99.9999% (Six Nines), bump reserve to 2. Otherwise 1.
    n_reserve = 2 if avail_req >= 99.9999 else 1
    
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
    # Even without BESS, high avail might demand more reserve
    n_reserve = 2 if avail_req >= 99.9999 else 1
    bess_power = 0
    bess_energy = 0

n_maint = math.ceil(n_running * maint_outage_pct) + 1
n_total = n_running + n_reserve + n_maint
installed_cap = n_total * unit_site_cap

hr_penalty = 1.05 if not use_bess else 1.0
real_elec_eff = elec_eff_iso / hr_penalty
net_eff = real_elec_eff * (1 - total_plant_loss_pct)
heat_rate_net_kj = 3600 / net_eff
heat_rate_net_btu = heat_rate_net_kj * 0.9478

# --- C. THERMAL & CHP CALCS ---
heat_input_mw = p_gross_req / real_elec_eff
heat_exhaust_mw = heat_input_mw * 0.28
heat_jacket_mw = heat_input_mw * 0.18
total_heat_rec_mw = heat_exhaust_mw + heat_jacket_mw

if include_chp:
    cooling_exhaust = heat_exhaust_mw * cop_double
    cooling_jacket = heat_jacket_mw * cop_single
    total_cooling_mw = cooling_exhaust + cooling_jacket
    
    cooling_demand_mw = p_it * 1.05 
    cooling_coverage_pct = min(100.0, (total_cooling_mw / cooling_demand_mw) * 100)
    water_cons_m3_hr = total_cooling_mw * 1.8 
else:
    total_cooling_mw = 0
    cooling_coverage_pct = 0
    if "Water" in cooling_method:
        water_cons_m3_hr = (p_net_req - p_it) * 1.5
    else:
        water_cons_m3_hr = 0.0

water_cons_daily_m3 = water_cons_m3_hr * 24

# --- D. LNG & FUEL ---
total_gas_energy_day_mmbtu = (p_gross_req * 24 * 3412.14) / (real_elec_eff * 1e6)
gas_vol_day_m3 = total_gas_energy_day_mmbtu * 28.26

lng_vol_day_m3 = 0
required_storage_m3 = 0
num_tanks = 0
trucks_per_day = 0

if include_lng:
    lng_vol_day_m3 = gas_vol_day_m3 / 600
    required_storage_m3 = lng_vol_day_m3 * autonomy_days
    tank_vol_sel = 3000
    num_tanks = math.ceil(required_storage_m3 / tank_vol_sel)
    trucks_per_day = math.ceil(lng_vol_day_m3 / 50)

# --- E. EMISSIONS ---
attenuation = 20 * math.log10(dist_neighbor_m)
noise_rec = source_noise_dba + (10 * math.log10(n_running)) - attenuation

total_bhp = p_gross_req * 1341
nox_tpy = (raw_nox * total_bhp * 8760) / 907185
req_scr = nox_tpy > limit_nox_tpy

urea_vol_yr = 0
if req_scr:
    urea_vol_yr = p_gross_req * 1.5 * 8760 
    
# --- F. FINANCIALS (CAPEX & LCOE) ---
capex_data = {
    "Item": [
        "Generation Units (Mechanical)", 
        "Electrical (GSU/Substation)", 
        "Civil & Engineering",
        "Tri-Gen (CHP) Plant",
        "BESS System",
        "LNG Infrastructure"
    ],
    "Cost ($M)": [
        155.0 * (installed_cap/147.0),
        19.0 * (installed_cap/147.0),
        10.0 * (installed_cap/147.0),
        18.0 * (installed_cap/147.0) if include_chp else 0,
        30.0 * (bess_power/50.0) if use_bess else 0,
        22.0 * (num_tanks/10.0) if include_lng else 0
    ]
}
df_capex = pd.DataFrame(capex_data)

# --- G. FOOTPRINT ---
area_gen = n_total * 200 
area_lng = num_tanks * 600 
area_chp = total_cooling_mw * 20 if include_chp else (p_net_req * 10) 
area_bess = bess_power * 30 
area_sub = 2500
total_area_m2 = (area_gen + area_lng + area_chp + area_bess + area_sub) * 1.2

# ==============================================================================
# 3. DASHBOARD OUTPUT
# ==============================================================================

if is_imperial:
    disp_lng_store = required_storage_m3 * 264.172 
    u_vol_lng = "gal"
    disp_cooling = total_cooling_mw * 284.345 
    u_cool = "TR"
    disp_water = water_cons_daily_m3 * 264.172 
    disp_area = total_area_m2 * 10.764 
    disp_dist = dist_neighbor_m * 3.28
else:
    disp_lng_store = required_storage_m3
    u_vol_lng = "m³"
    disp_cooling = total_cooling_mw
    u_cool = "MWt"
    disp_water = water_cons_daily_m3
    disp_area = total_area_m2
    disp_dist = dist_neighbor_m

# --- TOP KPIS ---
c1, c2, c3, c4 = st.columns(4)
c1.metric(t["kpi_net"], f"{p_net_req:.1f} MW", f"Gross: {p_gross_req:.1f} MW")
c2.metric("Heat Rate", f"{heat_rate_net_btu:,.0f} Btu/kWh", f"Eff: {net_eff*100:.1f}%")
c3.metric("Rec. Voltage", rec_voltage, f"Current: {current_13_8:,.0f} A")
c4.metric(t["kpi_pue"], f"{pue_calc:.3f}", f"Cooling: {cooling_mode}")

st.divider()

# --- TABS ---
t1, t2, t3, t4 = st.tabs(["⚙️ Engineering", "🧪 Physics & Env", "❄️ Tri-Gen & LNG", "💰 Financials"])

with t1:
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Power Balance")
        
        # Dynamic Table based on mode
        if include_chp:
            comps = ["Critical IT", "DC Auxiliaries", "CHP Pumps", "Plant Parasitics", "TOTAL GROSS"]
            vals = [p_it, p_it*dc_aux_pct, p_cooling_elec_new, p_parasitic, p_gross_req]
        else:
            comps = ["Critical IT", f"Cooling & Aux (PUE {pue_calc:.2f})", "Plant Parasitics", "TOTAL GROSS"]
            vals = [p_it, p_net_req - p_it, p_parasitic, p_gross_req]
            
        df_bal = pd.DataFrame({"Component": comps, "Power (MW)": vals})
        st.dataframe(df_bal.style.format({"Power (MW)": "{:.2f}"}), use_container_width=True)
        
        if use_bess:
            st.success("✅ **BESS Enabled:** Fleet runs at High Efficiency (Base Load).")
            st.metric("BESS Sizing", f"{bess_power:.1f} MW / {bess_energy:.1f} MWh")
        else:
            st.warning("⚠️ **No BESS:** Fleet runs De-Rated (Spinning Reserve) to cover Step Loads.")
            
    with col2:
        st.subheader("Fleet Strategy")
        st.write(f"**Target Avail:** {avail_req}%")
        st.write(f"**Technology:** {tech_type} ({unit_size_iso} MW)")
        st.write(f"**Site Capacity:** {unit_site_cap:.2f} MW")
        st.markdown("---")
        st.write(f"**Running (N):** {n_running}")
        st.write(f"**Reserve (+{n_reserve}):** {n_reserve}")
        st.write(f"**Maintenance (+{n_maint}):** {n_maint}")
        st.metric("Total Installed Fleet", f"{n_total} Units", f"{installed_cap:.1f} MW Total")

with t2:
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Site Physics")
        st.write(f"**Distance to Neighbor:** {disp_dist:.0f} {u_dist}")
        st.write(f"**Attenuation:** -{attenuation:.1f} dB")
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
            st.write(f"❄️ **Cooling Generated:** {total_cooling_mw:.1f} MWc ({disp_cooling:,.0f} {u_cool})")
            st.metric("Cooling Coverage", f"{cooling_coverage_pct:.1f}%")
        else:
            st.info(f"Using **{cooling_method}** for cooling.")
            st.write(f"**Electrical Cooling Load:** {cooling_load_elec:.1f} MW")
        
        st.metric(f"Water Consumption (WUE)", f"{disp_water:,.0f} {u_water}")

    with col2:
        st.subheader("LNG Sovereignty")
        if include_lng:
            st.write(f"**Target Autonomy:** {autonomy_days} Days")
            st.write(f"**Daily Consumption:** {gas_vol_day_m3:,.0f} m³ Gas")
            st.metric("Required LNG Storage", f"{disp_lng_store:,.0f} {u_vol_lng}")
            st.write(f"**Tanks Required:** {num_tanks}x Vertical Cryo")
            st.error(f"🚨 **Emergency Logistics:** {trucks_per_day} Trucks/Day if pipeline fails.")
        else:
            st.warning("LNG Plant Disabled (100% Pipeline Dependent)")

with t4:
    st.subheader("Financial Feasibility (Editable)")
    
    # Editable CAPEX
    st.markdown("👇 **Edit CAPEX values below to update LCOE:**")
    edited_df = st.data_editor(df_capex, num_rows="dynamic", use_container_width=True)
    
    # Recalculate based on edits
    total_capex = edited_df["Cost ($M)"].sum()
    
    # LCOE Calculation
    mwh_year = p_net_req * 8760
    fuel_cost_year = (heat_input_mw * 3.41214 * 8760) * gas_price
    om_cost_year = mwh_year * 12.0
    
    crf = (wacc * (1 + wacc)**project_years) / ((1 + wacc)**project_years - 1)
    capex_annualized = (total_capex * 1e6) * crf
    
    total_annual_cost = fuel_cost_year + om_cost_year + capex_annualized
    lcoe = total_annual_cost / (mwh_year * 1000)
    
    # Comparison
    savings = (grid_price - lcoe) * mwh_year * 1000
    
    c_f1, c_f2, c_f3 = st.columns(3)
    c_f1.metric("Total CAPEX", f"${total_capex:.1f} M")
    c_f2.metric("LCOE (Prime)", f"${lcoe:.4f} / kWh")
    c_f3.metric("Annual Savings vs Grid", f"${savings/1e6:.1f} M", f"Grid: ${grid_price:.3f}")

    # Sensitivity Analysis
    st.markdown("### 📊 Sensitivity: Gas Price Impact")
    gas_range = np.linspace(2, 12, 20)
    lcoe_range = []
    for g in gas_range:
        fc = (heat_input_mw * 3.41214 * 8760) * g
        tc = fc + om_cost_year + capex_annualized
        lcoe_range.append(tc / (mwh_year * 1000))
        
    df_sens = pd.DataFrame({"Gas Price ($/MMBtu)": gas_range, "LCOE ($/kWh)": lcoe_range})
    fig = px.line(df_sens, x="Gas Price ($/MMBtu)", y="LCOE ($/kWh)", markers=True, title="LCOE Sensitivity to Fuel Price")
    fig.add_hline(y=grid_price, line_dash="dash", annotation_text="Grid Price", line_color="red")
    st.plotly_chart(fig, use_container_width=True)

# --- FOOTER ---
st.markdown("---")
st.caption("CAT Prime Power Calculator | v2026.4")