import streamlit as st
import pandas as pd
import numpy as np
import math
import plotly.express as px
import plotly.graph_objects as go

# --- PAGE CONFIGURATION ---
st.set_page_config(page_title="CAT Bridge Solutions Designer v37", page_icon="🌉", layout="wide")

# ==============================================================================
# 0. HYBRID DATA LIBRARY
# ==============================================================================

bridge_rental_library = {
    "XGC1900": {
        "description": "Gas Rental Unit (G3516H) - High Efficiency",
        "fuels": ["Natural Gas"],
        "type": "High Speed",
        "iso_rating_mw": {60: 1.9, 50: 1.9}, 
        "electrical_efficiency": 0.392,
        "heat_rate_lhv": 8780,
        "step_load_pct": 25.0, 
        "emissions_nox": 0.5,
        "default_maint": 5.0, "default_for": 2.0,
        "est_asset_value_kw": 850.0, "est_mob_kw": 80.0,
        "reactance_xd_2": 0.14
    },
    "TM2500": {
        "description": "Mobile Gas Turbine (34 MW) - Aero",
        "fuels": ["Natural Gas", "Diesel"], 
        "type": "Gas Turbine",
        "iso_rating_mw": {60: 34.0, 50: 34.0}, 
        "electrical_efficiency": 0.370,
        "heat_rate_lhv": 9220,
        "step_load_pct": 15.0, 
        "emissions_nox": 0.6,
        "default_maint": 3.0, "default_for": 1.5,
        "est_asset_value_kw": 900.0, "est_mob_kw": 120.0,
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
        "default_maint": 2.0, "default_for": 1.0,
        "est_asset_value_kw": 950.0, "est_mob_kw": 60.0,
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
        "default_maint": 2.0, "default_for": 1.0,
        "est_asset_value_kw": 900.0, "est_mob_kw": 70.0,
        "reactance_xd_2": 0.18
    },
    "XQ2280": {
        "description": "Diesel Power Module (3516C) - Tier 4 Final",
        "fuels": ["Diesel"],
        "type": "High Speed",
        "iso_rating_mw": {60: 1.825, 50: 1.6}, 
        "electrical_efficiency": 0.380, 
        "heat_rate_lhv": 9000,
        "step_load_pct": 80.0, 
        "emissions_nox": 0.6,
        "default_maint": 4.0, "default_for": 1.0,
        "est_asset_value_kw": 600.0, "est_mob_kw": 50.0,
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
        "default_maint": 4.0, "default_for": 1.5,
        "est_asset_value_kw": 550.0, "est_mob_kw": 50.0,
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
        "default_maint": 3.0, "default_for": 1.0,
        "est_asset_value_kw": 500.0, "est_mob_kw": 40.0,
        "reactance_xd_2": 0.12
    }
}

# ==============================================================================
# 1. INPUTS (SIDEBAR)
# ==============================================================================

with st.sidebar:
    st.image("https://img.icons8.com/color/96/bridge.png", width=60)
    st.header("Global Settings")
    c_glob1, c_glob2 = st.columns(2)
    unit_system = c_glob1.radio("Units", ["Metric (SI)", "Imperial (US)"])
    freq_hz = c_glob2.radio("Frequency", [60, 50])

    is_imperial = "Imperial" in unit_system
    is_50hz = freq_hz == 50

    # 1. Profile
    st.header("1. Data Center Profile")
    dc_type_sel = st.selectbox("Data Center Type", ["AI Factory (Training)", "Hyperscale Standard"])
    is_ai = "AI" in dc_type_sel
    
    def_step_load = 50.0 if is_ai else 25.0
    def_use_bess = True if is_ai else False
    
    p_it = st.number_input("Critical IT Load (MW)", 1.0, 1000.0, 50.0, step=10.0)
    pue_input = st.number_input("Design PUE", 1.0, 2.0, 1.35, step=0.01)
    
    avail_req = st.number_input("Availability Target (%)", 90.0, 99.99999, 99.99, format="%.5f")
    step_load_req = st.number_input("Block Load / Step Req (%)", 0.0, 100.0, def_step_load, help="% of IT load that hits instantly")
    dist_loss_pct = st.number_input("Distribution Losses (%)", 0.0, 10.0, 1.0) / 100.0

    st.divider()

    # 2. Tech
    st.header("2. Technology & Fuel")
    fuel_type_sel = st.selectbox("Fuel Source", ["Natural Gas", "Diesel", "Propane"])
    
    avail_models = [k for k, v in bridge_rental_library.items() if fuel_type_sel in v['fuels']]
    selected_model = st.selectbox("Select Bridge Unit", avail_models)
    eng_data = bridge_rental_library[selected_model]
    st.info(f"**{eng_data['description']}**")

    # Storage Vars Initialization
    virtual_pipe_mode = "None"
    methane_number = 80
    tank_unit_cap = 1.0 
    tank_mob_cost = 0.0
    tank_area_unit = 0.0
    truck_capacity = 8000.0
    storage_days = 0

    if fuel_type_sel == "Natural Gas":
        methane_number = st.number_input("Methane Number (MN)", 30, 100, 80)
        gas_source = st.radio("Supply Method", ["Pipeline", "Virtual Pipeline (LNG)", "Virtual Pipeline (CNG)"])
        
        if gas_source == "Pipeline":
            virtual_pipe_mode = "Pipeline"
        elif "LNG" in gas_source:
            virtual_pipe_mode = "LNG"
            st.markdown("🔹 **LNG Storage**")
            storage_days = st.number_input("Autonomy (Days)", 1, 30, 5)
            c1, c2 = st.columns(2)
            tank_unit_cap = c1.number_input("ISO Tank Cap (Gal)", 1000, 20000, 10000)
            tank_mob_cost = c2.number_input("Mob Cost/Tank ($)", 0, 50000, 5000)
            truck_capacity = st.number_input("Truck Delivery Vol (Gal)", 1000, 15000, 10000)
            tank_area_unit = 40.0
        elif "CNG" in gas_source:
            virtual_pipe_mode = "CNG"
            st.markdown("🔹 **CNG Storage**")
            storage_days = st.number_input("Autonomy (Days)", 1, 30, 1)
            c1, c2 = st.columns(2)
            tank_unit_cap = c1.number_input("Trailer Cap (scf)", 50000, 1000000, 350000)
            tank_mob_cost = c2.number_input("Mob Cost/Trailer ($)", 0, 50000, 2000)
            truck_capacity = tank_unit_cap 
            tank_area_unit = 60.0
            
    elif fuel_type_sel == "Diesel":
        virtual_pipe_mode = "Diesel"
        st.markdown("🔹 **Diesel Storage**")
        storage_days = st.number_input("Autonomy (Days)", 1, 30, 3)
        c1, c2 = st.columns(2)
        tank_unit_cap = c1.number_input("Frac Tank Cap (Gal)", 1000, 50000, 20000)
        tank_mob_cost = c2.number_input("Mob Cost/Tank ($)", 0, 50000, 2500)
        truck_capacity = st.number_input("Truck Delivery Vol (Gal)", 1000, 15000, 8000)
        tank_area_unit = 50.0
        
    elif fuel_type_sel == "Propane":
        virtual_pipe_mode = "Propane"
        st.markdown("🔹 **LPG Storage**")
        storage_days = st.number_input("Autonomy (Days)", 1, 30, 5)
        c1, c2 = st.columns(2)
        tank_unit_cap = c1.number_input("Bullet Cap (Gal)", 1000, 100000, 30000)
        tank_mob_cost = c2.number_input("Mob Cost/Tank ($)", 0, 50000, 5000)
        truck_capacity = st.number_input("Truck Delivery Vol (Gal)", 1000, 15000, 9000)
        tank_area_unit = 80.0

    st.divider()
    
    def_mw = eng_data['iso_rating_mw'][freq_hz]
    unit_size_iso = st.number_input("Unit Prime Rating (ISO MW)", 0.1, 100.0, def_mw, format="%.3f")
    step_load_cap = st.number_input("Unit Step Load Capability (%)", 0.0, 100.0, eng_data['step_load_pct'])
    
    st.markdown("⚠️ **Parasitic Load**")
    gen_parasitic_pct = st.number_input("Gen. Parasitic Load (%)", 0.0, 10.0, 2.5) / 100.0

    st.caption("Availability Parameters (N+M+S)")
    c_r1, c_r2 = st.columns(2)
    maint_outage_pct = c_r1.number_input("Maint. Unavail (%)", 0.0, 20.0, float(eng_data.get('default_maint', 5.0))) / 100.0
    forced_outage_pct = c_r2.number_input("Forced Outage Rate (%)", 0.0, 20.0, float(eng_data.get('default_for', 2.0))) / 100.0

    st.divider()
    
    st.header("3. Site & Conditions")
    manual_derate_pct = st.number_input("Site Derating (%)", 0.0, 50.0, 5.0)
    derate_factor_calc = 1.0 - (manual_derate_pct / 100.0)
    
    st.divider()

    st.header("4. Strategy")
    use_bess = st.checkbox("Include BESS (Optimization)", value=def_use_bess)

    st.header("7. Financials")
    if fuel_type_sel == "Natural Gas":
        fuel_price_unit = st.number_input("Gas Price (USD/MMBtu)", 1.0, 20.0, 5.0)
        fuel_price_mmbtu = fuel_price_unit
    elif fuel_type_sel == "Diesel":
        fuel_price_unit = st.number_input("Diesel Price (USD/Gal)", 1.0, 10.0, 3.50)
        fuel_price_mmbtu = fuel_price_unit / 0.138
    else: 
        fuel_price_unit = st.number_input("Propane Price (USD/Gal)", 1.0, 10.0, 1.50)
        fuel_price_mmbtu = fuel_price_unit / 0.091 

    if virtual_pipe_mode in ["LNG", "CNG"]:
        vp_premium = st.number_input("Virtual Pipe Premium ($/MMBtu)", 0.0, 15.0, 4.0)
        fuel_price_mmbtu += vp_premium

    gen_mob_cost = st.number_input("Mob/Install Cost (USD/kW)", 10.0, 1000.0, eng_data['est_mob_kw'])
    cap_charge = st.number_input("Capacity Charge (USD/MW-mo)", 5000.0, 100000.0, 28000.0, step=1000.0)
    
    revenue_per_mw_mo = st.number_input("Revenue Loss (USD/MW/mo)", 10000.0, 1000000.0, 150000.0)
    months_saved = st.number_input("Months Saved", 1, 60, 18)

    # Buyout Params (For Tab 4)
    st.caption("Post-Grid Strategy Options")
    buyout_pct = st.number_input("Buyout Residual Value (%)", 0.0, 100.0, 20.0)
    ref_new_capex = eng_data['est_asset_value_kw']
    vpp_arb_spread = st.number_input("VPP Arbitrage ($/MWh)", 0.0, 200.0, 40.0)
    vpp_cap_pay = st.number_input("VPP Capacity ($/MW-yr)", 0.0, 100000.0, 28000.0)

# ==============================================================================
# 2. CALCULATION ENGINE (PRIME ALGORITHM v3 - AGGRESSIVE)
# ==============================================================================

# A. BASE LOADS
p_total_site_load = p_it * pue_input
p_dist_loss = p_total_site_load * dist_loss_pct
p_net_gen_req = p_total_site_load + p_dist_loss 

# B. FLEET SIZING - THE PRIME ALGORITHM (Strict G2/G3 Constraints)
unit_site_cap = unit_size_iso * derate_factor_calc
step_mw_req_site = p_it * (step_load_req / 100.0)

driver_txt = "N/A"
n_steady = 0
n_transient = 0
n_headroom = 0

if use_bess:
    # BESS Optimized
    target_load_factor = 0.95 
    n_base_mw = p_net_gen_req / (1 - gen_parasitic_pct)
    n_running = math.ceil(n_base_mw / (unit_site_cap * target_load_factor))
    
    bess_power = max(step_mw_req_site, unit_site_cap)
    bess_energy = bess_power * 2
    driver_txt = "Steady State (BESS Optimized)"
else:
    # NO BESS - HARD CONSTRAINTS
    # 1. Steady State
    n_steady = math.ceil(p_net_gen_req / (unit_site_cap * 0.90))
    
    # 2. Transient Stiffness (The Prime Killer for Gas)
    # Total Fleet Step Cap must >= Step Requirement
    unit_step_mw_cap = unit_site_cap * (step_load_cap / 100.0)
    n_transient = math.ceil(step_mw_req_site / unit_step_mw_cap)
    
    # 3. Headroom
    n_headroom = n_steady
    while True:
        total_cap = n_headroom * unit_site_cap
        total_parasitics = n_headroom * (unit_size_iso * gen_parasitic_pct)
        current_load = p_net_gen_req + total_parasitics
        if (total_cap - current_load) >= step_mw_req_site:
            break
        n_headroom += 1
        
    n_running = max(n_steady, n_transient, n_headroom)
    
    if n_running == n_transient: driver_txt = f"Transient Stiffness (Step: {step_load_cap}%)"
    elif n_running == n_headroom: driver_txt = "Spinning Reserve (Headroom)"
    else: driver_txt = "Steady State Load"
            
    bess_power = 0; bess_energy = 0

# --- FLEET STRATEGY RESTORED ---
n_maint = math.ceil(n_running * maint_outage_pct) 
n_forced_buffer = math.ceil(n_running * forced_outage_pct) 
n_reserve = max(n_forced_buffer, 1) # Standard logic for Tier 1-2
if avail_req > 99.99: n_reserve = max(n_reserve, 2) # Tier 4 check

n_total = n_running + n_maint + n_reserve
installed_cap_site = n_total * unit_site_cap

# --- C. THERMODYNAMICS & EFFICIENCY (AGGRESSIVE CURVE) ---
total_parasitics_mw = n_running * (unit_size_iso * gen_parasitic_pct)
p_gross_total = p_net_gen_req + total_parasitics_mw
real_load_factor = p_gross_total / (n_running * unit_site_cap)

# Efficiency Curve Correction
base_eff = eng_data['electrical_efficiency']
type_tech = bridge_rental_library[selected_model].get('type', 'High Speed')

if type_tech == "High Speed": 
    # Recip Engine Curve (Aggressive drop below 60%)
    if real_load_factor >= 0.75: 
        eff_factor = 1.0
    elif real_load_factor >= 0.50:
        eff_factor = 0.85 + (0.6 * (real_load_factor - 0.50)) 
    else:
        eff_factor = 0.65 + (1.0 * (real_load_factor - 0.30))
else: 
    # Turbine: Linear but steep drop
    eff_factor = 1.0 - (0.8 * (1.0 - real_load_factor))

eff_factor = max(eff_factor, 0.50) # Floor

gross_eff_site = base_eff * eff_factor
gross_hr_lhv = 3412.14 / gross_eff_site

# NET HR = Total Fuel Input / USEFUL Load (IT + Cooling)
# Penalizes both Engine Efficiency (Numerator goes up) AND Parasitics/Losses (Denominator is fixed)
total_fuel_input_mmbtu = p_gross_total * (gross_hr_lhv / 1e6) 
net_hr_lhv = (total_fuel_input_mmbtu * 1e6) / p_total_site_load

# HHV Conversion
hhv_factor = 1.108 if fuel_type_sel == "Natural Gas" else (1.06 if fuel_type_sel == "Diesel" else 1.09)
net_hr_hhv = net_hr_lhv * hhv_factor

# --- D. LOGISTICS ---
total_mmbtu_day = total_fuel_input_mmbtu * 24

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

# --- E. ELECTRICAL SIZING (RESTORED) ---
grid_connected = True 
grid_mva_sc = 500.0 if grid_connected else 0.0 
xd_pu = eng_data.get('reactance_xd_2', 0.15)
gen_mva_total = installed_cap_site / 0.8
gen_sc_mva = gen_mva_total / xd_pu
total_sc_mva = gen_sc_mva + grid_mva_sc

# Voltage Logic
if is_50hz:
    op_voltage_kv = 11.0 if p_gross_total < 35 else 33.0
else:
    op_voltage_kv = 13.8 if p_gross_total < 45 else 34.5

isc_ka = total_sc_mva / (math.sqrt(3) * op_voltage_kv)
rec_breaker = 63
for b in [25, 31.5, 40, 50, 63]:
    if b > (isc_ka * 1.1): rec_breaker = b; break

# --- F. FINANCIALS ---
gen_mwh_yr = p_gross_total * 8760
fuel_cost_mwh = (net_hr_lhv / 1e6) * fuel_price_mmbtu * 1000

rental_cost_yr = (n_running * unit_site_cap) * cap_charge * 12
rental_cost_mwh = rental_cost_yr / (p_net_gen_req * 8760)
var_om = 21.5
lcoe_bridge = fuel_cost_mwh + rental_cost_mwh + var_om
lcoe_utility = 120.0

gross_rev = p_it * revenue_per_mw_mo * months_saved
cost_energy_prem = (lcoe_bridge - lcoe_utility) * (p_net_gen_req * 730 * months_saved) / 1e6
capex_total = ((n_running * unit_site_cap * 1000 * gen_mob_cost) + log_capex) / 1e6
net_benefit = (gross_rev/1e6) - cost_energy_prem - capex_total

# --- CALCULATIONS FOR TAB 4 (FUTURE VALUE) ---
total_asset_value_m = (installed_cap_site * 1000 * ref_new_capex) / 1e6
buyout_price_m = total_asset_value_m * (buyout_pct / 100.0)
savings_vs_new = total_asset_value_m - buyout_price_m

rev_arb = installed_cap_site * vpp_arb_spread * 365 
rev_cap = installed_cap_site * vpp_cap_pay
total_vpp_yr_m = (rev_arb + rev_cap) / 1e6

# --- G. FOOTPRINT ---
area_gen_total = n_total * 150 
area_bess_total = bess_power * 30 
area_sub_total = 2500 
total_area_m2 = (area_gen_total + storage_area_m2 + area_bess_total + area_sub_total) * 1.2 

# ==============================================================================
# 3. DASHBOARD
# ==============================================================================

c1, c2, c3, c4 = st.columns(4)
c1.metric("Bridge Capacity", f"{p_net_gen_req:.1f} MW", f"IT Load: {p_it:.1f} MW")
c2.metric("Net Heat Rate (LHV)", f"{net_hr_lhv:,.0f} Btu/kWh", f"LF: {real_load_factor*100:.1f}%")
c3.metric("LCOE (Bridge)", f"${lcoe_bridge:.2f}/MWh", f"Fuel: ${fuel_cost_mwh:.2f}")
c4.metric("Net Benefit", f"${net_benefit:.1f} M", f"Saved: {months_saved} Mo")

st.divider()

t1, t2, t3, t4 = st.tabs(["⚙️ Engineering & Fleet", "🏗️ Logistics & Site", "💰 Business Case", "🔮 Future Value"])

with t1:
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.subheader("🔥 Heat Rate Analysis")
        st.write(f"**Strategy:** {'BESS Optimized' if use_bess else 'Spinning Reserve'}")
        st.write(f"**Driver:** {driver_txt}")
        
        st.metric("Gross HR (Engine)", f"{gross_hr_lhv:,.0f}", f"Eff: {gross_eff_site*100:.1f}%")
        st.metric("Net HR (System)", f"{net_hr_lhv:,.0f}", f"Delta: +{net_hr_lhv-gross_hr_lhv:,.0f}")
        st.info(f"**HHV (Billing):** {net_hr_hhv:,.0f} Btu/kWh")
        
        if not use_bess and real_load_factor < 0.50:
            st.error(f"⛔ **CRITICAL:** RICE LF {real_load_factor*100:.1f}% < 50%. Wet Stacking Risk.")

    with col2:
        st.subheader("🚜 Fleet Strategy (N+M+S)")
        st.write(f"**Model:** {selected_model}")
        st.write(f"**Site Rating:** {unit_site_cap:.2f} MW")
        st.write(f"**Avg. Load Factor:** {real_load_factor*100:.1f}%")
        st.markdown("---")
        st.write(f"**N (Running):** {n_running}")
        st.write(f"**M (Maintenance):** {n_maint}")
        st.write(f"**S (Standby):** {n_reserve}")
        st.metric("Total Fleet", f"{n_total} Units", f"{installed_cap_site:.1f} MW Total")
        
        if use_bess:
            st.info(f"⚡ **BESS:** {bess_power:.1f} MW / {bess_energy:.1f} MWh")

    with col3:
        st.subheader("⚡ Electrical Sizing")
        st.write(f"**Voltage:** {op_voltage_kv} kV")
        st.write(f"**Gen Contribution:** {gen_sc_mva:.1f} MVA")
        st.metric("Total Short Circuit", f"{isc_ka:.1f} kA")
        st.success(f"✅ Breaker: **{rec_breaker} kA**")
        
        st.markdown("---")
        st.write("**Loss Breakdown:**")
        st.write(f"• Parasitics: {total_parasitics_mw:.2f} MW")
        st.write(f"• Dist. Loss: {p_dist_loss:.2f} MW")

with t2:
    st.subheader(f"Logistics: {virtual_pipe_mode}")
    if virtual_pipe_mode != "Pipeline":
        c_l1, c_l2 = st.columns(2)
        c_l1.metric("Daily Volume", log_text)
        c_l1.metric("Assets Req.", f"{num_tanks} Tanks")
        c_l2.metric("Storage Area", f"{storage_area_m2:.0f} m²")
        c_l2.metric("Logistics CAPEX", f"${log_capex:,.0f}")
    else:
        st.success("Pipeline Connected")
        
    st.divider()
    st.subheader("Footprint")
    col_name = f"Area ({'ft²' if is_imperial else 'm²'})"
    df_foot = pd.DataFrame({
        "Zone": ["Generation", "Fuel/Logistics", "BESS", "Substation", "Total"],
        col_name: [
            area_gen_total, storage_area_m2, area_bess_total, area_sub_total, total_area_m2
        ]
    })
    st.dataframe(df_foot.style.format({col_name: "{:,.0f}"}), use_container_width=True)

with t3:
    st.subheader("Financial Waterfall")
    fig_water = go.Figure(go.Waterfall(
        name = "20", orientation = "v",
        measure = ["relative", "relative", "relative", "total"],
        x = ["Gross Revenue", "Energy Premium", "Mob & Storage Cost", "NET BENEFIT"],
        textposition = "outside",
        text = [f"+{gross_rev/1e6:.1f}M", f"-{cost_energy_prem:.1f}M", f"-{capex_total:.1f}M", f"${net_benefit:.1f}M"],
        y = [gross_rev/1e6, -cost_energy_prem, -capex_total, net_benefit],
        connector = {"line":{"color":"rgb(63, 63, 63)"}},
    ))
    st.plotly_chart(fig_water, use_container_width=True)
    
    st.divider()
    st.subheader("LCOE Structure")
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
st.caption("CAT Bridge Solutions Designer v37 | Full Engineering Suite")
