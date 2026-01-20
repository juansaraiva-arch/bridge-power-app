import streamlit as st
import pandas as pd
import numpy as np
import math
import plotly.express as px
import plotly.graph_objects as go

# --- PAGE CONFIGURATION ---
st.set_page_config(page_title="CAT Bridge Solutions Designer v26", page_icon="🌉", layout="wide")

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
        "step_load_pct": 40.0,
        "emissions_nox": 0.5,
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
        "step_load_pct": 20.0,
        "emissions_nox": 0.6,
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
        "est_asset_value_kw": 500.0, "est_mob_kw": 40.0,
        "reactance_xd_2": 0.12
    }
}

# ==============================================================================
# 1. INPUTS
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
    
    def_step_load = 40.0 if is_ai else 10.0
    def_use_bess = True if is_ai else False
    
    p_it = st.number_input("Critical IT Load (MW)", 1.0, 1000.0, 50.0, step=10.0)
    pue_input = st.number_input("Design PUE", 1.0, 2.0, 1.35, step=0.01)
    
    step_load_req = st.number_input("Expected Step Load (%)", 0.0, 100.0, def_step_load)
    dist_loss_pct = st.number_input("Distribution Losses (%)", 0.0, 10.0, 1.0) / 100.0

    st.divider()

    # 2. Tech
    st.header("2. Technology & Fuel")
    fuel_type_sel = st.selectbox("Fuel Source", ["Natural Gas", "Diesel", "Propane"])
    
    avail_models = [k for k, v in bridge_rental_library.items() if fuel_type_sel in v['fuels']]
    if not avail_models: st.error("No units found"); st.stop()
            
    selected_model = st.selectbox("Select Bridge Unit", avail_models)
    eng_data = bridge_rental_library[selected_model]
    st.info(f"**{eng_data['description']}**")

    # Storage Vars
    virtual_pipe_mode = "None"
    methane_number = 80
    tank_unit_cap, tank_mob_cost, tank_area_unit, storage_days = 1.0, 0.0, 0.0, 0

    if fuel_type_sel == "Natural Gas":
        methane_number = st.number_input("Methane Number (MN)", 30, 100, 80)
        gas_source = st.radio("Supply Method", ["Pipeline", "Virtual Pipeline (LNG)", "Virtual Pipeline (CNG)"])
        
        if gas_source == "Pipeline": virtual_pipe_mode = "Pipeline"
        elif "LNG" in gas_source:
            virtual_pipe_mode = "LNG"
            st.markdown("🔹 **LNG Storage**")
            storage_days = st.number_input("Autonomy (Days)", 1, 30, 5)
            tank_unit_cap = st.number_input("ISO Tank Cap (Gal)", 1000, 20000, 10000)
            tank_mob_cost = st.number_input("Mob Cost/Tank ($)", 0, 50000, 5000)
            tank_area_unit = 40.0
        elif "CNG" in gas_source:
            virtual_pipe_mode = "CNG"
            st.markdown("🔹 **CNG Storage**")
            storage_days = st.number_input("Autonomy (Days)", 1, 30, 1)
            tank_unit_cap = st.number_input("Trailer Cap (scf)", 50000, 1000000, 350000)
            tank_mob_cost = st.number_input("Mob Cost/Trailer ($)", 0, 50000, 2000)
            tank_area_unit = 60.0
            
    elif fuel_type_sel == "Diesel":
        virtual_pipe_mode = "Diesel"
        st.markdown("🔹 **Diesel Storage**")
        storage_days = st.number_input("Autonomy (Days)", 1, 30, 3)
        tank_unit_cap = st.number_input("Frac Tank Cap (Gal)", 1000, 50000, 20000)
        tank_mob_cost = st.number_input("Mob Cost/Tank ($)", 0, 50000, 2500)
        tank_area_unit = 50.0
        
    elif fuel_type_sel == "Propane":
        virtual_pipe_mode = "Propane"
        st.markdown("🔹 **LPG Storage**")
        storage_days = st.number_input("Autonomy (Days)", 1, 30, 5)
        tank_unit_cap = st.number_input("Tank Cap (Gal)", 1000, 100000, 30000)
        tank_mob_cost = st.number_input("Mob Cost/Tank ($)", 0, 50000, 5000)
        tank_area_unit = 80.0

    st.divider()
    # Tech Specs Override
    def_mw = eng_data['iso_rating_mw'][freq_hz]
    unit_size_iso = st.number_input("Unit Prime Rating (ISO MW)", 0.1, 100.0, def_mw, format="%.3f")
    step_load_cap = st.number_input("Unit Step Load Capability (%)", 0.0, 100.0, eng_data['step_load_pct'])
    
    st.markdown("⚠️ **Parasitic Load**")
    st.info("Defined as % of Nameplate Rating (Fixed per running unit)")
    gen_parasitic_pct = st.number_input("Gen. Parasitic Load (%)", 0.0, 10.0, 2.5) / 100.0

    st.divider()
    
    # 3. Site
    st.header("3. Site & Conditions")
    manual_derate_pct = st.number_input("Site Derating (%)", 0.0, 50.0, 5.0)
    derate_factor_calc = 1.0 - (manual_derate_pct / 100.0)
    
    st.divider()

    # 4. Strategy
    st.header("4. Strategy")
    use_bess = st.checkbox("Include BESS (Optimization)", value=def_use_bess)

    # 7. Financials
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

# ==============================================================================
# 2. CALCULATION ENGINE (PHYSICS & THERMODYNAMICS)
# ==============================================================================

# A. BASE LOADS
p_total_site_load = p_it * pue_input
p_dist_loss = p_total_site_load * dist_loss_pct
p_net_gen_req = p_total_site_load + p_dist_loss # Power required at Gen Bus

# B. FLEET SIZING & OPERATING POINT
unit_site_cap = unit_size_iso * derate_factor_calc
step_mw_req_site = p_it * (step_load_req / 100.0)

# --- CRITICAL LOGIC: N_RUNNING & LOAD FACTOR ---
if use_bess:
    # BESS handles steps. Run engines at optimal base load.
    target_load_factor = 0.95 
    n_base_mw = p_net_gen_req / (1 - gen_parasitic_pct) # Estimate gross
    n_running = math.ceil(n_base_mw / (unit_site_cap * target_load_factor))
    
    bess_power = max(step_mw_req_site, unit_site_cap)
    bess_energy = bess_power * 2 # 2 hr duration
else:
    # No BESS. Engines provide Spinning Reserve for Step Load.
    # Logic: Available Headroom >= Step Load
    n_min = math.ceil(p_net_gen_req / unit_site_cap)
    n_running = n_min
    while True:
        # Step Capability = Running Cap * Step% (or remaining headroom)
        total_cap_mw = n_running * unit_site_cap
        # Parasitics depend on N (Fixed per unit)
        total_parasitics_mw = n_running * (unit_size_iso * gen_parasitic_pct) # Fixed load per unit
        
        # Available Gross Capacity for Load
        avail_gross_for_load = total_cap_mw - total_parasitics_mw
        
        # Current Load %
        gross_load_needed = p_net_gen_req + total_parasitics_mw
        current_load_pct = gross_load_needed / total_cap_mw
        
        # Headroom Check
        headroom_mw = total_cap_mw - gross_load_needed
        
        # Transient Check (Can the fleet accept the step?)
        step_capacity_mw = total_cap_mw * (step_load_cap / 100.0)
        
        # Valid if Headroom > Step AND Step Cap > Step
        if headroom_mw >= step_mw_req_site and step_capacity_mw >= step_mw_req_site:
            break
        n_running += 1
    
    bess_power = 0; bess_energy = 0

# --- C. THERMODYNAMICS & EFFICIENCY ---
# 1. Total Parasitics (Fixed Physics: Fans/Pumps run per unit)
total_parasitics_mw = n_running * (unit_size_iso * gen_parasitic_pct)

# 2. Total Gross Generation Required
p_gross_total = p_net_gen_req + total_parasitics_mw

# 3. Real Fleet Load Factor
real_load_factor = p_gross_total / (n_running * unit_site_cap)

# 4. Part-Load Efficiency Correction (RICE vs Turbine)
base_eff = eng_data['electrical_efficiency']
type_tech = eng_data['type']

if type_tech == "High Speed": 
    # Recip Engine Curve: Stable high, drops below 75%
    if real_load_factor >= 0.75: 
        eff_factor = 1.0
    else: 
        # Quadratic decay below 75%
        eff_factor = 1.0 - (0.5 * (0.75 - real_load_factor)**2)
else: 
    # Turbine: Linear drop
    eff_factor = 1.0 - (0.6 * (1.0 - real_load_factor))

gross_eff_site = base_eff * eff_factor
gross_hr_lhv = 3412.14 / gross_eff_site

# 5. NET HEAT RATE (The Metric that matters)
# Net HR = Fuel Input (MMBtu) / Net Output (MWh)
fuel_input_mmbtu_hr = p_gross_total * (gross_hr_lhv / 1e6)
net_output_mw = p_it # Useful IT Load (Strict Net) OR p_net_gen_req (Facility Net). 
# Standard Industry Practice: Net HR usually refers to Facility Net (Post-Aux, Pre-Dist Loss)
# But let's calculate based on p_net_gen_req (Output from Gen Bus - Aux)
net_hr_lhv = (fuel_input_mmbtu_hr * 1e6) / p_net_gen_req

# HHV Conversion
hhv_factor = 1.108 if fuel_type_sel == "Natural Gas" else (1.06 if fuel_type_sel == "Diesel" else 1.09)
net_hr_hhv = net_hr_lhv * hhv_factor

# --- D. LOGISTICS ---
total_mmbtu_day = fuel_input_mmbtu_hr * 24
num_tanks = 0; log_capex = 0; log_text = "Pipeline"

if virtual_pipe_mode == "LNG":
    vol_day = total_mmbtu_day * 12.5
    num_tanks = math.ceil((vol_day * storage_days)/tank_unit_cap)
    log_capex = num_tanks * tank_mob_cost
    log_text = f"LNG: {vol_day:,.0f} gpd"
elif virtual_pipe_mode == "CNG":
    vol_day = total_mmbtu_day * 1000
    num_tanks = math.ceil((vol_day * storage_days)/tank_unit_cap)
    log_capex = num_tanks * tank_mob_cost
    log_text = f"CNG: {vol_day/1e6:.2f} MMscfd"
elif virtual_pipe_mode in ["Diesel", "Propane"]:
    conv = 7.3 if virtual_pipe_mode == "Diesel" else 11.0
    vol_day = total_mmbtu_day * conv
    num_tanks = math.ceil((vol_day * storage_days)/tank_unit_cap)
    log_capex = num_tanks * tank_mob_cost
    log_text = f"{virtual_pipe_mode}: {vol_day:,.0f} gpd"

# --- E. FINANCIALS ---
gen_mwh_yr = p_gross_total * 8760 # Fuel is paid on Gross
# Fuel Cost per MWh (Useful)
# Cost = (Fuel Input / Net Gen) * Price
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

# ==============================================================================
# 3. DASHBOARD
# ==============================================================================

c1, c2, c3, c4 = st.columns(4)
c1.metric("Net Capacity (IT)", f"{p_it:.1f} MW", f"Gross Needed: {p_gross_total:.1f} MW")
c2.metric("Net Heat Rate (LHV)", f"{net_hr_lhv:,.0f} Btu/kWh", f"Load Factor: {real_load_factor*100:.1f}%")
c3.metric("LCOE (Bridge)", f"${lcoe_bridge:.2f}/MWh", f"Fuel: ${fuel_cost_mwh:.2f}")
c4.metric("Net Benefit", f"${net_benefit:.1f} M", f"Saved: {months_saved} Mo")

st.divider()

t1, t2, t3 = st.tabs(["⚙️ Thermodynamics", "🏗️ Logistics & Site", "💰 Business Case"])

with t1:
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("🔥 Heat Rate Analysis")
        st.write(f"**Operating Strategy:** {'BESS Optimized (High Load)' if use_bess else 'Spinning Reserve (Part Load)'}")
        
        col_a, col_b = st.columns(2)
        col_a.metric("Gross HR (Engine)", f"{gross_hr_lhv:,.0f}", f"Eff: {gross_eff_site*100:.1f}%")
        col_b.metric("Net HR (System)", f"{net_hr_lhv:,.0f}", f"Delta: +{net_hr_lhv-gross_hr_lhv:,.0f}")
        
        st.info(f"**Billing Heat Rate (HHV):** {net_hr_hhv:,.0f} Btu/kWh")
        
        st.markdown("---")
        st.write("**Loss Breakdown:**")
        st.write(f"• Engines Running: **{n_running}** units")
        st.write(f"• Total Parasitics: **{total_parasitics_mw:.2f} MW** (Fixed fans/pumps)")
        st.write(f"• Dist. Losses: **{p_dist_loss:.2f} MW**")
        
        if not use_bess:
            st.warning(f"⚠️ No BESS Penalty: Running {n_running - math.ceil(p_net_gen_req/unit_site_cap)} extra units for spinning reserve increases parasitic load and degrades engine efficiency.")

    with col2:
        st.subheader("Power Balance")
        df_bal = pd.DataFrame({
            "Stage": ["IT Load", "+ Cooling/PUE", "+ Dist. Losses", "= Net Gen Req", "+ Parasitics", "= GROSS GEN"],
            "MW": [p_it, p_total_site_load-p_it, p_dist_loss, p_net_gen_req, total_parasitics_mw, p_gross_total]
        })
        st.dataframe(df_bal.style.format({"MW": "{:.2f}"}), use_container_width=True)

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

# --- FOOTER ---
st.markdown("---")
st.caption("CAT Bridge Solutions Designer v26 | Physics-Based Thermodynamics")
