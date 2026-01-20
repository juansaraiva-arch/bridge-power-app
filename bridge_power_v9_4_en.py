import streamlit as st
import pandas as pd
import numpy as np
import math
import plotly.express as px
import plotly.graph_objects as go

# --- PAGE CONFIGURATION ---
st.set_page_config(page_title="CAT Bridge Solutions Designer v29 (Prime Algo)", page_icon="🌉", layout="wide")

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
    
    def_step_load = 40.0 if is_ai else 10.0
    def_use_bess = True if is_ai else False
    
    p_it = st.number_input("Critical IT Load (MW)", 1.0, 1000.0, 50.0, step=10.0)
    pue_input = st.number_input("Design PUE", 1.0, 2.0, 1.35, step=0.01)
    
    avail_req = st.number_input("Availability Target (%)", 90.0, 99.99999, 99.99, format="%.5f")
    step_load_req = st.number_input("Expected Step Load (%)", 0.0, 100.0, def_step_load)
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
    st.info("Defined as % of Nameplate Rating (Fixed per running unit)")
    gen_parasitic_pct = st.number_input("Gen. Parasitic Load (%)", 0.0, 10.0, 2.5) / 100.0

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

    # Buyout Params
    buyout_pct = 20.0
    ref_new_capex = eng_data['est_asset_value_kw']
    vpp_arb_spread = 40.0
    vpp_cap_pay = 28000.0

# ==============================================================================
# 2. CALCULATION ENGINE (PRIME ALGORITHM)
# ==============================================================================

# A. BASE LOADS
p_total_site_load = p_it * pue_input
p_dist_loss = p_total_site_load * dist_loss_pct
p_net_gen_req = p_total_site_load + p_dist_loss # Power required at Gen Bus

# B. FLEET SIZING - THE PRIME ALGORITHM
unit_site_cap = unit_size_iso * derate_factor_calc
step_mw_req_site = p_it * (step_load_req / 100.0)

# Variables for Engineering Dashboard
driver_txt = "N/A"
n_steady = 0
n_transient = 0
n_headroom = 0

if use_bess:
    # BESS Strategy: BESS takes the hit. Engines run purely for load.
    # Target Load Factor: 95% (Sweet Spot)
    # Solve for N:  (N * Unit * 0.95) - (N * Unit * Parasitic%) = Net_Load
    # N * Unit * (0.95 - Parasitic%) = Net_Load
    effective_capacity = unit_site_cap * (0.95 - gen_parasitic_pct)
    n_running = math.ceil(p_net_gen_req / effective_capacity)
    
    bess_power = max(step_mw_req_site, unit_site_cap)
    bess_energy = bess_power * 2
    driver_txt = "Steady State (BESS Optimized)"
else:
    # PRIME ALGORITHM: NO BESS
    # We must solve for N iteratively because Parasitics depend on N.
    # We assume N starts at minimum and increments until all 3 Vectors are satisfied.
    
    n_running = math.ceil(p_net_gen_req / unit_site_cap) # Initial guess
    
    while True:
        # 1. Calculate Real Physics for current N
        total_parasitics_mw = n_running * (unit_size_iso * gen_parasitic_pct) # Fixed load per unit
        p_gross_needed = p_net_gen_req + total_parasitics_mw
        total_gross_cap = n_running * unit_site_cap
        
        # 2. VECTOR A: Steady State Capacity
        # Do we have enough gross cap to cover load + parasitics?
        is_steady_ok = total_gross_cap >= p_gross_needed
        
        # 3. VECTOR B: Transient Stiffness (Step Capability)
        # Can the running fleet accept the step MW without tripping?
        # Fleet Step Cap = N * Unit_Cap * Step_%
        fleet_step_cap = total_gross_cap * (step_load_cap / 100.0)
        is_transient_ok = fleet_step_cap >= step_mw_req_site
        
        # 4. VECTOR C: Spinning Reserve (Headroom)
        # Is there enough empty space to fit the step?
        # Empty Space = Total Cap - Current Load
        available_headroom = total_gross_cap - p_gross_needed
        is_headroom_ok = available_headroom >= step_mw_req_site
        
        if is_steady_ok and is_transient_ok and is_headroom_ok:
            # Determine Driver for Dashboard display
            if fleet_step_cap < (step_mw_req_site * 1.1): # Close call on Transient
                driver_txt = "Transient Stiffness (Step %)"
            elif available_headroom < (step_mw_req_site * 1.1):
                driver_txt = "Spinning Reserve (Headroom)"
            else:
                driver_txt = "Steady State Load"
            break
        
        n_running += 1
        
        if n_running > 200: # Safety break
            break
            
    bess_power = 0; bess_energy = 0

# --- C. THERMODYNAMICS & EFFICIENCY ---
total_parasitics_mw = n_running * (unit_size_iso * gen_parasitic_pct)
p_gross_total = p_net_gen_req + total_parasitics_mw
real_load_factor = p_gross_total / (n_running * unit_site_cap)

# Part-Load Efficiency Correction
base_eff = eng_data['electrical_efficiency']
type_tech = bridge_rental_library[selected_model].get('type', 'High Speed')

if type_tech == "High Speed": 
    if real_load_factor >= 0.75: eff_factor = 1.0
    else: eff_factor = 1.0 - (0.5 * (0.75 - real_load_factor)**2)
else: 
    eff_factor = 1.0 - (0.6 * (1.0 - real_load_factor))

gross_eff_site = base_eff * eff_factor
gross_hr_lhv = 3412.14 / gross_eff_site

# NET HR = Fuel (Btu) / Useful Load (IT + Cooling)
# Excludes Distribution losses from the denominator to penalize them
total_fuel_input_mmbtu = p_gross_total * (gross_hr_lhv / 1e6) 
net_hr_lhv = (total_fuel_input_mmbtu * 1e6) / p_total_site_load

# HHV Conversion
hhv_factor = 1.108 if fuel_type_sel == "Natural Gas" else (1.06 if fuel_type_sel == "Diesel" else 1.09)
net_hr_hhv = net_hr_lhv * hhv_factor

n_maint = math.ceil(n_running * 0.05) 
n_forced_buffer = math.ceil(n_running * 0.02) 
n_reserve = max(n_forced_buffer, 1) 

n_total = n_running + n_maint + n_reserve
installed_cap_site = n_total * unit_site_cap

# --- D. LOGISTICS ---
total_mmbtu_day = total_fuel_input_mmbtu * 24

# Initializing vars
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

# --- E. FINANCIALS ---
gen_mwh_yr = p_gross_total * 8760
fuel_cost_mwh = (net_hr_lhv / 1e6) * fuel_price_mmbtu * 1000

rental_cost_yr = installed_cap_site * cap_charge * 12
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
c1.metric("Bridge Capacity", f"{p_net_gen_req:.1f} MW", f"IT Load: {p_it:.1f} MW")
c2.metric("Fleet Configuration", f"{n_total} Units", f"Run: {n_running} | Driver: {driver_txt}")
c3.metric("LCOE (Bridge)", f"${lcoe_bridge:.2f}/MWh", f"Fuel: ${fuel_cost_mwh:.2f}")
c4.metric("Net Benefit", f"${net_benefit:.1f} M", f"Saved: {months_saved} Mo")

st.divider()

t1, t2, t3 = st.tabs(["⚙️ Thermodynamics", "🏗️ Logistics & Site", "💰 Business Case"])

with t1:
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("🔥 Heat Rate Analysis")
        st.write(f"**Operating Strategy:** {'BESS Optimized (High Load)' if use_bess else 'Spinning Reserve (Part Load)'}")
        st.write(f"**Sizing Driver:** {driver_txt}")
        
        col_a, col_b = st.columns(2)
        col_a.metric("Gross HR (Engine)", f"{gross_hr_lhv:,.0f}", f"Eff: {gross_eff_site*100:.1f}%")
        col_b.metric("Net HR (System)", f"{net_hr_lhv:,.0f}", f"Delta: +{net_hr_lhv-gross_hr_lhv:,.0f}")
        
        st.info(f"**Billing Heat Rate (HHV):** {net_hr_hhv:,.0f} Btu/kWh")
        
        st.markdown("---")
        st.write("**Loss Breakdown:**")
        st.write(f"• Engines Running: **{n_running}** units")
        st.write(f"• Load Factor: **{real_load_factor*100:.1f}%**")
        st.write(f"• Total Parasitics: **{total_parasitics_mw:.2f} MW** (Fixed per unit)")
        st.write(f"• Dist. Losses: **{p_dist_loss:.2f} MW**")
        
        if not use_bess and real_load_factor < 0.75:
            st.warning(f"⚠️ **Efficiency Penalty:** Low load factor ({real_load_factor*100:.1f}%) required for Spinning Reserve is increasing Fuel Consumption.")

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
st.caption("CAT Bridge Solutions Designer v29 | Powered by Prime Engineering Engine")
