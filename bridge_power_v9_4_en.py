import streamlit as st
import pandas as pd
import numpy as np
import math
import plotly.express as px
import plotly.graph_objects as go

# --- PAGE CONFIGURATION ---
st.set_page_config(page_title="Bridge Power Design Engine V9.6", page_icon="🏭", layout="wide")

# ==============================================================================
# 0. GLOBAL SETTINGS
# ==============================================================================

with st.sidebar:
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
    u_alt = "ft"
else:
    u_temp = "°C"
    u_dist = "m"
    u_area_s = "m²"
    u_area_l = "Ha"
    u_vol = "L"
    u_mass = "Tonnes"
    u_alt = "masl"

# Dictionary
t = {
    "title": "🏭 Bridge Power Design Engine V9.6",
    "subtitle": "**Total Engineering Suite.**\nCalculates: Availability, BESS, Urea Logistics, Footprint, and **Business Case (Step 8)**.",
    "sb_1_title": "1. Data Center Profile",
    "dc_type_label": "Data Center Type",
    "dc_opts": ["AI Factory (Training/Inference)", "Standard Hyperscale"],
    "p_max": "Critical IT Load (MW)",
    "step_load": "Expected Step Load (%)",
    "voltage_label": "Connection Voltage (kV)",
    "dist_loss": "Distribution Losses (%)",
    "aux_load": "Campus Auxiliaries (%)",
    "sb_2_title": "2. Generation Technology",
    "tech_opts": ["RICE (Reciprocating Engine)", "Gas Turbine (Aero)"],
    "unit_iso": "ISO Prime Rating (MW)",
    "eff_iso": "ISO Thermal Efficiency (%)",
    "parasitic": "Gen. Parasitic Load (%)",
    "step_cap": "Step Load Capability (%)",
    "emi_title": "Native Emissions",
    "sb_3_title": "3. Site Physics & Neighbors",
    "derate_opts": ["Automatic", "Manual"],
    "temp": "Avg Max Temp",
    "alt": "Altitude",
    "mn": "Methane Number (Gas)",
    "urban_int": "Urban Integration",
    "bldg_h": "Tallest Nearby Building",
    "dist_n": "Distance to Neighbor",
    "n_opts": ["Industrial", "Residential/Sensitive"],
    "source_noise": "Source Noise @ 1m (dBA)",
    "sb_4_title": "4. Regulatory & Urea",
    "reg_opts": ["USA - EPA Major", "EU Standard", "LatAm / Unregulated"],
    "urea_days": "Urea Autonomy (Days)",
    "sb_5_title": "5. Reliability & BESS",
    "avail_target": "Availability Target (%)",
    "use_bess": "Include BESS (Synthetic Inertia)",
    "maint": "Maintenance Unavailability (%)",
    # NEW STEP 8 INPUTS
    "sb_6_title": "6. Business & Strategy (Step 8)",
    "fuel_p": "Gas Price ($/MMBtu)",
    "cap_charge": "Capacity Charge ($/MW-mo)",
    "var_om": "Variable O&M ($/MWh)",
    "grid_rate": "Utility Grid Rate ($/kWh)",
    "contract_dur": "Contract Duration (Years)",
    "buyout_res": "Buyout Residual Value (%)",
    "new_capex": "Ref. New CAPEX ($/kW)",
    "vpp_arb": "VPP Arbitrage ($/MWh)",
    "vpp_cap": "VPP Capacity ($/MW-yr)",
    "vpp_anc": "VPP Ancillary ($/MW-yr)",
    
    "tab_tech": "⚙️ Engineering",
    "tab_area": "🏗️ Footprint",
    "tab_env": "🧪 Env & Physics",
    "tab_biz": "💰 Business Case"
}

st.title(t["title"])
st.markdown(t["subtitle"])

# ==============================================================================
# 1. INPUTS (SIDEBAR)
# ==============================================================================

with st.sidebar:
    # --- 1. PROFILE ---
    st.header(t["sb_1_title"])
    dc_type_sel = st.selectbox(t["dc_type_label"], t["dc_opts"])
    is_ai = "AI" in dc_type_sel
    
    def_step_load = 40.0 if is_ai else 10.0
    def_use_bess = True if is_ai else False
    
    p_max = st.number_input(t["p_max"], 10.0, 1000.0, 100.0, step=10.0)
    step_load_req = st.number_input(t["step_load"], 0.0, 100.0, def_step_load)
    voltage_kv = st.number_input(t["voltage_label"], 0.4, 500.0, 34.5, step=0.5)
    dist_loss_pct = st.number_input(t["dist_loss"], 0.0, 10.0, 3.0) / 100
    aux_load_pct = st.number_input(t["aux_load"], 0.0, 15.0, 2.5) / 100

    st.divider()

    # --- 2. TECHNOLOGY ---
    st.header(t["sb_2_title"])
    tech_type_sel = st.selectbox("Technology", t["tech_opts"])
    is_rice = "RICE" in tech_type_sel
    
    if is_rice:
        def_mw, def_eff, def_par, def_step_cap = 2.5, 46.0, 2.5, 65.0
        def_maint, def_noise_source, def_nox = 5.0, 85.0, 1.0
    else: 
        def_mw, def_eff, def_par, def_step_cap = 35.0, 38.0, 0.5, 20.0
        def_maint, def_noise_source, def_nox = 3.0, 90.0, 0.6

    unit_size_iso = st.number_input(t["unit_iso"], 1.0, 100.0, def_mw)
    eff_gen_base = st.number_input(t["eff_iso"], 20.0, 65.0, def_eff)
    parasitic_pct = st.number_input(t["parasitic"], 0.0, 10.0, def_par) / 100
    gen_step_cap = st.number_input(t["step_cap"], 0.0, 100.0, def_step_cap)
    
    st.caption(t["emi_title"])
    c_e1, c_e2 = st.columns(2)
    raw_nox = c_e1.number_input("NOx (g/bhp-hr)", 0.0, 10.0, def_nox)
    raw_co = c_e2.number_input("CO (g/bhp-hr)", 0.0, 10.0, 0.5)

    st.divider()

    # --- 3. PHYSICS ---
    st.header(t["sb_3_title"])
    derate_method_sel = st.radio("Derate Method", t["derate_opts"])
    is_auto_derate = "Auto" in derate_method_sel
    derate_factor_calc = 1.0
    
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
        derate_factor_calc = 1.0 - min(0.5, loss_temp + loss_alt + loss_mn)
    else:
        manual_derate = st.number_input("Manual Derate (%)", 0.0, 50.0, 0.0)
        derate_factor_calc = 1.0 - (manual_derate / 100.0)

    unit_size_site = unit_size_iso * derate_factor_calc

    st.caption(t["urban_int"])
    if is_imperial:
        nearby_bldg_ft = st.number_input(f"{t['bldg_h']} ({u_dist})", 15.0, 350.0, 40.0)
        dist_neighbor_ft = st.number_input(f"{t['dist_n']} ({u_dist})", 30.0, 6500.0, 328.0)
        nearby_building_h_m = nearby_bldg_ft / 3.28084
        dist_neighbor_m = dist_neighbor_ft / 3.28084
    else:
        nearby_building_h_m = st.number_input(f"{t['bldg_h']} ({u_dist})", 5.0, 100.0, 12.0)
        dist_neighbor_m = st.number_input(f"{t['dist_n']} ({u_dist})", 10.0, 2000.0, 100.0)

    neighbor_type_sel = st.selectbox("Neighbor Type", t["n_opts"])
    noise_limit = 70.0 if "Industrial" in neighbor_type_sel else 55.0
    source_noise_dba = st.number_input(t["source_noise"], 60.0, 120.0, def_noise_source)

    st.divider()

    # --- 4. REGULATORY ---
    st.header(t["sb_4_title"])
    reg_zone = st.selectbox("Region", t["reg_opts"])
    limit_nox_tpy = 250.0 if "EPA Major" in reg_zone else (100.0 if "Virginia" in reg_zone else 9999.0)
    urea_days = st.number_input(t["urea_days"], 1, 30, 7)

    st.header(t["sb_5_title"])
    avail_target = st.number_input("Availability Target (%)", 90.00, 99.99999, 99.999, format="%.5f")
    use_bess = st.checkbox(t["use_bess"], value=def_use_bess)
    maint_unav = st.number_input(t["maint"], 0.0, 20.0, def_maint) / 100
    
    st.divider()

    # --- 6. BUSINESS & STRATEGY (STEP 8) ---
    st.header(t["sb_6_title"])
    st.caption("EaaS / PPA Parameters")
    fuel_price = st.number_input(t["fuel_p"], 1.0, 20.0, 5.0) # $5/MMBtu default
    cap_charge = st.number_input(t["cap_charge"], 5000.0, 50000.0, 20000.0, step=1000.0) # $20k/MW default
    var_om = st.number_input(t["var_om"], 0.0, 100.0, 21.50) # $21.50/MWh default
    grid_rate_kwh = st.number_input(t["grid_rate"], 0.01, 0.50, 0.092, format="%.3f") # $0.092/kWh default
    
    # CONTRACT DURATION INPUT
    contract_years = st.number_input(t["contract_dur"], 1, 20, 5) 
    
    st.caption(f"Asset Transfer (Year {contract_years})")
    buyout_pct = st.number_input(t["buyout_res"], 0.0, 100.0, 20.0) # 20%
    new_asset_capex = st.number_input(t["new_capex"], 100.0, 2000.0, 500.0) # $500/kW
    
    st.caption("Future VPP Revenue")
    vpp_arb_spread = st.number_input(t["vpp_arb"], 0.0, 200.0, 40.0) # $/MWh spread
    vpp_cap_pay = st.number_input(t["vpp_cap"], 0.0, 100000.0, 28000.0) # $/MW-yr
    vpp_anc_pay = st.number_input(t["vpp_anc"], 0.0, 100000.0, 15000.0) # $/MW-yr

# ==============================================================================
# 2. CALCULATION ENGINE
# ==============================================================================

# A. LOAD & POWER
p_aux_mw = p_max * aux_load_pct 
p_dist_loss = (p_max + p_aux_mw) * dist_loss_pct
p_net_gen_req = p_max + p_aux_mw + p_dist_loss 
p_gross_gen_req = p_net_gen_req / (1 - parasitic_pct)

# B. FLEET SIZING
n_base = math.ceil(p_gross_gen_req / unit_size_site)
req_step_mw = p_max * (step_load_req / 100.0)
min_redundancy = 2 if avail_target >= 99.99 else 1

if use_bess:
    bess_mw = req_step_mw + unit_size_site 
    bess_mwh = bess_mw * 2 
    n_spin = min_redundancy
else:
    bess_mw = 0
    bess_mwh = 0
    # Spin Logic
    n_calc = n_base
    while True:
        total_mw = n_calc * unit_size_site
        total_step_cap = total_mw * (gen_step_cap / 100.0)
        if total_step_cap >= req_step_mw and (total_mw >= p_gross_gen_req):
            break
        n_calc += 1
    n_spin = max(n_calc - n_base, min_redundancy)
    # Ensure N+X covers load
    if ((n_base + n_spin - min_redundancy) * unit_size_site) < p_gross_gen_req:
        n_spin += 1

n_online = n_base + n_spin
n_maint = math.ceil(n_online * maint_unav)
n_total = n_online + n_maint
installed_cap_site = n_total * unit_size_site

# C. PHYSICS & ENV
attenuation_geo = 20 * math.log10(dist_neighbor_m / 1.0)
noise_total = source_noise_dba - attenuation_geo + (10 * math.log10(n_online))
req_attenuation = max(0, noise_total - noise_limit)

min_stack = nearby_building_h_m * 1.5
nox_tpy = (raw_nox * (p_gross_gen_req * 1341) * 8760) / 907185
req_scr = nox_tpy > limit_nox_tpy

# Urea
if req_scr:
    urea_yr_l = (p_gross_gen_req * 1.5) * 8760
    urea_store_l = (urea_yr_l / 365) * urea_days
    num_tanks = math.ceil(urea_store_l / 30000)
    trucks_yr = math.ceil(urea_yr_l / 25000)
else:
    urea_yr_l = 0; urea_store_l = 0; num_tanks = 0; trucks_yr = 0

# D. AREAS (CORRECTED)
area_gen = n_total * (140.0 if is_rice else 200.0)
area_bess = bess_mwh * 25.0
area_sub = 5500.0 if voltage_kv >= 115 else 2500.0
area_gas = 800.0
area_scr = (400.0 + (num_tanks * 50)) if req_scr else 0.0

area_subtotal = area_gen + area_bess + area_sub + area_gas + area_scr
area_tot_m2 = area_subtotal * 1.2 # +20% roads
area_ha = area_tot_m2 / 10000.0

# E. ENGINEERING ECONOMICS (EFFICIENCY)
lf = (p_gross_gen_req / (n_online * unit_size_site))
eff_factor = 1.0 - (0.6 * (1.0 - lf)**3) if is_rice and lf < 1.0 else 1.0
real_eff = eff_gen_base * max(0.5, eff_factor)
heat_rate_btu = (3412.14 / (real_eff/100))

# F. BUSINESS CASE (STEP 8 LOGIC)
# 1. LCOE Bridge Power
gen_mwh_yr = p_gross_gen_req * 8760

# --- FIX UNIT ERROR: (Btu/kWh / 1e6) * $/MMBtu = $/kWh. Need $/MWh -> Multiply by 1000 ---
fuel_cost_mwh = ((heat_rate_btu / 1e6) * fuel_price) * 1000 

# Fixed Cost spread over generation (Capacity Charge)
fixed_cost_yr = installed_cap_site * cap_charge * 12
fixed_cost_mwh = fixed_cost_yr / gen_mwh_yr
# Total LCOE
lcoe_bridge = fuel_cost_mwh + fixed_cost_mwh + var_om
# Utility Reference
lcoe_utility = grid_rate_kwh * 1000

# 2. Buyout Option (Dynamic Year)
buyout_cost_m = (installed_cap_site * 1000 * new_asset_capex * (buyout_pct/100)) / 1e6
new_plant_cost_m = (installed_cap_site * 1000 * new_asset_capex) / 1e6
savings_buyout = new_plant_cost_m - buyout_cost_m

# 3. VPP Revenue (Future Standby)
rev_arb = installed_cap_site * vpp_arb_spread * 365 # Assume 1 cycle/day full capacity
rev_cap = installed_cap_site * vpp_cap_pay
rev_anc = installed_cap_site * vpp_anc_pay
total_vpp_yr_m = (rev_arb + rev_cap + rev_anc) / 1e6

# ==============================================================================
# 3. DASHBOARD
# ==============================================================================

# Conversions for Display
if is_imperial:
    d_area_l = area_ha * 2.471; u_al = "Acres"; d_area_s = area_tot_m2 * 10.764; u_as = "ft²"
    d_urea = urea_yr_l * 0.264; u_vol = "gal"; d_mass = nox_tpy * 1.102; u_mass = "Tons"
    d_stack = min_stack * 3.28; u_dst = "ft"
else:
    d_area_l = area_ha; u_al = "Ha"; d_area_s = area_tot_m2; u_as = "m²"
    d_urea = urea_yr_l; u_vol = "L"; d_mass = nox_tpy; u_mass = "Tonnes"
    d_stack = min_stack; u_dst = "m"

# --- TOP KPIs (RESTORED HEAT RATE) ---
c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("IT Capacity", f"{p_max} MW", f"Gross: {p_gross_gen_req:.1f} MW")
c2.metric("Generators", f"{n_total} Units", f"N+S+M: {n_base}+{n_spin}+{n_maint}")
c3.metric("Heat Rate", f"{heat_rate_btu:,.0f} Btu/kWh", f"Eff: {real_eff:.1f}%")
c4.metric("LCOE (Bridge)", f"${lcoe_bridge:.2f}/MWh", f"Grid: ${lcoe_utility:.2f}")
c5.metric("Compliance", "FAIL" if (req_scr or req_attenuation > 0) else "OK", f"NOx: {d_mass:.0f} {u_mass}")

st.divider()

t1, t2, t3, t4 = st.tabs([t["tab_tech"], t["tab_area"], t["tab_env"], t["tab_biz"]])

with t1:
    c_a, c_b = st.columns(2)
    with c_a:
        st.subheader("Power Balance")
        df_bal = pd.DataFrame({
            "Item": ["Critical IT", "Auxiliaries", "Dist. Losses", "Gen. Parasitics", "TOTAL GROSS"],
            "MW": [p_max, p_aux_mw, p_dist_loss, (p_gross_gen_req - p_net_gen_req), p_gross_gen_req]
        })
        st.dataframe(df_bal.style.format({"MW": "{:.2f}"}), use_container_width=True)
    with c_b:
        st.subheader("Fleet Strategy")
        st.write(f"**Availability:** {avail_target}% (Redundancy N+{min_redundancy})")
        st.write(f"**Units:** {n_base} Run + {n_spin} Spin + {n_maint} Maint")
        if use_bess:
            st.success(f"**BESS Enabled:** {bess_mw:.1f} MW / {bess_mwh:.1f} MWh")
        else:
            st.warning("No BESS: High Spinning Reserve required.")

with t2:
    st.subheader("Footprint Estimate")
    df_area = pd.DataFrame({
        "Zone": ["Generation", "BESS", "Substation", "Gas ERM", "SCR/Urea", "Roads"],
        f"Area ({u_as})": [
            area_gen * (10.764 if is_imperial else 1),
            area_bess * (10.764 if is_imperial else 1),
            area_sub * (10.764 if is_imperial else 1),
            area_gas * (10.764 if is_imperial else 1),
            area_scr * (10.764 if is_imperial else 1),
            (area_tot_m2 - area_subtotal) * (10.764 if is_imperial else 1)
        ]
    })
    st.dataframe(df_area.style.format({f"Area ({u_as})": "{:,.0f}"}), use_container_width=True)
    st.metric("Total Land", f"{d_area_l:.2f} {u_al}")

with t3:
    c_e1, c_e2 = st.columns(2)
    with c_e1:
        st.subheader("Acoustics")
        st.write(f"Source: {source_noise_dba} dBA | Fleet (+{10*math.log10(n_online):.1f} dB)")
        st.write(f"Receiver: {noise_total:.1f} dBA | Limit: {noise_limit}")
        if req_attenuation > 0: st.error(f"Violation: -{req_attenuation:.1f} dB barrier needed")
        else: st.success("Noise Compliant")
    with c_e2:
        st.subheader("Emissions & Urea")
        st.write(f"NOx: {d_mass:,.0f} {u_mass}/yr")
        if req_scr:
            st.warning("SCR Required")
            st.write(f"Urea: {d_urea:,.0f} {u_vol}/yr")
            st.write(f"Tanks: {num_tanks}x 30kL | Trucks: {trucks_yr}/yr")

with t4:
    st.header("💰 Business Case & Financial Strategy (Step 8)")
    
    # 1. LCOE Comparison
    st.subheader("1. Bridge Power LCOE vs Utility")
    
    col_l1, col_l2 = st.columns([1, 2])
    
    with col_l1:
        st.metric("Net Heat Rate", f"{heat_rate_btu:,.0f} Btu/kWh")
        st.markdown(f"""
        **Bridge LCOE:** :red[**${lcoe_bridge:.2f}**] / MWh
        * Fuel: ${fuel_cost_mwh:.2f}
        * Capacity (Lease): ${fixed_cost_mwh:.2f}
        * O&M (Var): ${var_om:.2f}
        
        **Utility LCOE:** :green[**${lcoe_utility:.2f}**] / MWh
        """)
        
        delta = lcoe_bridge - lcoe_utility
        if delta > 0: st.info(f"Premium: +${delta:.2f}/MWh (Cost of Speed)")
        else: st.success(f"Savings: -${abs(delta):.2f}/MWh (Competitive!)")

    with col_l2:
        lcoe_data = pd.DataFrame({
            "Cost Component": ["Fuel", "Capacity (Lease)", "Variable O&M", "Utility Tariff"],
            "$/MWh": [fuel_cost_mwh, fixed_cost_mwh, var_om, lcoe_utility],
            "Type": ["Bridge", "Bridge", "Bridge", "Utility"]
        })
        fig_lcoe = px.bar(lcoe_data, x="Type", y="$/MWh", color="Cost Component", title="LCOE Composition", text_auto='.1f')
        st.plotly_chart(fig_lcoe, use_container_width=True)

    st.divider()

    # 2. Buyout & Transition
    st.subheader(f"2. Asset Transfer Strategy (Year {contract_years})")
    c_b1, c_b2 = st.columns(2)
    with c_b1:
        st.metric("Total Installed Fleet", f"{installed_cap_site:.1f} MW")
        st.metric("Est. Buyout Price (Residual)", f"${buyout_cost_m:.1f} M", f"{buyout_pct}% of New")
    with c_b2:
        st.metric("Cost of New Plant (Ref)", f"${new_plant_cost_m:.1f} M")
        st.success(f"**Potential CAPEX Avoided:** ${savings_buyout:.1f} Million")

    st.divider()

    # 3. Revenue Stacking
    st.subheader("3. Future VPP Revenue Potential (Standby Mode)")
    st.markdown("Annual revenue estimation from Grid Services once connected to Utility.")
    
    rev_data = pd.DataFrame({
        "Service": ["Energy Arbitrage", "Capacity Payments", "Ancillary (Freq/Inertia)"],
        "Revenue ($M/yr)": [rev_arb/1e6, rev_cap/1e6, rev_anc/1e6]
    })
    
    c_r1, c_r2 = st.columns([2, 1])
    with c_r1:
        fig_rev = px.bar(rev_data, x="Service", y="Revenue ($M/yr)", color="Service", title=f"Total VPP Revenue: ${total_vpp_yr_m:.1f}M / year")
        st.plotly_chart(fig_rev, use_container_width=True)
    with c_r2:
        st.info("Logic: Assets operate as Virtual Power Plant (VPP) when not in emergency, offsetting maintenance costs.")

# --- FOOTER ---
st.markdown("---")
st.caption("Bridge Power Engine V9.6 | Fixed Fuel Calculation & Heat Rate Display")
