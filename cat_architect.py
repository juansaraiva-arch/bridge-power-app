import streamlit as st
import pandas as pd
import numpy as np
import math
import plotly.express as px
import plotly.graph_objects as go
import copy

# --- PAGE CONFIGURATION ---
st.set_page_config(page_title="CAT Power Master Architect", page_icon="🏗️", layout="wide")

# ==============================================================================
# 1. THE DATA & PHYSICS ENGINE (LOGIC LAYER)
# ==============================================================================

# Data Library
leps_gas_library = {
    "XGC1900": {"type": "High Speed", "iso_mw": 1.9, "eff": 0.392, "hr": 8780, "step": 25.0, "nox": 0.5, "cost_kw": 775.0, "inst_kw": 300.0, "xd": 0.14},
    "G3520FR": {"type": "High Speed", "iso_mw": 2.5, "eff": 0.386, "hr": 8836, "step": 40.0, "nox": 0.5, "cost_kw": 575.0, "inst_kw": 650.0, "xd": 0.14},
    "G3520K":  {"type": "High Speed", "iso_mw": 2.4, "eff": 0.453, "hr": 7638, "step": 15.0, "nox": 0.3, "cost_kw": 575.0, "inst_kw": 650.0, "xd": 0.13},
    "CG260-16":{"type": "High Speed", "iso_mw": 3.96,"eff": 0.434, "hr": 7860, "step": 10.0, "nox": 0.5, "cost_kw": 675.0, "inst_kw": 1100.0,"xd": 0.15},
    "Titan 130":{"type": "Turbine",   "iso_mw": 16.5,"eff": 0.354, "hr": 9630, "step": 15.0, "nox": 0.6, "cost_kw": 775.0, "inst_kw": 1000.0,"xd": 0.18},
    "G20CM34": {"type": "Med Speed",  "iso_mw": 9.76,"eff": 0.475, "hr": 7480, "step": 10.0, "nox": 0.5, "cost_kw": 700.0, "inst_kw": 1250.0,"xd": 0.16}
}

def calculate_kpis(inputs):
    """
    Pure function: Takes a dictionary of inputs, returns a dictionary of calculated KPIs.
    Does NOT render Streamlit widgets.
    """
    res = {} # Results dictionary
    
    # 1. Unpack Inputs
    model_key = inputs.get("model", "G3520K")
    spec = leps_gas_library[model_key]
    
    p_it = inputs.get("p_it", 100.0)
    dc_aux = inputs.get("dc_aux", 0.05)
    dist_loss = inputs.get("dist_loss", 0.01)
    gen_parasitic = inputs.get("gen_parasitic", 0.025)
    
    use_chp = inputs.get("use_chp", True)
    cooling_method = inputs.get("cooling_method", "Water Cooled")
    pue_input = inputs.get("pue_input", 1.45)
    
    # Derates
    site_temp = inputs.get("site_temp", 35)
    site_alt = inputs.get("site_alt", 100)
    mn = inputs.get("mn", 80)
    
    # Calc Derate
    loss_temp = max(0, (site_temp - 25) * 0.01)
    loss_alt = max(0, (site_alt - 100) * 0.0001)
    loss_mn = max(0, (75 - mn) * 0.005)
    derate = 1.0 - (loss_temp + loss_alt + loss_mn)
    
    # 2. Power Balance
    if use_chp:
        p_cooling_elec = p_it * 0.03 # Absorption pumps
        p_net = p_it * (1 + dc_aux) + p_cooling_elec
        pue = p_net / p_it
    else:
        p_net = p_it * pue_input
        pue = pue_input
        
    p_gross = (p_net * (1 + dist_loss)) / (1 - gen_parasitic)
    
    # 3. Fleet Sizing
    unit_site_cap = spec['iso_mw'] * derate
    if inputs.get("use_bess", False):
        n_run = math.ceil(p_gross / (unit_site_cap * 0.95))
    else:
        n_run = math.ceil(p_gross / (unit_site_cap * 0.90))
        
    n_total = n_run + inputs.get("n_maint", 1) + inputs.get("n_reserve", 1)
    installed_mw = n_total * unit_site_cap
    
    # 4. CAPEX
    capex_gen = n_total * 1000 * inputs.get("cost_kw", spec['cost_kw'])
    capex_inst = n_total * 1000 * inputs.get("inst_kw", spec['inst_kw'])
    
    capex_bess = 0
    if inputs.get("use_bess", False):
        mw_bess = unit_site_cap
        mwh_bess = mw_bess * 2
        capex_bess = (mw_bess * 1000 * 120) + (mwh_bess * 1000 * 280)
        
    capex_log = 0
    if inputs.get("has_lng", False):
        mmbtu_hr = p_gross * (spec['hr']/1000) # approx
        vol_day = (mmbtu_hr * 24) * 12.5
        n_tanks = math.ceil((vol_day * 5) / 10000) # 5 days autonomy
        capex_log = n_tanks * 50000 # tank mob cost placeholder
        
    capex_chp = 0
    if use_chp:
        capex_chp = capex_gen * 0.20
        
    total_capex = (capex_gen + capex_inst + capex_bess + capex_log + capex_chp) / 1e6 # in M USD
    
    # 5. OPEX & LCOE
    hr_site = spec['hr'] # Simplification for engine
    if not use_chp: hr_site = hr_site # No CHP benefit on HR directly here, but PUE logic handles fuel load
    
    fuel_mmbtu_yr = p_gross * (hr_site/1000) * 8760
    
    gas_price = inputs.get("gas_price", 6.5)
    if inputs.get("is_lng_primary", False): gas_price += inputs.get("vp_premium", 4.0)
    
    cost_fuel = fuel_mmbtu_yr * gas_price
    cost_om = (p_net * 8760 * inputs.get("om_var", 12.0))
    
    # Financials
    wacc = inputs.get("wacc", 0.08)
    years = inputs.get("years", 20)
    crf = (wacc * (1 + wacc)**years) / ((1 + wacc)**years - 1)
    
    capex_ann = total_capex * 1e6 * crf
    total_ann_cost = cost_fuel + cost_om + capex_ann
    
    lcoe = total_ann_cost / (p_net * 8760 * 1000) # $/kWh
    
    # Populate Results
    res['p_net'] = p_net
    res['p_gross'] = p_gross
    res['n_total'] = n_total
    res['installed_mw'] = installed_mw
    res['pue'] = pue
    res['total_capex'] = total_capex
    res['lcoe'] = lcoe
    res['fuel_cost'] = cost_fuel
    res['capex_ann'] = capex_ann
    res['model'] = model_key
    res['hr_net'] = (fuel_mmbtu_yr * 1e6) / (p_net * 8760 * 1000) # Net Heat Rate Btu/kWh
    
    return res

# ==============================================================================
# 2. STATE MANAGEMENT (PROJECT & SCENARIOS)
# ==============================================================================

# Initialize Session State
if 'project' not in st.session_state:
    st.session_state['project'] = {
        "name": "New Data Center Project",
        "scenarios": {
            "Base Case": {} # Will hold inputs
        }
    }
    st.session_state['active_scenario'] = "Base Case"

# Default Inputs (The "Truth")
defaults = {
    "model": "G3520K", "p_it": 100.0, "dc_aux": 0.05, "dist_loss": 0.01, "gen_parasitic": 0.025,
    "use_chp": True, "cooling_method": "Water Cooled", "pue_input": 1.45,
    "use_bess": True, "has_lng": True, "is_lng_primary": False, "vp_premium": 4.0,
    "site_temp": 35, "site_alt": 100, "mn": 80,
    "n_maint": 1, "n_reserve": 1,
    "cost_kw": 575.0, "inst_kw": 650.0,
    "gas_price": 6.5, "om_var": 12.0, "wacc": 0.08, "years": 20,
    "target_lcoe": 0.11
}

# Helper: Get current input value (Cascade: Scenario -> Base -> Default)
def get_val(key):
    scen = st.session_state['active_scenario']
    proj = st.session_state['project']
    
    # 1. Look in active scenario
    if key in proj['scenarios'][scen]:
        return proj['scenarios'][scen][key]
    
    # 2. If active is not Base, look in Base
    if scen != "Base Case" and key in proj['scenarios']["Base Case"]:
        return proj['scenarios']["Base Case"][key]
        
    # 3. Return Default
    return defaults[key]

# Helper: Set input value
def set_val(key, value):
    scen = st.session_state['active_scenario']
    st.session_state['project']['scenarios'][scen][key] = value

# ==============================================================================
# 3. SIDEBAR (CONTROLLER)
# ==============================================================================

with st.sidebar:
    st.header("🎛️ Project Manager")
    
    # Project Name
    proj_name = st.text_input("Project Name", st.session_state['project']['name'])
    st.session_state['project']['name'] = proj_name
    
    # Scenario Management
    st.divider()
    st.subheader("Scenarios")
    
    # Selector
    scenario_list = list(st.session_state['project']['scenarios'].keys())
    active_scen = st.selectbox("Active Scenario", scenario_list, index=scenario_list.index(st.session_state['active_scenario']))
    st.session_state['active_scenario'] = active_scen
    
    # Add New Scenario
    new_scen_name = st.text_input("New Scenario Name", placeholder="e.g. High Gas Price")
    if st.button("Create Scenario"):
        if new_scen_name and new_scen_name not in scenario_list:
            # Copy base case inputs to new scenario
            st.session_state['project']['scenarios'][new_scen_name] = st.session_state['project']['scenarios']["Base Case"].copy()
            st.success(f"Created {new_scen_name}")
            st.rerun()
            
    st.info(f"Editing: **{active_scen}**")
    if active_scen != "Base Case":
        st.caption("Changes made here are specific to this scenario.")

# ==============================================================================
# 4. MAIN INTERFACE
# ==============================================================================

tab_edit, tab_compare, tab_report = st.tabs(["📝 Edit Inputs", "📊 Compare Scenarios", "📄 Report View"])

# --- TAB 1: EDIT INPUTS ---
with tab_edit:
    st.title(f"🛠️ {st.session_state['project']['name']} - {st.session_state['active_scenario']}")
    
    c1, c2, c3 = st.columns(3)
    
    with c1:
        st.subheader("1. Load & Tech")
        
        # Tech Selection
        curr_model = get_val("model")
        new_model = st.selectbox("Generator Model", list(leps_gas_library.keys()), index=list(leps_gas_library.keys()).index(curr_model))
        if new_model != curr_model: set_val("model", new_model)
        
        # IT Load
        curr_it = get_val("p_it")
        new_it = st.number_input("Critical IT Load (MW)", 1.0, 1000.0, float(curr_it))
        if new_it != curr_it: set_val("p_it", new_it)
        
        # CHP
        curr_chp = get_val("use_chp")
        new_chp = st.checkbox("Include Tri-Gen (CHP)", curr_chp)
        if new_chp != curr_chp: set_val("use_chp", new_chp)
        
        if not new_chp:
            curr_pue = get_val("pue_input")
            new_pue = st.number_input("Target PUE (Elec)", 1.1, 2.0, float(curr_pue))
            if new_pue != curr_pue: set_val("pue_input", new_pue)

    with c2:
        st.subheader("2. Infrastructure")
        
        # BESS
        curr_bess = get_val("use_bess")
        new_bess = st.checkbox("Include BESS", curr_bess)
        if new_bess != curr_bess: set_val("use_bess", new_bess)
        
        # LNG
        curr_lng = get_val("has_lng")
        new_lng = st.checkbox("Include LNG Storage", curr_lng)
        if new_lng != curr_lng: set_val("has_lng", new_lng)
        
        # Reliability
        c_r1, c_r2 = st.columns(2)
        curr_res = get_val("n_reserve")
        new_res = c_r1.number_input("N+ Reserve", 0, 5, int(curr_res))
        if new_res != curr_res: set_val("n_reserve", new_res)

    with c3:
        st.subheader("3. Financials")
        
        # Gas Price
        curr_gas = get_val("gas_price")
        new_gas = st.number_input("Gas Price ($/MMBtu)", 1.0, 50.0, float(curr_gas))
        if new_gas != curr_gas: set_val("gas_price", new_gas)
        
        # CAPEX Override
        st.caption("CAPEX Overrides ($/kW)")
        curr_ckw = get_val("cost_kw")
        new_ckw = st.number_input("Gen Equipment", 100.0, 2000.0, float(curr_ckw))
        if new_ckw != curr_ckw: set_val("cost_kw", new_ckw)
        
        # Target
        curr_tgt = get_val("target_lcoe")
        new_tgt = st.number_input("Target LCOE ($/kWh)", 0.05, 0.50, float(curr_tgt))
        if new_tgt != curr_tgt: set_val("target_lcoe", new_tgt)

    # LIVE PREVIEW OF CURRENT SCENARIO
    st.divider()
    
    # Compile current inputs for the engine
    current_inputs = {k: get_val(k) for k in defaults.keys()}
    results = calculate_kpis(current_inputs)
    
    st.markdown("#### ⚡ Live Scenario Preview")
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("LCOE", f"${results['lcoe']:.4f}/kWh", help="Levelized Cost of Energy")
    k2.metric("CAPEX", f"${results['total_capex']:.1f} M")
    k3.metric("Fleet", f"{results['n_total']}x {results['model']}")
    k4.metric("Net Capacity", f"{results['p_net']:.1f} MW")

# --- TAB 2: COMPARE SCENARIOS ---
with tab_compare:
    st.title("📊 Scenario Comparator")
    
    if len(st.session_state['project']['scenarios']) < 2:
        st.info("Create more scenarios in the sidebar to compare them.")
    else:
        # Build Comparison Data
        comp_data = []
        metrics_to_show = ["lcoe", "total_capex", "n_total", "model", "pue", "hr_net", "fuel_cost"]
        
        for sc_name, sc_inputs_partial in st.session_state['project']['scenarios'].items():
            # Merge defaults with scenario overrides
            full_inputs = defaults.copy()
            full_inputs.update(sc_inputs_partial)
            
            # Run Engine
            res = calculate_kpis(full_inputs)
            
            row = {"Scenario": sc_name}
            row.update({k: res[k] for k in metrics_to_show})
            comp_data.append(row)
            
        df_comp = pd.DataFrame(comp_data)
        df_comp.set_index("Scenario", inplace=True)
        
        # Transpose for "Side-by-Side" view
        df_display = df_comp.T
        
        # Formatting
        st.dataframe(
            df_display.style.format("{:.4f}", subset=pd.IndexSlice[["lcoe"], :])
                             .format("${:,.1f} M", subset=pd.IndexSlice[["total_capex"], :])
                             .format("{:,.0f}", subset=pd.IndexSlice[["hr_net", "fuel_cost"], :])
                             .highlight_min(axis=1, color="lightgreen", subset=["lcoe", "total_capex", "pue"])
                             .highlight_max(axis=1, color="lightpink", subset=["lcoe", "total_capex", "pue"]),
            use_container_width=True,
            height=400
        )
        
        # Visual Comparison (Bar Chart)
        st.subheader("LCOE Comparison")
        fig_comp = px.bar(df_comp, x=df_comp.index, y="lcoe", color="model", text_auto=".4f", title="LCOE by Scenario")
        st.plotly_chart(fig_comp, use_container_width=True)

# --- TAB 3: REPORT ---
with tab_report:
    st.title("📄 Executive Summary")
    st.write("This view allows exporting the analysis.")
    
    # Generate a simple text summary
    base_res = calculate_kpis(defaults)
    best_scen = None
    min_lcoe = 999
    
    for sc_name, sc_inputs_partial in st.session_state['project']['scenarios'].items():
        full_inputs = defaults.copy()
        full_inputs.update(sc_inputs_partial)
        res = calculate_kpis(full_inputs)
        if res['lcoe'] < min_lcoe:
            min_lcoe = res['lcoe']
            best_scen = sc_name
            
    st.success(f"**Recommendation:** The best performing scenario is **{best_scen}** with an LCOE of **${min_lcoe:.4f}/kWh**.")
    
    st.markdown("""
    **Next Steps:**
    1.  Validate gas price assumptions for the selected scenario.
    2.  Confirm site area availability for the required footprint.
    3.  Proceed to Stage 2 Engineering.
    """)
    
    if st.button("🖨️ Export to Excel (Simulation)"):
        st.toast("Report generated! (Functionality placeholder)")

# --- FOOTER ---
st.markdown("---")
st.caption("CAT Power Master Architect | v1.0 | Project & Scenario Management System")