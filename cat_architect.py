import streamlit as st
import pandas as pd
import numpy as np
import math
import plotly.express as px
import json
import io

# --- PAGE CONFIGURATION ---
st.set_page_config(page_title="CAT Power Master Architect v2.0", page_icon="🏗️", layout="wide")

# ==============================================================================
# 1. THE DATA & PHYSICS ENGINE (LOGIC LAYER)
# ==============================================================================

leps_gas_library = {
    "XGC1900": {"type": "High Speed", "iso_mw": 1.9, "eff": 0.392, "hr": 8780, "step": 25.0, "nox": 0.5, "cost_kw": 775.0, "inst_kw": 300.0, "xd": 0.14},
    "G3520FR": {"type": "High Speed", "iso_mw": 2.5, "eff": 0.386, "hr": 8836, "step": 40.0, "nox": 0.5, "cost_kw": 575.0, "inst_kw": 650.0, "xd": 0.14},
    "G3520K":  {"type": "High Speed", "iso_mw": 2.4, "eff": 0.453, "hr": 7638, "step": 15.0, "nox": 0.3, "cost_kw": 575.0, "inst_kw": 650.0, "xd": 0.13},
    "CG260-16":{"type": "High Speed", "iso_mw": 3.96,"eff": 0.434, "hr": 7860, "step": 10.0, "nox": 0.5, "cost_kw": 675.0, "inst_kw": 1100.0,"xd": 0.15},
    "Titan 130":{"type": "Turbine",   "iso_mw": 16.5,"eff": 0.354, "hr": 9630, "step": 15.0, "nox": 0.6, "cost_kw": 775.0, "inst_kw": 1000.0,"xd": 0.18},
    "G20CM34": {"type": "Med Speed",  "iso_mw": 9.76,"eff": 0.475, "hr": 7480, "step": 10.0, "nox": 0.5, "cost_kw": 700.0, "inst_kw": 1250.0,"xd": 0.16}
}

# FULL ENGINE from Light Version logic
def calculate_kpis(inputs):
    res = {}
    
    # --- A. Unpack & Setup ---
    model_key = inputs.get("model", "G3520K")
    spec = leps_gas_library[model_key]
    
    # Load & PUE
    p_it = inputs.get("p_it", 100.0)
    dc_aux = inputs.get("dc_aux", 0.05)
    use_chp = inputs.get("use_chp", True)
    
    if use_chp:
        p_cooling_elec = p_it * 0.03 
        p_net = p_it * (1 + dc_aux) + p_cooling_elec
        pue = p_net / p_it
    else:
        p_net = p_it * inputs.get("pue_input", 1.45)
        pue = inputs.get("pue_input", 1.45)
        
    dist_loss = inputs.get("dist_loss", 0.01)
    gen_parasitic = inputs.get("gen_parasitic", 0.025)
    p_gross = (p_net * (1 + dist_loss)) / (1 - gen_parasitic)
    
    # --- B. Derates ---
    loss_temp = max(0, (inputs.get("site_temp", 35) - 25) * 0.01)
    loss_alt = max(0, (inputs.get("site_alt", 100) - 100) * 0.0001)
    loss_mn = max(0, (75 - inputs.get("mn", 80)) * 0.005)
    derate = 1.0 - (loss_temp + loss_alt + loss_mn)
    unit_site_cap = spec['iso_mw'] * derate
    
    # --- C. Fleet Sizing ---
    target_lf = 0.95 if inputs.get("use_bess", True) else 0.90
    n_run = math.ceil(p_gross / (unit_site_cap * target_lf))
    n_total = n_run + inputs.get("n_maint", 1) + inputs.get("n_reserve", 1)
    installed_mw = n_total * unit_site_cap
    
    # --- D. CAPEX ---
    # Generators
    capex_gen = n_total * 1000 * inputs.get("cost_kw", spec['cost_kw'])
    capex_inst = n_total * 1000 * inputs.get("inst_kw", spec['inst_kw'])
    
    # BESS
    capex_bess = 0
    if inputs.get("use_bess", True):
        mw_bess = unit_site_cap * (1 + inputs.get("n_bess_red", 0))
        mwh_bess = mw_bess * 2
        capex_bess = (mw_bess * 1000 * inputs.get("cost_bess_inv", 120)) + (mwh_bess * 1000 * inputs.get("cost_bess_kwh", 280))
        
    # Logistics (LNG)
    capex_log = 0
    fuel_mmbtu_hr = p_gross * (spec['hr']/1000)
    if inputs.get("has_lng", True):
        vol_day = (fuel_mmbtu_hr * 24) * 12.5 # approx gal/mmbtu
        n_tanks = math.ceil((vol_day * inputs.get("lng_days", 5)) / 10000)
        capex_log = n_tanks * 50000 
        
    # Pipeline
    if not inputs.get("is_lng_primary", False):
        capex_log += (inputs.get("dist_pipe", 1000) * 200) # Simple pipe cost estimate
        
    # Emissions
    capex_emis = 0
    total_bhp = p_gross * 1341
    nox_tpy = (spec['nox'] * total_bhp * 8760) / 907185
    limit_nox = 250.0 if inputs.get("reg_zone") == "USA - EPA Major" else 9999
    if nox_tpy > limit_nox:
        capex_emis = installed_mw * 1000 * 60 # SCR cost
        
    # Tri-Gen
    capex_chp = 0
    if use_chp:
        capex_chp = capex_gen * 0.20
        
    total_capex = (capex_gen + capex_inst + capex_bess + capex_log + capex_emis + capex_chp) / 1e6
    
    # --- E. OPEX & Financials ---
    gas_price = inputs.get("gas_price", 6.5)
    if inputs.get("is_lng_primary", False): gas_price += inputs.get("vp_premium", 4.0)
    
    # Heat Rate Adjustment (Simple curve)
    lf_real = p_gross / (n_run * unit_site_cap)
    eff_factor = 1.0 if lf_real > 0.75 else (0.85 + (0.6*(lf_real-0.5)))
    hr_site = spec['hr'] / max(0.5, eff_factor)
    
    fuel_cost_yr = p_gross * (hr_site/1000) * gas_price * 8760
    om_cost_yr = p_net * 8760 * inputs.get("om_var", 12.0)
    if inputs.get("use_bess", True): om_cost_yr += (unit_site_cap * 1000 * 10) # BESS fixed O&M
    
    # Repowering
    repower_ann = 0
    if inputs.get("use_bess", True):
        # Simplified annualized repowering
        repower_ann = (capex_bess * 0.6) / 10 # Battery replacement annuity approx
        
    wacc = inputs.get("wacc", 0.08)
    years = inputs.get("years", 20)
    crf = (wacc * (1 + wacc)**years) / ((1 + wacc)**years - 1)
    
    capex_ann = total_capex * 1e6 * crf
    total_ann_cost = fuel_cost_yr + om_cost_yr + capex_ann + repower_ann
    
    lcoe = total_ann_cost / (p_net * 8760 * 1000)
    
    # --- F. Pack Results ---
    res = {
        "model": model_key,
        "n_total": n_total,
        "p_net": p_net,
        "pue": pue,
        "total_capex": total_capex,
        "lcoe": lcoe,
        "fuel_cost": fuel_cost_yr,
        "om_cost": om_cost_yr,
        "capex_ann": capex_ann,
        "hr_net": (p_gross * (hr_site/1000) * 1e6) / (p_net), # Btu/kWh net
        "nox_tpy": nox_tpy,
        "avail_est": 1.0 - (math.pow(0.02, inputs.get("n_reserve", 1))) # Rough prob
    }
    return res

# ==============================================================================
# 2. STATE & PERSISTENCE (DATABASE LAYER)
# ==============================================================================

# Default Master Input List (Expanded from v68)
defaults = {
    # 1. Load & Tech
    "model": "G3520K", "p_it": 100.0, "dc_aux": 0.05, "use_chp": True, "pue_input": 1.45,
    "volt_kv": 13.8, "step_req": 40.0,
    # 2. Site
    "site_temp": 35, "site_alt": 100, "mn": 80, "reg_zone": "LatAm / No-Reg",
    "dist_neighbor": 100.0,
    # 3. Infra
    "use_bess": True, "n_bess_red": 0, "has_lng": True, "is_lng_primary": False,
    "dist_pipe": 1000.0, "lng_days": 5, "n_maint": 1, "n_reserve": 1,
    # 4. Costs
    "cost_kw": 575.0, "inst_kw": 650.0, "cost_bess_kwh": 280.0, "cost_bess_inv": 120.0,
    "gas_price": 6.5, "vp_premium": 4.0, "om_var": 12.0, 
    "wacc": 0.08, "years": 20, "target_lcoe": 0.11
}

if 'project' not in st.session_state:
    st.session_state['project'] = {
        "name": "Project Alpha",
        "created_at": str(pd.Timestamp.now()),
        "scenarios": { "Base Case": defaults.copy() }
    }
    st.session_state['active_scenario'] = "Base Case"

def get_val(key):
    scen = st.session_state['active_scenario']
    return st.session_state['project']['scenarios'][scen].get(key, defaults[key])

def set_val(key, value):
    scen = st.session_state['active_scenario']
    st.session_state['project']['scenarios'][scen][key] = value

# ==============================================================================
# 3. SIDEBAR (CONTROLLER)
# ==============================================================================

with st.sidebar:
    st.title("CAT Architect v2.0")
    
    # --- Persistence Section ---
    with st.expander("💾 Project File (Database)", expanded=False):
        # Export
        proj_data = json.dumps(st.session_state['project'], indent=2)
        st.download_button(
            label="Download Project (.json)",
            data=proj_data,
            file_name=f"{st.session_state['project']['name']}.json",
            mime="application/json",
        )
        
        # Import
        uploaded_file = st.file_uploader("Load Project", type=["json"])
        if uploaded_file is not None:
            try:
                data = json.load(uploaded_file)
                st.session_state['project'] = data
                st.session_state['active_scenario'] = list(data['scenarios'].keys())[0]
                st.success("Project Loaded!")
                st.rerun()
            except Exception as e:
                st.error(f"Error loading file: {e}")

    # --- Scenario Manager ---
    st.divider()
    st.text_input("Project Name", value=st.session_state['project']['name'], key="proj_name_input", on_change=lambda: st.session_state['project'].update({"name": st.session_state.proj_name_input}))
    
    scenarios = list(st.session_state['project']['scenarios'].keys())
    active = st.selectbox("Active Scenario", scenarios, index=scenarios.index(st.session_state['active_scenario']))
    st.session_state['active_scenario'] = active
    
    new_scen = st.text_input("New Scenario", placeholder="Name...")
    if st.button("➕ Create Scenario"):
        if new_scen and new_scen not in scenarios:
            # Clone current active scenario
            st.session_state['project']['scenarios'][new_scen] = st.session_state['project']['scenarios'][active].copy()
            st.success(f"Created {new_scen}")
            st.rerun()

# ==============================================================================
# 4. MAIN INTERFACE
# ==============================================================================

tab_edit, tab_comp, tab_rep = st.tabs(["📝 Scenario Editor", "📊 Comparative Analysis", "📑 Report"])

# --- TAB 1: FULL EDITOR ---
with tab_edit:
    st.subheader(f"Editing: {st.session_state['active_scenario']}")
    
    # 4 Column Layout for Density
    c1, c2, c3, c4 = st.columns(4)
    
    # --- COLUMN 1: LOAD & TECH ---
    with c1:
        st.markdown("### ⚡ Load & Tech")
        
        # Model
        curr = get_val("model")
        sel = st.selectbox("Generator", list(leps_gas_library.keys()), index=list(leps_gas_library.keys()).index(curr))
        if sel != curr: set_val("model", sel)
        
        # Power
        curr = get_val("p_it")
        val = st.number_input("IT Load (MW)", 1.0, 1000.0, float(curr))
        if val != curr: set_val("p_it", val)
        
        curr = get_val("dc_aux")
        val = st.number_input("DC Aux (%)", 0.0, 20.0, float(curr)*100) / 100
        if val != curr: set_val("dc_aux", val)
        
        # Cooling
        curr = get_val("use_chp")
        val = st.checkbox("Tri-Gen (CHP)", curr)
        if val != curr: set_val("use_chp", val)
        
        if not val:
            curr = get_val("pue_input")
            v2 = st.number_input("Target PUE", 1.0, 2.0, float(curr))
            if v2 != curr: set_val("pue_input", v2)

    # --- COLUMN 2: SITE & DERATES ---
    with c2:
        st.markdown("### 🌍 Site Conditions")
        
        curr = get_val("site_temp")
        val = st.slider("Max Temp (°C)", 0, 55, int(curr))
        if val != curr: set_val("site_temp", val)
        
        curr = get_val("site_alt")
        val = st.number_input("Altitude (m)", 0, 5000, int(curr))
        if val != curr: set_val("site_alt", val)
        
        curr = get_val("mn")
        val = st.number_input("Methane #", 30, 100, int(curr))
        if val != curr: set_val("mn", val)
        
        curr = get_val("reg_zone")
        val = st.selectbox("Emissions", ["LatAm / No-Reg", "EU Standard", "USA - EPA Major"], index=["LatAm / No-Reg", "EU Standard", "USA - EPA Major"].index(curr))
        if val != curr: set_val("reg_zone", val)

    # --- COLUMN 3: INFRASTRUCTURE ---
    with c3:
        st.markdown("### 🔋 Infrastructure")
        
        # BESS
        curr = get_val("use_bess")
        val = st.checkbox("BESS Active", curr)
        if val != curr: set_val("use_bess", val)
        
        if val:
            curr = get_val("n_bess_red")
            v2 = st.number_input("BESS Redundancy (N+)", 0, 2, int(curr))
            if v2 != curr: set_val("n_bess_red", v2)
            
        # LNG
        curr = get_val("has_lng")
        val = st.checkbox("LNG Storage", curr)
        if val != curr: set_val("has_lng", val)
        
        if val:
            curr = get_val("is_lng_primary")
            v2 = st.checkbox("Virtual Pipeline (100%)", curr)
            if v2 != curr: set_val("is_lng_primary", v2)
            
        # Fleet
        c3a, c3b = st.columns(2)
        curr = get_val("n_reserve")
        v1 = c3a.number_input("Res.", 0, 5, int(curr))
        if v1 != curr: set_val("n_reserve", v1)
        
        curr = get_val("n_maint")
        v2 = c3b.number_input("Maint.", 0, 5, int(curr))
        if v2 != curr: set_val("n_maint", v2)

    # --- COLUMN 4: FINANCIALS ---
    with c4:
        st.markdown("### 💰 Financials")
        
        curr = get_val("gas_price")
        val = st.number_input("Gas ($/MMBtu)", 0.5, 50.0, float(curr))
        if val != curr: set_val("gas_price", val)
        
        if get_val("is_lng_primary"):
            curr = get_val("vp_premium")
            val = st.number_input("Logistics Premium", 0.0, 10.0, float(curr))
            if val != curr: set_val("vp_premium", val)
            
        curr = get_val("cost_kw")
        val = st.number_input("Genset ($/kW)", 100.0, 2000.0, float(curr))
        if val != curr: set_val("cost_kw", val)
        
        curr = get_val("wacc")
        val = st.number_input("WACC (%)", 0.0, 20.0, float(curr)*100) / 100
        if val != curr: set_val("wacc", val)

    # --- LIVE RESULTS BAR ---
    st.divider()
    inputs = {k: get_val(k) for k in defaults.keys()}
    res = calculate_kpis(inputs)
    
    k1, k2, k3, k4, k5 = st.columns(5)
    k1.metric("LCOE", f"${res['lcoe']:.4f}/kWh")
    k2.metric("CAPEX", f"${res['total_capex']:.1f} M")
    k3.metric("Fleet", f"{res['n_total']}x {res['model']}")
    k4.metric("Net HR", f"{res['hr_net']:.0f} Btu/kWh")
    k5.metric("Availability", f"{res['avail_est']*100:.4f}%")

# --- TAB 2: COMPARATOR ---
with tab_comp:
    st.header("Scenario Comparison")
    
    if len(st.session_state['project']['scenarios']) < 1:
        st.write("No scenarios.")
    else:
        all_res = []
        for name, params in st.session_state['project']['scenarios'].items():
            # Merge defaults
            full_params = defaults.copy()
            full_params.update(params)
            r = calculate_kpis(full_params)
            r['Scenario'] = name
            all_res.append(r)
            
        df = pd.DataFrame(all_res)
        df = df.set_index('Scenario')
        
        # Select KPIs
        df_view = df[['lcoe', 'total_capex', 'fuel_cost', 'n_total', 'model', 'hr_net']].copy()
        
        # Style
        st.dataframe(
            df_view.style.format({
                'lcoe': '${:.4f}', 
                'total_capex': '${:,.1f}M', 
                'fuel_cost': '${:,.0f}/yr',
                'hr_net': '{:,.0f}'
            }).highlight_min(subset=['lcoe', 'total_capex'], color='lightgreen', axis=0)
              .highlight_max(subset=['lcoe', 'total_capex'], color='lightpink', axis=0),
            use_container_width=True
        )
        
        # Chart
        c_chart1, c_chart2 = st.columns(2)
        fig1 = px.bar(df_view, x=df_view.index, y='lcoe', color='model', title="LCOE by Scenario", text_auto='.4f')
        c_chart1.plotly_chart(fig1, use_container_width=True)
        
        fig2 = px.bar(df_view, x=df_view.index, y='total_capex', title="CAPEX by Scenario", text_auto='.1f')
        c_chart2.plotly_chart(fig2, use_container_width=True)

# --- TAB 3: REPORT ---
with tab_rep:
    st.header("Project Executive Report")
    st.write(f"**Project:** {st.session_state['project']['name']}")
    st.write(f"**Date:** {st.session_state['project']['created_at']}")
    
    best_scen = df_view['lcoe'].idxmin()
    best_val = df_view['lcoe'].min()
    
    st.success(f"🏆 **Recommended Strategy:** The optimal scenario is **{best_scen}** with an LCOE of **${best_val:.4f}/kWh**.")
    
    st.markdown("### Detailed Configuration (Best Scenario)")
    best_params = st.session_state['project']['scenarios'][best_scen]
    st.json(best_params)
    
    # Placeholder for PDF export logic
    st.info("To save this report, use the browser's 'Print to PDF' function or download the Project JSON in the sidebar.")

# --- FOOTER ---
st.markdown("---")
st.caption("CAT Power Master Architect | v2.0 | Full Edition with Persistence")
