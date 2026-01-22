import streamlit as st
import pandas as pd
import numpy as np
import math
import plotly.express as px
import json
import io

# --- PAGE CONFIGURATION ---
st.set_page_config(page_title="CAT Power Master Architect v3.1", page_icon="🏗️", layout="wide")

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

def calculate_kpis(inputs):
    res = {}
    
    # --- A. Unpack & Setup ---
    model_key = inputs.get("model", "G3520K")
    spec = leps_gas_library[model_key]
    is_imperial = inputs.get("unit_system") == "Imperial (US)"
    
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
    
    # --- B. Derates (Manual vs Auto) ---
    if inputs.get("derate_mode") == "Manual":
        derate = 1.0 - (inputs.get("manual_derate", 5.0) / 100.0)
    else:
        # Auto Calc
        loss_temp = max(0, (inputs.get("site_temp", 35) - 25) * 0.01)
        loss_alt = max(0, (inputs.get("site_alt", 100) - 100) * 0.0001)
        loss_mn = max(0, (75 - inputs.get("mn", 80)) * 0.005)
        derate = 1.0 - (loss_temp + loss_alt + loss_mn)
        
    unit_site_cap = spec['iso_mw'] * derate
    
    # --- C. Fleet Sizing & Reliability ---
    # 1. Steady State Requirement
    target_lf = 0.95 if inputs.get("use_bess", True) else 0.90
    n_run = math.ceil(p_gross / (unit_site_cap * target_lf))
    
    # 2. Reliability Loop (Probabilistic)
    n_maint = math.ceil(n_run * inputs.get("maint_outage_pct", 0.05))
    
    # Probabilities
    prob_unit_ok = 1.0 - inputs.get("forced_outage_pct", 0.02)
    target_av = inputs.get("avail_req", 99.99) / 100.0
    
    n_reserve = 0
    sys_reliability = 0.0
    
    # Simple limit to avoid infinite loop
    for r in range(0, 10):
        n_pool = n_run + r
        prob_sys = 0.0
        # Binomial distribution logic: Prob of having at least n_run units available
        for k in range(n_run, n_pool + 1):
            comb = math.comb(n_pool, k)
            prob = comb * (prob_unit_ok ** k) * ((1 - prob_unit_ok) ** (n_pool - k))
            prob_sys += prob
        
        if prob_sys >= target_av:
            n_reserve = r
            sys_reliability = prob_sys
            break
        n_reserve = r # Update loop default
        sys_reliability = prob_sys

    n_total = n_run + n_maint + n_reserve
    installed_mw = n_total * unit_site_cap
    
    # --- D. Noise Calculation ---
    dist_m = inputs.get("dist_neighbor", 100.0)
    # Attenuation Formula: L2 = L1 - 20log(d) - 8 (Hemispherical propagation approx)
    attenuation = 20 * math.log10(dist_m) + 8 
    source_dba = inputs.get("source_noise_dba", 85.0)
    # Logarithmic addition of sources
    total_source = source_dba + (10 * math.log10(n_run))
    noise_at_neighbor = total_source - attenuation
    noise_limit = inputs.get("noise_limit", 70.0)
    noise_excess = max(0, noise_at_neighbor - noise_limit)
    
    # --- E. CAPEX ---
    # Generators (Use editable inputs)
    capex_gen = n_total * 1000 * inputs.get("cost_kw", spec['cost_kw'])
    capex_inst = n_total * 1000 * inputs.get("inst_kw", spec['inst_kw'])
    
    # BESS
    capex_bess = 0
    if inputs.get("use_bess", True):
        # Sizing BESS for Step Load
        step_mw_req = p_it * (inputs.get("step_load_req", 40.0)/100)
        mw_bess_req = max(step_mw_req, unit_site_cap) * (1 + inputs.get("n_bess_red", 0))
        mwh_bess = mw_bess_req * 2 
        capex_bess = (mw_bess_req * 1000 * inputs.get("cost_bess_inv", 120)) + (mwh_bess * 1000 * inputs.get("cost_bess_kwh", 280))
        
    # Logistics (LNG)
    capex_log = 0
    fuel_mmbtu_hr = p_gross * (spec['hr']/1000)
    if inputs.get("has_lng", True):
        vol_day = (fuel_mmbtu_hr * 24) * 12.5 
        n_tanks = math.ceil((vol_day * inputs.get("lng_days", 5)) / inputs.get("tank_size", 10000))
        capex_log = n_tanks * inputs.get("tank_cost", 50000)
        
    # Pipeline
    if not inputs.get("is_lng_primary", False):
        capex_log += (inputs.get("dist_pipe", 1000) * 200) 
        
    # Emissions (SCR / Oxicat)
    capex_emis = 0
    total_bhp = p_gross * 1341
    nox_tpy = (spec['nox'] * total_bhp * 8760) / 907185
    limit_nox = 250.0 if inputs.get("reg_zone") == "USA - EPA Major" else (150.0 if inputs.get("reg_zone") == "EU Standard" else 9999)
    
    if nox_tpy > limit_nox:
        capex_emis += installed_mw * 1000 * inputs.get("cost_scr", 60.0)
    if inputs.get("force_oxicat", False):
        capex_emis += installed_mw * 1000 * inputs.get("cost_oxicat", 15.0)
        
    # Tri-Gen (Editable cost)
    capex_chp = 0
    if use_chp:
        capex_chp = capex_gen * inputs.get("chp_cost_factor", 0.20)
        
    total_capex = (capex_gen + capex_inst + capex_bess + capex_log + capex_emis + capex_chp) / 1e6
    
    # --- F. OPEX & Financials ---
    gas_price = inputs.get("gas_price", 6.5)
    if inputs.get("is_lng_primary", False): gas_price += inputs.get("vp_premium", 4.0)
    
    # Heat Rate Adjustment
    lf_real = p_gross / (n_run * unit_site_cap)
    eff_factor = 1.0 if lf_real > 0.75 else (0.85 + (0.6*(lf_real-0.5)))
    hr_site = spec['hr'] / max(0.5, eff_factor)
    
    fuel_cost_yr = p_gross * (hr_site/1000) * gas_price * 8760
    om_cost_yr = p_net * 8760 * inputs.get("om_var", 12.0)
    if inputs.get("use_bess", True): om_cost_yr += (mw_bess_req * 1000 * inputs.get("bess_om", 10.0))
    
    # Repowering
    repower_ann = 0
    if inputs.get("use_bess", True):
        repower_ann = ((mwh_bess * 1000 * inputs.get("cost_bess_kwh", 280)) / 1e6) / ((1+inputs.get("wacc", 0.08))**10) * 1e6 
        
    wacc = inputs.get("wacc", 0.08)
    years = inputs.get("years", 20)
    crf = (wacc * (1 + wacc)**years) / ((1 + wacc)**years - 1)
    
    capex_ann = total_capex * 1e6 * crf
    total_ann_cost = fuel_cost_yr + om_cost_yr + capex_ann + repower_ann
    
    lcoe = total_ann_cost / (p_net * 8760 * 1000)
    
    # Net HR for Display
    hr_net_btu = (p_gross * (hr_site/1000) * 1e6) / (p_net)
    
    # Pack Results
    res = {
        "model": model_key,
        "n_total": n_total,
        "n_run": n_run,
        "n_reserve": n_reserve,
        "p_net": p_net,
        "pue": pue,
        "total_capex": total_capex,
        "lcoe": lcoe,
        "fuel_cost": fuel_cost_yr,
        "hr_net_btu": hr_net_btu,
        "nox_tpy": nox_tpy,
        "sys_reliability": sys_reliability,
        "noise_val": noise_at_neighbor,
        "noise_limit": noise_limit,
        "noise_excess": noise_excess
    }
    return res

# ==============================================================================
# 2. STATE & PERSISTENCE
# ==============================================================================

# Updated Defaults including new fields
defaults = {
    # 1. Global & Site
    "unit_system": "Metric (SI)", "freq": 60, 
    "derate_mode": "Auto-Calculate", "manual_derate": 5.0,
    "site_temp": 35, "site_alt": 100, "mn": 80, "reg_zone": "LatAm / No-Reg", 
    "dist_neighbor": 100.0, "noise_limit": 70.0, "source_noise_dba": 85.0,
    
    # 2. Load
    "dc_type": "AI Factory", "p_it": 100.0, "dc_aux": 0.05, "avail_req": 99.99, 
    "step_load_req": 40.0, "volt_kv": 13.8,
    
    # 3. Technology
    "model": "G3520K", "cost_kw": 575.0, "inst_kw": 650.0,
    "maint_outage_pct": 0.05, "forced_outage_pct": 0.02, # Stats inputs
    "gen_parasitic": 0.025,
    "use_bess": True, "n_bess_red": 0, "cost_bess_kwh": 280.0, "cost_bess_inv": 120.0, "bess_om": 10.0,
    
    # 4. Logistics
    "use_chp": True, "chp_cost_factor": 0.20, "pue_input": 1.45, "dist_loss": 0.01,
    "has_lng": True, "is_lng_primary": False, "lng_days": 5, "tank_size": 10000.0, "tank_cost": 50000.0,
    "dist_pipe": 1000.0, 
    "cost_scr": 60.0, "cost_oxicat": 15.0, "force_oxicat": False,
    
    # 5. Economics
    "gas_price": 6.5, "vp_premium": 4.0, "om_var": 12.0, "grid_price": 0.15,
    "wacc": 0.08, "years": 20, "target_lcoe": 0.11
}

# --- AUTO-HEALING STATE ---
if 'project' not in st.session_state:
    st.session_state['project'] = {
        "name": "Project Alpha",
        "created_at": str(pd.Timestamp.now()),
        "scenarios": { "Base Case": defaults.copy() }
    }
    st.session_state['active_scenario'] = "Base Case"
else:
    if 'created_at' not in st.session_state['project']:
        st.session_state['project']['created_at'] = str(pd.Timestamp.now())

def get_val(key):
    scen = st.session_state['active_scenario']
    return st.session_state['project']['scenarios'][scen].get(key, defaults.get(key, 0))

def set_val(key, value):
    scen = st.session_state['active_scenario']
    st.session_state['project']['scenarios'][scen][key] = value

# ==============================================================================
# 3. SIDEBAR
# ==============================================================================

with st.sidebar:
    st.title("CAT Architect v3.1")
    
    with st.expander("💾 Project File", expanded=False):
        proj_data = json.dumps(st.session_state['project'], indent=2)
        st.download_button("Download (.json)", proj_data, f"{st.session_state['project']['name']}.json", "application/json")
        uploaded_file = st.file_uploader("Load", type=["json"])
        if uploaded_file:
            data = json.load(uploaded_file)
            st.session_state['project'] = data
            st.session_state['active_scenario'] = list(data['scenarios'].keys())[0]
            st.rerun()

    st.divider()
    st.text_input("Project Name", value=st.session_state['project']['name'], key="proj_name_input", on_change=lambda: st.session_state['project'].update({"name": st.session_state.proj_name_input}))
    
    scenarios = list(st.session_state['project']['scenarios'].keys())
    active = st.selectbox("Active Scenario", scenarios, index=scenarios.index(st.session_state['active_scenario']))
    st.session_state['active_scenario'] = active
    
    new_scen = st.text_input("New Scenario", placeholder="Name...")
    if st.button("➕ Create Scenario"):
        if new_scen and new_scen not in scenarios:
            st.session_state['project']['scenarios'][new_scen] = st.session_state['project']['scenarios'][active].copy()
            st.rerun()

# ==============================================================================
# 4. MAIN INTERFACE
# ==============================================================================

tab_edit, tab_comp, tab_rep = st.tabs(["📝 Scenario Editor", "📊 Comparative Analysis", "📑 Report"])

# --- TAB 1: FULL EDITOR ---
with tab_edit:
    st.subheader(f"Editing: {st.session_state['active_scenario']}")
    
    t_glob, t_load, t_tech, t_log, t_fin = st.tabs(["🌍 Global & Site", "🏗️ Load & Config", "⚙️ Technology", "🚚 Logistics & BOP", "💰 Economics"])
    
    # 1. GLOBAL & SITE
    with t_glob:
        c1, c2 = st.columns(2)
        with c1:
            curr = get_val("unit_system")
            v = st.radio("Units", ["Metric (SI)", "Imperial (US)"], horizontal=True, index=0 if "Metric" in curr else 1)
            if v != curr: set_val("unit_system", v)
            
            curr = get_val("reg_zone")
            v = st.selectbox("Regulatory Zone", ["LatAm / No-Reg", "EU Standard", "USA - EPA Major"], index=["LatAm / No-Reg", "EU Standard", "USA - EPA Major"].index(curr))
            if v != curr: set_val("reg_zone", v)
            
            st.markdown("**Noise Constraints**")
            curr = get_val("dist_neighbor")
            v = st.number_input("Dist. Neighbor (m)", 10.0, 5000.0, float(curr))
            if v != curr: set_val("dist_neighbor", v)
            
            curr = get_val("noise_limit")
            v = st.number_input("Limit at Property Line (dBA)", 40.0, 100.0, float(curr))
            if v != curr: set_val("noise_limit", v)
            
        with c2:
            st.markdown("**Derate Settings**")
            curr = get_val("derate_mode")
            v = st.radio("Mode", ["Auto-Calculate", "Manual"], horizontal=True, index=0 if curr=="Auto-Calculate" else 1)
            if v != curr: set_val("derate_mode", v)
            
            if v == "Manual":
                curr = get_val("manual_derate")
                v = st.number_input("Manual Derate (%)", 0.0, 50.0, float(curr))
                if v != curr: set_val("manual_derate", v)
            else:
                curr = get_val("site_temp")
                v = st.slider("Max Ambient Temp (°C)", 0, 55, int(curr))
                if v != curr: set_val("site_temp", v)
                
                curr = get_val("site_alt")
                v = st.number_input("Altitude (m)", 0, 5000, int(curr))
                if v != curr: set_val("site_alt", v)
                
                curr = get_val("mn")
                v = st.number_input("Methane Number (MN)", 30, 100, int(curr))
                if v != curr: set_val("mn", v)

    # 2. LOAD & CONFIG
    with t_load:
        c1, c2 = st.columns(2)
        with c1:
            curr = get_val("p_it")
            v = st.number_input("Critical IT Load (MW)", 1.0, 1000.0, float(curr))
            if v != curr: set_val("p_it", v)
            
            curr = get_val("dc_aux")
            v = st.number_input("DC Aux (%)", 0.0, 20.0, float(curr)*100) / 100
            if v != curr: set_val("dc_aux", v)
            
        with c2:
            curr = get_val("avail_req")
            v = st.number_input("Availability Target (%)", 90.0, 99.9999, float(curr), format="%.4f")
            if v != curr: set_val("avail_req", v)
            
            curr = get_val("volt_kv")
            v = st.number_input("Connection Voltage (kV)", 0.4, 230.0, float(curr))
            if v != curr: set_val("volt_kv", v)

    # 3. TECHNOLOGY (Updated for Reliability Stats)
    with t_tech:
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**Generators**")
            curr = get_val("model")
            v = st.selectbox("Model Selection", list(leps_gas_library.keys()), index=list(leps_gas_library.keys()).index(curr))
            if v != curr: set_val("model", v)
            
            # Reliability Stats Inputs
            st.caption("Reliability Statistics")
            c1a, c1b = st.columns(2)
            curr = get_val("maint_outage_pct")
            v = c1a.number_input("Maint. Outage (%)", 0.0, 20.0, float(curr)*100)/100
            if v != curr: set_val("maint_outage_pct", v)
            
            curr = get_val("forced_outage_pct")
            v = c1b.number_input("Forced Outage (%)", 0.0, 10.0, float(curr)*100)/100
            if v != curr: set_val("forced_outage_pct", v)
            
            curr = get_val("gen_parasitic")
            v = st.number_input("Gen Parasitics (%)", 0.0, 10.0, float(curr)*100)/100
            if v != curr: set_val("gen_parasitic", v)

        with c2:
            st.markdown("**BESS**")
            curr = get_val("use_bess")
            v = st.checkbox("Enable BESS", curr)
            if v != curr: set_val("use_bess", v)
            
            if v:
                curr = get_val("n_bess_red")
                v = st.number_input("BESS Redundancy (N+)", 0, 2, int(curr))
                if v != curr: set_val("n_bess_red", v)
                
                curr = get_val("cost_bess_kwh")
                v = st.number_input("Battery Cost ($/kWh)", 100.0, 1000.0, float(curr))
                if v != curr: set_val("cost_bess_kwh", v)

    # 4. LOGISTICS & BOP
    with t_log:
        c1, c2, c3 = st.columns(3)
        with c1:
            st.markdown("**Fuel Logistics**")
            curr = get_val("has_lng")
            v = st.checkbox("Include LNG Storage", curr)
            if v != curr: set_val("has_lng", v)
            
            if v:
                curr = get_val("is_lng_primary")
                v = st.checkbox("LNG is Primary (Virtual Pipe)", curr)
                if v != curr: set_val("is_lng_primary", v)
                
                curr = get_val("tank_cost")
                v = st.number_input("Unit Tank Cost ($)", 1000.0, 200000.0, float(curr))
                if v != curr: set_val("tank_cost", v)

        with c2:
            st.markdown("**Cooling**")
            curr = get_val("use_chp")
            v = st.checkbox("Tri-Gen (Absorption)", curr)
            if v != curr: set_val("use_chp", v)
            
            if not v:
                curr = get_val("pue_input")
                v = st.number_input("Target PUE (Elec)", 1.05, 2.0, float(curr))
                if v != curr: set_val("pue_input", v)

        with c3:
            st.markdown("**Emissions**")
            curr = get_val("cost_scr")
            v = st.number_input("SCR Cost ($/kW)", 0.0, 200.0, float(curr))
            if v != curr: set_val("cost_scr", v)

    # 5. ECONOMICS (Updated for Granular CAPEX)
    with t_fin:
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**Unit Costs (CAPEX)**")
            curr = get_val("cost_kw")
            v = st.number_input("Genset Equip ($/kW)", 100.0, 2000.0, float(curr))
            if v != curr: set_val("cost_kw", v)
            
            curr = get_val("inst_kw")
            v = st.number_input("Installation/BOP ($/kW)", 50.0, 2000.0, float(curr))
            if v != curr: set_val("inst_kw", v)
            
            if get_val("use_chp"):
                curr = get_val("chp_cost_factor")
                v = st.number_input("Tri-Gen Cost Factor (vs Gen)", 0.1, 1.0, float(curr))
                if v != curr: set_val("chp_cost_factor", v)
            
        with c2:
            st.markdown("**OPEX & Financials**")
            curr = get_val("gas_price")
            v = st.number_input("Gas Price ($/MMBtu)", 0.5, 50.0, float(curr))
            if v != curr: set_val("gas_price", v)
            
            curr = get_val("target_lcoe")
            v = st.number_input("Target LCOE ($/kWh)", 0.05, 0.50, float(curr))
            if v != curr: set_val("target_lcoe", v)

    # --- LIVE RESULTS BAR ---
    st.divider()
    inputs = {k: get_val(k) for k in defaults.keys()}
    res = calculate_kpis(inputs)
    
    # Heat Rate Unit Logic
    is_imp = get_val("unit_system") == "Imperial (US)"
    hr_val = res['hr_net_btu'] if is_imp else (res['hr_net_btu'] * 0.001055)
    hr_unit = "Btu/kWh" if is_imp else "MJ/kWh"
    
    k1, k2, k3, k4, k5 = st.columns(5)
    k1.metric("LCOE", f"${res['lcoe']:.4f}/kWh")
    k2.metric("CAPEX", f"${res['total_capex']:.1f} M")
    k3.metric("Fleet", f"{res['n_total']}x {res['model']}", f"N+{res['n_reserve']}")
    k4.metric("Net HR", f"{hr_val:,.0f} {hr_unit}")
    k5.metric("Availability", f"{res['sys_reliability']*100:.4f}%")
    
    # Noise Warning
    if res['noise_excess'] > 0:
        st.error(f"🔊 Noise Limit Exceeded: {res['noise_val']:.1f} dBA > {res['noise_limit']} dBA. Check Logistics tab.")

# --- TAB 2: COMPARATOR ---
with tab_comp:
    st.header("Scenario Comparison")
    
    if len(st.session_state['project']['scenarios']) < 1:
        st.write("No scenarios.")
    else:
        all_res = []
        for name, params in st.session_state['project']['scenarios'].items():
            full_params = defaults.copy()
            full_params.update(params)
            r = calculate_kpis(full_params)
            r['Scenario'] = name
            
            # Unit Conversion for DF
            is_imp_loc = full_params.get("unit_system") == "Imperial (US)"
            r['hr_display'] = r['hr_net_btu'] if is_imp_loc else (r['hr_net_btu'] * 0.001055)
            r['unit'] = "Btu" if is_imp_loc else "MJ"
            
            all_res.append(r)
            
        df = pd.DataFrame(all_res)
        df = df.set_index('Scenario')
        
        df_view = df[['lcoe', 'total_capex', 'fuel_cost', 'n_total', 'model', 'hr_display', 'sys_reliability']].copy()
        df_view['sys_reliability'] = df_view['sys_reliability'] * 100
        
        st.dataframe(
            df_view.style.format({
                'lcoe': '${:.4f}', 
                'total_capex': '${:,.1f}M', 
                'fuel_cost': '${:,.0f}/yr',
                'hr_display': '{:,.0f}',
                'sys_reliability': '{:.4f}%'
            }).highlight_min(subset=['lcoe', 'total_capex'], color='lightgreen', axis=0)
              .highlight_max(subset=['lcoe', 'total_capex'], color='lightpink', axis=0),
            use_container_width=True
        )
        
        fig1 = px.bar(df_view, x=df_view.index, y='lcoe', color='model', title="LCOE by Scenario", text_auto='.4f')
        st.plotly_chart(fig1, use_container_width=True)

# --- TAB 3: REPORT ---
with tab_rep:
    st.header("Project Executive Report")
    st.write(f"**Project:** {st.session_state['project']['name']}")
    
    best_scen = df_view['lcoe'].idxmin()
    best_val = df_view['lcoe'].min()
    
    st.success(f"🏆 **Recommended Strategy:** The optimal scenario is **{best_scen}** with an LCOE of **${best_val:.4f}/kWh**.")
    st.json(st.session_state['project']['scenarios'][best_scen])

# --- FOOTER ---
st.markdown("---")
st.caption("CAT Power Master Architect | v3.1 | Advanced Engine")
