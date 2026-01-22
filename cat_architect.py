import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
import numpy as np
import math
import plotly.express as px
import plotly.graph_objects as go
import json
import time

# --- PAGE CONFIGURATION ---
st.set_page_config(page_title="CAT Architect v5.4", page_icon="🏗️", layout="wide")

# --- CSS FOR REPORT FORMATTING (FIXED MARGINS) ---
st.markdown("""
<style>
    /* Estilos de Impresión Profesional */
    @media print {
        /* Configuración de Página */
        @page {
            size: landscape;
            margin: 1.5cm;
        }
        
        /* Ocultar elementos de UI */
        [data-testid="stSidebar"], [data-testid="stHeader"], .stApp > header, 
        footer, .stButton, .stDeployButton, [data-testid="stToolbar"] { 
            display: none !important; 
        }
        
        /* Ajuste del Contenedor Principal */
        .block-container { 
            padding: 0 !important; 
            margin: 0 !important;
            max-width: 100% !important;
            box-shadow: none !important;
        }
        
        /* Texto y Tablas */
        body {
            font-family: "Helvetica Neue", Helvetica, Arial, sans-serif;
            color: black !important;
            font-size: 10pt;
        }
        
        /* Salto de Página Controlado */
        .page-break { 
            page-break-before: always !important; 
            break-before: always !important;
            display: block; 
            height: 0; 
            visibility: hidden;
        }
        
        /* Asegurar que los gráficos se vean completos */
        .js-plotly-plot {
            page-break-inside: avoid;
            margin-bottom: 20px;
        }
    }
    
    /* Botón de Impresión en Pantalla */
    .print-btn-container { text-align: center; margin-top: 20px; margin-bottom: 20px; }
</style>
""", unsafe_allow_html=True)

# ==============================================================================
# 1. PHYSICS ENGINE
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
    model_key = inputs.get("model", "G3520K")
    spec = leps_gas_library[model_key]
    
    # Load
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
    
    # Derate
    if inputs.get("derate_mode") == "Manual":
        derate = 1.0 - (inputs.get("manual_derate", 5.0) / 100.0)
    else:
        loss_temp = max(0, (inputs.get("site_temp", 35) - 25) * 0.01)
        loss_alt = max(0, (inputs.get("site_alt", 100) - 100) * 0.0001)
        loss_mn = max(0, (75 - inputs.get("mn", 80)) * 0.005)
        derate = 1.0 - (loss_temp + loss_alt + loss_mn)
        
    unit_site_cap = spec['iso_mw'] * derate
    
    # Fleet Sizing
    target_lf = 0.95 if inputs.get("use_bess", True) else 0.90
    n_run = math.ceil(p_gross / (unit_site_cap * target_lf))
    n_maint = math.ceil(n_run * inputs.get("maint_outage_pct", 0.05))
    
    prob_unit_ok = 1.0 - inputs.get("forced_outage_pct", 0.02)
    target_av = inputs.get("avail_req", 99.99) / 100.0
    
    n_reserve = 0
    prob_gen_sys = 0.0
    
    for r in range(0, 20):
        n_pool = n_run + r
        p_accum = 0.0
        for k in range(n_run, n_pool + 1):
            comb = math.comb(n_pool, k)
            p_accum += comb * (prob_unit_ok ** k) * ((1 - prob_unit_ok) ** (n_pool - k))
        
        if p_accum >= target_av:
            n_reserve = r
            prob_gen_sys = p_accum
            break
        n_reserve = r
        prob_gen_sys = p_accum

    n_total = n_run + n_maint + n_reserve
    installed_mw = n_total * unit_site_cap
    
    # BESS
    capex_bess = 0
    mw_bess_total = 0
    mwh_bess_total = 0
    prob_bess_sys = 1.0
    n_bess_red = 0
    bess_desc = "None"
    
    if inputs.get("use_bess", True):
        step_mw_req = p_it * (inputs.get("step_load_req", 40.0)/100)
        mw_bess_req = max(step_mw_req, unit_site_cap) 
        bess_fail_rate = inputs.get("bess_maint_pct", 0.01) + inputs.get("bess_for_pct", 0.005)
        
        for r in range(0, 10):
            sys_unavail = (bess_fail_rate ** (1 + r))
            sys_avail = 1.0 - sys_unavail
            if sys_avail >= target_av:
                n_bess_red = r
                prob_bess_sys = sys_avail
                break
            n_bess_red = r
            prob_bess_sys = sys_avail
            
        mw_bess_total = mw_bess_req * (1 + n_bess_red)
        mwh_bess_total = mw_bess_total * 2
        capex_bess = (mw_bess_total * 1000 * inputs.get("cost_bess_inv", 120)) + (mwh_bess_total * 1000 * inputs.get("cost_bess_kwh", 280))
        bess_desc = f"{mw_bess_total:.1f} MW / {mwh_bess_total:.1f} MWh (N+{n_bess_red})"

    sys_reliability = prob_gen_sys * prob_bess_sys

    # Capex Agg
    capex_gen = n_total * 1000 * inputs.get("cost_kw", spec['cost_kw'])
    capex_inst = n_total * 1000 * inputs.get("inst_kw", spec['inst_kw'])
    capex_log = 0
    fuel_mmbtu_hr = p_gross * (spec['hr']/1000)
    
    if inputs.get("has_lng", True):
        vol_day = (fuel_mmbtu_hr * 24) * 12.5 
        n_tanks = math.ceil((vol_day * inputs.get("lng_days", 5)) / inputs.get("tank_size", 10000))
        capex_log = (n_tanks * inputs.get("tank_cost", 50000)) + inputs.get("tank_mob", 5000)
        
    if not inputs.get("is_lng_primary", False):
        capex_log += (inputs.get("dist_pipe", 1000) * 200) 
        
    capex_emis = 0
    total_bhp = p_gross * 1341
    nox_tpy = (spec['nox'] * total_bhp * 8760) / 907185
    limit_nox = 250.0 if "EPA" in inputs.get("reg_zone", "") else (150.0 if "EU" in inputs.get("reg_zone", "") else 9999)
    req_scr = nox_tpy > limit_nox
    if req_scr: capex_emis += installed_mw * 1000 * inputs.get("cost_scr", 60.0)
    if inputs.get("force_oxicat", False): capex_emis += installed_mw * 1000 * inputs.get("cost_oxicat", 15.0)
        
    capex_chp = 0
    if use_chp: capex_chp = capex_gen * inputs.get("chp_cost_factor", 0.20)
        
    total_capex = (capex_gen + capex_inst + capex_bess + capex_log + capex_emis + capex_chp) / 1e6
    
    # Financials
    gas_price = inputs.get("gas_price", 6.5)
    if inputs.get("is_lng_primary", False): gas_price += inputs.get("vp_premium", 4.0)
    
    lf_real = p_gross / (n_run * unit_site_cap)
    eff_factor = 1.0 if lf_real > 0.75 else (0.85 + (0.6*(lf_real-0.5)))
    hr_site = spec['hr'] / max(0.5, eff_factor)
    
    fuel_cost_yr = p_gross * (hr_site/1000) * gas_price * 8760
    om_cost_yr = p_net * 8760 * inputs.get("om_var", 12.0)
    if inputs.get("use_bess", True): om_cost_yr += (mw_bess_total * 1000 * inputs.get("bess_om", 10.0))
    
    wacc = inputs.get("wacc", 0.08)
    years = inputs.get("years", 20)
    crf = (wacc * (1 + wacc)**years) / ((1 + wacc)**years - 1)
    
    repower_ann = 0
    if inputs.get("use_bess", True):
        repower_val = (mwh_bess_total * 1000 * inputs.get("cost_bess_kwh", 280)) / 1e6
        repower_ann = (repower_val / ((1+wacc)**10)) * 1e6 * crf 
    
    capex_ann = total_capex * 1e6 * crf
    total_ann_cost = fuel_cost_yr + om_cost_yr + capex_ann + repower_ann
    
    lcoe = total_ann_cost / (p_net * 8760 * 1000)
    
    # Noise
    dist_m = inputs.get("dist_neighbor", 100.0)
    attenuation = 20 * math.log10(dist_m) + 8
    total_source = inputs.get("source_noise_dba", 85.0) + (10 * math.log10(n_run))
    noise_at_neighbor = total_source - attenuation
    noise_excess = max(0, noise_at_neighbor - inputs.get("noise_limit", 70.0))
    
    res = {
        "model": model_key, "n_total": n_total, "n_reserve": n_reserve, "n_bess_red": n_bess_red,
        "p_net": p_net, "pue": pue, "total_capex": total_capex, "lcoe": lcoe,
        "fuel_cost": fuel_cost_yr, "om_cost": om_cost_yr, "capex_ann": capex_ann,
        "hr_net_btu": (fuel_mmbtu_hr * 1e6) / (p_net * 1000),
        "sys_reliability": sys_reliability, "bess_desc": bess_desc,
        "noise_val": noise_at_neighbor, "noise_limit": inputs.get("noise_limit", 70.0), "noise_excess": noise_excess
    }
    return res

# ==============================================================================
# 2. STATE & DEFAULT CONFIG
# ==============================================================================

defaults = {
    # Metadata
    "client": "", "location": "", "quote_ref": "", "contact_name": "", "contact_email": "", "contact_phone": "", "prepared_by": "",
    # Global
    "unit_system": "Metric (SI)", "freq": 60, "derate_mode": "Auto-Calculate", "manual_derate": 5.0,
    "site_temp": 35, "site_alt": 100, "mn": 80, "reg_zone": "LatAm / No-Reg",
    "dist_neighbor": 100.0, "noise_limit": 70.0, "source_noise_dba": 85.0,
    # Load
    "dc_type": "AI Factory", "p_it": 100.0, "dc_aux": 0.05, "avail_req": 99.99, "step_load_req": 40.0, "volt_kv": 13.8,
    # Tech
    "model": "G3520K", "gen_parasitic": 0.025, "maint_outage_pct": 0.05, "forced_outage_pct": 0.02,
    "use_bess": True, "bess_maint_pct": 0.01, "bess_for_pct": 0.005, "cost_bess_kwh": 280.0, "cost_bess_inv": 120.0, "bess_om": 10.0,
    # Logistics
    "use_chp": True, "chp_cost_factor": 0.20, "cop_double": 1.2, "cop_single": 0.7, "pue_input": 1.45, "dist_loss": 0.01,
    "has_lng": True, "is_lng_primary": False, "lng_days": 5, "tank_size": 10000.0, "tank_cost": 50000.0, "tank_mob": 5000.0,
    "dist_pipe": 1000.0, "cost_scr": 60.0, "cost_oxicat": 15.0, "force_oxicat": False, "urea_days": 7,
    # Econ
    "cost_kw": 575.0, "inst_kw": 650.0, "gas_price": 6.5, "vp_premium": 4.0, "om_var": 12.0, "grid_price": 0.15,
    "wacc": 0.08, "years": 20, "target_lcoe": 0.11
}

if 'project' not in st.session_state:
    st.session_state['project'] = {
        "name": "New Project",
        "created_at": str(pd.Timestamp.now().strftime("%Y-%m-%d")),
        "client": "", "location": "", "quote_ref": "", "contact_name": "", "contact_email": "", "contact_phone": "", "prepared_by": "",
        "scenarios": { "Base Case": defaults.copy() }
    }
    st.session_state['active_scenario'] = "Base Case"
else:
    # State Auto-Heal
    if "client" not in st.session_state['project']:
        st.session_state['project'].update({
            "client": "", "location": "", "quote_ref": "", 
            "contact_name": "", "contact_email": "", "contact_phone": "", "prepared_by": ""
        })

def get_val(key):
    scen = st.session_state['active_scenario']
    return st.session_state['project']['scenarios'][scen].get(key, defaults.get(key, 0))

def set_val(key, value):
    scen = st.session_state['active_scenario']
    st.session_state['project']['scenarios'][scen][key] = value

def create_next_scenario():
    i = 1
    while f"Scenario {i}" in st.session_state['project']['scenarios']:
        i += 1
    new_name = f"Scenario {i}"
    curr = st.session_state['active_scenario']
    st.session_state['project']['scenarios'][new_name] = st.session_state['project']['scenarios'][curr].copy()
    st.session_state['active_scenario'] = new_name
    return new_name

def rename_scenario(old_name, new_name):
    if new_name and new_name != old_name:
        if new_name not in st.session_state['project']['scenarios']:
            data = st.session_state['project']['scenarios'].pop(old_name)
            st.session_state['project']['scenarios'][new_name] = data
            st.session_state['active_scenario'] = new_name
            st.rerun()
        else:
            st.warning("Name exists!")

# ==============================================================================
# 3. SIDEBAR
# ==============================================================================

with st.sidebar:
    st.title("CAT Architect v5.4")
    
    st.markdown("### 📂 Project Data")
    st.session_state['project']['client'] = st.text_input("Client", st.session_state['project']['client'], placeholder="Customer Name")
    st.session_state['project']['location'] = st.text_input("Location", st.session_state['project']['location'], placeholder="Site City/Country")
    c_ref1, c_ref2 = st.columns(2)
    st.session_state['project']['quote_ref'] = c_ref1.text_input("Ref #", st.session_state['project']['quote_ref'], placeholder="QT-123")
    st.session_state['project']['prepared_by'] = c_ref2.text_input("Engineer", st.session_state['project'].get('prepared_by',''), placeholder="Your Name")
    
    with st.expander("👤 Contact Details"):
        st.session_state['project']['contact_name'] = st.text_input("Contact Name", st.session_state['project']['contact_name'])
        st.session_state['project']['contact_email'] = st.text_input("Email", st.session_state['project']['contact_email'])
        st.session_state['project']['contact_phone'] = st.text_input("Phone", st.session_state['project']['contact_phone'])

    st.divider()
    
    # Save/Load Vertically Stacked (Fixed Layout)
    proj_json = json.dumps(st.session_state['project'], indent=2)
    st.download_button("💾 Save Project", proj_json, f"{st.session_state['project']['name']}.json", "application/json", use_container_width=True)
    
    uploaded_file = st.file_uploader("📂 Load Project", type=["json"], label_visibility="collapsed")
    if uploaded_file:
        data = json.load(uploaded_file)
        st.session_state['project'] = data
        st.session_state['active_scenario'] = list(data['scenarios'].keys())[0]
        st.success("Loaded!")
        time.sleep(0.5)
        st.rerun()
        
    with st.expander("Save As..."):
        new_proj_name = st.text_input("Filename", value=st.session_state['project']['name'])
        if st.button("Update Filename"):
            st.session_state['project']['name'] = new_proj_name
            st.rerun()

    st.divider()
    st.markdown("### 🎛️ Scenarios")
    if st.button("➕ Create Scenario", use_container_width=True):
        new_name = create_next_scenario()
        st.success(f"Created {new_name}")
        st.rerun()
        
    scenarios = list(st.session_state['project']['scenarios'].keys())
    try:
        idx = scenarios.index(st.session_state['active_scenario'])
    except:
        idx = 0
        st.session_state['active_scenario'] = scenarios[0]
        
    active = st.selectbox("Active Scenario", scenarios, index=idx)
    st.session_state['active_scenario'] = active

# ==============================================================================
# 4. DATA CALCULATION FOR ALL SCENARIOS (GLOBAL SCOPE)
# ==============================================================================

all_res = []
for name, params in st.session_state['project']['scenarios'].items():
    full = defaults.copy()
    full.update(params)
    r = calculate_kpis(full)
    r['Scenario'] = name
    
    is_imp_loc = full.get("unit_system") == "Imperial (US)"
    r['Net Heat Rate'] = r['hr_net_btu'] if is_imp_loc else (r['hr_net_btu'] * 0.001055)
    r['HR Unit'] = "Btu/kWh" if is_imp_loc else "MJ/kWh"
    r['Availability (%)'] = r['sys_reliability'] * 100
    
    r['LCOE ($/kWh)'] = r['lcoe']
    r['CAPEX (M USD)'] = r['total_capex']
    r['Fuel Cost ($/yr)'] = r['fuel_cost']
    r['Generator Model'] = r['model']
    r['Total Units'] = r['n_total']
    
    all_res.append(r)
    
df = pd.DataFrame(all_res).set_index('Scenario')

cols_show = ['LCOE ($/kWh)', 'CAPEX (M USD)', 'Fuel Cost ($/yr)', 'Total Units', 'Generator Model', 'Net Heat Rate', 'HR Unit', 'Availability (%)']

# Pre-calculate Charts for Report (Fixes NameError)
fig_print_1 = px.bar(df, x=df.index, y='LCOE ($/kWh)', color='Generator Model', text_auto='.4f', title="LCOE Comparison")
fig_print_1.update_layout(height=400)

fig_print_2 = px.bar(df, x=df.index, y='CAPEX (M USD)', text_auto='.1f', title="CAPEX Comparison", color_discrete_sequence=['#EF553B'])
fig_print_2.update_layout(height=400)

fig_print_3 = px.bar(df, x=df.index, y='Fuel Cost ($/yr)', text_auto='.0s', title="Annual Fuel Cost Comparison", color_discrete_sequence=['#00CC96'])
fig_print_3.update_layout(height=400)

# ==============================================================================
# 5. TABS & INTERFACE
# ==============================================================================

tab_edit, tab_comp, tab_rep = st.tabs(["📝 Scenario Editor", "📊 Comparative Analysis", "📑 Report"])

with tab_edit:
    c_title, c_rename = st.columns([2, 1])
    c_title.subheader(f"Editing: {st.session_state['active_scenario']}")
    with c_rename:
        new_name_input = st.text_input("Rename Scenario", value=st.session_state['active_scenario'], label_visibility="collapsed")
        if new_name_input != st.session_state['active_scenario']:
            rename_scenario(st.session_state['active_scenario'], new_name_input)

    t1, t2, t3, t4, t5 = st.tabs(["🌍 Global & Site", "🏗️ Load & Config", "⚙️ Technology", "🚚 Logistics & BOP", "💰 Economics"])
    
    # 1. Global
    with t1:
        c1, c2 = st.columns(2)
        with c1:
            curr = get_val("unit_system")
            v = st.radio("Unit System", ["Metric (SI)", "Imperial (US)"], horizontal=True, index=0 if "Metric" in curr else 1)
            if v != curr: set_val("unit_system", v)
            is_imp = "Imperial" in v
            u_temp, u_dist = ("°F", "ft") if is_imp else ("°C", "m")
            
            curr = get_val("reg_zone")
            v = st.selectbox("Regulatory Zone", ["LatAm / No-Reg", "EU Standard", "USA - EPA Major"], index=["LatAm / No-Reg", "EU Standard", "USA - EPA Major"].index(curr))
            if v != curr: set_val("reg_zone", v)
            
            st.markdown("##### 🔊 Noise Constraints")
            curr = get_val("dist_neighbor")
            v = st.number_input(f"Dist. to Neighbor ({u_dist})", 10.0, 10000.0, float(curr))
            if v != curr: set_val("dist_neighbor", v)
            
            curr = get_val("noise_limit")
            v = st.number_input("Limit at Property (dBA)", 40.0, 100.0, float(curr))
            if v != curr: set_val("noise_limit", v)
            
        with c2:
            st.markdown("##### 🌡️ Derate Settings")
            curr = get_val("derate_mode")
            v = st.radio("Derate Mode", ["Auto-Calculate", "Manual"], horizontal=True, index=0 if curr=="Auto-Calculate" else 1)
            if v != curr: set_val("derate_mode", v)
            
            if v == "Manual":
                curr = get_val("manual_derate")
                v = st.number_input("Manual Derate (%)", 0.0, 50.0, float(curr))
                if v != curr: set_val("manual_derate", v)
            else:
                curr = get_val("site_temp")
                v = st.slider(f"Max Ambient Temp ({u_temp})", 0, 120 if is_imp else 55, int(curr))
                if v != curr: set_val("site_temp", v)
                
                curr = get_val("site_alt")
                v = st.number_input(f"Altitude ({u_dist})", 0, 15000 if is_imp else 5000, int(curr))
                if v != curr: set_val("site_alt", v)
                
                curr = get_val("mn")
                v = st.number_input("Methane Number (MN)", 30, 100, int(curr))
                if v != curr: set_val("mn", v)

    # 2. Load
    with t2:
        c1, c2 = st.columns(2)
        with c1:
            curr = get_val("p_it")
            v = st.number_input("Critical IT Load (MW)", 1.0, 1000.0, float(curr))
            if v != curr: set_val("p_it", v)
            
            curr = get_val("dc_aux")
            v = st.number_input("DC Aux (%)", 0.0, 20.0, float(curr)*100)/100
            if v != curr: set_val("dc_aux", v)
            
        with c2:
            curr = get_val("avail_req")
            v = st.number_input("Availability Target (%)", 90.0, 99.9999, float(curr), format="%.4f")
            if v != curr: set_val("avail_req", v)
            
            curr = get_val("step_load_req")
            v = st.number_input("Step Load Req (%)", 0.0, 100.0, float(curr))
            if v != curr: set_val("step_load_req", v)
            
            curr = get_val("volt_kv")
            v = st.number_input("Connection Voltage (kV)", 0.4, 230.0, float(curr))
            if v != curr: set_val("volt_kv", v)

    # 3. Technology
    with t3:
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("### ⚙️ Generators")
            curr = get_val("model")
            v = st.selectbox("Model Selection", list(leps_gas_library.keys()), index=list(leps_gas_library.keys()).index(curr))
            if v != curr: set_val("model", v)
            
            st.info("ℹ️ Reserve units are auto-calculated to meet Target Availability.")
            
            c1a, c1b = st.columns(2)
            curr = get_val("maint_outage_pct")
            v = c1a.number_input("Maint. Factor (%)", 0.0, 20.0, float(curr)*100)/100
            if v != curr: set_val("maint_outage_pct", v)
            
            curr = get_val("forced_outage_pct")
            v = c1b.number_input("Forced Outage (%)", 0.0, 10.0, float(curr)*100)/100
            if v != curr: set_val("forced_outage_pct", v)
            
            curr = get_val("gen_parasitic")
            v = st.number_input("Gen Parasitics (%)", 0.0, 10.0, float(curr)*100)/100
            if v != curr: set_val("gen_parasitic", v)

        with c2:
            st.markdown("### 🔋 BESS")
            curr = get_val("use_bess")
            v = st.checkbox("Enable BESS", curr)
            if v != curr: set_val("use_bess", v)
            
            if v:
                st.info("ℹ️ BESS Redundancy is auto-calculated to meet Target Availability.")
                
                c2a, c2b = st.columns(2)
                curr = get_val("bess_maint_pct")
                v = c2a.number_input("BESS Maint (%)", 0.0, 10.0, float(curr)*100)/100
                if v != curr: set_val("bess_maint_pct", v)
                
                curr = get_val("bess_for_pct")
                v = c2b.number_input("BESS FOR (%)", 0.0, 10.0, float(curr)*100)/100
                if v != curr: set_val("bess_for_pct", v)
                
                curr = get_val("bess_om")
                v = st.number_input("BESS O&M ($/kW-yr)", 0.0, 100.0, float(curr))
                if v != curr: set_val("bess_om", v)

    # 4. Logistics
    with t4:
        c1, c2, c3 = st.columns(3)
        with c1:
            st.markdown("**Fuel Logistics**")
            curr = get_val("has_lng")
            v = st.checkbox("Include LNG Storage", curr)
            if v != curr: set_val("has_lng", v)
            
            if v:
                curr = get_val("is_lng_primary")
                v = st.checkbox("LNG is Primary", curr)
                if v != curr: set_val("is_lng_primary", v)
                
                curr = get_val("lng_days")
                v = st.number_input("LNG Autonomy (Days)", 1, 60, int(curr))
                if v != curr: set_val("lng_days", v)
                
                curr = get_val("tank_size")
                v = st.number_input("Tank Size (Gal)", 1000.0, 100000.0, float(curr))
                if v != curr: set_val("tank_size", v)
                
                curr = get_val("tank_cost")
                v = st.number_input("Unit Tank Cost ($)", 1000.0, 200000.0, float(curr))
                if v != curr: set_val("tank_cost", v)
                
                curr = get_val("tank_mob")
                v = st.number_input("Mob Cost ($)", 0.0, 50000.0, float(curr))
                if v != curr: set_val("tank_mob", v)

        with c2:
            st.markdown("**Cooling (CHP)**")
            curr = get_val("use_chp")
            v = st.checkbox("Tri-Gen (Absorption)", curr)
            if v != curr: set_val("use_chp", v)
            
            if v:
                curr = get_val("cop_double")
                v = st.number_input("COP (Double Effect)", 0.5, 2.0, float(curr))
                if v != curr: set_val("cop_double", v)
                
                curr = get_val("cop_single")
                v = st.number_input("COP (Single Effect)", 0.4, 1.5, float(curr))
                if v != curr: set_val("cop_single", v)
            else:
                curr = get_val("pue_input")
                v = st.number_input("Target PUE (Elec)", 1.05, 2.0, float(curr))
                if v != curr: set_val("pue_input", v)

        with c3:
            st.markdown("**Emissions**")
            curr = get_val("cost_scr")
            v = st.number_input("SCR Cost ($/kW)", 0.0, 200.0, float(curr))
            if v != curr: set_val("cost_scr", v)
            
            curr = get_val("urea_days")
            v = st.number_input("Urea Storage (Days)", 1, 30, int(curr))
            if v != curr: set_val("urea_days", v)
            
            curr = get_val("force_oxicat")
            v = st.checkbox("Force Oxicat", curr)
            if v != curr: set_val("force_oxicat", v)
            
            if v:
                curr = get_val("cost_oxicat")
                v = st.number_input("Oxicat Cost ($/kW)", 0.0, 100.0, float(curr))
                if v != curr: set_val("cost_oxicat", v)

    # 5. Economics
    with t5:
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("### 💰 OPEX Inputs")
            curr = get_val("gas_price")
            v = st.number_input("Gas Price ($/MMBtu)", 0.5, 50.0, float(curr))
            if v != curr: set_val("gas_price", v)
            
            if get_val("is_lng_primary"):
                curr = get_val("vp_premium")
                v = st.number_input("LNG Premium ($/MMBtu)", 0.0, 10.0, float(curr))
                if v != curr: set_val("vp_premium", v)
                
            curr = get_val("om_var")
            v = st.number_input("Variable O&M ($/MWh)", 1.0, 100.0, float(curr))
            if v != curr: set_val("om_var", v)
            
            curr = get_val("target_lcoe")
            v = st.number_input("Target LCOE ($/kWh)", 0.05, 0.50, float(curr))
            if v != curr: set_val("target_lcoe", v)
            
        with c2:
            st.markdown("### 🏗️ CAPEX Unit Costs (Editable)")
            curr = get_val("cost_kw")
            v = st.number_input("Genset Equipment ($/kW)", 100.0, 2000.0, float(curr))
            if v != curr: set_val("cost_kw", v)
            
            curr = get_val("inst_kw")
            v = st.number_input("Installation/BOP ($/kW)", 50.0, 2000.0, float(curr))
            if v != curr: set_val("inst_kw", v)
            
            if get_val("use_chp"):
                curr = get_val("chp_cost_factor")
                v = st.number_input("Tri-Gen Cost Factor (vs Gen)", 0.1, 1.0, float(curr))
                if v != curr: set_val("chp_cost_factor", v)
                
            if get_val("use_bess"):
                curr = get_val("cost_bess_kwh")
                v = st.number_input("Battery Cost ($/kWh)", 100.0, 1000.0, float(curr))
                if v != curr: set_val("cost_bess_kwh", v)
                
                curr = get_val("cost_bess_inv")
                v = st.number_input("BESS Inverter ($/kW)", 50.0, 1000.0, float(curr))
                if v != curr: set_val("cost_bess_inv", v)

    # --- LIVE RESULTS ---
    st.divider()
    inputs = {k: get_val(k) for k in defaults.keys()}
    res = calculate_kpis(inputs)
    
    is_imp = get_val("unit_system") == "Imperial (US)"
    hr_val = res['hr_net_btu'] if is_imp else (res['hr_net_btu'] * 0.001055)
    hr_unit = "Btu/kWh" if is_imp else "MJ/kWh"
    
    # Conditional Formatting for LIVE Metric
    if is_imp:
        hr_fmt = f"{hr_val:,.0f}" # 0 decimals for Btu
    else:
        hr_fmt = f"{hr_val:,.2f}" # 2 decimals for MJ
    
    k1, k2, k3, k4, k5 = st.columns(5)
    k1.metric("LCOE", f"${res['lcoe']:.4f}/kWh")
    k2.metric("CAPEX", f"${res['total_capex']:.1f} M")
    k3.metric("Fleet", f"{res['n_total']}x {res['model']}", f"N+{res['n_reserve']}")
    k4.metric("Net HR", f"{hr_fmt} {hr_unit}")
    k5.metric("Availability", f"{res['sys_reliability']*100:.4f}%")
    
    if res['bess_desc'] != "None":
        st.info(f"🔋 BESS: {res['bess_desc']}")
        
    if res['noise_excess'] > 0:
        st.error(f"🔊 Noise Violation: {res['noise_val']:.1f} dBA > Limit {res['noise_limit']} dBA. Requires mitigation.")

with tab_comp:
    st.header("Comparison")
    
    # Conditional Number Formatting
    first_scen_inputs = st.session_state['project']['scenarios'][list(st.session_state['project']['scenarios'].keys())[0]]
    is_imp_global = "Imperial" in first_scen_inputs.get('unit_system', 'Metric')
    hr_format_table = '{:,.0f}' if is_imp_global else '{:,.2f}'
    
    st.dataframe(
        df[cols_show].style.format({
            'LCOE ($/kWh)': '${:.4f}', 
            'CAPEX (M USD)': '${:,.1f}', 
            'Fuel Cost ($/yr)': '${:,.0f}', 
            'Net Heat Rate': hr_format_table,
            'Availability (%)': '{:.4f}%'
        }).highlight_min(subset=['LCOE ($/kWh)', 'CAPEX (M USD)'], color='lightgreen').highlight_max(subset=['LCOE ($/kWh)', 'CAPEX (M USD)'], color='lightpink'),
        use_container_width=True
    )
    
    c_p1, c_p2, c_p3 = st.columns(3)
    c_p1.plotly_chart(fig_print_1, use_container_width=True, key="comp_lcoe")
    c_p2.plotly_chart(fig_print_2, use_container_width=True, key="comp_capex")
    c_p3.plotly_chart(fig_print_3, use_container_width=True, key="comp_fuel")

# --- TAB 3: REPORT GENERATOR ---
with tab_rep:
    # --- JAVASCRIPT PRINT TRIGGER (More Robust) ---
    components.html("""
    <script>
    function printPage() {
        window.parent.document.title = "CAT_Project_Report";
        window.parent.print();
    }
    </script>
    <div style="display: flex; justify-content: center; margin: 20px;">
        <button onclick="printPage()" class="print-btn">
            🖨️ Print Executive Report to PDF
        </button>
    </div>
    <style>
    .print-btn {
        background-color: #FFCD11; 
        color: black; 
        border: 2px solid black; 
        padding: 12px 24px; 
        font-size: 16px; 
        font-weight: bold; 
        border-radius: 4px; 
        cursor: pointer;
    }
    </style>
    """, height=80)

    # PAGE 1 HEADER
    proj = st.session_state['project']
    best_scen_name = df['LCOE ($/kWh)'].idxmin()
    best_data = df.loc[best_scen_name]
    best_inputs = defaults.copy()
    best_inputs.update(st.session_state['project']['scenarios'][best_scen_name])
    best_kpis = calculate_kpis(best_inputs)

    st.markdown(f"## **{proj['name']}**")
    
    meta_html = f"""
    <table style="width:100%; border: 1px solid #ddd; border-collapse: collapse; margin-bottom: 20px;">
        <tr>
            <td style="padding: 8px; border: 1px solid #ddd;"><strong>Client:</strong> {proj.get('client','')}</td>
            <td style="padding: 8px; border: 1px solid #ddd;"><strong>Location:</strong> {proj.get('location','')}</td>
            <td style="padding: 8px; border: 1px solid #ddd;"><strong>Ref:</strong> {proj.get('quote_ref','')}</td>
        </tr>
        <tr>
            <td style="padding: 8px; border: 1px solid #ddd;"><strong>Contact:</strong> {proj.get('contact_name','')}</td>
            <td style="padding: 8px; border: 1px solid #ddd;"><strong>Email:</strong> {proj.get('contact_email','')}</td>
            <td style="padding: 8px; border: 1px solid #ddd;"><strong>Prepared By:</strong> {proj.get('prepared_by','')}</td>
        </tr>
    </table>
    """
    st.markdown(meta_html, unsafe_allow_html=True)

    st.success(f"🏆 **Recommended Strategy:** {best_scen_name}")

    k1, k2, k3 = st.columns(3)
    k1.metric("Target LCOE", f"${best_data['LCOE ($/kWh)']:.4f}/kWh")
    k2.metric("Total Project CAPEX", f"${best_data['CAPEX (M USD)']:.1f} M")
    k3.metric("Annual Fuel Cost", f"${best_data['Fuel Cost ($/yr)']:,.0f}")

    # Heat Rate Formatting for Report Text
    hr_val_rep = best_data['Net Heat Rate']
    hr_unit_rep = best_data['HR Unit']
    if "Btu" in hr_unit_rep:
        hr_str_rep = f"{hr_val_rep:,.0f}"
    else:
        hr_str_rep = f"{hr_val_rep:,.2f}"

    st.subheader("Technical Configuration")
    t1, t2 = st.columns(2)
    with t1:
        st.markdown(f"""
        * **Generator Model:** {best_data['Generator Model']}
        * **Fleet Config:** {best_data['Total Units']} units (N+{best_kpis['n_reserve']})
        * **Availability:** {best_data['Availability (%)']:.4f}%
        * **Cooling:** {'Tri-Gen (Absorption)' if best_inputs.get('use_chp') else 'Electric'}
        """)
    with t2:
        st.markdown(f"""
        * **BESS Capacity:** {best_kpis['bess_desc']}
        * **Net Heat Rate:** {hr_str_rep} {hr_unit_rep}
        * **PUE:** {best_kpis['pue']:.3f}
        * **LNG Storage:** {best_inputs.get('lng_days')} days autonomy
        """)

    st.subheader("Cost Structure")
    labels = ['Fuel', 'CAPEX Amortization', 'O&M']
    values = [best_kpis['fuel_cost'], best_kpis['capex_ann'], best_kpis['om_cost']]
    fig_donut = go.Figure(data=[go.Pie(labels=labels, values=values, hole=.4)])
    fig_donut.update_layout(height=300, margin=dict(t=0, b=0, l=0, r=0))
    st.plotly_chart(fig_donut, use_container_width=True, key="rep_donut")

    # --- PAGE BREAK ---
    st.markdown('<div class="page-break"></div>', unsafe_allow_html=True)

    # --- PAGE 2: COMPARISON ---
    st.markdown("## 📊 Scenario Analysis")
    
    st.table(df[cols_show].style.format({
        'LCOE ($/kWh)': '${:.4f}', 
        'CAPEX (M USD)': '${:,.1f}', 
        'Fuel Cost ($/yr)': '${:,.0f}', 
        'Net Heat Rate': hr_format_table,
        'Availability (%)': '{:.4f}%'
    }))
    
    st.markdown("### Comparative Charts")
    c_p1, c_p2 = st.columns(2)
    c_p1.plotly_chart(fig_print_1, use_container_width=True, key="rep_lcoe")
    c_p2.plotly_chart(fig_print_2, use_container_width=True, key="rep_capex")
    st.plotly_chart(fig_print_3, use_container_width=True, key="rep_fuel")

# --- FOOTER ---
st.markdown("---")
st.caption(f"Prepared by: {st.session_state['project'].get('prepared_by')} | CAT Architect v5.4")
