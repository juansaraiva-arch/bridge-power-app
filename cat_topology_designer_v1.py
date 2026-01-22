import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
import numpy as np
import math
import graphviz
import json
import time
from scipy.stats import binom

# --- PAGE CONFIGURATION ---
st.set_page_config(page_title="CAT Topology Designer v2.1", page_icon="⚡", layout="wide")

# --- CSS FOR PRINTING ---
st.markdown("""
<style>
    @media print {
        [data-testid="stSidebar"], [data-testid="stHeader"], .stApp > header, footer, .stButton { display: none !important; }
        .block-container { padding: 0 !important; margin: 0 !important; }
        .page-break { page-break-before: always !important; display: block; height: 0; }
    }
    .print-btn {
        background-color: #FFCD11; color: black; border: 2px solid black; 
        padding: 10px 20px; font-weight: bold; cursor: pointer; width: 100%; margin-bottom: 20px;
    }
    .warning-box {
        background-color: #ffcccc; border: 1px solid #ff0000; padding: 10px; border-radius: 5px; color: #990000; margin-bottom: 10px;
    }
</style>
""", unsafe_allow_html=True)

# ==============================================================================
# 1. CORE ALGORITHMS
# ==============================================================================

def calculate_reliability(n_needed, n_total, maint_pct, for_pct):
    p_avail = 1.0 - ((maint_pct/100.0) + (for_pct/100.0))
    prob_success = 0.0
    for k in range(n_needed, n_total + 1):
        prob_success += binom.pmf(k, n_total, p_avail)
    return prob_success

def solve_topology_v2(inputs):
    res = {'warnings': []} # Store validation warnings
    
    # --- A. LOAD CALCULATIONS ---
    p_it = inputs['p_it']
    p_facility = p_it * (1 + (inputs['dc_aux']/100.0)) 
    p_gen_term = p_facility / (1 - (inputs['dist_loss']/100.0))
    p_gross_req = p_gen_term / (1 - (inputs['gen_parasitic']/100.0))
    
    # --- B. GENERATOR SIZING ---
    gen_mw = inputs['gen_rating']
    derate = 1.0 
    if inputs['derate_mode'] == 'Auto-Calculate':
        derate = 1.0 - (max(0, (inputs['temp'] - 25)*0.01) + max(0, (inputs['alt'] - 100)*0.0001))
    
    gen_site_cap = gen_mw * derate
    n_run = math.ceil(p_gross_req / gen_site_cap)
    
    target_avail = inputs['req_avail'] / 100.0
    n_reserve = 0
    calculated_avail = 0.0
    
    for r in range(0, 20):
        avail = calculate_reliability(n_run, n_run + r, inputs['gen_maint'], inputs['gen_for'])
        if avail >= target_avail:
            n_reserve = r
            calculated_avail = avail
            break
        n_reserve = r
        calculated_avail = avail
            
    n_total_gens = n_run + n_reserve
    
    res['gen'] = {
        'load_gross': p_gross_req,
        'n_run': n_run,
        'n_reserve': n_reserve,
        'total_units': n_total_gens,
        'site_cap': gen_site_cap,
        'calc_avail': calculated_avail
    }
    
    # --- C. VOLTAGE LOGIC (SMART SELECTION) ---
    
    def calc_amps(mw, kv):
        return (mw * 1e6) / (math.sqrt(3) * (kv * 1000) * 0.8)

    # 1. Voltage Decision
    voltage_kv = inputs['volts_kv'] # Default manual
    
    if inputs['volts_mode'] == 'Auto-Recommend':
        # Smart Logic: Iterate voltages and pick first viable
        candidates = [0.48, 4.16, 13.8, 34.5]
        recommended = 13.8 # Fallback
        
        for v in candidates:
            total_amps = calc_amps(p_gross_req, v)
            # Criteria: Manageable total current (e.g., < 5000A is single bus feasible, < 20000A is multi-bus feasible)
            # If amps are massive (>20kA), we need higher voltage to avoid excessive copper/busbars.
            if total_amps < 10000: # Below 10kA total is a reasonable distribution limit
                recommended = v
                break
        
        voltage_kv = recommended
        
    # --- D. ELECTRICAL PHYSICS CHECK ---
    
    i_nom_gen = (gen_site_cap * 1e6) / (math.sqrt(3) * (voltage_kv * 1000) * 0.8)
    i_sc_gen = i_nom_gen / inputs['gen_xd']
    
    # Validation for Warnings (The "Intelligence" Layer)
    total_sys_amps = n_total_gens * i_nom_gen
    
    if total_sys_amps > 12000: # e.g. 10MW @ 480V = 15kA continuous
        res['warnings'].append(f"⚠️ **High Current Warning:** Total system current is {total_sys_amps:,.0f} A. This voltage ({voltage_kv} kV) may require excessive cabling and busbar copper. Consider increasing voltage.")
        
    if voltage_kv <= 0.6 and p_gross_req > 5.0:
        res['warnings'].append(f"⚠️ **Low Voltage Warning:** Generating {p_gross_req:.1f} MW at {voltage_kv*1000}V is inefficient. Recommended: Medium Voltage (13.8 kV).")

    # --- E. CLUSTERING (SWITCHGEAR SIZING) ---
    
    MAX_BUS_AMP = 4000.0
    MAX_BUS_KA = 63000.0 
    
    fit_amp = math.floor((MAX_BUS_AMP * 0.9) / i_nom_gen)
    fit_ka = math.floor((MAX_BUS_KA * 0.95) / i_sc_gen)
    
    # If chosen voltage creates massive SC, warn user
    if i_sc_gen > 63000: # Single gen exceeds breaker cap (unlikely but possible at LV large scale)
        res['warnings'].append("🛑 **Critical Safety:** Single Generator Short Circuit contribution exceeds standard 63kA rating. Reactors required.")
    
    max_gens_per_bus = max(1, min(fit_amp, fit_ka))
    
    num_swgr = math.ceil(n_total_gens / max_gens_per_bus)
    gens_per_swgr = math.ceil(n_total_gens / num_swgr)
    
    bus_load_amps = gens_per_swgr * i_nom_gen
    bus_fault_ka = (gens_per_swgr * i_sc_gen) / 1000.0
    
    std_ratings = [1200, 2000, 3000, 4000, 5000]
    rec_bus_rating = next((x for x in std_ratings if x >= bus_load_amps), 5000)
    rec_gen_cb = next((x for x in std_ratings if x >= i_nom_gen), 1200)
    
    res['elec'] = {
        'voltage': voltage_kv,
        'i_gen': i_nom_gen,
        'i_sc_gen': i_sc_gen,
        'num_swgr': num_swgr,
        'gens_per_swgr': gens_per_swgr,
        'bus_amps_actual': bus_load_amps,
        'bus_ka_actual': bus_fault_ka,
        'rec_rating': rec_bus_rating,
        'rec_gen_cb': rec_gen_cb
    }
    
    # --- F. BESS SIZING ---
    res['bess'] = {'active': False}
    step_load_mw = p_it * (inputs['step_load_req']/100.0)
    gen_step_cap = (n_run * gen_site_cap) * (inputs['gen_step_cap']/100.0)
    
    need_bess = (gen_step_cap < step_load_mw) or (inputs['bess_manual_active'])
    
    if need_bess:
        mw_shortfall = max(0, step_load_mw - gen_step_cap)
        if inputs['bess_manual_active']: mw_shortfall = max(mw_shortfall, inputs['bess_manual_mw'])
        
        n_bess_run = math.ceil(mw_shortfall / inputs['bess_inv_mw'])
        
        n_bess_res = 0
        bess_avail = 0.0
        for r in range(0, 10):
            avail = calculate_reliability(n_bess_run, n_bess_run + r, inputs['bess_maint'], inputs['bess_for'])
            if avail >= target_avail:
                n_bess_res = r
                bess_avail = avail
                break
            n_bess_res = r
            bess_avail = avail
            
        total_bess = n_bess_run + n_bess_res
        
        res['bess'] = {
            'active': True,
            'reason': "Step Load Support" if gen_step_cap < step_load_mw else "Manual Request",
            'total_units': total_bess,
            'n_run': n_bess_run,
            'n_res': n_bess_res,
            'total_mw': total_bess * inputs['bess_inv_mw'],
            'total_mwh': total_bess * inputs['bess_cap_mwh'],
            'per_bus': math.ceil(total_bess / num_swgr)
        }

    # --- G. DISTRIBUTION ---
    feeder_cap_mw = (math.sqrt(3) * voltage_kv * 600 * 0.9) / 1000 
    total_feeders = math.ceil(p_gross_req / feeder_cap_mw)
    feeders_per_bus = math.ceil(total_feeders / num_swgr)
    
    res['dist'] = {
        'total_feeders': total_feeders,
        'feeders_per_bus': feeders_per_bus,
        'feeder_amp_rating': 1200
    }
    
    return res

# ==============================================================================
# 2. UI & INPUTS
# ==============================================================================

if 'design_v2' not in st.session_state:
    st.session_state['design_v2'] = {
        # Profile
        'dc_type': 'AI Factory', 'p_it': 100.0, 'dc_aux': 15.0, 'req_avail': 99.999, 'step_load_req': 40.0,
        'volts_mode': 'Auto-Recommend', 'volts_kv': 13.8,
        'derate_mode': 'Auto-Calculate', 'temp': 35, 'alt': 100,
        # Gens
        'gen_rating': 2.5, 'dist_loss': 1.5, 'gen_parasitic': 3.0, 'gen_step_cap': 25.0, 'gen_xd': 0.14,
        'gen_maint': 4.0, 'gen_for': 1.0,
        # BESS
        'bess_manual_active': True, 'bess_manual_mw': 20.0, 'bess_inv_mw': 3.8, 'bess_cap_mwh': 5.0,
        'bess_maint': 2.0, 'bess_for': 0.5
    }

def get(k): return st.session_state['design_v2'][k]
def set_k(k, v): st.session_state['design_v2'][k] = v

with st.sidebar:
    st.title("Inputs")
    
    # 1. Data Center Profile
    with st.expander("1. Data Center Profile", expanded=True):
        st.selectbox("Data Center Type", ["Hyperscale", "AI Factory", "Colocation", "Enterprise"], key='dc_type', on_change=lambda: set_k('dc_type', st.session_state.dc_type))
        st.number_input("Critical IT Load (MW)", 1.0, 1000.0, float(get('p_it')), key='p_it', on_change=lambda: set_k('p_it', st.session_state.p_it))
        st.number_input("DC Aux / PUE Overhead (%)", 0.0, 100.0, float(get('dc_aux')), key='dc_aux', on_change=lambda: set_k('dc_aux', st.session_state.dc_aux))
        st.number_input("Required Availability (%)", 90.0, 99.9999, float(get('req_avail')), format="%.4f", key='req_avail', on_change=lambda: set_k('req_avail', st.session_state.req_avail))
        st.number_input("Step Load Req (%)", 0.0, 100.0, float(get('step_load_req')), key='step_load_req', on_change=lambda: set_k('step_load_req', st.session_state.step_load_req))
        
        c1, c2 = st.columns(2)
        v_mode = c1.selectbox("Voltage Selection", ["Auto-Recommend", "Manual"], index=0 if get('volts_mode')=='Auto-Recommend' else 1, key='v_mode', on_change=lambda: set_k('volts_mode', st.session_state.v_mode))
        
        if v_mode == 'Manual':
            c2.number_input("Manual Voltage (kV)", 0.208, 230.0, float(get('volts_kv')), key='v_kv', on_change=lambda: set_k('volts_kv', st.session_state.v_kv))
        else:
            c2.info("Will optimize for Ampacity")
            
        d_mode = st.selectbox("Derate Mode", ["Auto-Calculate", "Manual"], index=0 if get('derate_mode')=='Auto-Calculate' else 1)
        if d_mode == 'Auto-Calculate':
            c1, c2 = st.columns(2)
            c1.slider("Temp (°C)", 0, 55, int(get('temp')), key='temp', on_change=lambda: set_k('temp', st.session_state.temp))
            c2.number_input("Alt (m)", 0, 3000, int(get('alt')), key='alt', on_change=lambda: set_k('alt', st.session_state.alt))

    # 2. Generators
    with st.expander("2. Generators", expanded=True):
        c1, c2 = st.columns(2)
        c1.number_input("Rating (ISO MW)", 0.5, 20.0, float(get('gen_rating')), key='g_rat', on_change=lambda: set_k('gen_rating', st.session_state.g_rat))
        c2.number_input("Step Cap (%)", 0.0, 100.0, float(get('gen_step_cap')), key='g_step', on_change=lambda: set_k('gen_step_cap', st.session_state.g_step))
        
        c1, c2 = st.columns(2)
        c1.number_input("Xd\" (p.u.)", 0.05, 0.5, float(get('gen_xd')), key='g_xd', on_change=lambda: set_k('gen_xd', st.session_state.g_xd))
        c2.empty() 
        
        st.caption("Losses & Parasitics")
        c1, c2 = st.columns(2)
        c1.number_input("Dist. Loss (%)", 0.0, 10.0, float(get('dist_loss')), key='d_loss', on_change=lambda: set_k('dist_loss', st.session_state.d_loss))
        c2.number_input("Parasitics (%)", 0.0, 10.0, float(get('gen_parasitic')), key='g_par', on_change=lambda: set_k('gen_parasitic', st.session_state.g_par))
        
        st.caption("Reliability Factors")
        c1, c2 = st.columns(2)
        c1.number_input("Maint (%)", 0.0, 20.0, float(get('gen_maint')), key='g_maint', on_change=lambda: set_k('gen_maint', st.session_state.g_maint))
        c2.number_input("FOR (%)", 0.0, 20.0, float(get('gen_for')), key='g_for', on_change=lambda: set_k('gen_for', st.session_state.g_for))

    # 3. BESS
    with st.expander("3. BESS Strategy"):
        st.checkbox("Force BESS Inclusion", value=get('bess_manual_active'), key='b_act', on_change=lambda: set_k('bess_manual_active', st.session_state.b_act))
        c1, c2 = st.columns(2)
        c1.number_input("Inv. MW", 0.5, 6.0, float(get('bess_inv_mw')), key='b_inv', on_change=lambda: set_k('bess_inv_mw', st.session_state.b_inv))
        c2.number_input("Unit MWh", 1.0, 10.0, float(get('bess_cap_mwh')), key='b_cap', on_change=lambda: set_k('bess_cap_mwh', st.session_state.b_cap))
        
        st.caption("Reliability Factors")
        c1, c2 = st.columns(2)
        c1.number_input("BESS Maint (%)", 0.0, 10.0, float(get('bess_maint')), key='b_maint', on_change=lambda: set_k('bess_maint', st.session_state.b_maint))
        c2.number_input("BESS FOR (%)", 0.0, 10.0, float(get('bess_for')), key='b_for', on_change=lambda: set_k('bess_for', st.session_state.b_for))

# CALCULATION TRIGGER
res = solve_topology_v2(st.session_state['design_v2'])

# ==============================================================================
# 3. MAIN DASHBOARD
# ==============================================================================

st.title("CAT Electrical Topology Designer v2.1")
st.caption(f"Configured for: {get('dc_type')} | Target Availability: {get('req_avail')}%")

# WARNINGS DISPLAY
if res['warnings']:
    for w in res['warnings']:
        st.error(w)

# --- EXECUTIVE SUMMARY ---
col1, col2, col3, col4 = st.columns(4)
col1.metric("Gross Load Required", f"{res['gen']['load_gross']:.1f} MW", f"IT Load: {get('p_it')} MW")
col2.metric("Generation Fleet", f"{res['gen']['total_units']} Units", f"N+{res['gen']['n_reserve']} Redundancy")
col3.metric("Calculated Availability", f"{res['gen']['calc_avail']*100:.5f}%", f"Target: {get('req_avail')}%")
col4.metric("System Voltage", f"{res['elec']['voltage']} kV", "Manual" if get('volts_mode')=='Manual' else "Auto-Selected")

st.divider()

# --- TABS ---
t_arch, t_specs, t_report = st.tabs(["📐 Topology Architecture", "📋 Equipment Specs", "📄 Engineering Report"])

with t_arch:
    st.subheader("Auto-Generated Single Line Diagram Structure")
    
    # GRAPHVIZ
    dot = graphviz.Digraph()
    dot.attr(rankdir='TB', compound='true')
    
    dot.node('RING', f'Distribution Ring / Utility\n{res["elec"]["voltage"]} kV', shape='doubleoctagon', fillcolor='#E0E0E0', style='filled')
    
    num_swgr = res['elec']['num_swgr']
    gens_per = res['elec']['gens_per_swgr']
    
    cols = st.columns(min(num_swgr, 4)) 
    for i, c in enumerate(cols):
        c.info(f"**Bus {i+1}**: {gens_per} Gens")
    
    for b in range(1, num_swgr + 1):
        bus_name = f'SWGR_{b}'
        label = f"Switchgear {b}\n{res['elec']['rec_rating']} A | {res['elec']['bus_ka_actual']:.1f} kA"
        dot.node(bus_name, label, shape='rect', fillcolor='#FFCD11', style='filled', width='3')
        
        dot.edge(bus_name, 'RING', label=f"{res['dist']['feeders_per_bus']}x Feeders")
        
        gen_grp = f'GEN_GRP_{b}'
        dot.node(gen_grp, f"{gens_per}x Generators\n{get('gen_rating')} MW", shape='folder')
        dot.edge(gen_grp, bus_name)
        
        if res['bess']['active']:
            bess_grp = f'BESS_GRP_{b}'
            bess_count = res['bess']['per_bus']
            dot.node(bess_grp, f"{bess_count}x BESS Units", shape='component', fillcolor='#90EE90', style='filled')
            dot.edge(bess_grp, bus_name)

    st.graphviz_chart(dot, use_container_width=True)

with t_specs:
    c1, c2 = st.columns(2)
    
    with c1:
        st.markdown("### 🏭 Switchgear Definition")
        df_sw = pd.DataFrame({
            "Parameter": ["Total Switchgears (Buses)", "Generators per Bus", "Calculated Bus Current", "Recommended Bus Rating", "Calculated Short Circuit", "Rec. Short Circuit Rating"],
            "Value": [
                f"{res['elec']['num_swgr']}",
                f"{res['elec']['gens_per_swgr']}",
                f"{res['elec']['bus_amps_actual']:.0f} A",
                f"**{res['elec']['rec_rating']} A**",
                f"{res['elec']['bus_ka_actual']:.1f} kA",
                f"**63 kA**" if res['elec']['bus_ka_actual'] > 40 else "**40 kA**"
            ]
        })
        st.table(df_sw)
        
        st.markdown("### 🔌 Distribution")
        st.info(f"**Total Feeders Required:** {res['dist']['total_feeders']} Circuits (approx {res['dist']['feeders_per_bus']} per bus)")
        
    with c2:
        st.markdown("### ⚙️ Generation Details")
        st.table(pd.DataFrame({
            "Parameter": ["Rated Capacity (ISO)", "Site Derated Capacity", "Run Requirement", "Redundancy Calculated", "Total Fleet Count"],
            "Value": [
                f"{get('gen_rating')} MW",
                f"{res['gen']['site_cap']:.3f} MW",
                f"{res['gen']['n_run']} Units",
                f"N+{res['gen']['n_reserve']}",
                f"**{res['gen']['total_units']} Units**"
            ]
        }))
        
        if res['bess']['active']:
            st.markdown("### 🔋 BESS Details")
            st.table(pd.DataFrame({
                "Parameter": ["Total Containers", "Inverter Total", "Energy Total", "Configuration"],
                "Value": [
                    f"{res['bess']['total_units']}",
                    f"{res['bess']['total_mw']:.1f} MW",
                    f"{res['bess']['total_mwh']:.1f} MWh",
                    f"{res['bess']['n_run']} Run + {res['bess']['n_res']} Spare"
                ]
            }))

with t_report:
    components.html("""
    <script>function printRep(){window.parent.print();}</script>
    <div style="text-align:center; margin:20px;"><button onclick="printRep()" class="print-btn">🖨️ Print Engineering Report</button></div>
    <style>.print-btn{background:#FFCD11;border:2px solid black;padding:10px 20px;font-weight:bold;cursor:pointer;}</style>
    """, height=80)
    
    st.header("Engineering Topology Report")
    st.caption(f"Generated on {time.strftime('%Y-%m-%d')}")
    
    if res['warnings']:
        st.markdown("### ⚠️ Design Warnings")
        for w in res['warnings']:
            st.warning(w)
    
    st.markdown("#### 1. Executive Summary")
    st.write(f"The facility requires a Gross Generation Capacity of **{res['gen']['load_gross']:.1f} MW** to support a Critical IT Load of **{get('p_it')} MW**.")
    st.write(f"To meet the availability target of **{get('req_avail')}%**, the algorithm recommends a fleet of **{res['gen']['total_units']} generators** (N+{res['gen']['n_reserve']}).")
    
    st.markdown("#### 2. Electrical Architecture")
    st.write(f"Due to physical constraints (Ampacity/kA), the system is segmented into **{res['elec']['num_swgr']} Parallel Switchgear Buses**.")
    st.write(f"- **Voltage:** {res['elec']['voltage']} kV")
    st.write(f"- **Bus Rating:** {res['elec']['rec_rating']} A")
    
    st.markdown("#### 3. BESS Integration")
    if res['bess']['active']:
        st.write(f"A BESS system of **{res['bess']['total_units']} containers** is included for {res['bess']['reason']}.")
    else:
        st.write("BESS not required for this topology.")
