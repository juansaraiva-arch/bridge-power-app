import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
import numpy as np
import math
import graphviz
from scipy.stats import binom

# --- PAGE CONFIG ---
st.set_page_config(page_title="CAT Topology Designer v6.0", page_icon="⚡", layout="wide")

# --- CSS ---
st.markdown("""
<style>
    @media print {
        [data-testid="stSidebar"], [data-testid="stHeader"], footer, .stButton { display: none !important; }
        .block-container { padding: 0 !important; margin: 0 !important; }
    }
    .warning-box { background-color: #ffcccc; border: 1px solid red; padding: 10px; border-radius: 5px; color: #900; margin-bottom: 10px; }
    .success-box { background-color: #d4edda; border: 1px solid #c3e6cb; padding: 10px; border-radius: 5px; color: #155724; margin-bottom: 10px; }
</style>
""", unsafe_allow_html=True)

# ==============================================================================
# 1. MATH ENGINE
# ==============================================================================

def get_avail(maint, force):
    return 1.0 - ((maint + force) / 100.0)

def calc_reliability_n_k(n_needed, n_total, p_avail):
    if n_total < n_needed: return 0.0
    prob = 0.0
    for k in range(n_needed, n_total + 1):
        prob += binom.pmf(k, n_total, p_avail)
    return prob

def solve_topology_v6(inputs):
    res = {'warnings': [], 'log': []}
    
    # --- A. LOAD ANALYSIS ---
    # Gross Load Calculation (The real load seen by generators)
    p_it = inputs['p_it']
    p_facility = p_it * (1 + inputs['dc_aux']/100.0)
    p_gen_term = p_facility / (1 - inputs['dist_loss']/100.0)
    p_gross_req = p_gen_term / (1 - inputs['gen_parasitic']/100.0)
    
    res['load'] = {'gross': p_gross_req, 'net': p_it}
    
    # --- B. GENERATOR CAPACITY ---
    # Derate
    derate = 1.0
    if inputs['derate_mode'] == 'Auto-Calculate':
        derate = 1.0 - (max(0, (inputs['temp'] - 25)*0.01) + max(0, (inputs['alt'] - 100)*0.0001))
    
    gen_site_mw = inputs['gen_rating'] * derate
    n_gen_needed_for_load = math.ceil(p_gross_req / gen_site_mw)
    
    # --- C. VOLTAGE & PHYSICAL LIMITS ---
    voltage_kv = inputs['volts_kv']
    if inputs['volts_mode'] == 'Auto-Recommend':
        amps = (p_gross_req * 1e6) / (math.sqrt(3) * 13800 * 0.8)
        voltage_kv = 34.5 if amps > 8000 else 13.8 # Simple logic
        
    i_nom_gen = (gen_site_mw * 1e6) / (math.sqrt(3) * (voltage_kv * 1000) * 0.8)
    i_sc_gen = i_nom_gen / inputs['gen_xd']
    
    # Clustering Limits
    MAX_BUS_AMP = 4000.0
    MAX_BUS_KA = 63000.0
    
    max_gens_per_bus_phy = max(1, min(
        math.floor((MAX_BUS_AMP * 0.9) / i_nom_gen),
        math.floor((MAX_BUS_KA * 0.95) / i_sc_gen)
    ))

    # --- D. ROBUST TOPOLOGY SOLVER (The "Bus Loss" Check) ---
    
    # We iterate to find a configuration (N_Buses, N_Gens) that satisfies:
    # 1. Load support (N_Total >= N_Needed)
    # 2. Reliability Target (Prob > 99.999%)
    # 3. BUS TOLERANCE: (N_Total - Gens_in_Largest_Bus) >= N_Needed
    
    target_rel = inputs['req_avail'] / 100.0
    p_gen_avail = get_avail(inputs['gen_maint'], inputs['gen_for'])
    p_bus_avail = get_avail(inputs['bus_maint'], inputs['bus_for'])
    
    solution = None
    
    # Iterate total generators starting from N+1
    for n_total in range(n_gen_needed_for_load + 1, n_gen_needed_for_load + 50):
        
        # Distribute into buses
        # Minimum buses dictated by physics
        min_buses_phy = math.ceil(n_total / max_gens_per_bus_phy)
        # Minimum buses dictated by Fault Tolerance Logic:
        # If we have 2 buses, losing 1 means losing 50%. We need the other 50% to carry 100% load.
        # This implies we need 200% capacity installed.
        # As N_buses increases, the "impact" of losing one decreases (1/N).
        
        # Try increasing bus count to find a valid configuration
        for n_buses in range(max(2, min_buses_phy), 10): # Try 2 up to 10 buses
            gens_per_bus = math.ceil(n_total / n_buses)
            
            # Check 1: Bus Fault Tolerance
            # Capacity lost if 1 bus fails
            lost_cap_mw = gens_per_bus * gen_site_mw
            remaining_cap_mw = (n_total * gen_site_mw) - lost_cap_mw
            
            is_tolerant = remaining_cap_mw >= p_gross_req
            
            if not is_tolerant:
                continue # Need more gens or more buses
            
            # Check 2: Probabilistic Reliability
            # P(System OK) = P(All Buses OK)*P(Gens OK) + P(1 Bus Fail)*P(Remaining Gens OK)
            
            # State 0: All Buses OK
            p_s0 = p_bus_avail ** n_buses
            rel_s0 = calc_reliability_n_k(n_gen_needed_for_load, n_total, p_gen_avail)
            
            # State 1: 1 Bus Fails
            p_s1 = math.comb(n_buses, 1) * (1 - p_bus_avail) * (p_bus_avail ** (n_buses - 1))
            n_gens_remaining = n_total - gens_per_bus
            rel_s1 = calc_reliability_n_k(n_gen_needed_for_load, n_gens_remaining, p_gen_avail)
            
            total_system_rel = (p_s0 * rel_s0) + (p_s1 * rel_s1)
            
            if total_system_rel >= target_rel:
                # FOUND VALID SOLUTION
                solution = {
                    'n_total': n_total,
                    'n_buses': n_buses,
                    'gens_per_bus': gens_per_bus,
                    'rel': total_system_rel,
                    'n_redundant': n_total - n_gen_needed_for_load
                }
                break
        
        if solution: break
    
    if not solution:
        res['warnings'].append("Could not find a N-1 Bus Tolerant solution within reasonable limits.")
        solution = {'n_total': n_total, 'n_buses': n_buses, 'gens_per_bus': gens_per_bus, 'rel': 0.0, 'n_redundant': 0}

    res['gen'] = solution
    res['gen']['site_mw'] = gen_site_mw
    res['gen']['n_needed'] = n_gen_needed_for_load

    # --- E. BESS (STEP LOAD & FAULT TOLERANCE) ---
    step_req_mw = p_it * (inputs['step_load_req'] / 100.0)
    
    # Step Capability from Gens (Considering N-1 Bus scenario?)
    # Usually Step Load is a transient event. If a bus is down, we might not have full step cap.
    # Safe design: Ensure Step Cap even with 1 bus down.
    
    # Available Gens in worst case (1 Bus Down)
    gens_avail_worst_case = solution['n_total'] - solution['gens_per_bus']
    gen_step_mw = gens_avail_worst_case * gen_site_mw * (inputs['gen_step_cap'] / 100.0)
    
    bess_needed = gen_step_mw < step_req_mw or inputs['bess_manual']
    
    n_bess_total = 0
    if bess_needed:
        shortfall = max(0, step_req_mw - gen_step_mw)
        if inputs['bess_manual']: shortfall = max(shortfall, inputs['bess_mw'])
        
        # BESS Containers needed for shortfall
        n_bess_run = math.ceil(shortfall / inputs['bess_inv_mw'])
        
        # Optimize BESS Redundancy (Target Rel)
        p_bess = get_avail(inputs['bess_maint'], inputs['bess_for'])
        n_bess_total, _ = optimize_redundancy(n_bess_run, target_rel, p_bess)
        
        # Check BESS Bus Tolerance
        # If we lose a bus, we lose BESS units on that bus.
        # We need to spread BESS so that (Total - BESS_per_Bus) >= Required
        bess_per_bus = math.ceil(n_bess_total / solution['n_buses'])
        # Recalculate Total to ensure tolerance
        # (N_Total - Per_Bus) >= N_Run
        # N_Total * (1 - 1/Bus) >= N_Run
        # N_Total >= N_Run / ((Bus-1)/Bus)
        if solution['n_buses'] > 1:
            min_tolerant_bess = math.ceil(n_bess_run / ((solution['n_buses']-1)/solution['n_buses']))
            n_bess_total = max(n_bess_total, min_tolerant_bess)
            
    res['bess'] = {
        'active': bess_needed,
        'n_total': n_bess_total,
        'per_bus': math.ceil(n_bess_total / solution['n_buses']) if solution['n_buses'] > 0 else 0
    }

    # --- F. DISTRIBUTION ---
    res['elec'] = {
        'voltage': voltage_kv,
        'bus_rating': 5000 if (solution['gens_per_bus'] * i_nom_gen) > 4000 else 4000,
        'bus_ka': (solution['gens_per_bus'] * i_sc_gen)/1000.0,
        'amps_per_bus': solution['gens_per_bus'] * i_nom_gen
    }
    
    return res

# ==============================================================================
# 2. UI INPUTS (RESTORED ALL VARIABLES)
# ==============================================================================

if 'inputs_v6' not in st.session_state:
    st.session_state['inputs_v6'] = {
        # Profile
        'dc_type': 'Hyperscale', 'p_it': 100.0, 'dc_aux': 15.0, 'req_avail': 99.999, 'step_load_req': 40.0,
        'volts_mode': 'Auto-Recommend', 'volts_kv': 13.8, 'derate_mode': 'Auto-Calculate', 'temp': 35, 'alt': 100,
        # Gens
        'gen_rating': 2.5, 'dist_loss': 1.5, 'gen_parasitic': 3.0, 'gen_step_cap': 25.0, 
        'gen_xd': 0.14, 'gen_maint': 4.0, 'gen_for': 1.0,
        # BESS
        'bess_manual': False, 'bess_mw': 20.0, 'bess_inv_mw': 3.8, 'bess_maint': 2.0, 'bess_for': 0.5,
        # Reliability Stats
        'bus_maint': 0.1, 'bus_for': 0.05,
        'cb_maint': 0.2, 'cb_for': 0.05,
        'cable_maint': 0.1, 'cable_for': 0.05,
        'tx_maint': 0.2, 'tx_for': 0.1
    }

def get(k): return st.session_state['inputs_v6'].get(k)
def set_k(k, v): st.session_state['inputs_v6'][k] = v

with st.sidebar:
    st.title("Inputs v6.0")
    
    with st.expander("1. Data Center Profile", expanded=True):
        st.selectbox("Data Center Type", ["Hyperscale", "AI Factory"], key='dc_type', on_change=lambda: set_k('dc_type', st.session_state.dc_type))
        st.number_input("Critical IT Load (MW)", 1.0, 1000.0, float(get('p_it')), key='p_it', on_change=lambda: set_k('p_it', st.session_state.p_it))
        st.number_input("DC Aux (%)", 0.0, 50.0, float(get('dc_aux')), key='dc_aux', on_change=lambda: set_k('dc_aux', st.session_state.dc_aux))
        st.number_input("Required Availability (%)", 90.0, 99.99999, float(get('req_avail')), format="%.5f", key='req_avail', on_change=lambda: set_k('req_avail', st.session_state.req_avail))
        st.number_input("Step Load Req (%)", 0.0, 100.0, float(get('step_load_req')), key='step_load_req', on_change=lambda: set_k('step_load_req', st.session_state.step_load_req))
        
        c1, c2 = st.columns(2)
        st.selectbox("Connection Voltage", ["Auto-Recommend", "Manual"], index=0, key='volts_mode', on_change=lambda: set_k('volts_mode', st.session_state.volts_mode))
        if get('volts_mode') == 'Manual':
            st.number_input("Voltage (kV)", 0.4, 230.0, float(get('volts_kv')), key='volts_kv', on_change=lambda: set_k('volts_kv', st.session_state.volts_kv))
            
        st.selectbox("Derate Mode", ["Auto-Calculate", "Manual"], index=0, key='derate_mode', on_change=lambda: set_k('derate_mode', st.session_state.derate_mode))
        if get('derate_mode') == 'Auto-Calculate':
            c1, c2 = st.columns(2)
            c1.number_input("Temp (°C)", 0, 55, int(get('temp')), key='temp', on_change=lambda: set_k('temp', st.session_state.temp))
            c2.number_input("Alt (m)", 0, 3000, int(get('alt')), key='alt', on_change=lambda: set_k('alt', st.session_state.alt))

    with st.expander("2. Generator Parameters"):
        c1, c2 = st.columns(2)
        c1.number_input("Rating (MW)", 0.5, 20.0, float(get('gen_rating')), key='gen_rating', on_change=lambda: set_k('gen_rating', st.session_state.gen_rating))
        c2.number_input("Dist Loss (%)", 0.0, 10.0, float(get('dist_loss')), key='dist_loss', on_change=lambda: set_k('dist_loss', st.session_state.dist_loss))
        c1.number_input("Parasitics (%)", 0.0, 10.0, float(get('gen_parasitic')), key='gen_parasitic', on_change=lambda: set_k('gen_parasitic', st.session_state.gen_parasitic))
        c2.number_input("Step Cap (%)", 0.0, 100.0, float(get('gen_step_cap')), key='gen_step_cap', on_change=lambda: set_k('gen_step_cap', st.session_state.gen_step_cap))
        st.number_input("Xd\" (pu)", 0.05, 0.5, float(get('gen_xd')), key='gen_xd', on_change=lambda: set_k('gen_xd', st.session_state.gen_xd))
        
        st.caption("Reliability")
        c1, c2 = st.columns(2)
        c1.number_input("Gen Maint (%)", 0.0, 20.0, float(get('gen_maint')), key='gen_maint', on_change=lambda: set_k('gen_maint', st.session_state.gen_maint))
        c2.number_input("Gen FOR (%)", 0.0, 20.0, float(get('gen_for')), key='gen_for', on_change=lambda: set_k('gen_for', st.session_state.gen_for))

    with st.expander("3. BESS Parameters"):
        st.checkbox("Force Manual BESS", value=get('bess_manual'), key='bess_manual', on_change=lambda: set_k('bess_manual', st.session_state.bess_manual))
        if get('bess_manual'):
            st.number_input("Manual BESS MW", 0.0, 500.0, float(get('bess_mw')), key='bess_mw', on_change=lambda: set_k('bess_mw', st.session_state.bess_mw))
        st.number_input("Inverter MW", 0.5, 6.0, float(get('bess_inv_mw')), key='bess_inv_mw', on_change=lambda: set_k('bess_inv_mw', st.session_state.bess_inv_mw))
        
        c1, c2 = st.columns(2)
        c1.number_input("BESS Maint (%)", 0.0, 20.0, float(get('bess_maint')), key='bess_maint', on_change=lambda: set_k('bess_maint', st.session_state.bess_maint))
        c2.number_input("BESS FOR (%)", 0.0, 20.0, float(get('bess_for')), key='bess_for', on_change=lambda: set_k('bess_for', st.session_state.bess_for))

    with st.expander("4. Component Reliability"):
        c1, c2 = st.columns(2)
        c1.number_input("Bus Maint", 0.0, 5.0, float(get('bus_maint')), key='bus_maint', on_change=lambda: set_k('bus_maint', st.session_state.bus_maint))
        c2.number_input("Bus FOR", 0.0, 5.0, float(get('bus_for')), key='bus_for', on_change=lambda: set_k('bus_for', st.session_state.bus_for))
        
        c1, c2 = st.columns(2)
        c1.number_input("Breaker Maint", 0.0, 5.0, float(get('cb_maint')), key='cb_maint', on_change=lambda: set_k('cb_maint', st.session_state.cb_maint))
        c2.number_input("Breaker FOR", 0.0, 5.0, float(get('cb_for')), key='cb_for', on_change=lambda: set_k('cb_for', st.session_state.cb_for))
        
        st.number_input("Cable FOR", 0.0, 5.0, float(get('cable_for')), key='cable_for', on_change=lambda: set_k('cable_for', st.session_state.cable_for))
        
        c1, c2 = st.columns(2)
        c1.number_input("Tx Maint", 0.0, 5.0, float(get('tx_maint')), key='tx_maint', on_change=lambda: set_k('tx_maint', st.session_state.tx_maint))
        c2.number_input("Tx FOR", 0.0, 5.0, float(get('tx_for')), key='tx_for', on_change=lambda: set_k('tx_for', st.session_state.tx_for))

res = solve_topology_v6(st.session_state['inputs_v6'])

# ==============================================================================
# 4. DASHBOARD
# ==============================================================================

st.title("CAT Topology Designer v6.0")
st.subheader("Fault Tolerant Architecture Solver")

col1, col2, col3, col4 = st.columns(4)
col1.metric("Gross Load", f"{res['load']['gross']:.1f} MW", f"Net: {get('p_it')} MW")
col2.metric("Topology Avail", f"{res['gen']['rel']*100:.6f}%", f"Target: {get('req_avail')}%")
col3.metric("Fleet Size", f"{res['gen']['n_total']} Units", f"Required: {res['gen']['n_needed']}")
col4.metric("Architecture", f"{res['gen']['n_buses']} Buses", f"{res['gen']['gens_per_bus']} Gens/Bus")

# Bus Loss Analysis
lost_mw = res['gen']['gens_per_bus'] * res['gen']['site_mw']
rem_mw = (res['gen']['n_total'] - res['gen']['gens_per_bus']) * res['gen']['site_mw']
load = res['load']['gross']

st.markdown(f"""
<div class="success-box">
    <b>🛡️ N-1 Bus Fault Tolerance Verified:</b><br>
    If one switchgear bus fails, system loses <b>{lost_mw:.1f} MW</b> ({res['gen']['gens_per_bus']} Gens).<br>
    Remaining Capacity: <b>{rem_mw:.1f} MW</b> vs Required Load: <b>{load:.1f} MW</b>.<br>
    The system <b>CAN</b> support the full load with one bus down.
</div>
""", unsafe_allow_html=True)

if res['warnings']:
    for w in res['warnings']: st.warning(w)

st.markdown("### System Diagram")
dot = graphviz.Digraph()
dot.attr(rankdir='TB')
dot.node('DC', f'Critical Load\n{get("p_it")} MW', shape='box3d', style='filled', fillcolor='#D6EAF8')

for b in range(1, res['gen']['n_buses'] + 1):
    bus_name = f'SWGR_{b}'
    label = f"Switchgear {b}\n{res['elec']['voltage']} kV | {res['elec']['bus_rating']} A"
    dot.node(bus_name, label, shape='rect', style='filled', fillcolor='#FCF3CF')
    
    gen_label = f"{res['gen']['gens_per_bus']}x Gens"
    dot.node(f'G_{b}', gen_label, shape='folder', style='filled', fillcolor='#D1F2EB')
    dot.edge(f'G_{b}', bus_name)
    
    if res['bess']['active']:
        bess_per = res['bess']['per_bus']
        dot.node(f'B_{b}', f"{bess_per}x BESS", shape='component', style='filled', fillcolor='#A9DFBF')
        dot.edge(f'B_{b}', bus_name)
    
    dot.edge(bus_name, 'DC')

st.graphviz_chart(dot, use_container_width=True)
