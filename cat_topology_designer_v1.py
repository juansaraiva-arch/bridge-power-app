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
st.set_page_config(page_title="CAT Topology Designer v4.0", page_icon="⚡", layout="wide")

# --- CSS FOR PRINTING ---
st.markdown("""
<style>
    @media print {
        [data-testid="stSidebar"], [data-testid="stHeader"], .stApp > header, footer, .stButton { display: none !important; }
        .block-container { padding: 0 !important; margin: 0 !important; }
        .page-break { page-break-before: always !important; display: block; height: 50px; content: " "; }
        .js-plotly-plot { margin-bottom: 20px !important; width: 100% !important; }
    }
    .print-btn { background-color: #FFCD11; color: black; border: 2px solid black; padding: 12px; font-weight: bold; cursor: pointer; width: 100%; }
</style>
""", unsafe_allow_html=True)

# ==============================================================================
# 1. CORE RELIABILITY ENGINE
# ==============================================================================

def calculate_complex_reliability(
    n_needed, num_groups, units_per_group, 
    prob_unit_avail, prob_group_avail
):
    """
    Generic reliability calculator for grouped architectures (Buses -> Gens OR Buses -> Feeders).
    Calculates P(System Success) considering Group (Bus) failures AND Unit (Gen/Feeder) failures.
    """
    total_reliability = 0.0
    total_units = num_groups * units_per_group
    
    # State 0: All Groups Available
    p_state0 = prob_group_avail ** num_groups
    
    # Prob of N units working given State 0
    p_units_ok_s0 = 0.0
    for k in range(n_needed, total_units + 1):
        p_units_ok_s0 += binom.pmf(k, total_units, prob_unit_avail)
    
    total_reliability += p_state0 * p_units_ok_s0
    
    # State 1: One Group Failed (N-1 Groups)
    if num_groups > 1:
        p_state1 = math.comb(num_groups, 1) * (1 - prob_group_avail) * (prob_group_avail ** (num_groups - 1))
        
        units_avail_s1 = (num_groups - 1) * units_per_group
        if units_avail_s1 >= n_needed:
            p_units_ok_s1 = 0.0
            for k in range(n_needed, units_avail_s1 + 1):
                p_units_ok_s1 += binom.pmf(k, units_avail_s1, prob_unit_avail)
            
            total_reliability += p_state1 * p_units_ok_s1
            
    # Note: Neglecting State 2 (2 buses fail) as probability is typically < 1e-6 for high-reliability gear
    
    return total_reliability

def solve_topology_v4(inputs):
    res = {'warnings': []}
    
    # --- A. LOADS ---
    p_it = inputs['p_it']
    p_gross_req = (p_it * (1 + inputs['dc_aux']/100)) / ((1 - inputs['dist_loss']/100) * (1 - inputs['gen_parasitic']/100))
    
    # --- B. COMPONENT AVAILABILITY CALCULATIONS ---
    # Convert Maint/FOR % to Availability Probability (0.0 - 1.0)
    
    # 1. Generator Unit Avail
    a_gen = 1.0 - ((inputs['gen_maint'] + inputs['gen_for']) / 100.0)
    
    # 2. Switchgear Bus Avail
    a_bus = 1.0 - ((inputs['bus_maint'] + inputs['bus_for']) / 100.0)
    
    # 3. Distribution Path Avail (Feeder Breaker + Cable + Transformer)
    # Series reliability: if any component fails, the path fails.
    u_cb = (inputs['cb_maint'] + inputs['cb_for']) / 100.0
    u_cable = (inputs['cable_maint'] + inputs['cable_for']) / 100.0
    u_tx = (inputs['tx_maint'] + inputs['tx_for']) / 100.0
    
    a_path = (1.0 - u_cb) * (1.0 - u_cable) * (1.0 - u_tx)
    
    target_avail = inputs['req_avail'] / 100.0
    
    # --- C. VOLTAGE & PHYSICS ---
    voltage_kv = inputs['volts_kv']
    if inputs['volts_mode'] == 'Auto-Recommend':
        if p_gross_req < 2.5: voltage_kv = 0.48
        elif p_gross_req < 15.0: voltage_kv = 4.16
        else: voltage_kv = 13.8 # Standard for large DCs
    
    # Generator Derate
    gen_mw = inputs['gen_rating']
    derate = 1.0 
    if inputs['derate_mode'] == 'Auto-Calculate':
        derate = 1.0 - (max(0, (inputs['temp'] - 25)*0.01) + max(0, (inputs['alt'] - 100)*0.0001))
    gen_site_cap = gen_mw * derate
    
    # --- D. UPSTREAM OPTIMIZATION (Generators & Buses) ---
    
    i_nom_gen = (gen_site_cap * 1e6) / (math.sqrt(3) * (voltage_kv * 1000) * 0.8)
    i_sc_gen = i_nom_gen / inputs['gen_xd']
    
    # Bus Constraints
    MAX_BUS_AMP = 4000.0
    MAX_BUS_KA = 63000.0
    max_gens_per_bus = max(1, min(
        math.floor((MAX_BUS_AMP * 0.9) / i_nom_gen),
        math.floor((MAX_BUS_KA * 0.95) / i_sc_gen)
    ))
    
    min_gens_load = math.ceil(p_gross_req / gen_site_cap)
    
    # Generator Loop
    final_gen_config = None
    
    for n_total in range(min_gens_load, min_gens_load + 20):
        n_buses = math.ceil(n_total / max_gens_per_bus)
        if target_avail > 0.999 and n_buses < 2: n_buses = 2 # Force redundancy for high tier
        
        gens_per_bus = math.ceil(n_total / n_buses)
        
        sys_rel = calculate_complex_reliability(
            min_gens_load, n_buses, gens_per_bus, a_gen, a_bus
        )
        
        if sys_rel >= target_avail:
            final_gen_config = {
                'total': n_buses * gens_per_bus,
                'n_buses': n_buses,
                'per_bus': gens_per_bus,
                'rel': sys_rel
            }
            break
            
    if not final_gen_config: # Fallback if loop finishes without hitting target
        final_gen_config = {'total': min_gens_load+20, 'n_buses': math.ceil((min_gens_load+20)/max_gens_per_bus), 'per_bus': max_gens_per_bus, 'rel': sys_rel}
        res['warnings'].append("⚠️ Generator Availability Target NOT met with Reasonable Redundancy.")

    # --- E. DOWNSTREAM OPTIMIZATION (Feeders & Trafos) ---
    
    # 1. Determine Capacity per Path
    # Standard Feeder: 1200A Breaker -> usually derated to 1000A or cable limit
    # Or Transformer limited.
    # Let's assume standard 2.5 MVA or 3 MVA distribution blocks for Hyperscale
    block_mva = 3.0 # Typical Substation / Transformer size
    block_mw = block_mva * 0.9 # PF
    
    min_paths_load = math.ceil(p_gross_req / block_mw)
    
    # 2. Distribution Sizing Loop
    final_dist_config = None
    
    # We distribute paths across the existing Buses determined in Step D
    n_buses_dist = final_gen_config['n_buses']
    
    for n_paths_total in range(min_paths_load, min_paths_load + 20):
        paths_per_bus = math.ceil(n_paths_total / n_buses_dist)
        
        dist_rel = calculate_complex_reliability(
            min_paths_load, n_buses_dist, paths_per_bus, a_path, a_bus
        )
        
        if dist_rel >= target_avail:
            final_dist_config = {
                'total': n_buses_dist * paths_per_bus,
                'per_bus': paths_per_bus,
                'rel': dist_rel,
                'block_mw': block_mw
            }
            break
            
    if not final_dist_config:
        final_dist_config = {'total': min_paths_load+10, 'per_bus': math.ceil((min_paths_load+10)/n_buses_dist), 'rel': dist_rel, 'block_mw': block_mw}
    
    # --- F. RESULTS ---
    
    res['gen'] = {
        'load_gross': p_gross_req,
        'total_units': final_gen_config['total'],
        'site_cap': gen_site_cap,
        'calc_avail': final_gen_config['rel']
    }
    
    bus_amps = final_gen_config['per_bus'] * i_nom_gen
    
    res['elec'] = {
        'voltage': voltage_kv,
        'num_swgr': final_gen_config['n_buses'],
        'gens_per_swgr': final_gen_config['per_bus'],
        'rec_rating': 5000 if bus_amps > 4000 else (4000 if bus_amps > 3000 else 3000),
        'bus_ka': final_gen_config['per_bus'] * i_sc_gen / 1000.0
    }
    
    res['dist'] = {
        'total_feeders': final_dist_config['total'],
        'feeders_per_bus': final_dist_config['per_bus'],
        'block_mw': final_dist_config['block_mw'],
        'calc_avail': final_dist_config['rel']
    }
    
    # BESS (Simplified pass-through for v4)
    res['bess'] = {'active': inputs['bess_manual_active'], 'total_units': 0}
    
    return res

# ==============================================================================
# 2. UI INPUTS
# ==============================================================================

if 'design_v4' not in st.session_state:
    st.session_state['design_v4'] = {
        'p_it': 100.0, 'dc_aux': 15.0, 'req_avail': 99.999, 'volts_mode': 'Auto-Recommend', 'volts_kv': 13.8,
        'derate_mode': 'Auto-Calculate', 'temp': 35, 'alt': 100,
        'gen_rating': 2.5, 'dist_loss': 1.5, 'gen_parasitic': 3.0, 'gen_xd': 0.14, 'gen_maint': 2.0, 'gen_for': 2.0,
        'bus_maint': 0.1, 'bus_for': 0.05,
        'cb_maint': 0.2, 'cb_for': 0.05,
        'cable_maint': 0.0, 'cable_for': 0.1,
        'tx_maint': 0.2, 'tx_for': 0.2,
        'bess_manual_active': False
    }

def get(k): return st.session_state['design_v4'].get(k, 0)
def set_k(k, v): st.session_state['design_v4'][k] = v

with st.sidebar:
    st.title("CAT Topology v4.0")
    
    with st.expander("1. Project & Load", expanded=True):
        st.number_input("IT Load (MW)", 10.0, 500.0, float(get('p_it')), key='p_it', on_change=lambda: set_k('p_it', st.session_state.p_it))
        st.number_input("Target Availability (%)", 90.0, 99.99999, float(get('req_avail')), format="%.5f", key='req_avail', on_change=lambda: set_k('req_avail', st.session_state.req_avail))
        
    with st.expander("2. Generation Tech"):
        st.number_input("Gen Rating (MW)", 1.0, 20.0, float(get('gen_rating')), key='gen_rating', on_change=lambda: set_k('gen_rating', st.session_state.gen_rating))
        st.number_input("Gen Maint (%)", 0.0, 10.0, float(get('gen_maint')), key='gen_maint', on_change=lambda: set_k('gen_maint', st.session_state.gen_maint))
        st.number_input("Gen FOR (%)", 0.0, 10.0, float(get('gen_for')), key='gen_for', on_change=lambda: set_k('gen_for', st.session_state.gen_for))

    with st.expander("3. Distribution Reliability (IEEE 493)", expanded=True):
        st.info("Annual Unavailability inputs (%)")
        c1, c2 = st.columns(2)
        c1.number_input("Bus Maint (%)", 0.0, 5.0, float(get('bus_maint')), step=0.01, key='bus_maint', on_change=lambda: set_k('bus_maint', st.session_state.bus_maint))
        c2.number_input("Bus FOR (%)", 0.0, 5.0, float(get('bus_for')), step=0.01, key='bus_for', on_change=lambda: set_k('bus_for', st.session_state.bus_for))
        
        c3, c4 = st.columns(2)
        c3.number_input("Breaker Maint", 0.0, 5.0, float(get('cb_maint')), step=0.01, key='cb_maint', on_change=lambda: set_k('cb_maint', st.session_state.cb_maint))
        c4.number_input("Breaker FOR", 0.0, 5.0, float(get('cb_for')), step=0.01, key='cb_for', on_change=lambda: set_k('cb_for', st.session_state.cb_for))
        
        c5, c6 = st.columns(2)
        c5.number_input("Transformer Maint", 0.0, 5.0, float(get('tx_maint')), step=0.01, key='tx_maint', on_change=lambda: set_k('tx_maint', st.session_state.tx_maint))
        c6.number_input("Transformer FOR", 0.0, 5.0, float(get('tx_for')), step=0.01, key='tx_for', on_change=lambda: set_k('tx_for', st.session_state.tx_for))
        
        st.number_input("Cable FOR (%)", 0.0, 5.0, float(get('cable_for')), step=0.01, key='cable_for', on_change=lambda: set_k('cable_for', st.session_state.cable_for))

res = solve_topology_v4(st.session_state['design_v4'])

# ==============================================================================
# 3. DASHBOARD
# ==============================================================================

st.title("Reliability-Based Topology Designer")

k1, k2, k3 = st.columns(3)
k1.metric("Gen Reliability", f"{res['gen']['calc_avail']*100:.5f}%", f"{res['gen']['total_units']} Units")
k2.metric("Dist. Reliability", f"{res['dist']['calc_avail']*100:.5f}%", f"{res['dist']['total_feeders']} Feeders")
k3.metric("Switchgear", f"{res['elec']['num_swgr']} Buses", f"{res['elec']['rec_rating']} A")

st.markdown("### Topology Diagram")
dot = graphviz.Digraph()
dot.attr(rankdir='TB')

# Distribution Level
dot.node('DC', f'Critical Load\n{get("p_it")} MW\nTarget: {get("req_avail")}%', shape='box3d', style='filled', fillcolor='#AED6F1')

for b in range(1, res['elec']['num_swgr']+1):
    bus_name = f'BUS_{b}'
    dot.node(bus_name, f"Switchgear {b}\n{res['elec']['voltage']} kV", shape='rect', style='filled', fillcolor='#F7DC6F')
    
    # Gen Cluster
    dot.node(f'G_{b}', f"{res['elec']['gens_per_swgr']}x Gens", shape='folder')
    dot.edge(f'G_{b}', bus_name)
    
    # Distribution Cluster
    feeders = res['dist']['feeders_per_bus']
    dist_label = f"{feeders}x Feeders\n(CB + Cable + Trafo)"
    dot.node(f'D_{b}', dist_label, shape='ellipse')
    dot.edge(bus_name, f'D_{b}')
    dot.edge(f'D_{b}', 'DC')

st.graphviz_chart(dot, use_container_width=True)

with st.expander("Detailed Engineering Report"):
    st.write(res)
