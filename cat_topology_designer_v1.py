import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
import numpy as np
import math
import graphviz
import time
from scipy.stats import binom

# --- PAGE CONFIG ---
st.set_page_config(page_title="CAT Topology Designer v5.1", page_icon="⚡", layout="wide")

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
# 1. RELIABILITY MATH ENGINE (UPDATED FOR SPLIT-BUS)
# ==============================================================================

def get_avail_prob(maint_pct, for_pct):
    return 1.0 - ((maint_pct / 100.0) + (for_pct / 100.0))

def calc_k_out_of_n_reliability(n_needed, n_total, p_unit_avail):
    if n_total < n_needed: return 0.0
    prob_success = 0.0
    for k in range(n_needed, n_total + 1):
        prob_success += binom.pmf(k, n_total, p_unit_avail)
    return prob_success

def optimize_redundancy(n_needed, target_rel, p_unit_avail):
    for added_redundancy in range(0, 30):
        n_total = n_needed + added_redundancy
        rel = calc_k_out_of_n_reliability(n_needed, n_total, p_unit_avail)
        if rel >= target_rel:
            return n_total, rel
    return n_needed + 30, rel

# ==============================================================================
# 2. MAIN ALGORITHM
# ==============================================================================

def solve_topology_v5_1(inputs):
    res = {'warnings': [], 'metrics': {}}
    
    # --- A. LOADS ---
    p_it = inputs['p_it']
    p_gross_req = (p_it * (1 + inputs['dc_aux']/100.0)) / ((1 - inputs['dist_loss']/100.0) * (1 - inputs['gen_parasitic']/100.0))
    res['load'] = {'gross_mw': p_gross_req}

    # --- B. VOLTAGE & PHYSICS ---
    voltage_kv = inputs['volts_kv']
    
    def get_total_amps(mw, kv):
        return (mw * 1e6) / (math.sqrt(3) * (kv * 1000) * 0.8)

    if inputs['volts_mode'] == 'Auto-Recommend':
        candidates = [0.48, 4.16, 13.8, 34.5]
        voltage_kv = 34.5
        for v in candidates:
            amps = get_total_amps(p_gross_req, v)
            if amps < 12000: # Increased threshold slightly for split bus
                voltage_kv = v
                break
    
    # --- C. GENERATION SIZING ---
    derate = 1.0
    if inputs['derate_mode'] == 'Auto-Calculate':
        derate = 1.0 - (max(0, (inputs['temp'] - 25)*0.01) + max(0, (inputs['alt'] - 100)*0.0001))
    
    gen_site_mw = inputs['gen_rating'] * derate
    n_gen_needed = math.ceil(p_gross_req / gen_site_mw)
    
    p_gen_avail = get_avail_prob(inputs['gen_maint'], inputs['gen_for'])
    target_rel = inputs['req_avail'] / 100.0
    
    n_gen_total, gen_rel_calc = optimize_redundancy(n_gen_needed, target_rel, p_gen_avail)
    
    res['gen'] = {
        'n_needed': n_gen_needed,
        'n_total': n_gen_total,
        'rel': gen_rel_calc,
        'site_mw': gen_site_mw
    }

    # --- D. TOPOLOGY & SPLIT BUS LOGIC ---
    i_nom_gen = (gen_site_mw * 1e6) / (math.sqrt(3) * (voltage_kv * 1000) * 0.8)
    i_sc_gen = i_nom_gen / inputs['gen_xd']
    
    LIMIT_BUS_AMP = 4000.0
    LIMIT_BUS_KA = 63000.0
    
    # Clustering
    max_gens_amp = math.floor((LIMIT_BUS_AMP * 0.9) / i_nom_gen)
    max_gens_ka = math.floor((LIMIT_BUS_KA * 0.95) / i_sc_gen)
    max_gens_per_section = max(1, min(max_gens_amp, max_gens_ka))
    
    # A "Split Bus" Switchgear has 2 sections. 
    # Total gens per physical lineup = 2 * max_gens_per_section
    gens_per_physical_swgr = max_gens_per_section * 2
    
    num_swgr = math.ceil(n_gen_total / gens_per_physical_swgr)
    # Ensure minimum 2 switchgears for Tier IV / High Avail logic usually, but Split Bus helps.
    
    gens_per_swgr = math.ceil(n_gen_total / num_swgr)
    
    # Logic check: Does the Split Bus hold the amps?
    # Each SECTION (A and B) handles half the current.
    section_amps = (gens_per_swgr / 2) * i_nom_gen
    section_ka = (gens_per_swgr / 2) * i_sc_gen
    
    # Bus Redundancy Check (N-1 Section)
    # Failure scenario: We lose ONE SECTION (half a switchgear) due to internal fault.
    # Gens lost = gens_per_swgr / 2
    gens_remaining = n_gen_total - (gens_per_swgr / 2)
    bus_tolerant = gens_remaining >= n_gen_needed
    
    if not bus_tolerant:
        # Optimization: Add Gens to make it tolerant
        needed_spares = n_gen_needed - gens_remaining
        if needed_spares > 0:
            n_gen_total += math.ceil(needed_spares)
            res['warnings'].append(f"ℹ️ Added generators to ensure N-1 Section Fault Tolerance.")
            bus_tolerant = True

    res['elec'] = {
        'voltage': voltage_kv,
        'num_swgr': num_swgr,
        'gens_per_swgr': gens_per_swgr,
        'section_rating': 4000 if section_amps > 3000 else 3000,
        'section_ka': section_ka / 1000.0,
        'bus_tolerant': bus_tolerant
    }

    # --- E. BESS & DIST (Simplified) ---
    res['bess'] = {'active': False, 'n_total': 0} # Placeholder
    # (BESS logic same as v5.0, omitted to focus on diagram)
    
    # Distribution Feeders
    dist_block_mw = 2.5 
    n_feeders_total = math.ceil(p_gross_req / dist_block_mw) + 2 # N+2 feeders
    res['dist'] = {'n_total': n_feeders_total}

    res['metrics'] = {'total_rel': gen_rel_calc, 'target': target_rel} # Simplified metric

    return res

# ==============================================================================
# 3. UI INPUTS
# ==============================================================================

if 'inputs_v5' not in st.session_state:
    st.session_state['inputs_v5'] = {
        'dc_type': 'Hyperscale', 'p_it': 100.0, 'dc_aux': 15.0, 'req_avail': 99.999,
        'volts_mode': 'Auto-Recommend', 'volts_kv': 13.8, 'derate_mode': 'Auto-Calculate', 'temp': 35, 'alt': 100,
        'gen_rating': 2.5, 'dist_loss': 1.5, 'gen_parasitic': 3.0, 'gen_xd': 0.14, 'gen_maint': 4.0, 'gen_for': 1.0,
    }

def get(k): return st.session_state['inputs_v5'].get(k)
def set_k(k, v): st.session_state['inputs_v5'][k] = v

with st.sidebar:
    st.title("Inputs v5.1")
    st.number_input("IT Load (MW)", 1.0, 500.0, float(get('p_it')), key='p_it', on_change=lambda: set_k('p_it', st.session_state.p_it))
    st.number_input("Target Availability (%)", 90.0, 99.99999, float(get('req_avail')), format="%.5f", key='req_avail', on_change=lambda: set_k('req_avail', st.session_state.req_avail))
    st.caption("Using Split-Bus Internal Topology Logic")

res = solve_topology_v5_1(st.session_state['inputs_v5'])

# ==============================================================================
# 4. DASHBOARD & DIAGRAMS
# ==============================================================================

st.title("CAT Topology Designer v5.1")
st.subheader("Detail: Split-Bus (Main-Tie-Main) Architecture")

if res['elec']['bus_tolerant']:
    st.markdown('<div class="success-box">✅ <b>Fault Tolerant:</b> System can survive the loss of any internal Bus Section (Half-Switchgear) without dropping critical load.</div>', unsafe_allow_html=True)
else:
    st.warning("System relies on all switchgear sections.")

t_overview, t_internal = st.tabs(["🗺️ System Overview", "🔍 Switchgear 'Black Box' Detail"])

with t_overview:
    dot = graphviz.Digraph()
    dot.attr(rankdir='TB')
    
    # Ring Bus
    dot.node('R', f'Distribution Ring\n{res["elec"]["voltage"]} kV', shape='doubleoctagon', fillcolor='#E0E0E0', style='filled')
    
    for b in range(1, res['elec']['num_swgr'] + 1):
        sw_name = f"SWGR_{b}"
        # Representing the Switchgear as a Cluster
        with dot.subgraph(name=f'cluster_sw_{b}') as c:
            c.attr(style='filled', color='#FFF9C4', label=f'Switchgear Lineup {b}')
            
            # Split Bus Nodes
            c.node(f'Bus_{b}_A', 'Bus A', shape='rect', style='filled', fillcolor='#FFCD11')
            c.node(f'Bus_{b}_B', 'Bus B', shape='rect', style='filled', fillcolor='#FFCD11')
            
            # Tie Breaker Edge
            c.edge(f'Bus_{b}_A', f'Bus_{b}_B', label='N.C. Tie', style='bold')
            
            # Gens
            gens_per_side = math.ceil(res['elec']['gens_per_swgr'] / 2)
            c.node(f'G_{b}_A', f'{gens_per_side}x Gens', shape='folder')
            c.node(f'G_{b}_B', f'{gens_per_side}x Gens', shape='folder')
            
            c.edge(f'G_{b}_A', f'Bus_{b}_A')
            c.edge(f'G_{b}_B', f'Bus_{b}_B')
            
        # Connections to Ring
        dot.edge(f'Bus_{b}_A', 'R', label='Feeder A')
        dot.edge(f'Bus_{b}_B', 'R', label='Feeder B')

    st.graphviz_chart(dot, use_container_width=True)

with t_internal:
    st.markdown("### Internal Connection Detail (Typical for all Switchgears)")
    st.write("This diagram details the internal connections of **ONE** Switchgear lineup to prevent total block loss.")
    
    detail_dot = graphviz.Digraph()
    detail_dot.attr(rankdir='LR') # Left to Right for detail
    
    # Internal Components
    with detail_dot.subgraph(name='cluster_enclosure') as c:
        c.attr(label='Switchgear Enclosure (Metal Clad)', style='dashed')
        
        # Section A
        c.node('BA', 'BUS A\n(Rating: 3000A)', shape='rect', style='filled', fillcolor='#FFCD11', width='2')
        c.node('MA', 'Main Breaker A', shape='square', style='filled', fillcolor='#F5B041')
        c.edge('MA', 'BA')
        
        # Section B
        c.node('BB', 'BUS B\n(Rating: 3000A)', shape='rect', style='filled', fillcolor='#FFCD11', width='2')
        c.node('MB', 'Main Breaker B', shape='square', style='filled', fillcolor='#F5B041')
        c.edge('MB', 'BB')
        
        # Tie
        c.node('TIE', 'Tie Breaker\n(Normally Closed)', shape='circle', style='filled', fillcolor='#FFFFFF')
        c.edge('BA', 'TIE', dir='none')
        c.edge('TIE', 'BB', dir='none')
        
        # Gens
        gens_half = math.ceil(res['elec']['gens_per_swgr'] / 2)
        c.node('GA', f'Generators 1-{gens_half}', shape='folder')
        c.node('GB', f'Generators {gens_half+1}-{res["elec"]["gens_per_swgr"]}', shape='folder')
        
        c.edge('GA', 'BA', label=f'{gens_half}x Breakers')
        c.edge('GB', 'BB', label=f'{gens_half}x Breakers')

    st.graphviz_chart(detail_dot, use_container_width=True)
    
    st.info("""
    **Logic:**
    1. **Normal:** Tie Closed. Bus A and B share load.
    2. **Fault on Bus A:** Main Breaker A Trips. Tie Breaker Trips.
    3. **Result:** Bus A is de-energized (Generators 1-12 lost). **Bus B remains energized** (Generators 13-24 online).
    4. **Impact:** System loses 50% of this block, not 100%.
    """)
