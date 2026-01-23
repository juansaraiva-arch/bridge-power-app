import streamlit as st
import pandas as pd
import numpy as np
import math
import graphviz
from scipy.stats import binom

# --- PAGE CONFIG ---
st.set_page_config(page_title="CAT Topology v19.1 (Stable)", page_icon="🚜", layout="wide")

# --- CSS ---
st.markdown("""
<style>
    .step-box { background-color: #f0f8ff; padding: 15px; border-radius: 5px; border-left: 5px solid #007bff; margin-bottom: 15px; }
    .success-box { background-color: #d4edda; padding: 15px; border-radius: 5px; border-left: 5px solid #28a745; margin-bottom: 15px; }
    .fail-box { background-color: #f8d7da; padding: 15px; border-radius: 5px; border-left: 5px solid #dc3545; margin-bottom: 15px; }
    .metric-value { font-size: 24px; font-weight: bold; }
    .metric-label { font-size: 12px; color: #555; text-transform: uppercase; }
</style>
""", unsafe_allow_html=True)

# ==============================================================================
# 0. DATA LIBRARIES & UTILS
# ==============================================================================

# LIBRERÍA ACTUALIZADA (Excel Source)
CAT_LIBRARY = {
    "XGC1900 (1.9 MW)":   {"mw": 1.9,   "xd": 0.16, "step_cap": 25.0},
    "G3520FR (2.5 MW)":   {"mw": 2.5,   "xd": 0.16, "step_cap": 25.0},
    "G3520K (2.4 MW)":    {"mw": 2.4,   "xd": 0.16, "step_cap": 25.0},
    "CG260 (3.96 MW)":    {"mw": 3.957, "xd": 0.15, "step_cap": 25.0},
    "G20CM34 (9.76 MW)":  {"mw": 9.76,  "xd": 0.14, "step_cap": 20.0},
    "Titan 130 (16.5 MW)":{"mw": 16.5,  "xd": 0.14, "step_cap": 15.0},
    "Titan 250 (23.2 MW)":{"mw": 23.2,  "xd": 0.14, "step_cap": 15.0},
    "Titan 350 (38.0 MW)":{"mw": 38.0,  "xd": 0.14, "step_cap": 15.0}
}

def calc_avail(mtbf, mttr):
    """Calcula disponibilidad A = MTBF / (MTBF + MTTR)"""
    if (mtbf + mttr) <= 0: return 0.0
    return mtbf / (mtbf + mttr)

def get_unavailability(mtbf, mttr):
    """Calcula indisponibilidad U = MTTR / (MTBF + MTTR)"""
    if (mtbf + mttr) <= 0: return 1.0
    return mttr / (mtbf + mttr)

def rel_k_out_n(n_needed, n_total, p_unit):
    """Calcula confiabilidad de sistema redundante paralelo k-de-n"""
    if n_total < n_needed: return 0.0
    prob = 0.0
    # Suma acumulativa de la PDF binomial
    for k in range(n_needed, n_total + 1):
        prob += binom.pmf(k, n_total, p_unit)
    return prob

def get_n_for_reliability(n_needed, target_avail, p_unit_avail):
    """Itera para encontrar el N total necesario para cumplir el Target"""
    for added_redundancy in range(0, 20):
        n_total = n_needed + added_redundancy
        prob = rel_k_out_n(n_needed, n_total, p_unit_avail)
        if prob >= target_avail:
            return n_total, prob
    return n_needed + 20, 0.0 

def calc_amps(mw, kv):
    return (mw * 1e6) / (math.sqrt(3) * kv * 1000)

def calc_sc_ka(mw_gen, xd, kv, n_gens):
    mva_gen = mw_gen / 0.8
    i_base = (mva_gen * 1e6) / (math.sqrt(3) * kv * 1000)
    i_sc_unit = i_base / xd
    return (i_sc_unit * n_gens) / 1000.0

# ==============================================================================
# 1. SIDEBAR INPUTS
# ==============================================================================

with st.sidebar:
    st.title("Inputs v19.1")
    
    with st.expander("1. Project & Load", expanded=True):
        p_it = st.number_input("IT Load (MW)", 10.0, 500.0, 100.0)
        target_avail_pct = st.number_input("Target Availability (%)", 99.0, 99.99999, 99.999, format="%.5f")
        target_avail = target_avail_pct / 100.0
        
        step_req_pct = st.number_input("Step Load Req (%)", 0.0, 100.0, 40.0)
        
        volts_mode = st.selectbox("Voltage Selection", ["Auto-Calculate", "Manual"])
        manual_kv = 13.8
        if volts_mode == "Manual":
            manual_kv = st.number_input("Voltage (kV)", 0.4, 115.0, 13.8)

    with st.expander("2. Generation (From Excel)", expanded=True):
        gen_model = st.selectbox("Generator Model", list(CAT_LIBRARY.keys()))
        gen_defaults = CAT_LIBRARY[gen_model]
        
        st.info(f"**Selected:** {gen_model}")
        
        c1, c2 = st.columns(2)
        gen_xd = c1.number_input("Xd\" (pu)", 0.05, 0.5, gen_defaults['xd'])
        gen_step_cap = c2.number_input("Step Cap (%)", 0.0, 100.0, gen_defaults['step_cap'])
        
        gen_specs = {"mw": gen_defaults['mw'], "xd": gen_xd, "step_cap": gen_step_cap}
        
        st.caption("Reliability (IEEE 493)")
        gen_mtbf = st.number_input("Gen MTBF (h)", 100, 50000, 2500)
        gen_mttr = st.number_input("Gen MTTR (h)", 1, 1000, 24)

    with st.expander("3. BESS Parameters"):
        bess_inv_mw = st.number_input("BESS Inverter Unit (MW)", 0.1, 10.0, 3.8)
        bess_duration_min = st.number_input("Duration (min)", 1, 60, 5)
        st.caption("BESS Reliability")
        bess_mtbf = st.number_input("BESS MTBF (h)", 100, 50000, 8000)
        bess_mttr = st.number_input("BESS MTTR (h)", 1, 1000, 24)

    with st.expander("4. Switchgear & Topology"):
        st.caption("Component Reliability")
        bus_mtbf = st.number_input("Bus MTBF (h)", 100000, 5000000, 1000000)
        bus_mttr = st.number_input("Bus MTTR (h)", 1, 1000, 12)
        cb_mtbf = st.number_input("Breaker MTBF (h)", 50000, 1000000, 300000)
        cb_mttr = st.number_input("Breaker MTTR (h)", 1, 100, 4)
        
        bus_amp_limit = st.number_input("Bus Amp Limit (A)", 1000, 6000, 4000)
        sc_limit_ka = st.number_input("Short Circuit Limit (kA)", 25, 100, 63)

# ==============================================================================
# 2. MAIN LOGIC FLOW
# ==============================================================================

st.title("🚜 CAT Topology Workflow v19.1")

# --- STEP 1: GENERATION SIZING ---
st.markdown("### Step 1: Generation Fleet Sizing")

dc_aux = 15.0 
dist_loss = 1.5 
parasitics = 3.0 
p_gross = (p_it * (1 + dc_aux/100)) / ((1 - dist_loss/100) * (1 - parasitics/100))

n_needed_load = math.ceil(p_gross / gen_specs['mw'])
p_gen_avail = calc_avail(gen_mtbf, gen_mttr)
n_total_gens, gen_sys_rel = get_n_for_reliability(n_needed_load, target_avail, p_gen_avail)

c1, c2, c3 = st.columns(3)
c1.metric("Gross Load", f"{p_gross:.2f} MW")
c2.metric("Gens Needed (Load)", f"{n_needed_load}")
c3.metric("Gens Total (Reliability)", f"{n_total_gens} (N+{n_total_gens - n_needed_load})")

st.markdown(f"<div class='step-box'>Generating System Reliability: <b>{gen_sys_rel*100:.6f}%</b> (Target: {target_avail_pct}%)</div>", unsafe_allow_html=True)


# --- STEP 2: BESS SIZING (BRIDGE POWER) ---
st.markdown("### Step 2: BESS Sizing (Bridge Power)")

# Logic: BESS Sizing = 100% of Gross Load
bess_target_mw = p_gross
n_bess_units = math.ceil(bess_target_mw / bess_inv_mw)
bess_installed_mw = n_bess_units * bess_inv_mw
bess_energy_mwh = bess_installed_mw * (bess_duration_min / 60.0)

# BESS Reliability (Assume N+1 block redundancy)
n_bess_total_inst = n_bess_units + 1
p_bess_avail = calc_avail(bess_mtbf, bess_mttr)
bess_sys_rel = rel_k_out_n(n_bess_units, n_bess_total_inst, p_bess_avail)

c1, c2, c3 = st.columns(3)
c1.metric("Bridge Power Target", f"{bess_target_mw:.1f} MW (100% Load)")
c2.metric("BESS Units (N+1)", f"{n_bess_total_inst} x {bess_inv_mw} MW")
c3.metric("Energy Storage", f"{bess_energy_mwh:.2f} MWh ({bess_duration_min} min)")


# --- STEP 3: TOPOLOGY & VOLTAGE ---
st.markdown("### Step 3: Voltage & Topology Analysis")

# Auto-Voltage
if volts_mode == "Manual":
    calc_kv = manual_kv
else:
    raw_amps_13 = calc_amps(p_gross, 13.8)
    if raw_amps_13 > 8000:
        calc_kv = 34.5
        st.info("⚡ Auto-Logic: Selected **34.5 kV** (High Load).")
    else:
        calc_kv = 13.8
        st.info("⚡ Auto-Logic: Selected **13.8 kV**.")

# Reliability Analysis (Cut-Set)
u_gen_unit = get_unavailability(gen_mtbf, gen_mttr)
u_bus = get_unavailability(bus_mtbf, bus_mttr)
u_cb = get_unavailability(cb_mtbf, cb_mttr)

# --- Option A: Ring ---
u_access_ring = u_cb + u_bus
p_gen_effective_ring = 1.0 - (u_gen_unit + u_access_ring)
n_ring, rel_ring = get_n_for_reliability(n_needed_load, target_avail, p_gen_effective_ring)
total_rel_ring = rel_ring * bess_sys_rel

# --- Option B: BaaH ---
u_access_baah = (u_cb * u_cb) + (u_bus * u_cb) + (u_cb * u_bus) + (u_bus * u_bus)
p_gen_effective_baah = 1.0 - (u_gen_unit + u_access_baah)
n_baah, rel_baah = get_n_for_reliability(n_needed_load, target_avail, p_gen_effective_baah)
total_rel_baah = rel_baah * bess_sys_rel

# Selection
final_topo = "Ring"
final_n_gens = n_ring

if total_rel_ring >= target_avail:
    st.markdown(f"<div class='success-box'>✅ **Ring Topology** meets target! (Avail: {total_rel_ring*100:.6f}%)</div>", unsafe_allow_html=True)
else:
    if total_rel_baah >= target_avail:
        st.markdown(f"<div class='success-box'>🚀 **Breaker-and-a-Half** selected to meet target. (Ring Failed at {total_rel_ring*100:.6f}%)</div>", unsafe_allow_html=True)
        final_topo = "BaaH"
        final_n_gens = n_baah
    else:
        st.markdown(f"<div class='fail-box'>💀 Target not met even with BaaH ({total_rel_baah*100:.6f}%). Using BaaH as best option.</div>", unsafe_allow_html=True)
        final_topo = "BaaH"
        final_n_gens = n_baah

# --- STEP 4: PHYSICS & DIAGRAM ---
st.markdown("### Step 4: Validation & Layout")

sc_ka = calc_sc_ka(gen_specs['mw'], gen_specs['xd'], calc_kv, final_n_gens)
if sc_ka < sc_limit_ka:
    st.markdown(f"✅ Short Circuit **{sc_ka:.2f} kA** is OK (< {sc_limit_ka} kA)")
else:
    st.markdown(f"❌ Short Circuit **{sc_ka:.2f} kA** EXCEEDS limit. Try increasing voltage.")

# Diagram
# CORRECCIÓN DE ERROR: Inicialización correcta de Graphviz
dot = graphviz.Digraph()
dot.attr(rankdir='TB')

if final_topo == "BaaH":
    dot.node('A', 'Main Bus A', shape='rect', width='10', style='filled', fillcolor='black', fontcolor='white')
    dot.node('B', 'Main Bus B', shape='rect', width='10', style='filled', fillcolor='black', fontcolor='white')
    
    # Draw Bays (Gen + BESS included in bays)
    n_bays_draw = min(4, math.ceil((final_n_gens + n_bess_total_inst)/2))
    
    for i in range(1, n_bays_draw + 1):
        with dot.subgraph(name=f'bay_{i}') as bay:
            bay.edge('A', f'CB{i}1', dir='none')
            bay.edge(f'CB{i}1', f'CB{i}2', dir='none')
            bay.edge(f'CB{i}2', f'CB{i}3', dir='none')
            bay.edge(f'CB{i}3', 'B', dir='none')
            
            bay.node(f'CB{i}1', shape='square', label='', width='0.5', fixedsize='true')
            bay.node(f'CB{i}2', shape='square', label='', width='0.5', fixedsize='true')
            bay.node(f'CB{i}3', shape='square', label='', width='0.5', fixedsize='true')
            
            bay.node(f'Source{i}', 'Gen/BESS', shape='circle')
            bay.edge(f'CB{i}1', f'Source{i}', label='Tap', dir='none')

else: # Ring
    amps = calc_amps(final_n_gens * gen_specs['mw'], calc_kv)
    n_bus = math.ceil(amps/bus_amp_limit)
    if n_bus < 2: n_bus = 2
    
    # Bus Nodes
    for i in range(1, n_bus+1):
        dot.node(f'B{i}', f'Bus {i}', shape='rect', style='filled', fillcolor='#FFCD11')
        dot.node(f'G{i}', 'Gens', shape='folder')
        dot.edge(f'G{i}', f'B{i}')
    
    # Ties
    for i in range(1, n_bus):
        dot.edge(f'B{i}', f'B{i+1}', label='Tie')
    dot.edge(f'B{n_bus}', 'B1', label='Tie')

st.graphviz_chart(dot, use_container_width=True)
