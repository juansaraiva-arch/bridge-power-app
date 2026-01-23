import streamlit as st
import pandas as pd
import numpy as np
import math
import graphviz
from scipy.stats import binom

# --- PAGE CONFIG ---
st.set_page_config(page_title="CAT Topology v19.0 (Library Fixed)", page_icon="🚜", layout="wide")

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

# LIBRERÍA ACTUALIZADA DESDE TU EXCEL (Nameplate Capacity ISO)
# Nota: X"d y Step Cap son estimados estándar (no estaban en el CSV), editables en la GUI.
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

# --- FUNCIONES MATEMÁTICAS (DEFINIDAS ANTES DE SU USO) ---

def calc_avail(mtbf, mttr):
    """Calcula disponibilidad A = MTBF / (MTBF + MTTR)"""
    if (mtbf + mttr) <= 0: return 0.0
    return mtbf / (mtbf + mttr)

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
    st.title("Inputs v19.0")
    
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
        
        # Permitir editar X"d y Step Cap ya que no estaban en el CSV
        c1, c2 = st.columns(2)
        gen_xd = c1.number_input("Xd\" (pu)", 0.05, 0.5, gen_defaults['xd'])
        gen_step_cap = c2.number_input("Step Cap (%)", 0.0, 100.0, gen_defaults['step_cap'])
        
        # Actualizar specs con valores editables
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

st.title("🚜 CAT Topology Workflow v19.0")

# --- STEP 1: GENERATION SIZING ---
st.markdown("### Step 1: Generation Fleet Sizing")

# Load calc
dc_aux = 15.0 # %
dist_loss = 1.5 # %
parasitics = 3.0 # %
p_gross = (p_it * (1 + dc_aux/100)) / ((1 - dist_loss/100) * (1 - parasitics/100))

# N Needed for Load
n_needed_load = math.ceil(p_gross / gen_specs['mw'])

# N Needed for Reliability
p_gen_avail = calc_avail(gen_mtbf, gen_mttr)
n_total_gens, gen_sys_rel = get_n_for_reliability(n_needed_load, target_avail, p_gen_avail)

c1, c2, c3 = st.columns(3)
c1.metric("Gross Load", f"{p_gross:.2f} MW")
c2.metric("Gens Needed (Load)", f"{n_needed_load}")
c3.metric("Gens Total (Reliability)", f"{n_total_gens} (N+{n_total_gens - n_needed_load})")

st.markdown(f"<div class='step-box'>Generating System Reliability: <b>{gen_sys_rel*100:.6f}%</b> (Target: {target_avail_pct}%)</div>", unsafe_allow_html=True)


# --- STEP 2: BESS SIZING ---
st.markdown("### Step 2: BESS Sizing (Step Load)")

# Fix: Use full load step requirement as BESS sizing basis (Conservative AI approach)
step_mw = p_it * (step_req_pct / 100.0)
n_bess_units = math.ceil(step_mw / bess_inv_mw)
bess_installed_mw = n_bess_units * bess_inv_mw
bess_energy_mwh = bess_installed_mw * (bess_duration_min / 60.0)

# BESS Reliability (Assume N+1 internal redundancy for block reliability)
n_bess_total_inst = n_bess_units + 1
p_bess_avail = calc_avail(bess_mtbf, bess_mttr)
# AHORA SÍ: La función rel_k_out_n está definida antes de llamarla
bess_sys_rel = rel_k_out_n(n_bess_units, n_bess_total_inst, p_bess_avail)

c1, c2, c3 = st.columns(3)
c1.metric("Step Load Req", f"{step_mw:.1f} MW")
c2.metric("BESS Units (N+1)", f"{n_bess_total_inst} x {bess_inv_mw} MW")
c3.metric("Energy", f"{bess_energy_mwh:.2f} MWh")


# --- STEP 3: TOPOLOGY & VOLTAGE ---
st.markdown("### Step 3: Voltage & Topology Analysis")

# Auto-Voltage Logic
if volts_mode == "Manual":
    calc_kv = manual_kv
else:
    # Estimate Total Amps at generation to decide voltage
    raw_amps_13 = calc_amps(p_gross, 13.8)
    if raw_amps_13 > 8000: 
        calc_kv = 34.5
        st.info("⚡ Auto-Logic: Selected **34.5 kV** due to high load (>8000A at 13.8kV).")
    else:
        calc_kv = 13.8
        st.info("⚡ Auto-Logic: Selected **13.8 kV** (Standard distribution).")

# Bus Splitting based on Amps
total_amps_sys = calc_amps(n_total_gens * gen_specs['mw'], calc_kv)
n_buses = math.ceil(total_amps_sys / bus_amp_limit)
if n_buses < 2: n_buses = 2 

gens_per_bus = math.ceil(n_total_gens / n_buses)
n_total_gens_final = gens_per_bus * n_buses 
amps_per_bus = calc_amps(gens_per_bus * gen_specs['mw'], calc_kv)

c1, c2, c3 = st.columns(3)
c1.metric("Selected Voltage", f"{calc_kv} kV")
c2.metric("Number of Buses", f"{n_buses}")
c3.metric("Amps per Bus", f"{amps_per_bus:.0f} A")

# --- RELIABILITY CALCULATION (Ring vs BaaH) ---
st.markdown("#### Reliability Check")

p_bus = calc_avail(bus_mtbf, bus_mttr)
p_cb = calc_avail(cb_mtbf, cb_mttr)

# 1. Ring Topology (Split Bus)
prob_all_buses = p_bus ** n_buses
prob_gens_ok = rel_k_out_n(n_needed_load, n_total_gens_final, p_gen_avail) 

# Consider N-1 Bus case
prob_1_bus_fail = math.comb(n_buses, 1) * (1 - p_bus) * (p_bus**(n_buses-1))
cap_n_1 = (n_total_gens_final - gens_per_bus) * gen_specs['mw']
n_1_tolerant = cap_n_1 >= p_gross
prob_rem_gens_ok = rel_k_out_n(n_needed_load, n_total_gens_final - gens_per_bus, p_gen_avail)

topo_avail_ring = (prob_all_buses * prob_gens_ok) + (prob_1_bus_fail * prob_rem_gens_ok if n_1_tolerant else 0)
total_avail_ring = topo_avail_ring * bess_sys_rel 

ring_pass = total_avail_ring >= target_avail

# 2. Breaker-and-a-Half (BaaH)
# Reliability limited by Gen+Breaker chain (Bus is redundant)
p_chain = p_gen_avail * p_cb 
topo_avail_baah = rel_k_out_n(n_needed_load, n_total_gens_final, p_chain)
total_avail_baah = topo_avail_baah * bess_sys_rel

baah_pass = total_avail_baah >= target_avail

final_topo = "Ring"

if ring_pass:
    st.markdown(f"<div class='success-box'>✅ **Ring Topology** meets target! (Avail: {total_avail_ring*100:.6f}%)</div>", unsafe_allow_html=True)
else:
    st.markdown(f"<div class='fail-box'>❌ **Ring Topology** FAILED (Avail: {total_avail_ring*100:.6f}%). N-1 Tolerant: {n_1_tolerant}</div>", unsafe_allow_html=True)
    st.write("...Checking Breaker-and-a-Half...")
    if baah_pass:
        st.markdown(f"<div class='success-box'>🚀 **Breaker-and-a-Half** meets target! (Avail: {total_avail_baah*100:.6f}%)</div>", unsafe_allow_html=True)
        final_topo = "BaaH"
    else:
        st.markdown(f"<div class='fail-box'>💀 **BaaH** also failed ({total_avail_baah*100:.6f}%). Check MTBF inputs or add more redundancy.</div>", unsafe_allow_html=True)
        final_topo = "Failed"

# --- STEP 4: SHORT CIRCUIT CHECK ---
st.markdown("### Step 4: Physics Validation (Short Circuit)")

if final_topo != "Failed":
    sc_ka = calc_sc_ka(gen_specs['mw'], gen_specs['xd'], calc_kv, n_total_gens_final)
    
    if sc_ka < sc_limit_ka:
        st.markdown(f"<div class='success-box'>✅ Short Circuit **{sc_ka:.2f} kA** is within limit ({sc_limit_ka} kA).</div>", unsafe_allow_html=True)
    else:
        st.markdown(f"<div class='fail-box'>❌ Short Circuit **{sc_ka:.2f} kA** EXCEEDS limit. Recommendation: Increase Voltage or add Reactors.</div>", unsafe_allow_html=True)
        
    # --- DIAGRAM ---
    st.markdown("### Final Architecture")
    dot = graphviz.Digraph(rankdir='TB')
    
    if final_topo == "BaaH":
        # Draw BaaH with 2 Main Buses
        dot.node('A', 'Main Bus A', shape='rect', width='10', style='filled', fillcolor='black', fontcolor='white')
        dot.node('B', 'Main Bus B', shape='rect', width='10', style='filled', fillcolor='black', fontcolor='white')
        # Draw sample bays
        for i in range(1, 4): 
            with dot.subgraph(name=f'bay_{i}') as bay:
                # 3 Breakers vertical
                cb1 = f'CB_{i}_1'; cb2 = f'CB_{i}_2'; cb3 = f'CB_{i}_3'
                bay.node(cb1, '', shape='square', width='0.5', style='filled', fillcolor='white')
                bay.node(cb2, '', shape='square', width='0.5', style='filled', fillcolor='white')
                bay.node(cb3, '', shape='square', width='0.5', style='filled', fillcolor='white')
                
                # Connections
                dot.edge('A', cb1, dir='none')
                dot.edge(cb1, cb2, dir='none')
                dot.edge(cb2, cb3, dir='none')
                dot.edge(cb3, 'B', dir='none')
                
                # Taps
                bay.node(f'G{i}', 'Gens', shape='circle')
                bay.edge(cb1, f'G{i}', label='Tap 1', dir='none')
                
                bay.node(f'L{i}', 'Load', shape='invtriangle')
                bay.edge(cb2, f'L{i}', label='Tap 2', dir='none')

    else:
        # Ring / Split Bus
        with dot.subgraph(name='cluster_main') as c:
            c.attr(style='invis')
            for i in range(1, n_buses+1):
                c.node(f'B{i}', f'Bus {i}\n{amps_per_bus:.0f}A', shape='rect', style='filled', fillcolor='#FFCD11')
                c.node(f'G{i}', f'{gens_per_bus}x Gens', shape='folder')
                c.edge(f'G{i}', f'B{i}')
        # Ties
        for i in range(1, n_buses+1):
            nxt = i+1 if i < n_buses else 1
            dot.edge(f'B{i}', f'B{nxt}', label='Tie')

    st.graphviz_chart(dot, use_container_width=True)
    ```
