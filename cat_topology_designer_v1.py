import streamlit as st
import pandas as pd
import numpy as np
import math
import graphviz
from scipy.stats import binom

# --- PAGE CONFIG ---
st.set_page_config(page_title="CAT Topology v18.0 (Workflow Mode)", page_icon="🚜", layout="wide")

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

CAT_LIBRARY = {
    "C175-20 (3000 ekW)": {"mw": 3.0, "xd": 0.14, "step_cap": 25.0},
    "C175-16 (2500 ekW)": {"mw": 2.5, "xd": 0.16, "step_cap": 25.0},
    "3516E (2000 ekW)":   {"mw": 2.0, "xd": 0.15, "step_cap": 20.0},
    "3512E (1500 ekW)":   {"mw": 1.5, "xd": 0.17, "step_cap": 20.0},
    "C32 (1000 ekW)":     {"mw": 1.0, "xd": 0.14, "step_cap": 30.0},
}

def calc_avail(mtbf, mttr):
    return mtbf / (mtbf + mttr) if (mtbf + mttr) > 0 else 0.0

def get_n_for_reliability(n_needed, target_avail, p_unit_avail):
    """Calcula N total (k-out-of-n) para cumplir target usando Binomial"""
    for added_redundancy in range(0, 20):
        n_total = n_needed + added_redundancy
        prob = 0.0
        # Probabilidad de que funcionen al menos n_needed
        for k in range(n_needed, n_total + 1):
            prob += binom.pmf(k, n_total, p_unit_avail)
        
        if prob >= target_avail:
            return n_total, prob
    return n_needed + 20, 0.0 # Fail safe

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
    st.title("Inputs v18.0")
    
    with st.expander("1. Project & Load", expanded=True):
        p_it = st.number_input("IT Load (MW)", 10.0, 500.0, 100.0)
        target_avail_pct = st.number_input("Target Availability (%)", 99.0, 99.99999, 99.999, format="%.5f")
        target_avail = target_avail_pct / 100.0
        
        step_req_pct = st.number_input("Step Load Req (%)", 0.0, 100.0, 40.0)
        
        volts_mode = st.selectbox("Voltage Selection", ["Auto-Calculate", "Manual"])
        manual_kv = 13.8
        if volts_mode == "Manual":
            manual_kv = st.number_input("Voltage (kV)", 0.4, 115.0, 13.8)

    with st.expander("2. Generation (CAT Lib)", expanded=True):
        gen_model = st.selectbox("Generator Model", list(CAT_LIBRARY.keys()))
        gen_specs = CAT_LIBRARY[gen_model]
        
        st.write(f"**Specs:** {gen_specs['mw']} MW | X\"d: {gen_specs['xd']}")
        
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

st.title("🚜 CAT Topology Workflow v18.0")

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

step_mw = p_it * (step_req_pct / 100.0)
n_bess_units = math.ceil(step_mw / bess_inv_mw)
bess_installed_mw = n_bess_units * bess_inv_mw
bess_energy_mwh = bess_installed_mw * (bess_duration_min / 60.0)

# BESS Reliability (k-out-of-n, assume needed is total - 1 for high avail, or just parallel block)
# Let's assume we need all to cover full step, so we add +1 redundancy
n_bess_total = n_bess_units + 1
p_bess_avail = calc_avail(bess_mtbf, bess_mttr)
bess_sys_rel = rel_k_out_n(n_bess_units, n_bess_total, p_bess_avail)

c1, c2, c3 = st.columns(3)
c1.metric("Step Load Req", f"{step_mw:.1f} MW")
c2.metric("BESS Units (N+1)", f"{n_bess_total} x {bess_inv_mw} MW")
c3.metric("Energy", f"{bess_energy_mwh:.2f} MWh")


# --- STEP 3: TOPOLOGY & VOLTAGE ---
st.markdown("### Step 3: Voltage & Topology Analysis")

# Auto-Voltage Logic
if volts_mode == "Manual":
    calc_kv = manual_kv
else:
    # Estimate Total Amps at generation to decide voltage
    # This is rough, refinement happens in bus split
    raw_amps_13 = calc_amps(p_gross, 13.8)
    if raw_amps_13 > 8000: # 8000A is hard to handle even with 2 buses
        calc_kv = 34.5
        st.info("⚡ Auto-Logic: Selected **34.5 kV** due to high load (>8000A at 13.8kV).")
    else:
        calc_kv = 13.8
        st.info("⚡ Auto-Logic: Selected **13.8 kV** (Standard distribution).")

# Bus Splitting based on Amps
total_amps_sys = calc_amps(n_total_gens * gen_specs['mw'], calc_kv)
n_buses = math.ceil(total_amps_sys / bus_amp_limit)
# Force even number for symmetry if possible, min 2 for redundancy
if n_buses < 2: n_buses = 2 

gens_per_bus = math.ceil(n_total_gens / n_buses)
# Update total to match symmetric bus loading
n_total_gens_final = gens_per_bus * n_buses 
amps_per_bus = calc_amps(gens_per_bus * gen_specs['mw'], calc_kv)

c1, c2, c3 = st.columns(3)
c1.metric("Selected Voltage", f"{calc_kv} kV")
c2.metric("Number of Buses", f"{n_buses}")
c3.metric("Amps per Bus", f"{amps_per_bus:.0f} A")

# --- RELIABILITY CALCULATION (Ring vs BaaH) ---
st.markdown("#### Reliability Check")

# Base component probs
p_bus = calc_avail(bus_mtbf, bus_mttr)
p_cb = calc_avail(cb_mtbf, cb_mttr)

# 1. Ring Topology (Split Bus)
# System works if: (Buses OK AND Gens OK) OR (1 Bus Down AND Rem Gens OK)
# Simplified Markov State Probability
prob_all_buses = p_bus ** n_buses
prob_gens_ok = rel_k_out_n(n_needed_load, n_total_gens_final, p_gen_avail) # Ignoring breakers for simplicity here

prob_1_bus_fail = math.comb(n_buses, 1) * (1 - p_bus) * (p_bus**(n_buses-1))
prob_rem_gens_ok = rel_k_out_n(n_needed_load, n_total_gens_final - gens_per_bus, p_gen_avail)

# Check N-1 Bus Capacity Tolerance
cap_n_1 = (n_total_gens_final - gens_per_bus) * gen_specs['mw']
n_1_tolerant = cap_n_1 >= p_gross

topo_avail_ring = (prob_all_buses * prob_gens_ok) + (prob_1_bus_fail * prob_rem_gens_ok if n_1_tolerant else 0)
total_avail_ring = topo_avail_ring * bess_sys_rel # Combining with BESS

ring_pass = total_avail_ring >= target_avail

# 2. Breaker-and-a-Half (BaaH)
# Bus failure is negligible (redundant). Limit is Gen+Breaker chain.
p_chain = p_gen_avail * p_cb # Gen available AND its breaker closed
topo_avail_baah = rel_k_out_n(n_needed_load, n_total_gens_final, p_chain)
total_avail_baah = topo_avail_baah * bess_sys_rel

baah_pass = total_avail_baah >= target_avail

# DECISION LOGIC
final_topo = "Ring"
final_avail = total_avail_ring

if ring_pass:
    st.markdown(f"<div class='success-box'>✅ **Ring Topology** meets target! (Avail: {total_avail_ring*100:.6f}%)</div>", unsafe_allow_html=True)
else:
    st.markdown(f"<div class='fail-box'>❌ **Ring Topology** FAILED (Avail: {total_avail_ring*100:.6f}%). N-1 Tolerant: {n_1_tolerant}</div>", unsafe_allow_html=True)
    st.write("...Checking Breaker-and-a-Half...")
    if baah_pass:
        st.markdown(f"<div class='success-box'>🚀 **Breaker-and-a-Half** meets target! (Avail: {total_avail_baah*100:.6f}%)</div>", unsafe_allow_html=True)
        final_topo = "BaaH"
        final_avail = total_avail_baah
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
        dot.node('A', 'Main Bus A', shape='rect', width='10', style='filled', fillcolor='black', fontcolor='white')
        dot.node('B', 'Main Bus B', shape='rect', width='10', style='filled', fillcolor='black', fontcolor='white')
        for i in range(1, 4): # Draw 3 sample bays
            dot.node(f'G{i}', f'Bay {i}\nGens', shape='circle')
            dot.edge('A', f'G{i}', label='CB')
            dot.edge(f'G{i}', 'B', label='CB')
    else:
        # Ring
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
