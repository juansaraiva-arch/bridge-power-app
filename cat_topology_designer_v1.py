import streamlit as st
import pandas as pd
import numpy as np
import math
import graphviz
from scipy.stats import binom

# --- PAGE CONFIG ---
st.set_page_config(page_title="CAT Topology v20.2 (Stable)", page_icon="🧮", layout="wide")

# --- CSS ---
st.markdown("""
<style>
    .math-box { background-color: #e8f4f8; padding: 15px; border-radius: 5px; font-family: monospace; border-left: 5px solid #17a2b8; margin-bottom: 10px; }
    .success-box { background-color: #d4edda; padding: 15px; border-radius: 5px; border-left: 5px solid #28a745; margin-bottom: 15px; }
    .fail-box { background-color: #f8d7da; padding: 15px; border-radius: 5px; border-left: 5px solid #dc3545; margin-bottom: 15px; }
    .metric-value { font-size: 24px; font-weight: bold; }
    .metric-label { font-size: 12px; color: #555; text-transform: uppercase; }
</style>
""", unsafe_allow_html=True)

# ==============================================================================
# 0. DATA & UTILS
# ==============================================================================

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
    if (mtbf + mttr) <= 0: return 0.0
    return mtbf / (mtbf + mttr)

def get_unavailability(mtbf, mttr):
    if (mtbf + mttr) <= 0: return 1.0
    return mttr / (mtbf + mttr)

def rel_k_out_n(n_needed, n_total, p_unit):
    if n_total < n_needed: return 0.0
    prob = 0.0
    for k in range(n_needed, n_total + 1):
        prob += binom.pmf(k, n_total, p_unit)
    return prob

def get_n_for_reliability(n_needed, target_avail, p_unit_avail):
    """
    Optimizes N (Redundancy) to meet a specific target reliability.
    Returns: (n_total, resulting_reliability)
    """
    for added_redundancy in range(0, 50): # Search up to N+50 if needed
        n_total = n_needed + added_redundancy
        prob = rel_k_out_n(n_needed, n_total, p_unit_avail)
        if prob >= target_avail:
            return n_total, prob
    return n_needed + 50, 0.0 

def calc_amps(mw, kv):
    return (mw * 1e6) / (math.sqrt(3) * kv * 1000)

def calc_sc_ka(mw_gen, xd, kv, n_gens):
    mva_gen = mw_gen / 0.8
    i_base = (mva_gen * 1e6) / (math.sqrt(3) * kv * 1000)
    i_sc_unit = i_base / xd
    return (i_sc_unit * n_gens) / 1000.0

# ==============================================================================
# 1. SIDEBAR
# ==============================================================================

with st.sidebar:
    st.title("Inputs v20.2")
    
    with st.expander("1. Project & Load", expanded=True):
        p_it = st.number_input("IT Load (MW)", 10.0, 500.0, 100.0)
        target_avail_pct = st.number_input("Target Availability (%)", 99.0, 99.99999, 99.999, format="%.5f")
        target_avail = target_avail_pct / 100.0
        
        volts_mode = st.selectbox("Voltage Selection", ["Auto-Calculate", "Manual"])
        manual_kv = 13.8
        if volts_mode == "Manual":
            manual_kv = st.number_input("Voltage (kV)", 0.4, 115.0, 13.8)

    with st.expander("2. Generation", expanded=True):
        gen_model = st.selectbox("Generator Model", list(CAT_LIBRARY.keys()))
        gen_defaults = CAT_LIBRARY[gen_model]
        
        c1, c2 = st.columns(2)
        gen_xd = c1.number_input("Xd\" (pu)", 0.05, 0.5, gen_defaults['xd'])
        gen_step_cap = c2.number_input("Step Cap (%)", 0.0, 100.0, gen_defaults['step_cap'])
        gen_specs = {"mw": gen_defaults['mw'], "xd": gen_xd, "step_cap": gen_step_cap}
        
        st.caption("Gen Reliability")
        gen_mtbf = st.number_input("Gen MTBF (h)", 100, 50000, 2500)
        gen_mttr = st.number_input("Gen MTTR (h)", 1, 1000, 24)

    with st.expander("3. BESS (Bridge Power)"):
        bess_inv_mw = st.number_input("BESS Inverter Unit (MW)", 0.1, 10.0, 3.8)
        bess_duration_min = st.number_input("Duration (min)", 1, 60, 5)
        st.caption("BESS Reliability")
        bess_mtbf = st.number_input("BESS MTBF (h)", 100, 50000, 8000)
        bess_mttr = st.number_input("BESS MTTR (h)", 1, 1000, 24)

    with st.expander("4. Substation Reliability"):
        bus_mtbf = st.number_input("Bus MTBF (h)", 100000, 5000000, 1000000)
        bus_mttr = st.number_input("Bus MTTR (h)", 1, 1000, 12)
        cb_mtbf = st.number_input("Breaker MTBF (h)", 50000, 1000000, 200000)
        cb_mttr = st.number_input("Breaker MTTR (h)", 1, 100, 8)
        
        bus_amp_limit = st.number_input("Bus Amp Limit (A)", 1000, 6000, 4000)
        sc_limit_ka = st.number_input("Short Circuit Limit (kA)", 25, 100, 63)

# ==============================================================================
# 2. CALCULATION ENGINE
# ==============================================================================

st.title("🚜 CAT Topology Workflow v20.2")
st.caption("Stable Release: Corrected Graphs + Math Breakdown")

# --- STEP 1: LOAD & VOLTAGE ---
dc_aux = 15.0; dist_loss = 1.5; parasitics = 3.0 
p_gross = (p_it * (1 + dc_aux/100)) / ((1 - dist_loss/100) * (1 - parasitics/100))

if volts_mode == "Manual":
    calc_kv = manual_kv
else:
    raw_amps_13 = calc_amps(p_gross, 13.8)
    calc_kv = 34.5 if raw_amps_13 > 8000 else 13.8

# --- STEP 2: BESS SIZING & RELIABILITY ---
target_bess_subsys = 0.999999 

bess_target_mw = p_gross
n_bess_needed = math.ceil(bess_target_mw / bess_inv_mw)
p_bess_unit_avail = calc_avail(bess_mtbf, bess_mttr)

# OPTIMIZE BESS N: Iteratively add redundancy until target met
n_bess_total, bess_rel_actual = get_n_for_reliability(n_bess_needed, target_bess_subsys, p_bess_unit_avail)

bess_installed_mw = n_bess_total * bess_inv_mw
bess_energy_mwh = bess_installed_mw * (bess_duration_min / 60.0)

# --- STEP 3: GENERATION & TOPOLOGY (BAAH PRIORITY) ---

u_gen_unit = get_unavailability(gen_mtbf, gen_mttr)
u_bus = get_unavailability(bus_mtbf, bus_mttr)
u_cb = get_unavailability(cb_mtbf, cb_mttr)

# BaaH Logic: Loss of access requires concurrent failures
u_access_baah = (u_cb * u_cb) + (u_bus * u_cb) + (u_cb * u_bus) + (u_bus * u_bus)
p_gen_effective_baah = 1.0 - (u_gen_unit + u_access_baah)

n_gen_needed = math.ceil(p_gross / gen_specs['mw'])
# Since Total = Gen_Sys * BESS_Sys, we need Gen_Sys >= Target / BESS_Sys
target_gen_sys = target_avail / bess_rel_actual if bess_rel_actual > 0 else target_avail

n_gen_total, gen_sys_rel = get_n_for_reliability(n_gen_needed, target_gen_sys, p_gen_effective_baah)
total_system_avail = gen_sys_rel * bess_rel_actual

# --- RESULTS DISPLAY ---

# 1. SUMMARY
if total_system_avail >= target_avail:
    st.markdown(f"<div class='success-box'>✅ **Target Met!** System Availability: **{total_system_avail*100:.7f}%**</div>", unsafe_allow_html=True)
else:
    st.markdown(f"<div class='fail-box'>❌ **Target Missed.** Best Possible: {total_system_avail*100:.7f}%</div>", unsafe_allow_html=True)

c1, c2, c3, c4 = st.columns(4)
c1.metric("Voltage", f"{calc_kv} kV")
c2.metric("Gen Fleet", f"{n_gen_total} Units", f"N+{n_gen_total - n_gen_needed}")
c3.metric("BESS Fleet", f"{n_bess_total} Units", f"N+{n_bess_total - n_bess_needed}")
c4.metric("Short Circuit", f"{calc_sc_ka(gen_specs['mw'], gen_specs['xd'], calc_kv, n_gen_total):.1f} kA")

# 2. MATH BREAKDOWN (THE "WHY")
with st.expander("🧮 CALCULATION BREAKDOWN (Why this result?)", expanded=True):
    st.markdown("### 1. BESS Subsystem Optimization")
    st.markdown(f"""
    * **Unit Reliability:** $A_{{unit}} = \\frac{{{bess_mtbf}}}{{{bess_mtbf} + {bess_mttr}}} = {p_bess_unit_avail:.6f}$
    * **Need:** {n_bess_needed} units for {bess_target_mw:.1f} MW.
    * **Iteration:**
        * N+{n_bess_total - n_bess_needed - 1} ({n_bess_total-1} units): Failed.
        * **N+{n_bess_total - n_bess_needed} ({n_bess_total} units):** {bess_rel_actual:.8f} (Success)
    """)
    
    st.markdown("### 2. Generator Subsystem (Breaker-and-a-Half)")
    st.markdown(f"""
    * **Gen Unit Unavail ($U_{{gen}}$):** {u_gen_unit:.6f}
    * **Grid Access Unavail ($U_{{access}}$):** {u_access_baah:.10f} (Negligible due to BaaH redundancy)
    * **Effective Gen Reliability:** $P_{{eff}} = 1 - (U_{{gen}} + U_{{access}}) = {p_gen_effective_baah:.6f}$
    * **Fleet Optimization (Binomial):**
        * Need {n_gen_needed} units.
        * Installed {n_gen_total} units (N+{n_gen_total - n_gen_needed}).
        * Resulting Reliability: **{gen_sys_rel:.8f}**
    """)
    
    st.markdown("### 3. Total System")
    st.markdown(f"""
    $$ A_{{Total}} = A_{{GenSys}} \\times A_{{BESSSys}} $$
    $$ {total_system_avail:.9f} = {gen_sys_rel:.9f} \\times {bess_rel_actual:.9f} $$
    """)

# 3. DIAGRAM
st.markdown("### Architecture Layout")

# --- CORRECCIÓN DE BUG AQUÍ ---
dot = graphviz.Digraph() 
dot.attr(rankdir='TB') # Ahora se asigna como atributo, no en constructor
# ------------------------------

dot.node('A', 'Bus A', shape='rect', width='10', style='filled', fillcolor='black', fontcolor='white')
dot.node('B', 'Bus B', shape='rect', width='10', style='filled', fillcolor='black', fontcolor='white')

n_draw = min(5, math.ceil((n_gen_total + n_bess_total)/2))
for i in range(1, n_draw + 1):
    with dot.subgraph(name=f'bay_{i}') as bay:
        bay.node(f'CB{i}1', shape='square', label='', style='filled', fillcolor='white')
        bay.node(f'CB{i}2', shape='square', label='', style='filled', fillcolor='white')
        bay.node(f'CB{i}3', shape='square', label='', style='filled', fillcolor='white')
        bay.edge('A', f'CB{i}1', dir='none')
        bay.edge(f'CB{i}1', f'CB{i}2', dir='none')
        bay.edge(f'CB{i}2', f'CB{i}3', dir='none')
        bay.edge(f'CB{i}3', 'B', dir='none')
        
        bay.node(f'S{i}', 'Source', shape='circle')
        bay.edge(f'CB{i}1', f'S{i}', label='Tap', dir='none')

st.graphviz_chart(dot, use_container_width=True)
