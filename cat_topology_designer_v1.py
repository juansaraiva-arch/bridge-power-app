import streamlit as st
import pandas as pd
import numpy as np
import math
import graphviz
from scipy.stats import binom

# --- PAGE CONFIG ---
st.set_page_config(page_title="CAT Topology Designer v9.0 (Auto-Tier)", page_icon="⚡", layout="wide")

# --- CSS ---
st.markdown("""
<style>
    @media print {
        [data-testid="stSidebar"], [data-testid="stHeader"], footer, .stButton { display: none !important; }
        .block-container { padding: 0 !important; margin: 0 !important; }
    }
    .success-box { background-color: #d4edda; border: 1px solid #c3e6cb; padding: 15px; border-radius: 5px; color: #155724; margin-bottom: 10px; }
    .info-box { background-color: #cce5ff; border: 1px solid #b8daff; padding: 15px; border-radius: 5px; color: #004085; margin-bottom: 10px; }
    .error-box { background-color: #f8d7da; border: 1px solid #f5c6cb; padding: 15px; border-radius: 5px; color: #721c24; margin-bottom: 10px; }
    .kpi-card { background-color: #f8f9fa; padding: 10px; border-radius: 5px; text-align: center; border-left: 5px solid #005cbf; box-shadow: 1px 1px 3px rgba(0,0,0,0.1); }
    .metric-value { font-size: 22px; font-weight: bold; }
    .metric-label { font-size: 13px; color: #666; }
</style>
""", unsafe_allow_html=True)

# ==============================================================================
# 1. RELIABILITY MATH & PHYSICS
# ==============================================================================

def calc_availability_mtbf(mtbf, mttr):
    if mtbf + mttr == 0: return 0.0
    return mtbf / (mtbf + mttr)

def reliability_k_out_of_n(n_needed, n_total, p_unit):
    if n_total < n_needed: return 0.0
    prob = 0.0
    for k in range(n_needed, n_total + 1):
        prob += binom.pmf(k, n_total, p_unit)
    return prob

def reliability_series(components):
    rel = 1.0
    for r in components: rel *= r
    return rel

def calc_short_circuit(voltage_kv, gen_mva, xd_pu, num_gens_parallel):
    if voltage_kv == 0 or xd_pu == 0: return 999999.0, 0.0
    # I_base = MVA / (sqrt(3)*kV)
    i_base = (gen_mva * 1e6) / (math.sqrt(3) * (voltage_kv * 1000))
    # I_sc_unit = I_base / X"d
    i_sc_unit = i_base / xd_pu
    # Worst case Ring Bus: All gens contribute to fault
    i_sc_total = i_sc_unit * num_gens_parallel
    return i_sc_total, i_base

# ==============================================================================
# 2. CORE OPTIMIZER (Single Voltage)
# ==============================================================================

def optimize_for_voltage(inputs, test_kv, p_gross_req, a_components):
    """
    Intenta resolver la topología para un voltaje específico.
    Retorna: (Success_Bool, Solution_Dict, Reason_If_Fail)
    """
    gen_rating_site = inputs['gen_rating'] * (1.0 - (max(0, (inputs['temp']-25)*0.01)))
    gen_mva = gen_rating_site / 0.8
    
    a_gen, a_bus, a_dist_path = a_components
    
    min_gens = math.ceil(p_gross_req / gen_rating_site)
    best_sol = None
    fail_reasons = []

    # Search Loop
    for n_total in range(min_gens, min_gens + 40): # Fleet sizing
        for n_buses in range(2, 12): # Bus sizing
            gens_per_bus = math.ceil(n_total / n_buses)
            real_total = n_buses * gens_per_bus
            
            # 1. Physics Check
            i_sc_total, i_nom_unit = calc_short_circuit(test_kv, gen_mva, inputs['gen_xd'], real_total)
            bus_amp_load = gens_per_bus * i_nom_unit
            
            phy_pass = True
            if i_sc_total > 63000: 
                phy_pass = False
                fail_reasons.append(f"KA_FAIL:{test_kv}kV")
            if bus_amp_load > 4000: 
                phy_pass = False
                fail_reasons.append(f"AMP_FAIL:{test_kv}kV")
            
            if not phy_pass: continue

            # 2. Fault Tolerance (N-1 Bus)
            surviving_gens = real_total - gens_per_bus - 1
            if (surviving_gens * gen_rating_site) < p_gross_req:
                continue # Not N-1 tolerant yet

            # 3. Availability Check (RBD)
            n_needed = math.ceil(p_gross_req / gen_rating_site)
            
            # P(Sys) = P(All Buses) * P(Gens) + P(1 Bus Down) * P(Rem Gens)
            p_buses_ok = a_bus ** n_buses
            rel_s0 = reliability_k_out_of_n(n_needed, real_total, a_gen)
            
            p_1bus_down = math.comb(n_buses, 1) * (1-a_bus) * (a_bus**(n_buses-1))
            rel_s1 = reliability_k_out_of_n(n_needed, real_total - gens_per_bus, a_gen)
            
            sys_avail = (p_buses_ok * rel_s0) + (p_1bus_down * rel_s1)
            
            # Dist leg
            m_feeders = math.ceil(p_gross_req / 2.5) + 2
            dist_avail = reliability_k_out_of_n(m_feeders-2, m_feeders, a_dist_path)
            
            total_avail = sys_avail * dist_avail
            
            if total_avail >= (inputs['req_avail']/100.0):
                # SUCCESS! Found a valid config
                return True, {
                    'n_buses': n_buses, 'n_total': real_total, 'gens_per_bus': gens_per_bus,
                    'voltage': test_kv, 'bus_ka': i_sc_total/1000.0, 'bus_amps': bus_amp_load,
                    'avail': total_avail, 'n_feeders': m_feeders
                }, None
                
    return False, None, list(set(fail_reasons))

# ==============================================================================
# 3. MASTER SOLVER (Auto-Tier Logic)
# ==============================================================================

def solve_topology_v9(inputs):
    res = {'pass': False, 'log': []}
    
    # Load Prep
    p_it = inputs['p_it']
    p_gross_req = (p_it * (1 + inputs['dc_aux']/100.0)) / ((1 - inputs['dist_loss']/100.0) * (1 - inputs['gen_parasitic']/100.0))
    res['load'] = {'gross': p_gross_req}
    
    # Avail Prep
    a_gen = calc_availability_mtbf(inputs['gen_mtbf'], inputs['gen_mttr'])
    a_bus = calc_availability_mtbf(inputs['bus_mtbf'], inputs['bus_mttr'])
    a_dist = reliability_series([
        calc_availability_mtbf(inputs['cb_mtbf'], inputs['cb_mttr']),
        calc_availability_mtbf(inputs['tx_mtbf'], inputs['tx_mttr']),
        calc_availability_mtbf(inputs['cable_mtbf'], inputs['cable_mttr'])
    ])
    
    # Voltage Selection Strategy
    if inputs['volts_mode'] == 'Manual':
        candidates = [inputs['volts_kv']]
    else:
        # Standard Utility Tiers
        candidates = [13.8, 34.5, 69.0] 
        # Add 4.16 only if load is small
        if p_gross_req < 10.0: candidates.insert(0, 4.16)
        if p_gross_req < 2.0: candidates.insert(0, 0.48)
        
    final_sol = None
    
    for kv in candidates:
        success, sol, reasons = optimize_for_voltage(inputs, kv, p_gross_req, (a_gen, a_bus, a_dist))
        
        if success:
            final_sol = sol
            res['log'].append(f"✅ Voltage {kv} kV: **SUCCESS**")
            break # Stop at the lowest voltage that works
        else:
            limit_msg = ", ".join(reasons[:2]) if reasons else "Availability/Tolerance"
            res['log'].append(f"⚠️ Voltage {kv} kV: Failed ({limit_msg}) -> Trying next...")
            
    if final_sol:
        res['pass'] = True
        res['sol'] = final_sol
    else:
        res['pass'] = False
        res['error'] = "Could not find a valid configuration even after checking all voltage tiers. Consider increasing Gen Rating or reducing Availability Target."

    return res

# ==============================================================================
# 4. UI INPUTS
# ==============================================================================

if 'inputs_v9' not in st.session_state:
    st.session_state['inputs_v9'] = {
        'p_it': 100.0, 'dc_aux': 15.0, 'req_avail': 99.999, 'volts_mode': 'Auto-Recommend', 'volts_kv': 13.8,
        'gen_rating': 2.5, 'gen_xd': 0.14, 'temp': 35, 'alt': 100,
        'dist_loss': 1.5, 'gen_parasitic': 3.0,
        # Default MTBF/MTTR (IEEE 493)
        'gen_mtbf': 1500, 'gen_mttr': 24, 
        'bus_mtbf': 500000, 'bus_mttr': 12,
        'cb_mtbf': 300000, 'cb_mttr': 8,
        'cable_mtbf': 500000, 'cable_mttr': 24,
        'tx_mtbf': 200000, 'tx_mttr': 72
    }

def get(k): return st.session_state['inputs_v9'].get(k)
def set_k(k, v): st.session_state['inputs_v9'][k] = v

with st.sidebar:
    st.title("Inputs v9.0")
    
    with st.expander("1. Load & Strategy", expanded=True):
        st.number_input("IT Load (MW)", 1.0, 500.0, float(get('p_it')), key='p_it', on_change=lambda: set_k('p_it', st.session_state.p_it))
        st.number_input("Target Avail (%)", 90.0, 99.99999, float(get('req_avail')), format="%.5f", key='req_avail', on_change=lambda: set_k('req_avail', st.session_state.req_avail))
        st.selectbox("Voltage Mode", ["Auto-Recommend", "Manual"], index=0, key='volts_mode', on_change=lambda: set_k('volts_mode', st.session_state.volts_mode))
        if get('volts_mode') == 'Manual':
            st.number_input("Manual kV", 0.4, 69.0, float(get('volts_kv')), key='volts_kv', on_change=lambda: set_k('volts_kv', st.session_state.volts_kv))

    with st.expander("2. Reliability (IEEE 493)"):
        c1, c2 = st.columns(2)
        c1.number_input("Gen MTBF (h)", 100, 100000, int(get('gen_mtbf')), key='gen_mtbf', on_change=lambda: set_k('gen_mtbf', st.session_state.gen_mtbf))
        c2.number_input("Gen MTTR (h)", 1, 1000, int(get('gen_mttr')), key='gen_mttr', on_change=lambda: set_k('gen_mttr', st.session_state.gen_mttr))
        st.caption("Default values used for Bus/Cable/Tx if not edited.")

    with st.expander("3. Tech Specs"):
        st.number_input("Gen Rating MW", 0.5, 20.0, float(get('gen_rating')), key='gen_rating', on_change=lambda: set_k('gen_rating', st.session_state.gen_rating))
        st.number_input("Xd\" (pu)", 0.05, 0.5, float(get('gen_xd')), key='gen_xd', on_change=lambda: set_k('gen_xd', st.session_state.gen_xd))

res = solve_topology_v9(st.session_state['inputs_v9'])

# ==============================================================================
# 5. DASHBOARD
# ==============================================================================

st.title("CAT Topology Designer v9.0")
st.subheader("Auto-Correction Engine")

# Auto-Correction Log
if len(res['log']) > 0:
    with st.expander("🤖 Logic Execution Log", expanded=False):
        for entry in res['log']:
            st.markdown(entry)

if res['pass']:
    sol = res['sol']
    
    # If the voltage chosen is different from standard 13.8, highlight it
    if sol['voltage'] > 13.8:
        st.markdown(f"""
        <div class="info-box">
            ℹ️ <b>Auto-Correction Applied:</b> 
            Lower voltages failed physical constraints (Amps/kA). 
            System automatically upgraded to <b>{sol['voltage']} kV</b> to meet requirements.
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown('<div class="success-box">✅ <b>System Validated:</b> Architecture meets N-1 Bus Tolerance, Short Circuit Limits, and Availability Target.</div>', unsafe_allow_html=True)

    c1, c2, c3, c4 = st.columns(4)
    with c1: st.markdown(f'<div class="kpi-card"><div class="metric-value">{sol["avail"]*100:.6f}%</div><div class="metric-label">Availability</div></div>', unsafe_allow_html=True)
    with c2: st.markdown(f'<div class="kpi-card"><div class="metric-value">{sol["n_buses"]}</div><div class="metric-label">Switchgear Buses</div></div>', unsafe_allow_html=True)
    with c3: st.markdown(f'<div class="kpi-card"><div class="metric-value">{sol["bus_ka"]:.1f} kA</div><div class="metric-label">Short Circuit</div></div>', unsafe_allow_html=True)
    with c4: st.markdown(f'<div class="kpi-card"><div class="metric-value">{sol["n_total"]}</div><div class="metric-label">Total Generators</div></div>', unsafe_allow_html=True)

    st.divider()
    
    t1, t2 = st.tabs(["Diagram", "Specs"])
    
    with t1:
        dot = graphviz.Digraph()
        dot.attr(rankdir='LR')
        for i in range(1, sol['n_buses']+1):
            with dot.subgraph(name=f"cluster_{i}") as c:
                c.attr(label=f"Cluster {i}", color="grey")
                c.node(f"B{i}", f"Bus {i}\n{sol['voltage']} kV", shape="rect", style="filled", fillcolor="#FFCD11")
                c.node(f"G{i}", f"{sol['gens_per_bus']}x Gens", shape="folder", style="filled", fillcolor="#D1F2EB")
                c.edge(f"G{i}", f"B{i}")
        # Ring
        for i in range(1, sol['n_buses']+1):
             nxt = 1 if i == sol['n_buses'] else i+1
             dot.edge(f"B{i}", f"B{nxt}", label="Tie", dir="none")
        st.graphviz_chart(dot, use_container_width=True)
        

[Image of high voltage electrical substation switchgear]


else:
    st.markdown('<div class="error-box">❌ <b>Optimization Failed:</b> No valid configuration found in any voltage tier. Check inputs.</div>', unsafe_allow_html=True)
