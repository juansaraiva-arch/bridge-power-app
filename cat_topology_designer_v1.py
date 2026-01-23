import streamlit as st
import pandas as pd
import numpy as np
import math
import graphviz
from scipy.stats import binom

# --- PAGE CONFIG ---
st.set_page_config(page_title="CAT Topology v14.0 (Auto-BaaH)", page_icon="⚡", layout="wide")

# --- CSS ---
st.markdown("""
<style>
    @media print {
        [data-testid="stSidebar"], [data-testid="stHeader"], footer, .stButton { display: none !important; }
        .block-container { padding: 0 !important; margin: 0 !important; }
    }
    .upgrade-box { background-color: #e6f7ff; border: 1px solid #91d5ff; padding: 15px; border-radius: 5px; color: #0050b3; margin-bottom: 10px; }
    .success-box { background-color: #d4edda; border: 1px solid #c3e6cb; padding: 15px; border-radius: 5px; color: #155724; margin-bottom: 10px; }
    .kpi-card { background-color: #f8f9fa; padding: 10px; border-radius: 5px; text-align: center; border-left: 5px solid #2E86C1; box-shadow: 1px 1px 3px rgba(0,0,0,0.1); }
    .metric-value { font-size: 22px; font-weight: bold; }
    .metric-label { font-size: 13px; color: #666; }
</style>
""", unsafe_allow_html=True)

# ==============================================================================
# 1. MATH ENGINE
# ==============================================================================

def calc_avail(mtbf, mttr):
    if mtbf + mttr == 0: return 0.0
    return mtbf / (mtbf + mttr)

def rel_k_out_n(n_needed, n_total, p_unit):
    if n_total < n_needed: return 0.0
    prob = 0.0
    for k in range(n_needed, n_total + 1):
        prob += binom.pmf(k, n_total, p_unit)
    return prob

def calc_sc(kv, mva, xd, n_par):
    if kv == 0 or xd == 0: return 999999.0, 0.0
    i_base = (mva * 1e6) / (math.sqrt(3) * (kv * 1000))
    i_sc = (i_base / xd) * n_par
    return i_sc, i_base

# ==============================================================================
# 2. SOLVER (AUTO-TOPOLOGY UPGRADE)
# ==============================================================================

def solve_topology_v14(inputs):
    res = {'pass': False, 'log': [], 'diag': {'reasons': set(), 'best_attempt': None}}
    
    # Inputs processing
    p_it = inputs['p_it']
    p_gross = (p_it * (1 + inputs['dc_aux']/100)) / ((1 - inputs['dist_loss']/100) * (1 - inputs['gen_parasitic']/100))
    
    # Base Reliability
    a_gen = calc_avail(inputs['gen_mtbf'], inputs['gen_mttr'])
    a_bus = calc_avail(inputs['bus_mtbf'], inputs['bus_mttr'])
    a_bess = calc_avail(inputs['bess_mtbf'], inputs['bess_mttr'])
    a_dist = calc_avail(inputs['cb_mtbf'], inputs['cb_mttr']) * calc_avail(inputs['tx_mtbf'], inputs['tx_mttr'])
    
    gen_site_mw = inputs['gen_rating'] * (1.0 - (max(0, (inputs['temp']-25)*0.01)))
    gen_mva = gen_site_mw / 0.8
    
    # Voltage Strategy
    kv_list = [inputs['volts_kv']] if inputs['volts_mode'] == 'Manual' else [13.8, 34.5, 69.0]
    if inputs['volts_mode'] == 'Auto-Recommend' and p_gross < 10: kv_list.insert(0, 4.16)

    final_sol = None
    best_failed_attempt = None
    fail_reasons_global = set()

    # --- MAIN LOOP ---
    for kv in kv_list:
        min_gens = math.ceil(p_gross / gen_site_mw)
        
        for n_total in range(min_gens, min_gens + 60):
            for n_buses in range(2, 12):
                per_bus = math.ceil(n_total / n_buses)
                real_total = n_buses * per_bus
                
                # 1. Physics (Shared for Ring and BaaH)
                isc, i_nom = calc_sc(kv, gen_mva, inputs['gen_xd'], real_total)
                bus_amps = per_bus * i_nom
                
                phy_pass = True
                fail_reasons = []
                if isc > 63000: phy_pass=False; fail_reasons.append(f"SC > 63kA")
                if bus_amps > 4000: phy_pass=False; fail_reasons.append(f"Amps > 4000A")
                
                if not phy_pass:
                    # Update diagnostics even if physics fail
                    if not best_failed_attempt: 
                        best_failed_attempt = {'kv': kv, 'reasons': fail_reasons, 'avail': 0}
                    continue

                # 2. BESS (Shared)
                step_req_mw = p_it * (inputs['step_req']/100.0)
                gens_avail_step = real_total - per_bus 
                gen_step_mw = gens_avail_step * gen_site_mw * (inputs['gen_step_cap']/100.0)
                bess_needed_mw = max(0, step_req_mw - gen_step_mw)
                if inputs['bess_force']: bess_needed_mw = max(bess_needed_mw, inputs['bess_manual_mw'])
                
                n_bess = 0; bess_rel = 1.0
                if bess_needed_mw > 0:
                    n_bess = math.ceil(bess_needed_mw / inputs['bess_inv_mw'])
                    bess_per = math.ceil(n_bess / n_buses)
                    bess_rel = rel_k_out_n(n_bess, bess_per * n_buses, a_bess)

                # --- TOPOLOGY BRANCHING ---
                # Strategy: Try Ring first. If Avail fails, Try BaaH.
                
                topologies_to_try = ["Ring", "BaaH"]
                
                for topo in topologies_to_try:
                    
                    # N-1 Bus Tolerance Check
                    # For Ring: Lose 1 bus capacity.
                    # For BaaH: Bus loss doesn't lose capacity (Redundant), but we simulate breaker failure logic.
                    # Simplified: BaaH is inherently tolerant. Ring needs check.
                    
                    tol_pass = True
                    if topo == "Ring":
                        surviving_mw = (real_total - per_bus - 1) * gen_site_mw
                        if surviving_mw < p_gross: tol_pass = False
                    
                    if not tol_pass: 
                        fail_reasons.append("N-1 Tol Fail")
                        continue

                    # Reliability Calc
                    n_load = math.ceil(p_gross / gen_site_mw)
                    
                    if topo == "Ring":
                        p_bus_ok = a_bus ** n_buses
                        r_s0 = rel_k_out_n(n_load, real_total, a_gen)
                        p_bus_fail = math.comb(n_buses, 1) * (1-a_bus) * (a_bus**(n_buses-1))
                        r_s1 = rel_k_out_n(n_load, real_total - per_bus, a_gen)
                        gen_sys_rel = (p_bus_ok * r_s0) + (p_bus_fail * r_s1)
                    else: # BaaH
                        # Bus is fully redundant. Reliability is limited by Gens + Breakers.
                        # We assume Bus Avail ~ 1.0 due to parallel redundancy.
                        gen_sys_rel = rel_k_out_n(n_load, real_total, a_gen)
                    
                    total_avail = gen_sys_rel * bess_rel * a_dist
                    
                    avail_pass = total_avail >= (inputs['req_avail']/100.0)
                    
                    candidate = {
                        'kv': kv, 'n_buses': n_buses, 'n_total': real_total, 'per_bus': per_bus,
                        'avail': total_avail, 'isc': isc, 'amps': bus_amps, 'topo': topo,
                        'bess_active': (n_bess > 0), 'n_bess': n_bess, 'bess_mw': bess_needed_mw,
                        'gen_step': gen_step_mw, 'req_step': step_req_mw, 'load': p_gross,
                        'reasons': [f"Avail {total_avail*100:.5f}%"] if not avail_pass else []
                    }

                    if avail_pass:
                        final_sol = candidate
                        break # Found valid solution!
                    
                    # Track best failure
                    if not best_failed_attempt or total_avail > best_failed_attempt['avail']:
                        best_failed_attempt = candidate
                        for r in fail_reasons: fail_reasons_global.add(r)

                if final_sol: break
            if final_sol: break
        if final_sol: break

    res['pass'] = (final_sol is not None)
    res['sol'] = final_sol
    res['diag']['best_attempt'] = best_failed_attempt
    res['diag']['reasons'] = list(fail_reasons_global)
    return res

# ==============================================================================
# 3. UI INPUTS
# ==============================================================================

if 'inputs_v14' not in st.session_state:
    st.session_state['inputs_v14'] = {
        'p_it': 100.0, 'dc_aux': 15.0, 'req_avail': 99.999, 'volts_mode': 'Auto-Recommend', 'volts_kv': 13.8,
        'step_req': 40.0, 'gen_rating': 2.5, 'gen_xd': 0.14, 'gen_step_cap': 25.0,
        'dist_loss': 1.5, 'gen_parasitic': 3.0, 'temp': 35, 'alt': 100,
        'bess_force': False, 'bess_manual_mw': 20.0, 'bess_inv_mw': 3.8,
        'gen_mtbf': 2000, 'gen_mttr': 24, 'bess_mtbf': 5000, 'bess_mttr': 48,
        'bus_mtbf': 500000, 'bus_mttr': 12, 'cb_mtbf': 300000, 'cb_mttr': 8,
        'tx_mtbf': 200000, 'tx_mttr': 72
    }

def get(k): return st.session_state['inputs_v14'].get(k)
def set_k(k, v): st.session_state['inputs_v14'][k] = v

with st.sidebar:
    st.title("Inputs v14.0")
    with st.expander("1. Load & Strategy", expanded=True):
        st.number_input("IT Load (MW)", 1.0, 500.0, float(get('p_it')), key='p_it', on_change=lambda: set_k('p_it', st.session_state.p_it))
        st.number_input("Target Avail (%)", 90.0, 99.99999, float(get('req_avail')), format="%.5f", key='req_avail', on_change=lambda: set_k('req_avail', st.session_state.req_avail))
        st.number_input("AI Step Load (%)", 0.0, 100.0, float(get('step_req')), key='step_req', on_change=lambda: set_k('step_req', st.session_state.step_req))
        opt = st.selectbox("Voltage", ["Auto-Recommend", "Manual"], index=0, key='volts_mode', on_change=lambda: set_k('volts_mode', st.session_state.volts_mode))
        if opt == 'Manual': st.number_input("Manual kV", 0.4, 69.0, float(get('volts_kv')), key='volts_kv', on_change=lambda: set_k('volts_kv', st.session_state.volts_kv))

    with st.expander("2. Reliability"):
        c1, c2 = st.columns(2)
        c1.number_input("Gen MTBF", value=int(get('gen_mtbf')), key='gen_mtbf', on_change=lambda: set_k('gen_mtbf', st.session_state.gen_mtbf))
        c2.number_input("Gen MTTR", value=int(get('gen_mttr')), key='gen_mttr', on_change=lambda: set_k('gen_mttr', st.session_state.gen_mttr))

    with st.expander("3. Tech Specs"):
        c1, c2 = st.columns(2)
        c1.number_input("Gen MW", value=float(get('gen_rating')), key='gen_rating', on_change=lambda: set_k('gen_rating', st.session_state.gen_rating))
        st.checkbox("Force BESS", value=get('bess_force'), key='bess_force', on_change=lambda: set_k('bess_force', st.session_state.bess_force))

res = solve_topology_v14(st.session_state['inputs_v14'])

# ==============================================================================
# 4. DASHBOARD
# ==============================================================================

st.title("CAT Topology Designer v14.0")
st.subheader("Auto-Scaling Architecture (Ring → BaaH)")

if res['pass']:
    sol = res['sol']
    
    # NOTIFICATION AREA
    if sol['topo'] == "BaaH":
        st.markdown(f"""
        <div class="upgrade-box">
            🚀 <b>Auto-Upgrade Triggered:</b> 
            Standard Ring topology failed to meet {get('req_avail')}% availability. 
            System automatically upgraded to <b>Breaker-and-a-Half (BaaH)</b> topology.
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f'<div class="success-box">✅ <b>Optimal Solution:</b> Standard Iso-Parallel Ring meets all targets.</div>', unsafe_allow_html=True)

    # KPIS
    c1, c2, c3, c4 = st.columns(4)
    c1.markdown(f'<div class="kpi-card"><div class="metric-value">{sol["avail"]*100:.6f}%</div><div class="metric-label">Availability</div></div>', unsafe_allow_html=True)
    c2.markdown(f'<div class="kpi-card"><div class="metric-value">{sol["n_buses"]}</div><div class="metric-label">Buses / Bays</div></div>', unsafe_allow_html=True)
    c3.markdown(f'<div class="kpi-card"><div class="metric-value">{sol["bess_mw"]:.1f} MW</div><div class="metric-label">BESS Gap</div></div>', unsafe_allow_html=True)
    c4.markdown(f'<div class="kpi-card"><div class="metric-value">{sol["n_total"]}</div><div class="metric-label">Total Gens</div></div>', unsafe_allow_html=True)

    st.divider()
    
    t1, t2 = st.tabs(["📐 System Diagram", "🔍 Topology Detail"])
    
    with t1:
        dot = graphviz.Digraph()
        dot.attr(rankdir='TB', splines='ortho')
        
        # DYNAMIC DRAWING BASED ON TOPOLOGY
        if sol['topo'] == "Ring":
            # (Reuse Ring drawing logic from v13)
            with dot.subgraph(name='cluster_main') as c:
                c.attr(style='invis')
                for i in range(1, sol['n_buses'] + 1):
                    c.node(f'Bus_{i}', label=f'Bus {i}\n{sol["amps"]:.0f}A', shape='underline', width='2.5')
                    c.node(f'G_{i}', label='G', shape='circle')
                    c.edge(f'G_{i}', f'Bus_{i}')
            for i in range(1, sol['n_buses'] + 1):
                nxt = i + 1 if i < sol['n_buses'] else 1
                dot.edge(f'Bus_{i}', f'Bus_{nxt}', label='X', dir='none')
                
        else: # Breaker-and-a-Half Diagram
            st.caption("Displaying Breaker-and-a-Half Configuration (Double Bus)")
            # Two Main Buses
            dot.node('BusA', f'MAIN BUS A ({sol["kv"]}kV)', shape='rect', style='filled', fillcolor='#FFCD11', width='10')
            dot.node('BusB', f'MAIN BUS B ({sol["kv"]}kV)', shape='rect', style='filled', fillcolor='#FFCD11', width='10')
            
            # Bays
            for i in range(1, sol['n_buses'] + 1):
                with dot.subgraph(name=f'bay_{i}') as bay:
                    bay.attr(label=f'Bay {i}')
                    # 3 Breakers per Bay
                    cb1 = f'CB_{i}_1'
                    cb2 = f'CB_{i}_2'
                    cb3 = f'CB_{i}_3'
                    
                    dot.node(cb1, 'X', shape='square')
                    dot.node(cb2, 'X', shape='square')
                    dot.node(cb3, 'X', shape='square')
                    
                    # Connections
                    dot.edge('BusA', cb1, dir='none')
                    dot.edge(cb1, cb2, dir='none') # Between CB1 and CB2 connects Circuit 1
                    dot.edge(cb2, cb3, dir='none') # Between CB2 and CB3 connects Circuit 2
                    dot.edge(cb3, 'BusB', dir='none')
                    
                    # Circuits (Gens/Feeders)
                    # Node between CB1/CB2
                    j1 = f'J_{i}_1'
                    dot.node(j1, '', shape='point', width='0')
                    dot.edge(cb1, j1, dir='none', len='0.1') 
                    dot.edge(j1, cb2, dir='none', len='0.1')
                    dot.node(f'G_{i}', f'{sol["per_bus"]}x Gens', shape='folder')
                    dot.edge(j1, f'G_{i}')
                    
                    # Node between CB2/CB3
                    j2 = f'J_{i}_2'
                    dot.node(j2, '', shape='point', width='0')
                    dot.edge(cb2, j2, dir='none', len='0.1')
                    dot.edge(j2, cb3, dir='none', len='0.1')
                    dot.node(f'F_{i}', 'Feeders', shape='invtriangle')
                    dot.edge(j2, f'F_{i}')

        st.graphviz_chart(dot, use_container_width=True)

    with t2:
        if sol['topo'] == "BaaH":
            st.info("""
            **Breaker-and-a-Half Logic:**
            - **Redundancy:** 2 Main Buses + 3 Breakers per 2 Circuits.
            - **Fault Tolerance:** Can lose ANY Bus or ANY Breaker without losing the circuit.
            - **Cost:** Higher CAPEX, Maximum Availability.
            """)
            
        else:
            st.info("Standard Split-Bus Ring Logic.")
            sdot = graphviz.Digraph(rankdir='LR')
            sdot.node('A', 'Bus A', shape='rect', style='filled', fillcolor='#FFCD11')
            sdot.node('B', 'Bus B', shape='filled', fillcolor='#FFCD11')
            sdot.node('T', 'Tie', shape='circle')
            sdot.edge('A','T'); sdot.edge('T','B')
            st.graphviz_chart(sdot)

else:
    st.markdown('<div class="error-box">❌ <b>Optimization Failed:</b> Even Breaker-and-a-Half topology could not meet constraints.</div>', unsafe_allow_html=True)
    if res['diag']['best_attempt']:
        st.write(f"Best Attempt: {res['diag']['best_attempt']['avail']*100:.5f}%")
