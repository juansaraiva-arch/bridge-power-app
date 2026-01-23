import streamlit as st
import pandas as pd
import numpy as np
import math
import graphviz
from scipy.stats import binom

# --- PAGE CONFIG ---
st.set_page_config(page_title="CAT Topology v11.0 (Integrated)", page_icon="⚡", layout="wide")

# --- CSS ---
st.markdown("""
<style>
    @media print {
        [data-testid="stSidebar"], [data-testid="stHeader"], footer, .stButton { display: none !important; }
        .block-container { padding: 0 !important; margin: 0 !important; }
    }
    .info-box { background-color: #cce5ff; border: 1px solid #b8daff; padding: 15px; border-radius: 5px; color: #004085; margin-bottom: 10px; }
    .success-box { background-color: #d4edda; border: 1px solid #c3e6cb; padding: 15px; border-radius: 5px; color: #155724; margin-bottom: 10px; }
    .error-box { background-color: #f8d7da; border: 1px solid #f5c6cb; padding: 15px; border-radius: 5px; color: #721c24; margin-bottom: 10px; }
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
# 2. SOLVER (AUTO-VOLTAGE + BESS + PHYSICS)
# ==============================================================================

def solve_topology_v11(inputs):
    res = {'pass': False, 'log': [], 'diag': {'reasons': set(), 'best_attempt': None}}
    
    # Loads
    p_it = inputs['p_it']
    p_gross = (p_it * (1 + inputs['dc_aux']/100)) / ((1 - inputs['dist_loss']/100) * (1 - inputs['gen_parasitic']/100))
    
    # Availabilities
    a_gen = calc_avail(inputs['gen_mtbf'], inputs['gen_mttr'])
    a_bus = calc_avail(inputs['bus_mtbf'], inputs['bus_mttr'])
    a_bess = calc_avail(inputs['bess_mtbf'], inputs['bess_mttr'])
    a_dist = calc_avail(inputs['cb_mtbf'], inputs['cb_mttr']) * calc_avail(inputs['tx_mtbf'], inputs['tx_mttr'])
    
    # Gen Specs
    gen_site_mw = inputs['gen_rating'] * (1.0 - (max(0, (inputs['temp']-25)*0.01)))
    gen_mva = gen_site_mw / 0.8
    
    # --- VOLTAGE STRATEGY ---
    # Defines the order of evaluation. 
    # Logic: Start low. If physics fail, move up automatically.
    kv_list = [inputs['volts_kv']] if inputs['volts_mode'] == 'Manual' else [13.8, 34.5, 69.0]
    if inputs['volts_mode'] == 'Auto-Recommend' and p_gross < 10: 
        kv_list.insert(0, 4.16)
        if p_gross < 2: kv_list.insert(0, 0.48)

    final_sol = None
    best_failed_attempt = None
    
    # Iterate through Voltage Tiers
    for kv in kv_list:
        min_gens = math.ceil(p_gross / gen_site_mw)
        
        # Iterate Fleet Size & Topology
        for n_total in range(min_gens, min_gens + 50):
            for n_buses in range(2, 12):
                per_bus = math.ceil(n_total / n_buses)
                real_total = n_buses * per_bus
                
                fail_reasons = []
                
                # 1. Physics Check (The Filter)
                isc, i_nom = calc_sc(kv, gen_mva, inputs['gen_xd'], real_total)
                bus_amps = per_bus * i_nom
                
                phy_pass = True
                if isc > 63000: 
                    phy_pass = False; fail_reasons.append(f"SC > 63kA ({isc/1000:.1f}kA)")
                if bus_amps > 4000: 
                    phy_pass = False; fail_reasons.append(f"Amps > 4000A ({bus_amps:.0f}A)")
                
                # If physics fail, this config is invalid for this voltage.
                # But we don't break the loop, we try other bus configs or move to next voltage.
                
                # 2. Fault Tolerance (N-1 Bus)
                surviving_mw = (real_total - per_bus - 1) * gen_site_mw
                tol_pass = surviving_mw >= p_gross
                if not tol_pass: fail_reasons.append("N-1 Tolerance Fail")
                
                # 3. BESS & Step Load
                step_req_mw = p_it * (inputs['step_req']/100.0)
                gens_avail_step = real_total - per_bus # Worst case
                gen_step_mw = gens_avail_step * gen_site_mw * (inputs['gen_step_cap']/100.0)
                
                bess_needed_mw = max(0, step_req_mw - gen_step_mw)
                if inputs['bess_force']: bess_needed_mw = max(bess_needed_mw, inputs['bess_manual_mw'])
                
                n_bess_units = 0
                bess_rel = 1.0
                
                if bess_needed_mw > 0:
                    n_bess_units = math.ceil(bess_needed_mw / inputs['bess_inv_mw'])
                    bess_per_bus = math.ceil(n_bess_units / n_buses)
                    n_bess_real = bess_per_bus * n_buses
                    bess_rel = rel_k_out_n(n_bess_units, n_bess_real, a_bess)
                
                # 4. Reliability
                n_load = math.ceil(p_gross / gen_site_mw)
                p_bus_ok = a_bus ** n_buses
                r_s0 = rel_k_out_n(n_load, real_total, a_gen)
                p_bus_fail = math.comb(n_buses, 1) * (1-a_bus) * (a_bus**(n_buses-1))
                r_s1 = rel_k_out_n(n_load, real_total - per_bus, a_gen)
                
                gen_sys_rel = (p_bus_ok * r_s0) + (p_bus_fail * r_s1)
                total_avail = gen_sys_rel * bess_rel * a_dist
                
                avail_pass = total_avail >= (inputs['req_avail']/100.0)
                if not avail_pass: fail_reasons.append(f"Avail {total_avail*100:.5f}% < Target")

                # Solution Candidate
                sol_candidate = {
                    'kv': kv, 'n_buses': n_buses, 'n_total': real_total, 'per_bus': per_bus,
                    'avail': total_avail, 'isc': isc, 'amps': bus_amps,
                    'bess_active': (n_bess_units > 0), 'n_bess': n_bess_units if n_bess_units > 0 else 0,
                    'bess_mw': bess_needed_mw, 'gen_step': gen_step_mw, 'req_step': step_req_mw,
                    'load': p_gross, 'reasons': fail_reasons
                }

                if phy_pass and tol_pass and avail_pass:
                    final_sol = sol_candidate
                    break # Found valid config for this voltage
                
                # Track best failure for diagnostics
                if not best_failed_attempt: best_failed_attempt = sol_candidate
                elif sol_candidate['avail'] > best_failed_attempt['avail']: best_failed_attempt = sol_candidate

            if final_sol: break # Stop searching buses/gens
        if final_sol: break # Stop searching voltages

    res['pass'] = (final_sol is not None)
    res['sol'] = final_sol
    res['diag']['best_attempt'] = best_failed_attempt
    return res

# ==============================================================================
# 3. UI INPUTS (Restored from previous steps)
# ==============================================================================

if 'inputs_v11' not in st.session_state:
    st.session_state['inputs_v11'] = {
        'p_it': 100.0, 'dc_aux': 15.0, 'req_avail': 99.999, 'volts_mode': 'Auto-Recommend', 'volts_kv': 13.8,
        'step_req': 40.0, 'gen_rating': 2.5, 'gen_xd': 0.14, 'gen_step_cap': 25.0,
        'dist_loss': 1.5, 'gen_parasitic': 3.0, 'temp': 35, 'alt': 100,
        'bess_force': False, 'bess_manual_mw': 20.0, 'bess_inv_mw': 3.8,
        'gen_mtbf': 2000, 'gen_mttr': 24, 'bess_mtbf': 5000, 'bess_mttr': 48,
        'bus_mtbf': 500000, 'bus_mttr': 12, 'cb_mtbf': 300000, 'cb_mttr': 8,
        'tx_mtbf': 200000, 'tx_mttr': 72
    }

def get(k): return st.session_state['inputs_v11'].get(k)
def set_k(k, v): st.session_state['inputs_v11'][k] = v

with st.sidebar:
    st.title("Inputs v11.0")
    with st.expander("1. Load & Strategy", expanded=True):
        st.number_input("IT Load (MW)", 1.0, 500.0, float(get('p_it')), key='p_it', on_change=lambda: set_k('p_it', st.session_state.p_it))
        st.number_input("Target Avail (%)", 90.0, 99.99999, float(get('req_avail')), format="%.5f", key='req_avail', on_change=lambda: set_k('req_avail', st.session_state.req_avail))
        st.number_input("AI Step Load (%)", 0.0, 100.0, float(get('step_req')), key='step_req', on_change=lambda: set_k('step_req', st.session_state.step_req))
        
        opt = st.selectbox("Voltage", ["Auto-Recommend", "Manual"], index=0, key='volts_mode', on_change=lambda: set_k('volts_mode', st.session_state.volts_mode))
        if opt == 'Manual': st.number_input("Manual kV", 0.4, 69.0, float(get('volts_kv')), key='volts_kv', on_change=lambda: set_k('volts_kv', st.session_state.volts_kv))

    with st.expander("2. Reliability (MTBF/MTTR)"):
        c1, c2 = st.columns(2)
        c1.number_input("Gen MTBF", value=int(get('gen_mtbf')), key='gen_mtbf', on_change=lambda: set_k('gen_mtbf', st.session_state.gen_mtbf))
        c2.number_input("Gen MTTR", value=int(get('gen_mttr')), key='gen_mttr', on_change=lambda: set_k('gen_mttr', st.session_state.gen_mttr))
        c1.number_input("BESS MTBF", value=int(get('bess_mtbf')), key='bess_mtbf', on_change=lambda: set_k('bess_mtbf', st.session_state.bess_mtbf))
        c2.number_input("BESS MTTR", value=int(get('bess_mttr')), key='bess_mttr', on_change=lambda: set_k('bess_mttr', st.session_state.bess_mttr))

    with st.expander("3. Tech Specs"):
        c1, c2 = st.columns(2)
        c1.number_input("Gen MW", value=float(get('gen_rating')), key='gen_rating', on_change=lambda: set_k('gen_rating', st.session_state.gen_rating))
        c2.number_input("Gen Step Cap %", value=float(get('gen_step_cap')), key='gen_step_cap', on_change=lambda: set_k('gen_step_cap', st.session_state.gen_step_cap))
        st.checkbox("Force BESS", value=get('bess_force'), key='bess_force', on_change=lambda: set_k('bess_force', st.session_state.bess_force))

res = solve_topology_v11(st.session_state['inputs_v11'])

# ==============================================================================
# 4. DASHBOARD
# ==============================================================================

st.title("CAT Topology Designer v11.0")
st.caption("Integrated: Physics + BESS + ANSI Diagrams + Auto-Voltage")

if res['pass']:
    sol = res['sol']
    
    # AUTO-RECOMMEND FEEDBACK
    if get('volts_mode') == 'Auto-Recommend':
        if sol['kv'] > 13.8:
            st.markdown(f"""<div class="info-box">ℹ️ <b>Auto-Correction:</b> Voltage set to <b>{sol['kv']} kV</b> to manage Short Circuit levels ({sol['isc']/1000:.1f} kA). Lower voltages failed physics checks.</div>""", unsafe_allow_html=True)
        else:
            st.markdown(f"""<div class="success-box">✅ <b>Optimized:</b> Configuration valid at standard <b>{sol['kv']} kV</b>.</div>""", unsafe_allow_html=True)

    # KPI
    c1, c2, c3, c4 = st.columns(4)
    c1.markdown(f'<div class="kpi-card"><div class="metric-value">{sol["avail"]*100:.6f}%</div><div class="metric-label">Availability</div></div>', unsafe_allow_html=True)
    c2.markdown(f'<div class="kpi-card"><div class="metric-value">{sol["n_buses"]}</div><div class="metric-label">Buses</div></div>', unsafe_allow_html=True)
    c3.markdown(f'<div class="kpi-card"><div class="metric-value">{sol["bess_mw"]:.1f} MW</div><div class="metric-label">BESS Cap</div></div>', unsafe_allow_html=True)
    c4.markdown(f'<div class="kpi-card"><div class="metric-value">{sol["n_total"]}</div><div class="metric-label">Total Gens</div></div>', unsafe_allow_html=True)

    st.divider()
    
    # TABS (Preserving ANSI + Detailed Specs)
    t1, t2 = st.tabs(["📐 ANSI Single Line Diagram", "📋 Detailed Specs & Internal Logic"])
    
    with t1:
        # ANSI DIAGRAM (From v10)
        dot = graphviz.Digraph()
        dot.attr(rankdir='TB', splines='ortho', nodesep='0.6')
        with dot.subgraph(name='cluster_main') as c:
            c.attr(style='invis')
            for i in range(1, sol['n_buses'] + 1):
                c.node(f'Bus_{i}', label=f'Bus {i}\n{sol["amps"]:.0f}A', shape='underline', width='2.5')
                c.node(f'G_{i}', label='G', shape='circle', fixedsize='true', width='0.6')
                c.edge(f'G_{i}', f'Bus_{i}', label='CB')
                if sol['bess_active']:
                    c.node(f'B_{i}', label='BESS', shape='box3d')
                    c.edge(f'B_{i}', f'Bus_{i}', label='CB')
                c.node(f'F_{i}', label=f'Feeders', shape='invtriangle')
                c.edge(f'Bus_{i}', f'F_{i}', label='CB')
        # Ties
        for i in range(1, sol['n_buses'] + 1):
            nxt = i + 1 if i < sol['n_buses'] else 1
            dot.edge(f'Bus_{i}', f'Bus_{nxt}', label='Tie X', dir='none')
        st.graphviz_chart(dot, use_container_width=True)
        
    with t2:
        # DETAILED SPECS (Restored Split Bus Detail from v7/8)
        c1, c2 = st.columns([1, 1])
        with c1:
            st.markdown("### Technical Data")
            st.write(f"**Gross Load:** {sol['load']:.1f} MW")
            st.write(f"**Short Circuit:** {sol['isc']/1000:.2f} kA (Limit 63kA)")
            st.write(f"**Bus Amps:** {sol['amps']:.0f} A (Limit 4000A)")
            if sol['bess_active']:
                st.write(f"**BESS Status:** {sol['n_bess']} Units (Step Load Support)")
            else:
                st.write("**BESS Status:** Not Required (Gens handle Step)")
        
        with c2:
            st.markdown("### 🔍 Switchgear Internal Logic (Split-Bus)")
            st.info("Showing Internal 'Main-Tie-Main' redundancy per Switchgear.")
            # Simple Split Bus Graph
            sdot = graphviz.Digraph()
            sdot.attr(rankdir='LR')
            sdot.node('BA', 'Bus A', shape='rect', style='filled', fillcolor='#FFCD11')
            sdot.node('BB', 'Bus B', shape='rect', style='filled', fillcolor='#FFCD11')
            sdot.node('Tie', 'Tie (NC)', shape='circle')
            sdot.edge('BA', 'Tie', dir='none')
            sdot.edge('Tie', 'BB', dir='none')
            sdot.node('GA', f'{math.ceil(sol["per_bus"]/2)}x Gens', shape='folder')
            sdot.node('GB', f'{int(sol["per_bus"]/2)}x Gens', shape='folder')
            sdot.edge('GA', 'BA')
            sdot.edge('GB', 'BB')
            st.graphviz_chart(sdot, use_container_width=True)

else:
    # FAILURE DIAGNOSTICS
    st.markdown('<div class="error-box">❌ <b>Analysis Failed:</b> Could not find a valid configuration even after Auto-Correction.</div>', unsafe_allow_html=True)
    
    best = res['diag']['best_attempt']
    if best:
        st.write("**Why?** Here is the failure analysis of the best attempt:")
        for r in best['reasons']:
            st.warning(f"⚠️ {r}")
        st.code(f"Best Attempt: {best['kv']} kV | {best['avail']*100:.5f}% Avail | {best['isc']/1000:.1f} kA SC")
