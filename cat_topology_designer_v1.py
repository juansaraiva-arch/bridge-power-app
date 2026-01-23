import streamlit as st
import pandas as pd
import numpy as np
import math
import graphviz
from scipy.stats import binom

# --- PAGE CONFIG ---
st.set_page_config(page_title="CAT Topology v10.0 (AI-Step & ANSI SLD)", page_icon="⚡", layout="wide")

# --- CSS ---
st.markdown("""
<style>
    @media print {
        [data-testid="stSidebar"], [data-testid="stHeader"], footer, .stButton { display: none !important; }
        .block-container { padding: 0 !important; margin: 0 !important; }
    }
    .success-box { background-color: #d4edda; border: 1px solid #c3e6cb; padding: 15px; border-radius: 5px; color: #155724; margin-bottom: 10px; }
    .kpi-card { background-color: #f8f9fa; padding: 10px; border-radius: 5px; text-align: center; border-left: 5px solid #2E86C1; box-shadow: 1px 1px 3px rgba(0,0,0,0.1); }
    .metric-value { font-size: 22px; font-weight: bold; }
    .metric-label { font-size: 13px; color: #666; }
</style>
""", unsafe_allow_html=True)

# ==============================================================================
# 1. RELIABILITY & PHYSICS ENGINE
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
# 2. SOLVER (Included BESS Step Logic)
# ==============================================================================

def solve_topology_v10(inputs):
    res = {'pass': False, 'log': []}
    
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
    
    # Voltage Strategy
    kv_list = [inputs['volts_kv']] if inputs['volts_mode'] == 'Manual' else [13.8, 34.5, 69.0]
    if inputs['volts_mode'] == 'Auto-Recommend' and p_gross < 10: kv_list.insert(0, 4.16)

    final_sol = None
    
    for kv in kv_list:
        min_gens = math.ceil(p_gross / gen_site_mw)
        
        # Loop Configs
        for n_total in range(min_gens, min_gens + 40):
            for n_buses in range(2, 10):
                per_bus = math.ceil(n_total / n_buses)
                real_total = n_buses * per_bus
                
                # 1. Physics
                isc, i_nom = calc_sc(kv, gen_mva, inputs['gen_xd'], real_total)
                bus_amps = per_bus * i_nom
                
                if isc > 63000 or bus_amps > 4000: continue
                
                # 2. Fault Tolerance (N-1 Bus)
                surviving_mw = (real_total - per_bus - 1) * gen_site_mw
                if surviving_mw < p_gross: continue
                
                # 3. BESS & STEP LOAD LOGIC (CRITICAL)
                step_req_mw = p_it * (inputs['step_req']/100.0)
                # Gen Step Cap (Worst case: N-1 Bus condition)
                gens_avail_step = real_total - per_bus
                gen_step_mw = gens_avail_step * gen_site_mw * (inputs['gen_step_cap']/100.0)
                
                bess_needed_mw = max(0, step_req_mw - gen_step_mw)
                n_bess_units = 0
                bess_rel = 1.0
                
                if bess_needed_mw > 0 or inputs['bess_force']:
                    if inputs['bess_force']: bess_needed_mw = max(bess_needed_mw, inputs['bess_manual_mw'])
                    n_bess_units = math.ceil(bess_needed_mw / inputs['bess_inv_mw'])
                    
                    # Distribute BESS
                    bess_per_bus = math.ceil(n_bess_units / n_buses)
                    n_bess_real = bess_per_bus * n_buses
                    
                    # BESS Reliability (k-out-of-n)
                    # We need n_bess_units. We have n_bess_real.
                    bess_rel = rel_k_out_n(n_bess_units, n_bess_real, a_bess)
                
                # 4. System Reliability (Series)
                # Gen Subsystem
                n_load = math.ceil(p_gross / gen_site_mw)
                p_bus_ok = a_bus ** n_buses
                r_s0 = rel_k_out_n(n_load, real_total, a_gen)
                p_bus_fail = math.comb(n_buses, 1) * (1-a_bus) * (a_bus**(n_buses-1))
                r_s1 = rel_k_out_n(n_load, real_total - per_bus, a_gen)
                
                gen_sys_rel = (p_bus_ok * r_s0) + (p_bus_fail * r_s1)
                
                # Total Avail = Gen_Sys * BESS_Sys * Dist_Sys
                total_avail = gen_sys_rel * bess_rel * a_dist
                
                if total_avail >= (inputs['req_avail']/100.0):
                    final_sol = {
                        'n_buses': n_buses, 'n_total': real_total, 'per_bus': per_bus,
                        'kv': kv, 'isc': isc, 'amps': bus_amps, 'avail': total_avail,
                        'bess_active': (n_bess_units > 0), 'n_bess': n_bess_units if n_bess_units > 0 else 0,
                        'bess_mw': bess_needed_mw, 'gen_step': gen_step_mw, 'req_step': step_req_mw,
                        'load': p_gross
                    }
                    break
            if final_sol: break
        if final_sol: break
        
    res['pass'] = (final_sol is not None)
    res['sol'] = final_sol
    return res

# ==============================================================================
# 3. UI INPUTS
# ==============================================================================

if 'inputs_v10' not in st.session_state:
    st.session_state['inputs_v10'] = {
        'p_it': 100.0, 'dc_aux': 15.0, 'req_avail': 99.999, 'volts_mode': 'Auto-Recommend', 'volts_kv': 13.8,
        'step_req': 40.0, 'gen_rating': 2.5, 'gen_xd': 0.14, 'gen_step_cap': 25.0,
        'dist_loss': 1.5, 'gen_parasitic': 3.0, 'temp': 35, 'alt': 100,
        'bess_force': False, 'bess_manual_mw': 20.0, 'bess_inv_mw': 3.8,
        # MTBF/MTTR
        'gen_mtbf': 2000, 'gen_mttr': 24, 'bess_mtbf': 5000, 'bess_mttr': 48,
        'bus_mtbf': 500000, 'bus_mttr': 12, 'cb_mtbf': 300000, 'cb_mttr': 8,
        'tx_mtbf': 200000, 'tx_mttr': 72
    }

def get(k): return st.session_state['inputs_v10'].get(k)
def set_k(k, v): st.session_state['inputs_v10'][k] = v

with st.sidebar:
    st.title("Inputs v10.0")
    with st.expander("1. Load & AI Step", expanded=True):
        st.number_input("IT Load (MW)", 1.0, 500.0, float(get('p_it')), key='p_it', on_change=lambda: set_k('p_it', st.session_state.p_it))
        st.number_input("Target Avail (%)", 90.0, 99.99999, float(get('req_avail')), format="%.5f", key='req_avail', on_change=lambda: set_k('req_avail', st.session_state.req_avail))
        st.number_input("AI Step Load (%)", 0.0, 100.0, float(get('step_req')), key='step_req', on_change=lambda: set_k('step_req', st.session_state.step_req))
        
        opt = st.selectbox("Voltage", ["Auto-Recommend", "Manual"], index=0, key='volts_mode', on_change=lambda: set_k('volts_mode', st.session_state.volts_mode))
        if opt == 'Manual': st.number_input("kV", 0.4, 69.0, float(get('volts_kv')), key='volts_kv', on_change=lambda: set_k('volts_kv', st.session_state.volts_kv))

    with st.expander("2. Reliability Stats (MTBF/MTTR)"):
        c1, c2 = st.columns(2)
        c1.number_input("Gen MTBF", value=int(get('gen_mtbf')), key='gen_mtbf', on_change=lambda: set_k('gen_mtbf', st.session_state.gen_mtbf))
        c2.number_input("Gen MTTR", value=int(get('gen_mttr')), key='gen_mttr', on_change=lambda: set_k('gen_mttr', st.session_state.gen_mttr))
        c1.number_input("BESS MTBF", value=int(get('bess_mtbf')), key='bess_mtbf', on_change=lambda: set_k('bess_mtbf', st.session_state.bess_mtbf))
        c2.number_input("BESS MTTR", value=int(get('bess_mttr')), key='bess_mttr', on_change=lambda: set_k('bess_mttr', st.session_state.bess_mttr))
        st.caption("Standard IEEE 493 defaults loaded for Bus/Breakers.")

    with st.expander("3. Tech Specs"):
        c1, c2 = st.columns(2)
        c1.number_input("Gen MW", value=float(get('gen_rating')), key='gen_rating', on_change=lambda: set_k('gen_rating', st.session_state.gen_rating))
        c2.number_input("Gen Step Cap %", value=float(get('gen_step_cap')), key='gen_step_cap', on_change=lambda: set_k('gen_step_cap', st.session_state.gen_step_cap))
        st.checkbox("Force BESS", value=get('bess_force'), key='bess_force', on_change=lambda: set_k('bess_force', st.session_state.bess_force))

res = solve_topology_v10(st.session_state['inputs_v10'])

# ==============================================================================
# 4. DASHBOARD & ANSI DIAGRAM
# ==============================================================================

st.title("CAT Topology Designer v10.0")
st.subheader("AI-Ready Infrastructure (Step Load Analysis)")

if res['pass']:
    sol = res['sol']
    
    # KPI
    c1, c2, c3, c4 = st.columns(4)
    c1.markdown(f'<div class="kpi-card"><div class="metric-value">{sol["avail"]*100:.6f}%</div><div class="metric-label">System Avail</div></div>', unsafe_allow_html=True)
    c2.markdown(f'<div class="kpi-card"><div class="metric-value">{sol["load"]:.1f} MW</div><div class="metric-label">Gross Load</div></div>', unsafe_allow_html=True)
    c3.markdown(f'<div class="kpi-card"><div class="metric-value">{sol["n_buses"]}</div><div class="metric-label">Iso-Parallel Buses</div></div>', unsafe_allow_html=True)
    c4.markdown(f'<div class="kpi-card"><div class="metric-value">{sol["bess_mw"]:.1f} MW</div><div class="metric-label">BESS Gap Fill</div></div>', unsafe_allow_html=True)

    st.divider()
    
    # Step Load Report
    st.markdown("### ⚡ AI Step Load Performance")
    col_a, col_b = st.columns([1, 1])
    with col_a:
        st.write(f"**Required Step Load:** {sol['req_step']:.1f} MW")
        st.write(f"**Generator Capability:** {sol['gen_step']:.1f} MW (Immediate pickup)")
    with col_b:
        if sol['bess_active']:
            st.success(f"**BESS Support:** {sol['bess_mw']:.1f} MW provided by {sol['n_bess']} containers.")
            st.caption("BESS is Critical for Availability.")
        else:
            st.info("Generators can handle the AI load step without BESS.")

    # ANSI DIAGRAM
    st.markdown("### 📐 Single Line Diagram (ANSI/IEC Style)")
    
    dot = graphviz.Digraph()
    dot.attr(rankdir='TB', splines='ortho', nodesep='0.6', ranksep='0.6')
    
    # We want a horizontal bus layout.
    # Graphviz hack: Use a subgraph with rank=same for buses
    
    n_buses = sol['n_buses']
    
    # 1. DRAW BUSES (Aligned Horizontally)
    with dot.subgraph(name='cluster_main_bus') as c:
        c.attr(style='invis')
        for i in range(1, n_buses + 1):
            # BUS NODE (Visualized as a thick line using HTML label)
            bus_label = f'''<<TABLE BORDER="0" CELLBORDER="0" CELLSPACING="0">
                            <TR><TD BGCOLOR="black" WIDTH="150" HEIGHT="5"></TD></TR>
                            <TR><TD>Bus {i} ({sol['kv']}kV)</TD></TR>
                            </TABLE>>'''
            c.node(f'Bus_{i}', label=bus_label, shape='none')
    
    # 2. DRAW COMPONENTS FOR EACH BUS
    for i in range(1, n_buses + 1):
        bus_node = f'Bus_{i}'
        
        # Generator Group (Above Bus)
        gen_node = f'G_{i}'
        cb_gen_node = f'CB_G_{i}'
        
        dot.node(gen_node, label='G', shape='circle', width='0.6', fixedsize='true', style='bold')
        dot.node(cb_gen_node, label='', shape='square', width='0.4', fixedsize='true', style='filled', fillcolor='white')
        
        # Edge Gen -> CB -> Bus
        dot.edge(gen_node, cb_gen_node, dir='none')
        dot.edge(cb_gen_node, bus_node, dir='none')
        
        # Feeder Group (Below Bus)
        cb_feed_node = f'CB_F_{i}'
        feed_node = f'F_{i}'
        
        dot.node(cb_feed_node, label='', shape='square', width='0.4', fixedsize='true', style='filled', fillcolor='white')
        dot.node(feed_node, label=f'Feeder {i}', shape='invtriangle', style='filled', fillcolor='black', fontcolor='white')
        
        # Edge Bus -> CB -> Feeder
        dot.edge(bus_node, cb_feed_node, dir='none')
        dot.edge(cb_feed_node, feed_node, dir='none')
        
        # BESS (Side connection if active)
        if sol['bess_active']:
            bess_node = f'BESS_{i}'
            cb_bess = f'CB_B_{i}'
            dot.node(bess_node, label='BESS', shape='box3d')
            dot.node(cb_bess, label='', shape='square', width='0.4', style='filled', fillcolor='white')
            dot.edge(bus_node, cb_bess, dir='none')
            dot.edge(cb_bess, bess_node, dir='none')

    # 3. DRAW TIE BREAKERS (Between Buses)
    # To force horizontal alignment, we link Bus_1 -> Tie -> Bus_2
    # Constraint: rank=same is tricky with ortholines, let's try simple invisible edges for ordering
    
    for i in range(1, n_buses):
        tie_cb = f'Tie_{i}_{i+1}'
        dot.node(tie_cb, label='X', shape='square', width='0.4', fixedsize='true', style='bold')
        
        # Logic connection
        dot.edge(f'Bus_{i}', tie_cb, dir='none', constraint='false') # Don't disturb rank
        dot.edge(tie_cb, f'Bus_{i+1}', dir='none', constraint='false')
        
    # Closing the ring (Last to First) - Drawn as a long feedback line
    tie_last = f'Tie_{n_buses}_1'
    dot.node(tie_last, label='X', shape='square', width='0.4', fixedsize='true', style='bold')
    dot.edge(f'Bus_{n_buses}', tie_last, dir='none', constraint='false')
    dot.edge(tie_last, f'Bus_1', dir='none', constraint='false')

    st.graphviz_chart(dot, use_container_width=True)

else:
    st.error("No valid configuration found. Please check constraints.")
