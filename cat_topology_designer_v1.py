import streamlit as st
import pandas as pd
import numpy as np
import math
import graphviz
from scipy.stats import binom

# --- PAGE CONFIG ---
st.set_page_config(page_title="CAT Topology v15.0 (Pro Edition)", page_icon="⚡", layout="wide")

# --- CSS ---
st.markdown("""
<style>
    @media print {
        [data-testid="stSidebar"], [data-testid="stHeader"], footer, .stButton { display: none !important; }
        .block-container { padding: 0 !important; margin: 0 !important; }
    }
    .upgrade-box { background-color: #e6f7ff; border: 1px solid #91d5ff; padding: 15px; border-radius: 5px; color: #0050b3; margin-bottom: 15px; font-weight: bold; }
    .success-box { background-color: #d4edda; border: 1px solid #c3e6cb; padding: 15px; border-radius: 5px; color: #155724; margin-bottom: 15px; }
    .metric-container { background-color: #f8f9fa; padding: 15px; border-radius: 8px; border: 1px solid #e9ecef; }
    .metric-header { font-size: 16px; font-weight: bold; margin-bottom: 10px; color: #333; border-bottom: 2px solid #FFCD11; padding-bottom: 5px; }
    .sub-metric { font-size: 14px; margin-bottom: 5px; color: #555; }
    .sub-metric b { color: #000; }
</style>
""", unsafe_allow_html=True)

# ==============================================================================
# 1. MATH ENGINE (CORRECTED BAAH LOGIC)
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

def get_gen_states(n_total, n_needed, a_gen):
    # Estimate average number of units in maintenance based on unavailability
    n_maint_avg = round(n_total * (1.0 - a_gen))
    n_op = n_needed
    n_standby = n_total - n_op - n_maint_avg
    # Adjust if math gets weird due to rounding
    if n_standby < 0:
        n_maint_avg += n_standby # Reduce maint count
        n_standby = 0
    return n_op, n_standby, n_maint_avg

# ==============================================================================
# 2. SOLVER (BESS CORRECTION & AUTO-UPGRADE)
# ==============================================================================

def solve_topology_v15(inputs):
    res = {'pass': False, 'log': [], 'diag': {'best_attempt': None}}
    
    # Inputs
    p_it = inputs['p_it']
    p_gross = (p_it * (1 + inputs['dc_aux']/100)) / ((1 - inputs['dist_loss']/100) * (1 - inputs['gen_parasitic']/100))
    
    # Reliability Data
    a_gen = calc_avail(inputs['gen_mtbf'], inputs['gen_mttr'])
    a_bus_single = calc_avail(inputs['bus_mtbf'], inputs['bus_mttr'])
    # For BaaH, bus system is redundant: A_bus_sys = 1 - (1 - A_single)^2
    a_bus_redundant = 1.0 - (1.0 - a_bus_single)**2
    
    a_bess = calc_avail(inputs['bess_mtbf'], inputs['bess_mttr'])
    # Dist path assumed series of CB -> Tx
    a_dist_path = calc_avail(inputs['cb_mtbf'], inputs['cb_mttr']) * calc_avail(inputs['tx_mtbf'], inputs['tx_mttr'])

    # Gen Specs
    gen_site_mw = inputs['gen_rating'] * (1.0 - (max(0, (inputs['temp']-25)*0.01)))
    gen_mva = gen_site_mw / 0.8
    n_load = math.ceil(p_gross / gen_site_mw)

    # --- BESS SIZING (CORRECTED: FULL STEP COVERAGE) ---
    # BESS must cover the full AI step requirement instantly.
    bess_power_mw = p_it * (inputs['step_req'] / 100.0)
    # Energy: Assume 60 seconds (1 minute) support duration for stabilization
    bess_duration_h = 1.0 / 60.0 
    bess_energy_mwh = bess_power_mw * bess_duration_h

    n_bess = math.ceil(bess_power_mw / inputs['bess_inv_mw']) if bess_power_mw > 0 else 0
    # BESS Availability (N+1 redundancy assumed for high reliability)
    bess_rel_sys = rel_k_out_n(n_bess, n_bess + 1, a_bess) if n_bess > 0 else 1.0

    # Voltage Strategy (Hyperscale Logic)
    if inputs['volts_mode'] == 'Manual': kv_list = [inputs['volts_kv']]
    elif p_gross > 200: kv_list = [69.0]
    elif p_gross > 40: kv_list = [34.5, 69.0]
    else: kv_list = [13.8, 34.5]

    final_sol = None
    best_failed = None

    for kv in kv_list:
        min_gens = n_load + 2 # Start with N+2 minimum
        
        for n_total in range(min_gens, min_gens + 60):
            # Try typical bay configurations for large plants
            for n_bays in [4, 5, 6, 8, 10]:
                gens_per_bay = math.ceil(n_total / n_bays)
                real_total = n_bays * gens_per_bay
                
                # 1. Physics Check (SC & Amps)
                # Amps per main bus section in BaaH is roughly half total current under normal operation, 
                # but must handle full current of connected bays during contingency.
                # Simplified check: Check total current against main bus rating (e.g. 4000A or 5000A)
                isc, i_nom_total = calc_sc(kv, gen_mva, inputs['gen_xd'], real_total)
                total_amps = real_total * (gen_mva*1e6 / (math.sqrt(3)*kv*1000))
                
                # Stricter limits for 99.999%
                if isc > 63000 or total_amps > 5000: continue 

                # 2. Reliability Calculation (BaaH Logic)
                # In BaaH, a single bus fault does NOT drop generation. 
                # Reliability is dominated by Generators and Breakers.
                # We assume the redundant bus system availability is high enough (~1.0).
                
                # Gen System Reliability (k-out-of-n)
                gen_sys_rel = rel_k_out_n(n_load, real_total, a_gen)
                
                # Distribution System Reliability (N+2 feeders)
                n_feeders = math.ceil(p_gross / 2.5) + 2
                dist_sys_rel = rel_k_out_n(n_feeders-2, n_feeders, a_dist_path)

                # TOTAL SYSTEM AVAILABILITY
                total_avail = gen_sys_rel * bess_rel_sys * dist_sys_rel
                
                candidate = {
                    'kv': kv, 'n_bays': n_bays, 'n_total': real_total, 'per_bay': gens_per_bay,
                    'avail': total_avail, 'isc': isc, 'amps': total_amps, 'topo': 'BaaH',
                    'bess_mw': bess_power_mw, 'bess_mwh': bess_energy_mwh, 'n_bess': n_bess
                }

                if total_avail >= (inputs['req_avail']/100.0):
                    final_sol = candidate
                    break
                
                if not best_failed or total_avail > best_failed['avail']: best_failed = candidate

            if final_sol: break
        if final_sol: break

    res['pass'] = (final_sol is not None)
    res['sol'] = final_sol
    res['diag']['best_attempt'] = best_failed
    
    if final_sol:
        # Calculate Gen States
        nop, nstb, nmnt = get_gen_states(final_sol['n_total'], n_load, a_gen)
        res['sol']['states'] = (nop, nstb, nmnt)
        res['sol']['load_mw'] = p_gross

    return res

# ==============================================================================
# 3. UI INPUTS
# ==============================================================================

if 'inputs_v15' not in st.session_state:
    st.session_state['inputs_v15'] = {
        'p_it': 100.0, 'dc_aux': 15.0, 'req_avail': 99.999, 'volts_mode': 'Auto-Recommend', 'volts_kv': 34.5,
        'step_req': 40.0, 'gen_rating': 2.5, 'gen_xd': 0.14, 'bess_inv_mw': 3.8,
        'dist_loss': 1.5, 'gen_parasitic': 3.0, 'temp': 35, 'alt': 100,
        # High Reliability MTBF/MTTR Defaults
        'gen_mtbf': 3000, 'gen_mttr': 24, 'bess_mtbf': 8000, 'bess_mttr': 24,
        'bus_mtbf': 1000000, 'bus_mttr': 8, 'cb_mtbf': 500000, 'cb_mttr': 4,
        'tx_mtbf': 300000, 'tx_mttr': 48
    }

def get(k): return st.session_state['inputs_v15'].get(k)
def set_k(k, v): st.session_state['inputs_v15'][k] = v

with st.sidebar:
    st.title("Inputs v15.0")
    st.caption("Pro Edition: BaaH & Full BESS Sizing")
    with st.expander("1. Load & Strategy", expanded=True):
        st.number_input("IT Load (MW)", 10.0, 500.0, float(get('p_it')), key='p_it', on_change=lambda: set_k('p_it', st.session_state.p_it))
        st.number_input("Target Avail (%)", 99.0, 99.99999, float(get('req_avail')), format="%.5f", key='req_avail', on_change=lambda: set_k('req_avail', st.session_state.req_avail))
        st.number_input("AI Step Load (%)", 0.0, 100.0, float(get('step_req')), key='step_req', on_change=lambda: set_k('step_req', st.session_state.step_req))
        opt = st.selectbox("Voltage", ["Auto-Recommend", "Manual"], index=0, key='volts_mode', on_change=lambda: set_k('volts_mode', st.session_state.volts_mode))
        if opt == 'Manual': st.number_input("Manual kV", 0.4, 69.0, float(get('volts_kv')), key='volts_kv', on_change=lambda: set_k('volts_kv', st.session_state.volts_kv))

    with st.expander("2. Tech Specs"):
        c1, c2 = st.columns(2)
        c1.number_input("Gen MW", value=float(get('gen_rating')), key='gen_rating', on_change=lambda: set_k('gen_rating', st.session_state.gen_rating))
        c2.number_input("BESS Inv MW", value=float(get('bess_inv_mw')), key='bess_inv_mw', on_change=lambda: set_k('bess_inv_mw', st.session_state.bess_inv_mw))

    with st.expander("3. Reliability Data (High Spec)"):
        c1, c2 = st.columns(2)
        c1.number_input("Gen MTBF", value=int(get('gen_mtbf')), key='gen_mtbf', on_change=lambda: set_k('gen_mtbf', st.session_state.gen_mtbf))
        c2.number_input("Gen MTTR", value=int(get('gen_mttr')), key='gen_mttr', on_change=lambda: set_k('gen_mttr', st.session_state.gen_mttr))
        c1.number_input("BESS MTBF", value=int(get('bess_mtbf')), key='bess_mtbf', on_change=lambda: set_k('bess_mtbf', st.session_state.bess_mtbf))
        c2.number_input("BESS MTTR", value=int(get('bess_mttr')), key='bess_mttr', on_change=lambda: set_k('bess_mttr', st.session_state.bess_mttr))

res = solve_topology_v15(st.session_state['inputs_v15'])

# ==============================================================================
# 4. DASHBOARD
# ==============================================================================

st.title("CAT Topology Designer v15.0")
st.subheader("AI-Ready Hyperscale Architecture")

if res['pass']:
    sol = res['sol']
    nop, nstb, nmnt = sol['states']

    # NOTIFICATIONS
    st.markdown(f"""
    <div class="upgrade-box">
        ⚡ <b>High Availability Config Selected:</b> Breaker-and-a-Half (BaaH) topology applied to meet {get('req_avail')}% target.
    </div>
    <div class="success-box">
        ✅ <b>System Validated:</b> Solution found at <b>{sol['kv']} kV</b> meeting all criteria.
    </div>
    """, unsafe_allow_html=True)

    # DETAILED METRICS GRID
    c1, c2, c3 = st.columns(3)
    
    with c1:
        st.markdown("""
        <div class="metric-container">
            <div class="metric-header">🔌 Power & Voltage</div>
            <div class="sub-metric">Recommended Voltage: <b>{:.1f} kV</b></div>
            <div class="sub-metric">Gross Load: <b>{:.1f} MW</b></div>
            <div class="sub-metric">Short Circuit: <b>{:.1f} kA</b></div>
        </div>
        """.format(sol['kv'], sol['load_mw'], sol['isc']/1000), unsafe_allow_html=True)

    with c2:
        st.markdown("""
        <div class="metric-container">
            <div class="metric-header">🔋 BESS (AI Step Support)</div>
            <div class="sub-metric">Step Load Requirement: <b>{:.1f} MW</b> ({:.0f}%)</div>
            <div class="sub-metric">BESS Power: <b>{:.1f} MW</b> (Full Coverage)</div>
            <div class="sub-metric">BESS Energy (60s): <b>{:.2f} MWh</b></div>
        </div>
        """.format(sol['bess_mw'], get('step_req'), sol['bess_mw'], sol['bess_mwh']), unsafe_allow_html=True)
        
    with c3:
        st.markdown("""
        <div class="metric-container">
            <div class="metric-header">⚙️ Generation Fleet</div>
            <div class="sub-metric">Total Units: <b>{}</b> x {:.1f} MW</div>
            <div class="sub-metric">🟢 Operating: <b>{}</b></div>
            <div class="sub-metric">🟡 Standby: <b>{}</b> | 🔴 Maint (Avg): <b>{}</b></div>
        </div>
        """.format(sol['n_total'], get('gen_rating'), nop, nstb, nmnt), unsafe_allow_html=True)

    st.divider()
    
    # ANSI-STYLE BAAH DIAGRAM
    st.markdown("### 📐 Single Line Diagram: Breaker-and-a-Half (ANSI Style)")
    
    dot = graphviz.Digraph()
    # Use orthogonal lines and top-to-bottom flow for ANSI look
    dot.attr(rankdir='TB', splines='ortho', nodesep='1.0', ranksep='0.8')
    
    # MAIN BUSES (Thick horizontal lines at top and bottom)
    dot.node('BusA', '', shape='rect', style='filled', fillcolor='black', height='0.1', width='12', fixedsize='true')
    dot.node('BusB', '', shape='rect', style='filled', fillcolor='black', height='0.1', width='12', fixedsize='true')
    
    # Draw Bays
    for i in range(1, sol['n_bays'] + 1):
        # Use subgraphs to group breakers vertically
        with dot.subgraph(name=f'cluster_bay_{i}') as bay:
            bay.attr(style='invis')
            
            # Breakers (Squares)
            cb_top = f'CB_{i}_T'
            cb_mid = f'CB_{i}_M'
            cb_bot = f'CB_{i}_B'
            for cb in [cb_top, cb_mid, cb_bot]:
                bay.node(cb, label='', shape='square', width='0.5', fixedsize='true', style='bold')
            
            # Junction points for circuits
            j_top = f'J_{i}_T'
            j_bot = f'J_{i}_B'
            bay.node(j_top, shape='point', width='0.1')
            bay.node(j_bot, shape='point', width='0.1')
            
            # Connections Main Buses -> Breakers
            dot.edge('BusA', cb_top, dir='none', penwidth='2')
            dot.edge(cb_bot, 'BusB', dir='none', penwidth='2')
            
            # Breaker Stack Connections
            bay.edge(cb_top, j_top, dir='none')
            bay.edge(j_top, cb_mid, dir='none')
            bay.edge(cb_mid, j_bot, dir='none')
            bay.edge(j_bot, cb_bot, dir='none')
            
            # Circuits (Gens & Feeders)
            gen_node = f'G_{i}'
            feed_node = f'F_{i}'
            
            bay.node(gen_node, label=f'Gens {i}\n({sol["per_bay"]}x)', shape='circle', width='1.0')
            bay.node(feed_node, label=f'Feeders {i}', shape='invtriangle', width='1.0')
            
            # Connect Circuits to Junctions
            bay.edge(j_top, gen_node, dir='none', label='Circuit 1')
            bay.edge(j_bot, feed_node, dir='none', label='Circuit 2')

    st.graphviz_chart(dot, use_container_width=True)

else:
    st.error("❌ Optimization Failed. Even Breaker-and-a-Half topology could not meet constraints with current inputs.")
    if res['diag']['best_attempt']:
        st.write(f"Best Availability Achieved: {res['diag']['best_attempt']['avail']*100:.6f}%")
