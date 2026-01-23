import streamlit as st
import pandas as pd
import numpy as np
import math
import graphviz
from scipy.stats import binom

# --- PAGE CONFIG ---
st.set_page_config(page_title="CAT Topology v17.0 (Corrected Logic)", page_icon="⚡", layout="wide")

# --- CSS ---
st.markdown("""
<style>
    @media print {
        [data-testid="stSidebar"], [data-testid="stHeader"], footer, .stButton { display: none !important; }
        .block-container { padding: 0 !important; margin: 0 !important; }
    }
    .metric-container {
        background-color: #ffffff; 
        border: 1px solid #e0e0e0; 
        padding: 15px; 
        border-radius: 8px; 
        border-left: 5px solid #FFCD11;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
    .metric-header { font-size: 14px; font-weight: bold; color: #555; margin-bottom: 5px; text-transform: uppercase; }
    .metric-value { font-size: 24px; font-weight: bold; color: #000; }
    .metric-sub { font-size: 13px; color: #666; margin-top: 5px; }
    
    .success-box { background-color: #f6ffed; border: 1px solid #b7eb8f; padding: 15px; border-radius: 5px; color: #135200; }
    .warning-box { background-color: #fffbe6; border: 1px solid #ffe58f; padding: 15px; border-radius: 5px; color: #876800; }
</style>
""", unsafe_allow_html=True)

# ==============================================================================
# 1. MATH ENGINE (CORRECTED RELIABILITY)
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
    # Calculation of Operation, Standby, Maintenance
    # Maint is statistical average based on unavailability
    n_maint = int(round(n_total * (1.0 - a_gen)))
    n_op = n_needed
    n_standby = n_total - n_op - n_maint
    if n_standby < 0:
        # If reliability is low, we might be dipping into 'needed' reserves or under maintenance
        n_standby = 0
    return n_op, n_standby, n_maint

# ==============================================================================
# 2. SOLVER (BAAH + HIGH CAPACITY BESS)
# ==============================================================================

def solve_topology_v17(inputs):
    res = {'pass': False, 'sol': None}
    
    # 1. LOAD & SPECS
    p_it = inputs['p_it']
    p_gross = (p_it * (1 + inputs['dc_aux']/100)) / ((1 - inputs['dist_loss']/100) * (1 - inputs['gen_parasitic']/100))
    
    gen_site_mw = inputs['gen_rating'] * (1.0 - (max(0, (inputs['temp']-25)*0.01)))
    gen_mva = gen_site_mw / 0.8
    
    # 2. VOLTAGE (Hyperscale Logic)
    if inputs['volts_mode'] == 'Manual': kv_list = [inputs['volts_kv']]
    elif p_gross > 200: kv_list = [69.0]
    elif p_gross > 40: kv_list = [34.5, 69.0]
    else: kv_list = [13.8, 34.5]

    # 3. RELIABILITY DATA
    a_gen = calc_avail(inputs['gen_mtbf'], inputs['gen_mttr'])
    # Breaker-and-a-Half Logic:
    # Bus Availability is NOT a bottleneck. We treat it as 1.0 (Double Bus Redundancy).
    # The limit is the Breaker/Generator chain.
    a_cb = calc_avail(inputs['cb_mtbf'], inputs['cb_mttr'])
    a_chain = a_gen * a_cb # Gen available AND its breaker closed

    # 4. BESS SIZING (The "User Rule": Match Generation if Step is high)
    # If Step Load > 20%, we size BESS aggressively to match N+1 Generation Capacity.
    # This provides full buffering for AI transients.
    step_req_mw = p_it * (inputs['step_req']/100.0)
    
    # Calculate Needed Gens first to size BESS relative to Fleet
    n_needed_load = math.ceil(p_gross / gen_site_mw)
    gen_fleet_capacity = (n_needed_load + 1) * gen_site_mw # N+1 Baseline
    
    if inputs['step_req'] >= 30.0:
        # User Rule: For high step, BESS ~ Gen Capacity
        bess_target_mw = gen_fleet_capacity
        bess_sizing_mode = "Matched to Gen Capacity (High Inertia)"
    else:
        # Standard: Cover the step
        bess_target_mw = max(step_req_mw, 5.0) # Min 5MW
        bess_sizing_mode = "Matched to Step Requirement"

    n_bess = math.ceil(bess_target_mw / inputs['bess_inv_mw'])
    bess_inst_mw = n_bess * inputs['bess_inv_mw']
    
    # BESS Reliability (Parallel Redundancy)
    a_bess = calc_avail(inputs['bess_mtbf'], inputs['bess_mttr'])
    # We assume N+1 redundancy inside the BESS block
    rel_bess = rel_k_out_n(n_bess - 1, n_bess, a_bess)

    # 5. OPTIMIZATION LOOP
    final_sol = None
    
    for kv in kv_list:
        # Start with sufficient redundancy (N+2 minimum for Tier IV compliance)
        min_total = n_needed_load + 2
        
        for n_total in range(min_total, min_total + 40):
            # Try BaaH Bay Configurations
            # Each Bay supports 2 Circuits. 
            # Ideally: 1 Gen + 1 Feeder per bay.
            # Total Bays needed = Max(N_Gens, N_Feeders)
            
            n_feeders = math.ceil(p_gross / 2.5) # 2.5MW blocks
            n_circuits = n_total + n_feeders
            n_bays = math.ceil(n_circuits / 2)
            
            # Physics Check
            isc, i_nom = calc_sc(kv, gen_mva, inputs['gen_xd'], n_total)
            # Main Bus Amps (Total Load flow)
            total_amps = (p_gross * 1e6) / (math.sqrt(3) * kv * 1000)
            
            if isc > 63000: continue
            if total_amps > 5000: continue # 5000A bus is standard for heavy industry
            
            # Reliability Calculation (BaaH)
            # Bus faults ignored (Prob ~ 0).
            # Calc Gen Fleet Reliability
            rel_gen_sys = rel_k_out_n(n_needed_load, n_total, a_chain)
            
            # Total Avail
            # We assume Feeder/Dist reliability is handled by downstream dual feeds
            total_avail = rel_gen_sys * rel_bess
            
            if total_avail >= (inputs['req_avail']/100.0):
                final_sol = {
                    'kv': kv, 'n_bays': n_bays, 'n_total': n_total,
                    'avail': total_avail, 'isc': isc, 'amps': total_amps,
                    'bess_mw': bess_inst_mw, 'bess_mode': bess_sizing_mode, 'n_bess': n_bess,
                    'load': p_gross
                }
                break
        if final_sol: break
            
    res['pass'] = (final_sol is not None)
    res['sol'] = final_sol
    
    if final_sol:
        res['sol']['states'] = get_gen_states(final_sol['n_total'], n_needed_load, a_gen)

    return res

# ==============================================================================
# 3. UI INPUTS
# ==============================================================================

if 'inputs_v17' not in st.session_state:
    st.session_state['inputs_v17'] = {
        'p_it': 100.0, 'dc_aux': 15.0, 'req_avail': 99.999, 'volts_mode': 'Auto-Recommend', 'volts_kv': 34.5,
        'step_req': 40.0, 'gen_rating': 2.5, 'gen_xd': 0.14, 'bess_inv_mw': 3.8,
        'dist_loss': 1.5, 'gen_parasitic': 3.0, 'temp': 35, 'alt': 100,
        'gen_mtbf': 3000, 'gen_mttr': 24, 'bess_mtbf': 8000, 'bess_mttr': 24,
        'cb_mtbf': 500000, 'cb_mttr': 4 # High reliability breakers
    }

def get(k): return st.session_state['inputs_v17'].get(k)
def set_k(k, v): st.session_state['inputs_v17'][k] = v

with st.sidebar:
    st.title("Inputs v17.0")
    st.caption("Mode: Corrected BaaH & Heavy BESS")
    
    with st.expander("1. Project Data", expanded=True):
        st.number_input("IT Load (MW)", 10.0, 500.0, float(get('p_it')), key='p_it', on_change=lambda: set_k('p_it', st.session_state.p_it))
        st.number_input("Target Avail (%)", 99.0, 99.99999, float(get('req_avail')), format="%.5f", key='req_avail', on_change=lambda: set_k('req_avail', st.session_state.req_avail))
        st.number_input("AI Step Load (%)", 0.0, 100.0, float(get('step_req')), key='step_req', on_change=lambda: set_k('step_req', st.session_state.step_req))
        opt = st.selectbox("Voltage", ["Auto-Recommend", "Manual"], index=0, key='volts_mode', on_change=lambda: set_k('volts_mode', st.session_state.volts_mode))
        if opt == 'Manual': st.number_input("kV", 0.4, 69.0, float(get('volts_kv')), key='volts_kv', on_change=lambda: set_k('volts_kv', st.session_state.volts_kv))

    with st.expander("2. Specs"):
        c1, c2 = st.columns(2)
        c1.number_input("Gen MW", value=float(get('gen_rating')), key='gen_rating', on_change=lambda: set_k('gen_rating', st.session_state.gen_rating))
        c2.number_input("BESS Inv MW", value=float(get('bess_inv_mw')), key='bess_inv_mw', on_change=lambda: set_k('bess_inv_mw', st.session_state.bess_inv_mw))

res = solve_topology_v17(st.session_state['inputs_v17'])

# ==============================================================================
# 4. DASHBOARD
# ==============================================================================

st.title("CAT Topology Designer v17.0")
st.subheader("High Availability Substation (ANSI/IEC Standard)")

if res['pass']:
    sol = res['sol']
    nop, nstb, nmnt = sol['states']
    
    # 1. NOTIFICATIONS
    st.markdown(f"""
    <div class="success-box">
        <b>✅ Solution Validated:</b> Breaker-and-a-Half Topology at <b>{sol['kv']} kV</b>. 
        Meets {sol['avail']*100:.6f}% availability.
    </div>
    """, unsafe_allow_html=True)

    # 2. METRICS GRID
    c1, c2, c3, c4 = st.columns(4)
    
    with c1:
        st.markdown(f"""
        <div class="metric-container">
            <div class="metric-header">VOLTAGE LEVEL</div>
            <div class="metric-value">{sol['kv']} kV</div>
            <div class="metric-sub">Auto-Selected for {sol['load']:.0f} MW</div>
        </div>""", unsafe_allow_html=True)

    with c2:
        st.markdown(f"""
        <div class="metric-container">
            <div class="metric-header">BESS CAPACITY</div>
            <div class="metric-value">{sol['bess_mw']:.1f} MW</div>
            <div class="metric-sub">{sol['bess_mode']}</div>
        </div>""", unsafe_allow_html=True)

    with c3:
        st.markdown(f"""
        <div class="metric-container">
            <div class="metric-header">GEN FLEET</div>
            <div class="metric-value">{sol['n_total']} Units</div>
            <div class="metric-sub">Config: N+{sol['n_total'] - nop}</div>
        </div>""", unsafe_allow_html=True)

    with c4:
        st.markdown(f"""
        <div class="metric-container">
            <div class="metric-header">FAULT LEVELS</div>
            <div class="metric-value">{sol['isc']/1000:.1f} kA</div>
            <div class="metric-sub">Bus Amps: {sol['amps']:.0f} A</div>
        </div>""", unsafe_allow_html=True)

    st.divider()
    
    c_diag, c_data = st.columns([2, 1])
    
    with c_diag:
        st.markdown("### 📐 Substation Diagram (ANSI Breaker-and-a-Half)")
        # ANSI BAAH VISUALIZATION
        dot = graphviz.Digraph()
        dot.attr(rankdir='TB', splines='ortho', nodesep='0.8', ranksep='0.5')
        
        # MAIN BUSES (Parallel Horizontal Bars)
        dot.node('BusA', 'MAIN BUS A', shape='rect', style='filled', fillcolor='black', fontcolor='white', height='0.3', width='10', fixedsize='true')
        dot.node('BusB', 'MAIN BUS B', shape='rect', style='filled', fillcolor='black', fontcolor='white', height='0.3', width='10', fixedsize='true')
        
        # Draw 3 Representative Bays
        # BAY 1
        with dot.subgraph(name='cluster_bay1') as b1:
            b1.attr(label='Bay 1 (Typical)', style='dashed')
            # 3 Breakers vertical
            b1.node('CB1_1', '', shape='square', style='filled', fillcolor='white', width='0.5', fixedsize='true')
            b1.node('CB1_2', '', shape='square', style='filled', fillcolor='white', width='0.5', fixedsize='true')
            b1.node('CB1_3', '', shape='square', style='filled', fillcolor='white', width='0.5', fixedsize='true')
            # Connections
            b1.edge('BusA', 'CB1_1', dir='none', penwidth='2')
            b1.edge('CB1_1', 'CB1_2', dir='none')
            b1.edge('CB1_2', 'CB1_3', dir='none')
            b1.edge('CB1_3', 'BusB', dir='none', penwidth='2')
            # Circuits
            b1.node('G1', f'Gen Group 1', shape='circle')
            b1.edge('CB1_1', 'G1', label='Tap 1')
            b1.node('F1', 'Feeders 1', shape='invtriangle')
            b1.edge('CB1_2', 'F1', label='Tap 2')

        # BAY 2 (BESS)
        with dot.subgraph(name='cluster_bay2') as b2:
            b2.attr(label='Bay 2 (BESS/Gen)', style='dashed')
            b2.node('CB2_1', '', shape='square', style='filled', fillcolor='white', width='0.5', fixedsize='true')
            b2.node('CB2_2', '', shape='square', style='filled', fillcolor='white', width='0.5', fixedsize='true')
            b2.node('CB2_3', '', shape='square', style='filled', fillcolor='white', width='0.5', fixedsize='true')
            
            b2.edge('BusA', 'CB2_1', dir='none', penwidth='2')
            b2.edge('CB2_1', 'CB2_2', dir='none')
            b2.edge('CB2_2', 'CB2_3', dir='none')
            b2.edge('CB2_3', 'BusB', dir='none', penwidth='2')
            
            b2.node('BESS1', 'BESS Array', shape='box3d')
            b2.edge('CB2_1', 'BESS1')
            b2.node('G2', 'Gen Group 2', shape='circle')
            b2.edge('CB2_2', 'G2')

        st.graphviz_chart(dot, use_container_width=True)

    with c_data:
        st.markdown("### 📋 Generation State Breakdown")
        st.table(pd.DataFrame({
            "Status": ["Running (Load)", "Standby (Reserve)", "Maintenance (Avg)"],
            "Count": [nop, nstb, nmnt],
            "Capacity (MW)": [f"{nop*get('gen_rating'):.1f}", f"{nstb*get('gen_rating'):.1f}", f"{nmnt*get('gen_rating'):.1f}"]
        }))
        
        st.markdown("### ℹ️ Reliability Note")
        st.info("""
        **Why Breaker-and-a-Half?**
        - Fault on Bus A -> System runs on Bus B.
        - Fault on Bus B -> System runs on Bus A.
        - **Zero Generation Loss** for single bus faults.
        - Allows breaker maintenance without circuit outage.
        """)

else:
    st.error("No valid solution found. Try increasing BESS parameters or Gen Rating.")
