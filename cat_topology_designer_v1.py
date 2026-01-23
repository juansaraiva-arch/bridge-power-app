import streamlit as st
import pandas as pd
import numpy as np
import math
import graphviz
from scipy.stats import binom

# --- PAGE CONFIG ---
st.set_page_config(page_title="CAT Topology v16.0 (High Precision)", page_icon="⚡", layout="wide")

# --- CSS ---
st.markdown("""
<style>
    @media print {
        [data-testid="stSidebar"], [data-testid="stHeader"], footer, .stButton { display: none !important; }
        .block-container { padding: 0 !important; margin: 0 !important; }
    }
    .metric-box {
        background-color: #f0f2f6; border-left: 5px solid #000; padding: 15px; border-radius: 5px; margin-bottom: 10px;
    }
    .metric-title { font-size: 14px; color: #555; font-weight: bold; }
    .metric-value { font-size: 24px; font-weight: bold; color: #000; }
    .metric-sub { font-size: 12px; color: #666; }
    
    .success-box { background-color: #d4edda; border: 1px solid #c3e6cb; padding: 15px; border-radius: 5px; color: #155724; }
    .error-box { background-color: #f8d7da; border: 1px solid #f5c6cb; padding: 15px; border-radius: 5px; color: #721c24; }
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
    # Cumulative Binomial Probability (P(X >= k))
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
# 2. SOLVER (FULL STEP COVERAGE & BAAH LOGIC)
# ==============================================================================

def solve_topology_v16(inputs):
    res = {'pass': False, 'log': []}
    
    # 1. LOAD ANALYSIS
    p_it = inputs['p_it']
    # Gross Load (Gen Terminals)
    p_gross = (p_it * (1 + inputs['dc_aux']/100)) / ((1 - inputs['dist_loss']/100) * (1 - inputs['gen_parasitic']/100))
    
    # 2. BESS SIZING (Conservative: Full Step Coverage)
    # The BESS must handle the step load entirely to protect the gens.
    step_mw_req = p_it * (inputs['step_req'] / 100.0)
    
    # BESS Power & Units
    n_bess = math.ceil(step_mw_req / inputs['bess_inv_mw']) if step_mw_req > 0 else 0
    bess_installed_mw = n_bess * inputs['bess_inv_mw']
    
    # BESS Energy (Duration Support)
    # Assuming 5 minutes (typical for bridge to gens ramp up)
    duration_min = 5.0 
    bess_energy_mwh = bess_installed_mw * (duration_min / 60.0)

    # 3. GENERATION & TOPOLOGY
    a_gen = calc_avail(inputs['gen_mtbf'], inputs['gen_mttr'])
    # In BaaH, Bus Avail is effectively 1.0 for the system (redundant paths).
    # Reliability is limited by the Gen+Breaker chain.
    a_breaker = calc_avail(inputs['cb_mtbf'], inputs['cb_mttr'])
    a_gen_chain = a_gen * a_breaker # Probability that Gen AND its Breaker are OK
    
    gen_site_mw = inputs['gen_rating'] * (1.0 - (max(0, (inputs['temp']-25)*0.01)))
    gen_mva = gen_site_mw / 0.8
    
    # Voltage Selection (Hyperscale Hard Logic)
    if inputs['volts_mode'] == 'Manual': kv_list = [inputs['volts_kv']]
    elif p_gross > 200: kv_list = [69.0, 115.0]
    elif p_gross > 40: kv_list = [34.5, 69.0]
    else: kv_list = [13.8, 34.5]

    final_sol = None
    
    for kv in kv_list:
        n_needed_load = math.ceil(p_gross / gen_site_mw)
        min_total = n_needed_load + 2 # Minimum N+2 redundancy
        
        for n_total in range(min_total, min_total + 40):
            # Try Bay Configurations (Even numbers preferred for BaaH symmetry)
            for n_bays in [4, 6, 8, 10, 12]:
                per_bay = math.ceil(n_total / n_bays)
                real_total = n_bays * per_bay
                
                # A. Physics (Short Circuit)
                isc, i_nom = calc_sc(kv, gen_mva, inputs['gen_xd'], real_total)
                # Bus Amps: In BaaH, current splits, but check worst case (one bus down)
                bus_amps = real_total * i_nom
                
                if isc > 63000: continue
                # Allow higher amps for BaaH buses (often rated 4000-5000A)
                if bus_amps > 5000: continue 
                
                # B. Reliability Calculation (BaaH)
                # We need 'n_needed_load' generators to be available.
                # In BaaH, loss of a bus does NOT lose gens. 
                # So we calculate simple k-out-of-n for the fleet.
                
                gen_rel = rel_k_out_n(n_needed_load, real_total, a_gen_chain)
                
                # BESS Reliability (Assume N+1 internal redundancy for the block)
                bess_rel = 0.9999 # Simplified for high-tier BESS
                
                # Dist Reliability
                dist_rel = 0.99995 # Simplified High Tier
                
                total_avail = gen_rel * bess_rel * dist_rel
                
                if total_avail >= (inputs['req_avail']/100.0):
                    # STATES CALCULATION
                    n_maint = round(real_total * (1 - a_gen))
                    n_op = n_needed_load
                    n_standby = real_total - n_op - n_maint
                    if n_standby < 0: n_standby = 0 # Correction
                    
                    final_sol = {
                        'kv': kv, 'n_bays': n_bays, 'n_total': real_total, 'per_bay': per_bay,
                        'avail': total_avail, 'isc': isc, 'amps': bus_amps,
                        'bess_mw': step_mw_req, 'bess_inst_mw': bess_installed_mw, 
                        'bess_mwh': bess_energy_mwh, 'n_bess': n_bess,
                        'states': (n_op, n_standby, n_maint),
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

if 'inputs_v16' not in st.session_state:
    st.session_state['inputs_v16'] = {
        'p_it': 100.0, 'dc_aux': 15.0, 'req_avail': 99.999, 'volts_mode': 'Auto-Recommend', 'volts_kv': 34.5,
        'step_req': 40.0, 'gen_rating': 2.5, 'gen_xd': 0.14, 'bess_inv_mw': 3.8,
        'dist_loss': 1.5, 'gen_parasitic': 3.0, 'temp': 35, 'alt': 100,
        # IEEE 493 High Reliability
        'gen_mtbf': 3000, 'gen_mttr': 24, 'bess_mtbf': 8000, 'bess_mttr': 24,
        'bus_mtbf': 2000000, 'bus_mttr': 12, 'cb_mtbf': 500000, 'cb_mttr': 6
    }

def get(k): return st.session_state['inputs_v16'].get(k)
def set_k(k, v): st.session_state['inputs_v16'][k] = v

with st.sidebar:
    st.title("Inputs v16.0")
    with st.expander("1. Project Profile", expanded=True):
        st.number_input("IT Load (MW)", 10.0, 500.0, float(get('p_it')), key='p_it', on_change=lambda: set_k('p_it', st.session_state.p_it))
        st.number_input("Target Avail (%)", 99.0, 99.99999, float(get('req_avail')), format="%.5f", key='req_avail', on_change=lambda: set_k('req_avail', st.session_state.req_avail))
        st.number_input("AI Step Load (%)", 0.0, 100.0, float(get('step_req')), key='step_req', on_change=lambda: set_k('step_req', st.session_state.step_req))
        opt = st.selectbox("Voltage", ["Auto-Recommend", "Manual"], index=0, key='volts_mode', on_change=lambda: set_k('volts_mode', st.session_state.volts_mode))
        if opt == 'Manual': st.number_input("kV", 0.4, 230.0, float(get('volts_kv')), key='volts_kv', on_change=lambda: set_k('volts_kv', st.session_state.volts_kv))

    with st.expander("2. Tech Specs"):
        c1, c2 = st.columns(2)
        c1.number_input("Gen MW", value=float(get('gen_rating')), key='gen_rating', on_change=lambda: set_k('gen_rating', st.session_state.gen_rating))
        c2.number_input("BESS Inv MW", value=float(get('bess_inv_mw')), key='bess_inv_mw', on_change=lambda: set_k('bess_inv_mw', st.session_state.bess_inv_mw))
        
    with st.expander("3. Reliability Data"):
        c1, c2 = st.columns(2)
        c1.number_input("Gen MTBF", value=int(get('gen_mtbf')), key='gen_mtbf', on_change=lambda: set_k('gen_mtbf', st.session_state.gen_mtbf))
        c2.number_input("Gen MTTR", value=int(get('gen_mttr')), key='gen_mttr', on_change=lambda: set_k('gen_mttr', st.session_state.gen_mttr))

res = solve_topology_v16(st.session_state['inputs_v16'])

# ==============================================================================
# 4. DASHBOARD
# ==============================================================================

st.title("CAT Topology Designer v16.0")
st.caption("Architecture: Breaker-and-a-Half (Double Bus) | AI Step Support: Full Coverage")

if res['pass']:
    sol = res['sol']
    nop, nstb, nmnt = sol['states']
    
    # 1. SUMMARY METRICS
    st.markdown("### 📊 Executive Summary")
    c1, c2, c3, c4 = st.columns(4)
    
    with c1:
        st.markdown(f"""
        <div class="metric-box">
            <div class="metric-title">SYSTEM AVAILABILITY</div>
            <div class="metric-value">{sol['avail']*100:.6f}%</div>
            <div class="metric-sub">Target: {get('req_avail')}%</div>
        </div>""", unsafe_allow_html=True)
        
    with c2:
        st.markdown(f"""
        <div class="metric-box">
            <div class="metric-title">RECOMMENDED VOLTAGE</div>
            <div class="metric-value">{sol['kv']} kV</div>
            <div class="metric-sub">Short Circuit: {sol['isc']/1000:.1f} kA</div>
        </div>""", unsafe_allow_html=True)
        
    with c3:
        st.markdown(f"""
        <div class="metric-box">
            <div class="metric-title">GENERATION FLEET</div>
            <div class="metric-value">{sol['n_total']} Units</div>
            <div class="metric-sub">Architecture: {sol['n_bays']} Bays</div>
        </div>""", unsafe_allow_html=True)
        
    with c4:
        st.markdown(f"""
        <div class="metric-box">
            <div class="metric-title">BESS CAPACITY</div>
            <div class="metric-value">{sol['bess_inst_mw']:.1f} MW</div>
            <div class="metric-sub">Step Req: {sol['bess_mw']:.1f} MW</div>
        </div>""", unsafe_allow_html=True)

    # 2. DETAILED BREAKDOWN
    st.divider()
    c1, c2 = st.columns([1, 1])
    
    with c1:
        st.markdown("### 🔋 BESS & Generation Detail")
        st.write("#### Generator Status Logic")
        st.table(pd.DataFrame({
            "State": ["🟢 Operating (Load)", "🟡 Standby (Redundancy)", "🔴 Maintenance (Avg)"],
            "Units": [nop, nstb, nmnt],
            "Capacity (MW)": [f"{nop*get('gen_rating'):.1f}", f"{nstb*get('gen_rating'):.1f}", f"{nmnt*get('gen_rating'):.1f}"]
        }))
        
        st.write("#### BESS Sizing (Conservative AI Support)")
        st.write(f"- **Step Load Requirement:** {sol['bess_mw']:.1f} MW ({get('step_req')}%)")
        st.write(f"- **Installed Power:** {sol['bess_inst_mw']:.1f} MW ({sol['n_bess']} x {get('bess_inv_mw')} MW)")
        st.write(f"- **Energy Storage:** {sol['bess_mwh']:.2f} MWh (5 min support)")
    
    with c2:
        st.markdown("### 📐 Topology Diagram (ANSI BaaH)")
        
        # ANSI DIAGRAM LOGIC
        dot = graphviz.Digraph()
        dot.attr(rankdir='TB', splines='ortho', nodesep='0.5', ranksep='0.5')
        
        # Main Buses (Horizontal Bars)
        dot.node('BusA', '', shape='rect', style='filled', fillcolor='black', height='0.05', width='8', fixedsize='true')
        dot.node('BusB', '', shape='rect', style='filled', fillcolor='black', height='0.05', width='8', fixedsize='true')
        
        # Draw Bays (Vertical)
        # We'll draw 2 representative bays to keep diagram clean, with "..." if more
        
        display_bays = min(sol['n_bays'], 3) # Show max 3 bays visually
        
        for i in range(1, display_bays + 1):
            with dot.subgraph(name=f'bay_{i}') as bay:
                # Nodes
                cb_top = f'CB_{i}_1'
                cb_mid = f'CB_{i}_2'
                cb_bot = f'CB_{i}_3'
                gen = f'G_{i}'
                load = f'L_{i}'
                
                bay.node(cb_top, '', shape='square', width='0.4', style='bold')
                bay.node(cb_mid, '', shape='square', width='0.4', style='bold')
                bay.node(cb_bot, '', shape='square', width='0.4', style='bold')
                
                bay.node(gen, f'{sol["per_bay"]}x Gens', shape='circle')
                bay.node(load, 'Feeder', shape='invtriangle')
                
                # Connectivity
                dot.edge('BusA', cb_top, dir='none')
                dot.edge(cb_top, cb_mid, dir='none')
                dot.edge(cb_mid, cb_bot, dir='none')
                dot.edge(cb_bot, 'BusB', dir='none')
                
                # Taps
                # Tap between Top/Mid -> Gen
                j1 = f'j1_{i}'
                dot.node(j1, '', shape='point', width='0')
                dot.edge(cb_top, j1, dir='none', len='0.2')
                dot.edge(j1, cb_mid, dir='none', len='0.2')
                dot.edge(j1, gen, dir='none')
                
                # Tap between Mid/Bot -> Load
                j2 = f'j2_{i}'
                dot.node(j2, '', shape='point', width='0')
                dot.edge(cb_mid, j2, dir='none', len='0.2')
                dot.edge(j2, cb_bot, dir='none', len='0.2')
                dot.edge(j2, load, dir='none')

        st.graphviz_chart(dot, use_container_width=True)
        st.caption(f"Showing {display_bays} of {sol['n_bays']} Bays. Configuration: Breaker-and-a-Half.")

else:
    st.markdown('<div class="error-box">❌ <b>Analysis Failed:</b> Constraints too tight.</div>', unsafe_allow_html=True)
