import streamlit as st
import pandas as pd
import numpy as np
import math
import graphviz
from scipy.stats import binom

# --- PAGE CONFIG ---
st.set_page_config(page_title="CAT Topology v10.1 (Diagnostic + AI)", page_icon="⚡", layout="wide")

# --- CSS ---
st.markdown("""
<style>
    @media print {
        [data-testid="stSidebar"], [data-testid="stHeader"], footer, .stButton { display: none !important; }
        .block-container { padding: 0 !important; margin: 0 !important; }
    }
    .error-box { background-color: #f8d7da; border: 1px solid #f5c6cb; padding: 15px; border-radius: 5px; color: #721c24; margin-bottom: 10px; }
    .warning-box { background-color: #fff3cd; border: 1px solid #ffeeba; padding: 15px; border-radius: 5px; color: #856404; margin-bottom: 10px; }
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
# 2. SOLVER WITH DIAGNOSTICS
# ==============================================================================

def solve_topology_v10_1(inputs):
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
    
    # Voltage Strategy
    kv_list = [inputs['volts_kv']] if inputs['volts_mode'] == 'Manual' else [13.8, 34.5, 69.0]
    if inputs['volts_mode'] == 'Auto-Recommend' and p_gross < 10: kv_list.insert(0, 4.16)

    final_sol = None
    best_failed_attempt = None
    
    # Track reasons across all attempts
    fail_reasons_global = set()

    for kv in kv_list:
        min_gens = math.ceil(p_gross / gen_site_mw)
        
        # Loop Configs
        for n_total in range(min_gens, min_gens + 50):
            for n_buses in range(2, 12):
                per_bus = math.ceil(n_total / n_buses)
                real_total = n_buses * per_bus
                
                current_fail_reasons = []
                
                # --- CHECK 1: PHYSICS ---
                isc, i_nom = calc_sc(kv, gen_mva, inputs['gen_xd'], real_total)
                bus_amps = per_bus * i_nom
                
                phy_pass = True
                if isc > 63000: 
                    phy_pass = False; current_fail_reasons.append(f"SC > 63kA ({isc/1000:.1f}kA)")
                if bus_amps > 4000: 
                    phy_pass = False; current_fail_reasons.append(f"Amps > 4000A ({bus_amps:.0f}A)")
                
                # --- CHECK 2: FAULT TOLERANCE ---
                surviving_mw = (real_total - per_bus - 1) * gen_site_mw
                tol_pass = surviving_mw >= p_gross
                if not tol_pass: current_fail_reasons.append("N-1 Tolerance Fail")
                
                # --- CHECK 3: BESS & STEP LOAD ---
                step_req_mw = p_it * (inputs['step_req']/100.0)
                gens_avail_step = real_total - per_bus # Worst case step
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
                
                # --- CHECK 4: RELIABILITY ---
                n_load = math.ceil(p_gross / gen_site_mw)
                p_bus_ok = a_bus ** n_buses
                r_s0 = rel_k_out_n(n_load, real_total, a_gen)
                p_bus_fail = math.comb(n_buses, 1) * (1-a_bus) * (a_bus**(n_buses-1))
                r_s1 = rel_k_out_n(n_load, real_total - per_bus, a_gen)
                
                gen_sys_rel = (p_bus_ok * r_s0) + (p_bus_fail * r_s1)
                total_avail = gen_sys_rel * bess_rel * a_dist
                
                avail_pass = total_avail >= (inputs['req_avail']/100.0)
                if not avail_pass: current_fail_reasons.append(f"Avail Low ({total_avail*100:.5f}%)")

                # --- RESULT EVALUATION ---
                attempt_data = {
                    'n_buses': n_buses, 'n_total': real_total, 'kv': kv,
                    'avail': total_avail, 'isc': isc, 'amps': bus_amps,
                    'reasons': current_fail_reasons, 'valid': False
                }

                if phy_pass and tol_pass and avail_pass:
                    # SUCCESS
                    attempt_data['valid'] = True
                    attempt_data.update({
                        'per_bus': per_bus, 'bess_active': (n_bess_units > 0),
                        'n_bess': n_bess_units, 'bess_mw': bess_needed_mw,
                        'gen_step': gen_step_mw, 'req_step': step_req_mw, 'load': p_gross
                    })
                    final_sol = attempt_data
                    break
                
                # Update Diagnostics (Best Failed Attempt)
                # Logic: Prefer passing Physics > Passing Tol > Highest Avail
                if not final_sol:
                    update_best = False
                    if best_failed_attempt is None: update_best = True
                    else:
                        # Score: Phy(100) + Tol(50) + Avail(0-1)
                        curr_score = (100 if phy_pass else 0) + (50 if tol_pass else 0) + total_avail
                        best_score = (100 if not any("kA" in r or "Amps" in r for r in best_failed_attempt['reasons']) else 0) + \
                                     (50 if not any("Tolerance" in r for r in best_failed_attempt['reasons']) else 0) + \
                                     best_failed_attempt['avail']
                        if curr_score > best_score: update_best = True
                    
                    if update_best:
                        best_failed_attempt = attempt_data
                        
                    for r in current_fail_reasons: fail_reasons_global.add(r)

            if final_sol: break
        if final_sol: break
        
    res['pass'] = (final_sol is not None)
    res['sol'] = final_sol
    res['diag']['reasons'] = list(fail_reasons_global)
    res['diag']['best_attempt'] = best_failed_attempt
    return res

# ==============================================================================
# 3. UI INPUTS
# ==============================================================================

if 'inputs_v10_1' not in st.session_state:
    st.session_state['inputs_v10_1'] = {
        'p_it': 100.0, 'dc_aux': 15.0, 'req_avail': 99.999, 'volts_mode': 'Auto-Recommend', 'volts_kv': 13.8,
        'step_req': 40.0, 'gen_rating': 2.5, 'gen_xd': 0.14, 'gen_step_cap': 25.0,
        'dist_loss': 1.5, 'gen_parasitic': 3.0, 'temp': 35, 'alt': 100,
        'bess_force': False, 'bess_manual_mw': 20.0, 'bess_inv_mw': 3.8,
        # MTBF/MTTR (Defaults IEEE)
        'gen_mtbf': 2000, 'gen_mttr': 24, 'bess_mtbf': 5000, 'bess_mttr': 48,
        'bus_mtbf': 500000, 'bus_mttr': 12, 'cb_mtbf': 300000, 'cb_mttr': 8,
        'tx_mtbf': 200000, 'tx_mttr': 72
    }

def get(k): return st.session_state['inputs_v10_1'].get(k)
def set_k(k, v): st.session_state['inputs_v10_1'][k] = v

with st.sidebar:
    st.title("Inputs v10.1")
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
        st.caption("Defaults loaded for Bus/BESS/Tx/CB.")

    with st.expander("3. Tech Specs"):
        c1, c2 = st.columns(2)
        c1.number_input("Gen MW", value=float(get('gen_rating')), key='gen_rating', on_change=lambda: set_k('gen_rating', st.session_state.gen_rating))
        c2.number_input("Gen Step Cap %", value=float(get('gen_step_cap')), key='gen_step_cap', on_change=lambda: set_k('gen_step_cap', st.session_state.gen_step_cap))
        st.checkbox("Force BESS", value=get('bess_force'), key='bess_force', on_change=lambda: set_k('bess_force', st.session_state.bess_force))

res = solve_topology_v10_1(st.session_state['inputs_v10_1'])

# ==============================================================================
# 4. DASHBOARD & DIAGNOSTICS
# ==============================================================================

st.title("CAT Topology Designer v10.1")
st.subheader("AI-Ready Infrastructure & Diagnostics")

if res['pass']:
    sol = res['sol']
    
    # KPI
    c1, c2, c3, c4 = st.columns(4)
    c1.markdown(f'<div class="kpi-card"><div class="metric-value">{sol["avail"]*100:.6f}%</div><div class="metric-label">System Avail</div></div>', unsafe_allow_html=True)
    c2.markdown(f'<div class="kpi-card"><div class="metric-value">{sol["load"]:.1f} MW</div><div class="metric-label">Gross Load</div></div>', unsafe_allow_html=True)
    c3.markdown(f'<div class="kpi-card"><div class="metric-value">{sol["n_buses"]}</div><div class="metric-label">Buses @ {sol["kv"]}kV</div></div>', unsafe_allow_html=True)
    c4.markdown(f'<div class="kpi-card"><div class="metric-value">{sol["bess_mw"]:.1f} MW</div><div class="metric-label">BESS Gap Fill</div></div>', unsafe_allow_html=True)

    st.divider()
    
    # ANSI DIAGRAM
    dot = graphviz.Digraph()
    dot.attr(rankdir='TB', splines='ortho', nodesep='0.6')
    
    n_buses = sol['n_buses']
    # Drawing logic similar to v10 but concise
    with dot.subgraph(name='cluster_main') as c:
        c.attr(style='invis')
        for i in range(1, n_buses + 1):
            c.node(f'Bus_{i}', label=f'Bus {i}\n{sol["amps"]:.0f}A', shape='underline', width='2.5')
            c.node(f'G_{i}', label='G', shape='circle')
            c.edge(f'G_{i}', f'Bus_{i}')
            if sol['bess_active']:
                c.node(f'B_{i}', label='BESS', shape='box3d')
                c.edge(f'Bus_{i}', f'B_{i}')
            
            c.node(f'F_{i}', label=f'Feeder {i}', shape='invtriangle')
            c.edge(f'Bus_{i}', f'F_{i}')
            
    # Ring Ties
    for i in range(1, n_buses + 1):
        nxt = i + 1 if i < n_buses else 1
        dot.edge(f'Bus_{i}', f'Bus_{nxt}', label='X', dir='none')

    st.graphviz_chart(dot, use_container_width=True)

else:
    # --- FAILURE ANALYSIS ---
    st.markdown('<div class="error-box">❌ <b>No Solution Found:</b> All configurations failed constraints.</div>', unsafe_allow_html=True)
    
    diag = res['diag']
    best = diag['best_attempt']
    reasons = diag['reasons']
    
    c1, c2 = st.columns(2)
    
    with c1:
        st.markdown("### 🔍 Why did it fail?")
        # Parse Reasons for Recommendations
        if any("SC >" in r for r in reasons):
            st.error(f"**Short Circuit Too High:** Found values > 63kA. \n\n*Recommendation:* Increase Voltage (try 34.5kV or 69kV).")
        if any("Amps >" in r for r in reasons):
            st.error(f"**Bus Current Too High:** Found values > 4000A. \n\n*Recommendation:* Increase Voltage or Split into more Buses.")
        if any("Tolerance" in r for r in reasons):
            st.warning("**N-1 Bus Tolerance Failed:** Generators too small relative to load.\n\n*Recommendation:* Use larger Generator Rating.")
        if any("Avail Low" in r for r in reasons):
            st.warning("**Availability Target Missed:** Even physically valid options didn't reach 99.999%.\n\n*Recommendation:* Improve MTTR (Response time) or add BESS redundancy.")

    with c2:
        if best:
            st.markdown("### 📉 Best Failed Candidate")
            st.write("This was the closest configuration:")
            st.code(f"""
            Voltage: {best['kv']} kV
            Buses:   {best['n_buses']}
            Total Gens: {best['n_total']}
            
            Short Circuit: {best['isc']/1000:.1f} kA
            Availability:  {best['avail']*100:.5f} %
            
            FAILURE REASONS:
            {', '.join(best['reasons'])}
            """)
