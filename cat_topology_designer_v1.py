import streamlit as st
import pandas as pd
import numpy as np
import math
import graphviz
from scipy.stats import binom

# --- PAGE CONFIG ---
st.set_page_config(page_title="CAT Topology v13.0 (Expert Diagnostic)", page_icon="⚡", layout="wide")

# --- CSS ---
st.markdown("""
<style>
    @media print {
        [data-testid="stSidebar"], [data-testid="stHeader"], footer, .stButton { display: none !important; }
        .block-container { padding: 0 !important; margin: 0 !important; }
    }
    .fail-card { background-color: #ffe6e6; border-left: 5px solid #ff4d4d; padding: 15px; border-radius: 5px; color: #990000; }
    .rec-card { background-color: #e6f7ff; border-left: 5px solid #1890ff; padding: 15px; border-radius: 5px; color: #0050b3; }
    .success-card { background-color: #f6ffed; border-left: 5px solid #52c41a; padding: 15px; border-radius: 5px; color: #135200; }
    .kpi-card { background-color: #f8f9fa; padding: 10px; border-radius: 5px; text-align: center; border: 1px solid #ddd; }
    .metric-value { font-size: 20px; font-weight: bold; }
    .metric-label { font-size: 12px; color: #666; }
</style>
""", unsafe_allow_html=True)

# ==============================================================================
# 1. ADVANCED RELIABILITY ENGINE
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

def calc_topology_reliability(n_buses, n_needed, n_total, per_bus, a_gen, a_bus, topology_type="Ring"):
    """
    Calculates Gen Subsystem reliability based on Topology Architecture.
    """
    # 1. Standard Ring / Split Bus (N-1 Bus Tolerance)
    # Failure Mode: Lose 1 Bus -> Lose (per_bus) Gens.
    if topology_type == "Ring":
        p_buses_ok = a_bus ** n_buses
        rel_s0 = rel_k_out_n(n_needed, n_total, a_gen)
        
        p_1bus_down = math.comb(n_buses, 1) * (1-a_bus) * (a_bus**(n_buses-1))
        rel_s1 = rel_k_out_n(n_needed, n_total - per_bus, a_gen)
        
        return (p_buses_ok * rel_s0) + (p_1bus_down * rel_s1)

    # 2. Breaker-and-a-Half (BaaH) / Double Bus
    # Failure Mode: Bus failure DOES NOT disconnect generators (they switch to the other bus).
    # We model this as "Perfect Bus" for the generator connection (virtually).
    # The risk shifts entirely to the Generator Breakers themselves.
    elif topology_type == "BaaH":
        # Effective Bus Availability approaches 1.0 because of redundancy
        a_bus_effective = 1.0 - (1.0 - a_bus)**2 
        # For simplicity in this engine, we assume Bus is reliable enough to be ignored, 
        # and calculate pure k-out-of-n of generators.
        return rel_k_out_n(n_needed, n_total, a_gen)

    return 0.0

def calc_sc(kv, mva, xd, n_par):
    if kv == 0 or xd == 0: return 999999.0, 0.0
    i_base = (mva * 1e6) / (math.sqrt(3) * (kv * 1000))
    i_sc = (i_base / xd) * n_par
    return i_sc, i_base

# ==============================================================================
# 2. SOLVER (DIAGNOSTIC MODE)
# ==============================================================================

def solve_topology_v13(inputs):
    res = {'pass': False, 'best_attempt': None, 'recommendations': []}
    
    # Load & Avail Prep
    p_it = inputs['p_it']
    p_gross = (p_it * (1 + inputs['dc_aux']/100)) / ((1 - inputs['dist_loss']/100) * (1 - inputs['gen_parasitic']/100))
    
    a_gen = calc_avail(inputs['gen_mtbf'], inputs['gen_mttr'])
    a_bus = calc_avail(inputs['bus_mtbf'], inputs['bus_mttr'])
    a_bess = calc_avail(inputs['bess_mtbf'], inputs['bess_mttr'])
    a_dist = calc_avail(inputs['cb_mtbf'], inputs['cb_mttr']) * calc_avail(inputs['tx_mtbf'], inputs['tx_mttr'])
    
    gen_site_mw = inputs['gen_rating'] * (1.0 - (max(0, (inputs['temp']-25)*0.01)))
    gen_mva = gen_site_mw / 0.8
    
    # Voltage Logic
    kv_list = [inputs['volts_kv']] if inputs['volts_mode'] == 'Manual' else [13.8, 34.5, 69.0]
    if inputs['volts_mode'] == 'Auto-Recommend' and p_gross < 10: kv_list.insert(0, 4.16)

    best_attempt = None
    
    for kv in kv_list:
        min_gens = math.ceil(p_gross / gen_site_mw)
        
        for n_total in range(min_gens, min_gens + 60):
            for n_buses in range(2, 10):
                per_bus = math.ceil(n_total / n_buses)
                real_total = n_buses * per_bus
                
                # --- 1. PHYSICS ---
                isc, i_nom = calc_sc(kv, gen_mva, inputs['gen_xd'], real_total)
                bus_amps = per_bus * i_nom
                if isc > 63000 or bus_amps > 4000: 
                    # Store as failure if needed
                    continue
                
                # --- 2. BESS & STEP ---
                # (Simplified for v13 logic clarity - assumes BESS is solved or forced)
                step_req_mw = p_it * (inputs['step_req']/100.0)
                gens_avail_step = real_total - per_bus 
                gen_step_mw = gens_avail_step * gen_site_mw * (inputs['gen_step_cap']/100.0)
                bess_needed_mw = max(0, step_req_mw - gen_step_mw)
                
                bess_rel = 1.0
                n_bess = 0
                if bess_needed_mw > 0 or inputs['bess_force']:
                    if inputs['bess_force']: bess_needed_mw = max(bess_needed_mw, inputs['bess_manual_mw'])
                    n_bess = math.ceil(bess_needed_mw / inputs['bess_inv_mw'])
                    bess_rel = rel_k_out_n(n_bess, n_bess + 1, a_bess) # Assume N+1 BESS
                
                # --- 3. RELIABILITY (RING) ---
                n_load = math.ceil(p_gross / gen_site_mw)
                
                # N-1 Bus Check
                surviving_mw = (real_total - per_bus - 1) * gen_site_mw
                tol_pass = surviving_mw >= p_gross
                
                # Calc Probability
                gen_sys_rel = calc_topology_reliability(n_buses, n_load, real_total, per_bus, a_gen, a_bus, "Ring")
                total_avail = gen_sys_rel * bess_rel * a_dist
                
                # Snapshot this attempt
                attempt = {
                    'kv': kv, 'n_buses': n_buses, 'n_total': real_total, 'per_bus': per_bus,
                    'avail': total_avail, 'isc': isc, 'amps': bus_amps, 'tol_pass': tol_pass,
                    'bess_mw': bess_needed_mw, 'n_bess': n_bess, 'load': p_gross
                }
                
                # Check Validity
                if tol_pass and total_avail >= (inputs['req_avail']/100.0):
                    res['pass'] = True
                    res['sol'] = attempt
                    res['sol']['topo'] = "Ring"
                    return res
                
                # Update Best Attempt (The one closest to target)
                if best_attempt is None or total_avail > best_attempt['avail']:
                    best_attempt = attempt

    # IF WE REACH HERE, NO SOLUTION FOUND
    res['pass'] = False
    res['best_attempt'] = best_attempt
    
    # --- DIAGNOSTIC ENGINE ---
    # Why did it fail?
    if best_attempt:
        gap = (inputs['req_avail']/100.0) - best_attempt['avail']
        
        # Sim 1: Try Breaker-and-a-Half (BaaH) Logic
        baah_rel_gen = calc_topology_reliability(best_attempt['n_buses'], 
                                                 math.ceil(p_gross/gen_site_mw), 
                                                 best_attempt['n_total'], 
                                                 best_attempt['per_bus'], 
                                                 a_gen, a_bus, "BaaH")
        # Recalc total
        # Note: BESS and Dist assumed same for comparison
        baah_total_avail = baah_rel_gen * bess_rel * a_dist
        
        if baah_total_avail >= (inputs['req_avail']/100.0):
            res['recommendations'].append({
                'type': 'TOPOLOGY',
                'msg': f"Upgrade Topology to **Breaker-and-a-Half (BaaH)**.",
                'detail': f"Standard Ring Ring reached {best_attempt['avail']*100:.5f}%. BaaH would achieve **{baah_total_avail*100:.5f}%**.",
                'impact': 'HIGH'
            })
        
        # Sim 2: Check Gen MTBF Sensitivity
        # If we double Gen MTBF, does it pass?
        a_gen_imp = calc_avail(inputs['gen_mtbf']*2, inputs['gen_mttr'])
        gen_sys_rel_imp = calc_topology_reliability(best_attempt['n_buses'], math.ceil(p_gross/gen_site_mw), best_attempt['n_total'], best_attempt['per_bus'], a_gen_imp, a_bus, "Ring")
        avail_imp = gen_sys_rel_imp * bess_rel * a_dist
        
        if avail_imp >= (inputs['req_avail']/100.0):
             res['recommendations'].append({
                'type': 'MAINTENANCE',
                'msg': f"Improve Generator Reliability (MTBF).",
                'detail': f"Doubling Gen MTBF to {inputs['gen_mtbf']*2}h would raise avail to **{avail_imp*100:.5f}%**.",
                'impact': 'MEDIUM'
            })
            
        # Sim 3: Check N-1 Tolerance
        if not best_attempt['tol_pass']:
             res['recommendations'].append({
                'type': 'CAPACITY',
                'msg': f"Insufficient Generators for N-1 Bus Fault.",
                'detail': "System collapses if one bus fails. Add more generators per bus.",
                'impact': 'CRITICAL'
            })

    return res

# ==============================================================================
# 3. UI INPUTS
# ==============================================================================

if 'inputs_v13' not in st.session_state:
    st.session_state['inputs_v13'] = {
        'p_it': 100.0, 'dc_aux': 15.0, 'req_avail': 99.999, 'volts_mode': 'Auto-Recommend', 'volts_kv': 13.8,
        'step_req': 40.0, 'gen_rating': 2.5, 'gen_xd': 0.14, 'gen_step_cap': 25.0,
        'dist_loss': 1.5, 'gen_parasitic': 3.0, 'temp': 35, 'alt': 100,
        'bess_force': False, 'bess_manual_mw': 20.0, 'bess_inv_mw': 3.8,
        'gen_mtbf': 2000, 'gen_mttr': 24, 'bess_mtbf': 5000, 'bess_mttr': 48,
        'bus_mtbf': 500000, 'bus_mttr': 12, 'cb_mtbf': 300000, 'cb_mttr': 8,
        'tx_mtbf': 200000, 'tx_mttr': 72
    }

def get(k): return st.session_state['inputs_v13'].get(k)
def set_k(k, v): st.session_state['inputs_v13'][k] = v

with st.sidebar:
    st.title("Inputs v13.0")
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
        st.caption("Standard defaults for Bus/CB/Tx")

    with st.expander("3. Tech Specs"):
        c1, c2 = st.columns(2)
        c1.number_input("Gen MW", value=float(get('gen_rating')), key='gen_rating', on_change=lambda: set_k('gen_rating', st.session_state.gen_rating))
        c2.number_input("Gen Step Cap %", value=float(get('gen_step_cap')), key='gen_step_cap', on_change=lambda: set_k('gen_step_cap', st.session_state.gen_step_cap))
        st.checkbox("Force BESS", value=get('bess_force'), key='bess_force', on_change=lambda: set_k('bess_force', st.session_state.bess_force))

res = solve_topology_v13(st.session_state['inputs_v13'])

# ==============================================================================
# 4. DASHBOARD
# ==============================================================================

st.title("CAT Topology Designer v13.0")
st.caption("Expert System: Diagnostic & Advanced Topologies")

if res['pass']:
    sol = res['sol']
    st.markdown(f'<div class="success-card">✅ <b>PASS:</b> Configuration valid at <b>{sol["kv"]} kV</b> with Iso-Parallel Ring.</div>', unsafe_allow_html=True)
    
    c1, c2, c3, c4 = st.columns(4)
    c1.markdown(f'<div class="kpi-card"><div class="metric-value">{sol["avail"]*100:.6f}%</div><div class="metric-label">Availability</div></div>', unsafe_allow_html=True)
    c2.markdown(f'<div class="kpi-card"><div class="metric-value">{sol["n_buses"]}</div><div class="metric-label">Buses</div></div>', unsafe_allow_html=True)
    c3.markdown(f'<div class="kpi-card"><div class="metric-value">{sol["bess_mw"]:.1f} MW</div><div class="metric-label">BESS Gap</div></div>', unsafe_allow_html=True)
    c4.markdown(f'<div class="kpi-card"><div class="metric-value">{sol["n_total"]}</div><div class="metric-label">Generators</div></div>', unsafe_allow_html=True)
    
    st.divider()
    t1, t2 = st.tabs(["Diagram", "Specs"])
    with t1:
        dot = graphviz.Digraph()
        dot.attr(rankdir='TB', splines='ortho', nodesep='0.6')
        with dot.subgraph(name='cluster_main') as c:
            c.attr(style='invis')
            for i in range(1, sol['n_buses'] + 1):
                c.node(f'Bus_{i}', label=f'Bus {i}\n{sol["amps"]:.0f}A', shape='underline', width='2.5')
                c.node(f'G_{i}', label='G', shape='circle')
                c.edge(f'G_{i}', f'Bus_{i}')
        for i in range(1, sol['n_buses'] + 1):
            nxt = i + 1 if i < sol['n_buses'] else 1
            dot.edge(f'Bus_{i}', f'Bus_{nxt}', label='X', dir='none')
        st.graphviz_chart(dot, use_container_width=True)
        

else:
    # --- EXPERT DIAGNOSTIC MODE ---
    st.markdown('<div class="fail-card">❌ <b>TARGET NOT MET:</b> The Standard Ring Topology cannot achieve the required availability or physical limits with current parameters.</div>', unsafe_allow_html=True)
    
    best = res['best_attempt']
    if best:
        st.markdown(f"### 📉 Status: Gap Analysis")
        
        c1, c2 = st.columns(2)
        with c1:
            target = get('req_avail')
            achieved = best['avail']*100
            gap = target - achieved
            st.metric("Target", f"{target:.6f}%")
            st.metric("Best Achieved (Ring)", f"{achieved:.6f}%", delta=f"-{gap:.6f}%", delta_color="inverse")
            st.write(f"**Best Configuration Found:** {best['n_buses']} Buses @ {best['kv']} kV")
        
        with c2:
            st.markdown("### 💡 Expert Recommendations")
            
            if not res['recommendations']:
                st.info("Try increasing generator redundancy manually or reducing MTTR.")
                
            for rec in res['recommendations']:
                st.markdown(f"""
                <div class="rec-card">
                    <b>{rec['type']}: {rec['msg']}</b><br>
                    {rec['detail']}
                </div>
                <br>
                """, unsafe_allow_html=True)
                
    st.divider()
    st.markdown("### Reference: High Availability Topologies")
    st.image("https://www.electrical-knowhow.com/wp-content/uploads/2012/05/Breaker-and-a-half-Scheme-1.jpg", caption="Breaker-and-a-Half Topology (Suggested for >99.999%)", width=400)
