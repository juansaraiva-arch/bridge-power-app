import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
import numpy as np
import math
import graphviz
from scipy.stats import binom

# --- PAGE CONFIG ---
st.set_page_config(page_title="CAT Topology Designer v8.1 (Diagnostic)", page_icon="⚡", layout="wide")

# --- CSS ---
st.markdown("""
<style>
    @media print {
        [data-testid="stSidebar"], [data-testid="stHeader"], footer, .stButton { display: none !important; }
        .block-container { padding: 0 !important; margin: 0 !important; }
    }
    .warning-box { background-color: #fff3cd; border: 1px solid #ffeeba; padding: 15px; border-radius: 5px; color: #856404; margin-bottom: 10px; }
    .error-box { background-color: #f8d7da; border: 1px solid #f5c6cb; padding: 15px; border-radius: 5px; color: #721c24; margin-bottom: 10px; }
    .success-box { background-color: #d4edda; border: 1px solid #c3e6cb; padding: 15px; border-radius: 5px; color: #155724; margin-bottom: 10px; }
    .kpi-card { background-color: #f8f9fa; padding: 10px; border-radius: 5px; text-align: center; border-left: 5px solid #333; box-shadow: 1px 1px 3px rgba(0,0,0,0.1); }
    .metric-value { font-size: 22px; font-weight: bold; }
    .metric-label { font-size: 13px; color: #666; }
</style>
""", unsafe_allow_html=True)

# ==============================================================================
# 1. RELIABILITY MATH
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

# ==============================================================================
# 2. PHYSICS ENGINE
# ==============================================================================

def calc_short_circuit(voltage_kv, gen_mva, xd_pu, num_gens_parallel):
    if voltage_kv == 0 or xd_pu == 0: return 999999.0, 0.0
    i_base = (gen_mva * 1e6) / (math.sqrt(3) * (voltage_kv * 1000))
    i_sc_unit = i_base / xd_pu
    i_sc_total = i_sc_unit * num_gens_parallel
    return i_sc_total, i_base

# ==============================================================================
# 3. SOLVER WITH DIAGNOSTICS (v8.1)
# ==============================================================================

def solve_topology_v8_1(inputs):
    res = {'pass': False, 'diagnostics': {}}
    
    # --- A. LOAD ---
    p_it = inputs['p_it']
    p_gross_req = (p_it * (1 + inputs['dc_aux']/100.0)) / ((1 - inputs['dist_loss']/100.0) * (1 - inputs['gen_parasitic']/100.0))
    res['load'] = {'gross': p_gross_req}
    
    # --- B. AVAILABILITY PROBS ---
    a_gen = calc_availability_mtbf(inputs['gen_mtbf'], inputs['gen_mttr'])
    a_bus = calc_availability_mtbf(inputs['bus_mtbf'], inputs['bus_mttr'])
    # Distribution Path
    a_cb = calc_availability_mtbf(inputs['cb_mtbf'], inputs['cb_mttr'])
    a_tx = calc_availability_mtbf(inputs['tx_mtbf'], inputs['tx_mttr'])
    a_cable = calc_availability_mtbf(inputs['cable_mtbf'], inputs['cable_mttr'])
    a_dist_path = reliability_series([a_cb, a_tx, a_cable])
    
    # --- C. OPTIMIZATION & DIAGNOSTICS ---
    voltage_kv = inputs['volts_kv']
    if inputs['volts_mode'] == 'Auto-Recommend':
        voltage_kv = 13.8 # Default start
        
    gen_rating_site = inputs['gen_rating'] * (1.0 - (max(0, (inputs['temp']-25)*0.01)))
    gen_mva = gen_rating_site / 0.8
    
    best_solution = None
    
    # Diagnostic Trackers (To report "Best Attempt" if failure)
    diag_max_avail = 0.0
    diag_min_sc = 999999.0
    diag_min_amps = 999999.0
    diag_best_attempt = None
    diag_fail_reasons = set()
    
    min_gens = math.ceil(p_gross_req / gen_rating_site)
    
    # Search Space
    for n_total in range(min_gens, min_gens + 40):
        for n_buses in range(2, 10):
            gens_per_bus = math.ceil(n_total / n_buses)
            real_total = n_buses * gens_per_bus
            
            # 1. Physics Calc
            i_sc_total, i_nom_unit = calc_short_circuit(voltage_kv, gen_mva, inputs['gen_xd'], real_total)
            bus_amp_load = gens_per_bus * i_nom_unit
            
            # Track Diagnostics
            diag_min_sc = min(diag_min_sc, i_sc_total)
            diag_min_amps = min(diag_min_amps, bus_amp_load)
            
            # Physics Constraints
            phy_pass = True
            fail_reason = []
            
            if i_sc_total > 63000: 
                phy_pass = False
                fail_reason.append("Short Circuit > 63kA")
                diag_fail_reasons.add("Short Circuit Limit Exceeded")
            
            if bus_amp_load > 4000: 
                phy_pass = False
                fail_reason.append("Bus Amps > 4000A")
                diag_fail_reasons.add("Bus Ampacity Exceeded")
            
            # 2. Availability Calc (Run it even if physics fail, just to see potential)
            # Gen Subsystem (N-1 Bus Tolerant Logic)
            n_needed = math.ceil(p_gross_req / gen_rating_site)
            
            # RBD: P(Sys) = P(Buses OK) * P(Gens OK) + P(1 Bus Down) * P(Rem Gens OK)
            p_buses_ok = a_bus ** n_buses
            rel_s0 = reliability_k_out_of_n(n_needed, real_total, a_gen)
            
            p_1bus_down = math.comb(n_buses, 1) * (1-a_bus) * (a_bus**(n_buses-1))
            rel_s1 = reliability_k_out_of_n(n_needed, real_total - gens_per_bus, a_gen)
            
            sys_avail = (p_buses_ok * rel_s0) + (p_1bus_down * rel_s1)
            
            # Dist Subsystem
            m_feeders = math.ceil(p_gross_req / 2.5) # 2.5MW blocks
            dist_avail = reliability_k_out_of_n(m_feeders, m_feeders + 2, a_dist_path)
            
            total_rel = sys_avail * dist_avail
            diag_max_avail = max(diag_max_avail, total_rel)
            
            avail_pass = total_rel >= (inputs['req_avail']/100.0)
            if not avail_pass:
                fail_reason.append(f"Avail {total_rel*100:.4f}% < Target")
                diag_fail_reasons.add("Availability Target Missed")

            # Store "Best Attempt" (The one that passed physics or had best avail)
            # Preference: Passing Physics > Highest Avail
            if phy_pass:
                if diag_best_attempt is None or (not diag_best_attempt['phy_pass']) or (total_rel > diag_best_attempt['avail']):
                    diag_best_attempt = {
                        'n_buses': n_buses, 'n_total': real_total, 
                        'bus_ka': i_sc_total/1000.0, 'bus_amps': bus_amp_load, 
                        'avail': total_rel, 'phy_pass': True, 'reasons': fail_reason
                    }
            elif diag_best_attempt is None: # First iteration
                 diag_best_attempt = {
                        'n_buses': n_buses, 'n_total': real_total, 
                        'bus_ka': i_sc_total/1000.0, 'bus_amps': bus_amp_load, 
                        'avail': total_rel, 'phy_pass': False, 'reasons': fail_reason
                    }

            # 3. Check Success
            if phy_pass and avail_pass:
                # N-1 Bus Capacity Check
                rem_mw = (real_total - gens_per_bus) * gen_rating_site
                if rem_mw >= p_gross_req:
                    best_solution = {
                        'n_buses': n_buses, 'n_total': real_total, 'gens_per_bus': gens_per_bus,
                        'bus_ka': i_sc_total/1000.0, 'bus_amps': bus_amp_load, 'avail': total_rel,
                        'voltage': voltage_kv, 'dist_feeders': m_feeders + 2
                    }
                    break
        if best_solution: break
    
    # --- RESULT PROCESSING ---
    if best_solution:
        res['pass'] = True
        res['sol'] = best_solution
    else:
        res['pass'] = False
        # GENERATE INTELLIGENT RECOMMENDATIONS
        recs = []
        
        # Scenario 1: Physics Failure (SC or Amps)
        if "Short Circuit Limit Exceeded" in diag_fail_reasons or "Bus Ampacity Exceeded" in diag_fail_reasons:
            recs.append(f"❌ **Physical Limits Exceeded:** Current config reaches **{diag_min_sc/1000.0:.1f} kA** / **{diag_min_amps:.0f} A**.")
            
            # Calculate hypothetical next voltage
            next_kv = 34.5 if voltage_kv == 13.8 else (13.8 if voltage_kv == 4.16 else 69.0)
            if next_kv != voltage_kv:
                # Quick calc
                _, i_base_next = calc_short_circuit(next_kv, gen_mva, inputs['gen_xd'], 1)
                est_amps = (diag_min_amps * voltage_kv) / next_kv
                est_ka = (diag_min_sc * voltage_kv) / next_kv
                recs.append(f"💡 **Recommendation:** Increase voltage to **{next_kv} kV**. Estimated results: **{est_amps:.0f} A** / **{est_ka/1000.0:.1f} kA** (likely to pass).")
            else:
                recs.append("💡 **Recommendation:** Split into more buses or use High-Impedance Reactors.")

        # Scenario 2: Availability Failure
        if "Availability Target Missed" in diag_fail_reasons:
            recs.append(f"❌ **Availability Gap:** Max achieved **{diag_max_avail*100:.5f}%** vs Target **{inputs['req_avail']:.5f}%**.")
            
            # Identify weak link
            if a_gen < 0.99:
                recs.append("💡 **Weak Link:** Generator MTBF/MTTR is poor. Improve maintenance contracts.")
            if a_bus < 0.999:
                recs.append("💡 **Weak Link:** Bus reliability is low. Check Switchgear quality inputs.")
            if a_dist_path < 0.9995:
                recs.append("💡 **Weak Link:** Distribution path (Trafo/Cable) failure rate is high. Add N+2 or N+3 redundancy on feeders.")
                
        res['diagnostics'] = {
            'recs': recs,
            'best_attempt': diag_best_attempt
        }

    return res

# ==============================================================================
# 2. UI INPUTS (Standard v8 Inputs)
# ==============================================================================

if 'inputs_v8' not in st.session_state:
    st.session_state['inputs_v8'] = {
        'p_it': 100.0, 'dc_aux': 15.0, 'req_avail': 99.999, 'volts_mode': 'Auto-Recommend', 'volts_kv': 13.8,
        'gen_rating': 2.5, 'gen_xd': 0.14, 'temp': 35, 'alt': 100,
        'dist_loss': 1.5, 'gen_parasitic': 3.0,
        # Default MTBF/MTTR
        'gen_mtbf': 1500, 'gen_mttr': 24, 
        'bus_mtbf': 500000, 'bus_mttr': 12,
        'cb_mtbf': 300000, 'cb_mttr': 8,
        'cable_mtbf': 500000, 'cable_mttr': 24,
        'tx_mtbf': 200000, 'tx_mttr': 72
    }

def get(k): return st.session_state['inputs_v8'].get(k)
def set_k(k, v): st.session_state['inputs_v8'][k] = v

with st.sidebar:
    st.title("Inputs v8.1 (Diagnostic)")
    
    with st.expander("1. Load & Voltage", expanded=True):
        st.number_input("IT Load (MW)", 1.0, 500.0, float(get('p_it')), key='p_it', on_change=lambda: set_k('p_it', st.session_state.p_it))
        st.number_input("Target Avail (%)", 90.0, 99.99999, float(get('req_avail')), format="%.5f", key='req_avail', on_change=lambda: set_k('req_avail', st.session_state.req_avail))
        st.selectbox("Voltage", ["Auto-Recommend", "Manual"], index=0, key='volts_mode', on_change=lambda: set_k('volts_mode', st.session_state.volts_mode))
        if get('volts_mode') == 'Manual':
            st.number_input("kV", 0.4, 69.0, float(get('volts_kv')), key='volts_kv', on_change=lambda: set_k('volts_kv', st.session_state.volts_kv))

    with st.expander("2. Reliability Data (MTBF/MTTR)", expanded=True):
        c1, c2 = st.columns(2)
        c1.number_input("Gen MTBF (h)", 100, 100000, int(get('gen_mtbf')), key='gen_mtbf', on_change=lambda: set_k('gen_mtbf', st.session_state.gen_mtbf))
        c2.number_input("Gen MTTR (h)", 1, 1000, int(get('gen_mttr')), key='gen_mttr', on_change=lambda: set_k('gen_mttr', st.session_state.gen_mttr))
        
        c1, c2 = st.columns(2)
        c1.number_input("Bus MTBF (h)", 10000, 1000000, int(get('bus_mtbf')), key='bus_mtbf', on_change=lambda: set_k('bus_mtbf', st.session_state.bus_mtbf))
        c2.number_input("Bus MTTR (h)", 1, 1000, int(get('bus_mttr')), key='bus_mttr', on_change=lambda: set_k('bus_mttr', st.session_state.bus_mttr))
        
        # Simplified Distribution inputs
        c1, c2 = st.columns(2)
        c1.number_input("Trafo MTBF", 10000, 1000000, int(get('tx_mtbf')), key='tx_mtbf', on_change=lambda: set_k('tx_mtbf', st.session_state.tx_mtbf))
        c2.number_input("Trafo MTTR", 1, 1000, int(get('tx_mttr')), key='tx_mttr', on_change=lambda: set_k('tx_mttr', st.session_state.tx_mttr))

    with st.expander("3. Tech Specs"):
        st.number_input("Gen Rating MW", 0.5, 20.0, float(get('gen_rating')), key='gen_rating', on_change=lambda: set_k('gen_rating', st.session_state.gen_rating))
        st.number_input("Xd\" (pu)", 0.05, 0.5, float(get('gen_xd')), key='gen_xd', on_change=lambda: set_k('gen_xd', st.session_state.gen_xd))

res = solve_topology_v8_1(st.session_state['inputs_v8'])

# ==============================================================================
# 3. DASHBOARD
# ==============================================================================

st.title("CAT Topology Designer v8.1")
st.subheader("Diagnostic Engine")

if res['pass']:
    sol = res['sol']
    st.markdown('<div class="success-box">✅ <b>Optimization Successful:</b> Configuration meets all Physics and Availability constraints.</div>', unsafe_allow_html=True)
    
    c1, c2, c3, c4 = st.columns(4)
    with c1: st.markdown(f'<div class="kpi-card"><div class="metric-value">{sol["avail"]*100:.6f}%</div><div class="metric-label">System Avail</div></div>', unsafe_allow_html=True)
    with c2: st.markdown(f'<div class="kpi-card"><div class="metric-value">{sol["n_buses"]}</div><div class="metric-label">Switchgear Buses</div></div>', unsafe_allow_html=True)
    with c3: st.markdown(f'<div class="kpi-card"><div class="metric-value">{sol["bus_ka"]:.1f} kA</div><div class="metric-label">Short Circuit</div></div>', unsafe_allow_html=True)
    with c4: st.markdown(f'<div class="kpi-card"><div class="metric-value">{sol["n_total"]}</div><div class="metric-label">Total Gens</div></div>', unsafe_allow_html=True)
    
    st.divider()
    
    # Diagram logic (Simplified for pass)
    dot = graphviz.Digraph()
    dot.attr(rankdir='LR')
    for i in range(1, sol['n_buses']+1):
        dot.node(f'B{i}', f'Bus {i}\n{sol["voltage"]} kV', shape='rect', style='filled', fillcolor='#FFCD11')
    st.graphviz_chart(dot, use_container_width=True)

else:
    # --- DIAGNOSTIC FAILURE REPORT ---
    diag = res['diagnostics']
    attempt = diag['best_attempt']
    
    st.markdown('<div class="error-box">❌ <b>Optimization Failed:</b> Unable to meet all constraints simultaneously.</div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("### 🔍 Failure Analysis")
        for rec in diag['recs']:
            st.markdown(rec)
            
    with col2:
        if attempt:
            st.markdown("### 📉 Best Failed Attempt Metrics")
            st.write("This configuration came closest but still failed:")
            st.table(pd.DataFrame({
                "Metric": ["Calculated Avail", "Bus Count", "Short Circuit", "Bus Current"],
                "Value": [
                    f"{attempt['avail']*100:.5f}%",
                    f"{attempt['n_buses']}",
                    f"{attempt['bus_ka']:.1f} kA",
                    f"{attempt['bus_amps']:.0f} A"
                ],
                "Status": [
                    "❌ Too Low" if attempt['avail'] < (get('req_avail')/100) else "✅ OK",
                    "-",
                    "❌ Too High (>63)" if attempt['bus_ka'] > 63 else "✅ OK",
                    "❌ Too High (>4000)" if attempt['bus_amps'] > 4000 else "✅ OK"
                ]
            }))
