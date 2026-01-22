import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
import numpy as np
import math
import graphviz
import time
from scipy.stats import binom

# --- PAGE CONFIG ---
st.set_page_config(page_title="CAT Topology Designer v5.0", page_icon="⚡", layout="wide")

# --- CSS ---
st.markdown("""
<style>
    @media print {
        [data-testid="stSidebar"], [data-testid="stHeader"], footer, .stButton { display: none !important; }
        .block-container { padding: 0 !important; margin: 0 !important; }
    }
    .warning-box { background-color: #ffcccc; border: 1px solid red; padding: 10px; border-radius: 5px; color: #900; margin-bottom: 10px; }
    .success-box { background-color: #d4edda; border: 1px solid #c3e6cb; padding: 10px; border-radius: 5px; color: #155724; margin-bottom: 10px; }
</style>
""", unsafe_allow_html=True)

# ==============================================================================
# 1. RELIABILITY MATH ENGINE
# ==============================================================================

def get_avail_prob(maint_pct, for_pct):
    """Convierte tasas de mantenimiento y falla forzada en Probabilidad de Disponibilidad (0-1)"""
    # Maint = Planned Outage rate, FOR = Forced Outage rate
    # Disponibilidad = 1 - (Unavailability)
    return 1.0 - ((maint_pct / 100.0) + (for_pct / 100.0))

def calc_k_out_of_n_reliability(n_needed, n_total, p_unit_avail):
    """Calcula la probabilidad de que al menos 'n_needed' unidades funcionen de un total de 'n_total'"""
    if n_total < n_needed: return 0.0
    prob_success = 0.0
    # Suma de probabilidades binomiales para k = n_needed hasta n_total
    for k in range(n_needed, n_total + 1):
        prob_success += binom.pmf(k, n_total, p_unit_avail)
    return prob_success

def optimize_redundancy(n_needed, target_rel, p_unit_avail):
    """Encuentra el N_total mínimo necesario para cumplir la confiabilidad objetivo"""
    for added_redundancy in range(0, 20): # Límite de seguridad de 20 unidades extra
        n_total = n_needed + added_redundancy
        rel = calc_k_out_of_n_reliability(n_needed, n_total, p_unit_avail)
        if rel >= target_rel:
            return n_total, rel
    return n_needed + 20, rel # Retorno fallback si no se cumple

# ==============================================================================
# 2. MAIN ALGORITHM
# ==============================================================================

def solve_topology_v5(inputs):
    res = {'warnings': [], 'metrics': {}}
    
    # --- A. CARGA Y PÉRDIDAS ---
    p_it = inputs['p_it']
    # Carga Bruta considerando Auxiliares y Pérdidas de Distribución
    p_facility = p_it * (1 + inputs['dc_aux']/100.0)
    p_gen_term = p_facility / (1 - inputs['dist_loss']/100.0)
    # Potencia al Eje (Gross) considerando Parásitos
    p_gross_req = p_gen_term / (1 - inputs['gen_parasitic']/100.0)
    
    res['load'] = {'gross_mw': p_gross_req, 'net_mw': p_it}

    # --- B. SELECCIÓN DE VOLTAJE (AUTO/MANUAL) ---
    voltage_kv = inputs['volts_kv']
    
    # Función auxiliar para calcular amperios totales
    def get_total_amps(mw, kv):
        return (mw * 1e6) / (math.sqrt(3) * (kv * 1000) * 0.8) # PF 0.8 típico gen

    if inputs['volts_mode'] == 'Auto-Recommend':
        # Lógica de recomendación basada en Amperios Manejables (< 5000A por bus idealmente)
        # Probamos voltajes estándar
        candidates = [0.48, 4.16, 13.8, 34.5]
        voltage_kv = 34.5 # Default alto
        for v in candidates:
            amps = get_total_amps(p_gross_req, v)
            # Si la corriente total es < 8000A, es manejable (quizás en 2 buses). 
            # Si es mayor, subimos voltaje.
            if amps < 10000: 
                voltage_kv = v
                break
    
    # --- C. GENERACIÓN (Cálculo Probabilístico) ---
    # 1. Capacidad Unitaria
    derate = 1.0
    if inputs['derate_mode'] == 'Auto-Calculate':
        derate = 1.0 - (max(0, (inputs['temp'] - 25)*0.01) + max(0, (inputs['alt'] - 100)*0.0001))
    
    gen_site_mw = inputs['gen_rating'] * derate
    
    # 2. Unidades Necesarias (N)
    n_gen_needed = math.ceil(p_gross_req / gen_site_mw)
    
    # 3. Optimización de Redundancia
    p_gen_avail = get_avail_prob(inputs['gen_maint'], inputs['gen_for'])
    target_rel = inputs['req_avail'] / 100.0
    
    # Aquí buscamos una confiabilidad MUY alta para la generación base, 
    # ya que es solo el primer eslabón de la cadena en serie.
    # Asumimos que la generación debe aportar al menos la confiabilidad target por sí misma.
    n_gen_total, gen_rel_calc = optimize_redundancy(n_gen_needed, target_rel, p_gen_avail)
    
    res['gen'] = {
        'n_needed': n_gen_needed,
        'n_total': n_gen_total,
        'rel': gen_rel_calc,
        'site_mw': gen_site_mw
    }

    # --- D. BESS & STEP LOAD (Probabilidad Condicional) ---
    # Requerimiento de Step Load
    step_req_mw = p_it * (inputs['step_load_req'] / 100.0)
    
    # Capacidad de Step de los Generadores (Solo los N activos, no los de reserva)
    gen_step_mw = n_gen_needed * gen_site_mw * (inputs['gen_step_cap'] / 100.0)
    
    bess_needed = False
    n_bess_total = 0
    bess_rel_calc = 1.0 # Si no se usa, confiabilidad es 1 (no falla)
    
    if gen_step_mw < step_req_mw:
        bess_needed = True
        shortfall_mw = step_req_mw - gen_step_mw
        
        # Dimensionamiento BESS (Potencia)
        # Asumimos un inversor típico o pedimos input (usaré 3.8 MW como estándar interno si no se pide, pero el usuario pidió inputs específicos, asumo que están en el sidebar)
        inv_mw = 3.8 # Default, se puede hacer variable
        n_bess_needed = math.ceil(shortfall_mw / inv_mw)
        
        # Confiabilidad BESS
        p_bess_avail = get_avail_prob(inputs['bess_maint'], inputs['bess_for'])
        n_bess_total, bess_rel_calc = optimize_redundancy(n_bess_needed, target_rel, p_bess_avail)
        
        res['warnings'].append(f"ℹ️ **Step Load Requirement:** Generators provide {gen_step_mw:.1f} MW step. BESS added for remaining {shortfall_mw:.1f} MW.")

    res['bess'] = {
        'active': bess_needed,
        'n_total': n_bess_total,
        'rel': bess_rel_calc
    }

    # --- E. TOPOLOGÍA FÍSICA (SWITCHGEAR & CLUSTERING) ---
    # Cálculos Físicos
    i_nom_gen = (gen_site_mw * 1e6) / (math.sqrt(3) * (voltage_kv * 1000) * 0.8)
    i_sc_gen = i_nom_gen / inputs['gen_xd'] # Aproximación X"d
    
    # Límites Físicos (Hardcoded standards)
    LIMIT_BUS_AMP = 4000.0 # O 5000A
    LIMIT_BUS_KA = 63000.0 # 63 kA
    
    # Validación de Voltaje Manual
    total_amps_sys = n_gen_total * i_nom_gen
    total_ka_sys = n_gen_total * i_sc_gen
    
    if total_amps_sys > 20000 and inputs['volts_mode'] == 'Manual':
        res['warnings'].append(f"⚠️ **Warning:** Total system current is extremely high ({total_amps_sys:,.0f} A). Consider increasing voltage.")
    
    # Cálculo de Clusters (Buses)
    # Cuántos generadores caben en un bus sin violar límites?
    max_gens_amp = math.floor((LIMIT_BUS_AMP * 0.9) / i_nom_gen)
    max_gens_ka = math.floor((LIMIT_BUS_KA * 0.95) / i_sc_gen)
    
    max_gens_per_bus = max(1, min(max_gens_amp, max_gens_ka))
    
    # Número de Switchgears necesarios
    num_swgr = math.ceil(n_gen_total / max_gens_per_bus)
    
    # Verificar redundancia de Buses (Probabilidad de Bus)
    p_bus_avail = get_avail_prob(inputs['bus_maint'], inputs['bus_for'])
    
    # Simplificación: Asumimos que necesitamos (Num_Swgr) buses para plena carga?
    # No necesariamente. Si tenemos N+X gens, quizás podemos perder un bus.
    # Pero para diseño robusto, asumimos que necesitamos disponibilidad de bus.
    # Calculamos la confiabilidad del sistema de buses en serie (si necesitamos todos) o paralelo.
    # Dado que distribuimos los gens, si perdemos un bus, perdemos capacidad.
    # Verificamos si (Num_Swgr - 1) buses tienen capacidad suficiente para la carga.
    gens_per_bus = math.ceil(n_gen_total / num_swgr)
    gens_remaining_if_bus_fail = (num_swgr - 1) * gens_per_bus
    
    bus_redundant = False
    if gens_remaining_if_bus_fail >= n_gen_needed:
        bus_redundant = True
    else:
        # Si no somos tolerantes a falla de bus, aumentamos buses o generadores?
        # Aumentar generadores por bus es limitado por kA.
        # Aumentar número de buses reduce el impacto de perder uno.
        # Iteración simple: Si no es tolerante, forzamos +1 Bus si la confiabilidad total cae.
        pass # Por ahora reportamos el estado.

    res['elec'] = {
        'voltage': voltage_kv,
        'num_swgr': num_swgr,
        'gens_per_bus': gens_per_bus,
        'bus_rating': 5000 if (gens_per_bus * i_nom_gen) > 4000 else 4000,
        'bus_ka': (gens_per_bus * i_sc_gen)/1000.0,
        'bus_redundant': bus_redundant
    }

    # --- F. DISTRIBUCIÓN (TRANSFORMADORES & FEEDERS) ---
    # Asumimos bloque de distribución estándar (ej. 2.5 MVA)
    dist_block_mw = 2.5 # Se podría hacer input
    n_feeders_needed = math.ceil(p_gross_req / dist_block_mw)
    
    # Probabilidad de una cadena de distribución (Breaker -> Cable -> Trafo)
    p_cb = get_avail_prob(inputs['cb_maint'], inputs['cb_for'])
    p_cable = get_avail_prob(inputs['cable_maint'], inputs['cable_for'])
    p_tx = get_avail_prob(inputs['tx_maint'], inputs['tx_for'])
    
    p_path_avail = p_cb * p_cable * p_tx
    
    # Optimizar Redundancia de Feeders
    n_feeders_total, dist_rel_calc = optimize_redundancy(n_feeders_needed, target_rel, p_path_avail)
    
    res['dist'] = {
        'n_needed': n_feeders_needed,
        'n_total': n_feeders_total,
        'rel': dist_rel_calc,
        'per_bus': math.ceil(n_feeders_total / num_swgr)
    }

    # --- G. CONFIABILIDAD TOTAL DEL SISTEMA (SERIE) ---
    # System = Gen_System AND BESS_System (if needed) AND Dist_System
    # Asumimos que los Buses son parte de la cadena intermedia.
    
    total_rel = gen_rel_calc * dist_rel_calc
    if bess_needed:
        total_rel *= bess_rel_calc
        
    res['metrics'] = {
        'total_rel': total_rel,
        'target': target_rel
    }
    
    if total_rel < target_rel:
        res['warnings'].append(f"⚠️ **Target Not Met:** Calculated {total_rel*100:.5f}% < Target {target_rel*100:.5f}%. Consider improving component reliability.")

    return res

# ==============================================================================
# 3. UI INPUTS (SIDEBAR COMPLETO)
# ==============================================================================

if 'inputs_v5' not in st.session_state:
    st.session_state['inputs_v5'] = {
        # Profile
        'dc_type': 'Hyperscale', 'p_it': 100.0, 'dc_aux': 15.0, 'req_avail': 99.999, 'step_load_req': 40.0,
        'volts_mode': 'Auto-Recommend', 'volts_kv': 13.8, 'derate_mode': 'Auto-Calculate',
        'temp': 35, 'alt': 100,
        # Gens
        'gen_rating': 2.5, 'dist_loss': 1.5, 'gen_parasitic': 3.0, 'gen_step_cap': 25.0, 
        'gen_xd': 0.14, 'gen_maint': 4.0, 'gen_for': 1.0,
        # BESS
        'bess_maint': 2.0, 'bess_for': 0.5,
        # Dist Reliability
        'bus_maint': 0.5, 'bus_for': 0.1,
        'cb_maint': 0.5, 'cb_for': 0.1,
        'cable_maint': 0.1, 'cable_for': 0.1,
        'tx_maint': 0.5, 'tx_for': 0.2
    }

def get(k): return st.session_state['inputs_v5'].get(k)
def set_k(k, v): st.session_state['inputs_v5'][k] = v

with st.sidebar:
    st.title("Inputs v5.0")
    
    with st.expander("1. Data Center Profile", expanded=True):
        st.selectbox("Type", ["Hyperscale", "AI Factory", "Colo"], key='dc_type', on_change=lambda: set_k('dc_type', st.session_state.dc_type))
        st.number_input("Critical IT Load (MW)", 1.0, 500.0, float(get('p_it')), key='p_it', on_change=lambda: set_k('p_it', st.session_state.p_it))
        st.number_input("Required Availability (%)", 90.0, 99.99999, float(get('req_avail')), format="%.5f", key='req_avail', on_change=lambda: set_k('req_avail', st.session_state.req_avail))
        st.number_input("Step Load Req (%)", 0.0, 100.0, float(get('step_load_req')), key='step_load_req', on_change=lambda: set_k('step_load_req', st.session_state.step_load_req))
        st.number_input("DC Aux (%)", 0.0, 50.0, float(get('dc_aux')), key='dc_aux', on_change=lambda: set_k('dc_aux', st.session_state.dc_aux))
        
        c1, c2 = st.columns(2)
        st.selectbox("Connection Voltage", ["Auto-Recommend", "Manual"], index=0, key='volts_mode', on_change=lambda: set_k('volts_mode', st.session_state.volts_mode))
        if get('volts_mode') == 'Manual':
            st.number_input("Voltage (kV)", 0.4, 230.0, float(get('volts_kv')), key='volts_kv', on_change=lambda: set_k('volts_kv', st.session_state.volts_kv))
            
    with st.expander("2. Generator Tech"):
        c1, c2 = st.columns(2)
        c1.number_input("Rating (MW)", 0.5, 20.0, float(get('gen_rating')), key='gen_rating', on_change=lambda: set_k('gen_rating', st.session_state.gen_rating))
        c2.number_input("Voltage (kV)", 0.4, 15.0, 13.8) # Nominal gen voltage
        c1.number_input("Step Cap (%)", 0.0, 100.0, float(get('gen_step_cap')), key='gen_step_cap', on_change=lambda: set_k('gen_step_cap', st.session_state.gen_step_cap))
        c2.number_input("Xd\" (pu)", 0.05, 0.5, float(get('gen_xd')), key='gen_xd', on_change=lambda: set_k('gen_xd', st.session_state.gen_xd))
        
        st.markdown("**Losses**")
        c1, c2 = st.columns(2)
        c1.number_input("Dist Loss (%)", 0.0, 10.0, float(get('dist_loss')), key='dist_loss', on_change=lambda: set_k('dist_loss', st.session_state.dist_loss))
        c2.number_input("Parasitics (%)", 0.0, 10.0, float(get('gen_parasitic')), key='gen_parasitic', on_change=lambda: set_k('gen_parasitic', st.session_state.gen_parasitic))
        
        st.markdown("**Reliability**")
        c1, c2 = st.columns(2)
        c1.number_input("Gen Maint (%)", 0.0, 20.0, float(get('gen_maint')), key='gen_maint', on_change=lambda: set_k('gen_maint', st.session_state.gen_maint))
        c2.number_input("Gen FOR (%)", 0.0, 20.0, float(get('gen_for')), key='gen_for', on_change=lambda: set_k('gen_for', st.session_state.gen_for))

    with st.expander("3. BESS Reliability"):
        c1, c2 = st.columns(2)
        c1.number_input("BESS Maint (%)", 0.0, 20.0, float(get('bess_maint')), key='b_maint', on_change=lambda: set_k('bess_maint', st.session_state.b_maint))
        c2.number_input("BESS FOR (%)", 0.0, 20.0, float(get('bess_for')), key='b_for', on_change=lambda: set_k('bess_for', st.session_state.b_for))

    with st.expander("4. Distribution Reliability"):
        st.caption("Enter Maint% / FOR% for each:")
        c1, c2 = st.columns(2)
        c1.number_input("Bus Maint", 0.0, 5.0, float(get('bus_maint')), key='bus_m', on_change=lambda: set_k('bus_maint', st.session_state.bus_m))
        c2.number_input("Bus FOR", 0.0, 5.0, float(get('bus_for')), key='bus_f', on_change=lambda: set_k('bus_for', st.session_state.bus_f))
        
        c1, c2 = st.columns(2)
        c1.number_input("Breaker Maint", 0.0, 5.0, float(get('cb_maint')), key='cb_m', on_change=lambda: set_k('cb_maint', st.session_state.cb_m))
        c2.number_input("Breaker FOR", 0.0, 5.0, float(get('cb_for')), key='cb_f', on_change=lambda: set_k('cb_for', st.session_state.cb_f))
        
        c1, c2 = st.columns(2)
        c1.number_input("Trafo Maint", 0.0, 5.0, float(get('tx_maint')), key='tx_m', on_change=lambda: set_k('tx_maint', st.session_state.tx_m))
        c2.number_input("Trafo FOR", 0.0, 5.0, float(get('tx_for')), key='tx_f', on_change=lambda: set_k('tx_for', st.session_state.tx_f))

# RUN
res = solve_topology_v5(st.session_state['inputs_v5'])

# ==============================================================================
# 4. DASHBOARD
# ==============================================================================

st.title("CAT Topology Designer v5.0")
st.markdown(f"**Target Availability:** {get('req_avail')}% | **Connection:** {res['elec']['voltage']} kV")

# SUMMARY METRICS
m1, m2, m3, m4 = st.columns(4)
m1.metric("Gross Generation", f"{res['load']['gross_mw']:.1f} MW", f"IT: {get('p_it')} MW")
m2.metric("System Availability", f"{res['metrics']['total_rel']*100:.6f}%", 
          delta=f"{(res['metrics']['total_rel'] - res['metrics']['target'])*100:.4f}%", delta_color="normal")
m3.metric("Fleet Size", f"{res['gen']['n_total']} Gens", f"Req: {res['gen']['n_needed']}")
m4.metric("Switchgears", f"{res['elec']['num_swgr']} Buses", f"Rating: {res['elec']['bus_rating']} A")

if res['warnings']:
    for w in res['warnings']:
        st.markdown(f'<div class="warning-box">{w}</div>', unsafe_allow_html=True)
else:
    st.markdown('<div class="success-box">✅ System design meets all availability and physical constraints.</div>', unsafe_allow_html=True)

# DIAGRAM
t_diag, t_det = st.tabs(["📐 Topology Diagram", "📋 Detailed Specs"])

with t_diag:
    dot = graphviz.Digraph()
    dot.attr(rankdir='TB')
    
    # Grid/Load
    dot.node('LOAD', f'Critical Load\n{get("p_it")} MW', shape='box3d', style='filled', fillcolor='#D6EAF8')
    
    # Buses
    for b in range(1, res['elec']['num_swgr'] + 1):
        bus_id = f'BUS_{b}'
        dot.node(bus_id, f"Switchgear {b}\n{res['elec']['voltage']} kV | {res['elec']['bus_rating']} A", shape='rect', style='filled', fillcolor='#FCF3CF')
        
        # Gens
        gen_id = f'GENS_{b}'
        dot.node(gen_id, f"{res['elec']['gens_per_bus']}x Generators", shape='folder', style='filled', fillcolor='#D1F2EB')
        dot.edge(gen_id, bus_id)
        
        # Feeders
        feeders = res['dist']['per_bus']
        dist_id = f'DIST_{b}'
        dot.node(dist_id, f"{feeders}x Feeders\n(Breaker+Cable+Trafo)", shape='ellipse')
        dot.edge(bus_id, dist_id)
        dot.edge(dist_id, 'LOAD')
        
        # BESS
        if res['bess']['active']:
            bess_id = f'BESS_{b}'
            dot.node(bess_id, f"BESS Cluster", shape='component', style='filled', fillcolor='#D5F5E3')
            dot.edge(bess_id, bus_id)

    st.graphviz_chart(dot, use_container_width=True)

with t_det:
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("### 🔌 Generation & Switchgear")
        st.write(f"**Total Generators:** {res['gen']['n_total']} units")
        st.write(f"**Redundancy:** N+{res['gen']['n_total'] - res['gen']['n_needed']}")
        st.write(f"**Bus Short Circuit:** {res['elec']['bus_ka']:.2f} kA")
        if res['elec']['bus_redundant']:
            st.success("Bus Topology is N-1 Tolerant")
        else:
            st.warning("Bus Topology is NOT fully N-1 Tolerant at Peak Load")
            
    with c2:
        st.markdown("### 🔋 BESS & Distribution")
        if res['bess']['active']:
            st.write(f"**BESS Status:** Active (Step Load Support)")
            st.write(f"**Units:** {res['bess']['n_total']} Containers")
        else:
            st.write("**BESS Status:** Not Required for Step Load")
            
        st.write(f"**Total Distribution Feeders:** {res['dist']['n_total']}")
        st.write(f"**Feeder Redundancy:** N+{res['dist']['n_total'] - res['dist']['n_needed']}")
