import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
import numpy as np
import math
import graphviz
from scipy.stats import binom

# --- PAGE CONFIG ---
st.set_page_config(page_title="CAT Topology Designer v7.0", page_icon="⚡", layout="wide")

# --- CSS ---
st.markdown("""
<style>
    @media print {
        [data-testid="stSidebar"], [data-testid="stHeader"], footer, .stButton { display: none !important; }
        .block-container { padding: 0 !important; margin: 0 !important; }
    }
    .warning-box { background-color: #f8d7da; border: 1px solid #f5c6cb; padding: 15px; border-radius: 5px; color: #721c24; margin-bottom: 10px; }
    .success-box { background-color: #d4edda; border: 1px solid #c3e6cb; padding: 15px; border-radius: 5px; color: #155724; margin-bottom: 10px; }
    .kpi-card { background-color: #f0f2f6; padding: 10px; border-radius: 5px; text-align: center; border-left: 5px solid #FFCD11; }
</style>
""", unsafe_allow_html=True)

# ==============================================================================
# 1. ROBUST SOLVER ENGINE (ISO-PARALLEL RING)
# ==============================================================================

def get_avail(maint, force):
    return 1.0 - ((maint + force) / 100.0)

def solve_topology_v7(inputs):
    res = {'warnings': [], 'log': []}
    
    # --- A. CARGA BRUTA ---
    p_it = inputs['p_it']
    p_gross_req = (p_it * (1 + inputs['dc_aux']/100.0)) / ((1 - inputs['dist_loss']/100.0) * (1 - inputs['gen_parasitic']/100.0))
    res['load'] = {'gross': p_gross_req, 'net': p_it}
    
    # --- B. CAPACIDAD UNITARIA ---
    derate = 1.0
    if inputs['derate_mode'] == 'Auto-Calculate':
        derate = 1.0 - (max(0, (inputs['temp'] - 25)*0.01) + max(0, (inputs['alt'] - 100)*0.0001))
    
    gen_site_mw = inputs['gen_rating'] * derate
    n_needed_pure = math.ceil(p_gross_req / gen_site_mw)
    
    # --- C. VOLTAJE Y FISICA ---
    # Selección de voltaje basada en amperaje total
    voltage_kv = inputs['volts_kv']
    if inputs['volts_mode'] == 'Auto-Recommend':
        amps_pure = (p_gross_req * 1e6) / (math.sqrt(3) * 13800 * 0.8)
        # Si la corriente total > 4000A (un solo bus), necesitamos MV. Si es > 10kA, sugerimos 34.5 o más buses.
        voltage_kv = 34.5 if amps_pure > 12000 else 13.8 
        if p_gross_req < 5.0: voltage_kv = 0.48

    i_nom_gen = (gen_site_mw * 1e6) / (math.sqrt(3) * (voltage_kv * 1000) * 0.8)
    i_sc_gen = i_nom_gen / inputs['gen_xd']
    
    # Límites Físicos del Switchgear
    LIMIT_BUS_AMP = 4000.0 # 4000A es el estándar robusto
    LIMIT_BUS_KA = 63000.0
    
    # Máximo de generadores por sección física para no fundir barras
    max_gens_phy = max(1, min(
        math.floor((LIMIT_BUS_AMP * 0.9) / i_nom_gen),
        math.floor((LIMIT_BUS_KA * 0.95) / i_sc_gen)
    ))

    # --- D. ALGORITMO DE ANILLO N+2 (The "P-050" Logic) ---
    # Buscamos una configuración (Buses x Gens) que cumpla:
    # 1. Capacidad Restante >= Carga, incluso si perdemos 1 Bus COMPLETO + 2 Generadores extras.
    # 2. Amperaje y kA dentro de límites por Bus.
    
    best_config = None
    
    # Iteramos numero de Buses (Clusters)
    # Mínimo 3 buses para hacer un anillo decente, máximo 8 (complejidad)
    for n_buses in range(3, 10):
        
        # Iteramos generadores por bus
        # Empezamos con el mínimo necesario para cubrir carga
        min_gens_per_bus = math.ceil(n_needed_pure / n_buses)
        
        for gens_per_bus in range(min_gens_per_bus, max_gens_phy + 1):
            
            total_gens = n_buses * gens_per_bus
            
            # --- CRITICAL TEST: N-1 BUS + N-2 GEN ---
            # Escenario: Falla Bus #1 (perdemos gens_per_bus) Y fallan 2 gens en otros buses
            gens_lost_scenario = gens_per_bus + 2 
            gens_remaining = total_gens - gens_lost_scenario
            
            capacity_remaining = gens_remaining * gen_site_mw
            
            if capacity_remaining >= p_gross_req:
                # Valid configuration found!
                
                # Check reliability probability just in case
                p_gen = get_avail(inputs['gen_maint'], inputs['gen_for'])
                # Simplified Probability: P(System OK) approx 1 if specific scenario passed
                # We assume Bus Rel is high.
                
                best_config = {
                    'n_buses': n_buses,
                    'gens_per_bus': gens_per_bus,
                    'total_gens': total_gens,
                    'redundancy': total_gens - n_needed_pure,
                    'bus_amps': gens_per_bus * i_nom_gen,
                    'bus_ka': (gens_per_bus * i_sc_gen) / 1000.0,
                    'surviving_margin_mw': capacity_remaining - p_gross_req
                }
                break
        
        if best_config: break
        
    if not best_config:
        res['warnings'].append("❌ No solution found within physical limits. Consider increasing Voltage or Switchgear Ratings.")
        best_config = {'n_buses': 0, 'total_gens': 0, 'gens_per_bus':0, 'bus_amps':0, 'bus_ka':0, 'redundancy':0}

    res['gen'] = best_config
    res['gen']['site_mw'] = gen_site_mw
    res['elec'] = {'voltage': voltage_kv}

    # --- E. BESS Y DISTRIBUCIÓN ---
    # BESS logic (Step Load)
    step_req = p_it * (inputs['step_load_req'] / 100.0)
    # Step Cap en peor escenario (Bus perdido)
    gens_avail_worst = best_config['total_gens'] - best_config['gens_per_bus']
    gen_step_avail = gens_avail_worst * gen_site_mw * (inputs['gen_step_cap'] / 100.0)
    
    bess_active = False
    n_bess_total = 0
    if gen_step_avail < step_req or inputs['bess_manual']:
        bess_active = True
        shortfall = max(0, step_req - gen_step_avail)
        if inputs['bess_manual']: shortfall = max(shortfall, inputs['bess_mw'])
        n_bess_total = math.ceil(shortfall / inputs['bess_inv_mw'])
        # Spread BESS across buses for redundancy
        bess_per_bus = math.ceil(n_bess_total / best_config['n_buses'])
        n_bess_total = bess_per_bus * best_config['n_buses'] # Equalize
        
    res['bess'] = {'active': bess_active, 'n_total': n_bess_total}
    
    # Distribution Feeders (N+2 Redundancy per loop logic)
    feeder_cap = 2.5 # MW
    n_feeders = math.ceil(p_gross_req / feeder_cap) + 2
    res['dist'] = {'n_feeders': n_feeders}

    return res

# ==============================================================================
# 2. UI INPUTS
# ==============================================================================

if 'inputs_v7' not in st.session_state:
    st.session_state['inputs_v7'] = {
        'p_it': 100.0, 'dc_aux': 15.0, 'req_avail': 99.999, 'step_load_req': 40.0,
        'volts_mode': 'Auto-Recommend', 'volts_kv': 13.8, 'derate_mode': 'Auto-Calculate', 'temp': 35, 'alt': 100,
        'gen_rating': 2.5, 'dist_loss': 1.5, 'gen_parasitic': 3.0, 'gen_xd': 0.14, 'gen_step_cap': 25.0,
        'gen_maint': 4.0, 'gen_for': 1.0,
        'bess_manual': False, 'bess_mw': 20.0, 'bess_inv_mw': 3.8, 'bess_maint': 2.0, 'bess_for': 0.5,
        'bus_maint': 0.1, 'bus_for': 0.05
    }

def get(k): return st.session_state['inputs_v7'].get(k)
def set_k(k, v): st.session_state['inputs_v7'][k] = v

with st.sidebar:
    st.title("Inputs v7.0")
    with st.expander("1. Data Center Profile", expanded=True):
        st.number_input("Critical IT Load (MW)", 1.0, 500.0, float(get('p_it')), key='p_it', on_change=lambda: set_k('p_it', st.session_state.p_it))
        st.number_input("Required Availability (%)", 90.0, 99.99999, float(get('req_avail')), format="%.5f", key='req_avail', on_change=lambda: set_k('req_avail', st.session_state.req_avail))
        st.number_input("Step Load Req (%)", 0.0, 100.0, float(get('step_load_req')), key='step_load_req', on_change=lambda: set_k('step_load_req', st.session_state.step_load_req))
        st.number_input("DC Aux (%)", 0.0, 50.0, float(get('dc_aux')), key='dc_aux', on_change=lambda: set_k('dc_aux', st.session_state.dc_aux))
        c1, c2 = st.columns(2)
        st.selectbox("Voltage", ["Auto-Recommend", "Manual"], index=0, key='volts_mode', on_change=lambda: set_k('volts_mode', st.session_state.volts_mode))
        if get('volts_mode') == 'Manual':
            st.number_input("kV", 0.4, 69.0, float(get('volts_kv')), key='volts_kv', on_change=lambda: set_k('volts_kv', st.session_state.volts_kv))
        st.selectbox("Derate", ["Auto-Calculate", "Manual"], index=0, key='derate_mode', on_change=lambda: set_k('derate_mode', st.session_state.derate_mode))
        if get('derate_mode') == 'Auto-Calculate':
            c1, c2 = st.columns(2)
            c1.number_input("Temp C", 0, 55, int(get('temp')), key='temp', on_change=lambda: set_k('temp', st.session_state.temp))
            c2.number_input("Alt m", 0, 3000, int(get('alt')), key='alt', on_change=lambda: set_k('alt', st.session_state.alt))

    with st.expander("2. Tech Specs", expanded=True):
        c1, c2 = st.columns(2)
        c1.number_input("Gen MW", 0.5, 20.0, float(get('gen_rating')), key='gen_rating', on_change=lambda: set_k('gen_rating', st.session_state.gen_rating))
        c2.number_input("Xd\" pu", 0.05, 0.5, float(get('gen_xd')), key='gen_xd', on_change=lambda: set_k('gen_xd', st.session_state.gen_xd))
        c1.number_input("Step Cap %", 0.0, 100.0, float(get('gen_step_cap')), key='gen_step_cap', on_change=lambda: set_k('gen_step_cap', st.session_state.gen_step_cap))
        c2.number_input("Losses %", 0.0, 20.0, float(get('dist_loss')), key='dist_loss', on_change=lambda: set_k('dist_loss', st.session_state.dist_loss))
        c1.number_input("Parasitics %", 0.0, 20.0, float(get('gen_parasitic')), key='gen_parasitic', on_change=lambda: set_k('gen_parasitic', st.session_state.gen_parasitic))
        st.caption("Reliability")
        c1, c2 = st.columns(2)
        c1.number_input("Maint %", 0.0, 20.0, float(get('gen_maint')), key='gen_maint', on_change=lambda: set_k('gen_maint', st.session_state.gen_maint))
        c2.number_input("FOR %", 0.0, 20.0, float(get('gen_for')), key='gen_for', on_change=lambda: set_k('gen_for', st.session_state.gen_for))

    with st.expander("3. BESS & Components"):
        st.checkbox("Force BESS", value=get('bess_manual'), key='bess_manual', on_change=lambda: set_k('bess_manual', st.session_state.bess_manual))
        c1, c2 = st.columns(2)
        c1.number_input("Inv MW", 0.5, 6.0, float(get('bess_inv_mw')), key='bess_inv_mw', on_change=lambda: set_k('bess_inv_mw', st.session_state.bess_inv_mw))
        c2.number_input("BESS Manual MW", 0.0, 500.0, float(get('bess_mw')), key='bess_mw', on_change=lambda: set_k('bess_mw', st.session_state.bess_mw))
        st.caption("Component Reliability (Maint/FOR)")
        st.number_input("Bus Maint %", 0.0, 5.0, float(get('bus_maint')), key='bus_maint', on_change=lambda: set_k('bus_maint', st.session_state.bus_maint))

res = solve_topology_v7(st.session_state['inputs_v7'])

# ==============================================================================
# 3. DASHBOARD
# ==============================================================================

st.title("CAT Topology Designer v7.0")
st.subheader("Iso-Parallel Ring Architecture")

# SUMMARY METRICS
c1, c2, c3, c4 = st.columns(4)
with c1:
    st.markdown(f"""<div class="kpi-card"><h3>{res['load']['gross']:.1f} MW</h3><p>Gross Generation Load</p></div>""", unsafe_allow_html=True)
with c2:
    st.markdown(f"""<div class="kpi-card"><h3>{res['gen']['total_gens']} Units</h3><p>Total Fleet (N+{res['gen']['redundancy']})</p></div>""", unsafe_allow_html=True)
with c3:
    st.markdown(f"""<div class="kpi-card"><h3>{res['gen']['n_buses']} Buses</h3><p>{res['gen']['gens_per_bus']} Gens/Bus</p></div>""", unsafe_allow_html=True)
with c4:
    st.markdown(f"""<div class="kpi-card"><h3>{res['elec']['voltage']} kV</h3><p>{res['gen']['bus_ka']:.1f} kA / Bus</p></div>""", unsafe_allow_html=True)

st.divider()

if res['gen']['n_buses'] > 0:
    st.markdown(f"""
    <div class="success-box">
        <h4>🛡️ High Resilience "N-1 Bus + 2 Gen" Configuration</h4>
        This topology is designed to withstand the catastrophic loss of <b>1 Complete Bus Section</b> ({res['gen']['gens_per_bus']} Generators lost) 
        PLUS the random failure of <b>2 additional Generators</b> elsewhere in the ring.<br>
        <ul>
            <li><b>Worst Case Capacity Loss:</b> {(res['gen']['gens_per_bus'] + 2) * res['gen']['site_mw']:.1f} MW</li>
            <li><b>Remaining Capacity:</b> {(res['gen']['total_gens'] - res['gen']['gens_per_bus'] - 2) * res['gen']['site_mw']:.1f} MW</li>
            <li><b>Required Load:</b> {res['load']['gross']:.1f} MW</li>
            <li><b>Result:</b> <b>PASS</b> (System holds load)</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)
else:
    st.error("Optimization Failed. Increase voltage or generator size.")

# --- DIAGRAMMING ---
t_sld, t_specs = st.tabs(["📐 Ring Topology Diagram", "📋 Detailed Specs"])

with t_sld:
    st.markdown("**Diagram Note:** This schematic represents an **Iso-Parallel Ring**. All buses are interconnected via Tie-Breakers (and potentially Current Limiting Reactors).")
    
    dot = graphviz.Digraph()
    dot.attr(rankdir='LR', splines='ortho') # Left-to-Right for Ring layout
    
    # Create the Ring of Buses
    n_buses = res['gen']['n_buses']
    
    for i in range(1, n_buses + 1):
        bus_name = f'BUS_{i}'
        
        # Subgraph for each Cluster
        with dot.subgraph(name=f'cluster_{i}') as c:
            c.attr(label=f'Bus Cluster {i}', style='dashed', color='grey')
            
            # Bus Bar
            c.node(bus_name, f'Bus {i}\n{res["gen"]["bus_amps"]:.0f}A', shape='rect', style='filled', fillcolor='#FFCD11', width='2')
            
            # Generators
            c.node(f'G_{i}', f'{res["gen"]["gens_per_bus"]}x Gens', shape='folder', style='filled', fillcolor='#D1F2EB')
            c.edge(f'G_{i}', bus_name)
            
            # BESS
            if res['bess']['active']:
                bess_per = math.ceil(res['bess']['n_total']/n_buses)
                c.node(f'B_{i}', f'{bess_per}x BESS', shape='component', style='filled', fillcolor='#A9DFBF')
                c.edge(f'B_{i}', bus_name)
                
            # Output Feeder Group
            feeders_per = math.ceil(res['dist']['n_feeders']/n_buses)
            c.node(f'F_{i}', f'{feeders_per}x Feeders', shape='ellipse')
            c.edge(bus_name, f'F_{i}')

    # Create Ring Connections (Tie Breakers)
    for i in range(1, n_buses + 1):
        current_bus = f'BUS_{i}'
        next_bus_idx = i + 1 if i < n_buses else 1
        next_bus = f'BUS_{next_bus_idx}'
        
        # Tie Breaker Node
        tie_name = f'Tie_{i}_{next_bus_idx}'
        dot.node(tie_name, 'TIE\n(NC)', shape='circle', width='0.8', fixedsize='true', style='filled', fillcolor='white')
        
        dot.edge(current_bus, tie_name, dir='none', color='black', penwidth='2')
        dot.edge(tie_name, next_bus, dir='none', color='black', penwidth='2')

    st.graphviz_chart(dot, use_container_width=True)

with t_specs:
    st.write(res)
