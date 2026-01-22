import streamlit as st
import pandas as pd
import numpy as np
import math
import graphviz

# --- PAGE CONFIGURATION ---
st.set_page_config(page_title="CAT Electrical Topology Designer v1.0", page_icon="⚡", layout="wide")

# ==============================================================================
# 1. ENGINEERING LIBRARY (Based on uploaded file & standards)
# ==============================================================================

# Default Voltages derived from file snippet (e.g., 13.8kV standard)
gen_library = {
    "G3520K":  {"iso_mw": 2.5, "pf": 0.8, "voltage_kv": 13.8, "xd_pu": 0.12}, # High Speed
    "G3520FR": {"iso_mw": 2.5, "pf": 0.8, "voltage_kv": 13.8, "xd_pu": 0.14}, # High Speed
    "CG260-16":{"iso_mw": 4.0, "pf": 0.8, "voltage_kv": 13.8, "xd_pu": 0.15}, # High Speed
    "G20CM34": {"iso_mw": 9.8, "pf": 0.8, "voltage_kv": 11.0, "xd_pu": 0.16}, # Medium Speed
    "Titan 130":{"iso_mw": 16.5,"pf": 0.8, "voltage_kv": 13.8, "xd_pu": 0.18}, # Turbine
}

# ==============================================================================
# 2. TOPOLOGY ALGORITHM
# ==============================================================================

def solve_topology(inputs):
    res = {}
    model = gen_library[inputs['gen_model']]
    
    # --- A. GENERATION BLOCKS (CLUSTERING) ---
    # 1. Derated Capacity
    derate = 1.0 - (max(0, (inputs['temp'] - 25)*0.01) + max(0, (inputs['alt'] - 100)*0.0001))
    gen_mw_site = model['iso_mw'] * derate
    
    # 2. Total Gens Needed
    n_gens_run = math.ceil(inputs['load_mw'] / gen_mw_site)
    n_gens_total = n_gens_run + inputs['redundancy']
    
    # 3. Current per Gen (Amps)
    # I = P_w / (sqrt(3) * V * PF)
    i_gen_nom = (gen_mw_site * 1e6) / (math.sqrt(3) * (model['voltage_kv'] * 1000) * model['pf'])
    
    # 4. Short Circuit Contribution (Approx)
    # I_sc = I_nom / X''d
    i_sc_gen = i_gen_nom / model['xd_pu']
    
    # 5. Switchgear Limits logic
    sw_amp_limit = inputs['sw_rating_a'] * 0.85 # Safety margin
    sw_ka_limit = inputs['sw_ka_rating'] * 1000 # Convert to Amps
    
    # Max gens per bus based on Ampacity
    max_gens_amp = math.floor(sw_amp_limit / i_gen_nom)
    
    # Max gens per bus based on Short Circuit (kA)
    # Assuming standard utility contribution or worst case closed tie
    # This is a simplified check: Total Gen SC < Switchgear Rating
    max_gens_ka = math.floor(sw_ka_limit / i_sc_gen)
    
    # Determine Cluster Size
    max_gens_per_bus = min(max_gens_amp, max_gens_ka)
    
    # Force at least 1 bus if calculation fails or user inputs weird data
    if max_gens_per_bus < 1: max_gens_per_bus = 1
        
    num_buses = math.ceil(n_gens_total / max_gens_per_bus)
    gens_per_bus = math.ceil(n_gens_total / num_buses)
    
    res['gen'] = {
        'total_units': n_gens_total,
        'mw_site': gen_mw_site,
        'amps_per_gen': i_gen_nom,
        'ka_per_gen': i_sc_gen / 1000,
        'num_buses': num_buses,
        'units_per_bus': gens_per_bus,
        'bus_amps': gens_per_bus * i_gen_nom,
        'bus_ka': gens_per_bus * (i_sc_gen / 1000)
    }
    
    # --- B. BESS DIMENSIONING (PHYSICAL) ---
    res['bess'] = {'active': False}
    if inputs['use_bess']:
        # Energy Constraint
        req_mwh = inputs['bess_mwh']
        cnt_energy = math.ceil(req_mwh / inputs['cont_mwh'])
        
        # Power Constraint (Step Load)
        req_mw = inputs['bess_mw']
        cnt_power = math.ceil(req_mw / inputs['cont_mw'])
        
        n_containers = max(cnt_energy, cnt_power)
        
        res['bess'] = {
            'active': True,
            'total_containers': n_containers,
            'total_mwh': n_containers * inputs['cont_mwh'],
            'total_mw': n_containers * inputs['cont_mw'],
            'containers_per_bus': math.ceil(n_containers / num_buses)
        }

    # --- C. TRANSFORMERS (STEP-UP) ---
    res['trafo'] = {'needed': False}
    if inputs['grid_kv'] != model['voltage_kv']:
        res['trafo'] = {
            'needed': True,
            'type': "GSU (Step-Up)",
            'ratio': f"{model['voltage_kv']} / {inputs['grid_kv']} kV",
            # Sizing: Total Bus Capacity / PF
            'mva_bus': (gens_per_bus * gen_mw_site) / 0.9 # Trafo PF assumption
        }

    return res

# ==============================================================================
# 3. UI & INPUTS
# ==============================================================================

with st.sidebar:
    st.header("🎛️ Design Parameters")
    
    with st.expander("1. Global & Site", expanded=True):
        load_mw = st.number_input("IT Load (MW)", 10.0, 500.0, 100.0)
        temp = st.slider("Ambient Temp (°C)", 0, 55, 35)
        alt = st.number_input("Altitude (m)", 0, 4000, 100)
        grid_kv = st.selectbox("Grid/Distribution Voltage (kV)", [13.8, 34.5, 69.0, 115.0, 230.0], index=1)
        
    with st.expander("2. Technology", expanded=True):
        gen_model = st.selectbox("Generator Model", list(gen_library.keys()))
        redundancy = st.number_input("Spare Gens (N+)", 0, 10, 2)
        
    with st.expander("3. Switchgear Constraints"):
        sw_rating = st.selectbox("Bus Rating (A)", [1200, 2000, 3000, 4000], index=3)
        sw_ka = st.selectbox("Short Circuit (kA)", [25, 40, 50, 63], index=3)
        
    with st.expander("4. BESS Strategy"):
        use_bess = st.checkbox("Include BESS", value=True)
        if use_bess:
            bess_mw = st.number_input("Required Power (MW)", 0.0, 500.0, 20.0)
            bess_mwh = st.number_input("Required Energy (MWh)", 0.0, 1000.0, 40.0)
            st.markdown("---")
            st.caption("Container Specs (Editable)")
            cont_mw = st.number_input("Inverter Capacity (MW/Unit)", 0.5, 6.0, 3.8)
            cont_mwh = st.number_input("Energy Capacity (MWh/Unit)", 1.0, 10.0, 5.0)
        else:
            bess_mw, bess_mwh, cont_mw, cont_mwh = 0,0,0,0

# Input Packet
inputs = {
    'load_mw': load_mw, 'temp': temp, 'alt': alt, 'grid_kv': grid_kv,
    'gen_model': gen_model, 'redundancy': redundancy,
    'sw_rating_a': sw_rating, 'sw_ka_rating': sw_ka,
    'use_bess': use_bess, 'bess_mw': bess_mw, 'bess_mwh': bess_mwh,
    'cont_mw': cont_mw, 'cont_mwh': cont_mwh
}

# RUN SOLVER
res = solve_topology(inputs)
gen_data = res['gen']

# ==============================================================================
# 4. MAIN DASHBOARD
# ==============================================================================

st.title("CAT Electrical Topology Designer v1.0")

# --- KPI ROW ---
k1, k2, k3, k4 = st.columns(4)
k1.metric("Architecture", f"{gen_data['num_buses']}x Parallel Buses", f"{gen_data['units_per_bus']} Gens/Bus")
k2.metric("Total Generation", f"{gen_data['total_units']} Units", f"{gen_data['mw_site']:.2f} MW/unit")
k3.metric("Switchgear Load", f"{gen_data['bus_amps']:.0f} A / {inputs['sw_rating_a']} A", f"{gen_data['bus_amps']/inputs['sw_rating_a']*100:.0f}% Loading")
k4.metric("Short Circuit est.", f"{gen_data['bus_ka']:.1f} kA", f"Limit: {inputs['sw_ka_rating']} kA")

# --- TABS ---
t_diagram, t_specs, t_report = st.tabs(["📐 Single Line Diagram (SLD)", "📋 Equipment Specs", "📄 Engineering Report"])

with t_diagram:
    st.subheader("System Topology Visualization")
    
    # GRAPHVIZ DIAGRAM GENERATION
    dot = graphviz.Digraph()
    dot.attr(rankdir='TB') # Top to Bottom
    
    # Grid / Distribution Level
    dot.node('GRID', f'Data Center Ring\n{inputs["grid_kv"]} kV', shape='doubleoctagon', style='filled', fillcolor='#E0E0E0')
    
    for b in range(1, gen_data['num_buses'] + 1):
        bus_name = f'BUS_{b}'
        # Switchgear Bus Node
        label_bus = f"SWGR BUS {b}\n{gen_library[gen_model]['voltage_kv']} kV | {inputs['sw_rating_a']} A"
        dot.node(bus_name, label_bus, shape='rect', style='filled', fillcolor='#FFCD11', width='4')
        
        # Connect Bus to Grid (via Trafo if needed)
        if res['trafo']['needed']:
            tx_name = f'TX_{b}'
            dot.node(tx_name, f"GSU {b}\n{res['trafo']['mva_bus']:.1f} MVA", shape='trapezium')
            dot.edge(bus_name, tx_name)
            dot.edge(tx_name, 'GRID')
        else:
            dot.edge(bus_name, 'GRID', label='Direct')
            
        # Generators
        with dot.subgraph(name=f'cluster_gen_{b}') as c:
            c.attr(label=f'Generation Block {b}', style='dashed')
            # Draw representative gens (not all to avoid clutter)
            c.node(f'G_{b}_1', f'{gen_data["units_per_bus"]}x {gen_model}', shape='circle')
            dot.edge(f'G_{b}_1', bus_name)
            
        # BESS
        if res['bess']['active']:
            bess_units = res['bess']['containers_per_bus']
            b_node = f'BESS_{b}'
            dot.node(b_node, f"BESS Block\n{bess_units}x Containers", shape='component', fillcolor='#90EE90', style='filled')
            dot.edge(b_node, bus_name)

    st.graphviz_chart(dot, use_container_width=True)
    st.info("ℹ️ Diagram represents the logical grouping. Physical connections may use Tie-Breakers between buses (Iso-Parallel).")

with t_specs:
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("### 🏭 Switchgear Specification")
        st.table(pd.DataFrame({
            "Parameter": ["Voltage Class", "Bus Rating", "KA Rating", "Main Breakers", "Feeder Breakers", "BESS Breakers"],
            "Value": [
                f"{'15 kV' if gen_library[gen_model]['voltage_kv'] > 1 else '600 V'}",
                f"{inputs['sw_rating_a']} A",
                f"{inputs['sw_ka_rating']} kA",
                f"{gen_data['units_per_bus']} per Bus",
                "1-2 per Bus (to Ring)",
                f"{res['bess'].get('containers_per_bus', 0)} per Bus"
            ]
        }))
        
    with c2:
        st.markdown("### 🔋 BESS Specification")
        if res['bess']['active']:
            st.table(pd.DataFrame({
                "Parameter": ["Total Containers", "Total Power", "Total Energy", "Inverter/Unit", "Energy/Unit"],
                "Value": [
                    f"{res['bess']['total_containers']} Units",
                    f"{res['bess']['total_mw']:.1f} MW",
                    f"{res['bess']['total_mwh']:.1f} MWh",
                    f"{inputs['cont_mw']} MW",
                    f"{inputs['cont_mwh']} MWh"
                ]
            }))
        else:
            st.warning("BESS Not Included in Design")

with t_report:
    st.header("Engineering Design Summary")
    st.write(f"**Project:** Data Center {load_mw} MW Power Plant")
    st.write(f"**Topology Strategy:** Modular Block Architecture")
    
    st.markdown("#### 1. Generation Blocks")
    st.write(f"To satisfy the demand of **{load_mw} MW** with **N+{inputs['redundancy']}** redundancy, the system is segmented into **{gen_data['num_buses']}** independent buses.")
    st.write(f"This ensures that the short circuit level remains approximately **{gen_data['bus_ka']:.1f} kA**, below the safety limit of **{inputs['sw_ka_rating']} kA**.")
    
    st.markdown("#### 2. Electrical Distribution")
    if res['trafo']['needed']:
        st.write(f"Voltage step-up is required from **{gen_library[gen_model]['voltage_kv']} kV** to **{inputs['grid_kv']} kV**.")
        st.write(f"Recommended: **{gen_data['num_buses']}x Step-Up Transformers** (one per bus) rated at approx **{res['trafo']['mva_bus']:.1f} MVA** each.")
    else:
        st.write("Voltage matches Grid/Distribution. Direct connection supported (Reactors may be needed for Tie-Breakers).")