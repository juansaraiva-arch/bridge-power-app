import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint

# --- PAGE CONFIG ---
st.set_page_config(page_title="CAT Stability Sim v2.0", page_icon="📉", layout="wide")

# --- CSS ---
st.markdown("""
<style>
    .section-header { font-size: 16px; font-weight: bold; color: #444; margin-top: 10px; margin-bottom: 5px; border-bottom: 1px solid #ddd; }
    .param-help { font-size: 12px; color: #666; font-style: italic; }
    .success-box { background-color: #d4edda; padding: 10px; border-radius: 5px; border-left: 5px solid #28a745; }
    .fail-box { background-color: #f8d7da; padding: 10px; border-radius: 5px; border-left: 5px solid #dc3545; }
</style>
""", unsafe_allow_html=True)

# ==============================================================================
# 1. DATA LIBRARY (From Excel + Dynamic Defaults)
# ==============================================================================

# H (Inertia) defaults: Recip=0.8-1.2s, Turbine=3.0-5.0s
CAT_LIBRARY = {
    "XGC1900 (1.9 MW)":   {"mw": 1.9,   "type": "Recip",   "h_def": 1.0},
    "G3520FR (2.5 MW)":   {"mw": 2.5,   "type": "Recip",   "h_def": 1.0},
    "G3520K (2.4 MW)":    {"mw": 2.4,   "type": "Recip",   "h_def": 1.0},
    "CG260 (3.96 MW)":    {"mw": 3.957, "type": "Recip",   "h_def": 1.2},
    "G20CM34 (9.76 MW)":  {"mw": 9.76,  "type": "Recip",   "h_def": 1.5},
    "Titan 130 (16.5 MW)":{"mw": 16.5,  "type": "Turbine", "h_def": 4.0},
    "Titan 250 (23.2 MW)":{"mw": 23.2,  "type": "Turbine", "h_def": 4.5},
    "Titan 350 (38.0 MW)":{"mw": 38.0,  "type": "Turbine", "h_def": 5.0}
}

# ==============================================================================
# 2. INPUT SIDEBAR (4 Categories)
# ==============================================================================

with st.sidebar:
    st.title("Inputs v2.0")
    
    # --- 1. PROJECT & LOAD ---
    with st.expander("1. Project & Load Profile", expanded=True):
        p_it = st.number_input("IT Load (MW)", 1.0, 500.0, 100.0)
        dc_aux = st.number_input("Auxiliaries & Cooling (%)", 0.0, 50.0, 15.0)
        
        st.markdown('<div class="section-header">Simulated Event</div>', unsafe_allow_html=True)
        base_load_pct = st.number_input("Initial Base Load (%)", 10.0, 100.0, 50.0, help="Load on gensets BEFORE the step.")
        step_req_pct = st.number_input("AI Step Load (%)", 0.0, 100.0, 40.0, help="Sudden load increase to simulate.")

    # --- 2. GENERATION FLEET ---
    with st.expander("2. Generation Fleet (Dynamics)", expanded=True):
        gen_model = st.selectbox("Generator Model", list(CAT_LIBRARY.keys()))
        specs = CAT_LIBRARY[gen_model]
        
        # Calculate Gross Load to recommend N
        p_gross_total = p_it * (1 + dc_aux/100.0)
        n_rec = int(np.ceil(p_gross_total / specs['mw'])) + 1
        
        n_gens_op = st.number_input("Operating Units (N)", 1, 50, n_rec, help="Number of units running and synchronized.")
        
        st.markdown('<div class="section-header">Dynamic Parameters</div>', unsafe_allow_html=True)
        h_const = st.number_input("Inertia Constant H (s)", 0.1, 10.0, float(specs['h_def']), help="Kinetic energy buffer. Vital for frequency stability.")
        
        tau_def = 0.5 if specs['type'] == 'Recip' else 1.5
        tau_gov = st.number_input("Governor Response (s)", 0.05, 5.0, tau_def, help="Turbo lag / Fuel valve delay.")
        
        gen_d = st.number_input("Damping Factor D", 0.0, 5.0, 1.0, help="Natural load damping.")

    # --- 3. BESS SYSTEM ---
    with st.expander("3. BESS System (Fast Response)", expanded=True):
        bess_mode = st.selectbox("Control Mode", ["Grid Forming (Virtual Inertia)", "Grid Following (PQ)", "Disabled"])
        
        if bess_mode != "Disabled":
            # Auto-size suggestion based on Step Load
            step_mw = p_it * (step_req_pct/100.0)
            bess_cap = st.number_input("BESS Power (MW)", 0.0, 500.0, step_mw, help="Should cover the step load for best results.")
            
            bess_response = st.number_input("Response Time (ms)", 1, 1000, 50, help="Inverter latency (detection + activation).")
            bess_droop = st.number_input("Freq Droop (%)", 0.1, 10.0, 1.0, help="Lower % = More aggressive injection.")
        else:
            bess_cap = 0.0
            bess_response = 999
            bess_droop = 5.0

    # --- 4. LIMITS ---
    with st.expander("4. Stability Limits (Criteria)"):
        nadir_limit = st.number_input("Min Frequency Limit (Hz)", 40.0, 59.5, 57.0, help="ANSI 81U Trip level.")
        rocof_limit = st.number_input("Max RoCoF (Hz/s)", 0.1, 10.0, 2.0, help="Rate of Change of Frequency limit.")
        settling_time_limit = st.number_input("Max Recovery Time (s)", 1.0, 20.0, 5.0)

# ==============================================================================
# 3. PRE-CALCULATION & UI FEEDBACK
# ==============================================================================

st.title("⚡ CAT Transient Stability Simulator v2.0")
st.markdown(f"**Scenario:** {step_req_pct}% AI Load Step on {n_gens_op}x {gen_model}")

# Calculate Initial State
total_gen_cap_mw = n_gens_op * specs['mw']
mw_base = p_gross_total * (base_load_pct/100.0)
mw_step = p_gross_total * (step_req_pct/100.0)
mw_final = mw_base + mw_step

# System Inertia (Total)
sys_mva = total_gen_cap_mw / 0.8
e_kinetic = sys_mva * h_const # MW-s

col1, col2, col3, col4 = st.columns(4)
col1.metric("System Inertia", f"{e_kinetic:.1f} MWs")
col2.metric("Base Load", f"{mw_base:.1f} MW", f"{(mw_base/total_gen_cap_mw)*100:.0f}% Cap")
col3.metric("Step Load", f"+{mw_step:.1f} MW", "Disturbance")
col4.metric("Total Final Load", f"{mw_final:.1f} MW", f"{(mw_final/total_gen_cap_mw)*100:.0f}% Cap")

if mw_final > total_gen_cap_mw:
    st.error(f"⚠️ **OVERLOAD WARNING:** Final load ({mw_final:.1f} MW) exceeds generation capacity ({total_gen_cap_mw:.1f} MW). Simulation will likely crash.")

# ==============================================================================
# 4. PLACEHOLDER FOR SIMULATION (NEXT STEP)
# ==============================================================================

st.info("👆 Adjust parameters in the sidebar and click 'Run Simulation' to solve the Swing Equation.")

if st.button("🚀 Run Transient Simulation", type="primary"):
    # (Here we will plug in the ODE Solver in the next step)
    # Just a placeholder graph for now to show UI layout
    t = np.linspace(0, 10, 100)
    f_dummy = 60.0 - 1.5 * np.exp(-t) * np.sin(2*np.pi*t) 
    
    st.write("---")
    st.subheader("Simulation Output (Preview)")
    
    c1, c2 = st.columns([3, 1])
    with c1:
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(t, f_dummy, label="Frequency (Hz)")
        ax.axhline(nadir_limit, color='r', linestyle='--', label='Limit')
        ax.set_ylabel("Hz")
        ax.grid(True, alpha=0.3)
        ax.legend()
        st.pyplot(fig)
    
    with c2:
        st.markdown("#### Status")
        st.markdown(f"**Nadir:** 58.5 Hz")
        st.markdown(f"<div class='success-box'>✅ PASS</div>", unsafe_allow_html=True)