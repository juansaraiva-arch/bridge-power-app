import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint

# --- PAGE CONFIG ---
st.set_page_config(page_title="CAT Stability Sim v3.1 (Pulse Dynamics)", page_icon="💓", layout="wide")

# --- CSS ---
st.markdown("""
<style>
    .section-header { font-size: 16px; font-weight: bold; color: #444; margin-top: 10px; margin-bottom: 5px; border-bottom: 1px solid #ddd; }
    .success-box { background-color: #d4edda; padding: 10px; border-radius: 5px; border-left: 5px solid #28a745; }
    .fail-box { background-color: #f8d7da; padding: 10px; border-radius: 5px; border-left: 5px solid #dc3545; }
    .warning-box { background-color: #fff3cd; padding: 10px; border-radius: 5px; border-left: 5px solid #ffc107; }
</style>
""", unsafe_allow_html=True)

# ==============================================================================
# 1. DATA LIBRARY
# ==============================================================================
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
# 2. INPUT SIDEBAR
# ==============================================================================

with st.sidebar:
    st.title("Inputs v3.1 (AI Dynamics)")
    
    with st.expander("1. Project & Load", expanded=True):
        p_it = st.number_input("IT Load (MW)", 1.0, 5000.0, 100.0) 
        dc_aux = st.number_input("Auxiliaries & Cooling (%)", 0.0, 50.0, 15.0)
        
        st.markdown('<div class="section-header">AI Load Behavior</div>', unsafe_allow_html=True)
        base_load_pct = st.number_input("Base Load (%)", 10.0, 100.0, 50.0)
        step_req_pct = st.number_input("AI Spike Peak (%)", 0.0, 100.0, 40.0)
        
        # --- NEW PARAMETERS: DURATION & TYPE ---
        load_type = st.radio("Fluctuation Type", ["Single Pulse (Spike)", "Step (Permanent)"])
        if load_type == "Single Pulse (Spike)":
            pulse_duration = st.number_input("Spike Duration (s)", 0.1, 10.0, 2.0, help="Typical Inference/Checkpoint duration.")
        else:
            pulse_duration = 999.0 # Effectively infinite

    with st.expander("2. Generation Fleet", expanded=True):
        gen_model = st.selectbox("Generator Model", list(CAT_LIBRARY.keys()))
        specs = CAT_LIBRARY[gen_model]
        
        p_gross_total = p_it * (1 + dc_aux/100.0)
        n_rec_calc = int(np.ceil(p_gross_total / specs['mw'])) + 1
        MAX_GENS_HARD_LIMIT = 5000 
        n_default_safe = min(n_rec_calc, MAX_GENS_HARD_LIMIT)
        
        n_gens_op = st.number_input("Operating Units (N)", 1, MAX_GENS_HARD_LIMIT, n_default_safe)
        
        st.markdown('<div class="section-header">Dynamic Parameters</div>', unsafe_allow_html=True)
        h_const = st.number_input("Inertia Constant H (s)", 0.1, 20.0, float(specs['h_def']))
        tau_gov = st.number_input("Governor Response (s)", 0.05, 10.0, 0.5)

    with st.expander("3. BESS System"):
        bess_mode = st.selectbox("Control Mode", ["Grid Forming", "Disabled"])
        if bess_mode != "Disabled":
            step_mw = p_it * (step_req_pct/100.0)
            bess_cap = st.number_input("BESS Power (MW)", 0.0, 5000.0, step_mw)
            bess_response = st.number_input("Response Time (ms)", 10, 2000, 50)
        else:
            bess_cap = 0.0
            bess_response = 1000

    with st.expander("4. Criteria"):
        nadir_limit = st.number_input("Min Freq (Hz)", 40.0, 59.9, 57.0)
        overshoot_limit = st.number_input("Max Freq (Hz)", 60.1, 70.0, 63.0, help="High frequency trip limit.")

# ==============================================================================
# 3. PHYSICS ENGINE (UPDATED FOR PULSES)
# ==============================================================================

def system_dynamics_pulse(y, t, params):
    freq_dev, p_mech_dev, p_bess_dev = y
    
    # Unpack
    H = max(params['H'], 0.1) 
    Sys_MVA = max(params['Sys_MVA'], 0.1)
    Tau_g = max(params['Tau_g'], 0.02)
    Tau_b = max(params['Tau_b'], 0.02)
    
    P_spike = params['P_spike']
    P_bess_cap = params['P_bess_cap']
    Duration = params['Duration']
    
    # 1. Electrical Load Profile (PULSE LOGIC)
    # Load rises at t=1.0 and falls at t=1.0+Duration
    if 1.0 <= t < (1.0 + Duration):
        p_elec_dev = P_spike
    else:
        p_elec_dev = 0.0
        
    # 2. Governor (Mechanical Power)
    target_mech = p_elec_dev 
    d_pmech_dt = (target_mech - p_mech_dev) / Tau_g
    
    # 3. BESS Response
    target_bess = 0.0
    if params['Bess_Enabled']:
        # If load is high, inject. If load drops, stop injecting.
        # Simple logic: follow the load dev, capped by rating
        req_bess = p_elec_dev 
        target_bess = min(P_bess_cap, req_bess)
        
    d_pbess_dt = (target_bess - p_bess_dev) / Tau_b
    
    # 4. Swing Equation
    p_acc_mw = p_mech_dev + p_bess_dev - p_elec_dev
    f0 = 60.0
    d_freq_dt = (p_acc_mw / Sys_MVA) * (f0 / (2 * H))
    
    return [d_freq_dt, d_pmech_dt, d_pbess_dt]

# ==============================================================================
# 4. SIMULATION EXECUTION
# ==============================================================================

st.title("⚡ CAT Stability Sim v3.1 (Pulse Analysis)")
st.markdown(f"**Scenario:** {load_type} of {step_req_pct}% Load")

# Stats
total_gen_cap_mw = n_gens_op * specs['mw']
mw_base = p_gross_total * (base_load_pct/100.0)
mw_step = p_gross_total * (step_req_pct/100.0)
sys_mva = total_gen_cap_mw / 0.8

c1, c2, c3 = st.columns(3)
c1.metric("System Inertia", f"{sys_mva * h_const:.1f} MWs")
c2.metric("Spike Magnitude", f"{mw_step:.1f} MW")
c3.metric("Spike Duration", f"{pulse_duration} s" if pulse_duration < 900 else "Infinite")

if st.button("🚀 Run Pulse Simulation", type="primary"):
    
    sim_params = {
        'H': h_const,
        'Tau_g': tau_gov,
        'Tau_b': bess_response / 1000.0,
        'P_spike': mw_step,
        'P_bess_cap': bess_cap,
        'Sys_MVA': sys_mva,
        'Bess_Enabled': (bess_mode != "Disabled"),
        'Duration': pulse_duration
    }
    
    t_end = 15.0 
    t = np.linspace(0, t_end, 1500)
    y0 = [0.0, 0.0, 0.0]
    
    try:
        sol = odeint(system_dynamics_pulse, y0, t, args=(sim_params,))
        
        freq_trace = 60.0 + sol[:, 0]
        p_mech_trace = mw_base + sol[:, 1]
        p_bess_trace = sol[:, 2]
        p_load_trace = [mw_base + (mw_step if 1.0 <= ti < (1.0 + pulse_duration) else 0) for ti in t]
        
        nadir = np.min(freq_trace)
        peak_freq = np.max(freq_trace)
        
        # --- VISUALIZATION ---
        st.write("---")
        col_g, col_kpi = st.columns([3, 1])
        
        with col_g:
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
            
            # Frequency
            ax1.plot(t, freq_trace, 'b-', linewidth=2, label="Freq")
            ax1.axhline(nadir_limit, color='r', linestyle='--', label="Low Limit")
            ax1.axhline(overshoot_limit, color='orange', linestyle='--', label="High Limit")
            
            # Highlight Pulse Area
            ax1.axvspan(1.0, 1.0+pulse_duration, color='gray', alpha=0.1, label="Load Active")
            
            ax1.set_ylabel("Hz")
            ax1.grid(True, alpha=0.3)
            ax1.legend(loc="upper right")
            ax1.set_title("Frequency Response (Overshoot & Undershoot)")
            
            # Power
            ax2.plot(t, p_load_trace, 'k--', label="Load (AI Pulse)")
            ax2.plot(t, p_mech_trace, 'g-', label="Gen (Mech)")
            if sim_params['Bess_Enabled']:
                ax2.plot(t, p_bess_trace, 'm-', label="BESS")
                
            ax2.set_ylabel("MW")
            ax2.set_xlabel("Time (s)")
            ax2.grid(True, alpha=0.3)
            ax2.legend()
            
            st.pyplot(fig)
            
        with col_kpi:
            st.subheader("Metrics")
            
            st.metric("Lowest Freq (Nadir)", f"{nadir:.3f} Hz", delta=f"{nadir-60.0:.3f}")
            st.metric("Highest Freq (Peak)", f"{peak_freq:.3f} Hz", delta=f"{peak_freq-60.0:.3f}")
            
            # Logic for Pass/Fail
            if nadir < nadir_limit:
                st.markdown(f"<div class='fail-box'>❌ <b>TRIP:</b> Under-Frequency</div>", unsafe_allow_html=True)
            elif peak_freq > overshoot_limit:
                 st.markdown(f"<div class='fail-box'>❌ <b>TRIP:</b> Over-Frequency (Overshoot)</div>", unsafe_allow_html=True)
            else:
                st.markdown(f"<div class='success-box'>✅ <b>PASS:</b> Stable</div>", unsafe_allow_html=True)
                
            if peak_freq > 61.0 and not sim_params['Bess_Enabled']:
                st.info("ℹ️ Note the Overshoot when load drops. BESS can help absorb this recoil.")

    except Exception as e:
        st.error(f"Error: {str(e)}")
