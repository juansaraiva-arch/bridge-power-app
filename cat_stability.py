import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint

# --- PAGE CONFIG ---
st.set_page_config(page_title="CAT Stability Sim v4.0 (Diagnostic)", page_icon="🩺", layout="wide")

# --- CSS ---
st.markdown("""
<style>
    .section-header { font-size: 16px; font-weight: bold; color: #444; margin-top: 10px; margin-bottom: 5px; border-bottom: 1px solid #ddd; }
    .success-box { background-color: #d4edda; padding: 15px; border-radius: 5px; border-left: 5px solid #28a745; margin-bottom: 10px; }
    .fail-box { background-color: #f8d7da; padding: 15px; border-radius: 5px; border-left: 5px solid #dc3545; margin-bottom: 10px; }
    .rec-box { background-color: #e2e3e5; padding: 15px; border-radius: 5px; border-left: 5px solid #383d41; margin-top: 10px; }
    .metric-value { font-size: 24px; font-weight: bold; }
</style>
""", unsafe_allow_html=True)

# ==============================================================================
# 1. DATA LIBRARY
# ==============================================================================
CAT_LIBRARY = {
    "XGC1900 (1.9 MW)":   {"mw": 1.9,   "type": "Recip",   "h_def": 1.0, "tau_def": 0.5},
    "G3520FR (2.5 MW)":   {"mw": 2.5,   "type": "Recip",   "h_def": 1.0, "tau_def": 0.6},
    "G3520K (2.4 MW)":    {"mw": 2.4,   "type": "Recip",   "h_def": 1.0, "tau_def": 0.6},
    "CG260 (3.96 MW)":    {"mw": 3.957, "type": "Recip",   "h_def": 1.2, "tau_def": 0.8},
    "G20CM34 (9.76 MW)":  {"mw": 9.76,  "type": "Recip",   "h_def": 1.5, "tau_def": 1.0},
    "Titan 130 (16.5 MW)":{"mw": 16.5,  "type": "Turbine", "h_def": 4.0, "tau_def": 1.5},
    "Titan 250 (23.2 MW)":{"mw": 23.2,  "type": "Turbine", "h_def": 4.5, "tau_def": 2.0},
    "Titan 350 (38.0 MW)":{"mw": 38.0,  "type": "Turbine", "h_def": 5.0, "tau_def": 2.5}
}

# ==============================================================================
# 2. INPUT SIDEBAR
# ==============================================================================

with st.sidebar:
    st.title("Inputs v4.0")
    st.caption("Diagnostic & Recommendations Mode")
    
    with st.expander("1. Project & Load", expanded=True):
        p_it = st.number_input("IT Load (MW)", 1.0, 5000.0, 100.0) 
        dc_aux = st.number_input("Auxiliaries & Cooling (%)", 0.0, 50.0, 15.0)
        
        st.markdown('<div class="section-header">AI Pulse Profile</div>', unsafe_allow_html=True)
        base_load_pct = st.number_input("Base Load (%)", 10.0, 100.0, 50.0)
        step_req_pct = st.number_input("AI Spike Peak (%)", 0.0, 100.0, 40.0)
        
        load_type = st.radio("Load Pattern", ["Single Pulse (Inference)", "Step (Training)"])
        pulse_duration = st.number_input("Pulse Duration (s)", 0.5, 20.0, 3.0) if load_type == "Single Pulse (Inference)" else 999.0

    with st.expander("2. Generation Fleet", expanded=True):
        gen_model = st.selectbox("Generator Model", list(CAT_LIBRARY.keys()))
        specs = CAT_LIBRARY[gen_model]
        
        p_gross_total = p_it * (1 + dc_aux/100.0)
        n_rec_calc = int(np.ceil(p_gross_total / specs['mw'])) + 1
        MAX_GENS = 5000 
        n_default = min(n_rec_calc, MAX_GENS)
        
        n_gens_op = st.number_input("Operating Units (N)", 1, MAX_GENS, n_default, help="Try increasing this to solve trips!")
        
        st.markdown('<div class="section-header">Physics</div>', unsafe_allow_html=True)
        h_const = st.number_input("Inertia H (s)", 0.1, 20.0, float(specs['h_def']))
        tau_gov = st.number_input("Governor Lag (s)", 0.05, 10.0, float(specs.get('tau_def', 0.5)))

    with st.expander("3. BESS Mitigation (Optional)", expanded=True):
        # OPTIONAL BESS CHECKBOX
        enable_bess = st.checkbox("Enable BESS", value=False)
        
        if enable_bess:
            step_mw = p_it * (step_req_pct/100.0)
            bess_cap = st.number_input("BESS Power (MW)", 0.0, 5000.0, step_mw)
            bess_response = st.number_input("Response Time (ms)", 10, 2000, 50)
        else:
            bess_cap = 0.0
            bess_response = 1000

    with st.expander("4. Safety Limits"):
        nadir_limit = st.number_input("Min Freq (Hz)", 50.0, 59.9, 57.0)
        overshoot_limit = st.number_input("Max Freq (Hz)", 60.1, 70.0, 63.0)

# ==============================================================================
# 3. PHYSICS ENGINE
# ==============================================================================

def system_dynamics(y, t, params):
    freq_dev, p_mech_dev, p_bess_dev = y
    
    H = max(params['H'], 0.1); Sys_MVA = max(params['Sys_MVA'], 0.1)
    Tau_g = max(params['Tau_g'], 0.05); Tau_b = max(params['Tau_b'], 0.02)
    
    P_spike = params['P_spike']
    P_bess_cap = params['P_bess_cap']
    Duration = params['Duration']
    
    # Load
    p_elec_dev = P_spike if 1.0 <= t < (1.0 + Duration) else 0.0
        
    # Governor
    target_mech = p_elec_dev 
    d_pmech_dt = (target_mech - p_mech_dev) / Tau_g
    
    # BESS
    target_bess = 0.0
    if params['Bess_Enabled']:
        target_bess = min(P_bess_cap, p_elec_dev)
        
    d_pbess_dt = (target_bess - p_bess_dev) / Tau_b
    
    # Swing
    p_acc_mw = p_mech_dev + p_bess_dev - p_elec_dev
    f0 = 60.0
    d_freq_dt = (p_acc_mw / Sys_MVA) * (f0 / (2 * H))
    
    return [d_freq_dt, d_pmech_dt, d_pbess_dt]

# ==============================================================================
# 4. SIMULATION EXECUTION & DIAGNOSTICS
# ==============================================================================

st.title("⚡ CAT Stability Sim v4.0")
st.markdown(f"**Scenario:** {step_req_pct}% Pulse | **Config:** {n_gens_op}x Gens | **BESS:** {'ON' if enable_bess else 'OFF'}")

# Stats
total_gen_cap_mw = n_gens_op * specs['mw']
mw_base = p_gross_total * (base_load_pct/100.0)
mw_step = p_gross_total * (step_req_pct/100.0)
sys_mva = total_gen_cap_mw / 0.8

c1, c2, c3 = st.columns(3)
c1.metric("System Inertia", f"{sys_mva * h_const:.1f} MWs")
c2.metric("Spike Magnitude", f"{mw_step:.1f} MW")
c3.metric("Load Step", f"+{mw_step:.1f} MW")

if st.button("🚀 Run Simulation", type="primary"):
    
    sim_params = {
        'H': h_const, 'Tau_g': tau_gov, 'Tau_b': bess_response / 1000.0,
        'P_spike': mw_step, 'P_bess_cap': bess_cap, 'Sys_MVA': sys_mva,
        'Bess_Enabled': enable_bess, 'Duration': pulse_duration
    }
    
    t_end = max(15.0, pulse_duration + 10.0)
    t = np.linspace(0, t_end, 2000)
    y0 = [0.0, 0.0, 0.0]
    
    try:
        sol = odeint(system_dynamics, y0, t, args=(sim_params,))
        
        freq_trace = 60.0 + sol[:, 0]
        p_mech_trace = mw_base + sol[:, 1]
        p_bess_trace = sol[:, 2]
        p_load_trace = [mw_base + (mw_step if 1.0 <= ti < (1.0 + pulse_duration) else 0) for ti in t]
        
        nadir = np.min(freq_trace)
        peak_freq = np.max(freq_trace)
        
        # --- PLOTTING ---
        st.write("---")
        col_g, col_kpi = st.columns([3, 1])
        
        with col_g:
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
            ax1.plot(t, freq_trace, 'b-', linewidth=2, label="Freq")
            ax1.axhline(nadir_limit, color='r', linestyle='--', label="Min Limit")
            ax1.axhline(overshoot_limit, color='orange', linestyle='--', label="Max Limit")
            if pulse_duration < 100:
                ax1.axvspan(1.0, 1.0+pulse_duration, color='gray', alpha=0.1)
            ax1.set_ylabel("Frequency (Hz)")
            ax1.grid(True, alpha=0.3); ax1.legend(loc="upper right")
            ax1.set_title("Frequency Response")
            
            ax2.plot(t, p_load_trace, 'k--', label="Load")
            ax2.plot(t, p_mech_trace, 'g-', label="Gens")
            if enable_bess: ax2.plot(t, p_bess_trace, 'm-', label="BESS")
            ax2.set_ylabel("MW"); ax2.set_xlabel("Time (s)")
            ax2.grid(True, alpha=0.3); ax2.legend(loc="upper right")
            st.pyplot(fig)
            
        with col_kpi:
            st.subheader("Diagnostics")
            st.metric("Nadir", f"{nadir:.3f} Hz", delta=f"{nadir-60.0:.3f}")
            st.metric("Peak", f"{peak_freq:.3f} Hz", delta=f"{peak_freq-60.0:.3f}")
            
            # --- TRIP ANALYSIS ---
            trip_low = nadir < nadir_limit
            trip_high = peak_freq > overshoot_limit
            
            if not trip_low and not trip_high:
                st.markdown(f"<div class='success-box'>✅ <b>PASS</b><br>System is Stable.</div>", unsafe_allow_html=True)
            else:
                st.markdown(f"<div class='fail-box'>❌ <b>TRIP DETECTED</b></div>", unsafe_allow_html=True)
                if trip_low: st.error(f"Under-Frequency: {nadir:.2f} Hz < {nadir_limit} Hz")
                if trip_high: st.error(f"Over-Frequency: {peak_freq:.2f} Hz > {overshoot_limit} Hz")

        # --- RECOMMENDATION ENGINE ---
        if trip_low or trip_high:
            st.markdown("### 🩺 Recommendations to Avoid Trip")
            
            # Recommendation 1: More Inertia (Gens)
            # Heuristic: Double the inertia usually halves the freq deviation approx.
            # Estimate needed inertia ratio
            freq_drop = 60.0 - nadir
            allowed_drop = 60.0 - nadir_limit
            ratio_needed = freq_drop / allowed_drop if allowed_drop > 0 else 2.0
            new_gens = int(n_gens_op * ratio_needed * 1.2) # 20% safety margin
            
            st.markdown(f"<div class='rec-box'><b>Option A: Increase Spinning Reserve (Inertia)</b><br>"
                        f"Add more generators online to make the system 'heavier' and slower to drop.<br>"
                        f"👉 Try increasing operating units to approx: <b>{new_gens} Units</b></div>", unsafe_allow_html=True)
            
            # Recommendation 2: BESS
            if not enable_bess:
                st.markdown(f"<div class='rec-box'><b>Option B: Enable BESS (Fast Response)</b><br>"
                            f"Batteries respond in ms, covering the load step instantly while engines spool up.<br>"
                            f"👉 Enable BESS with at least: <b>{mw_step:.1f} MW</b> capacity.</div>", unsafe_allow_html=True)
            elif bess_cap < mw_step:
                st.markdown(f"<div class='rec-box'><b>Option B: Increase BESS Capacity</b><br>"
                            f"Your BESS ({bess_cap} MW) is smaller than the Load Spike ({mw_step:.1f} MW).<br>"
                            f"👉 Increase BESS to match the spike: <b>{mw_step:.1f} MW</b>.</div>", unsafe_allow_html=True)
            else:
                 st.markdown(f"<div class='rec-box'><b>Option B: Tune BESS Response</b><br>"
                            f"The BESS is big enough but might be responding too slowly.<br>"
                            f"👉 Try reducing BESS Response Time (current: {bess_response} ms).</div>", unsafe_allow_html=True)

    except Exception as e:
        st.error(f"Error: {str(e)}")
