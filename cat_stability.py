import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint

# --- PAGE CONFIG ---
st.set_page_config(page_title="CAT Hybrid Optimizer v5.5 (Stable UI)", page_icon="⚡", layout="wide")

# --- CSS (SCOPED TO AVOID CONFLICTS) ---
st.markdown("""
<style>
    /* Usamos prefijos 'cat-' para evitar conflictos con clases nativas de Streamlit */
    .cat-section-header { font-size: 16px; font-weight: bold; color: #444; margin-top: 10px; margin-bottom: 5px; border-bottom: 1px solid #ddd; }
    
    .cat-box-success { 
        background-color: #d4edda; 
        padding: 15px; 
        border-radius: 8px; 
        border-left: 5px solid #28a745; 
        color: #155724;
        margin-bottom: 10px;
    }
    
    .cat-box-fail { 
        background-color: #f8d7da; 
        padding: 15px; 
        border-radius: 8px; 
        border-left: 5px solid #dc3545; 
        color: #721c24;
        margin-bottom: 10px;
    }
    
    .cat-box-opt { 
        background-color: #e3f2fd; 
        padding: 15px; 
        border-radius: 8px; 
        border-left: 5px solid #0d47a1; 
        color: #084298;
    }
    
    /* Estabiliza las métricas */
    div[data-testid="stMetricValue"] {
        font-size: 28px !important;
        font-weight: bold !important;
    }
</style>
""", unsafe_allow_html=True)

# ==============================================================================
# 1. DATA LIBRARY
# ==============================================================================
CAT_LIBRARY = {
    "XGC1900 (1.9 MW)":   {"mw": 1.9,   "type": "Recip", "h_def": 1.0, "tau_def": 0.5, "hr": 8.78, "capex": 800},
    "G3520FR (2.5 MW)":   {"mw": 2.5,   "type": "Recip", "h_def": 1.0, "tau_def": 0.6, "hr": 8.83, "capex": 600},
    "G3520K (2.4 MW)":    {"mw": 2.4,   "type": "Recip", "h_def": 1.0, "tau_def": 0.6, "hr": 7.64, "capex": 600},
    "CG260 (3.96 MW)":    {"mw": 3.96,  "type": "Recip", "h_def": 1.2, "tau_def": 0.8, "hr": 7.86, "capex": 700},
    "G20CM34 (9.76 MW)":  {"mw": 9.76,  "type": "Recip", "h_def": 1.5, "tau_def": 1.0, "hr": 7.48, "capex": 750},
    "Titan 130 (16.5 MW)":{"mw": 16.5,  "type": "Turbine","h_def": 4.0,"tau_def": 1.5, "hr": 9.63, "capex": 800},
    "Titan 250 (23.2 MW)":{"mw": 23.2,  "type": "Turbine","h_def": 4.5,"tau_def": 2.0, "hr": 8.67, "capex": 800},
    "Titan 350 (38.0 MW)":{"mw": 38.0,  "type": "Turbine","h_def": 5.0,"tau_def": 2.5, "hr": 8.50, "capex": 800}
}

# ==============================================================================
# 2. INPUT SIDEBAR
# ==============================================================================

with st.sidebar:
    st.title("Inputs v5.5")
    
    # --- TECHNICAL ---
    with st.expander("1. Project & Load", expanded=True):
        p_it = st.number_input("IT Load (MW)", 1.0, 5000.0, 100.0) 
        dc_aux = st.number_input("Auxiliaries (%)", 0.0, 50.0, 15.0)
        base_load_pct = st.number_input("Base Load (%)", 10.0, 100.0, 50.0)
        step_req_pct = st.number_input("AI Spike (%)", 0.0, 100.0, 40.0)
        
        st.markdown("---")
        load_type = st.radio("Load Pattern", ["Single Pulse (Inference)", "Step (Training)"])
        if load_type == "Single Pulse (Inference)":
            pulse_duration = st.number_input("Pulse Duration (s)", 0.5, 60.0, 5.0)
        else:
            pulse_duration = 9999.0 

    with st.expander("2. Generation Fleet"):
        gen_model = st.selectbox("Generator Model", list(CAT_LIBRARY.keys()))
        specs = CAT_LIBRARY[gen_model]
        
        p_gross_total = p_it * (1 + dc_aux/100.0)
        n_rec = int(np.ceil(p_gross_total / specs['mw'])) + 1
        n_gens_op = st.number_input("Operating Units (N)", 1, 5000, n_rec)
        
        h_const = st.number_input("Inertia H (s)", 0.1, 20.0, float(specs['h_def']))
        tau_gov = st.number_input("Gov Lag (s)", 0.05, 10.0, float(specs.get('tau_def', 0.5)))

    with st.expander("3. BESS Config", expanded=True):
        enable_bess = st.checkbox("Enable BESS", value=True)
        step_mw = p_it * (step_req_pct/100.0)
        
        auto_size_bess = st.checkbox("Auto-Size BESS (Optimizer)", value=True)
        
        if enable_bess:
            bess_cap_manual = st.number_input("Manual BESS Power (MW)", 0.0, 5000.0, step_mw, disabled=auto_size_bess)
            bess_response = st.number_input("Response (ms)", 10, 2000, 50)
        else:
            bess_cap_manual = 0.0
            bess_response = 1000

    # --- ECONOMIC ---
    with st.expander("4. Economics (LCOE)", expanded=True):
        fuel_price = st.number_input("Fuel Price ($/MMBtu)", 1.0, 50.0, 6.0)
        op_hours = st.number_input("Op Hours/Year", 100, 8760, 3000)
        
        capex_gen_kwe = st.number_input("Gen CAPEX ($/kW)", 100, 2000, specs['capex'])
        capex_bess_kwh = st.number_input("BESS Energy ($/kWh)", 100, 1000, 250)
        capex_bess_kw = st.number_input("BESS Power ($/kW)", 50, 500, 100)
        
        bess_dur_hours = 0.5 
        discount_rate = 0.08
        project_years = 10

    with st.expander("5. Limits"):
        nadir_limit = st.number_input("Min Freq (Hz)", 50.0, 59.9, 57.0)
        overshoot_limit = st.number_input("Max Freq (Hz)", 60.1, 70.0, 63.0)

# ==============================================================================
# 3. ENGINES
# ==============================================================================

def system_dynamics(y, t, params):
    freq_dev, p_mech_dev, p_bess_dev = y
    H = max(params['H'], 0.1); Sys_MVA = max(params['Sys_MVA'], 0.1)
    Tau_g = max(params['Tau_g'], 0.05); Tau_b = max(params['Tau_b'], 0.02)
    P_spike = params['P_spike']; P_bess_cap = params['P_bess_cap']; Duration = params['Duration']
    
    p_elec_dev = P_spike if 1.0 <= t < (1.0 + Duration) else 0.0
    
    target_bess = 0.0
    if params['Bess_Enabled']:
        target_bess = min(P_bess_cap, p_elec_dev)
    d_pbess_dt = (target_bess - p_bess_dev) / Tau_b
    
    net_load = p_elec_dev - p_bess_dev
    d_pmech_dt = (net_load - p_mech_dev) / Tau_g
    
    p_acc_mw = p_mech_dev + p_bess_dev - p_elec_dev
    f0 = 60.0
    d_freq_dt = (p_acc_mw / Sys_MVA) * (f0 / (2 * H))
    
    return [d_freq_dt, d_pmech_dt, d_pbess_dt]

def calculate_lcoe(n_gens, bess_mw, sim_stable, nadir_val):
    gen_cap_mw = n_gens * specs['mw']
    capex_gens = gen_cap_mw * 1000 * capex_gen_kwe
    capex_bess = (bess_mw * 1000 * capex_bess_kw) + (bess_mw * bess_dur_hours * 1000 * capex_bess_kwh)
    
    total_capex = capex_gens + capex_bess
    crf = (discount_rate * (1+discount_rate)**project_years) / ((1+discount_rate)**project_years - 1)
    capex_annual = total_capex * crf
    
    load_mw_base = p_gross_total * (base_load_pct/100.0)
    lf = load_mw_base / max(0.1, gen_cap_mw)
    hr_penalty = 1.0 + 1.0 * ((1.0 - max(0.1, lf))**3) 
    hr_actual = specs['hr'] * hr_penalty
    fuel_annual = load_mw_base * hr_actual * fuel_price * op_hours
    
    om_annual = total_capex * 0.02
    total_mwh_year = load_mw_base * op_hours
    
    lcoe_mwh = (capex_annual + fuel_annual + om_annual) / total_mwh_year
    lcoe_kwh = lcoe_mwh / 1000.0
    
    if not sim_stable:
        violation = max(0, nadir_limit - nadir_val) 
        lcoe_kwh += 10.0 + (violation * 1.0)
        
    return lcoe_kwh

# ==============================================================================
# 4. OPTIMIZATION LOOP
# ==============================================================================

def run_optimization_robust():
    progress_text = "Running Auto-Sizing Optimization..."
    my_bar = st.progress(0, text=progress_text)
    
    mw_base = p_gross_total * (base_load_pct/100.0)
    mw_peak = p_gross_total * ((base_load_pct + step_req_pct)/100.0)
    
    n_min_base = int(np.ceil(mw_base / specs['mw']))
    n_min_peak = int(np.ceil(mw_peak / specs['mw']))
    
    # Search Range
    n_start = max(1, n_min_base) 
    n_end = int(n_min_peak * 1.3) + 2
    n_range = range(n_start, n_end + 1)
    
    if auto_size_bess and enable_bess:
        bess_max_search = step_mw * 1.5 
        b_range = np.linspace(0, bess_max_search, 8)
    elif enable_bess:
        b_range = [0.0, bess_cap_manual]
    else:
        b_range = [0.0]
    
    results = []
    total_iters = len(n_range) * len(b_range)
    iter_count = 0
    
    for n in n_range:
        for b_mw in b_range:
            sys_mva = (n * specs['mw']) / 0.8
            sim_p = {
                'H': h_const, 'Tau_g': tau_gov, 'Tau_b': bess_response/1000.0,
                'P_spike': step_mw, 'P_bess_cap': b_mw, 'Sys_MVA': sys_mva,
                'Bess_Enabled': True, 'Duration': pulse_duration
            }
            
            t = np.linspace(0, 10, 200)
            try:
                sol = odeint(system_dynamics, [0,0,0], t, args=(sim_p,))
                freq = 60.0 + sol[:, 0]
                nadir = np.min(freq)
                peak = np.max(freq)
                is_stable = (nadir >= nadir_limit) and (peak <= overshoot_limit)
            except:
                is_stable = False; nadir=0; peak=99
            
            cost = calculate_lcoe(n, b_mw, is_stable, nadir)
            lcoe_clean = cost
            if not is_stable:
                 lcoe_clean = cost - (10.0 + (max(0, nadir_limit - nadir) * 1.0))
            
            total_cap = (n * specs['mw']) + b_mw
            if total_cap < mw_peak:
                is_stable = False
                cost += 5.0
            
            results.append({
                "Gens": n, "BESS_MW": b_mw, "Stable": is_stable, 
                "LCOE_Penalized": cost, "LCOE": lcoe_clean, "Nadir": nadir
            })
            
            iter_count += 1
            if iter_count % 5 == 0:
                my_bar.progress(min(100, int(iter_count/total_iters * 100)))
            
    my_bar.empty()
    return pd.DataFrame(results)

# ==============================================================================
# 5. UI DISPLAY
# ==============================================================================

st.title("⚡ CAT Hybrid Optimizer v5.5")
st.markdown("**Status:** UI Stabilized. Physics & Economics Enabled.")

tab1, tab2 = st.tabs(["🚀 Single Simulation", "💰 Global Optimization"])

# --- TAB 1: PHYSICS VISUALIZATION ---
with tab1:
    bess_val_tab1 = bess_cap_manual if enable_bess else 0.0
    
    if st.button("Run Simulation Analysis", type="primary"):
        sys_mva = (n_gens_op * specs['mw']) / 0.8
        sim_params = {
            'H': h_const, 'Tau_g': tau_gov, 'Tau_b': bess_response / 1000.0,
            'P_spike': step_mw, 'P_bess_cap': bess_val_tab1, 'Sys_MVA': sys_mva,
            'Bess_Enabled': enable_bess, 'Duration': pulse_duration
        }
        
        t_end = max(15.0, pulse_duration + 5.0)
        t = np.linspace(0, t_end, 1500)
        
        sol = odeint(system_dynamics, [0,0,0], t, args=(sim_params,))
        freq = 60.0 + sol[:, 0]
        nadir = np.min(freq)
        
        mw_base = p_gross_total * (base_load_pct/100.0)
        p_mech = mw_base + sol[:, 1]
        p_bess = sol[:, 2]
        p_load = [mw_base + (step_mw if 1.0 <= ti < (1.0 + pulse_duration) else 0) for ti in t]
        
        # --- FIX: COLUMN WIDTH RATIO [2, 1] for Stability ---
        c1, c2 = st.columns([2, 1])
        
        with c1:
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
            
            # Ax1
            ax1.plot(t, freq, label="Freq", color='blue')
            ax1.axhline(nadir_limit, color='red', linestyle='--', label="Trip")
            if pulse_duration < 100:
                ax1.axvspan(1.0, 1.0+pulse_duration, color='gray', alpha=0.1)
            ax1.set_ylabel("Freq (Hz)"); ax1.grid(True, alpha=0.3); ax1.legend()
            ax1.set_title(f"Frequency Response (Nadir: {nadir:.2f} Hz)")
            
            # Ax2
            ax2.plot(t, p_load, 'k--', label="Total Load")
            ax2.plot(t, p_mech, 'g-', label="Gens (Mech)")
            if enable_bess:
                ax2.plot(t, p_bess, 'm-', label="BESS (Inj)")
            ax2.set_ylabel("Power (MW)"); ax2.set_xlabel("Time (s)")
            ax2.grid(True, alpha=0.3); ax2.legend()
            
            st.pyplot(fig)
            
        with c2:
            # --- FIX: USE CONTAINER TO PREVENT JITTER ---
            with st.container(border=True):
                st.subheader("Results")
                
                is_stable = (nadir >= nadir_limit)
                lcoe = calculate_lcoe(n_gens_op, bess_val_tab1, is_stable, nadir)
                real_lcoe = lcoe if is_stable else lcoe - (10.0 + (max(0, nadir_limit - nadir) * 1.0))
                
                # Standard Metrics (Stable Font)
                st.metric("LCOE", f"${real_lcoe:.3f}/kWh")
                st.metric("Nadir", f"{nadir:.3f} Hz")
                st.metric("Spike Load", f"{step_mw:.1f} MW")
                
                st.write("---")
                if is_stable: 
                    st.markdown('<div class="cat-box-success">✅ <b>SYSTEM STABLE</b></div>', unsafe_allow_html=True)
                else: 
                    st.markdown('<div class="cat-box-fail">❌ <b>SYSTEM TRIP</b><br>Under-Frequency Event</div>', unsafe_allow_html=True)
                    if not enable_bess:
                        st.info("💡 Tip: Enable BESS to absorb the load spike.")

# --- TAB 2: OPTIMIZER ---
with tab2:
    if st.button("🔎 Run Optimizer"):
        df_opt = run_optimization_robust()
        df_viable = df_opt[df_opt['Stable'] == True].sort_values("LCOE_Penalized")
        
        if not df_viable.empty:
            best = df_viable.iloc[0]
            
            # --- RESULTS LAYOUT ---
            st.markdown(f"### ✅ Optimal Configuration Found")
            
            col_res1, col_res2, col_res3 = st.columns(3)
            with col_res1:
                st.markdown(f"""
                <div class="cat-box-opt">
                    <div style="font-size:14px">OPTIMAL GENERATORS</div>
                    <div style="font-size:24px; font-weight:bold">{int(best['Gens'])} Units</div>
                    <div style="font-size:12px">{gen_model}</div>
                </div>""", unsafe_allow_html=True)
            
            with col_res2:
                st.markdown(f"""
                <div class="cat-box-opt">
                    <div style="font-size:14px">OPTIMAL BESS</div>
                    <div style="font-size:24px; font-weight:bold">{best['BESS_MW']:.1f} MW</div>
                    <div style="font-size:12px">Peak Shaving</div>
                </div>""", unsafe_allow_html=True)
                
            with col_res3:
                st.markdown(f"""
                <div class="cat-box-success">
                    <div style="font-size:14px">LOWEST LCOE</div>
                    <div style="font-size:24px; font-weight:bold">${best['LCOE']:.3f}</div>
                    <div style="font-size:12px">per kWh</div>
                </div>""", unsafe_allow_html=True)
            
            # Battle Card
            df_gen_only = df_opt[(df_opt['BESS_MW'] == 0) & (df_opt['Stable'] == True)].sort_values("LCOE_Penalized")
            st.write("---")
            
            if not df_gen_only.empty:
                gen_best = df_gen_only.iloc[0]
                diff = gen_best['LCOE'] - best['LCOE']
                st.info(f"**Traditional:** {int(gen_best['Gens'])} Gens @ ${gen_best['LCOE']:.3f}/kWh")
                st.success(f"**Hybrid:** {int(best['Gens'])} Gens + {best['BESS_MW']:.1f}MW BESS @ ${best['LCOE']:.3f}/kWh\n\n**Savings:** ${diff:.3f}/kWh")
            else:
                st.warning("Only Hybrid configurations can stabilize this load.")
            
            # Map
            fig, ax = plt.subplots(figsize=(10, 6))
            stable = df_opt[df_opt['Stable']]
            unstable = df_opt[~df_opt['Stable']]
            ax.scatter(unstable['Gens'], unstable['BESS_MW'], c='lightgray', marker='x', alpha=0.5, label='Trip')
            sc = ax.scatter(stable['Gens'], stable['BESS_MW'], c=stable['LCOE'], cmap='viridis_r', s=100, edgecolors='k')
            plt.colorbar(sc, label='LCOE ($/kWh)')
            ax.scatter(best['Gens'], best['BESS_MW'], c='red', s=300, marker='*', label='Optimal')
            ax.set_xlabel("Generators"); ax.set_ylabel("BESS (MW)"); ax.legend(); st.pyplot(fig)
        else:
            st.error("No stable configuration found.")
