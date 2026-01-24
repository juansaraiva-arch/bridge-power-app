import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint

# --- PAGE CONFIG ---
st.set_page_config(page_title="CAT Hybrid Optimizer v5.3 (Auto-Size)", page_icon="🤖", layout="wide")

# --- CSS ---
st.markdown("""
<style>
    .section-header { font-size: 16px; font-weight: bold; color: #444; margin-top: 10px; margin-bottom: 5px; border-bottom: 1px solid #ddd; }
    .success-box { background-color: #d4edda; padding: 10px; border-radius: 5px; border-left: 5px solid #28a745; }
    .fail-box { background-color: #f8d7da; padding: 10px; border-radius: 5px; border-left: 5px solid #dc3545; }
    .opt-box { background-color: #e3f2fd; padding: 15px; border-radius: 5px; border-left: 5px solid #0d47a1; }
    .metric-value { font-size: 24px; font-weight: bold; }
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
    st.title("Inputs v5.3")
    st.caption("Auto-Sizing Enabled")
    
    # --- TECHNICAL ---
    with st.expander("1. Project & Load", expanded=True):
        p_it = st.number_input("IT Load (MW)", 1.0, 5000.0, 100.0) 
        dc_aux = st.number_input("Auxiliaries (%)", 0.0, 50.0, 15.0)
        base_load_pct = st.number_input("Base Load (%)", 10.0, 100.0, 50.0)
        step_req_pct = st.number_input("AI Spike (%)", 0.0, 100.0, 40.0)
        pulse_duration = 5.0 

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
        
        # --- NUEVA LÓGICA: AUTO SIZE ---
        auto_size_bess = st.checkbox("Auto-Size BESS for Optimization", value=True, help="If checked, Optimizer ignores the manual slider below and searches for the best size.")
        
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
    
    net_load_for_gov = p_elec_dev - p_bess_dev
    target_mech = net_load_for_gov
    d_pmech_dt = (target_mech - p_mech_dev) / Tau_g
    
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
    
    if gen_cap_mw == 0: lf = 1.0 
    else: lf = load_mw_base / gen_cap_mw
    
    hr_penalty = 1.0 + 1.0 * ((1.0 - max(0.1, lf))**3) 
    hr_actual = specs['hr'] * hr_penalty
    
    fuel_cost_hr = load_mw_base * hr_actual * fuel_price 
    fuel_annual = fuel_cost_hr * op_hours
    
    om_annual = total_capex * 0.02
    total_mwh_year = load_mw_base * op_hours
    lcoe = (capex_annual + fuel_annual + om_annual) / total_mwh_year
    
    if not sim_stable:
        violation = max(0, nadir_limit - nadir_val) 
        lcoe += 10000 + (violation * 1000)
        
    return lcoe

# ==============================================================================
# 4. ROBUST OPTIMIZATION LOOP (AUTO-SIZE)
# ==============================================================================

def run_optimization_robust():
    progress_text = "Running Auto-Sizing Optimization..."
    my_bar = st.progress(0, text=progress_text)
    
    # 1. GENERATOR RANGE (Base Load -> Peak + Margin)
    mw_base = p_gross_total * (base_load_pct/100.0)
    mw_peak = p_gross_total * ((base_load_pct + step_req_pct)/100.0)
    
    n_min_base = int(np.ceil(mw_base / specs['mw']))
    n_min_peak = int(np.ceil(mw_peak / specs['mw']))
    
    n_start = max(1, n_min_base) 
    n_end = int(n_min_peak * 1.3) + 2
    n_range = range(n_start, n_end + 1)
    
    # 2. BESS RANGE (SMART LOGIC)
    # If Auto-Size: Search from 0 to 1.5x the Spike Magnitude
    # If Manual: Use only the user value (0 or Manual)
    
    if auto_size_bess and enable_bess:
        bess_max_search = step_mw * 1.5 
        # Create a grid of ~8 points to keep it fast
        b_range = np.linspace(0, bess_max_search, 8)
    elif enable_bess:
        b_range = [0.0, bess_cap_manual] # Compare None vs Manual
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
            
            # Physics
            t = np.linspace(0, 10, 200)
            try:
                sol = odeint(system_dynamics, [0,0,0], t, args=(sim_p,))
                freq = 60.0 + sol[:, 0]
                nadir = np.min(freq)
                peak = np.max(freq)
                is_stable = (nadir >= nadir_limit) and (peak <= overshoot_limit)
            except:
                is_stable = False; nadir=0; peak=99
            
            # Economics
            cost = calculate_lcoe(n, b_mw, is_stable, nadir)
            lcoe_clean = cost
            if not is_stable:
                 lcoe_clean = cost - (10000 + (max(0, nadir_limit - nadir) * 1000))
            
            # Capacity Hard Check
            total_cap = (n * specs['mw']) + b_mw
            if total_cap < mw_peak:
                is_stable = False
                cost += 5000
            
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
# 5. UI
# ==============================================================================

st.title("⚡ CAT Hybrid Optimizer v5.3")
st.markdown("**Status:** Auto-Sizing Enabled. Finding the economic sweet spot.")

tab1, tab2 = st.tabs(["🚀 Single Simulation", "💰 Global Optimization"])

with tab1:
    bess_val_tab1 = bess_cap_manual if enable_bess else 0.0
    if st.button("Run Single Check", type="primary"):
        sys_mva = (n_gens_op * specs['mw']) / 0.8
        sim_params = {
            'H': h_const, 'Tau_g': tau_gov, 'Tau_b': bess_response / 1000.0,
            'P_spike': step_mw, 'P_bess_cap': bess_val_tab1, 'Sys_MVA': sys_mva,
            'Bess_Enabled': enable_bess, 'Duration': pulse_duration
        }
        t = np.linspace(0, 15, 1000)
        sol = odeint(system_dynamics, [0,0,0], t, args=(sim_params,))
        freq = 60.0 + sol[:, 0]
        nadir = np.min(freq)
        
        c1, c2 = st.columns([3, 1])
        with c1:
            fig, ax = plt.subplots(figsize=(10, 4))
            ax.plot(t, freq, label="Freq"); ax.axhline(nadir_limit, color='r', linestyle='--')
            ax.set_title(f"Nadir: {nadir:.3f} Hz"); ax.grid(True, alpha=0.3); st.pyplot(fig)
        with c2:
            is_stable = (nadir >= nadir_limit)
            lcoe = calculate_lcoe(n_gens_op, bess_val_tab1, is_stable, nadir)
            real_lcoe = lcoe if is_stable else lcoe - (10000 + (max(0, nadir_limit - nadir) * 1000))
            st.metric("LCOE", f"${real_lcoe:.2f}/MWh")
            if is_stable: st.success("STABLE")
            else: st.error("TRIP")

with tab2:
    if st.button("🔎 Run Optimizer"):
        df_opt = run_optimization_robust()
        df_viable = df_opt[df_opt['Stable'] == True].sort_values("LCOE_Penalized")
        
        if not df_viable.empty:
            best = df_viable.iloc[0]
            st.markdown(f"### ✅ Optimal Found")
            
            c1, c2, c3 = st.columns(3)
            c1.metric("Optimal Generators", f"{int(best['Gens'])} Units", f"{gen_model}")
            c2.metric("Optimal BESS", f"{best['BESS_MW']:.1f} MW", "Peak Shaving")
            c3.metric("Lowest LCOE", f"${best['LCOE']:.2f}/MWh", "Optimized")
            
            # Battle Card
            df_gen_only = df_opt[(df_opt['BESS_MW'] == 0) & (df_opt['Stable'] == True)].sort_values("LCOE_Penalized")
            st.write("---")
            st.subheader("⚔️ Hybrid vs. Traditional")
            
            if not df_gen_only.empty:
                gen_best = df_gen_only.iloc[0]
                diff = gen_best['LCOE'] - best['LCOE']
                
                col_a, col_b = st.columns(2)
                with col_a:
                    st.info(f"**Traditional:** {int(gen_best['Gens'])} Gens @ ${gen_best['LCOE']:.2f}/MWh")
                with col_b:
                    st.success(f"**Hybrid:** {int(best['Gens'])} Gens + {best['BESS_MW']:.1f}MW BESS @ ${best['LCOE']:.2f}/MWh\n\n**Savings:** {diff:.2f} $/MWh")
            else:
                st.warning("Only Hybrid configurations can stabilize this load.")
            
            # Map
            fig, ax = plt.subplots(figsize=(10, 6))
            stable = df_opt[df_opt['Stable']]
            unstable = df_opt[~df_opt['Stable']]
            ax.scatter(unstable['Gens'], unstable['BESS_MW'], c='lightgray', marker='x', alpha=0.5, label='Trip')
            sc = ax.scatter(stable['Gens'], stable['BESS_MW'], c=stable['LCOE'], cmap='viridis_r', s=100, edgecolors='k')
            plt.colorbar(sc, label='LCOE ($/MWh)')
            ax.scatter(best['Gens'], best['BESS_MW'], c='red', s=300, marker='*', label='Optimal')
            ax.set_xlabel("Generators"); ax.set_ylabel("BESS (MW)"); ax.legend(); st.pyplot(fig)
        else:
            st.error("No stable configuration found.")
