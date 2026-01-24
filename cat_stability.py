import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint

# --- PAGE CONFIG ---
st.set_page_config(page_title="CAT Hybrid Optimizer v5.1 (Resilient)", page_icon="💎", layout="wide")

# --- CSS ---
st.markdown("""
<style>
    .section-header { font-size: 16px; font-weight: bold; color: #444; margin-top: 10px; margin-bottom: 5px; border-bottom: 1px solid #ddd; }
    .success-box { background-color: #d4edda; padding: 10px; border-radius: 5px; border-left: 5px solid #28a745; }
    .fail-box { background-color: #f8d7da; padding: 10px; border-radius: 5px; border-left: 5px solid #dc3545; }
    .warn-box { background-color: #fff3cd; padding: 10px; border-radius: 5px; border-left: 5px solid #ffc107; }
    .opt-box { background-color: #cce5ff; padding: 15px; border-radius: 5px; border-left: 5px solid #004085; }
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
    st.title("Inputs v5.1 (Resilient)")
    
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

    with st.expander("3. BESS Config"):
        enable_bess = st.checkbox("Enable BESS", value=True)
        step_mw = p_it * (step_req_pct/100.0)
        bess_cap = st.number_input("BESS Power (MW)", 0.0, 5000.0, step_mw) if enable_bess else 0.0
        bess_response = st.number_input("Response (ms)", 10, 2000, 50)

    # --- ECONOMIC ---
    with st.expander("4. Economics (LCOE)", expanded=True):
        fuel_price = st.number_input("Fuel Price ($/MMBtu)", 1.0, 50.0, 4.0)
        op_hours = st.number_input("Op Hours/Year", 100, 8760, 2000)
        
        capex_gen_kwe = st.number_input("Gen CAPEX ($/kW)", 100, 2000, specs['capex'])
        capex_bess_kwh = st.number_input("BESS Energy ($/kWh)", 100, 1000, 300)
        capex_bess_kw = st.number_input("BESS Power ($/kW)", 50, 500, 150)
        
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
    
    target_mech = p_elec_dev 
    d_pmech_dt = (target_mech - p_mech_dev) / Tau_g
    
    target_bess = min(P_bess_cap, p_elec_dev) if params['Bess_Enabled'] else 0.0
    d_pbess_dt = (target_bess - p_bess_dev) / Tau_b
    
    p_acc_mw = p_mech_dev + p_bess_dev - p_elec_dev
    f0 = 60.0
    d_freq_dt = (p_acc_mw / Sys_MVA) * (f0 / (2 * H))
    
    return [d_freq_dt, d_pmech_dt, d_pbess_dt]

def calculate_lcoe(n_gens, bess_mw, sim_stable, nadir_val):
    # If unstable, we penalize but STILL calculate LCOE to show "what if" or trends
    # We add a massive penalty to the LCOE value itself if unstable
    
    # 1. CAPEX Annualized
    gen_cap_mw = n_gens * specs['mw']
    capex_gens = gen_cap_mw * 1000 * capex_gen_kwe
    capex_bess = (bess_mw * 1000 * capex_bess_kw) + (bess_mw * bess_dur_hours * 1000 * capex_bess_kwh)
    
    total_capex = capex_gens + capex_bess
    crf = (discount_rate * (1+discount_rate)**project_years) / ((1+discount_rate)**project_years - 1)
    capex_annual = total_capex * crf
    
    # 2. Fuel Cost (Non-Linear Efficiency)
    load_mw = p_gross_total * (base_load_pct/100.0)
    lf = load_mw / max(0.1, gen_cap_mw)
    
    hr_penalty = 1.0 + 0.8 * ((1.0 - max(0.1, lf))**2) # Quadratic penalty for low load
    hr_actual = specs['hr'] * hr_penalty
    
    fuel_cost_hr = load_mw * hr_actual * fuel_price 
    fuel_annual = fuel_cost_hr * op_hours
    
    # 3. O&M
    om_annual = total_capex * 0.02
    
    total_mwh_year = load_mw * op_hours
    lcoe = (capex_annual + fuel_annual + om_annual) / total_mwh_year
    
    # Penalty Logic for Optimization Sorting
    if not sim_stable:
        # Penalize proportional to how bad the failure was
        # This allows sorting "bad" solutions to find the "least bad"
        violation = max(0, nadir_limit - nadir_val) 
        lcoe += 1000 + (violation * 1000) 
        
    return lcoe

# ==============================================================================
# 4. ROBUST OPTIMIZATION LOOP
# ==============================================================================

def run_optimization_robust():
    progress_text = "Running Smart Grid Search..."
    my_bar = st.progress(0, text=progress_text)
    
    # --- SMART RANGES ---
    # 1. Generators: 
    # Start: Need enough to cover Base Load
    mw_base = p_gross_total * (base_load_pct/100.0)
    mw_peak_demand = p_gross_total * ((base_load_pct + step_req_pct)/100.0)
    
    n_min_base = int(np.ceil(mw_base / specs['mw']))
    n_min_peak = int(np.ceil(mw_peak_demand / specs['mw']))
    
    # Range: From Base coverage up to Peak + 50% Margin (for inertia)
    # This ensures we search DEEP into high-inertia configurations
    n_start = n_min_base
    n_end = int(n_min_peak * 1.5) + 5
    n_step = max(1, int((n_end - n_start) / 10)) # Limit resolution to avoid timeout
    
    n_range = range(n_start, n_end, n_step)
    
    # 2. BESS:
    # 0 to 120% of Spike
    bess_step_mw = max(1.0, step_mw / 4.0) # 5 steps of BESS
    b_range = np.arange(0, step_mw * 1.25, bess_step_mw)
    
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
            # Fast sim
            t = np.linspace(0, 10, 400)
            try:
                sol = odeint(system_dynamics, [0,0,0], t, args=(sim_p,))
                freq = 60.0 + sol[:, 0]
                nadir = np.min(freq)
                peak = np.max(freq)
                is_stable = (nadir >= nadir_limit) and (peak <= overshoot_limit)
            except:
                is_stable = False; nadir=0; peak=99
            
            cost = calculate_lcoe(n, b_mw, is_stable, nadir)
            
            # Remove the artificial penalty for the raw "LCOE_Clean" metric display
            lcoe_clean = cost
            if not is_stable:
                 lcoe_clean = cost - (1000 + (max(0, nadir_limit - nadir) * 1000))

            results.append({
                "Gens": n, "BESS_MW": b_mw, "Stable": is_stable, 
                "LCOE_Penalized": cost, "LCOE": lcoe_clean,
                "Nadir": nadir, "Peak": peak
            })
            
            iter_count += 1
            if iter_count % 5 == 0:
                my_bar.progress(min(100, int(iter_count/total_iters * 100)), text=f"Simulating: {n} Gens...")
            
    my_bar.empty()
    return pd.DataFrame(results)

# ==============================================================================
# 5. UI
# ==============================================================================

st.title("⚡ CAT Hybrid Optimizer v5.1")
st.markdown("**Objective:** Find the lowest LCOE that meets Stability limits.")

tab1, tab2 = st.tabs(["🚀 Single Simulation", "💰 Global Optimization"])

# --- TAB 1 ---
with tab1:
    if st.button("Run Single Check", type="primary"):
        sys_mva = (n_gens_op * specs['mw']) / 0.8
        sim_params = {
            'H': h_const, 'Tau_g': tau_gov, 'Tau_b': bess_response / 1000.0,
            'P_spike': step_mw, 'P_bess_cap': bess_cap, 'Sys_MVA': sys_mva,
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
            lcoe = calculate_lcoe(n_gens_op, bess_cap if enable_bess else 0, is_stable, nadir)
            # Remove penalty for display
            real_lcoe = lcoe if is_stable else lcoe - (1000 + (max(0, nadir_limit - nadir) * 1000))
            st.metric("LCOE", f"${real_lcoe:.2f}/MWh")
            if is_stable: st.success("STABLE")
            else: st.error("TRIP")

# --- TAB 2 ---
with tab2:
    st.info("Searching across Generator counts and BESS sizes...")
    
    if st.button("🔎 Run Optimizer"):
        df_opt = run_optimization_robust()
        
        df_viable = df_opt[df_opt['Stable'] == True].sort_values("LCOE_Penalized")
        
        # --- BEST SOLUTION FINDER ---
        if not df_viable.empty:
            # PERFECT CASE: Found stable solutions
            best = df_viable.iloc[0]
            status_msg = "✅ Optimal Configuration Found"
            status_color = "success-box"
        else:
            # FALLBACK CASE: No stable solution found
            # Pick the "Least Bad" (Lowest LCOE among those closest to stability)
            # Sort by Nadir (descending) to find closest to limit
            df_opt_sorted = df_opt.sort_values("Nadir", ascending=False)
            best = df_opt_sorted.iloc[0]
            status_msg = f"⚠️ Unstable (Best Effort). Freq reached {best['Nadir']:.2f} Hz (Limit {nadir_limit} Hz)."
            status_color = "warn-box"

        # --- DISPLAY RESULTS ---
        st.markdown(f"### {status_msg}")
        
        c1, c2, c3 = st.columns(3)
        c1.markdown(f"""
        <div class="opt-box">
            <div style="font-size:14px">GENERATORS</div>
            <div style="font-size:28px; font-weight:bold">{int(best['Gens'])} Units</div>
            <div style="font-size:12px">{gen_model}</div>
        </div>""", unsafe_allow_html=True)
        
        c2.markdown(f"""
        <div class="opt-box">
            <div style="font-size:14px">BESS SIZE</div>
            <div style="font-size:28px; font-weight:bold">{best['BESS_MW']:.1f} MW</div>
            <div style="font-size:12px">Response: {bess_response} ms</div>
        </div>""", unsafe_allow_html=True)
        
        c3.markdown(f"""
        <div class="{status_color}">
            <div style="font-size:14px">LCOE EST.</div>
            <div style="font-size:28px; font-weight:bold">${best['LCOE']:.2f}</div>
            <div style="font-size:12px">per MWh</div>
        </div>""", unsafe_allow_html=True)
        
        st.write("---")
        
        # --- BATTLE CARD (Compare to Gen Only) ---
        # Find best "Gen Only" scenario (BESS=0)
        df_gen_only = df_opt[df_opt['BESS_MW'] == 0].sort_values("LCOE_Penalized")
        if not df_gen_only.empty:
            best_gen = df_gen_only.iloc[0]
            
            st.subheader("⚔️ Comparison: Hybrid vs. Gen-Only")
            
            col_l, col_r = st.columns(2)
            with col_l:
                st.markdown(f"**Best Hybrid:** {int(best['Gens'])} Gens + {best['BESS_MW']:.1f} MW BESS")
                st.write(f"LCOE: **${best['LCOE']:.2f}** | Nadir: {best['Nadir']:.2f} Hz")
            with col_r:
                st.markdown(f"**Best Gen-Only:** {int(best_gen['Gens'])} Gens")
                st.write(f"LCOE: **${best_gen['LCOE']:.2f}** | Nadir: {best_gen['Nadir']:.2f} Hz")
                if not best_gen['Stable']:
                    st.caption("⚠️ Gen-Only option is UNSTABLE in this range.")
        
        # --- HEATMAP ---
        st.markdown("### 🗺️ Landscape Analysis")
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Plot unstable points
        unstable = df_opt[~df_opt['Stable']]
        if not unstable.empty:
            ax.scatter(unstable['Gens'], unstable['BESS_MW'], c='lightgray', marker='x', alpha=0.5, label='Unstable')
            
        # Plot stable points
        stable = df_opt[df_opt['Stable']]
        if not stable.empty:
            sc = ax.scatter(stable['Gens'], stable['BESS_MW'], c=stable['LCOE'], cmap='viridis_r', s=100, edgecolors='k', label='Stable')
            plt.colorbar(sc, ax=ax, label='LCOE ($/MWh)')
            
        # Highlight Selected
        ax.scatter(best['Gens'], best['BESS_MW'], c='red', s=200, marker='*', label='Selected', zorder=10)
        
        ax.set_xlabel("Generators (N)")
        ax.set_ylabel("BESS (MW)")
        ax.legend()
        ax.grid(True, alpha=0.3)
        st.pyplot(fig)
