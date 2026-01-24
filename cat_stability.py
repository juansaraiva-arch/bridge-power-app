import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint

# --- PAGE CONFIG ---
st.set_page_config(page_title="CAT Hybrid Optimizer v5.0", page_icon="💰", layout="wide")

# --- CSS ---
st.markdown("""
<style>
    .section-header { font-size: 16px; font-weight: bold; color: #444; margin-top: 10px; margin-bottom: 5px; border-bottom: 1px solid #ddd; }
    .success-box { background-color: #d4edda; padding: 10px; border-radius: 5px; border-left: 5px solid #28a745; }
    .fail-box { background-color: #f8d7da; padding: 10px; border-radius: 5px; border-left: 5px solid #dc3545; }
    .opt-box { background-color: #cce5ff; padding: 15px; border-radius: 5px; border-left: 5px solid #004085; }
    .metric-value { font-size: 24px; font-weight: bold; }
</style>
""", unsafe_allow_html=True)

# ==============================================================================
# 1. DATA LIBRARY (With Heat Rates from Excel)
# ==============================================================================
# Added 'heat_rate' (MMBtu/MWh) and 'capex' ($/kW est)
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
# 2. INPUT SIDEBAR (Economic & Technical)
# ==============================================================================

with st.sidebar:
    st.title("Inputs v5.0 (LCOE Optimizer)")
    
    # --- TECHNICAL ---
    with st.expander("1. Project & Load", expanded=True):
        p_it = st.number_input("IT Load (MW)", 1.0, 2000.0, 100.0) 
        dc_aux = st.number_input("Auxiliaries (%)", 0.0, 50.0, 15.0)
        base_load_pct = st.number_input("Base Load (%)", 10.0, 100.0, 50.0)
        step_req_pct = st.number_input("AI Spike (%)", 0.0, 100.0, 40.0)
        
        # Pulse defaults
        pulse_duration = 5.0 # Seconds

    with st.expander("2. Generation Fleet"):
        gen_model = st.selectbox("Generator Model", list(CAT_LIBRARY.keys()))
        specs = CAT_LIBRARY[gen_model]
        
        p_gross_total = p_it * (1 + dc_aux/100.0)
        n_rec = int(np.ceil(p_gross_total / specs['mw'])) + 1
        n_gens_op = st.number_input("Operating Units (N)", 1, 1000, n_rec)
        
        h_const = st.number_input("Inertia H (s)", 0.1, 20.0, float(specs['h_def']))
        tau_gov = st.number_input("Gov Lag (s)", 0.05, 10.0, float(specs.get('tau_def', 0.5)))

    with st.expander("3. BESS Config"):
        enable_bess = st.checkbox("Enable BESS", value=True)
        step_mw = p_it * (step_req_pct/100.0)
        bess_cap = st.number_input("BESS Power (MW)", 0.0, 2000.0, step_mw) if enable_bess else 0.0
        bess_response = st.number_input("Response (ms)", 10, 2000, 50)

    # --- ECONOMIC ---
    with st.expander("4. Economics (LCOE)", expanded=True):
        st.markdown('<div class="section-header">Fuel & OPEX</div>', unsafe_allow_html=True)
        fuel_price = st.number_input("Fuel Price ($/MMBtu)", 1.0, 50.0, 4.0, help="NatGas Price")
        op_hours = st.number_input("Op Hours/Year", 100, 8760, 2000, help="Hours running in Island Mode")
        
        st.markdown('<div class="section-header">CAPEX Assumptions</div>', unsafe_allow_html=True)
        capex_gen_kwe = st.number_input("Gen CAPEX ($/kW)", 100, 2000, specs['capex'])
        capex_bess_kwh = st.number_input("BESS Energy ($/kWh)", 100, 1000, 300)
        capex_bess_kw = st.number_input("BESS Power ($/kW)", 50, 500, 150)
        
        # BESS Duration for CAPEX calc
        bess_dur_hours = 0.5 # Assumed for stability/bridge BESS
        
        discount_rate = 0.08 # 8% WACC
        project_years = 10

    with st.expander("5. Limits"):
        nadir_limit = st.number_input("Min Freq (Hz)", 50.0, 59.9, 57.0)
        overshoot_limit = st.number_input("Max Freq (Hz)", 60.1, 70.0, 63.0)

# ==============================================================================
# 3. PHYSICS & ECONOMICS ENGINES
# ==============================================================================

def system_dynamics(y, t, params):
    # (Same robust ODE from v3/4)
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

def calculate_lcoe(n_gens, bess_mw, sim_stable):
    """
    Calculates LCOE ($/MWh) including CAPEX annualized + Fuel + Fixed OPEX.
    Penalizes Low Load operation.
    """
    if not sim_stable:
        return 99999.0 # Infinity cost for unstable systems
    
    # 1. CAPEX Annualized
    gen_cap_mw = n_gens * specs['mw']
    capex_gens = gen_cap_mw * 1000 * capex_gen_kwe
    
    capex_bess = (bess_mw * 1000 * capex_bess_kw) + (bess_mw * bess_dur_hours * 1000 * capex_bess_kwh)
    
    total_capex = capex_gens + capex_bess
    crf = (discount_rate * (1+discount_rate)**project_years) / ((1+discount_rate)**project_years - 1)
    capex_annual = total_capex * crf
    
    # 2. Fuel Cost (Non-Linear Efficiency Curve)
    # Load Factor = Base Load / Total Capacity
    load_mw = p_gross_total * (base_load_pct/100.0)
    lf = load_mw / gen_cap_mw
    
    # Heat Rate Penalty: HR increases as LF decreases.
    # Simple Model: HR_actual = HR_rated * (1 + 0.5 * (1 - LF)^2)
    # E.g. LF=1.0 -> Factor 1.0. LF=0.5 -> Factor 1.125. LF=0.1 -> Factor 1.4.
    hr_penalty = 1.0 + 0.8 * ((1.0 - max(0.1, lf))**2)
    hr_actual = specs['hr'] * hr_penalty
    
    fuel_cost_hr = load_mw * hr_actual * fuel_price # MW * MMBtu/MWh * $/MMBtu = $/h
    fuel_annual = fuel_cost_hr * op_hours
    
    # 3. O&M (Fixed approx 2% of Capex)
    om_annual = total_capex * 0.02
    
    # 4. Total Generation
    total_mwh_year = load_mw * op_hours
    
    lcoe = (capex_annual + fuel_annual + om_annual) / total_mwh_year
    return lcoe

# ==============================================================================
# 4. OPTIMIZATION LOOP (GRID SEARCH)
# ==============================================================================

def run_optimization():
    # Progress Bar
    progress_text = "Running Economic Optimization..."
    my_bar = st.progress(0, text=progress_text)
    
    # Grid Ranges
    # Gens: Start from N that barely covers load, up to +10 units
    n_min = int(np.ceil((p_gross_total*(base_load_pct/100)) / specs['mw']))
    n_range = range(n_min, n_min + 8)
    
    # BESS: 0 to 120% of Spike
    bess_step_mw = max(1.0, step_mw / 5.0)
    b_range = np.arange(0, step_mw * 1.2, bess_step_mw)
    
    results = []
    
    total_iters = len(n_range) * len(b_range)
    iter_count = 0
    
    for n in n_range:
        for b_mw in b_range:
            # 1. Run Physics
            sys_mva = (n * specs['mw']) / 0.8
            sim_p = {
                'H': h_const, 'Tau_g': tau_gov, 'Tau_b': bess_response/1000.0,
                'P_spike': step_mw, 'P_bess_cap': b_mw, 'Sys_MVA': sys_mva,
                'Bess_Enabled': True, 'Duration': pulse_duration
            }
            t = np.linspace(0, 10, 500)
            y0 = [0.0, 0.0, 0.0]
            try:
                sol = odeint(system_dynamics, y0, t, args=(sim_p,))
                freq = 60.0 + sol[:, 0]
                nadir = np.min(freq)
                peak = np.max(freq)
                
                is_stable = (nadir >= nadir_limit) and (peak <= overshoot_limit)
                
            except:
                is_stable = False
            
            # 2. Calculate Economics
            cost = calculate_lcoe(n, b_mw, is_stable)
            
            results.append({
                "Gens": n, "BESS_MW": b_mw, "Stable": is_stable, "LCOE": cost, 
                "Nadir": nadir, "Cap_MW": n*specs['mw']
            })
            
            iter_count += 1
            my_bar.progress(int(iter_count/total_iters * 100), text=f"Simulating: {n} Gens + {b_mw:.1f} MW BESS")
            
    my_bar.empty()
    return pd.DataFrame(results)

# ==============================================================================
# 5. MAIN UI
# ==============================================================================

st.title("⚡ CAT Hybrid Optimizer v5.0")
st.markdown("**Objective:** Find the lowest LCOE ($/MWh) that meets Frequency Stability limits.")

tab1, tab2 = st.tabs(["🚀 Single Simulation", "💰 Global Optimization"])

# --- TAB 1: SINGLE RUN (Validation) ---
with tab1:
    if st.button("Run Single Check", type="primary"):
        # (Same logic as v4 logic for single run visualization)
        # Re-using code for brevity in display
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
            ax.plot(t, freq, label="Freq")
            ax.axhline(nadir_limit, color='r', linestyle='--')
            ax.set_title(f"Nadir: {nadir:.3f} Hz")
            ax.grid(True, alpha=0.3)
            st.pyplot(fig)
        with c2:
            lcoe = calculate_lcoe(n_gens_op, bess_cap if enable_bess else 0, nadir >= nadir_limit)
            st.metric("LCOE", f"${lcoe:.2f}/MWh")
            if nadir < nadir_limit:
                st.error("TRIP DETECTED")
            else:
                st.success("STABLE")

# --- TAB 2: OPTIMIZER ---
with tab2:
    st.info("This will simulate multiple combinations to find the sweet spot.")
    
    if st.button("🔎 Run Optimizer"):
        df_opt = run_optimization()
        
        # Filter viable solutions
        df_viable = df_opt[df_opt['Stable'] == True].sort_values("LCOE")
        
        if df_viable.empty:
            st.error("No stable configuration found within search range! Try increasing Gen/BESS limits.")
        else:
            best = df_viable.iloc[0]
            
            # --- RESULTS ---
            st.markdown("### 🏆 Optimal Configuration")
            
            c1, c2, c3 = st.columns(3)
            c1.markdown(f"""
            <div class="opt-box">
                <div style="font-size:14px">OPTIMAL FLEET</div>
                <div style="font-size:28px; font-weight:bold">{int(best['Gens'])} Units</div>
                <div style="font-size:12px">of {gen_model}</div>
            </div>
            """, unsafe_allow_html=True)
            
            c2.markdown(f"""
            <div class="opt-box">
                <div style="font-size:14px">OPTIMAL BESS</div>
                <div style="font-size:28px; font-weight:bold">{best['BESS_MW']:.1f} MW</div>
                <div style="font-size:12px">Response: {bess_response} ms</div>
            </div>
            """, unsafe_allow_html=True)
            
            c3.markdown(f"""
            <div class="success-box">
                <div style="font-size:14px">LOWEST LCOE</div>
                <div style="font-size:28px; font-weight:bold">${best['LCOE']:.2f}</div>
                <div style="font-size:12px">per MWh</div>
            </div>
            """, unsafe_allow_html=True)
            
            st.write("---")
            
            # --- BATTLE CARD: Optimized vs Pure Gen ---
            # Pure Gen is the stable solution with 0 BESS (if exists) or lowest BESS
            df_pure_gen = df_viable[df_viable['BESS_MW'] == 0]
            
            if not df_pure_gen.empty:
                base = df_pure_gen.sort_values("LCOE").iloc[0]
                savings = base['LCOE'] - best['LCOE']
                savings_pct = (savings / base['LCOE']) * 100
                
                st.subheader("⚔️ Battle Card: Hybrid vs. Traditional")
                comp_data = {
                    "Metric": ["Generators (N)", "BESS (MW)", "LCOE ($/MWh)", "Annual Savings"],
                    "Traditional (Gen Only)": [f"{int(base['Gens'])}", "0.0 MW", f"${base['LCOE']:.2f}", "-"],
                    "Hybrid (Optimized)": [f"{int(best['Gens'])}", f"{best['BESS_MW']:.1f} MW", f"**${best['LCOE']:.2f}**", f"**{savings_pct:.1f}%**"]
                }
                st.table(pd.DataFrame(comp_data).set_index("Metric"))
                
                st.caption(f"*Annual Savings Estimate based on {p_gross_total:.0f}MW load: **${(savings * p_gross_total * op_hours / 1e6):.2f} Million/year***")
            
            # --- HEATMAP VIZ ---
            st.markdown("### 🗺️ Cost & Stability Map")
            
            fig, ax = plt.subplots(figsize=(10, 6))
            
            # Scatter plot of all trials
            # Grey for unstable, Color for stable (mapped to LCOE)
            unstable = df_opt[~df_opt['Stable']]
            stable = df_opt[df_opt['Stable']]
            
            ax.scatter(unstable['Gens'], unstable['BESS_MW'], c='lightgray', marker='x', label='Trip (Unstable)')
            sc = ax.scatter(stable['Gens'], stable['BESS_MW'], c=stable['LCOE'], cmap='viridis_r', s=100, edgecolors='k', label='Stable')
            
            # Highlight Best
            ax.scatter(best['Gens'], best['BESS_MW'], c='gold', s=300, marker='*', edgecolors='black', label='Optimal', zorder=10)
            
            cbar = plt.colorbar(sc, ax=ax)
            cbar.set_label('LCOE ($/MWh)')
            
            ax.set_xlabel("Number of Generators")
            ax.set_ylabel("BESS Capacity (MW)")
            ax.set_title("Optimization Landscape: LCOE vs. Stability")
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            st.pyplot(fig)
