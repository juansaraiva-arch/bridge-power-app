import streamlit as st
import pandas as pd
import numpy as np
import math
import sys
import plotly.express as px
import plotly.graph_objects as go
from io import BytesIO
from datetime import datetime

# PDF Generation with ReportLab (Full Professional Engine)
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib.colors import HexColor, white
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, 
    PageBreak, KeepTogether
)
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT, TA_JUSTIFY

# --- PAGE CONFIGURATION ---
st.set_page_config(page_title="CAT Bridge Power Enterprise", page_icon="🏭", layout="wide")

# ==============================================================================
# 0. HYBRID DATA LIBRARY - ENHANCED WITH DENSITY & RAMP RATE
# ==============================================================================

leps_gas_library = {
    "XGC1900": {
        "description": "Mobile Power Module (High Speed)",
        "type": "High Speed",
        "iso_rating_mw": 1.9,
        "electrical_efficiency": 0.392,
        "heat_rate_lhv": 8780,
        "step_load_pct": 25.0, 
        "ramp_rate_mw_s": 0.5,
        "emissions_nox": 0.5,
        "emissions_co": 2.5,
        "mtbf_hours": 50000,
        "maintenance_interval_hrs": 1000,
        "maintenance_duration_hrs": 48,
        "default_for": 2.0, 
        "default_maint": 5.0,
        "est_cost_kw": 775.0,      
        "est_install_kw": 300.0,
        "power_density_mw_per_m2": 0.010,
        "gas_pressure_min_psi": 1.5,
        "reactance_xd_2": 0.14
    },
    "G3520FR": {
        "description": "Fast Response Gen Set (High Speed)",
        "type": "High Speed",
        "iso_rating_mw": 2.5,
        "electrical_efficiency": 0.386,
        "heat_rate_lhv": 8836,
        "step_load_pct": 40.0,
        "ramp_rate_mw_s": 0.6,
        "emissions_nox": 0.5,
        "emissions_co": 2.1,
        "mtbf_hours": 48000,
        "maintenance_interval_hrs": 1000,
        "maintenance_duration_hrs": 48,
        "default_for": 2.0,
        "default_maint": 5.0,
        "est_cost_kw": 575.0,
        "est_install_kw": 650.0,
        "power_density_mw_per_m2": 0.010,
        "gas_pressure_min_psi": 1.5,
        "reactance_xd_2": 0.14
    },
    "G3520K": {
        "description": "High Efficiency Gen Set (High Speed)",
        "type": "High Speed",
        "iso_rating_mw": 2.4,
        "electrical_efficiency": 0.453,
        "heat_rate_lhv": 7638,
        "step_load_pct": 15.0,
        "ramp_rate_mw_s": 0.4,
        "emissions_nox": 0.3,
        "emissions_co": 2.3,
        "mtbf_hours": 52000,
        "maintenance_interval_hrs": 1000,
        "maintenance_duration_hrs": 48,
        "default_for": 2.5,
        "default_maint": 6.0,
        "est_cost_kw": 575.0,
        "est_install_kw": 650.0,
        "power_density_mw_per_m2": 0.010,
        "gas_pressure_min_psi": 1.5,
        "reactance_xd_2": 0.13
    },
    "Titan 130": {
        "description": "Solar Gas Turbine (16.5 MW)",
        "type": "Gas Turbine",
        "iso_rating_mw": 16.5,
        "electrical_efficiency": 0.354,
        "heat_rate_lhv": 9630,
        "step_load_pct": 15.0,
        "ramp_rate_mw_s": 2.0,
        "emissions_nox": 0.6,
        "emissions_co": 0.6,
        "mtbf_hours": 80000,
        "maintenance_interval_hrs": 8000,
        "maintenance_duration_hrs": 120,
        "default_for": 1.5,
        "default_maint": 2.0,
        "est_cost_kw": 775.0,
        "est_install_kw": 1000.0,
        "power_density_mw_per_m2": 0.020,
        "gas_pressure_min_psi": 300.0,
        "reactance_xd_2": 0.18
    },
    "G20CM34": {
        "description": "Medium Speed Baseload Platform",
        "type": "Medium Speed",
        "iso_rating_mw": 9.76,
        "electrical_efficiency": 0.475,
        "heat_rate_lhv": 7480,
        "step_load_pct": 10.0,
        "ramp_rate_mw_s": 0.3,
        "emissions_nox": 0.5,
        "emissions_co": 0.5,
        "mtbf_hours": 60000,
        "maintenance_interval_hrs": 2500,
        "maintenance_duration_hrs": 72,
        "default_for": 3.0, 
        "default_maint": 5.0,
        "est_cost_kw": 700.0,
        "est_install_kw": 1250.0,
        "power_density_mw_per_m2": 0.008,
        "gas_pressure_min_psi": 90.0,
        "reactance_xd_2": 0.16
    }
}

# ==============================================================================
# HELPER FUNCTIONS - THE V3 PHYSICS CORE
# ==============================================================================

def get_part_load_efficiency(base_eff, load_pct, gen_type):
    load_pct = max(0, min(100, load_pct))
    if gen_type == "High Speed":
        xp = [0, 25, 50, 75, 100]
        fp = [0.0, 0.70, 0.88, 0.96, 1.00]
    elif gen_type == "Medium Speed":
        xp = [0, 25, 50, 75, 100]
        fp = [0.0, 0.75, 0.91, 0.97, 1.00]
    elif gen_type == "Gas Turbine":
        xp = [0, 25, 50, 75, 100]
        fp = [0.0, 0.55, 0.78, 0.90, 1.00]
    else:
        return base_eff
    factor = np.interp(load_pct, xp, fp)
    return base_eff * factor

def transient_stability_check(xd_pu, num_units, step_load_pct):
    if num_units == 0: return False, 100.0
    equiv_xd = xd_pu / math.sqrt(num_units)
    voltage_sag = (step_load_pct/100) * equiv_xd * 100
    if voltage_sag > 15: # Bridge limit slightly higher than Perm (15% vs 10%)
        return False, voltage_sag
    return True, voltage_sag

def calculate_bess_requirements(p_avg, p_peak, step_load_pct, gen_ramp, gen_step, load_ramp_req, enable_black_start=False):
    step_mw_req = p_avg * (step_load_pct / 100)
    gen_step_mw = p_avg * (gen_step / 100)
    bess_step = max(0, step_mw_req - gen_step_mw)
    bess_peak = max(0, p_peak - p_avg)
    bess_ramp = max(0, (load_ramp_req - gen_ramp) * 10)
    bess_black = p_peak * 0.10 if enable_black_start else 0
    
    bess_power = max(bess_step, bess_peak, bess_ramp, bess_black)
    bess_energy = bess_power * 1.0 # 1h standard for bridge
    
    return bess_power, bess_energy, {
        "step_support": bess_step, "peak_shaving": bess_peak, "ramp_support": bess_ramp, "black_start": bess_black
    }

def calculate_bess_reliability_credit(bess_power, unit_cap):
    # En rental, el BESS permite apagar unidades de reserva
    # 1 MW de BESS confiable = 1 MW de Reserva Rodante
    credit_units = math.floor(bess_power / unit_cap)
    return credit_units

def optimize_fleet_rental(p_avg, p_peak, unit_cap, gen_data, use_bess):
    # Optimización específica para RENTAL (Minimizar n_running para bajar cuota mensual)
    # Pero respetando eficiencia (evitar wet stacking)
    if use_bess:
        n_min = math.ceil(p_avg / unit_cap) # BESS cubre picos
    else:
        n_min = math.ceil(p_peak / unit_cap)
        
    best_n = n_min
    best_score = -9999
    
    for n in range(n_min, n_min + 5):
        load_pct = (p_avg / (n * unit_cap)) * 100
        if load_pct < 40 or load_pct > 95: continue
        
        eff = get_part_load_efficiency(gen_data["electrical_efficiency"], load_pct, gen_data["type"])
        
        # En Rental: El costo fijo (n) pesa más que en CAPEX. Penalizamos fuerte el número de unidades.
        score = (eff * 100) - (n * 5) 
        
        if score > best_score:
            best_score = score
            best_n = n
    return best_n

# ==============================================================================
# 1. INPUTS & SIDEBAR
# ==============================================================================

with st.sidebar:
    st.image("https://img.icons8.com/color/96/generator.png", width=60)
    st.title("CAT Bridge Power")
    
    # --- 1. PROJECT ---
    st.header("1. Project Profile")
    dc_type = st.selectbox("Data Center Type", ["AI Factory (Training)", "Hyperscale Standard", "Colocation"])
    is_ai = "AI" in dc_type
    p_it = st.number_input("Critical IT Load (MW)", 1.0, 500.0, 50.0, step=5.0)
    pue = st.number_input("Target PUE", 1.05, 2.0, 1.25 if not is_ai else 1.15)
    
    p_total_avg = p_it * pue
    p_total_peak = p_total_avg * 1.10 # 10% overhead peak
    
    st.info(f"Avg Load: **{p_total_avg:.1f} MW**")
    
    # --- 2. PHYSICS ---
    st.header("2. Transient Physics")
    load_step_pct = st.slider("Max Load Step (%)", 0, 100, 50 if is_ai else 20)
    load_ramp = st.number_input("Ramp Rate (MW/s)", 0.1, 10.0, 2.0)
    
    # --- 3. TECHNOLOGY ---
    st.header("3. Technology")
    tech_filter = st.multiselect("Filter", ["High Speed", "Medium Speed", "Gas Turbine"], default=["High Speed"])
    avail_gens = {k: v for k, v in leps_gas_library.items() if v["type"] in tech_filter}
    selected_gen = st.selectbox("Generator Model", list(avail_gens.keys()))
    gen_data = avail_gens[selected_gen]
    
    use_bess = st.checkbox("Include BESS (Hybrid)", value=True)
    enable_black_start = st.checkbox("Black Start Capable", value=True)
    
    # --- 4. SITE ---
    st.header("4. Site Conditions")
    temp_c = st.slider("Max Ambient Temp (°C)", 20, 50, 35)
    alt_m = st.number_input("Altitude (m)", 0, 3000, 100)
    
    # Auto Derate Calculation (V3 Logic)
    derate = max(0.5, 1.0 - (max(0, temp_c-25)*0.01) - (alt_m/1000 * 0.1))
    unit_site_cap = gen_data["iso_rating_mw"] * derate
    
    # --- 5. BRIDGE BUSINESS MODEL (V9 LOGIC) ---
    st.header("5. Commercial Model")
    biz_model = st.radio("Financial Mode", ["Bridge / Rental (OPEX)", "Permanent / Purchase (CAPEX)"])
    is_rental = "Rental" in biz_model
    
    fuel_price = st.number_input("Gas Price ($/MMBtu)", 1.0, 20.0, 4.5)
    
    if is_rental:
        st.markdown("💰 **Rental Parameters**")
        rental_rate_kw = st.number_input("Gen Rental ($/kW-mo)", 15.0, 60.0, 28.0)
        bess_rate_kw = st.number_input("BESS Rental ($/kW-mo)", 20.0, 80.0, 35.0)
        contract_months = st.number_input("Contract Duration (Months)", 6, 60, 24)
        buyout_pct = st.number_input("Buyout Option @ End (%)", 0.0, 100.0, 20.0)
        capex_kw = 0.0 # No initial CAPEX
    else:
        st.markdown("🏗️ **Purchase Parameters**")
        capex_kw = st.number_input("Turnkey CAPEX ($/kW)", 500.0, 2000.0, 800.0)
        project_years = st.number_input("Project Life (Years)", 10, 30, 20)
        wacc = st.number_input("WACC (%)", 5.0, 15.0, 8.0) / 100
        rental_rate_kw = 0.0
        contract_months = project_years * 12

# ==============================================================================
# 2. CALCULATION CORE
# ==============================================================================

# A. Fleet Optimization
n_running = optimize_fleet_rental(p_total_avg, p_total_peak, unit_site_cap, gen_data, use_bess)

# B. BESS Sizing
if use_bess:
    bess_mw, bess_mwh, bess_bkdn = calculate_bess_requirements(
        p_total_avg, p_total_peak, load_step_pct, 
        gen_data["ramp_rate_mw_s"], gen_data["step_load_pct"], load_ramp, enable_black_start
    )
    # BESS Credit (Rental Savings)
    bess_credit = calculate_bess_reliability_credit(bess_mw, unit_site_cap)
    n_reserve = max(1, 2 - bess_credit) # Reducimos reserva física gracias al BESS
else:
    bess_mw, bess_mwh = 0, 0
    bess_bkdn = {}
    n_reserve = 2 # N+2 estándar para Bridge sin BESS

n_total = n_running + n_reserve
installed_mw = n_total * unit_site_cap

# C. Efficiency & Ops
load_pct = (p_total_avg / (n_running * unit_site_cap)) * 100
fleet_eff = get_part_load_efficiency(gen_data["electrical_efficiency"], load_pct, gen_data["type"])
fuel_mw = p_total_avg / fleet_eff
fuel_mmbtu_hr = fuel_mw * 3.412

# D. Transient Check
is_stable, voltage_sag = transient_stability_check(gen_data["reactance_xd_2"], n_running, load_step_pct)

# E. Financials
hours_mo = 730
fuel_cost_mo = fuel_mmbtu_hr * fuel_price * hours_mo
gen_rent_mo = (n_total * unit_site_cap * 1000) * rental_rate_kw
bess_rent_mo = (bess_mw * 1000 * bess_rate_kw)

if is_rental:
    monthly_bill = fuel_cost_mo + gen_rent_mo + bess_rent_mo
    lcoe = monthly_bill / (p_total_avg * hours_mo) * 1000 # $/MWh
    total_contract_cost = monthly_bill * contract_months
    
    # Buyout Logic
    # Asumimos que el precio 'nuevo' de referencia es $800/kW
    ref_new_price = n_total * unit_site_cap * 1000 * 800
    buyout_price = ref_new_price * (buyout_pct/100)
else:
    # CAPEX Logic (Simplified Annualized)
    total_capex = installed_mw * 1000 * capex_kw
    if use_bess: total_capex += (bess_mw * 1000 * 300) # BESS Purchase approx
    
    crf = (wacc * (1+wacc)**20) / ((1+wacc)**20 - 1)
    annual_capex = total_capex * crf
    annual_fuel = fuel_cost_mo * 12
    lcoe = (annual_capex + annual_fuel) / (p_total_avg * 8760) * 1000
    monthly_bill = lcoe * p_total_avg * hours_mo / 1000

# ==============================================================================
# 3. DASHBOARD
# ==============================================================================

st.title("🏭 CAT Bridge Power Enterprise")

# KPI ROW
k1, k2, k3, k4, k5 = st.columns(5)
k1.metric("Solution Type", "Hybrid Rental" if is_rental and use_bess else ("Rental" if is_rental else "Turnkey"))
k2.metric("Fleet Strategy", f"{n_running}+{n_reserve}", f"{selected_gen}")
k3.metric("Installed Cap", f"{installed_mw:.1f} MW", f"Load: {load_pct:.1f}%")
k4.metric("Monthly OPEX", f"${monthly_bill/1000:,.0f}k", f"LCOE: ${lcoe/1000:.3f}/kWh")
k5.metric("Stability", "OK" if is_stable else "RISK", f"Sag: {voltage_sag:.1f}%")

st.divider()

t1, t2, t3, t4 = st.tabs(["📊 System Design", "💰 Business Case", "⚡ Technical Perf.", "📄 Report"])

with t1:
    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Configuration Details")
        st.write(f"**Primary Tech:** {gen_data['type']} ({selected_gen})")
        st.write(f"**Site Derate:** {derate*100:.1f}% (Temp: {temp_c}C, Alt: {alt_m}m)")
        st.write(f"**Availability Strategy:** N+{n_reserve}")
        if use_bess:
            st.success(f"**BESS Integrated:** {bess_mw:.1f} MW / {bess_mwh:.1f} MWh")
            st.caption("BESS is providing Virtual Spinning Reserve, reducing rental units.")
    with c2:
        st.subheader("Load Analysis")
        st.metric("Peak Load", f"{p_total_peak:.1f} MW")
        st.metric("Avg Load", f"{p_total_avg:.1f} MW")
        st.metric("Step Load", f"{p_total_avg * load_step_pct/100:.1f} MW ({load_step_pct}%)")

with t2:
    if is_rental:
        st.subheader("Rental Financial Breakdown")
        fin_df = pd.DataFrame({
            "Category": ["Gen Rental", "BESS Rental", "Fuel (Est)"],
            "Monthly Cost": [gen_rent_mo, bess_rent_mo, fuel_cost_mo]
        })
        c_fin1, c_fin2 = st.columns([1, 2])
        c_fin1.dataframe(fin_df.style.format({"Monthly Cost": "${:,.0f}"}), use_container_width=True)
        c_fin1.metric("Total Contract Value", f"${total_contract_cost/1e6:.1f} M", f"{contract_months} Months")
        
        fig_fin = px.pie(fin_df, values="Monthly Cost", names="Category", title="Monthly Bill Composition", hole=0.4)
        c_fin2.plotly_chart(fig_fin, use_container_width=True)
        
        st.divider()
        st.subheader("Exit Strategy (End of Term)")
        c_ex1, c_ex2 = st.columns(2)
        c_ex1.metric("Buyout Price", f"${buyout_price/1e6:.1f} M", f"{buyout_pct}% of New Value")
        c_ex2.info("💡 **Bridge-to-Permanent:** Buying the assets allows conversion to Backup/Peaking plant for long-term grid support.")
    else:
        st.subheader("CAPEX Purchase Model")
        st.metric("Total Project CAPEX", f"${total_capex/1e6:.1f} M")
        st.metric("Annual OPEX (Fuel)", f"${annual_fuel/1e6:.1f} M")

with t3:
    st.subheader("Physics & Stability")
    c_tech1, c_tech2 = st.columns(2)
    with c_tech1:
        st.write("**Transient Response**")
        if is_stable:
            st.success(f"✅ Voltage Sag: {voltage_sag:.2f}% (Limit 15%)")
        else:
            st.error(f"❌ Voltage Sag: {voltage_sag:.2f}% (Limit 15%)")
            st.warning("Action: Add BESS or more generators.")
    with c_tech2:
        st.write("**Efficiency**")
        st.metric("Real Fleet Eff", f"{fleet_eff*100:.1f}%", f"Base: {gen_data['electrical_efficiency']*100:.1f}%")
        if load_pct < 40:
            st.warning("⚠️ Low Load Warning: Wet Stacking Risk")

with t4:
    st.header("📄 Executive Report")
    
    # PDF Generator Logic (Embedded V3 Engine)
    def generate_bridge_pdf():
        buffer = BytesIO()
        doc = SimpleDocTemplate(buffer, pagesize=letter, rightMargin=40, leftMargin=40, topMargin=40, bottomMargin=40)
        styles = getSampleStyleSheet()
        story = []
        
        # Header
        story.append(Paragraph("CAT BRIDGE POWER - EXECUTIVE BRIEF", styles['Title']))
        story.append(Spacer(1, 0.2*inch))
        
        # Summary
        story.append(Paragraph(f"<b>Project:</b> {dc_type} ({p_it} MW)", styles['Heading2']))
        summary_text = f"""
        This report outlines a <b>{biz_model}</b> solution for a critical load of {p_total_avg:.1f} MW.
        The system utilizes <b>{n_total} x {selected_gen}</b> units in an N+{n_reserve} configuration
        {'with BESS support' if use_bess else 'without BESS'}.
        """
        story.append(Paragraph(summary_text, styles['Normal']))
        story.append(Spacer(1, 0.2*inch))
        
        # Financials
        story.append(Paragraph("Financial Overview", styles['Heading2']))
        fin_data = [
            ["Metric", "Value"],
            ["Monthly Bill", f"${monthly_bill:,.0f}"],
            ["LCOE", f"${lcoe/1000:.3f}/kWh"],
            ["Contract Term", f"{contract_months} Months" if is_rental else f"{project_years} Years"],
            ["Total Contract Value", f"${total_contract_cost:,.0f}" if is_rental else f"${total_capex:,.0f}"]
        ]
        t = Table(fin_data, colWidths=[3*inch, 2.5*inch])
        t.setStyle(TableStyle([
            ('BACKGROUND', (0,0), (-1,0), HexColor('#FFCC00')),
            ('TEXTCOLOR', (0,0), (-1,0), HexColor('#000000')),
            ('ALIGN', (0,0), (-1,-1), 'LEFT'),
            ('GRID', (0,0), (-1,-1), 1, HexColor('#000000'))
        ]))
        story.append(t)
        
        # Technical
        story.append(Paragraph("Technical Specs", styles['Heading2']))
        tech_data = [
            ["Metric", "Value"],
            ["Configuration", f"{n_running} Run + {n_reserve} Res"],
            ["Installed Capacity", f"{installed_mw:.1f} MW"],
            ["Stability (Sag)", f"{voltage_sag:.1f}% {'(OK)' if is_stable else '(FAIL)'}"],
            ["Fleet Efficiency", f"{fleet_eff*100:.1f}%"]
        ]
        t2 = Table(tech_data, colWidths=[3*inch, 2.5*inch])
        t2.setStyle(TableStyle([('GRID', (0,0), (-1,-1), 1, HexColor('#000000'))]))
        story.append(t2)
        
        doc.build(story)
        buffer.seek(0)
        return buffer.getvalue()

    if st.button("📥 Generate Bridge Report"):
        pdf_bytes = generate_bridge_pdf()
        st.download_button("Download PDF", data=pdf_bytes, file_name="CAT_Bridge_Report.pdf", mime="application/pdf")
