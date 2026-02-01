import streamlit as st
import pandas as pd
import numpy as np
import math
import sys
import plotly.express as px
import plotly.graph_objects as go
from io import BytesIO
from datetime import datetime

# Importamos ReportLab para PDFs Profesionales (El motor de la V3)
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib.colors import HexColor, white
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, 
    PageBreak, KeepTogether
)
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT, TA_JUSTIFY

# --- CONFIGURACIÓN DE PÁGINA ---
st.set_page_config(page_title="CAT Bridge Power Enterprise", page_icon="🏭", layout="wide")

# ==============================================================================
# 0. LIBRERÍA DE DATOS HÍBRIDA (Datos Técnicos V3 + Datos Comerciales V9)
# ==============================================================================

leps_gas_library = {
    "XGC1900": {
        "type": "High Speed",
        "iso_rating_mw": 1.9,
        "electrical_efficiency": 0.392,
        "step_load_pct": 25.0, 
        "ramp_rate_mw_s": 0.5,
        "emissions_nox": 0.5,
        "est_rent_mo_kw": 28.0, # Costo Renta Mensual ($/kW)
        "est_install_kw": 150.0, # Costo Instalación ($/kW)
        "power_density_mw_per_m2": 0.010,
        "reactance_xd_2": 0.14
    },
    "XGC1375": {
        "type": "High Speed",
        "iso_rating_mw": 1.375,
        "electrical_efficiency": 0.380,
        "step_load_pct": 30.0, 
        "ramp_rate_mw_s": 0.4,
        "emissions_nox": 0.5,
        "est_rent_mo_kw": 30.0,
        "est_install_kw": 150.0,
        "power_density_mw_per_m2": 0.010,
        "reactance_xd_2": 0.14
    },
    "PMG 3516 (Diesel)": {
        "type": "Diesel", # Curva de eficiencia distinta
        "iso_rating_mw": 2.0,
        "electrical_efficiency": 0.400,
        "step_load_pct": 100.0, # Diesel aguanta todo
        "ramp_rate_mw_s": 1.5,
        "emissions_nox": 4.0,
        "est_rent_mo_kw": 35.0,
        "est_install_kw": 100.0,
        "power_density_mw_per_m2": 0.012,
        "reactance_xd_2": 0.12
    },
    "Titan 130 (Turbine)": {
        "type": "Gas Turbine",
        "iso_rating_mw": 16.5,
        "electrical_efficiency": 0.354,
        "step_load_pct": 15.0,
        "ramp_rate_mw_s": 2.0,
        "emissions_nox": 0.6,
        "est_rent_mo_kw": 22.0, # Menor costo por escala
        "est_install_kw": 400.0,
        "power_density_mw_per_m2": 0.020,
        "reactance_xd_2": 0.18
    }
}

# ==============================================================================
# 1. MOTOR FÍSICO AVANZADO (Importado de CAT-QuickSize-v3)
# ==============================================================================

def get_part_load_efficiency(base_eff, load_pct, gen_type):
    """Calcula eficiencia real usando curvas de interpolación (No fórmulas simples)."""
    load_pct = max(0, min(100, load_pct))
    
    if gen_type == "High Speed":
        xp = [0, 25, 50, 75, 100]
        fp = [0.0, 0.70, 0.88, 0.96, 1.00]
    elif gen_type == "Diesel":
        xp = [0, 25, 50, 75, 100]
        fp = [0.0, 0.75, 0.90, 0.98, 1.00]
    elif gen_type == "Gas Turbine":
        xp = [0, 25, 50, 75, 100]
        fp = [0.0, 0.55, 0.78, 0.90, 1.00]
    else:
        return base_eff

    factor = np.interp(load_pct, xp, fp)
    return base_eff * factor

def transient_stability_check(xd_pu, num_units, step_load_pct):
    """Verifica si el voltaje colapsa ante un golpe de carga."""
    if num_units == 0: return False, 100.0
    equiv_xd = xd_pu / math.sqrt(num_units)
    # Fórmula aproximada: Caída Voltaje % ~ Step Load % * Reactancia Eq
    voltage_sag = (step_load_pct/100) * equiv_xd * 100
    # Límite Bridge: 15% (un poco más tolerante que plantas permanentes)
    return voltage_sag <= 15, voltage_sag

def calculate_bess_requirements(p_avg, p_peak, step_load_pct, gen_ramp, gen_step, load_ramp_req, enable_black_start):
    """Dimensionamiento detallado de BESS por componentes."""
    step_mw_req = p_avg * (step_load_pct / 100)
    gen_step_mw = p_avg * (gen_step / 100)
    
    bess_step = max(0, step_mw_req - gen_step_mw)
    bess_peak = max(0, p_peak - p_avg)
    bess_ramp = max(0, (load_ramp_req - gen_ramp) * 10) # 10s buffer
    bess_black = p_peak * 0.10 if enable_black_start else 0
    
    bess_power = max(bess_step, bess_peak, bess_ramp, bess_black)
    bess_energy = bess_power * 1.0 # 1 hora estándar para Bridge
    
    return bess_power, bess_energy, {
        "Step Support": bess_step, "Peak Shaving": bess_peak, 
        "Ramp Support": bess_ramp, "Black Start": bess_black
    }

def optimize_fleet_rental(p_avg, p_peak, unit_cap, gen_data, use_bess):
    """Optimiza flota priorizando OPEX (Menos máquinas = Menos Renta)."""
    if use_bess:
        n_min = math.ceil(p_avg / unit_cap) # BESS cubre picos y transitorios
    else:
        n_min = math.ceil(p_peak / unit_cap) # Generadores cubren todo
        
    best_n = n_min
    best_score = -9999
    
    # Buscamos el punto dulce de eficiencia vs costo de renta
    for n in range(n_min, n_min + 5):
        load_pct = (p_avg / (n * unit_cap)) * 100
        if load_pct < 40 or load_pct > 98: continue # Evitar Wet Stacking o Sobrecarga
        
        eff = get_part_load_efficiency(gen_data["electrical_efficiency"], load_pct, gen_data["type"])
        
        # Score: Eficiencia alta (ahorro fuel) - Costo Renta (n)
        score = (eff * 100) - (n * 3) 
        if score > best_score:
            best_score = score
            best_n = n
            
    return best_n

# ==============================================================================
# 2. INTERFAZ DE USUARIO (SIDEBAR)
# ==============================================================================

with st.sidebar:
    st.image("https://img.icons8.com/color/96/generator.png", width=60)
    st.title("CAT Bridge Power Enterprise")
    
    # --- 1. PERFIL DEL PROYECTO ---
    st.header("1. Project Profile")
    dc_type = st.selectbox("Data Center Type", ["AI Factory (Training)", "Hyperscale Standard", "Colocation"])
    is_ai = "AI" in dc_type
    
    p_it = st.number_input("Critical IT Load (MW)", 1.0, 500.0, 50.0, step=5.0)
    pue = st.number_input("Target PUE", 1.05, 2.0, 1.25)
    
    p_total_avg = p_it * pue
    p_total_peak = p_total_avg * 1.10 # 10% overhead pico
    
    st.info(f"Avg Load: **{p_total_avg:.1f} MW**")
    
    # --- 2. FÍSICA TRANSITORIA ---
    st.header("2. Transient Physics")
    load_step_pct = st.slider("Max Load Step (%)", 0, 100, 50 if is_ai else 20)
    load_ramp = st.number_input("Ramp Rate (MW/s)", 0.1, 10.0, 2.0)
    
    # --- 3. TECNOLOGÍA ---
    st.header("3. Technology Solution")
    selected_gen = st.selectbox("Generator Model", list(leps_gas_library.keys()))
    gen_data = leps_gas_library[selected_gen]
    
    use_bess = st.checkbox("Include BESS (Hybrid)", value=True)
    enable_black_start = st.checkbox("Black Start Capable", value=True)
    
    # --- 4. SITIO Y DERATEO AVANZADO ---
    st.header("4. Site Conditions & Constraints")
    
    col_s1, col_s2 = st.columns(2)
    temp_c = col_s1.slider("Max Ambient Temp (°C)", 20, 50, 35)
    alt_m = col_s2.number_input("Altitude (m)", 0, 4000, 100, step=100)
    methane_number = st.slider("Gas Methane Number (MN)", 30, 100, 80, help="Calidad del gas (MN<70 requiere derateo fuerte)")
    
    # Cálculo de Derateo (Lógica CAT QuickSize v3)
    # 1. Temperatura: -1% por cada 1°C sobre 25°C
    derate_temp = 1.0 - max(0, (temp_c - 25) * 0.01)
    # 2. Altitud: -1% por cada 100m sobre 100m
    derate_alt = 1.0 - max(0, (alt_m - 100) * 0.0001)
    # 3. Combustible: -0.5% por cada punto de MN bajo 70
    derate_fuel = 1.0 - max(0, (75 - methane_number) * 0.005)
    
    derate = max(0.5, derate_temp * derate_alt * derate_fuel)
    unit_site_cap = gen_data["iso_rating_mw"] * derate
    
    st.caption(f"📉 **Site Rating:** {unit_site_cap:.2f} MW (Factor: {derate*100:.1f}%)")
    if derate < 0.85:
        st.warning(f"⚠️ High Derate detected! Check Cooling/Gas quality.")

    # Restricciones Ambientales
    limit_nox = st.selectbox("Emissions Limit (NOx)", ["No Limit", "World Bank (Standard)", "EPA Tier 4 (Strict)"])
    max_area_m2 = st.number_input("Max Available Area (m²)", 0, 50000, 0, help="Dejar en 0 si no hay limite")
    
    # --- 5. MODELO DE NEGOCIO (LOGICA V9) --
    st.header("5. Commercial Strategy")
    biz_model = st.radio("Financial Mode", ["Bridge / Rental (OPEX)", "Permanent / Purchase (CAPEX)"])
    is_rental = "Rental" in biz_model
    
    fuel_price = st.number_input("Gas Price ($/MMBtu)", 1.0, 20.0, 4.5)

    # --- NUEVO: TIME TO MARKET ---
    st.divider()
    st.markdown("⏱️ **Time-to-Market Economics**")
    enable_ttm = st.checkbox("Include Revenue Analysis?", value=True)
    
    if enable_ttm:
        grid_delay_mo = st.number_input("Grid Connection Delay (Months)", 0, 60, 18, help="Meses de espera por la red.")
        revenue_per_mw_mo = st.number_input("DC Revenue ($/MW-mo)", 10000, 1000000, 200000, step=10000, help="Facturación estimada por MW IT.")
    else:
        grid_delay_mo = 0
        revenue_per_mw_mo = 0
    
    st.divider()

    if is_rental:
        st.markdown("💰 **Rental Parameters**")
        rental_rate_kw = st.number_input("Gen Rental ($/kW-mo)", 10.0, 60.0, gen_data["est_rent_mo_kw"])
        bess_rate_kw = st.number_input("BESS Rental ($/kW-mo)", 15.0, 60.0, 35.0)
        contract_months = st.number_input("Contract Duration (Months)", 6, 60, 24)
        
        # --- NUEVO: Costos de Movilización ---
        st.markdown("🚚 **Logistics (One-Time)**")
        mob_cost = st.number_input("Mobilization ($)", 0, 1000000, 150000, step=10000)
        demob_cost = st.number_input("Demobilization ($)", 0, 1000000, 100000, step=10000)
        
        buyout_pct = st.number_input("Buyout Option @ End (%)", 0.0, 100.0, 20.0)
    else:
        st.markdown("🏗️ **Purchase Parameters**")
        capex_kw = st.number_input("Turnkey CAPEX ($/kW)", 500.0, 2000.0, 800.0)
        project_years = st.number_input("Project Life (Years)", 5, 30, 20)
        wacc = st.number_input("WACC (%)", 5.0, 15.0, 8.0) / 100
        # Valores dummy para evitar errores en lógica comparativa
        mob_cost = 0; demob_cost = 0; rental_rate_kw = 0; contract_months = project_years*12

# ==============================================================================
# 3. MOTOR DE CÁLCULO (CORE)
# ==============================================================================

# A. Optimización de Flota
n_running = optimize_fleet_rental(p_total_avg, p_total_peak, unit_site_cap, gen_data, use_bess)

# B. Dimensionamiento BESS
if use_bess:
    bess_mw, bess_mwh, bess_bkdn = calculate_bess_requirements(
        p_total_avg, p_total_peak, load_step_pct, 
        gen_data["ramp_rate_mw_s"], gen_data["step_load_pct"], load_ramp, enable_black_start
    )
    # BESS Credit (Ahorro de Renta): BESS reemplaza reserva rodante
    # Regla: 1 MW de BESS confiable = 1 MW de Generador en Reserva
    bess_credit_units = math.floor(bess_mw / unit_site_cap)
    n_reserve = max(1, 2 - bess_credit_units) # Mínimo 1 reserva física siempre
else:
    bess_mw, bess_mwh = 0, 0
    bess_bkdn = {}
    n_reserve = 2 # Estándar sin BESS

n_total = n_running + n_reserve
installed_mw = n_total * unit_site_cap

# C. Eficiencia y Combustible
load_pct = (p_total_avg / (n_running * unit_site_cap)) * 100
fleet_eff = get_part_load_efficiency(gen_data["electrical_efficiency"], load_pct, gen_data["type"])
fuel_mw = p_total_avg / fleet_eff
fuel_mmbtu_hr = fuel_mw * 3.412

# D. Estabilidad Transitoria
is_stable, voltage_sag = transient_stability_check(gen_data["reactance_xd_2"], n_running, load_step_pct)

# E. Finanzas
hours_mo = 730
fuel_cost_mo = fuel_mmbtu_hr * fuel_price * hours_mo
gen_rent_mo = (n_total * unit_site_cap * 1000) * rental_rate_kw
bess_rent_mo = (bess_mw * 1000 * bess_rate_kw)

if is_rental:
    monthly_bill = fuel_cost_mo + gen_rent_mo + bess_rent_mo
    lcoe = monthly_bill / (p_total_avg * hours_mo) * 1000 # $/MWh
    
    # Total Contract = Mensualidades + Mob + Demob
    total_contract_value = (monthly_bill * contract_months) + mob_cost + demob_cost
    
    # Lógica de Buyout
    ref_new_price = n_total * unit_site_cap * 1000 * 800 
    buyout_price = ref_new_price * (buyout_pct/100)
else:
    # CAPEX Simple
    total_capex = installed_mw * 1000 * capex_kw
    if use_bess: total_capex += (bess_mw * 1000 * 300)
    
    crf = (wacc * (1+wacc)**project_years) / ((1+wacc)**project_years - 1)
    annual_capex = total_capex * crf
    annual_fuel = fuel_cost_mo * 12
    lcoe = (annual_capex + annual_fuel) / (p_total_avg * 8760) * 1000
    monthly_bill = lcoe * p_total_avg * hours_mo / 1000
    mob_cost = 0 # Dummy para evitar error en gráfica

# ==============================================================================
# 3b. CÁLCULOS DE INGENIERÍA DETALLADA (Footprint & Emissions)
# ==============================================================================

# A. Emisiones y Urea
total_bhp = p_total_avg * 1341 # Convertir MW a BHP
nox_g_bhp_hr = gen_data["emissions_nox"]
nox_ton_yr = (nox_g_bhp_hr * total_bhp * 8760) / 907185 # Toneladas cortas

req_scr = False
scr_capex = 0
urea_opex_mo = 0

if limit_nox == "EPA Tier 4 (Strict)" and nox_g_bhp_hr > 0.1:
    req_scr = True
    # CAPEX SCR estimado: $60/kW
    scr_capex = installed_mw * 1000 * 60 
    # Consumo Urea estimado: 1.5% del consumo de combustible
    urea_liters_yr = (fuel_mmbtu_hr / 0.138) * 0.015 * 8760 # Approx
    urea_opex_mo = (urea_liters_yr / 12) * 0.50 # $0.50/litro

# B. Footprint (Huella Física)
# Datos QuickSize: Generador ~200m²/MW (incluye pasillos), BESS ~30m²/MW
area_gen = installed_mw * (1 / gen_data["power_density_mw_per_m2"])
area_bess = bess_mw * 30 if use_bess else 0
area_logistics = 200 # Zona de descarga básica
if is_rental: area_logistics += 300 # Patio de maniobras extra

total_area = area_gen + area_bess + area_logistics
area_status = "✅ OK"
if max_area_m2 > 0 and total_area > max_area_m2:
    area_status = "❌ OVERFLOW"

# ==============================================================================
# 4. DASHBOARD DE RESULTADOS
# ==============================================================================

# ==============================================================================
# 3b. MOTOR FÍSICO: EMISIONES Y FOOTPRINT (El Eslabón Perdido)
# ==============================================================================

# 1. CÁLCULO DE EMISIONES (NOx y Urea)
# Convertimos MW a BHP (Brake Horsepower) para usar factores de emisión estándar
total_bhp = p_total_avg * 1341 
nox_g_bhp_hr = gen_data["emissions_nox"]

# Toneladas anuales: (g/bhp-hr * bhp * horas) / g_per_ton
nox_ton_yr = (nox_g_bhp_hr * total_bhp * 8760) / 907185 

# Lógica de SCR (Catalizador)
req_scr = False
scr_capex = 0
urea_opex_mo = 0

# Verificamos si 'limit_nox' existe (lo agregaste en el Sidebar), si no, asumimos "No Limit"
limit_check = limit_nox if 'limit_nox' in locals() else "No Limit"

if limit_check == "EPA Tier 4 (Strict)" and nox_g_bhp_hr > 0.1:
    req_scr = True
    # Costo SCR aprox: $60/kW instalado
    scr_capex = installed_mw * 1000 * 60 
    
    # Consumo Urea: ~1.5% del consumo de combustible volumétrico (estimado simple)
    # 1 MMBtu gas ~ 28 m3. Urea ratio es complejo, usamos aproximación financiera:
    # Costo Urea ~ $1.5/MWh generado en motores Tier 4
    urea_opex_mo = p_total_avg * 730 * 1.5 

# 2. CÁLCULO DE FOOTPRINT (Huella Física)
# Factores de densidad (m2/MW) basados en CAT QuickSize V3
area_factor_gen = 1 / gen_data["power_density_mw_per_m2"]
area_gen = installed_mw * area_factor_gen

# BESS: Aprox 30 m2 por MW (contenedores + pasillos)
area_bess = bess_mw * 30 if use_bess else 0

# Logística: Zona de descarga y tanques
area_logistics = 200 
if is_rental: area_logistics += 300 # Patio de maniobras extra

total_area = area_gen + area_bess + area_logistics

# Verificación contra límite
area_status = "✅ OK"
max_area_check = max_area_m2 if 'max_area_m2' in locals() else 0

if max_area_check > 0 and total_area > max_area_check:
    area_status = "❌ OVERFLOW"

# 3. ACTUALIZACIÓN DE COSTOS (Inyectar SCR en el Flujo de Caja)
# Si se requiere SCR, sumamos su costo a la movilización (One-Time) o mensualidad
if req_scr:
    if is_rental:
        # En renta, el SCR suele cobrarse como un fee inicial o premium mensual
        # Aquí lo sumamos al costo de movilización para el análisis
        if 'mob_cost' in locals():
            mob_cost += scr_capex

st.title("🏭 CAT Bridge Power Enterprise")

# KPIs Principales
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
            st.caption(f"BESS provides equivalent of {bess_credit_units} generators in reserve.")
    with c2:
        st.subheader("Load Analysis")
        st.metric("Peak Load", f"{p_total_peak:.1f} MW")
        st.metric("Avg Load", f"{p_total_avg:.1f} MW")
        st.metric("Step Load", f"{p_total_avg * load_step_pct/100:.1f} MW ({load_step_pct}%)")

with t2:
    st.subheader("💰 Financial Decision Analysis")
    
    # --- 1. TIME TO MARKET (COST OF DELAY) ---
    if enable_ttm and grid_delay_mo > 0:
        # CÁLCULOS
        total_revenue_gained = p_it * revenue_per_mw_mo * grid_delay_mo
        cost_bridge_total = monthly_bill * grid_delay_mo
        cost_grid_total = (p_total_avg * 730 * 0.08) * grid_delay_mo # Asumiendo $0.08 red
        
        premium_paid = cost_bridge_total - cost_grid_total
        net_benefit = total_revenue_gained - premium_paid
        roi_ttm = (net_benefit / premium_paid) * 100 if premium_paid > 0 else 0
        
        c_ttm1, c_ttm2 = st.columns([1, 2])
        
        with c_ttm1:
            st.metric("Revenue Gained (Early Start)", f"${total_revenue_gained/1e6:,.1f} M", f"{grid_delay_mo} Months early")
            st.metric("Bridge Power Premium", f"-${premium_paid/1e6:,.1f} M", "Cost over Grid")
            st.metric("NET PROJECT BENEFIT", f"${net_benefit/1e6:,.1f} M", f"ROI: {roi_ttm:.0f}%", delta_color="normal")
            
        with c_ttm2:
            fig_waterfall = go.Figure(go.Waterfall(
                name = "20", orientation = "v",
                measure = ["relative", "relative", "total"],
                x = ["Revenue Gained", "Rental Premium Cost", "Net Benefit"],
                textposition = "outside",
                text = [f"+${total_revenue_gained/1e6:.1f}M", f"-${premium_paid/1e6:.1f}M", f"${net_benefit/1e6:.1f}M"],
                y = [total_revenue_gained, -premium_paid, net_benefit],
                connector = {"line":{"color":"rgb(63, 63, 63)"}},
            ))
            fig_waterfall.update_layout(title = "The Cost of Waiting vs. Starting Now", height=300)
            st.plotly_chart(fig_waterfall, use_container_width=True)
            
        st.divider()

    # --- 2. RENT VS BUY ANALYSIS ---
    st.subheader("Rent vs. Buy Analysis (Cumulative Cash Flow)")
    
    months_proj = list(range(1, 61))
    
    # RENTA: Incluye Mob al inicio
    cum_rent = [mob_cost + (monthly_bill * m) for m in months_proj]
    
    # COMPRA: CAPEX inicial + OPEX bajo
    purchase_capex = installed_mw * 1000 * 800 # Est. $800/kW
    if use_bess: purchase_capex += (bess_mw * 1000 * 300)
    monthly_opex_purchase = fuel_cost_mo + (installed_mw * 1000 * 15) 
    cum_buy = [purchase_capex + (monthly_opex_purchase * m) for m in months_proj]
    
    # Breakeven
    breakeven_month = next((i for i, (r, b) in enumerate(zip(cum_rent, cum_buy)) if r > b), None)
    
    c_chart, c_kpi = st.columns([2, 1])
    
    with c_chart:
        fig_fin = px.line(x=months_proj, y=[cum_rent, cum_buy], labels={"x": "Months", "value": "Cumulative Cost ($)"})
        fig_fin.data[0].name = "Rental Scenario"
        fig_fin.data[1].name = "Purchase Scenario"
        fig_fin.update_layout(title="Cumulative Cash Flow Comparison")
        if breakeven_month:
             fig_fin.add_vline(x=breakeven_month, line_dash="dot", annotation_text="Breakeven")
        st.plotly_chart(fig_fin, use_container_width=True)
        
    with c_kpi:
        if is_rental:
            st.metric("Total Contract Value", f"${total_contract_value/1e6:.1f} M", f"{contract_months} Months")
            st.metric("One-Time Costs", f"${(mob_cost+demob_cost)/1000:,.0f} k", "Mob + Demob")
            if breakeven_month:
                st.info(f"📉 **Breakeven:** Month {breakeven_month}")
            
            # Desglose OPEX
            fin_df = pd.DataFrame({
                "Category": ["Gen Rental", "BESS Rental", "Fuel (Est)"],
                "Monthly Cost": [gen_rent_mo, bess_rent_mo, fuel_cost_mo]
            })
            st.dataframe(fin_df.style.format({"Monthly Cost": "${:,.0f}"}), use_container_width=True, hide_index=True)

    if is_rental:
        st.info(f"💡 **Exit Strategy:** Buyout Option at Month {contract_months}: **${buyout_price/1e6:.2f} M** ({buyout_pct}% of Asset Value)")

    with t3:
        st.subheader("⚙️ Technical Engineering Analysis")
    
    # Definimos las columnas con nombres seguros
    c_tech1, c_tech2 = st.columns(2)
    
    with c_tech1:
        st.markdown("#### 🏗️ Site Footprint & Logistics")
        # Verificación de seguridad por si faltan los cálculos previos
        if 'total_area' in locals():
            c_ft1, c_ft2 = st.columns(2)
            c_ft1.metric("Total Area Required", f"{total_area:,.0f} m²", area_status)
            c_ft2.metric("Power Density", f"{installed_mw/total_area*1000:.1f} kW/m²")
            
            # Tabla de Áreas
            footprint_df = pd.DataFrame({
                "Zone": ["Generation Hall", "BESS Containers", "Logistics/Fuel"],
                "Area (m²)": [area_gen, area_bess, area_logistics]
            })
            st.dataframe(footprint_df, use_container_width=True, hide_index=True)
            
            if area_status == "❌ OVERFLOW":
                st.error(f"Site limit exceeded by {total_area - max_area_m2:,.0f} m²!")
        else:
            st.warning("⚠️ Calculation Engine not updated yet. Please apply Step 2 (Calculations).")

    # Usamos c_tech2 consistentemente
    with c_tech2:
        st.markdown("#### 🌍 Emissions & Compliance")
        if 'nox_ton_yr' in locals():
            c_em1, c_em2 = st.columns(2)
            c_em1.metric("NOx Potential", f"{nox_ton_yr:.1f} Ton/yr", f"Raw: {gen_data['emissions_nox']} g/bhp-hr")
            
            if req_scr:
                c_em2.error("SCR System Required")
                st.warning(f"⚠️ Strict limits require Aftertreatment (SCR).")
                st.write(f"• **SCR CAPEX:** ${scr_capex/1e6:.2f} M (Added to Mob)")
                st.write(f"• **Urea OPEX:** ${urea_opex_mo:,.0f} / month")
            else:
                c_em2.success("Standard Compliance OK")
                st.caption("Engine meets limits without extra hardware.")
        else:
            st.warning("⚠️ Waiting for emissions calculation...")

    st.divider()
    
    st.subheader("⚡ Transient Physics (Deep Dive)")
    c_phys1, c_phys2 = st.columns([1, 2])
    
    with c_phys1:
        st.metric("Voltage Sag", f"{voltage_sag:.2f}%", "Limit < 15%")
        st.metric("Step Load Capability", f"{p_total_avg * load_step_pct/100:.1f} MW", f"{load_step_pct}% Step")
    
    with c_phys2:
        if use_bess:
            st.markdown("**🔋 BESS Sizing Logic:**")
            if 'bess_bkdn' in locals() and bess_bkdn:
                # Gráfica de BESS
                bess_chart_data = pd.DataFrame({
                    "Driver": list(bess_bkdn.keys()),
                    "Power Req (MW)": list(bess_bkdn.values())
                })
                # Filtro para limpiar gráfica
                bess_chart_data = bess_chart_data[bess_chart_data["Power Req (MW)"] > 0.01]
                st.bar_chart(bess_chart_data.set_index("Driver"))
            else:
                st.info("BESS breakdown data not available.") 
    
        # Asegúrate de que esta línea esté alineada dentro de 'with t3:'
    if use_bess:
        st.markdown("### 🔋 BESS Sizing Breakdown")
        
        # 1. Preparar datos para gráfica
        bess_chart_data = pd.DataFrame({
            "Driver": list(bess_bkdn.keys()),
            "Power Req (MW)": list(bess_bkdn.values())
        })
        
        # 2. Filtrar valores insignificantes
        bess_chart_data = bess_chart_data[bess_chart_data["Power Req (MW)"] > 0.01]
        
        # 3. Crear Gráfico
        fig_bess = px.bar(
            bess_chart_data, 
            x="Driver", 
            y="Power Req (MW)",
            text="Power Req (MW)",
            title="BESS Sizing Drivers (MW required per function)",
            color="Driver",
            color_discrete_sequence=px.colors.qualitative.Safe
        )
        
        # 4. Estilizar
        fig_bess.update_traces(texttemplate='%{text:.1f} MW', textposition='outside')
        fig_bess.update_layout(showlegend=False, height=350, margin=dict(l=20, r=20, t=40, b=20))
        
        # 5. Renderizar
        st.plotly_chart(fig_bess, use_container_width=True)
        
        # Explicación contextual
        # Verificamos si bess_bkdn tiene datos antes de buscar el máximo
        if bess_bkdn:
            driver_max = max(bess_bkdn, key=bess_bkdn.get)
            st.caption(f"ℹ️ The BESS is sized to meet the largest requirement: **{driver_max} ({bess_bkdn[driver_max]:.1f} MW)**.")
        
        # Explicación contextual
        # Verificamos si bess_bkdn tiene datos antes de buscar el máximo
        if bess_bkdn:
            driver_max = max(bess_bkdn, key=bess_bkdn.get)
            st.caption(f"ℹ️ The BESS is sized to meet the largest requirement: **{driver_max} ({bess_bkdn[driver_max]:.1f} MW)**.")  
            # Explicación contextual
            driver_max = max(bess_bkdn, key=bess_bkdn.get)
            st.caption(f"ℹ️ The BESS is sized to meet the largest requirement: **{driver_max} ({bess_bkdn[driver_max]:.1f} MW)**.")   
            
    with c_tech2:
        st.write("**Efficiency & Fuel**")
        st.metric("Real Fleet Eff", f"{fleet_eff*100:.1f}%", f"Base: {gen_data['electrical_efficiency']*100:.1f}%")
        st.write(f"Fuel Consumption: {fuel_mmbtu_hr:,.0f} MMBtu/hr")
        if load_pct < 40:
            st.warning("⚠️ Low Load Warning: Wet Stacking Risk")

with t4:
    st.header("📄 Executive Report Generation")
    
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
            ["Total Contract Value", f"${total_contract_value:,.0f}" if is_rental else f"${total_capex:,.0f}"]
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
        story.append(Spacer(1, 0.2*inch))
        story.append(Paragraph("Technical Specifications", styles['Heading2']))
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
        
        # Disclaimer
        story.append(Spacer(1, 0.5*inch))
        story.append(Paragraph("Generated by CAT Bridge Power Enterprise v10.0", styles['Italic']))
        
        doc.build(story)
        buffer.seek(0)
        return buffer.getvalue()

    if st.button("📥 Generate PDF Report"):
        pdf_bytes = generate_bridge_pdf()
        st.download_button(
            label="Download PDF Report",
            data=pdf_bytes,
            file_name=f"CAT_Bridge_Report_{datetime.now().strftime('%Y%m%d')}.pdf",
            mime="application/pdf"
        )

# --- FOOTER ---
st.markdown("---")
st.caption("Calculation Engine: Fusion of V9.3 Business Logic + V3.0 Physics Core")











