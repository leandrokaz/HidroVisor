# -*- coding: utf-8 -*-
"""
Dashboard de Correlación Fluvial vs. Índices ENSO (NOAA)
Superposición de Serie Actual (2026) vs Eventos Históricos
Desarrollado con Streamlit
"""

import streamlit as st
import pandas as pd
import numpy as np
import psycopg2
import requests
from datetime import date
from plotly.subplots import make_subplots

# --- 1. CONFIGURACIÓN DE LA PÁGINA ---
st.set_page_config(
    page_title="SIyAH - Correlación Río vs NOAA",
    page_icon="🌊",
    layout="wide"
)

# --- DICCIONARIO DE ÍNDICES NOAA ---
NOAA_INDICES = {
    'NINO34': {
        'name': 'Multivariate ENSO Index (MEI V2)',
        'url': 'https://psl.noaa.gov/data/correlation/meiv2.data',
        'unit': 'std'
    },
    'ONI': {
        'name': 'Oceanic Niño Index (ONI)',
        'url': 'https://psl.noaa.gov/data/correlation/oni.data',
        'unit': 'Anomalía (°C)'
    },
    'SOI': {
        'name': 'Southern Oscillation Index (SOI - Atmosférico)',
        'url': 'https://psl.noaa.gov/data/correlation/soi.data',
        'unit': 'Index Unit (sigma)'
    }
}

# --- 2. FUNCIONES DE CARGA Y PARSEO CON CACHÉ ---

@st.cache_data(ttl=3600)
def fetch_estaciones():
    """Obtiene la lista de estaciones desde la API del INA"""
    try:
        url_estaciones = "https://alerta.ina.gob.ar/pub/datos/estaciones?auto=true&redId=10&format=json"
        response = requests.get(url_estaciones, timeout=10)
        response.raise_for_status() 
        data_json = response.json()
        
        df_estaciones = pd.DataFrame(data_json['data'])
        
        if 'tipo' in df_estaciones.columns:
            df_estaciones = df_estaciones[df_estaciones['tipo'].isin(['H', 'A'])]
            
        df_estaciones = df_estaciones[['sitecode', 'nombre']].dropna()
        df_estaciones['sitecode'] = df_estaciones['sitecode'].astype(int)
        df_estaciones = df_estaciones.drop_duplicates(subset=['sitecode'])
        
        return df_estaciones.sort_values(by='nombre')
    except Exception as e:
        print("Error al consumir la API de estaciones:", e)
        return pd.DataFrame([{'sitecode': 34, 'nombre': 'Pto. Pilcomayo (río Paraguay)'}])
    
    
@st.cache_data(ttl=86400)
def download_and_parse_noaa_index(url):
    """Descarga y parsea el formato PSL de la NOAA"""
    try:
        res = requests.get(url, timeout=10)
        res.raise_for_status()
        lines = res.text.split('\n')
        
        header = lines[0].split()
        if len(header) < 2:
            return pd.DataFrame()
        start_year, end_year = int(header[0]), int(header[1])
        
        records = []
        for line in lines[1:]:
            parts = line.split()
            if not parts:
                continue
            if len(parts) == 1:
                break
            if len(parts) >= 13:
                try:
                    yr = int(parts[0])
                    if yr < start_year or yr > end_year:
                        continue
                    monthly_vals = [float(x) for x in parts[1:13]]
                    for m_idx, val in enumerate(monthly_vals, start=1):
                        val_cleaned = val if val > -90.0 else np.nan
                        records.append({'year': yr, 'month': m_idx, 'value': val_cleaned})
                except ValueError:
                    continue
                    
        df = pd.DataFrame(records)
        if not df.empty:
            df['fecha'] = pd.to_datetime(df.apply(lambda r: f"{int(r['year'])}-{int(r['month'])}-15", axis=1))
        return df
    except Exception as ex:
        return pd.DataFrame()

def fetch_river_data(f_inicio, f_fin, station_id):
    """Consulta PostgreSQL y filtra datos fuera de rango extremo (-50 a 100m)"""
    try:
        conn = psycopg2.connect("dbname='meteorology' user='sololectura' host='correo.ina.gob.ar' port='9049'")
        sql_query = '''SELECT timestart as fecha, valor as nivel FROM alturas_all 
                       WHERE timestart BETWEEN %s AND %s AND unid=%s 
                       AND valor BETWEEN -50 AND 100'''
                       
        df = pd.read_sql_query(sql_query, conn, params=[f_inicio, f_fin, int(station_id)])
        conn.close()
        
        if not df.empty:
            df.loc[df['nivel'] <= -900, 'nivel'] = np.nan
            
            mediana = df['nivel'].median()
            desvio = df['nivel'].std()
            
            if pd.notna(desvio) and desvio > 0:
                df.loc[np.abs(df['nivel'] - mediana) > (5 * desvio), 'nivel'] = np.nan

        return df
    except Exception as e:
        st.error(f"Error consultando BBDD de hidrometría: {e}")
        return pd.DataFrame()

# --- 3. ENCABEZADO Y LOGO ---
col_head1, col_head2 = st.columns([3, 1])
with col_head1:
    st.title("Correlación Río vs Índice de El Niño (NOAA)")
    st.caption("Análisis comparativo temporal multianual entre hidrómetros locales y variables globales del Pacífico")

with col_head2:
    st.image("https://alerta.ina.gob.ar/img/Logo_SIyAH.png", width=220)

st.markdown("---")

# --- 4. PANEL DE CONTROLES ---
df_estaciones = fetch_estaciones()

c1, c2, c3 = st.columns([2, 1, 2])

with c1:
    estacion_nombre = st.selectbox(
        "Estación de Río (Eje Izquierdo):", 
        options=df_estaciones['nombre'].tolist(),
        index=0
    )
    station_id = int(df_estaciones[df_estaciones['nombre'] == estacion_nombre]['sitecode'].values[0])

with c2:
    river_agg = st.radio(
        "Filtro Temporal Río:", 
        options=["Serie Diaria", "Media Mensual"],
        index=0
    )

with c3:
    noaa_option_label = st.selectbox(
        "Índice de El Niño (Eje Derecho):",
        options=[info['name'] for info in NOAA_INDICES.values()]
    )
    selected_noaa_key = [k for k, v in NOAA_INDICES.items() if v['name'] == noaa_option_label][0]

time_range = st.radio(
    "Período / Evento de Referencia:",
    options=[
        "Últimos 2 Años", "Últimos 10 Años", 
        "Niño 1982-1983", "Niño 1991-1992", "Niño 1997-1998", "Niño 2015-2016", 
        "Serie Completa"
    ],
    horizontal=True
)

st.markdown("---")

# --- 5. LÓGICA DE PROCESAMIENTO Y GRÁFICO ---
index_meta = NOAA_INDICES[selected_noaa_key]
df_noaa_full = download_and_parse_noaa_index(index_meta['url'])

if not df_noaa_full.empty and df_noaa_full['value'].dropna().shape[0] > 0:
    
    today = date.today()
    is_event_comparison = time_range.startswith("Niño")
    
    # 1. Definir rango de fechas según selección
    if time_range == "Últimos 2 Años":
        f_inicio_hist = f"{today.year - 2}-{today.month:02d}-{today.day:02d}"
        f_fin_hist = today.strftime("%Y-%m-%d")
        title_suffix = "Últimos 2 Años"
    elif time_range == "Últimos 10 Años":
        f_inicio_hist = f"{today.year - 10}-{today.month:02d}-{today.day:02d}"
        f_fin_hist = today.strftime("%Y-%m-%d")
        title_suffix = "Últimos 10 Años"
    elif time_range == "Niño 1982-1983":
        f_inicio_hist, f_fin_hist = "1982-01-01", "1983-12-31"
        start_year_ref = 1982
        title_suffix = "Evento El Niño 1982-1983 vs Actual (2026)"
    elif time_range == "Niño 1991-1992":
        f_inicio_hist, f_fin_hist = "1991-01-01", "1992-12-31"
        start_year_ref = 1991
        title_suffix = "Evento El Niño 1991-1992 vs Actual (2026)"
    elif time_range == "Niño 1997-1998":
        f_inicio_hist, f_fin_hist = "1997-01-01", "1998-12-31"
        start_year_ref = 1997
        title_suffix = "Evento El Niño 1997-1998 vs Actual (2026)"
    elif time_range == "Niño 2015-2016":
        f_inicio_hist, f_fin_hist = "2015-01-01", "2016-12-31"
        start_year_ref = 2015
        title_suffix = "Evento El Niño 2015-2016 vs Actual (2026)"
    else:  # Serie Completa
        f_inicio_hist = df_noaa_full['fecha'].min().strftime("%Y-%m-%d")
        f_fin_hist = today.strftime("%Y-%m-%d")
        title_suffix = "Serie Histórica Completa"

    # Consultas de Datos Río y NOAA
    df_rio_hist = fetch_river_data(f_inicio_hist, f_fin_hist, station_id)
    df_noaa_hist = df_noaa_full[(df_noaa_full['fecha'] >= f_inicio_hist) & (df_noaa_full['fecha'] <= f_fin_hist)]

    # Si es evento Niño, consultamos adicionalmente la serie del año actual (2026)
    if is_event_comparison:
        f_inicio_curr = "2026-01-01"
        f_fin_curr = today.strftime("%Y-%m-%d")
        df_rio_curr = fetch_river_data(f_inicio_curr, f_fin_curr, station_id)
        df_noaa_curr = df_noaa_full[(df_noaa_full['fecha'] >= f_inicio_curr) & (df_noaa_full['fecha'] <= f_fin_curr)]

    fig = make_subplots(specs=[[{"secondary_y": True}]])

    # --- MODALIDAD 1: COMPARACIÓN SUPERPUESTA (EVENTOS NIÑO) ---
    if is_event_comparison:
        
        # A. Función auxiliar para normalizar la fecha de 2026 a la línea temporal histórica
        def map_2026_to_ref(dt, start_yr):
            # Mapea 2026 al año de inicio del evento (ej. 1997)
            try:
                return dt.replace(year=start_yr)
            except ValueError: # Manejo para año bisiesto (29 Feb)
                return dt.replace(year=start_yr, day=28)

        # 1. Río Histórico
        if not df_rio_hist.empty:
            df_rio_hist['fecha'] = pd.to_datetime(df_rio_hist['fecha'])
            df_rio_hist = df_rio_hist.set_index('fecha').resample('M' if river_agg == "Media Mensual" else '1D').mean().reset_index()
            fig.add_trace(
                dict(
                    x=df_rio_hist['fecha'], y=df_rio_hist['nivel'], mode='lines',
                    line={'width': 2, 'color': '#93c5fd', 'dash': 'dot'},
                    name=f"Río ({time_range})", connectgaps=True,
                    hovertemplate="<b>Histórico</b><br>Fecha: %{x|%d-%b-%Y}<br>Nivel: %{y:.2f} m<extra></extra>"
                ), secondary_y=False
            )

        # 2. Río Actual (2026 superpuesto)
        if not df_rio_curr.empty:
            df_rio_curr['fecha'] = pd.to_datetime(df_rio_curr['fecha'])
            df_rio_curr = df_rio_curr.set_index('fecha').resample('M' if river_agg == "Media Mensual" else '1D').mean().reset_index()
            df_rio_curr['fecha_mapped'] = df_rio_curr['fecha'].apply(lambda d: map_2026_to_ref(d, start_year_ref))
            
            fig.add_trace(
                dict(
                    x=df_rio_curr['fecha_mapped'], y=df_rio_curr['nivel'], mode='lines+markers',
                    line={'width': 3, 'color': '#1d4ed8'}, marker={'size': 4},
                    name="Río Actual (2026)", connectgaps=True,
                    hovertemplate="<b>Actual (2026)</b><br>Fecha: %{x|%d-%b}<br>Nivel: %{y:.2f} m<extra></extra>"
                ), secondary_y=False
            )

        # 3. NOAA Histórico
        if not df_noaa_hist.empty:
            df_noaa_h_clean = df_noaa_hist.dropna(subset=['value'])
            fig.add_trace(
                dict(
                    x=df_noaa_h_clean['fecha'], y=df_noaa_h_clean['value'], mode='lines',
                    line={'width': 2, 'color': '#fca5a5', 'dash': 'dot'},
                    name=f"NOAA ({time_range})",
                    hovertemplate="<b>NOAA Histórico</b><br>Mes: %{x|%b-%Y}<br>Valor: %{y:.2f}<extra></extra>"
                ), secondary_y=True
            )

        # 4. NOAA Actual (2026 superpuesto)
        if not df_noaa_curr.empty:
            df_noaa_c_clean = df_noaa_curr.dropna(subset=['value'])
            df_noaa_c_clean['fecha_mapped'] = df_noaa_c_clean['fecha'].apply(lambda d: map_2026_to_ref(d, start_year_ref))
            fig.add_trace(
                dict(
                    x=df_noaa_c_clean['fecha_mapped'], y=df_noaa_c_clean['value'], mode='lines+markers',
                    line={'width': 2.5, 'color': '#dc2626'}, marker={'size': 6},
                    name="NOAA Actual (2026)",
                    hovertemplate="<b>NOAA 2026</b><br>Mes: %{x|%b}<br>Valor: %{y:.2f}<extra></extra>"
                ), secondary_y=True
            )

    # --- MODALIDAD 2: VISUALIZACIÓN CONTINUA ESTÁNDAR ---
    else:
        if not df_rio_hist.empty:
            df_rio_hist['fecha'] = pd.to_datetime(df_rio_hist['fecha'])
            df_rio_hist = df_rio_hist.set_index('fecha').resample('ME' if river_agg == "Media Mensual" else '1D').mean().reset_index()
            fig.add_trace(
                dict(
                    x=df_rio_hist['fecha'], y=df_rio_hist['nivel'],
                    mode='lines+markers' if river_agg == "Media Mensual" else 'lines',
                    line={'width': 2.5, 'color': '#1d4ed8'}, connectgaps=True,
                    name=f"Nivel {estacion_nombre}",
                    hovertemplate="<b>%{x|%d-%b-%Y}</b><br>Nivel: %{y:.2f} m<extra></extra>"
                ), secondary_y=False
            )

        if not df_noaa_hist.empty:
            df_noaa_clean = df_noaa_hist.dropna(subset=['value'])
            fig.add_trace(
                dict(
                    x=df_noaa_clean['fecha'], y=df_noaa_clean['value'], mode='lines+markers',
                    line={'width': 2, 'color': '#dc2626', 'dash': 'dash'}, marker={'size': 6},
                    name=index_meta['name'],
                    hovertemplate="<b>%{x|%b-%Y}</b><br>Índice: %{y:.2f}<extra></extra>"
                ), secondary_y=True
            )

    # Ajuste de escala NOAA Y2
    noaa_vals = df_noaa_full['value'].dropna()
    range_y_noaa = [float(noaa_vals.min() - 0.2), float(noaa_vals.max() + 0.2)] if not noaa_vals.empty else [-3.0, 3.0]

    # Layout y Formato de Ejes
    fig.update_layout(
        title={
            'text': f'Análisis: <b>{estacion_nombre}</b> vs <b>{index_meta["name"]}</b><br><sub>{title_suffix}</sub>',
            'y': 0.94, 'x': 0.5, 'xanchor': 'center', 'font': {'size': 18}
        },
        xaxis={'title': 'Eje Temporal (Meses)', 'type': 'date', 'showgrid': True, 'gridcolor': '#f0f0f0'},
        yaxis={
            'title': f'Nivel {estacion_nombre} (m)', 'title_font': {'color': '#1d4ed8'}, 
            'tickfont': {'color': '#1d4ed8'}, 'showgrid': True, 'gridcolor': '#e5e7eb', 'autorange': True
        },
        yaxis2={
            'title': f"{index_meta['name']} ({index_meta['unit']})", 'title_font': {'color': '#dc2626'}, 
            'tickfont': {'color': '#dc2626'}, 'range': range_y_noaa, 'overlaying': 'y', 'side': 'right',
            'showgrid': False, 'zeroline': False
        },
        plot_bgcolor='white', paper_bgcolor='white',
        legend={'orientation': 'h', 'yanchor': 'top', 'y': -0.18, 'xanchor': 'center', 'x': 0.5},
        hovermode='x unified',
        height=580
    )

    st.plotly_chart(fig, use_container_width=True)

else:
    st.warning("No se pudieron cargar los datos de la NOAA en este momento.")

# --- 6. PIE DE PÁGINA ---
st.caption("Datos hidrológicos provistos por el **SIyAH - Instituto Nacional del Agua (INA)** | Índices climáticos provistos por la **NOAA Physical Sciences Laboratory (PSL)**")