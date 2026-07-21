# -*- coding: utf-8 -*-
"""
Dashboard de Correlación Fluvial vs. Índices ENSO (NOAA)
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
    """Obtiene la lista de estaciones desde la API del INA (Versión Estable)"""
    try:
        url_estaciones = "https://alerta.ina.gob.ar/pub/datos/estaciones?auto=true&redId=10&format=json"
        response = requests.get(url_estaciones)
        response.raise_for_status() 
        data_json = response.json()
        
        df_estaciones = pd.DataFrame(data_json['data'])
        df_estaciones = df_estaciones[['sitecode', 'nombre']].dropna()
        df_estaciones['sitecode'] = df_estaciones['sitecode'].astype(int)
        df_estaciones = df_estaciones.sort_values(by='nombre')
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
    """Consulta la base de datos PostgreSQL local"""
    try:
        conn = psycopg2.connect("dbname='meteorology' user='sololectura' host='correo.ina.gob.ar' port='9049'")
        sql_query = '''SELECT timestart as fecha, valor as nivel FROM alturas_all 
                       WHERE timestart BETWEEN %s AND %s AND unid=%s '''
        df = pd.read_sql_query(sql_query, conn, params=[f_inicio, f_fin, int(station_id)])
        conn.close()
        
        if not df.empty:
            df.loc[df['nivel'] <= -900, 'nivel'] = np.nan
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
        "Estación de Río (Eje Izquierdo - Azul):", 
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
        "Índice de El Niño (Eje Derecho - Rojo):",
        options=[info['name'] for info in NOAA_INDICES.values()]
    )
    selected_noaa_key = [k for k, v in NOAA_INDICES.items() if v['name'] == noaa_option_label][0]

time_range = st.radio(
    "Período de Visualización (Estándar e Históricos El Niño):",
    options=[
        "Últimos 2 Años", "Últimos 5 Años", 
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
    
    noaa_min_val = df_noaa_full['value'].min()
    noaa_max_val = df_noaa_full['value'].max()
    
    if pd.isna(noaa_min_val) or pd.isna(noaa_max_val):
        range_y_noaa = [-3.0, 3.0]
    else:
        span_noaa = noaa_max_val - noaa_min_val
        range_y_noaa = [float(noaa_min_val - (span_noaa * 0.05)), float(noaa_max_val + (span_noaa * 0.05))]

    # Selección de rango de fechas
    today = date.today()
    if time_range == "Últimos 2 Años":
        f_inicio = f"{today.year - 2}-{today.month:02d}-{today.day:02d}"
        f_fin = today.strftime("%Y-%m-%d")
        title_suffix = "Últimos 2 Años"
    elif time_range == "Últimos 5 Años":
        f_inicio = f"{today.year - 5}-{today.month:02d}-{today.day:02d}"
        f_fin = today.strftime("%Y-%m-%d")
        title_suffix = "Últimos 5 Años"
    elif time_range == "Niño 1982-1983":
        f_inicio, f_fin = "1982-01-01", "1983-12-31"
        title_suffix = "Evento El Niño 1982-1983"
    elif time_range == "Niño 1991-1992":
        f_inicio, f_fin = "1991-01-01", "1992-12-31"
        title_suffix = "Evento El Niño 1991-1992"
    elif time_range == "Niño 1997-1998":
        f_inicio, f_fin = "1997-01-01", "1998-12-31"
        title_suffix = "Evento El Niño 1997-1998"
    elif time_range == "Niño 2015-2016":
        f_inicio, f_fin = "2015-01-01", "2016-12-31"
        title_suffix = "Evento El Niño 2015-2016"
    else:  # Serie Completa
        f_inicio = df_noaa_full['fecha'].min().strftime("%Y-%m-%d")
        f_fin = today.strftime("%Y-%m-%d")
        title_suffix = "Serie Histórica Completa"

    # Consultar datos de Río
    df_rio = fetch_river_data(f_inicio, f_fin, station_id)
    df_noaa_filtered = df_noaa_full[(df_noaa_full['fecha'] >= f_inicio) & (df_noaa_full['fecha'] <= f_fin)]

    # Crear gráfico Plotly
    fig = make_subplots(specs=[[{"secondary_y": True}]])

    # 1. Serie Río
    if not df_rio.empty:
        df_rio['fecha'] = pd.to_datetime(df_rio['fecha'])
        df_rio = df_rio.set_index('fecha')

        if river_agg == "Media Mensual":
            df_rio = df_rio.resample('M').mean().reset_index()
            mode_rio, marker_config = 'lines+markers', {'size': 5}
            name_suffix, hover_fmt = "(Media Mensual)", "%b-%Y"
        else:
            df_rio = df_rio.resample('1D').mean().reset_index()
            mode_rio, marker_config = 'lines', {}
            name_suffix, hover_fmt = "(Diario)", "%d-%b-%Y"

        fig.add_trace(
            dict(
                x=df_rio['fecha'],
                y=df_rio['nivel'],
                mode=mode_rio,
                marker=marker_config,
                line={'width': 2.5, 'color': '#1d4ed8'},
                hovertemplate=f"<b>{estacion_nombre}</b><br>Fecha: %{{x|{hover_fmt}}}<br>Nivel: %{{y:.2f}} m<extra></extra>",
                name=f"Nivel {estacion_nombre} {name_suffix}"
            ),
            secondary_y=False
        )

    # 2. Serie NOAA
    if not df_noaa_filtered.empty:
        df_noaa_clean = df_noaa_filtered.dropna(subset=['value'])
        fig.add_trace(
            dict(
                x=df_noaa_clean['fecha'],
                y=df_noaa_clean['value'],
                mode='lines+markers',
                line={'width': 2, 'color': '#dc2626', 'dash': 'dash'},
                marker={'size': 6, 'color': '#dc2626'},
                hovertemplate=f"<b>{index_meta['name']}</b><br>Mes: %{{x|%b-%Y}}<br>Valor: %{{y:.2f}}<extra></extra>",
                name=index_meta['name']
            ),
            secondary_y=True
        )

    # Layout de la figura (Sintaxis actualizada)
    fig.update_layout(
        title={
            'text': f'Análisis: <b>{estacion_nombre}</b> vs <b>{index_meta["name"]}</b> ({title_suffix})',
            'y': 0.95, 'x': 0.5, 'xanchor': 'center', 'font': {'size': 18}
        },
        xaxis={'title': 'Línea de Tiempo', 'type': 'date', 'showgrid': True},
        yaxis={
            'title': f'Nivel {estacion_nombre} (m)', 
            'title_font': {'color': '#1d4ed8'}, 
            'tickfont': {'color': '#1d4ed8'}
        },
        yaxis2={
            'title': f"{index_meta['name']} ({index_meta['unit']})",
            'title_font': {'color': '#dc2626'}, 
            'tickfont': {'color': '#dc2626'},
            'range': range_y_noaa, 
            'overlaying': 'y', 
            'side': 'right'
        },
        plot_bgcolor='white', paper_bgcolor='white',
        legend={'orientation': 'h', 'yanchor': 'top', 'y': -0.15, 'xanchor': 'center', 'x': 0.5},
        hovermode='x unified',
        height=550
    )

    st.plotly_chart(fig, use_container_width=True)

else:
    st.warning("No se pudieron cargar los datos del índice de la NOAA en este momento. Intente recargar la página.")

# --- 6. PIE DE PÁGINA ---
st.caption("Datos fluviales provistos por el **SIyAH - Instituto Nacional del Agua (INA)** | Índices climáticos provistos por la **NOAA Physical Sciences Laboratory (PSL)**")