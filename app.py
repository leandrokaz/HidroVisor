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
    """Obtiene la lista de estaciones desde la API del INA filtrando varId (2 y 39)"""
    try:
        url_estaciones = "https://alerta.ina.gob.ar/pub/datos/estaciones?auto=true&redId=10&format=json"
        response = requests.get(url_estaciones, timeout=10)
        response.raise_for_status() 
        data_json = response.json()
        
        df_estaciones = pd.DataFrame(data_json['data'])
        
        if 'tipo' in df_estaciones.columns:
            df_estaciones = df_estaciones[df_estaciones['tipo'].isin(['H'] or ['A'])]
            
        df_estaciones = df_estaciones[['sitecode', 'nombre']].dropna()
        df_estaciones['sitecode'] = df_estaciones['sitecode'].astype(int)
        df_estaciones = df_estaciones.drop_duplicates(subset=['sitecode'])
        
        return df_estaciones
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
    try:
        conn = psycopg2.connect("dbname='meteorology' user='sololectura' host='correo.ina.gob.ar' port='9049'")
        
        # 1. Filtro base en SQL (elimina de entrada valores absurdos como -9999 o >999)
        sql_query = '''
            SELECT timestart as fecha, valor as nivel 
            FROM alturas_all 
            WHERE timestart BETWEEN %s AND %s 
              AND unid = %s 
              AND valor BETWEEN -50 AND 50
        '''
        df = pd.read_sql_query(sql_query, conn, params=[f_inicio, f_fin, int(station_id)])
        conn.close()
        
        if not df.empty:
            # 2. Filtro dinámico vectorizado por Z-Score Modificado (IQR/Mediana)
            median = df['nivel'].median()
            std = df['nivel'].std()
            
            if pd.notna(std) and std > 0:
                df.loc[np.abs(df['nivel'] - median) > (4 * std), 'nivel'] = np.nan
                
        return df
    except Exception as e:
        st.error(f"Error consultando BBDD de hidrometría: {e}")
        return pd.DataFrame()

# --- 3. ENCABEZADO Y LOGO ---
col_head1, col_head2 = st.columns([3, 1])
with col_head1:
    st.title("Seguimiento Hidrológico del Evento de El Niño")
    st.caption("Análisis comparativo temporal multianual entre hidrómetros locales y variables globales del Pacífico")

with col_head2:
    st.image("https://raw.githubusercontent.com/leandrokaz/HidroVisor/main/ina_version1.jpg", width=220)


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
    
    noaa_min_val = df_noaa_full['value'].min()
    noaa_max_val = df_noaa_full['value'].max()
    
    if pd.isna(noaa_min_val) or pd.isna(noaa_max_val):
        range_y_noaa = [-3.0, 3.0]
    else:
        span_noaa = noaa_max_val - noaa_min_val
        range_y_noaa = [float(noaa_min_val - (span_noaa * 0.05)), float(noaa_max_val + (span_noaa * 0.05))]

    today = date.today()
    is_event_comparison = time_range.startswith("Niño")
    
    # Selección de rango de fechas para el evento o período histórico
    if time_range == "Últimos 2 Años":
        f_inicio = f"{today.year - 2}-{today.month:02d}-{today.day:02d}"
        f_fin = today.strftime("%Y-%m-%d")
        title_suffix = "Últimos 2 Años"
    elif time_range == "Últimos 10 Años":
        f_inicio = f"{today.year - 10}-{today.month:02d}-{today.day:02d}"
        f_fin = today.strftime("%Y-%m-%d")
        title_suffix = "Últimos 10 Años"
    elif time_range == "Niño 1982-1983":
        f_inicio, f_fin = "1982-01-01", "1983-12-31"
        start_year_ref = 1982
        title_suffix = "Evento El Niño 1982-1983 vs Actual (2026)"
    elif time_range == "Niño 1991-1992":
        f_inicio, f_fin = "1991-01-01", "1992-12-31"
        start_year_ref = 1991
        title_suffix = "Evento El Niño 1991-1992 vs Actual (2026)"
    elif time_range == "Niño 1997-1998":
        f_inicio, f_fin = "1997-01-01", "1998-12-31"
        start_year_ref = 1997
        title_suffix = "Evento El Niño 1997-1998 vs Actual (2026)"
    elif time_range == "Niño 2015-2016":
        f_inicio, f_fin = "2015-01-01", "2016-12-31"
        start_year_ref = 2015
        title_suffix = "Evento El Niño 2015-2016 vs Actual (2026)"
    else:  # Serie Completa
        f_inicio = df_noaa_full['fecha'].min().strftime("%Y-%m-%d")
        f_fin = today.strftime("%Y-%m-%d")
        title_suffix = "Serie Histórica Completa"

    # Consultar datos de Río e Índice para el período seleccionado
    df_rio = fetch_river_data(f_inicio, f_fin, station_id)
    df_noaa_filtered = df_noaa_full[(df_noaa_full['fecha'] >= f_inicio) & (df_noaa_full['fecha'] <= f_fin)]

    # Si se selecciona un evento histórico El Niño, consultamos adicionalmente la serie actual (Enero 2026 a la fecha)
    if is_event_comparison:
        f_inicio_curr = "2026-01-01"
        f_fin_curr = today.strftime("%Y-%m-%d")
        df_rio_curr = fetch_river_data(f_inicio_curr, f_fin_curr, station_id)
        df_noaa_curr = df_noaa_full[(df_noaa_full['fecha'] >= f_inicio_curr) & (df_noaa_full['fecha'] <= f_fin_curr)]

    # Crear gráfico Plotly
    fig = make_subplots(specs=[[{"secondary_y": True}]])

    # Función interna para reasignar fechas de 2026 a la escala temporal del evento de referencia
    def map_2026_to_ref(dt, start_yr):
        try:
            return dt.replace(year=start_yr)
        except ValueError:  # Maneja el 29 de febrero si el año base no es bisiesto
            return dt.replace(year=start_yr, day=28)

    # 1. Serie Río (Histórica / Período Seleccionado)
    if not df_rio.empty:
        df_rio['fecha'] = pd.to_datetime(df_rio['fecha'])
        df_rio = df_rio.set_index('fecha')

        if river_agg == "Media Mensual":
            df_rio = df_rio.resample('ME').mean().reset_index()
            mode_rio, marker_config = 'lines+markers', {'size': 5}
            hover_fmt = "%b-%Y"
        else:
            df_rio = df_rio.resample('1D').mean().reset_index()
            mode_rio, marker_config = 'lines', {}
            hover_fmt = "%d-%b-%Y"

        rio_color = '#93c5fd' if is_event_comparison else '#1d4ed8'
        rio_dash = 'dot' if is_event_comparison else 'solid'
        rio_label = f"Nivel {estacion_nombre} ({time_range})" if is_event_comparison else f"Nivel {estacion_nombre}"

        fig.add_trace(
            dict(
                x=df_rio['fecha'],
                y=df_rio['nivel'],
                mode=mode_rio,
                marker=marker_config,
                line={'width': 2 if is_event_comparison else 2.5, 'color': rio_color, 'dash': rio_dash},
                hovertemplate=f"<b>{estacion_nombre} ({time_range})</b><br>Fecha: %{{x|{hover_fmt}}}<br>Nivel: %{{y:.2f}} m<extra></extra>",
                name=rio_label
            ),
            secondary_y=False
        )

    # 1b. Serie Río Actual (Superposición 2026 sobre el Evento Histórico)
    if is_event_comparison and not df_rio_curr.empty:
        df_rio_curr['fecha'] = pd.to_datetime(df_rio_curr['fecha'])
        df_rio_curr = df_rio_curr.set_index('fecha')

        if river_agg == "Media Mensual":
            df_rio_curr = df_rio_curr.resample('ME').mean().reset_index()
            mode_curr, marker_curr = 'lines+markers', {'size': 6}
        else:
            df_rio_curr = df_rio_curr.resample('1D').mean().reset_index()
            mode_curr, marker_curr = 'lines', {}

        df_rio_curr['fecha_mapped'] = df_rio_curr['fecha'].apply(lambda d: map_2026_to_ref(d, start_year_ref))

        fig.add_trace(
            dict(
                x=df_rio_curr['fecha_mapped'],
                y=df_rio_curr['nivel'],
                mode=mode_curr,
                marker=marker_curr,
                line={'width': 3, 'color': '#1d4ed8'},  # Azul oscuro solido
                hovertemplate=f"<b>{estacion_nombre} (Actual 2026)</b><br>Día/Mes: %{{x|%d-%b}}<br>Nivel: %{{y:.2f}} m<extra></extra>",
                name=f"Nivel {estacion_nombre} (Actual 2026)"
            ),
            secondary_y=False
        )

    # 2. Serie NOAA (Histórica / Período Seleccionado)
    if not df_noaa_filtered.empty:
        df_noaa_clean = df_noaa_filtered.dropna(subset=['value'])
        noaa_color = '#fca5a5' if is_event_comparison else '#dc2626'
        noaa_label = f"{index_meta['name']} ({time_range})" if is_event_comparison else index_meta['name']

        fig.add_trace(
            dict(
                x=df_noaa_clean['fecha'],
                y=df_noaa_clean['value'],
                mode='lines+markers',
                line={'width': 1.5 if is_event_comparison else 2, 'color': noaa_color, 'dash': 'dash'},
                marker={'size': 4 if is_event_comparison else 6, 'color': noaa_color},
                hovertemplate=f"<b>{index_meta['name']}</b><br>Mes: %{{x|%b-%Y}}<br>Valor: %{{y:.2f}}<extra></extra>",
                name=noaa_label
            ),
            secondary_y=True
        )

    # 2b. Serie NOAA Actual (Superposición 2026 sobre el Evento Histórico)
    if is_event_comparison and not df_noaa_curr.empty:
        df_noaa_curr_clean = df_noaa_curr.dropna(subset=['value'])
        if not df_noaa_curr_clean.empty:
            df_noaa_curr_clean['fecha_mapped'] = df_noaa_curr_clean['fecha'].apply(lambda d: map_2026_to_ref(d, start_year_ref))

            fig.add_trace(
                dict(
                    x=df_noaa_curr_clean['fecha_mapped'],
                    y=df_noaa_curr_clean['value'],
                    mode='lines+markers',
                    line={'width': 2.5, 'color': '#dc2626', 'dash': 'solid'},  # Rojo solido prominente
                    marker={'size': 7, 'color': '#dc2626'},
                    hovertemplate=f"<b>{index_meta['name']} (Actual 2026)</b><br>Mes: %{{x|%b}}<br>Valor: %{{y:.2f}}<extra></extra>",
                    name=f"{index_meta['name']} (Actual 2026)"
                ),
                secondary_y=True
            )

    # Layout de la figura
    fig.update_layout(
        title={
            'text': f'Análisis de Nivel Hidrométrico en <b>{estacion_nombre}</b> vs <b>{index_meta["name"]}</b> ({title_suffix})',
            'y': 0.95, 'x': 0.5, 'xanchor': 'center', 'font': {'size': 18}
        },
        xaxis={'title': 'Línea de Tiempo', 'type': 'date', 'showgrid': True, 'gridcolor': '#f0f0f0'},
        yaxis={
            'title': f'Nivel en estación hidrométrica {estacion_nombre} (m)', 
            'title_font': {'color': '#1d4ed8'}, 
            'tickfont': {'color': '#1d4ed8'},
            'showgrid': True,
            'gridcolor': '#e5e7eb'
        },
        yaxis2={
            'title': f"{index_meta['name']} ({index_meta['unit']})",
            'title_font': {'color': '#dc2626'}, 
            'tickfont': {'color': '#dc2626'},
            'range': range_y_noaa, 
            'overlaying': 'y', 
            'side': 'right',
            'showgrid': False,     # <-- Elimina la grilla del eje secundario
            'zeroline': False      # <-- ELIMINA LA LÍNEA DEL VALOR CERO EN EL EJE SECUNDARIO
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
st.caption("Datos hidrológicos provistos por el **SIyAH - Instituto Nacional del Agua (INA)** | Índices climáticos provistos por la **NOAA Physical Sciences Laboratory (PSL)**")