# app.py - GeoRiesgo Caracas para Web (Streamlit)
import streamlit as st
import numpy as np
import random
from datetime import datetime
import os

# Configuración de página
st.set_page_config(
    page_title="GeoRiesgo Caracas - SAT",
    page_icon="🚨",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Colores y configuración
PRIMARY_COLOR = "#003865"
ROJO = "#D52B1E"
GRIS_CLARO = "#F8F9FA"

# Mapeo de zonas
ZONES_GEO = {
    "valle_central": {"center": (10.5000, -66.8800), "name": "Valle Central"},
    "ladera_vulnerable": {"center": (10.5150, -66.8950), "name": "Laderas Medias"},
    "cerro_pendiente": {"center": (10.5300, -66.8650), "name": "Cerros Altos"},
    "quebrada": {"center": (10.4950, -66.9100), "name": "Cercanía a Quebradas"},
    "seguro": {"center": (10.4850, -66.8550), "name": "Zonas Consolidadas"}
}

RISK_COLORS = {
    "BAJO": "#90EE90",
    "MODERADO": "#FFFF99", 
    "ALTO": "#FFB366",
    "CRÍTICO": "#FF9999"
}

# Cache para modelo
@st.cache_resource
def load_model():
    """Cargar modelo si está disponible"""
    try:
        model_path = "mejor_modelo_libertador.h5"
        if os.path.exists(model_path):
            import tensorflow as tf
            from tensorflow import keras
            model = keras.models.load_model(model_path, compile=False)
            return model, True
    except Exception as e:
        st.warning(f"⚠️ Modelo no cargado: {e}")
    return None, False

# Funciones de generación de datos sintéticos
def crear_parche_dummy(tipo_zona):
    patch = np.zeros((24, 24, 4))
    if tipo_zona == "valle_central":
        patch[:, :, 0] = np.random.uniform(0.05, 0.15, (24, 24))
        patch[:, :, 1] = np.random.uniform(0.1, 0.25, (24, 24))
        patch[:, :, 2] = np.random.uniform(0.3, 0.5, (24, 24))
        patch[:, :, 3] = np.random.uniform(0.8, 1.0, (24, 24))
    elif tipo_zona == "cerro_pendiente":
        patch[:, :, 0] = np.random.uniform(0.45, 0.55, (24, 24))
        patch[:, :, 1] = np.random.uniform(0.6, 0.9, (24, 24))
        patch[:, :, 2] = np.random.uniform(0.2, 0.4, (24, 24))
        patch[:, :, 3] = np.random.uniform(0.5, 0.7, (24, 24))
    elif tipo_zona == "ladera_vulnerable":
        patch[:, :, 0] = np.random.uniform(0.2, 0.35, (24, 24))
        patch[:, :, 1] = np.random.uniform(0.4, 0.7, (24, 24))
        patch[:, :, 2] = np.random.uniform(0.1, 0.3, (24, 24))
        patch[:, :, 3] = np.random.uniform(0.6, 0.8, (24, 24))
    elif tipo_zona == "quebrada":
        patch[:, :, 0] = np.random.uniform(0.1, 0.25, (24, 24))
        patch[:, :, 1] = np.random.uniform(0.3, 0.6, (24, 24))
        patch[:, :, 2] = np.random.uniform(0.0, 0.07, (24, 24))
        patch[:, :, 3] = np.random.uniform(0.4, 0.6, (24, 24))
    elif tipo_zona == "seguro":
        patch[:, :, 0] = np.random.uniform(0.05, 0.15, (24, 24))
        patch[:, :, 1] = np.random.uniform(0.0, 0.1, (24, 24))
        patch[:, :, 2] = np.random.uniform(0.7, 1.0, (24, 24))
        patch[:, :, 3] = np.random.uniform(0.8, 1.0, (24, 24))
    return patch.astype(np.float32)

def crear_secuencia_climatica(tipo_lluvia):
    seq = np.zeros((1, 24, 5))
    if tipo_lluvia == "seco":
        precip = np.random.uniform(0, 0.5, 24)
        temp = np.random.uniform(24, 28, 24)
        humedad = np.random.uniform(60, 75, 24)
    elif tipo_lluvia == "moderado":
        precip = np.random.gamma(1.5, 1.2, 24)
        temp = 25 - 0.1 * precip + np.random.normal(0, 1, 24)
        humedad = 75 + 0.8 * precip + np.random.normal(0, 5, 24)
    elif tipo_lluvia == "intenso":
        precip = np.random.gamma(2.5, 2.0, 24)
        temp = 24 - 0.2 * precip + np.random.normal(0, 1, 24)
        humedad = 80 + 1.0 * precip + np.random.normal(0, 4, 24)
    elif tipo_lluvia == "extremo":
        precip = np.zeros(24)
        precip[16:] = np.random.gamma(4, 3, 8)
        temp = 23 - 0.3 * precip + np.random.normal(0, 1.5, 24)
        humedad = 85 + 1.2 * precip + np.random.normal(0, 3, 24)
    elif tipo_lluvia == "creciente":
        base = np.linspace(0.2, 3.0, 24)
        precip = base + np.random.normal(0, 0.3, 24)
        temp = 25 - 0.15 * precip + np.random.normal(0, 1.2, 24)
        humedad = 70 + 0.9 * precip + np.random.normal(0, 6, 24)
    
    for i in range(24):
        inicio_24h = max(0, i - 7)
        acum_24h = np.sum(precip[inicio_24h:i + 1]) * 3
        acum_72h = np.sum(precip[:i + 1]) * 3
        seq[0, i, 0] = precip[i]
        seq[0, i, 1] = temp[i]
        seq[0, i, 2] = humedad[i]
        seq[0, i, 3] = acum_24h
        seq[0, i, 4] = acum_72h
    return seq.astype(np.float32)

def analizar_riesgo(zona, clima, model, using_real_model):
    """Ejecutar análisis de riesgo"""
    if using_real_model and model:
        parche = crear_parche_dummy(zona)
        secuencia = crear_secuencia_climatica(clima)
        prob = float(model.predict([secuencia, parche[np.newaxis, ...]], verbose=0)[0][0])
        metodo = "MODELO CNN-LSTM REAL"
    else:
        base = {"valle_central": 0.10, "ladera_vulnerable": 0.45, "cerro_pendiente": 0.65, "quebrada": 0.75, "seguro": 0.15}
        clima_factor = {"seco": 0.5, "moderado": 1.0, "intenso": 1.8, "extremo": 2.5, "creciente": 1.5}
        prob = min(0.95, base[zona] * clima_factor[clima] * random.uniform(0.9, 1.1))
        prob = max(0.05, prob)
        metodo = "SIMULACIÓN REALISTA"
    
    if prob >= 0.7:
        nivel, color, icono, accion = "CRÍTICO", RISK_COLORS["CRÍTICO"], "🚨", "EVACUACIÓN INMEDIATA"
    elif prob >= 0.5:
        nivel, color, icono, accion = "ALTO", RISK_COLORS["ALTO"], "⚠️", "RESTRICCIÓN DE ACCESO"
    elif prob >= 0.3:
        nivel, color, icono, accion = "MODERADO", RISK_COLORS["MODERADO"], "🔶", "VIGILANCIA ESPECIAL"
    else:
        nivel, color, icono, accion = "BAJO", RISK_COLORS["BAJO"], "✅", "VIGILANCIA ESTÁNDAR"
    
    return {"prob": prob, "nivel": nivel, "color": color, "icono": icono, "accion": accion, "metodo": metodo}

# ==================== INTERFAZ PRINCIPAL ====================
def main():
    # Header
    st.markdown(f"""
        <div style='background-color: {PRIMARY_COLOR}; padding: 20px; border-radius: 10px; text-align: center;'>
            <h1 style='color: white; margin: 0;'>📍 GeoRiesgo Caracas</h1>
            <h3 style='color: white; margin: 5px 0 0 0;'>Sistema de Alerta Temprana - Municipio Libertador</h3>
        </div>
    """, unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Configuración")
        
        # Cargar modelo
        model, using_real_model = load_model()
        estado = "✅ CNN-LSTM ACTIVO" if using_real_model else "⚠️ MODO SIMULACIÓN"
        color_estado = "green" if using_real_model else "orange"
        st.markdown(f"<span style='color: {color_estado}; font-weight: bold;'>{estado}</span>", unsafe_allow_html=True)
        
        st.divider()
        
        # Selector de zona
        st.subheader("📍 Zona Geográfica")
        zona = st.radio(
            "Selecciona una zona:",
            options=["valle_central", "ladera_vulnerable", "cerro_pendiente", "quebrada", "seguro"],
            format_func=lambda x: ZONES_GEO[x]["name"],
            index=0
        )
        
        # Selector de clima
        st.subheader("🌦️ Escenario Climático")
        clima = st.radio(
            "Selecciona un escenario:",
            options=["seco", "moderado", "intenso", "extremo", "creciente"],
            format_func=lambda x: {
                "seco": "Seco (< 5 mm)",
                "moderado": "Moderado (15-25 mm)",
                "intenso": "Intenso (35-50 mm) ⚠️",
                "extremo": "Extremo (> 80 mm) 🚨",
                "creciente": "Creciente (alerta temprana)"
            }[x],
            index=1
        )
        
        st.divider()
        
        # Botón de análisis
        if st.button("🔍 ANALIZAR RIESGO", type="primary", use_container_width=True):
            st.session_state['analizar'] = True
        
        st.divider()
        
        # Leyenda
        st.subheader("🎨 Leyenda de Riesgo")
        for nivel, color in RISK_COLORS.items():
            st.markdown(f"<div style='display: flex; align-items: center; gap: 10px;'><div style='width: 20px; height: 20px; background-color: {color}; border-radius: 4px; border: 1px solid #666;'></div><span>{nivel}</span></div>", unsafe_allow_html=True)

    # Área principal
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Mapa placeholder (Streamlit no soporta tkintermapview nativamente)
        st.subheader("🗺️ Mapa Interactivo")
        
        # Opción 1: Usar st.map con datos estáticos
        map_data = []
        for zone_id, zone_info in ZONES_GEO.items():
            map_data.append({
                "lat": zone_info["center"][0],
                "lon": zone_info["center"][1],
                "zona": zone_info["name"]
            })
        
        import pandas as pd
        st.map(pd.DataFrame(map_data), zoom=11, use_container_width=True)
        
        # Nota sobre mapa avanzado
        st.info("💡 Para un mapa interactivo completo con polígonos de riesgo, considera integrar [Folium](https://python-visualization.github.io/folium/) o [Plotly Mapbox](https://plotly.com/python/mapbox-layers/).")
    
    with col2:
        # Resultados
        st.subheader("📊 Resultados")
        
        if st.session_state.get('analizar'):
            resultado = analizar_riesgo(zona, clima, model, using_real_model)
            
            # Tarjeta de resultado
            with st.container(border=True):
                st.markdown(f"<div style='text-align: center; padding: 15px; background-color: {resultado['color']}; border-radius: 10px;'>", unsafe_allow_html=True)
                st.markdown(f"<h1 style='margin: 0;'>{resultado['icono']}</h1>", unsafe_allow_html=True)
                st.markdown(f"<h3 style='margin: 10px 0; color: black;'>{resultado['nivel']}</h3>", unsafe_allow_html=True)
                st.markdown(f"<h2 style='margin: 0; color: black;'>{resultado['prob']*100:.1f}%</h2>", unsafe_allow_html=True)
                st.markdown("</div>", unsafe_allow_html=True)
                
                st.progress(resultado['prob'])
                st.markdown(f"**Acción recomendada:** {resultado['accion']}")
                st.markdown(f"*Método: {resultado['metodo']}*")
                st.markdown(f"*Generado: {datetime.now().strftime('%H:%M:%S')}*")
                
                # Explicación técnica
                with st.expander("💬 Explicación Técnica"):
                    explicaciones = {
                        "CRÍTICO": f"⚠️ RIESGO EXTREMO: Condiciones críticas detectadas. Similar a evento Vargas 1999. Requiere evacuación inmediata.",
                        "ALTO": f"⚠️ RIESGO ELEVADO: Alta probabilidad de deslizamientos en 6-12h. Requiere restricción de acceso.",
                        "MODERADO": f"🔶 RIESGO MODERADO: Condiciones favorables para deslizamientos en sectores vulnerables. Vigilancia cada 2h.",
                        "BAJO": f"✅ RIESGO BAJO: Condiciones estables sin amenaza inminente. Vigilancia estándar suficiente."
                    }
                    st.write(explicaciones[resultado['nivel']])
            
            st.session_state['analizar'] = False
        else:
            st.info("👈 Selecciona zona y escenario, luego haz clic en 'ANALIZAR RIESGO'")
    
    # Footer
    st.divider()
    st.markdown(f"""
        <div style='text-align: center; color: #666; font-size: 0.9em;'>
            GeoRiesgo Caracas v3.1 | Modelo: {'CNN-LSTM REAL' if using_real_model else 'SIMULACIÓN'} | © 2026 UNES
        </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
