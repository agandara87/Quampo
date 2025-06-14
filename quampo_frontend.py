import os
import tempfile
import pathlib
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
from datetime import datetime, timedelta
from quampo_backend import (
    procesar_imagen,
    generar_informe,
    generar_informe_llm,
    geocode_location,
    download_gee_image
)

# ────────────────────────────────────────────────────────────────────
# Configuración del entorno Streamlit
tmp_dir = tempfile.gettempdir()
os.environ["STREAMLIT_GLOBAL_CONFIG_DIR"] = f"{tmp_dir}/.streamlit"
os.environ["MPLCONFIGDIR"] = f"{tmp_dir}/.matplotlib"
os.environ["BROWSER_GATHERUSAGESTATS"] = "false"

st.set_page_config(page_title="Quampo – Análisis Satelital", layout="centered")
st.title("🛰️ Quanmpo – Análisis Satelital de Cultivos")
st.markdown(
    "Sube tu TIFF georreferenciado o, si no subes nada, usaremos GEE para descargar Sentinel-2 directamente."
)

# 1) Carga opcional de imagen TIFF
uploaded = st.file_uploader(
    "📤 Imagen satelital (.tif, .tiff)",
    type=["tif", "tiff"]
)
if uploaded and "img_bytes" not in st.session_state:
    st.session_state.img_bytes = uploaded.read()
    st.session_state.img_name = uploaded.name
    size_kb = len(st.session_state.img_bytes) / 1024
    st.success(f"Archivo **{st.session_state.img_name}** cargado ({size_kb:.1f} KB)")
    try:
        st.image(st.session_state.img_bytes, caption="Imagen cargada")
    except Exception:
        pass

# 2) Parámetros de análisis
date_col, meta_col = st.columns(2)
with date_col:
    fecha = st.date_input("📅 Fecha de referencia", datetime.today().date())
    cultivo = st.text_input("🌾 Cultivo (ej. Maíz, Soja, Barbecho)", "")
with meta_col:
    ubicacion = st.text_input("📍 Localidad (ciudad/región)", "")
    maps_link = st.text_input("🔗 Enlace de Google Maps (opcional)", "")
    fecha_siembra = st.date_input(
        "🌱 Fecha de siembra",
        datetime.today().date() - timedelta(days=30)
    )

# 3) Parámetros GEE (solo para descarga)
col1, col2 = st.columns(2)
with col1:
    window = st.slider("± Días alrededor de la fecha", 1, 30, 14)
with col2:
    cloud = st.slider("Nubosidad máx. (%)", 0, 100, 20)

# 4) Botones de acción
gen_col, reset_col = st.columns([1, 1])
if reset_col.button("🔄 Resetear todo", type="secondary"):
    st.session_state.clear()
    st.experimental_rerun()
generar = gen_col.button("Generar informe")

# 5) Flujo principal
if generar:
    # Validaciones
    if not cultivo:
        st.error("⚠️ Ingresa el nombre del cultivo.")
        st.stop()
    if not ubicacion and not maps_link:
        st.error("⚠️ Ingresa la localidad o pega un enlace de Google Maps.")
        st.stop()

    # 5.1) Obtener imagen (TIFF local o GEE)
    if "img_bytes" in st.session_state:
        tmp_path = pathlib.Path(tmp_dir) / st.session_state.img_name
        with open(tmp_path, "wb") as f:
            f.write(st.session_state.img_bytes)
        st.info("⏳ Procesando imagen TIFF subida…")
        try:
            promedios, indices, tipo, meta = procesar_imagen(str(tmp_path))
            fuente = f"Imagen subida: {st.session_state.img_name}"
            if not meta.get('crs'):
                st.warning("La imagen no tiene georreferenciación; el mapa NDVI no incluirá coordenadas reales.")
        except Exception as e:
            st.error(f"Error procesando la imagen subida: {e}")
            st.stop()
    else:
        # Extraer coordenadas de Google Maps URL o geocoding de texto
        if maps_link:
            import re
            m = re.search(r'@(-?\d+\.\d+),(-?\d+\.\d+)', maps_link)
            if not m:
                st.error("URL de Google Maps inválida: no encontré '@lat,lon'.")
                st.stop()
            lat, lon = float(m.group(1)), float(m.group(2))
        else:
            st.info("🔍 Geocoding de la localidad…")
            try:
                lat, lon = geocode_location(ubicacion)
            except Exception as e:
                st.error(f"Error geocodificando '{ubicacion}': {e}")
                st.stop()

        # Descargar y apilar GEE
        fecha_str = fecha.strftime("%Y-%m-%d")
        st.info("⌛️ Descargando imagen con GEE…")
        ruta_local, meta_gee = download_gee_image(lat, lon, fecha_str, window_days=window, max_cloud_pct=cloud)
        if not ruta_local:
            st.error("No se encontró imagen Sentinel-2 …")
            st.stop()
        st.write(f"{meta_gee['notice']} (fecha real: {meta_gee['actual_date']})")
        st.success(f"Imagen descargada: {pathlib.Path(ruta_local).name} (Fecha real: {meta_gee['actual_date']}, Nubosidad {meta_gee['cloud_pct']}%)")
        st.info("⌛️ Procesando imagen descargada…")
        try:
            promedios, indices, tipo, meta = procesar_imagen(ruta_local)
            fuente = f"GEE Sentinel-2: {meta_gee['actual_date']}"
        except Exception as e:
            st.error(f"Error procesando la imagen descargada: {e}")
            st.stop()

    # 5.2) Generar informe técnico
    st.info("📝 Generando informe técnico…")
    try:
        informe_tecnico = generar_informe(promedios, fecha.strftime("%Y-%m-%d"), cultivo, ubicacion, tipo, fecha_siembra.strftime("%Y-%m-%d"), fuente)
    except Exception as e:
        st.error(f"Error generando informe: {e}")
        st.stop()
    st.subheader("✅ Informe técnico completo")
    st.markdown(f"```\n{informe_tecnico}\n```")

    # 5.3) Informe agronómico LLM
    st.subheader("🤖 Informe agronómico profesional")
    with st.spinner("Llamando a la LLM..."):
        try:
            informe_llm = generar_informe_llm(informe_tecnico)
            st.markdown(informe_llm)
        except Exception as e:
            st.error(f"Error al generar informe LLM: {e}")

    # 5.4) Mapa NDVI
    if "NDVI" in indices:
        arr_ndvi = indices["NDVI"]
    elif "NDVI_orientativo" in indices:
        arr_ndvi = indices["NDVI_orientativo"]
    else:
        arr_ndvi = None
    if arr_ndvi is None:
        st.warning("No se encontró índice NDVI para visualizar.")
    else:
        fig, ax = plt.subplots()
        extent = None
        if meta.get("bounds"):
            b = meta["bounds"]
            extent = [b.left, b.right, b.bottom, b.top]
        im = ax.imshow(arr_ndvi, extent=extent, origin='upper') if extent else ax.imshow(arr_ndvi, origin='upper')
        ax.set_title("Mapa NDVI")
        ax.axis("off")
        plt.colorbar(im, ax=ax, fraction=0.04)
        st.subheader("🗺️ Mapa NDVI")
        st.pyplot(fig)

    # 5.5) Estadísticas de índices promedio
    st.subheader("📊 Estadísticas de índices promedio")
    stats_md = "| Índice | Promedio |\n|---|---:|\n"
    for k, v in promedios.items():
        stats_md += f"| {k} | {v:.3f} |\n"
    st.markdown(stats_md)
