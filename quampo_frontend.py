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
    get_satellite_image_dates,
    get_sentinel_url,
    download_sentinel_image 
)

# ────────────────────────────────────────────────────────────────────
# Configuración entorno Streamlit
tmp_dir = tempfile.gettempdir()
os.environ["STREAMLIT_GLOBAL_CONFIG_DIR"] = f"{tmp_dir}/.streamlit"
os.environ["MPLCONFIGDIR"] = f"{tmp_dir}/.matplotlib"
os.environ["BROWSER_GATHERUSAGESTATS"] = "false"

st.set_page_config(page_title="Quampo – Análisis Satelital", layout="centered")
st.title("🛰️ Quampo – Análisis Satelital de Cultivos")
st.markdown(
    "Sube tu imagen TIFF georreferenciada o, si no tienes imagen, deja en blanco y se usará Sentinel Hub."
)

# 1) Subir imagen satelital (solo TIFF para geoespacial)
uploaded = st.file_uploader(
    "📤 Imagen satelital (.tif, .tiff). Si no subes nada, se usará Sentinel Hub según la ubicación y fecha.",
    type=["tif", "tiff"]
)
if uploaded and "img_bytes" not in st.session_state:
    st.session_state["img_bytes"] = uploaded.read()
    st.session_state["img_name"] = uploaded.name
    size_kb = len(st.session_state["img_bytes"]) / 1024
    st.success(f"Archivo **{st.session_state['img_name']}** cargado ({size_kb:.1f} KB)")
    try:
        st.image(st.session_state["img_bytes"], caption="Imagen cargada")
    except Exception:
        pass

# 2) Metadatos
date_col, meta_col = st.columns(2)
with date_col:
    fecha = st.date_input("📅 Fecha de la imagen / análisis", datetime.today().date())
    cultivo = st.text_input("🌾 Cultivo (ej. Maíz, Soja)", "")
with meta_col:
    ubicacion = st.text_input("📍 Localidad (ciudad/región)", "")
    fecha_siembra = st.date_input("🌱 Fecha de siembra", datetime.today().date() - timedelta(days=30))

# 3) Botones
gen_col, reset_col = st.columns([1, 1])
generar = gen_col.button("Generar informe")
resetear = reset_col.button("🔄 Resetear todo", type="secondary")
if resetear:
    st.session_state.clear()
    st.experimental_rerun()

# 4) Flujo principal
if generar:
    # Validaciones
    if not cultivo:
        st.error("⚠️ Ingresa el nombre del cultivo.")
        st.stop()
    if not ubicacion:
        st.error("⚠️ Ingresa la localidad.")
        st.stop()

    # Procesar imagen subida o Sentinel
    if "img_bytes" in st.session_state:
        # Imagen subida
        tmp_path = pathlib.Path(tempfile.gettempdir()) / st.session_state["img_name"]
        with open(tmp_path, "wb") as f:
            f.write(st.session_state["img_bytes"])
        st.info("⏳ Procesando imagen subida...")
        try:
            promedios, indices, tipo, meta = procesar_imagen(str(tmp_path))
            fuente = f"Imagen subida: {st.session_state['img_name']}"
            # Aviso si no georef
            if meta.get('crs') is None:
                st.warning("La imagen subida no tiene georreferenciación; el mapa NDVI se mostrará en píxeles y se usará la ubicación manual para pronóstico/clima.")
        except Exception as e:
            st.error(f"Error procesando la imagen subida: {e}")
            st.stop()

    else:
        # --- Rama: usar Sentinel Hub ---
        st.info("🔍 No se subió imagen. Se intentará usar Sentinel Hub según ubicación y fecha.")
        # 1) Geocodificar
        try:
            lat, lon = geocode_location(ubicacion)
        except Exception as e:
            st.error(f"Error geocodificando '{ubicacion}': {e}")
            st.stop()

        fecha_str = fecha.strftime("%Y-%m-%d")

        # 2) Mostrar URLs encontradas (opcional, solo para referencia)
        imgs = get_satellite_image_dates(lat, lon)
        st.write("URLs de imágenes satelitales (referencia):")
        st.write(f"- Actual: {imgs.get('actual') or 'No encontrada'}")
        st.write(f"- Anterior (~30d): {imgs.get('anterior') or 'No encontrada'}")
        st.write(f"- Hace un año: {imgs.get('anio_atras') or 'No encontrada'}")

        # 3) Descargar la imagen “actual” usando la función del backend
        st.info("⏳ Descargando imagen Sentinel Hub localmente...")
        ruta_local = download_sentinel_image(lat, lon, fecha_str, size_m=5000, resolution=10, window_days=7)
        if not ruta_local:
            st.warning("No se encontró o no se pudo descargar imagen Sentinel en ±7 días de la fecha seleccionada.")
            st.stop()
        st.success(f"Imagen descargada localmente: {ruta_local}")

        # 4) Procesar la imagen descargada
        st.info("⏳ Procesando imagen Sentinel descargada...")
        try:
            promedios, indices, tipo, meta = procesar_imagen(str(ruta_local))
            fuente = f"Sentinel Hub: fecha {fecha_str}"
        except Exception as e:
            st.error(f"Error procesando imagen Sentinel: {e}")
            st.stop()


    # Generar informe técnico
    st.info("📝 Generando informe técnico...")
    try:
        informe_tecnico = generar_informe(
            promedios,
            fecha.strftime("%Y-%m-%d"),
            cultivo,
            ubicacion,
            tipo,
            fecha_siembra.strftime("%Y-%m-%d"),
            fuente
        )
    except Exception as e:
        st.error(f"Error generando informe: {e}")
        st.stop()

    st.subheader("✅ Informe técnico completo")
    st.markdown(f"```\n{informe_tecnico}\n```")

    # Informe LLM
    st.subheader("🤖 Informe agronómico profesional")
    with st.spinner("Llamando a la LLM..."):
        try:
            informe_llm = generar_informe_llm(informe_tecnico)
            st.markdown(informe_llm)
        except Exception as e:
            st.error(f"Error al generar informe LLM: {e}")

    # Visualización NDVI
    key = "NDVI" if "NDVI" in indices else "NDVI_orientativo"
    arr_ndvi = indices.get(key)
    if arr_ndvi is None:
        st.warning("No se encontró índice NDVI en los resultados.")
    else:
        fig, ax = plt.subplots()
        extent = meta.get('extent', None)
        if extent:
            im = ax.imshow(arr_ndvi, extent=extent, origin='upper')
            ax.set_title("Mapa NDVI")
        else:
            im = ax.imshow(arr_ndvi, origin='upper')
            ax.set_title("Mapa NDVI (sin georreferenciación)")
        ax.axis('off')
        plt.colorbar(im, ax=ax, fraction=0.04)
        st.subheader("🗺️ Mapa NDVI")
        st.pyplot(fig)

    # Estadísticas de promedios
    st.subheader("📊 Estadísticas de índices promedio")
    stats_md = "| Índice | Valor promedio |\n|---|---:|\n"
    for k, v in promedios.items():
        stats_md += f"| {k} | {v:.3f} |\n"
    st.markdown(stats_md)

    # Fin del flujo
