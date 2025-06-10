# quampo_frontend.py
import streamlit as st
import matplotlib.pyplot as plt
from datetime import datetime
from quampo_backend import (
    procesar_imagen,
    generar_informe,
    generar_informe_llm,
)

st.set_page_config(page_title="Quampo – Análisis Satelital", layout="centered")
st.title("🛰️ Quampo – Análisis Satelital de Cultivos")
st.markdown("Subí tu imagen y completá los datos para generar el informe.")

# ----------------------------------------------------------------------
# 1) Subida de la imagen ------------------------------------------------
# ----------------------------------------------------------------------
uploaded_file = st.file_uploader(
    "📤 Imagen satelital (.tif, .tiff, .jpg, .png)",
    type=["tif", "tiff", "jpg", "jpeg", "png"],
    help="Máx. 200 MB por archivo (límite de Hugging Face Spaces)",
)

# ↳ guardo **una sola vez** los bytes y el nombre en session_state
if uploaded_file is not None and "img_bytes" not in st.session_state:
    st.session_state["img_bytes"] = uploaded_file.read()
    st.session_state["img_name"] = uploaded_file.name
    st.success(f"Archivo **{uploaded_file.name}** cargado "
               f"({len(st.session_state['img_bytes'])/1024:.1f} KB).")

# -----------------------------------------------------------------------------
# 2) Formulario de metadatos ---------------------------------------------------
# -----------------------------------------------------------------------------
fecha = st.date_input("📅 Fecha de la imagen", datetime.today())
cultivo = st.text_input("🌾 Cultivo (ej. Maíz, Soja)")
ubicacion = st.text_input("📍 Ubicación (ej. Pergamino)")
fecha_siembra = st.date_input("🌱 Fecha de siembra", datetime.today())

# -----------------------------------------------------------------------------
# 3) Botones -------------------------------------------------------------------
# -----------------------------------------------------------------------------
col_btn1, col_btn2 = st.columns([1, 1])
generar = col_btn1.button("Generar informe", use_container_width=True)
resetear = col_btn2.button("🔄 Reset", type="secondary", use_container_width=True)

# ↳ Si el usuario quiere borrar y empezar de cero
if resetear:
    st.session_state.clear()
    st.experimental_rerun()

# -----------------------------------------------------------------------------
# 4) Generar informe -----------------------------------------------------------
# -----------------------------------------------------------------------------
if generar:
    # Verifico que haya imagen en memoria
    if "img_bytes" not in st.session_state:
        st.error("⚠️ Primero tenés que subir una imagen.")
        st.stop()

    # Guardo la imagen en disco temporal
    with open("temp_image.tif", "wb") as f:
        f.write(st.session_state["img_bytes"])

    st.info("⏳ Iniciando análisis…")
    try:
        prom, idx, tipo, meta = procesar_imagen("temp_image.tif")
    except Exception as e:
        st.exception(e)
        st.stop()

    # --- Advertencias visibles ---
    st.info(f"📷 Tipo de imagen: **{tipo}** | Bandas: **{meta['band_count']}**")
    if not meta["has_nir"]:
        st.warning("⚠️ No se detectó banda NIR. El NDVI es sólo estimativo (RGB).")

    # --- Informe técnico ---
    informe = generar_informe(
        prom,
        str(fecha),
        cultivo or "Desconocido",
        ubicacion or "Sin ubicación",
        tipo,
        str(fecha_siembra),
    )

    st.subheader("✅ Informe técnico")
    st.text(informe)

    # --- Informe LLM ---
    st.subheader("🤖 Informe agronómico profesional")
    with st.spinner("Generando informe con GPT…"):
        try:
            informe_llm = generar_informe_llm(informe)
            st.markdown(informe_llm)
        except Exception as e:
            st.error("Error al llamar a OpenAI:")
            st.exception(e)

    # --- Mapa NDVI ---
    if "NDVI" in idx or "NDVI_orientativo" in idx:
        ndvi_key = "NDVI" if "NDVI" in idx else "NDVI_orientativo"
        fig, ax = plt.subplots()
        im = ax.imshow(idx[ndvi_key], cmap="RdYlGn")
        ax.axis("off")
        plt.colorbar(im, ax=ax, fraction=0.036)
        subtitulo = "🖼 Mapa NDVI" if ndvi_key == "NDVI" else "🖼 NDVI estimado (RGB)"
        st.subheader(subtitulo)
        st.pyplot(fig, use_container_width=True)

        if ndvi_key == "NDVI_orientativo":
            st.caption("El NDVI fue estimado sin banda NIR; úsalo sólo como referencia.")

# -----------------------------------------------------------------------------
# (Opcional) mostrar estado para debug ----------------------------------------
# -----------------------------------------------------------------------------
# st.write("Session state:", st.session_state)  # -- descomenta si necesitas ver
