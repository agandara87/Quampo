# quampo_frontend.py
# ──────────────────────────────────────────────────────────────────────
import os, tempfile
tmp_dir = tempfile.gettempdir()
os.environ["STREAMLIT_GLOBAL_CONFIG_DIR"] = f"{tmp_dir}/.streamlit"
os.environ["MPLCONFIGDIR"]              = f"{tmp_dir}/.matplotlib"
os.environ["BROWSER_GATHERUSAGESTATS"]  = "false"
# ──────────────────────────────────────────────────────────────────────
import streamlit as st
import matplotlib.pyplot as plt
from datetime import datetime
from quampo_backend import procesar_imagen, generar_informe, generar_informe_llm

st.set_page_config(page_title="Quampo – Análisis Satelital", layout="centered")
st.title("🛰️ Quampo – Análisis Satelital de Cultivos")
st.markdown("Subí tu imagen y completá los datos para generar el informe.")

# 1) Uploader
uploaded_file = st.file_uploader(
    "📤 Imagen satelital (.tif, .tiff, .jpg, .png)",
    type=["tif", "tiff", "jpg", "jpeg", "png"],
)

if uploaded_file and "img_bytes" not in st.session_state:
    st.session_state["img_bytes"] = uploaded_file.read()
    st.session_state["img_name"]  = uploaded_file.name
    st.success(f"Archivo **{uploaded_file.name}** cargado "
               f"({len(st.session_state['img_bytes'])/1024:.1f} KB)")

# 2) Metadatos
col1, col2 = st.columns(2)
with col1:
    fecha         = st.date_input("📅 Fecha de la imagen", datetime.today())
    cultivo       = st.text_input("🌾 Cultivo (ej. Maíz, Soja)")
with col2:
    ubicacion     = st.text_input("📍 Ubicación", placeholder="Pergamino")
    fecha_siembra = st.date_input("🌱 Fecha de siembra", datetime.today())

# 3) Botones
col_g, col_r = st.columns([1,1])
generar  = col_g.button("Generar informe", use_container_width=True)
resetear = col_r.button("🔄 Reset", type="secondary", use_container_width=True)

if resetear:
    st.session_state.clear()
    st.experimental_rerun()

# 4) Generar informe
if generar:
    if "img_bytes" not in st.session_state:
        st.error("⚠️ Primero subí una imagen.")
        st.stop()

    with open("temp_image.tif", "wb") as f:
        f.write(st.session_state["img_bytes"])

    st.info("⏳ Procesando imagen…")
    prom, idx, tipo, meta = procesar_imagen("temp_image.tif")

    st.info(f"📷 Tipo: **{tipo}** | Bandas: **{meta['band_count']}**")
    if not meta["has_nir"]:
        st.warning("⚠️ Sin banda NIR → NDVI es sólo orientativo (RGB).")

    informe = generar_informe(
        prom, str(fecha), cultivo or "Desconocido",
        ubicacion or "Sin ubicación", tipo, str(fecha_siembra)
    )
    st.subheader("✅ Informe técnico"); st.text(informe)

    st.subheader("🤖 Informe agronómico profesional")
    with st.spinner("Llamando a GPT…"):
        st.markdown(generar_informe_llm(informe))

    if "NDVI" in idx or "NDVI_orientativo" in idx:
        ndvi_key = "NDVI" if "NDVI" in idx else "NDVI_orientativo"
        fig, ax = plt.subplots()
        im = ax.imshow(idx[ndvi_key], cmap="RdYlGn"); ax.axis("off")
        plt.colorbar(im, ax=ax, fraction=0.04)
        st.subheader("🖼 Mapa NDVI" if ndvi_key=="NDVI"
                     else "🖼 NDVI estimado (RGB)")
        st.pyplot(fig, use_container_width=True)
