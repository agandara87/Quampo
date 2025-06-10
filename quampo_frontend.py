# quampo_frontend.py
import streamlit as st
import matplotlib.pyplot as plt
from datetime import datetime
from quampo_backend import procesar_imagen, generar_informe, generar_informe_llm

st.set_page_config(page_title="Quampo", layout="centered")
st.title("🛰️ Quampo - Análisis Satelital de Cultivos")
st.write("Subí tu imagen y completá los datos para generar el informe.")

# Formulario para agrupar inputs y botón de envío
with st.form("analysis_form", clear_on_submit=False):
    uploaded_file = st.file_uploader(
        "📤 Imagen satelital (.tif, .jpg, .png)",
        type=["tif", "tiff", "jpg", "jpeg", "png"]
    )
    fecha = st.date_input("📅 Fecha de la imagen", datetime.today())
    cultivo = st.text_input("🌾 Cultivo (ej: Maíz, Soja)")
    ubicacion = st.text_input("📍 Ubicación (ej: Pergamino)")
    fecha_siembra = st.date_input("📅 Fecha de siembra", datetime.today())

    submit = st.form_submit_button("Generar informe")

if submit:
    # Debug inicial
    st.info("▶️ Iniciando análisis...")
    if not uploaded_file:
        st.error("❌ Primero tenés que subir una imagen.")
    elif not cultivo or not ubicacion:
        st.error("❌ Completar cultivo y ubicación.")
    else:
        # Guardar archivo con extensión real
        ext = uploaded_file.name.split(".")[-1]
        temp_path = f"temp_image.{ext}"
        with open(temp_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        st.success("✔️ Imagen guardada.")

        # Procesar imagen
        try:
            prom, idx, tipo, metadata = procesar_imagen(temp_path)
            st.info(f"🛠 Procesando como {tipo}, bandas: {metadata['band_count']}")
        except Exception as e:
            st.error(f"⚠️ Error al procesar la imagen: {e}")
            st.stop()

        # Generar informe técnico
        informe = generar_informe(
            prom,
            fecha.strftime("%Y-%m-%d"),
            cultivo,
            ubicacion,
            tipo,
            fecha_siembra.strftime("%Y-%m-%d")
        )
        st.subheader("✅ Informe técnico")
        st.text(informe)

        # Generar y mostrar informe LLM
        try:
            llm_report = generar_informe_llm(informe)
            st.subheader("🤖 Informe agronómico profesional")
            st.markdown(llm_report)
        except Exception as e:
            st.warning(f"🤖 No se pudo generar el informe LLM: {e}")

        # Mostrar mapa NDVI
        if "NDVI" in idx:
            st.subheader("🖼 Mapa NDVI real")
            fig, ax = plt.subplots()
            im = ax.imshow(idx["NDVI"], cmap="RdYlGn")
            ax.axis("off")
            fig.colorbar(im, ax=ax, label="NDVI")
            st.pyplot(fig)
        elif "NDVI_orientativo" in idx:
            st.subheader("🖼 Mapa NDVI estimado (RGB)")
            fig, ax = plt.subplots()
            im = ax.imshow(idx["NDVI_orientativo"], cmap="RdYlGn")
            ax.axis("off")
            fig.colorbar(im, ax=ax, label="NDVI (estimado)")
            st.warning("⚠️ Este NDVI fue estimado usando sólo RGB.")
            st.pyplot(fig)

