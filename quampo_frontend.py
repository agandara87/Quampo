import streamlit as st
import matplotlib.pyplot as plt
from datetime import datetime
from quampo_backend import procesar_imagen, generar_informe, generar_informe_llm

st.set_page_config(page_title="Quampo", layout="centered")
st.title("🛰️ Quampo - Análisis Satelital de Cultivos")
st.write("Subí tu imagen y completa los datos para generar el informe técnico y agronómico.")

# 1. Subida de archivo
uploaded_file = st.file_uploader("📤 Imagen satelital (.tif, .jpg, .png)", 
                                 type=["tif", "tiff", "jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Guardar con extensión real
    ext = uploaded_file.name.split(".")[-1]
    temp_path = f"temp_image.{ext}"
    with open(temp_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    # 2. Inputs del productor
    fecha = st.date_input("📅 Fecha de la imagen", datetime.today())
    cultivo = st.text_input("🌾 Cultivo (ej: Maíz, Soja)", "")
    ubicacion = st.text_input("📍 Ubicación (ej: Pergamino)", "")
    fecha_siembra = st.date_input("📅 Fecha de siembra", datetime.today())

    # 3. Botón para ejecutar el análisis
    if st.button("Generar informe"):
        # Ejecuta el procesamiento
        prom, idx, tipo, metadata = procesar_imagen(temp_path)
        informe = generar_informe(
            prom,
            fecha.strftime("%Y-%m-%d"),
            cultivo,
            ubicacion,
            tipo,
            fecha_siembra.strftime("%Y-%m-%d")
        )

        # 4. Advertencias
        if tipo == "RGB" and not metadata.get("has_nir", False):
            st.warning("⚠️ Imagen RGB detectada. El NDVI es solo estimativo (sin banda NIR).")
        if not metadata.get("has_nir", False):
            st.error("🚫 No se pudo calcular NDVI real; falta banda NIR.")

        # 5. Mostrar informe técnico
        st.subheader("✅ Informe técnico")
        st.text(informe)

        # 6. Mostrar informe LLM
        st.subheader("🤖 Informe agronómico profesional")
        llm_report = generar_informe_llm(informe)
        st.markdown(llm_report)

        # 7. Mapa NDVI
        if "NDVI" in idx:
            fig, ax = plt.subplots()
            im = ax.imshow(idx["NDVI"], cmap="RdYlGn")
            ax.axis("off")
            fig.colorbar(im, ax=ax, label="NDVI")
            st.subheader("🖼 Mapa NDVI")
            st.pyplot(fig)
        elif "NDVI_orientativo" in idx:
            fig, ax = plt.subplots()
            im = ax.imshow(idx["NDVI_orientativo"], cmap="RdYlGn")
            ax.axis("off")
            fig.colorbar(im, ax=ax, label="NDVI (estimado)")
            st.subheader("🖼 Mapa NDVI (Estimado RGB)")
            st.warning("Este NDVI fue estimado usando solo bandas RGB.")
            st.pyplot(fig)
else:
    st.info("📌 Cargá una imagen para habilitar el análisis.")

