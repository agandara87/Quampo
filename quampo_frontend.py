# quampo_frontend.py

import streamlit as st
import matplotlib.pyplot as plt
from datetime import datetime
from quampo_backend import procesar_imagen, generar_informe, generar_informe_llm

st.title("🛰️ Quampo - Análisis Satelital de Cultivos")

uploaded_file = st.file_uploader("📤 Subí tu imagen satelital", type=["tif", "jpg", "png"])

if uploaded_file:
    with open("temp_image.tif", "wb") as f:
        f.write(uploaded_file.getbuffer())

    fecha = st.date_input("📅 Fecha de la imagen", datetime.today())
    cultivo = st.text_input("🌾 Cultivo (ej: Maíz, Soja)", "")
    ubicacion = st.text_input("📍 Ubicación (ej: Pergamino)", "")
    fecha_siembra = st.date_input("📅 Fecha de siembra", datetime.today())

    if st.button("Generar informe"):
        # Procesar imagen
        prom, idx, tipo, metadata = procesar_imagen("temp_image.tif")

        # Mostrar información básica de la imagen
        st.info(f"📷 Tipo de imagen: {tipo} | Bandas detectadas: {metadata['band_count']}")
        if metadata["has_nir"]:
            st.success("✅ Banda NIR detectada correctamente.")
        else:
            st.warning("⚠️ No se detectó banda NIR. El NDVI es solo una estimación basada en RGB.")

        # Generar informe
        informe = generar_informe(prom, str(fecha), cultivo, ubicacion, tipo, str(fecha_siembra))

        # Advertencias clave
        if tipo == "RGB":
            st.warning("⚠️ Imagen RGB detectada. El NDVI fue estimado sin banda NIR. El resultado es solo orientativo.")
        if "NDVI" not in prom:
            st.error("🚫 No se pudo calcular NDVI porque falta la banda NIR.")

        # Mostrar informe técnico
        st.subheader("✅ Informe técnico")
        st.text(informe)

        # Mostrar informe LLM
        st.subheader("🤖 Informe agronómico profesional")
        st.markdown(generar_informe_llm(informe))

        # Mostrar mapa NDVI real o estimado
        if "NDVI" in idx:
            fig, ax = plt.subplots()
            im = ax.imshow(idx["NDVI"], cmap="RdYlGn")
            ax.axis("off")
            plt.colorbar(im, ax=ax)
            st.subheader("🖼 Mapa NDVI")
            st.pyplot(fig)
        elif "NDVI_orientativo" in idx:
            fig, ax = plt.subplots()
            im = ax.imshow(idx["NDVI_orientativo"], cmap="RdYlGn")
            ax.axis("off")
            plt.colorbar(im, ax=ax)
            st.subheader("🖼 Mapa NDVI (Estimado RGB)")
            st.warning("Este NDVI fue estimado usando solo las bandas RGB. Es solo una orientación, no diagnóstico real.")
            st.pyplot(fig)
