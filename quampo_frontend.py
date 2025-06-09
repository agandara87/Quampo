
# quampo_frontend.py

import streamlit as st
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
        prom, idx, tipo = procesar_imagen("temp_image.tif")
        informe = generar_informe(prom, str(fecha), cultivo, ubicacion, tipo, str(fecha_siembra))
        st.subheader("✅ Informe técnico")
        st.text(informe)
        st.subheader("🤖 Informe agronómico profesional")
        st.markdown(generar_informe_llm(informe))
