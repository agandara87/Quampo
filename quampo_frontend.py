# quampo_frontend.py
import streamlit as st
import matplotlib.pyplot as plt
from datetime import datetime
from quampo_backend import procesar_imagen, generar_informe, generar_informe_llm

st.set_page_config(page_title="Quampo", layout="centered")
st.title("🛰️ Quampo - Análisis Satelital de Cultivos")
st.write("Subí tu imagen y completa los datos para generar el informe.")

# 1) Subida de archivo
uploaded_file = st.file_uploader("📤 Imagen satelital (.tif, .jpg, .png)",
                                 type=["tif", "tiff", "jpg", "jpeg", "png"])
if not uploaded_file:
    st.info("📌 Cargá una imagen para habilitar el análisis.")
    st.stop()

# 2) Guardar el archivo con su extensión
ext = uploaded_file.name.split(".")[-1]
temp_path = f"temp_image.{ext}"
with open(temp_path, "wb") as f:
    f.write(uploaded_file.getbuffer())

# 3) Inputs adicionales
fecha = st.date_input("📅 Fecha de la imagen", datetime.today())
cultivo = st.text_input("🌾 Cultivo (ej: Maíz, Soja)")
ubicacion = st.text_input("📍 Ubicación (ej: Pergamino)")
fecha_siembra = st.date_input("📅 Fecha de siembra", datetime.today())

# 4) Botón para lanzar el análisis
if st.button("Generar informe"):
    # Procesar imagen
    prom, idx, tipo, metadata = procesar_imagen(temp_path)

    # Generar el informe técnico
    informe = generar_informe(
        prom,
        fecha.strftime("%Y-%m-%d"),
        cultivo,
        ubicacion,
        tipo,
        fecha_siembra.strftime("%Y-%m-%d")
    )

    # 5) Advertencias
    if not metadata.get("has_nir", False):
        st.warning("⚠️ No se detectó banda NIR — el NDVI real no puede calcularse.")
        if "NDVI_orientativo" in idx:
            st.info("ℹ️ Se mostró un NDVI estimado usando solo RGB.")

    # 6) Mostrar informe técnico
    st.subheader("✅ Informe técnico")
    st.text(informe)

    # 7) Mostrar informe LLM
    st.subheader("🤖 Informe agronómico profesional")
    st.markdown(generar_informe_llm(informe))

    # 8) Mapa NDVI
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
        fig.colorbar(im, ax=ax, label="NDVI estimado")
        st.pyplot(fig)


