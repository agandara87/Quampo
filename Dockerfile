# --------------------------------------
# Quampo – Dockerfile (HF Space)
# --------------------------------------

# 1) Imagen base: Debian bookworm slim con Python 3.11
FROM python:3.11-slim-bookworm

# 2) Dependencias del sistema necesarias para GDAL/Rasterio,
#    proyecciones y compilación de wheels nativos.
RUN apt-get update && apt-get install -y --no-install-recommends \
        gdal-bin libgdal-dev python3-gdal \
        proj-bin libproj-dev \
        build-essential gcc g++ make \
        libgl1-mesa-glx \
    && rm -rf /var/lib/apt/lists/*

# 3) Variables de entorno de GDAL y Proj (evita “data not found”)
ENV GDAL_DATA=/usr/share/gdal
ENV PROJ_LIB=/usr/share/proj

# 4) Configuración de carpetas temporales (HF no permite HOME=/root)
ENV HOME=/tmp
ENV STREAMLIT_GLOBAL_CONFIG_DIR=/tmp/.streamlit
ENV MPLCONFIGDIR=/tmp/.matplotlib
ENV BROWSER_GATHERUSAGESTATS=false
ENV STREAMLIT_SERVER_ENABLE_XSRF_PROTECTION=false

# 5) Puerto por defecto para correr localmente (HF lo sobrescribe)
ENV PORT=7860

# 6) Instalo primero requirements para cachear capa de pip
COPY requirements.txt /tmp/
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r /tmp/requirements.txt

# 7) Copio el resto de la app
COPY . /app
WORKDIR /app

# 8) Comando de arranque: forma *shell* para que $PORT se expanda
CMD streamlit run quampo_frontend.py \
    --server.port $PORT \
    --server.address 0.0.0.0
