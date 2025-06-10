# Imagen base con GDAL y Python
FROM osgeo/gdal:alpine-small-latest

# Instalar dependencias del sistema
RUN apt-get update && apt-get install -y \
    python3-pip \
    python3-opencv \
    python3-numpy \
    python3-matplotlib \
    python3-dev \
    libgdal-dev \
    libgl1-mesa-glx \
    && rm -rf /var/lib/apt/lists/*

# Definir variable de entorno para GDAL
ENV CPLUS_INCLUDE_PATH=/usr/include/gdal
ENV C_INCLUDE_PATH=/usr/include/gdal
ENV GDAL_VERSION=3.6.2

# Crear directorio de la app
WORKDIR /app

# Copiar archivos de la app
COPY . /app

# Instalar librerías de Python
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

# Ejecutar Streamlit al levantar el contenedor
CMD ["streamlit", "run", "quampo_frontend.py", "--server.port=7860", "--server.address=0.0.0.0"]
