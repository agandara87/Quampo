# Imagen base estable con GDAL y Python
FROM osgeo/gdal:ubuntu-full-3.6.2

# Seteamos la zona horaria para evitar prompts interactivos como el de tu screenshot
ENV TZ=America/Argentina/Buenos_Aires
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezon

# Instalar dependencias del sistema
RUN apt-get update && apt-get install -y \
    python3-pip \
    python3-opencv \
    python3-numpy \
    python3-matplotlib \
    libgl1-mesa-glx \
    && rm -rf /var/lib/apt/lists/*

# Establecer variables necesarias para Rasterio
ENV CPLUS_INCLUDE_PATH=/usr/include/gdal
ENV C_INCLUDE_PATH=/usr/include/gdal
ENV GDAL_VERSION=3.6.2

# Crear directorio y copiar código
WORKDIR /app
COPY . /app

# Instalar dependencias de Python
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

# Comando para ejecutar Streamlit
CMD ["streamlit", "run", "quampo_frontend.py", "--server.port=7860", "--server.address=0.0.0.0"]
