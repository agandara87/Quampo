FROM python:3.10-slim

# Evitar interacción al instalar tzdata
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=America/Argentina/Buenos_Aires

# Instalar dependencias de rasterio y zona horaria
RUN apt-get update && apt-get install -y \
    g++ \
    python3-dev \
    libgdal-dev \
    tzdata \
    && ln -fs /usr/share/zoneinfo/$TZ /etc/localtime \
    && dpkg-reconfigure --frontend noninteractive tzdata \
    && rm -rf /var/lib/apt/lists/*

# Crear directorio de la app
WORKDIR /app

# Copiar requerimientos y app
COPY requirements.txt .
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

COPY . .

CMD ["streamlit", "run", "quampo_frontend.py", "--server.port=7860", "--server.address=0.0.0.0"]

