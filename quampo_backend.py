import os
import json
import tempfile
import requests
import zipfile
import io
import numpy as np
import rasterio
from rasterio.errors import RasterioIOError
from datetime import datetime, timedelta
from openai import OpenAI
import ee

# === Configuration Constants ===
LLM_MODEL = "gpt-4o"

# Parámetros configurables (via env vars)
WINDOW_DAYS          = int(os.getenv("WINDOW_DAYS",      14))
CLOUD_THRESHOLD      = int(os.getenv("CLOUD_THRESHOLD",20))
SCALE                = int(os.getenv("SCALE",          30))
BUFFER_M             = int(os.getenv("BUFFER_M",      2500))
FORECAST_DAYS        = int(os.getenv("FORECAST_DAYS",   7))

LLM_TEMPERATURE      = float(os.getenv("LLM_TEMPERATURE",  0.7))
LLM_MAX_TOKENS       = int(os.getenv("LLM_MAX_TOKENS", 1024))
EXPLAIN_TEMPERATURE  = float(os.getenv("EXPLAIN_TEMPERATURE",0.3))
EXPLAIN_MAX_TOKENS   = int(os.getenv("EXPLAIN_MAX_TOKENS",   40))
PHENO_TEMPERATURE    = float(os.getenv("PHENO_TEMPERATURE",  0.3))
PHENO_MAX_TOKENS     = int(os.getenv("PHENO_MAX_TOKENS",    50))
PLAN_TEMPERATURE     = float(os.getenv("PLAN_TEMPERATURE",   0.5))
PLAN_MAX_TOKENS      = int(os.getenv("PLAN_MAX_TOKENS",    200))

# === Google Earth Engine Initialization ===
SA_PATH = 'service_account.json'
if not os.path.exists(SA_PATH):
    raise FileNotFoundError(f"Service account file not found: {SA_PATH}")
with open(SA_PATH) as f:
    sa_info = json.load(f)
credentials = ee.ServiceAccountCredentials(sa_info['client_email'], SA_PATH)
ee.Initialize(credentials)

# === API Keys and OpenAI Client ===
OPENAI_API_KEY      = os.getenv("OPENAI_API_KEY")
OPENWEATHER_API_KEY = os.getenv("OPENWEATHER_API_KEY")
GOOGLE_MAPS_API_KEY = os.getenv("GOOGLE_MAPS_API_KEY")
for key, val in [("OPENAI_API_KEY", OPENAI_API_KEY),
                 ("OPENWEATHER_API_KEY", OPENWEATHER_API_KEY),
                 ("GOOGLE_MAPS_API_KEY", GOOGLE_MAPS_API_KEY)]:
    if not val:
        raise EnvironmentError(f"Falta {key} en env vars")
client = OpenAI(api_key=OPENAI_API_KEY)

# === System prompt ===
system_prompt_llm = '''
Sos un asesor técnico agrónomo digital que trabaja para Quampo, una plataforma que analiza imágenes satelitales de cultivos.
Tu objetivo es interpretar un informe técnico que incluye:
- Índices de vegetación (NDVI, EVI, NDWI, SAVI, GNDVI, NDMI, NDRE, MSAVI).
- Datos climáticos actuales, pronóstico diario extendido y datos espaciales.
- Información del cultivo (tipo, ubicación) y fechas de imágenes satelitales.
- Fecha de siembra y cantidad de días desde siembra.
- Etapa fenológica estimada.

Incluí además:
- Geocodificación de la localidad (Google Maps).
- URLs de las imágenes satelitales más recientes, la anterior y un año atrás.
- Pronóstico extendido de 7 días.

Siempre hablá en español simple, con profesionalismo.
'''

# === Geocoding ===
def geocode_location(location_name):
    url = (
        f"https://maps.googleapis.com/maps/api/geocode/json?"
        f"address={requests.utils.quote(location_name)}&key={GOOGLE_MAPS_API_KEY}"
    )
    resp = requests.get(url, timeout=10)
    data = resp.json()
    if resp.status_code != 200 or data.get('status') != 'OK' or not data.get('results'):
        raise ValueError("No se pudo geocodificar la localidad")
    loc = data['results'][0]['geometry']['location']
    return loc['lat'], loc['lng']

# === Weather Retrieval ===
def obtener_clima_current(lat, lon):
    url = (
        f"https://api.openweathermap.org/data/2.5/weather?lat={lat}&lon={lon}"
        f"&units=metric&lang=es&appid={OPENWEATHER_API_KEY}"
    )
    resp = requests.get(url, timeout=10)
    data = resp.json()
    if resp.status_code != 200 or 'main' not in data:
        return None
    return {
        'descripcion': data['weather'][0]['description'].capitalize(),
        'temperatura': f"{data['main']['temp']}°C",
        'humedad':    f"{data['main']['humidity']}%",
        'lluvia':     f"{data.get('rain', {}).get('1h', 0)} mm"
    }

def get_extended_forecast(lat, lon):
    url = (
        f"https://api.openweathermap.org/data/2.5/onecall?lat={lat}&lon={lon}"
        f"&exclude=current,minutely,hourly,alerts&units=metric&lang=es&appid={OPENWEATHER_API_KEY}"
    )
    resp = requests.get(url, timeout=10)
    data = resp.json()
    if resp.status_code != 200 or 'daily' not in data:
        return []
    forecast = []
    for day in data['daily'][:FORECAST_DAYS]:
        local_dt = datetime.utcfromtimestamp(day['dt']) - timedelta(hours=3)
        forecast.append({
            'fecha': local_dt.strftime('%Y-%m-%d'),
            'desc':  day['weather'][0]['description'].capitalize(),
            'temp':  f"{day['temp']['day']}°C",
            'rain':  f"{day.get('rain', 0)} mm"
        })
    return forecast

# === Earth Engine Image Functions ===
def get_gee_image_url(lat, lon, date_str,
                      window_days=WINDOW_DAYS,
                      cloud_threshold=CLOUD_THRESHOLD,
                      scale=SCALE,
                      buffer_m=BUFFER_M):
    try:
        target = datetime.fromisoformat(date_str).date()
    except ValueError:
        return None, None, None
    start = (target - timedelta(days=window_days)).isoformat()
    end   = (target + timedelta(days=window_days)).isoformat()
    geom = ee.Geometry.Point([lon, lat])
    coll = (ee.ImageCollection('COPERNICUS/S2_SR')
            .filterBounds(geom)
            .filterMetadata('CLOUDY_PIXEL_PERCENTAGE','less_than',cloud_threshold))
    window = coll.filterDate(start, end).sort('CLOUDY_PIXEL_PERCENTAGE')
    if window.size().getInfo() > 0:
        image, notice = window.first(), f"Imagen en ventana {start}–{end}"
    else:
        before = coll.filterDate('2015-01-01', date_str).sort('system:time_start', False)
        if before.size().getInfo() == 0:
            return None, None, None
        image, notice = before.first(), f"Imagen previa antes de {date_str}"
    ts = image.get('system:time_start').getInfo()
    actual_date = datetime.utcfromtimestamp(ts/1000).strftime('%Y-%m-%d')
    region = geom.buffer(buffer_m).bounds().getInfo()['coordinates']
    url = image.getDownloadURL({
        'bands': ['B4','B3','B2','B8','B8A','B11'],
        'scale': scale,
        'crs': 'EPSG:4326',
        'region': region,
        'fileFormat': 'GEO_TIFF'
    })
    return url, actual_date, notice

def get_gee_image_dates(lat, lon):
    today    = datetime.utcnow().date().isoformat()
    prev     = (datetime.utcnow().date() - timedelta(days=30)).isoformat()
    year_ago = (datetime.utcnow().date() - timedelta(days=365)).isoformat()
    return {
        'actual': get_gee_image_url(lat, lon, today),
        'anterior': get_gee_image_url(lat, lon, prev),
        'anio_atras': get_gee_image_url(lat, lon, year_ago)
    }

def download_and_stack_gee_tif(url: str, output_path: str) -> str:
    resp = requests.get(url, stream=True)
    resp.raise_for_status()
    z = zipfile.ZipFile(io.BytesIO(resp.content))
    tifs = sorted([n for n in z.namelist() if n.lower().endswith('.tif')])
    with z.open(tifs[0]) as f0:
        with rasterio.open(io.BytesIO(f0.read())) as src0:
            meta = src0.meta.copy()
    meta.update(count=len(tifs))
    with rasterio.open(output_path, 'w', **meta) as dst:
        for i, name in enumerate(tifs, 1):
            with z.open(name) as f:
                with rasterio.open(io.BytesIO(f.read())) as src:
                    dst.write(src.read(1), i)
    return output_path

def download_gee_image(lat, lon, date_str,
                       window_days=WINDOW_DAYS,
                       max_cloud_pct=CLOUD_THRESHOLD,
                       scale=SCALE,
                       buffer_m=BUFFER_M):
    url, actual_date, notice = get_gee_image_url(lat, lon, date_str,
                                                  window_days, max_cloud_pct,
                                                  scale, buffer_m)
    if not url:
        return None, {}
    out_tif = tempfile.mktemp(suffix=".tif")
    download_and_stack_gee_tif(url, out_tif)
    return out_tif, {
        'requested_date': date_str,
        'actual_date': actual_date,
        'notice': notice,
        'cloud_pct': max_cloud_pct,
        'scale': scale,
        'buffer_m': buffer_m
    }

# === Glossary ===
glossary = {
    'NDVI':  {'interpretacion': [
        {'rango': [0.0, 0.3], 'significado': 'Baja actividad fotosintética / Estrés'},
        {'rango': [0.3, 0.6], 'significado': 'Moderada actividad fotosintética'},
        {'rango': [0.6, 1.0], 'significado': 'Buena salud del cultivo'}
    ]},
    'EVI':   {'interpretacion': [
        {'rango': [-1.0, 0.0], 'significado': 'Poca vegetación o suelo desnudo'},
        {'rango': [0.0, 0.2], 'significado': 'Estrés o vegetación escasa'},
        {'rango': [0.2, 0.5], 'significado': 'Vegetación moderada'},
        {'rango': [0.5, 1.0], 'significado': 'Vegetación densa y sana'}
    ]},
    'NDWI':  {'interpretacion': [
        {'rango': [-1.0, 0.0], 'significado': 'Humedad baja'},
        {'rango': [0.0, 0.3], 'significado': 'Humedad moderada'},
        {'rango': [0.3, 1.0], 'significado': 'Alta humedad o agua libre'}
    ]},
    'SAVI':  {'interpretacion': [
        {'rango': [-1.0, 0.0], 'significado': 'Estrés o suelo descubierto'},
        {'rango': [0.0, 0.2], 'significado': 'Vegetación escasa'},
        {'rango': [0.2, 0.5], 'significado': 'Vegetación moderada'},
        {'rango': [0.5, 1.0], 'significado': 'Vegetación densa'}
    ]},
    'GNDVI': {'interpretacion': [
        {'rango': [-1.0, 0.0], 'significado': 'Vegetación pobre o suelo desnudo'},
        {'rango': [0.0, 0.3], 'significado': 'Vegetación moderada'},
        {'rango': [0.3, 1.0], 'significado': 'Alta vegetación sana'}
    ]},
    'NDMI':  {'interpretacion': [
        {'rango': [-1.0,  0.0], 'significado': 'Sequía o muy baja humedad en biomasa'},
        {'rango': [ 0.0,  0.2], 'significado': 'Humedad moderada, posible estrés leve'},
        {'rango': [ 0.2,  0.5], 'significado': 'Buena humedad en biomasa'},
        {'rango': [ 0.5,  1.0], 'significado': 'Alta humedad en biomasa / Follaje muy hidratado'}
    ]},
    'NDRE': {'interpretacion': [
      {'rango': [-1.0, 0.0], 'significado': 'Clorofila muy baja / Estrés temprano'},
      {'rango': [0.0, 0.2], 'significado': 'Clorofila baja'},
      {'rango': [0.2, 0.5], 'significado': 'Clorofila moderada'},
      {'rango': [0.5, 1.0], 'significado': 'Alta concentración de clorofila'}
    ]},
    'MSAVI': {'interpretacion': [
      {'rango': [-1.0, 0.0], 'significado': 'Suelo desnudo / Vegetación muy escasa'},
      {'rango': [0.0, 0.2], 'significado': 'Vegetación incipiente'},
      {'rango': [0.2, 0.5], 'significado': 'Vegetación moderada con algo de suelo'},
      {'rango': [0.5, 1.0], 'significado': 'Vegetación densa, poco efecto suelo'}
    ]}
}

# === Interpret Index Deterministically ===
def interpretar_indice(valor, nombre):
    if nombre == 'EVI':
        evi = max(valor, -1.0)
        if evi > 0.8: return "Cobertura muy densa de vegetación"
        elif evi > 0.5: return "Vegetación densa"
        elif evi > 0.2: return "Vegetación moderada"
        elif evi >= 0:  return "Vegetación escasa o suelo desnudo"
        else: return "Superficie no vegetal (agua, sombra…)"
    for r in glossary[nombre]['interpretacion']:
        if r['rango'][0] <= valor <= r['rango'][1]:
            return r['significado']
    return 'Interpretación no disponible'

# === Explain Index via LLM ===
def explicar_indice_llm(valor, nombre, categoria):
    prompt = (
        f"Tengo un índice {nombre} con valor {valor:.3f}, que cae en '{categoria}'. "
        "Explícalo en una frase breve, enfocándote en el estado del cultivo y posibles acciones."
    )
    resp = client.chat.completions.create(
        model=LLM_MODEL,
        temperature=EXPLAIN_TEMPERATURE,
        max_tokens=EXPLAIN_MAX_TOKENS,
        messages=[
            {"role":"system","content":system_prompt_llm},
            {"role":"user","content":prompt}
        ]
    )
    return resp.choices[0].message.content.strip()

# === Phenological Stage via LLM ===
def etapa_fenologica_llm(cultivo, dias, ubicacion, clima_actual):
    clima_txt = ("Clima actual: "
                f"{clima_actual.get('descripcion','N/A')}, "
                f"{clima_actual.get('temperatura','N/A')}, humedad {clima_actual.get('humedad','N/A')}")
    user_prompt = (
        f"Para el cultivo '{cultivo}' en '{ubicacion}', con {dias if dias is not None else 'N/A'} días "
        f"desde siembra y estas condiciones: {clima_txt}, ¿en qué etapa fenológica está y por qué? "
        "Devuélvelo en una frase breve."
    )
    resp = client.chat.completions.create(
        model=LLM_MODEL,
        temperature=PHENO_TEMPERATURE,
        max_tokens=PHENO_MAX_TOKENS,
        messages=[
            {"role":"system","content":system_prompt_llm},
            {"role":"user","content":user_prompt}
        ]
    )
    return resp.choices[0].message.content.strip()

# === Generate Action Plan via LLM ===
def generar_plan_accion_llm(texto_informe):
    prompt = (
        f"A partir de este informe técnico, genera un plan de acción con sugerencias claras "
        f"y pasos a seguir para mejorar el cultivo:\n{texto_informe}\nDevuélvelo en un listado de puntos."
    )
    resp = client.chat.completions.create(
        model=LLM_MODEL,
        temperature=PLAN_TEMPERATURE,
        max_tokens=PLAN_MAX_TOKENS,
        messages=[
            {"role":"system","content":system_prompt_llm},
            {"role":"user","content":prompt}
        ]
    )
    return resp.choices[0].message.content.strip()

# === Satellite Image Processing ===
def procesar_imagen(path):
    """
    Abre un GeoTIFF multibanda en el orden fijo [B4, B3, B2, B8, B8A, B11]
    y calcula índices: NDVI, EVI, NDMI, NDWI, SAVI, GNDVI, NDRE, MSAVI.
    """
    import numpy as np
    import rasterio

    # 1) Abrir y leer todas las bandas
    with rasterio.open(path) as src:
        arr = src.read(masked=True)  # arr.shape esperado: (6, filas, cols)
        meta = {'count': src.count, 'crs': src.crs, 'bounds': src.bounds}

    # 2) Verificar que efectivamente hay 6 bandas
    if arr.shape[0] < 6:
        raise ValueError(f"Esperaba 6 bandas (B4,B3,B2,B8,B8A,B11), encontré {arr.shape[0]}")

    # 3) Extraer por posición fija (índice 0→B4, 1→B3, 2→B2, 3→B8, 4→B8A, 5→B11)
    R = arr[0]  # Rojo (B4)
    G = arr[1]  # Verde (B3)
    B = arr[2]  # Azul (B2)
    N = arr[3]  # NIR (B8)
    RE = arr[4] # Red-Edge (B8A)
    S = arr[5]  # SWIR1 (B11)

    # 4) Normalizar si vienen valores en un rango >1 (Digital Numbers)
    maxv = max(np.nanmax(ch) for ch in (R, G, B, N, RE, S))
    if maxv > 1:
        R, G, B, N, RE, S = [ch / maxv for ch in (R, G, B, N, RE, S)]

    # 5) Calcular índices
    # Nota: usamos 1e-5 para evitar división por cero
    funcs = {
        'NDVI':  (N - R) / (N + R + 1e-5),
        'EVI':   2.5 * (N - R) / (N + 6*R - 7.5*B + 1e-5),
        'NDMI':  (N - S) / (N + S + 1e-5),       # humedad biomasa
        'NDWI':  (G - N) / (G + N + 1e-5),       # verde–NIR (puedes omitirlo si solo quieres NDMI)
        'SAVI':  1.5 * (N - R) / (N + R + 0.5 + 1e-5),
        'GNDVI': (G - R) / (G + R + 1e-5),
        'NDRE':  (RE - R) / (RE + R + 1e-5),     # clorofila vía red-edge
        'MSAVI': (2*N + 1 - np.sqrt((2*N + 1)**2 - 8*(N - R))) / 2
    }

    # 6) Calcular promedio de cada índice
    promedios = {k: float(np.nanmean(v)) for k, v in funcs.items()}

    tipo = 'Multiespectral'
    return promedios, funcs, tipo, meta

# === Generate Report Text ===
def generar_informe(promedios, fecha, cultivo, ubicacion, tipo, fecha_siembra, fuente=None):
    lat, lon = geocode_location(ubicacion)
    imgs      = get_gee_image_dates(lat, lon)
    clima_act = obtener_clima_current(lat, lon)
    forecast7 = get_extended_forecast(lat, lon)
    try:
        dias = (datetime.strptime(fecha, '%Y-%m-%d') - datetime.strptime(fecha_siembra, '%Y-%m-%d')).days
    except:
        dias = None
    etapa = etapa_fenologica_llm(cultivo, dias, ubicacion, clima_act)

    lines = []
    if fuente:
        lines.append(f"Fuente de imagen: {fuente}")
    lines.append(f"🛰 Informe Quampo | Fecha análisis: {fecha}")
    lines.append(f"Cultivo: {cultivo} | Ubicación: {ubicacion} (lat:{lat:.4f}, lon:{lon:.4f})")
    lines.append(f"Tipo: {tipo} | Días desde siembra: {dias if dias is not None else 'N/A'} | Etapa: {etapa}")
    lines.append("-- Índices promedio --")
    for k, v in promedios.items():
        if np.isnan(v):
            lines.append(f"{k}: No hay datos suficientes")
            continue
        cat = interpretar_indice(v, k)
        lines.append(f"{k}: {v:.3f} → {cat}")
    # Nota: en este paso solo incluimos la parte técnica sin explicaciones LLM.
    # El pulido/integración la haremos en el siguiente prompt.

    # Sección clima actual
    if clima_act:
        lines.append("-- Clima actual --")
        lines.append(f"{clima_act['descripcion']}, {clima_act['temperatura']}, humedad {clima_act['humedad']}, lluvia 1h: {clima_act.get('lluvia','0 mm')}")
    else:
        lines.append("-- Clima actual: No disponible --")

    # Sección pronóstico
    if forecast7:
        lines.append("-- Pronóstico próximo (7 días) --")
        for d in forecast7:
            lines.append(f"{d['fecha']}: {d['desc']}, {d['temp']}, lluvia {d['rain']}")
    else:
        lines.append("-- Pronóstico próximo (7 días): No disponible --")

    # Sección imágenes
    lines.append("-- Imágenes satelitales --")
    for label, key in [("Actual", 'actual'), ("Anterior (~30d)", 'anterior'), ("Hace 1 año", 'anio_atras')]:
        url, fecha_img, notice = imgs[key]
        if url:
            lines.append(f"{label}: fecha {fecha_img} ({notice})")
            lines.append(f"  URL: {url}")
        else:
            lines.append(f"{label}: No disponible")

    # Combinar en un solo bloque de texto
    return "\n".join(lines)

# === Polish Report with LLM ===
def generar_informe_llm(texto_informe):
    prompt = (
        texto_informe
        + "\n\nPor favor, genera un análisis integrando los índices de vegetación con las condiciones climáticas actuales y el pronóstico: "
          "explica cómo el clima afecta esos índices y propone recomendaciones concretas de manejo (riego, fertilización, monitoreo, etc.). "
          "Devuélvelo en español simple y profesional, con secciones claras: Resumen, Interpretación detallada, Recomendaciones."
    )
    resp = client.chat.completions.create(
        model=LLM_MODEL,
        temperature=LLM_TEMPERATURE,
        max_tokens=LLM_MAX_TOKENS,
        messages=[
            {"role":"system","content": system_prompt_llm},
            {"role":"user","content": prompt}
        ]
    )
    return resp.choices[0].message.content.strip()

# === Final Orchestration ===
def crear_reporte(path_tif, fecha, cultivo, ubicacion, fecha_siembra, fuente=None):
    promedios, _, tipo, meta = procesar_imagen(path_tif)
    texto_prelim = generar_informe(promedios, fecha, cultivo, ubicacion, tipo, fecha_siembra, fuente)
    informe_final = generar_informe_llm(texto_prelim)
    return {'informe': informe_final}

