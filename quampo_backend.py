import os
import tempfile
import requests
import numpy as np
import rasterio
from datetime import datetime, timedelta
from openai import OpenAI
from sentinelhub import (
    SHConfig, BBox, CRS, SentinelHubRequest,
    DataCollection, MimeType, bbox_to_dimensions
)

# 1️⃣ Configuración global de credenciales
config = SHConfig()  # Lee SH_CLIENT_ID y SH_CLIENT_SECRET desde env vars

token = os.environ.get("OPENAI_API_KEY")
OPENWEATHER_API_KEY = os.getenv("OPENWEATHER_API_KEY")
GOOGLE_MAPS_API_KEY = os.getenv("GOOGLE_MAPS_API_KEY")

if not os.getenv("SH_CLIENT_ID") or not os.getenv("SH_CLIENT_SECRET"):
    raise EnvironmentError("Faltan credenciales de Sentinel Hub en env vars")
if not OPENWEATHER_API_KEY:
    raise EnvironmentError("Falta OPENWEATHER_API_KEY en env vars")
if not GOOGLE_MAPS_API_KEY:
    raise EnvironmentError("Falta GOOGLE_MAPS_API_KEY en env vars")
if not token:
    raise EnvironmentError("Falta OPENAI_API_KEY en env vars")

client = OpenAI(api_key=token)

# 2️⃣ Funciones Sentinel Hub

def get_sentinel_url(lat, lon, date_str,
                     size_m=5000,
                     resolution=10,
                     window_days=14,
                     max_cloud_coverage=1.0):
    """
    Busca la escena Sentinel-2 más limpia o más reciente en ±window_days alrededor de date_str.
    Retorna la URL del TIFF o None si no hay nada.
    """
    # 1. Validar fecha
    try:
        target = datetime.fromisoformat(date_str).date()
    except ValueError:
        print(f"[WARN] Fecha inválida: {date_str}")
        return None

    start = (target - timedelta(days=window_days)).isoformat()
    end   = (target + timedelta(days=window_days)).isoformat()

    # 2. Construir BBox y dimensiones
    half = size_m / 2
    bbox = BBox((
        lon - half/111320, lat - half/110540,
        lon + half/111320, lat + half/110540
    ), crs=CRS.WGS84)
    dims = bbox_to_dimensions(bbox, resolution=resolution)

    # 3. Evalscript básico RGB+NIR
    evalscript = """
        //VERSION=3
        function setup() {
            return {
                input: ["B04","B03","B02","B08"],
                output: { bands: 4 }
            };
        }
        function evaluatePixel(sample) {
            return [sample.B04, sample.B03, sample.B02, sample.B08];
        }
    """

    # 4. Intentar L2A NRT, luego L2A estándar, y por último L1C
    collections = [
        DataCollection.SENTINEL2_L2A,      # ideal, NRT si tu cuenta lo permite
        DataCollection.SENTINEL2_L2A,      # fallback a estándar
        DataCollection.SENTINEL2_L1C       # último recurso
    ]

    for coll in collections:
        print(f"[INFO] Buscando en {coll.name} entre {start} y {end}")
        input_data = SentinelHubRequest.input_data(
            coll,
            time_interval=(start, end),
            mosaicking_order='leastCC',
            other_args={
                'dataFilter': {
                    'maxCloudCoverage': max_cloud_coverage * 100
                }
            }
        )
        request = SentinelHubRequest(
            data_folder=None,
            evalscript=evalscript,
            input_data=[input_data],
            responses=[SentinelHubRequest.output_response('default', MimeType.TIFF)],
            bbox=bbox,
            size=dims,
            config=config
        )

        try:
            urls = request.get_url_list()
        except Exception as e:
            print(f"[WARN] Error al obtener URLs de {coll.name}: {e}")
            continue

        if urls:
            print(f"[INFO] {len(urls)} escenas encontradas en {coll.name}. Eligiendo la primera.")
            return urls[0]

        print(f"[INFO] No se encontraron escenas en {coll.name}.")

    return None


def get_satellite_image_dates(lat, lon, window_days=14):
    """
    Para tres fechas clave (hoy, -30d, -365d), devuelve la URL o None.
    """
    today    = datetime.utcnow().date().isoformat()
    prev     = (datetime.utcnow().date() - timedelta(days=30)).isoformat()
    year_ago = (datetime.utcnow().date() - timedelta(days=365)).isoformat()

    return {
        'actual':     get_sentinel_url(lat, lon, today, window_days=window_days),
        'anterior':   get_sentinel_url(lat, lon, prev, window_days=window_days),
        'anio_atras': get_sentinel_url(lat, lon, year_ago, window_days=window_days)
    }


def download_sentinel_image(lat, lon, date_str,
                            size_m=5000,
                            resolution=10,
                            window_days=14):
    """
    Descarga el TIFF en carpeta temporal. Usa L2A+L1C con window ampliado.
    Retorna ruta local o None.
    """
    url = get_sentinel_url(lat, lon, date_str,
                           size_m=size_m,
                           resolution=resolution,
                           window_days=window_days,
                           max_cloud_coverage=1.0)
    if not url:
        print("[INFO] No hay URL para descargar.")
        return None

    # Descarga
    try:
        resp = requests.get(url, timeout=60)
        resp.raise_for_status()
    except Exception as e:
        print(f"[ERROR] al descargar desde URL: {e}")
        return None

    # Guardar
    tmp_folder = tempfile.mkdtemp(prefix="sentinel_")
    local_path = os.path.join(tmp_folder, "sentinel_best.tif")
    with open(local_path, "wb") as f:
        f.write(resp.content)

    print(f"[INFO] Imagen descargada en {local_path}")
    return local_path

# 🧠 Prompt base para la LLM
system_prompt_llm = """
Sos un asesor técnico agrónomo digital que trabaja para Quampo, una plataforma que analiza imágenes satelitales de cultivos.
Tu objetivo es interpretar un informe técnico que incluye:
- Índices de vegetación (NDVI, EVI, NDWI, SAVI, GNDVI).
- Datos climáticos actuales, pronóstico diario extendido y datos espaciales.
- Información del cultivo (tipo, ubicación) y fechas de imágenes satelitales.
- Fecha de siembra y cantidad de días desde siembra.
- Etapa fenológica estimada.

Incluí además:
- Geocodificación de la localidad (Google Maps).
- URLs de las imágenes satelitales más recientes, la anterior y un año atrás.
- Pronóstico extendido de 7 días.

Siempre hablá en español simple, con profesionalismo.
"""

# 3️⃣ Otras funciones

# Geocodificación con Google Maps
def geocode_location(location_name):
    url = (
        f"https://maps.googleapis.com/maps/api/geocode/json?"
        f"address={requests.utils.quote(location_name)}&key={GOOGLE_MAPS_API_KEY}"
    )
    try:
        resp = requests.get(url, timeout=10)
        data = resp.json()
    except Exception as e:
        raise RuntimeError(f"Error de red geocodificando: {e}")

    if resp.status_code != 200 or data.get('status') != 'OK' or not data.get('results'):
        raise ValueError("No se pudo geocodificar la localidad")

    loc = data['results'][0]['geometry']['location']
    # Extraemos lat y lng del JSON:
    lat = loc['lat']
    lon = loc['lng']   # aquí tomamos loc['lng'] pero lo asignamos a variable llamada lon
    return lat, lon    # retornamos (lat, lon)

# Pronóstico extendido 7 días
def get_extended_forecast(lat, lon):
    url = (
        f"https://api.openweathermap.org/data/2.5/onecall?lat={lat}&lon={lon}"
        f"&exclude=current,minutely,hourly,alerts&units=metric&lang=es&appid={OPENWEATHER_API_KEY}"
    )
    try:
        resp = requests.get(url, timeout=10)
        data = resp.json()
    except Exception as e:
        print(f"[WARN] Error al obtener pronóstico: {e}")
        return []

    if resp.status_code != 200 or 'daily' not in data:
        return []

    forecast = []
    for day in data.get('daily', [])[:7]:
        date = datetime.utcfromtimestamp(day['dt']).strftime('%Y-%m-%d')
        desc = day['weather'][0]['description'].capitalize()
        temp = day['temp']['day']
        rain = day.get('rain', 0)
        forecast.append({'fecha': date, 'desc': desc, 'temp': f"{temp}°C", 'rain': f"{rain} mm"})
    return forecast

# Clima actual simplificado
def obtener_clima_current(lat, lon):
    url = (
        f"https://api.openweathermap.org/data/2.5/weather?lat={lat}&lon={lon}"
        f"&units=metric&lang=es&appid={OPENWEATHER_API_KEY}"
    )
    try:
        resp = requests.get(url, timeout=10)
        data = resp.json()
    except Exception as e:
        print(f"[WARN] Error al obtener clima actual: {e}")
        return None

    if resp.status_code != 200 or 'main' not in data:
        return None
    return {
        'descripcion': data['weather'][0]['description'].capitalize(),
        'temperatura': f"{data['main']['temp']}°C",
        'humedad': f"{data['main']['humidity']}%",
        'lluvia': f"{data.get('rain', {}).get('1h', 0)} mm"
    }

# Glosario interpretativo completo
glossary = {
    'NDVI': {'interpretacion': [
        {'rango': [0.0, 0.3], 'significado': 'Baja actividad fotosintética / Estrés'},
        {'rango': [0.3, 0.6], 'significado': 'Moderada actividad fotosintética'},
        {'rango': [0.6, 1.0], 'significado': 'Buena salud del cultivo'}
    ]},
    'EVI': {'interpretacion': [
        {'rango': [-1.0, 0.0], 'significado': 'Poca vegetación o suelo desnudo'},
        {'rango': [0.0, 0.2], 'significado': 'Estrés o vegetación escasa'},
        {'rango': [0.2, 0.5], 'significado': 'Vegetación moderada'},
        {'rango': [0.5, 1.0], 'significado': 'Vegetación densa y sana'}
    ]},
    'NDWI': {'interpretacion': [
        {'rango': [-1.0, 0.0], 'significado': 'Humedad baja'},
        {'rango': [0.0, 0.3], 'significado': 'Humedad moderada'},
        {'rango': [0.3, 1.0], 'significado': 'Alta humedad o agua libre'}
    ]},
    'SAVI': {'interpretacion': [
        {'rango': [-1.0, 0.0], 'significado': 'Estrés o suelo descubierto'},
        {'rango': [0.0, 0.2], 'significado': 'Vegetación escasa'},
        {'rango': [0.2, 0.5], 'significado': 'Vegetación moderada'},
        {'rango': [0.5, 1.0], 'significado': 'Vegetación densa'}
    ]},
    'GNDVI': {'interpretacion': [
        {'rango': [-1.0, 0.0], 'significado': 'Vegetación pobre o suelo desnudo'},
        {'rango': [0.0, 0.3], 'significado': 'Vegetación moderada'},
        {'rango': [0.3, 1.0], 'significado': 'Alta vegetación sana'}
    ]}
}

# Procesamiento de imagen con rasterio
def procesar_imagen(path):
    try:
        src = rasterio.open(path)
    except Exception as e:
        raise RuntimeError(f"Error abriendo imagen: {e}")

    with src:
        arr = src.read(masked=True)
        descs = src.descriptions or []
        meta = {
            'count': src.count,
            'crs': getattr(src, 'crs', None),
            'transform': getattr(src, 'transform', None),
            'bounds': getattr(src, 'bounds', None)
        }

    img = np.transpose(arr, (1, 2, 0))
    idx_map = {'red': 2, 'green': 1, 'blue': 0}
    # Detectar banda NIR por descripción
    for i, name in enumerate(descs):
        if name and 'nir' in name.lower():
            idx_map['nir'] = i
    # Si no hay descripción pero hay >=4 bandas, asumimos banda 3 es NIR
    if 'nir' not in idx_map and src.count >= 4:
        idx_map['nir'] = 3

    # Verificar que haya al menos bandas RGB
    if src.count < 3:
        raise ValueError("La imagen debe tener al menos 3 bandas (RGB)")

    R = img[:, :, idx_map['red']].astype(float)
    G = img[:, :, idx_map['green']].astype(float)
    B = img[:, :, idx_map['blue']].astype(float)
    N = img[:, :, idx_map.get('nir', 0)].astype(float) if 'nir' in idx_map else None
    has_nir = N is not None
    tipo = 'Multiespectral' if has_nir else 'RGB'

    if has_nir:
        max_val = max(np.nanmax(ch) for ch in (R, G, B, N))
        if max_val > 1:
            R, G, B, N = [ch / max_val for ch in (R, G, B, N)]

    indices, promedios = {}, {}
    # NDVI orientativo si solo RGB o si NIR real?
    or_ndvi = (G - R) / (G + R + 1e-5)
    indices['NDVI_orientativo'] = or_ndvi
    promedios['NDVI_orientativo'] = float(np.nanmean(or_ndvi))

    if has_nir:
        funcs = {
            'NDVI': (N - R) / (N + R + 1e-5),
            'EVI': 2.5 * (N - R) / (N + 6*R - 7.5*B + 1e-5),
            'NDWI': (G - N) / (G + N + 1e-5),
            'SAVI': 1.5 * (N - R) / (N + R + 0.5 + 1e-5),
            'GNDVI': (G - R) / (G + R + 1e-5)
        }
        for k, arr_v in funcs.items():
            indices[k] = arr_v
            promedios[k] = float(np.nanmean(arr_v))

    extent = None
    try:
        extent = (
            meta['bounds'].left, meta['bounds'].right,
            meta['bounds'].bottom, meta['bounds'].top
        ) if meta['bounds'] is not None else None
    except Exception:
        extent = None
    meta.update({'has_nir': has_nir, 'extent': extent})
    return promedios, indices, tipo, meta

# Interpretación de índices
def interpretar_indice(valor, nombre):
    for r in glossary.get(nombre, {}).get('interpretacion', []):
        if r['rango'][0] <= valor <= r['rango'][1]:
            return r['significado']
    return 'Interpretación no disponible'

# Etapa fenológica del cultivo
def etapa_fenologica(cultivo, dias):
    cultivo = cultivo.lower()
    if dias is None:
        return 'Fecha inválida'
    if cultivo == 'soja':
        return 'Emergencia' if dias < 30 else 'Floración' if dias < 60 else 'Llenado de granos/maduración'
    if cultivo == 'maíz':
        return 'Vegetativo temprano' if dias < 35 else 'Floración' if dias < 70 else 'Llenado de granos'
    if cultivo == 'trigo':
        return 'Macollaje' if dias < 30 else 'Espigazón' if dias < 70 else 'Maduración'
    return 'Etapa desconocida'

# Generar informe técnico completo con localidad y pronósticos
def generar_informe(promedios, fecha, cultivo, ubicacion, tipo, fecha_siembra, fuente=None):
    try:
        lat, lon = geocode_location(ubicacion)
    except Exception as e:
        raise RuntimeError(f"Error en geocodificación: {e}")

    imgs = get_satellite_image_dates(lat, lon)
    clima_act = obtener_clima_current(lat, lon)
    forecast7 = get_extended_forecast(lat, lon)

    try:
        dias = (datetime.strptime(fecha, '%Y-%m-%d') - datetime.strptime(fecha_siembra, '%Y-%m-%d')).days
    except Exception:
        dias = None
    etapa = etapa_fenologica(cultivo, dias)

    lines = []
    if fuente:
        lines.append(f"Fuente de imagen: {fuente}")
    lines.append(f"🛰 Informe Quampo | Fecha: {fecha}")
    lines.append(f"Cultivo: {cultivo}")
    lines.append(f"Ubicación: {ubicacion} (lat:{lat:.4f}, lon:{lon:.4f})")
    lines.append(f"Tipo de imagen: {tipo}")
    lines.append(f"Días desde siembra: {dias if dias is not None else 'N/A'}")
    lines.append(f"Etapa fenológica estimada: {etapa}")
    lines.append("-- Índices promedio --")
    for key, val in promedios.items():
        lines.append(f"{key}: {val:.3f} → {interpretar_indice(val, key)}")

    if clima_act:
        lines.append(
            f"Clima actual: {clima_act['descripcion']}, {clima_act['temperatura']}, "
            f"humedad {clima_act['humedad']}, lluvia {clima_act['lluvia']}"
        )

    lines.append("-- Pronóstico 7 días --")
    for d in forecast7:
        lines.append(f"{d['fecha']}: {d['desc']}, {d['temp']}, lluvia {d['rain']}")

    lines.append("-- Imágenes satelitales --")
    for label, url in [('Actual', imgs.get('actual')), ('Anterior (~30 días)', imgs.get('anterior')), ('Hace un año', imgs.get('anio_atras'))]:
        if url:
            lines.append(f"{label}: {url}")
        else:
            lines.append(f"{label}: no se encontró imagen en ±7 días")

    return "\n".join(lines)

# Generar informe agronómico con LLM
def generar_informe_llm(informe_tecnico):
    resp = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": system_prompt_llm},
            {"role": "user", "content": informe_tecnico}
        ],
        temperature=0.7,
        max_tokens=1024
    )
    return resp.choices[0].message.content.strip()

# Bloque de prueba local
if __name__ == "__main__":
    lat, lon = -34.6037, -58.3816
    print("URLs Sentinel Hub:", get_satellite_image_dates(lat, lon))
