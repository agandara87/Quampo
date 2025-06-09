
# 📦 Librerías necesarias
import cv2
import numpy as np
import matplotlib.pyplot as plt
import openai
import os
import requests
from datetime import datetime
from openai import OpenAI
import rasterio

# 🔐 Configurar tu API Key de OpenAI
client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
# 🌦 API Key de OpenWeatherMap
OPENWEATHER_API_KEY = "fa977f2964c89bf890b1502681dfa742"

# 🧠 Prompt base para la LLM
system_prompt_llm = """
Sos un asesor técnico agrónomo digital que trabaja para Quampo, una plataforma que analiza imágenes satelitales de cultivos.
Tu objetivo es interpretar un informe técnico que incluye:
- Índices de vegetación (NDVI, NDWI, SAVI, etc.).
- Datos climáticos actuales y pronóstico.
- Información del cultivo (tipo, ubicación).
- Fecha de siembra y cantidad de días desde siembra.
- Etapa fenológica estimada.

A partir de eso, redactá un informe empático, técnico y claro para un productor agropecuario. Incluí:
- Estado actual del cultivo según su etapa fenológica.
- Posibles causas de índices bajos o altos.
- Cómo influye el clima en esta etapa específica.
- Recomendaciones agronómicas prácticas.
- Alertas si hay riesgo climático o signos de estrés vegetal.

Siempre hablá en español simple, con profesionalismo. No repitas los datos textuales, usalos para **interpretar**.
"""

# 🌤 Función de clima real
def obtener_clima_openweather(ubicacion):
    url = f"http://api.openweathermap.org/data/2.5/weather?q={ubicacion}&appid={OPENWEATHER_API_KEY}&units=metric&lang=es"
    respuesta = requests.get(url)
    data = respuesta.json()

    if respuesta.status_code != 200 or "main" not in data:
        print("⚠️ Ubicación no válida o sin datos climáticos disponibles.")
        return {
            "actual": {"temperatura": "No disponible", "humedad": "No disponible", "lluvia": "0 mm", "descripcion": "No disponible"},
            "pronostico": {"estado": "No disponible", "lluvia": "0 mm"}
        }

    temp = f"{data['main']['temp']}°C"
    humedad = f"{data['main']['humidity']}%"
    descripcion = data['weather'][0]['description'].capitalize()
    lluvia = f"{data.get('rain', {}).get('1h', 0)} mm"

    return {
        "actual": {
            "temperatura": temp,
            "humedad": humedad,
            "lluvia": lluvia,
            "descripcion": descripcion
        },
        "pronostico": {
            "estado": descripcion,
            "lluvia": lluvia
        }
    }

# 📘 Glosario interpretativo NDVI
glossary = {
    "NDVI": {
        "interpretacion": [
            {"rango": [0.0, 0.3], "significado": "Baja actividad fotosintética / Estrés"},
            {"rango": [0.3, 0.6], "significado": "Moderada actividad fotosintética"},
            {"rango": [0.6, 1.0], "significado": "Buena salud del cultivo"}
        ]
    }
}

# 🧮 Funciones de índices
def calcular_ndvi(img): return (img[:,:,0] - img[:,:,2]) / (img[:,:,0] + img[:,:,2] + 1e-5)
def calcular_ndwi(img): return (img[:,:,0] - img[:,:,1]) / (img[:,:,0] + img[:,:,1] + 1e-5)
def calcular_savi(img): return 1.5 * (img[:,:,0] - img[:,:,2]) / (img[:,:,0] + img[:,:,2] + 0.5)
def calcular_evi(img): return 2.5 * (img[:,:,0] - img[:,:,2]) / (img[:,:,0] + 6 * img[:,:,2] - 7.5 * img[:,:,1] + 1)
def calcular_gndvi(img): return (img[:,:,0] - img[:,:,1]) / (img[:,:,0] + img[:,:,1] + 1e-5)

# 🛰 Procesamiento de imagen
import rasterio
import numpy as np

def procesar_imagen(path):
    with rasterio.open(path) as src:
        bandas = src.count
        nombres = src.descriptions  # nombres de las bandas (si están)

        print(f"📷 Imagen con {bandas} bandas.")
        if nombres:
            print("🧾 Bandas detectadas:", nombres)
        else:
            print("⚠️ No hay descripciones de bandas disponibles.")

        # Leer todas las bandas
        img = src.read()  # shape: (bandas, altura, ancho)

        # Normalizar a formato (altura, ancho, bandas)
        img = np.transpose(img, (1, 2, 0))

    tipo = "Multiespectral" if bandas > 3 else "RGB"

    # Heurística para usar bandas
    red = img[:, :, 2].astype(float) if bandas >= 3 else None
    green = img[:, :, 1].astype(float) if bandas >= 2 else None
    blue = img[:, :, 0].astype(float) if bandas >= 1 else None
    nir = img[:, :, 3].astype(float) if bandas >= 4 else None  # Asumimos que banda 4 = NIR, salvo que se detecte otra

    # Si hay descripciones, intentamos mapear más exactamente
    if nombres and "nir" in "".join(n.lower() for n in nombres if n):
        for idx, name in enumerate(nombres):
            if name and "nir" in name.lower():
                nir = img[:, :, idx].astype(float)
                print(f"✅ Banda NIR detectada en la posición {idx+1}")
                break

    indices = {}
    promedios = {}

    if nir is not None and red is not None:
        ndvi = (nir - red) / (nir + red + 1e-5)
        indices["NDVI"] = ndvi
        promedios["NDVI"] = float(np.nanmean(ndvi))
    elif green is not None and red is not None:
        print("⚠️ NDVI estimado con bandas RGB (sin NIR)")
        ndvi_est = (green - red) / (green + red + 1e-5)
        indices["NDVI_orientativo"] = ndvi_est
        promedios["NDVI_orientativo"] = float(np.nanmean(ndvi_est))

    # Otros índices
    if red is not None and green is not None and blue is not None:
        evi = 2.5 * (blue - red) / (blue + 6 * red - 7.5 * green + 1e-5)
        ndwi = (green - blue) / (green + blue + 1e-5)
        savi = 1.5 * (red - blue) / (red + blue + 0.5)
        gndvi = (green - red) / (green + red + 1e-5)

        indices.update({"EVI": evi, "NDWI": ndwi, "SAVI": savi, "GNDVI": gndvi})
        promedios.update({
            "EVI": float(np.nanmean(evi)),
            "NDWI": float(np.nanmean(ndwi)),
            "SAVI": float(np.nanmean(savi)),
            "GNDVI": float(np.nanmean(gndvi))
        })

    return promedios, indices, tipo
)

    # Otros índices (se pueden calcular igual con RGB)
    evi = 2.5 * (blue - red) / (blue + 6 * red - 7.5 * green + 1e-5)
    ndwi = (green - blue) / (green + blue + 1e-5)
    savi = 1.5 * (red - blue) / (red + blue + 0.5)
    gndvi = (green - red) / (green + red + 1e-5)

    indices["EVI"] = evi
    indices["NDWI"] = ndwi
    indices["SAVI"] = savi
    indices["GNDVI"] = gndvi

    promedios["EVI"] = float(np.nanmean(evi))
    promedios["NDWI"] = float(np.nanmean(ndwi))
    promedios["SAVI"] = float(np.nanmean(savi))
    promedios["GNDVI"] = float(np.nanmean(gndvi))

    return promedios, indices, tipo


# 📊 Interpretar índice
def interpretar_indice(valor, nombre):
    for r in glossary.get(nombre, {}).get("interpretacion", []):
        if r["rango"][0] <= valor <= r["rango"][1]: return r["significado"]
    return "Interpretación no disponible"

# 🌱 Etapa fenológica
def etapa_fenologica(cultivo, dias):
    cultivo = cultivo.lower()
    if cultivo == "soja":
        return "Emergencia" if dias < 30 else "Floración" if dias < 60 else "Llenado de granos / maduración"
    if cultivo == "maíz":
        return "Vegetativo temprano" if dias < 35 else "Floración" if dias < 70 else "Llenado de granos"
    if cultivo == "trigo":
        return "Macollaje" if dias < 30 else "Espigazón" if dias < 70 else "Maduración"
    return "Etapa desconocida"

# 📄 Generar informe técnico
def generar_informe(prom, fecha, cultivo, ubicacion, tipo, fecha_siembra):
    dias = (datetime.strptime(fecha, "%Y-%m-%d") - datetime.strptime(fecha_siembra, "%Y-%m-%d")).days
    etapa = etapa_fenologica(cultivo, dias)
    clima = obtener_clima_openweather(ubicacion)

    resumen = [
        f"🛰 Informe Quampo | Fecha: {fecha}",
        f"Cultivo: {cultivo}",
        f"Ubicación: {ubicacion}",
        f"Tipo de imagen: {tipo}",
        f"Días desde siembra: {dias} días",
        f"Etapa fenológica estimada: {etapa}"
    ]
    for k, v in prom.items():
        interpre = interpretar_indice(v, k)
        resumen.append(f"{k}: {v:.2f} → {interpre}")

    if tipo == "RGB" and "NDVI_orientativo" in prom:
        resumen.append("\n⚠️ Nota: El NDVI fue estimado a partir de bandas RGB. Es solo orientativo, ya que falta la banda NIR.")

    if "NDVI" not in prom:
        resumen.append("⚠️ Advertencia: No se calculó NDVI (falta banda NIR).")

    resumen.append(f"Clima actual: {clima['actual']['descripcion']}, {clima['actual']['temperatura']}, humedad {clima['actual']['humedad']}")
    resumen.append(f"🔮 Pronóstico: {clima['pronostico']['estado']}, {clima['pronostico']['lluvia']} de lluvia")

    return "\n".join(resumen)

# 🤖 Informe GPT

def generar_informe_llm(informe):
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": system_prompt_llm},
            {"role": "user", "content": f"Este es el informe técnico:\n\n{informe}\n\nRedactá un informe agronómico claro y profesional para el productor."}
        ]
    )
    return response.choices[0].message.content.strip()
