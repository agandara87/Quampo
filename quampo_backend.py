
# quampo_backend.py

import cv2
import numpy as np
import matplotlib.pyplot as plt
import openai
import requests
from datetime import datetime

openai.api_key = "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"
OPENWEATHER_API_KEY = "fa977f2964c89bf890b1502681dfa742"

system_prompt_llm = """Sos un asesor técnico agrónomo digital que trabaja para Quampo...
(Contenido omitido para brevedad; reemplaza aquí con tu `system_prompt_llm` completo)"""

glossary = {
    "NDVI": {
        "interpretacion": [
            {"rango": [0.0, 0.3], "significado": "Baja actividad fotosintética / Estrés"},
            {"rango": [0.3, 0.6], "significado": "Moderada actividad fotosintética"},
            {"rango": [0.6, 1.0], "significado": "Buena salud del cultivo"}
        ]
    }
}

def obtener_clima_openweather(ubicacion):
    url = f"http://api.openweathermap.org/data/2.5/weather?q={ubicacion}&appid={OPENWEATHER_API_KEY}&units=metric&lang=es"
    respuesta = requests.get(url)
    data = respuesta.json()
    if respuesta.status_code != 200 or "main" not in data:
        return {"actual": {"temperatura": "No disponible", "humedad": "No disponible", "lluvia": "0 mm", "descripcion": "No disponible"},
                "pronostico": {"estado": "No disponible", "lluvia": "0 mm"}}
    return {
        "actual": {
            "temperatura": f"{data['main']['temp']}°C",
            "humedad": f"{data['main']['humidity']}%",
            "lluvia": f"{data.get('rain', {}).get('1h', 0)} mm",
            "descripcion": data['weather'][0]['description'].capitalize()
        },
        "pronostico": {
            "estado": data['weather'][0]['description'].capitalize(),
            "lluvia": f"{data.get('rain', {}).get('1h', 0)} mm"
        }
    }

def procesar_imagen(path):
    img = cv2.imread(path, -1)
    if img is None or len(img.shape) < 3:
        raise ValueError("Imagen no válida")
    tipo = "Multiespectral" if img.shape[2] > 3 else "RGB"
    red, green, blue = img[:, :, 2].astype(float), img[:, :, 1].astype(float), img[:, :, 0].astype(float)
    nir = img[:, :, 3].astype(float) if tipo == "Multiespectral" else None

    indices, promedios = {}, {}
    if nir is not None:
        ndvi = (nir - red) / (nir + red + 1e-5)
        indices["NDVI"] = ndvi
        promedios["NDVI"] = float(np.nanmean(ndvi))
    else:
        ndvi_est = (green - red) / (green + red + 1e-5)
        indices["NDVI_orientativo"] = ndvi_est
        promedios["NDVI_orientativo"] = float(np.nanmean(ndvi_est))

    indices["EVI"] = 2.5 * (blue - red) / (blue + 6 * red - 7.5 * green + 1e-5)
    indices["NDWI"] = (green - blue) / (green + blue + 1e-5)
    indices["SAVI"] = 1.5 * (red - blue) / (red + blue + 0.5)
    indices["GNDVI"] = (green - red) / (green + red + 1e-5)

    for k in ["EVI", "NDWI", "SAVI", "GNDVI"]:
        promedios[k] = float(np.nanmean(indices[k]))

    return promedios, indices, tipo

def interpretar_indice(valor, nombre):
    for r in glossary.get(nombre, {}).get("interpretacion", []):
        if r["rango"][0] <= valor <= r["rango"][1]:
            return r["significado"]
    return "Interpretación no disponible"

def etapa_fenologica(cultivo, dias):
    cultivo = cultivo.lower()
    if cultivo == "soja":
        return "Emergencia" if dias < 30 else "Floración" if dias < 60 else "Llenado de granos / maduración"
    if cultivo == "maíz":
        return "Vegetativo temprano" if dias < 35 else "Floración" if dias < 70 else "Llenado de granos"
    if cultivo == "trigo":
        return "Macollaje" if dias < 30 else "Espigazón" if dias < 70 else "Maduración"
    return "Etapa desconocida"

def generar_informe(prom, fecha, cultivo, ubicacion, tipo, fecha_siembra):
    dias = (datetime.strptime(fecha, "%Y-%m-%d") - datetime.strptime(fecha_siembra, "%Y-%m-%d")).days
    etapa = etapa_fenologica(cultivo, dias)
    clima = obtener_clima_openweather(ubicacion)
    resumen = [f"🛰 Informe Quampo | Fecha: {fecha}", f"Cultivo: {cultivo}", f"Ubicación: {ubicacion}", f"Tipo de imagen: {tipo}",
               f"Días desde siembra: {dias} días", f"Etapa fenológica estimada: {etapa}"]
    for k, v in prom.items():
        interpre = interpretar_indice(v, k)
        resumen.append(f"{k}: {v:.2f} → {interpre}")
    if tipo == "RGB" and "NDVI_orientativo" in prom:
        resumen.append("⚠️ NDVI orientativo calculado con bandas RGB (sin NIR).")
    resumen.append(f"Clima actual: {clima['actual']['descripcion']}, {clima['actual']['temperatura']}, humedad {clima['actual']['humedad']}")
    resumen.append(f"🔮 Pronóstico: {clima['pronostico']['estado']}, {clima['pronostico']['lluvia']} de lluvia")
    return "\n".join(resumen)

def generar_informe_llm(informe):
    client = openai.OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
    response = openai.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": system_prompt_llm},
            {"role": "user", "content": f"Este es el informe técnico:\n\n{informe}\n\nRedactá un informe agronómico claro y profesional para el productor."}
        ]
    )
    return response.choices[0].message.content.strip()
