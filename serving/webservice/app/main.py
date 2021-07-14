import base64
import logging
import os
import pathlib
import ssl
from datetime import datetime

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from classifier import Resnetv2c3

ssl._create_default_https_context = ssl._create_unverified_context

app = FastAPI()

origins = [
    "http://localhost",
    "http://localhost:8080",
    "http://localhost:3000",
    "http://3.137.210.188:3000"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

SERVING_HOST = os.getenv("SERVING_HOST", "localhost")
SERVING_PORT = os.getenv("SERVING_PORT", "8501")
SERVING_BASE_URL = f'{SERVING_HOST}:{SERVING_PORT}/v1/models'
SERVING_MODEL_NAME = os.getenv("SERVING_MODEL_NAME", "retinopathy_v3-c3-resnet50v2")

RESNET_CLASS_NAMES_FILE = os.getenv("RESNET_CLASS_NAMES_FILE", str(pathlib.Path(
    "../../models/deap-retinopathy-v3-c3-BaseModel.RESNET50_v2", "1", "class_names.json")))

image_classifier = Resnetv2c3(RESNET_CLASS_NAMES_FILE, SERVING_BASE_URL, SERVING_MODEL_NAME)

diagnosis_mapping = {
    "No DR Signs": {
        "resultado": "Sin Signos de RD",
        "descripcion": "No se encuentran rasgos caraterísticos de la enfermedad"
    },
    "NPDR": {
        "resultado": "Con signos de RD No Proliferativo",
        "descripcion": "La imagen analizada presenta rasgos caraterísticos de la enfermedad"
    },
    "PDR": {
        "resultado": "Con signos de RD Proliferativo",
        "descripcion": "La imagen analizada presenta rasgos caraterísticos de la enfermedad"
    }
}


@app.get("/retinopatia")
def read_root():
    return {"status": "ok"}


# Se retorna siempre la misma imagen enviada en resultado_imagen
# Salvo que el algoritmo principal sea segmentación o deteccion
# en cuyo caso se envia esa imagen
@app.post("/retinopatia/diagnostico")
def get_classification(file: UploadFile = File(...)):
    data = file.file.read()
    data_str = base64.b64encode(data)

    if file.content_type.startswith('image/') is False:
        raise HTTPException(status_code=400, detail={
            "tipo": "formato",
            "mensaje": "No se pudo procesar la imagen"
        })

    try:
        # contents = file.file.read()
        start = datetime.now()
        predicted_class, probabilities = image_classifier.classify_image(file.file)
        observaciones = [f"{diagnosis_mapping[class_name]['resultado']} ({prob:.2f})" for class_name, prob in
                         probabilities]
        observaciones = "; ".join(observaciones)

        logging.info(f"Predicted Class: {predicted_class}")

        try:
            diagnosis = diagnosis_mapping[predicted_class]
        except KeyError:
            diagnosis = {
                "resultado": "Diágnostico inválido",
                "descripcion": "Se obtuvo un diagnóstico inválido."
            }

        time = datetime.now() - start
        return {
            "tipo": "Clasificación",
            "resultado_imagen": data_str,
            **diagnosis,
            "resultados_adicionales": [],
            # Solo a modo de ejemplo
            # "resultados_adicionales": [
            #     {
            #         "tipo": "Ejemplo 1",
            #         "resultado": "Ejemplo resultado text",
            #         "resultado_imagen": data_str,
            #         "descripcion": "Ejemplo descripcion",
            #         "file_name": file.filename
            #     }
            # ],
            "file_name": file.filename,
            "tiempo": time,
            "fecha": start.strftime("%d-%m-%Y %H:%M:%S"),
            "observaciones": observaciones,
            "info": {}
        }
    except Exception as error:
        logging.exception(error)
        # e = sys.exc_info()[1]
        raise HTTPException(status_code=500, detail={
            "tipo": "servidor",
            "mensaje": "Ha ocurrido un error en el servidor"
        })
