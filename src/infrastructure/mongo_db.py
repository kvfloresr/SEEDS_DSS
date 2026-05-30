import uuid
from datetime import datetime
from pymongo import MongoClient
from bson import ObjectId  
import gridfs
import numpy as np

MONGO_URI = "mongodb://localhost:27017/"
DB_NAME = "SEED_DSS"

_client = None

def get_db():
    global _client
    if _client is None:
        _client = MongoClient(MONGO_URI)
    return _client[DB_NAME]


def save_analysis_group_safe(files, predicted_class, probability, probability_vector, features, model_version="seed_cnn_v1"):
    """
    Guarda en MongoDB un grupo de análisis con imágenes, ajustado al esquema de 'reports'.
    """
    db = get_db()
    fs = gridfs.GridFS(db, collection="images_files")

    group_guid = str(uuid.uuid4())  
    analysis_id = str(uuid.uuid4())
    sample_id = ObjectId()  

    image_ids = []
    first_image_id = None

    for filename, image_bytes in files:
        # Subir imagen a GridFS
        file_id = fs.put(
            image_bytes,
            filename=filename,
            contentType="image/png",
            uploaded_at=datetime.utcnow()
        )

        image_id = str(uuid.uuid4())
        image_doc = {
            "image_id": image_id,
            "sample_guid": group_guid,
            "sample_id": str(sample_id),  
            "path": f"gridfs://{file_id}",
            "filename": filename,
            "gridfs_id": ObjectId(file_id),
            "captured_at": datetime.utcnow(),
            "preprocessed": False,
            "metadata": {"width": None, "height": None}
        }
        db.images.insert_one(image_doc)

        image_ids.append(image_id)
        if first_image_id is None:
            first_image_id = image_id

    analysis_doc = {
        "analysis_id": analysis_id,
        "image_id": first_image_id,
        "sample_guid": group_guid,
        "sample_id": str(sample_id),  
        "predicted_class": predicted_class,
        "probability": float(probability),
        "probability_vector": probability_vector or [],
        "features": features,
        "model_version": model_version,
        "path": f"/analyses/{analysis_id}.json",
        "captured_at": datetime.utcnow(),
        "processed_at": datetime.utcnow(),
        "reviewed": False,
        "review_notes": None
    }

    db.analyses.insert_one(analysis_doc)

    system_user_id = ObjectId("507f1f77bcf86cd799439011")  

    report_doc = {
        "report_id": ObjectId(),  # Genera un nuevo ObjectId
        "sample_id": sample_id,  # ObjectId generado arriba
        "generated_at": datetime.utcnow(),  # Date requerido
        "generated_by": system_user_id,  # ObjectId de usuario
        "summary": {  # Object requerido: estructura un resumen
            "predicted_class": predicted_class,
            "probability": probability,
            "total_images": len(files),
            "details": f"Análisis grupal de {len(files)} imágenes."
        },
        "status": "completed", 
        "attachments": image_ids 
    }

    db.reports.insert_one(report_doc)

    print(f"[MongoDB] Grupo guardado correctamente. analysis_id={analysis_id}, report_id={report_doc['report_id']}")
    return analysis_id, first_image_id

def get_analyses(limit=10):
    db = get_db()
    docs = db.analyses.find().sort("processed_at", -1).limit(limit)
    results = []
    for d in docs:
        captured = d.get("captured_at")
        processed = d.get("processed_at")
        results.append({
            "_id": str(d.get("_id")),  # Convertir ObjectId a string
            "analysis_id": d.get("analysis_id"),
            "image_id": d.get("image_id"),
            "sample_guid": d.get("sample_guid"),
            "sample_id": d.get("sample_id"),
            "predicted_class": d.get("predicted_class"),
            "probability": d.get("probability"),
            "probability_vector": d.get("probability_vector", []),
            "features": d.get("features", {}),
            "model_version": d.get("model_version"),
            "path": d.get("path"),
            "captured_at": captured.strftime("%Y-%m-%d %H:%M:%S") if captured else None,
            "processed_at": processed.strftime("%Y-%m-%d %H:%M:%S") if processed else None,
            "reviewed": d.get("reviewed"),
            "review_notes": d.get("review_notes")
        })
    return results

def get_reports(limit=10):
    db = get_db()
    docs = db.analyses.find().sort("processed_at", -1).limit(limit) 
    results = [] 
    for d in docs: 
        processed = d.get("processed_at") 
        results.append({ 
            "analysis_id": d.get("analysis_id"), 
            "image_id": d.get("image_id"), 
            "filename": d.get("path"), 
            "predicted_class": d.get("predicted_class"),
            "probability": d.get("probability"),
            "features": d.get("features"),
            "processed_at": processed.strftime("%Y-%m-%d %H:%M:%S") if processed else None 
        }) 
    return results  