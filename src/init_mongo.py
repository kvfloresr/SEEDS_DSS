from pymongo import MongoClient, ASCENDING, DESCENDING
import gridfs
from datetime import datetime
import pprint

MONGO_URI = "mongodb://localhost:27017/"
DB_NAME = "SeedDSS_NoSQL"

def get_client():
    return MongoClient(MONGO_URI)

def create_collections_and_validators(db):
    # Definición de validadores 
    usuarios_validator = {
        "$jsonSchema": {
            "bsonType": "object",
            "required": ["user_id", "nombre", "rol", "correo"],
            "properties": {
                "user_id": {"bsonType": "string"},
                "nombre": {"bsonType": "string"},
                "rol": {"enum": ["Laboratorista", "Administrador", "Productor"]},
                "correo": {"bsonType": "string"},
                "metadata": {"bsonType": "object"}
            }
        }
    }

    imagenes_validator = {
        "$jsonSchema": {
            "bsonType": "object",
            "required": ["imagen_id", "sample_id", "ruta", "fecha_captura"],
            "properties": {
                "imagen_id": {"bsonType": "string"},
                "sample_id": {"bsonType": "string"},
                "ruta": {"bsonType": "string"},
                "resolucion": {"bsonType": "string"},
                "procesada": {"bsonType": "bool"},
                "metadatos_captura": {"bsonType": "object"}
            }
        }
    }

    resultados_validator = {
        "$jsonSchema": {
            "bsonType": "object",
            "required": ["resultado_id", "imagen_id", "clasificacion", "confianza", "fecha_procesamiento"],
            "properties": {
                "resultado_id": {"bsonType": "string"},
                "imagen_id": {"bsonType": "string"},
                "clasificacion": {"bsonType": "string"},
                "confianza": {"bsonType": "double"},
                "fecha_procesamiento": {"bsonType": "date"},
                "caracteristicas": {"bsonType": "object"},
                "model_version": {"bsonType": "string"}
            }
        }
    }

    bitacora_validator = {
        "$jsonSchema": {
            "bsonType": "object",
            "required": ["usuario", "accion", "fecha_hora"],
            "properties": {
                "usuario": {"bsonType": "string"},
                "accion": {"bsonType": "string"},
                "entidad": {"bsonType": "string"},
                "detalle": {"bsonType": "string"},
                "fecha_hora": {"bsonType": "date"}
            }
        }
    }

    # Crear colecciones (si no existen) 
    existing = db.list_collection_names()

    if "usuarios" not in existing:
        db.create_collection("usuarios", validator=usuarios_validator)
    if "imagenes" not in existing:
        db.create_collection("imagenes", validator=imagenes_validator)
    if "resultados_ia" not in existing:
        db.create_collection("resultados_ia", validator=resultados_validator)
    if "bitacora" not in existing:
        db.create_collection("bitacora", validator=bitacora_validator)

    # Otras colecciones flexibles 
    for name in ["productores", "lotes", "muestras", "reportes"]:
        if name not in existing:
            db.create_collection(name)

def create_indexes(db):
    # Índices útiles para consultas frecuentes
    db.usuarios.create_index([("user_id", ASCENDING)], unique=True, name="idx_user_id")
    db.imagenes.create_index([("imagen_id", ASCENDING)], unique=True, name="idx_imagen_id")
    db.imagenes.create_index([("sample_id", ASCENDING)], name="idx_sample_id")
    db.resultados_ia.create_index([("resultado_id", ASCENDING)], unique=True, name="idx_result_id")
    db.resultados_ia.create_index([("imagen_id", ASCENDING)], name="idx_result_image")
    db.productores.create_index([("producer_id", ASCENDING)], unique=True, name="idx_producer_id")
    db.lotes.create_index([("lot_id", ASCENDING)], unique=True, name="idx_lot_id")
    db.muestras.create_index([("sample_id", ASCENDING)], unique=True, name="idx_sample_id_unique")
    db.bitacora.create_index([("fecha_hora", DESCENDING)], name="idx_bitacora_fecha")

def seed_initial_documents(db):
    # Usuarios de ejemplo
    usuarios = [
        {
            "user_id": "ADMIN01",
            "nombre": "Administrador SeedDSS",
            "rol": "Administrador",
            "correo": "admin@seeddss.local",
            "metadata": {"created_by": "init_script", "created_at": datetime.utcnow()}
        },
        {
            "user_id": "LAB01",
            "nombre": "Valeria Flores",
            "rol": "Laboratorista",
            "correo": "vflores@iniaf.gob.bo",
            "metadata": {"laboratorio": "Laboratorio Central", "created_at": datetime.utcnow()}
        },
        {
            "user_id": "PROD01",
            "nombre": "Semillas del Oriente S.R.L.",
            "rol": "Productor",
            "correo": "contacto@semillasoriente.bo",
            "metadata": {"registro_iniaf": "R-2023-0001", "created_at": datetime.utcnow()}
        }
    ]
    for u in usuarios:
        try:
            db.usuarios.insert_one(u)
        except Exception as e:
            print("Usuario insert skip (ya existe?):", e)

    # Productor / Lote / Muestra de ejemplo
    producer = {
        "producer_id": "PROD01",
        "nombre": "Semillas del Oriente S.R.L.",
        "registro_iniaf": "R-2023-0001",
        "ubicacion": {"departamento": "Santa Cruz", "municipio": "San Julián"},
        "contacto": {"telefono": "72100000", "email": "contacto@semillasoriente.bo"},
        "created_at": datetime.utcnow()
    }

    lot = {
        "lot_id": "LOT2025-0001",
        "producer_id": "PROD01",
        "variedad": "Soya-Alpha",
        "campana": "2025/1",
        "fecha_ingreso": datetime(2025, 9, 25),
        "superficie_ha": 50,
        "estado": "En evaluación"
    }

    sample = {
        "sample_id": "SAMPLE-LOT2025-0001-1",
        "lot_id": "LOT2025-0001",
        "fecha_recepcion": datetime.utcnow(),
        "tamano_muestra": 500,
        "observaciones": "Muestra representativa",
    }

    try:
        db.productores.insert_one(producer)
        db.lotes.insert_one(lot)
        db.muestras.insert_one(sample)
    except Exception as e:
        print("Seed insert error (ya existe?):", e)

    
    db.bitacora.insert_one({
        "usuario": "LAB01",
        "accion": "Inicialización Base",
        "entidad": "init_script",
        "detalle": "Creación de colecciones y documentos de ejemplo",
        "fecha_hora": datetime.utcnow()
    })

def demo_gridfs_bucket(db):
    # Inicializa GridFS 
    fs = gridfs.GridFS(db, collection="imagenes_files")
    print("GridFS bucket preparado: imagenes_files.files / imagenes_files.chunks")

def main():
    client = get_client()
    db = client[DB_NAME]
    print("Conectando a MongoDB:", MONGO_URI)
    print("Base de datos:", DB_NAME)

    create_collections_and_validators(db)
    print("Colecciones y validadores creados (si no existían).")

    create_indexes(db)
    print("Índices creados.")

    seed_initial_documents(db)
    print("Documentos iniciales insertados (usuarios, productor, lote, muestra).")

    demo_gridfs_bucket(db)

    print("\nColecciones actuales:")
    pprint.pprint(db.list_collection_names())

    print("\nEjemplo usuario LAB01:")
    print(db.usuarios.find_one({"user_id": "LAB01"}))

    client.close()

if __name__ == "__main__":
    main()
