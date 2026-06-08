import os
import matplotlib
matplotlib.use('Agg') 
import cv2
import matplotlib.pyplot as plt
import uuid
from datetime import datetime
import cv2
from flask import Flask, request, jsonify, send_file, abort
from flask_cors import CORS
import gridfs
import numpy as np
import io
from io import BytesIO
import time
from datetime import datetime
from werkzeug.utils import secure_filename
import base64
from src.application.utils.feature_extraction import analyze_morphology_visual
from src.application.services.analysis_service import analyze_files_preview, insert_report_group, analyze_image, preprocess_image, predict_step_by_step
from src.infrastructure.mongo_db import save_analysis_group_safe, get_db, get_reports as mongo_get_reports
from src.presentation.reports.pdf_report_generator import generate_pdf_in_memory
from flask_jwt_extended import JWTManager, jwt_required, create_access_token, get_jwt_identity, verify_jwt_in_request, get_jwt
from werkzeug.security import check_password_hash, generate_password_hash
from src.infrastructure.sql_db import (
    delete_producer, get_users, insert_user, get_lots, insert_lot, get_samples, insert_sample, get_reports_sql, get_sql_connection, 
    delete_lot, delete_sample, update_producer, update_user, delete_user, change_user_password, get_roles,
    insert_producer, start_wizard_session, update_wizard_session, get_wizard_session, rollback_wizard_session, get_producer_by_cod_or_name, get_or_insert_producer, get_producers
)
from pathlib import Path
import tensorflow as tf
from tensorflow import keras
import json
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
from src.tests.evaluate_real_cnn import predict_process, evaluate_model
from src.application.services.seed_counter import count_from_bytes
from flask import Response
from src.interface.camera_stream import gen_frames, latest, latest_jpeg, start_camera as _start_cam
from src.application.services.per_seed_analysis import analyze_per_seed



# Config
UPLOAD_TMP = os.environ.get("UPLOAD_TMP", "/tmp/seed_dss_uploads") 
os.makedirs(UPLOAD_TMP, exist_ok=True)

app = Flask(__name__)
CORS(app, resources={
    r"/api/*": {
        "origins": "*",
        "methods": ["GET", "POST", "PUT", "DELETE", "OPTIONS"],
        "allow_headers": ["Content-Type", "Authorization"],
        "supports_credentials": True
    }
}) 
app.config["JWT_SECRET_KEY"] = "seed_dss_secret_key"  
app.config["JWT_TOKEN_LOCATION"] = ["headers"]
app.config["JWT_ACCESS_TOKEN_EXPIRES"] = False

jwt = JWTManager(app)


try:
    BASE = Path(__file__).resolve().parents[2]   # raíz del proyecto

    DATA_DIR = BASE / "data" / "processed" / "test"
    MODEL_PATH = BASE / "models" / "best_model.keras"
    MODEL_DIR = BASE / "models"
    CLASS_PATH = BASE / "models" / "class_indices.json"
    IMG_SIZE = (128, 128)

    print(f"[DEBUG] BASE path: {BASE}")
    print(f"[DEBUG] MODEL_DIR: {MODEL_DIR}")
    print(f"[DEBUG] MODEL_PATH: {MODEL_PATH}")
    print(f"[DEBUG] CLASS_PATH: {CLASS_PATH}")

    if not MODEL_PATH.exists():
        print(f"[ERROR] Modelo no encontrado en {MODEL_PATH}")
        raise FileNotFoundError(f"Modelo no encontrado en {MODEL_PATH}")
    if not CLASS_PATH.exists():
        print(f"[ERROR] Clases no encontradas en {CLASS_PATH}")
        raise FileNotFoundError(f"Clases no encontradas en {CLASS_PATH}")

    print("[DEBUG] Cargando modelo...")
    model = keras.models.load_model(str(MODEL_PATH))
    print("[DEBUG] Modelo cargado exitosamente")

    print("[DEBUG] Cargando clases...")
    with open(CLASS_PATH, "r", encoding="utf-8") as f:
        idx2class = json.load(f)
    print(f"[DEBUG] Clases cargadas: {list(idx2class.keys())}")

    # Inyectar el modelo ya cargado en per_seed_analysis para evitar cargarlo
    # dos veces (TF falla en hilos secundarios al cargar el modelo de nuevo)
    import src.application.services.per_seed_analysis as _psa
    _psa._model = model
    _psa._idx2class = idx2class
    print("[DEBUG] Modelo compartido con per_seed_analysis OK")

    ## Pre-iniciar la cámara en segundo plano al arrancar el backend
    _start_cam(width=1280, height=720)
    print("[DEBUG] Cámara pre-iniciada en segundo plano")

    print("[DEBUG] Creando feature_extractor...")
    feature_extractor = tf.keras.Model(
        inputs=model.input,
        outputs=model.layers[-3].output
    )
    print("[INFO] Modelo y extractor cargados correctamente.")
except Exception as e:
    print(f"[ERROR] Error cargando modelo: {e}")
    import traceback
    traceback.print_exc()
    model = None
    idx2class = {}
    feature_extractor = None



@app.route("/api/analyze_group", methods=["POST"])
def analyze_group():
    try:
        files = request.files.getlist("files")
        generated_by = request.form.get("generated_by", "IA-System")
        sample_id = request.form.get("sample_id", str(uuid.uuid4()))
        observations = request.form.get("observations", "")

        result = analyze_files_preview(files, generated_by, sample_id, observations)
        return jsonify({
            "status": "ok",
            "analysis_id": result["analysis_id"],
            "predicted_class": result["predicted_class"],
            "probability": result["probability"],
            "features": result["features"],
            "probability_vector": result.get("probability_vector", [])  
        }), 200

    except Exception as e:
        app.logger.error(f"Error en analyze_group: {e}")
        return jsonify({"status": "error", "message": str(e)}), 500
    
@app.route("/api/reports", methods=["GET"])
def list_reports():
    try:
        limit = int(request.args.get("limit", 10))
        rows = mongo_get_reports(limit=limit) 
        return jsonify(rows), 200
    except Exception as e:
        app.logger.exception("Error en list_reports")
        return jsonify({"error": str(e)}), 500

@app.route("/api/get_overlay_images/<analysis_id>", methods=["GET"])
def get_overlay_images(analysis_id):
    try:
        db = get_db()
        fs = gridfs.GridFS(db, collection="images_files")
        
        print(f"[API] Buscando análisis con ID: {analysis_id}")
        
        analysis = db.analyses.find_one({"analysis_id": analysis_id})
        if not analysis:
            print(f"[API] Análisis no encontrado: {analysis_id}")
            return jsonify({"error": "Análisis no encontrado"}), 404
        
        sample_guid = analysis.get("sample_guid")
        print(f"[API] sample_guid del análisis: {sample_guid}")
        
        images = []
        for image_doc in db.images.find({"sample_guid": sample_guid}):
            filename = image_doc.get("filename", "")
            print(f"[API] Revisando imagen: {filename}")
            if "_overlay" in filename.lower():
                print(f"[API] Encontrada overlay: {filename}")
                file_data = fs.get(image_doc["gridfs_id"])
                import base64
                encoded = base64.b64encode(file_data.read()).decode('utf-8')
                images.append({
                    "filename": filename,
                    "image_base64": f"data:image/png;base64,{encoded}"
                })
        
        print(f"[API] Total overlays encontradas: {len(images)}")
        return jsonify({"images": images}), 200
    except Exception as e:
        print(f"[API] Error: {e}")
        return jsonify({"error": str(e)}), 500
    
@app.route("/api/download_report/<analysis_id>", methods=["GET"])
def download_report(analysis_id):
    try:
        pdf_buffer = generate_pdf_in_memory(analysis_id)  
        if not pdf_buffer:
            return jsonify({"error": "PDF no encontrado"}), 404
        
        pdf_buffer.seek(0)  
        return send_file(pdf_buffer, as_attachment=True,
                        download_name=f"Seed_Report_{analysis_id}.pdf",
                        mimetype="application/pdf")
    except Exception as e:
        app.logger.exception("Error en download_report")
        return jsonify({"error": str(e)}), 500

@app.route("/api/login", methods=["POST"])
def login():
    try:
        data = request.json
        email = data.get("email")
        password = data.get("password")
        if not email or not password:
            return jsonify({"error": "Email y contraseña requeridos"}), 400
        conn = get_sql_connection()
        cursor = conn.cursor()
        cursor.execute("""
            SELECT u.user_id, u.name, u.email, u.password, r.name AS role_name
            FROM Users u
            JOIN Roles r ON u.role_id = r.role_id
            WHERE u.email = ?
        """, (email,))
        row = cursor.fetchone()
        conn.close()
        if not row:
            return jsonify({"error": "Usuario no encontrado"}), 404
        db_pass = str(row[3])
        if db_pass.startswith("scrypt"):
            if not check_password_hash(db_pass, password):
                return jsonify({"error": "Contraseña incorrecta"}), 401
        else:
            if db_pass != password:
                return jsonify({"error": "Contraseña incorrecta"}), 401
        user = {
            "user_id": str(row[0]),
            "name": row[1],
            "email": row[2],
            "role_name": row[4]
        }
        # Crear token JWT con identity como string y claims adicionales
        access_token = create_access_token(
            identity=user["user_id"],  # Identity debe ser string
            additional_claims=user  # Claims adicionales
        )
        return jsonify({"status": "ok", "user": user, "token": access_token}), 200
    except Exception as e:
        app.logger.exception("Error en login")
        return jsonify({"error": str(e)}), 500

# Obtener usuario actual
@app.route("/api/me", methods=["GET"])
@jwt_required()
def get_current_user():
    current_user = get_jwt_identity()
    return jsonify(current_user), 200

@app.route("/api/users", methods=["GET", "POST", "PUT", "DELETE"])
def manage_users():
    if request.method == "GET":
        return jsonify(get_users()), 200
    
    # Solo una vez aquí
    try:
        verify_jwt_in_request()
        current_user = get_jwt()  # Cambia aquí para obtener claims
        if current_user["role_name"] not in ["Administrador", "Supervisor"]:
            return jsonify({"error": "No autorizado"}), 403
    except Exception as e:
        print(f"[ERROR] JWT inválido en manage_users: {e}")
        return jsonify({"error": "Token inválido"}), 401
    
    if request.method == "POST":
        data = request.json
        print(f"[DEBUG] POST /api/users data: {data}")
        role = data.get("role")
        if not role:
            return jsonify({"error": "role requerido"}), 400
        if current_user["role_name"] == "Supervisor" and role == "Administrador": 
            return jsonify({"error": "No puedes crear administradores"}), 403
        print(f"[DEBUG] Llamando insert_user con: name={data['name']}, email={data['email']}, role={role}")
        try:
            insert_user(data["name"], data["email"], generate_password_hash(data["password"]), role)
            return jsonify({"message": "Usuario creado"}), 201
        except ValueError as e:
            print(f"[ERROR] ValueError en insert_user: {e}")
            return jsonify({"error": str(e)}), 400
        except Exception as e:
            app.logger.exception("Error creando usuario")
            return jsonify({"error": str(e)}), 500
    
    if request.method == "PUT":
        data = request.json
        user_id = data.get("user_id")
        name = data.get("name")
        email = data.get("email")
        role_id = data.get("role_id")
        active = data.get("active")
        
        if not user_id:
            return jsonify({"error": "user_id requerido"}), 400
        
        try:
            update_user(user_id, name, email, role_id, active)
            return jsonify({"message": "Usuario actualizado"}), 200
        except ValueError as e:
            return jsonify({"error": str(e)}), 404
        except Exception as e:
            app.logger.exception("Error actualizando usuario")
            return jsonify({"error": str(e)}), 500
    if request.method == "DELETE":
        data = request.json
        user_id = data.get("user_id")
        
        if not user_id:
            return jsonify({"error": "user_id requerido"}), 400
        
        try:
            update_user(user_id, active="Inactivo")
            return jsonify({"message": "Usuario desactivado"}), 200
        except ValueError as e:
            return jsonify({"error": str(e)}), 404
        except Exception as e:
            app.logger.exception("Error desactivando usuario")
            return jsonify({"error": str(e)}), 500

@app.route("/api/roles", methods=["GET"])
def get_roles_endpoint():
    try:
        roles = get_roles()
        return jsonify(roles), 200
    except Exception as e:
        app.logger.exception("Error obteniendo roles")
        return jsonify({"error": str(e)}), 500

@app.route("/api/users/change_password", methods=["POST"])
def change_password():
    data = request.json
    user_id = data.get("user_id")
    new_password = data.get("new_password")
    
    if not user_id or not new_password:
        return jsonify({"error": "user_id y new_password requeridos"}), 400
    
    try:
        change_user_password(user_id, new_password)
        return jsonify({"message": "Contraseña cambiada"}), 200
    except ValueError as e:
        return jsonify({"error": str(e)}), 404
    except Exception as e:
        app.logger.exception("Error cambiando contraseña")
        return jsonify({"error": str(e)}), 500

# CRUD Lotes
@app.route("/api/lots", methods=["GET", "POST", "DELETE"])
def manage_lots():
    if request.method == "GET":
        # Obtener parámetros de filtro
        user_id = request.args.get("user_id")
        search = request.args.get("search")
        producer = request.args.get("producer")
        species = request.args.get("species")
        variety = request.args.get("variety")
        category = request.args.get("category")
        reception_from = request.args.get("reception_from")
        reception_to = request.args.get("reception_to")
        
        try:
            lots = get_lots(
                user_id=user_id,
                search=search,
                producer=producer,
                species=species,
                variety=variety,
                category=category,
                reception_from=reception_from,
                reception_to=reception_to
            )
            return jsonify(lots), 200
        except Exception as e:
            app.logger.exception("Error obteniendo lotes")
            return jsonify({"error": str(e)}), 500
    
    if request.method == "POST":
        data = request.json
        analyst = data.get("created_by", "system")
        
        try:
            lot_id = insert_lot(
                data["producer"],
                data["species"],
                data["variety"],
                data["category"],
                data["reception"],
                analyst
            )
            return jsonify({"message": "Lote creado", "lot_id": lot_id}), 201
        except Exception as e:
            app.logger.exception("Error creando lote")
            return jsonify({"error": str(e)}), 500
    
    if request.method == "DELETE":
        data = request.json
        lot_id = data.get("lot_id")
        user_id = data.get("user_id")
        if not lot_id or not user_id:
            return jsonify({"error": "lot_id y user_id requeridos"}), 400
        
        try:
            delete_lot(lot_id, user_id)
            return jsonify({"message": "Lote eliminado"}), 200
        except ValueError as e:
            return jsonify({"error": str(e)}), 403
        except Exception as e:
            app.logger.exception("Error eliminando lote")
            return jsonify({"error": str(e)}), 500

# CRUD Muestras
@app.route("/api/samples", methods=["GET", "POST", "DELETE"])
def manage_samples():
    if request.method == "GET":
        # Obtener parámetros de filtro
        user_id = request.args.get("user_id")
        search = request.args.get("search")
        lot_name = request.args.get("lot_name")
        sample_date_from = request.args.get("sample_date_from")
        sample_date_to = request.args.get("sample_date_to")
        analyst = request.args.get("analyst")
        
        try:
            samples = get_samples(
                user_id=user_id,
                search=search,
                lot_name=lot_name,
                sample_date_from=sample_date_from,
                sample_date_to=sample_date_to,
                analyst=analyst
            )
            return jsonify(samples), 200
        except Exception as e:
            app.logger.exception("Error obteniendo muestras")
            return jsonify({"error": str(e)}), 500
    if request.method == "POST":
        data = request.json
        lot_id = data.get("lot_id")
        sample_date = data.get("sample_date")
        analyst = data.get("analyst")
        observations = data.get("observations")
        
        if not lot_id or not sample_date or not analyst:
            return jsonify({"error": "lot_id, sample_date y analyst requeridos"}), 400
        
        try:
            sample_id = insert_sample(lot_id, sample_date, analyst, observations)
            return jsonify({"message": "Muestra creada", "sample_id": sample_id}), 201
        except Exception as e:
            app.logger.exception("Error creando muestra")
            return jsonify({"error": str(e)}), 500
        
    if request.method == "DELETE":
        data = request.json
        sample_id = data.get("sample_id")
        user_id = data.get("user_id")
        
        if not sample_id or not user_id:
            return jsonify({"error": "sample_id y user_id requeridos"}), 400
        try:
            delete_sample(sample_id, user_id)
            return jsonify({"message": "Muestra eliminada"}), 200
        except ValueError as e:
            return jsonify({"error": str(e)}), 403
        except Exception as e:
            app.logger.exception("Error eliminando muestra")
            return jsonify({"error": str(e)}), 500

# Historial de Análisis (Mongo)
@app.route("/api/analysis_history", methods=["GET"])
@jwt_required()
def analysis_history():
    limit = int(request.args.get("limit", 10))
    return jsonify(mongo_get_reports(limit)), 200

# Historial de Reportes (SQL)
@app.route("/api/reports_history", methods=["GET"])
@jwt_required()
def reports_history():
    limit = int(request.args.get("limit", 10))
    return jsonify(get_reports_sql(limit)), 200

# Historial de Análisis (Mongo)
@app.route("/api/analyses", methods=["GET"])  
def list_analyses():
    try:
        limit = int(request.args.get("limit", 10))
        rows = mongo_get_reports(limit=limit)
        return jsonify(rows), 200
    except Exception as e:
        app.logger.exception("Error en list_analyses")
        return jsonify({"error": str(e)}), 500

# Nuevos endpoints para wizard transaccional
@app.route("/api/start_wizard", methods=["POST"])
@jwt_required()
def start_wizard():
    print("[DEBUG] /api/start_wizard called")
    current_user = get_jwt()
    session_id = start_wizard_session(current_user["user_id"])
    return jsonify({"session_id": session_id}), 201

@app.route("/api/save_producer", methods=["POST"])
@jwt_required()
def save_producer():
    print("[DEBUG] /api/save_producer called")
    data = request.json
    session_id = data.get("session_id")
    if not session_id:
        return jsonify({"error": "session_id requerido"}), 400
    
    producer_id = get_or_insert_producer(data["name"], data["phone"], data["address"], data["cod_producer"])
    update_wizard_session(session_id, current_step=2, producer_id=producer_id)
    return jsonify({"producer_id": producer_id}), 201

@app.route("/api/save_lot", methods=["POST"])
@jwt_required()
def save_lot():
    print("[DEBUG] /api/save_lot called")
    data = request.json
    session_id = data.get("session_id")
    if not session_id:
        return jsonify({"error": "session_id requerido"}), 400
    
    session = get_wizard_session(session_id)
    if not session or not session["producer_id"]:
        return jsonify({"error": "Productor no guardado"}), 400
    
    lot_id = insert_lot(session["producer_id"], data["species"], data["variety"], data["category"], data["reception"], data.get("created_by"))
    update_wizard_session(session_id, current_step=3, lot_id=lot_id)
    return jsonify({"lot_id": lot_id}), 201

@app.route("/api/save_sample", methods=["POST"])
@jwt_required()
def save_sample():
    print("[DEBUG] /api/save_sample called")
    data = request.json
    session_id = data.get("session_id")
    if not session_id:
        return jsonify({"error": "session_id requerido"}), 400
    
    session = get_wizard_session(session_id)
    if not session or not session["lot_id"]:
        return jsonify({"error": "Lote no guardado"}), 400
    
    sample_id = insert_sample(session["lot_id"], session["producer_id"], data["sample_date"], data["analyst"], data["observations"])
    update_wizard_session(session_id, current_step=4, sample_id=sample_id)
    return jsonify({"sample_id": sample_id}), 201

@app.route("/api/save_report", methods=["POST"])
@jwt_required()
def save_report():
    print("[DEBUG] /api/save_report called")
    data = request.json
    session_id = data.get("session_id")
    sample_id = data.get("sample_id")
    predicted_class = data.get("predicted_class")
    probability = data.get("probability")
    features = data.get("features", {})
    observations = data.get("observations", "")
    
    if not session_id or not sample_id:
        return jsonify({"error": "session_id y sample_id requeridos"}), 400
    
    session = get_wizard_session(session_id)
    if not session or not session["sample_id"]:
        return jsonify({"error": "Muestra no guardada"}), 400
    
    current_user = get_jwt()
    report_id = insert_report_group(sample_id, current_user["user_id"], predicted_class, probability, features, observations)
    update_wizard_session(session_id, current_step=5, report_id=report_id)
    return jsonify({"report_id": report_id}), 201

@app.route("/api/rollback_wizard", methods=["POST"])
@jwt_required()
def rollback_wizard():
    print("[DEBUG] /api/rollback_wizard called")
    data = request.json
    session_id = data.get("session_id")
    if not session_id:
        return jsonify({"error": "session_id requerido"}), 400
    
    rollback_wizard_session(session_id)
    return jsonify({"message": "Rollback completado"}), 200

@app.route("/api/get_producer", methods=["GET"])
@jwt_required()
def get_producer():
    cod_or_name = request.args.get("cod_or_name")
    if not cod_or_name:
        return jsonify({"error": "cod_or_name requerido"}), 400
    producer = get_producer_by_cod_or_name(cod_or_name)
    if producer:
        return jsonify(producer), 200
    return jsonify({"message": "Productor no encontrado"}), 404




@app.route("/api/evaluate_model", methods=["GET"])
@jwt_required()
def evaluate_model_endpoint():
    try:
        current_user = get_jwt()
        result = evaluate_model(user=current_user.get("name", "Sistema"))
        if "error" in result:
            return jsonify(result), 500
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/api/predict", methods=["POST"])
@jwt_required()
def predict():
    try:
        if 'image' not in request.files:
            return jsonify({"error": "Archivo de imagen requerido"}), 400
        
        f = request.files['image']
        img = preprocess_image(f.read())
        if img is None:
            return jsonify({"error": "Imagen inválida"}), 400
        
        label, prob, preds = analyze_image(img)
        return jsonify({
            "label": label,
            "probability": prob,
            "probability_vector": preds
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    
@app.route("/api/process_image_steps", methods=["POST"])
@jwt_required()
def process_image_steps_endpoint():
    try:
        if 'image' not in request.files:
            return jsonify({"error": "Archivo de imagen requerido"}), 400
        
        f = request.files['image']
        result = predict_process(f.read())  
        if "error" in result:
            return jsonify(result), 500
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/api/predict_step_by_step", methods=["POST"])
@jwt_required()
def predict_step_by_step_endpoint():
    try:
        if 'image' not in request.files:
            return jsonify({"error": "Archivo de imagen requerido"}), 400
        
        f = request.files['image']
        result = predict_step_by_step(f.read())
        if "error" in result:
            return jsonify(result), 500
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/api/producers", methods=["GET", "POST", "PUT", "DELETE"])
def manage_producers():
    if request.method == "GET":
        try:
            search = request.args.get("search")
            return jsonify(get_producers(search=search)), 200
        except Exception as e:
            app.logger.exception("Error obteniendo productores")
            return jsonify({"error": str(e)}), 500

    if request.method == "POST":
        data = request.json
        try:
            producer_id = get_or_insert_producer(data["name"], data["phone"], data["address"], data["cod_producer"])
            return jsonify({"message": "Productor registrado", "producer_id": producer_id}), 201
        except Exception as e:
            app.logger.exception("Error creando productor")
            return jsonify({"error": str(e)}), 500

    if request.method == "PUT":
        data = request.json
        producer_id = data.get("producer_id")
        if not producer_id:
            return jsonify({"error": "producer_id requerido"}), 400
        try:
            update_producer(producer_id, data["name"], data["cod_producer"], data["phone"], data["address"])
            return jsonify({"message": "Productor actualizado"}), 200
        except Exception as e:
            app.logger.exception("Error actualizando productor")
            return jsonify({"error": str(e)}), 500

    if request.method == "DELETE":
        data = request.json
        producer_id = data.get("producer_id")
        if not producer_id:
            return jsonify({"error": "producer_id requerido"}), 400
        try:
            delete_producer(producer_id)
            return jsonify({"message": "Productor eliminado"}), 200
        except ValueError as e:
            return jsonify({"error": str(e)}), 400
        except Exception as e:
            app.logger.exception("Error eliminando productor")
            return jsonify({"error": str(e)}), 500
        
@app.route("/api/count_seeds", methods=["POST"])
def count_seeds():
    """Recibe UNA foto (campo 'file') y devuelve el conteo REAL de semillas."""
    try:
        f = request.files.get("file")
        if f is None:
            return jsonify({"error": "falta el archivo 'file'"}), 400
        params = {}
        for k in ("min_area", "max_area", "thresh"):
            if k in request.form:
                params[k] = int(request.form[k])
        result = count_from_bytes(f.read(), params)
        if "error" in result:
            return jsonify(result), 400
        return jsonify(result), 200
    except Exception as e:
        app.logger.exception("Error en count_seeds")
        return jsonify({"error": str(e)}), 500
    
@app.route("/api/camera/stream")
def camera_stream():
    camera = int(request.args.get("camera", 1))
    return Response(gen_frames(camera),
                    mimetype="multipart/x-mixed-replace; boundary=frame")

@app.route("/api/camera/counts")
def camera_counts():
    return jsonify(latest)

@app.route("/api/camera/snapshot")
def camera_snapshot():
    if latest_jpeg["data"] is None:
        return jsonify({"error": "camara iniciando"}), 503
    return Response(latest_jpeg["data"], mimetype="image/jpeg")

@app.route("/api/analyze_per_seed", methods=["POST"])
def analyze_per_seed_endpoint():
    try:
        files = request.files.getlist("files")
        if not files:
            return jsonify({"error": "No se recibieron imagenes"}), 400
        n_reject = int(request.form.get("n_reject", 0))
        n_total  = int(request.form.get("n_total",  0))
        mode     = request.form.get("mode", "multi")          # <-- AÑADIR
        result   = analyze_per_seed([f.read() for f in files], n_reject, n_total, mode)  # <-- pasar mode
        if "error" in result:
            return jsonify(result), 500
        return jsonify(result), 200
    except Exception as e:
        app.logger.exception("Error en analyze_per_seed")
        return jsonify({"error": str(e)}), 500
    
@app.route("/api/save_iniaf/<analysis_id>", methods=["POST"])
def save_iniaf(analysis_id):
    """Guarda el resumen INIAF en el documento del análisis para que el PDF lo incluya."""
    try:
        payload = request.json or {}
        db = get_db()
        res = db.analyses.update_one(
            {"analysis_id": analysis_id},
            {"$set": {"iniaf": payload}}
        )
        if res.matched_count == 0:
            return jsonify({"error": "Análisis no encontrado"}), 404
        return jsonify({"message": "INIAF guardado"}), 200
    except Exception as e:
        app.logger.exception("Error en save_iniaf")
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000, debug=False, threaded=True)
