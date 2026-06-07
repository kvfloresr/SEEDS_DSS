import uuid
import pyodbc
from datetime import datetime
from werkzeug.security import generate_password_hash

DB_PATH = "data/seed_dss.db"

def get_sql_connection():
    conn = pyodbc.connect(
        'DRIVER={ODBC Driver 17 for SQL Server};'
        'SERVER=localhost\\SQLEXPRESS;'
        'DATABASE=SeedDSS;'
        'Trusted_Connection=yes;'
    )
    return conn

def insert_role(name, description):
    conn = get_sql_connection()
    cursor = conn.cursor()
    conn.autocommit = False
    try:
        cursor.execute("INSERT INTO Roles (name, description) VALUES (?, ?)", (name, description))
        conn.commit()
        print(f"[OK] Rol '{name}' insertado.")
    except Exception as e:
        conn.rollback()
        raise e
    finally:
        conn.close()

def get_roles():
    conn = get_sql_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT role_id, name, description FROM Roles")
    rows = cursor.fetchall()
    conn.close()
    return [{"role_id": r[0], "name": r[1], "description": r[2]} for r in rows]

def insert_user(name, email, password, role_value="Laboratorista", active="Activo"):
    print(f"[DEBUG insert_user] Recibido: name='{name}', email='{email}', password='{password}', role_value='{role_value}', active='{active}'")
    
    conn = get_sql_connection()
    cursor = conn.cursor()
    conn.autocommit = False
    role_id = None
    try:
        role_id = int(role_value)
        cursor.execute("SELECT role_id FROM Roles WHERE role_id = ?", (role_id,))
        if not cursor.fetchone():
            raise ValueError(f"role_id {role_id} no existe.")
        print(f"[DEBUG insert_user] role_id válido: {role_id}")
    except ValueError:
        cursor.execute("SELECT role_id FROM Roles WHERE name = ?", (role_value,))
        row = cursor.fetchone()
        if not row:
            raise ValueError(f"El rol '{role_value}' no existe en la base de datos.")
        role_id = row[0]
        print(f"[DEBUG insert_user] role_id encontrado por nombre: {role_id}")
    
    cursor.execute("SELECT COUNT(*) FROM Users WHERE email = ?", (email,))
    exists = cursor.fetchone()[0]
    if exists:
        print(f"[INFO insert_user] Usuario con email '{email}' ya existe, no se insertó nuevamente.")
        conn.commit()  
        conn.close()
        return
    
    cursor.execute("""
        INSERT INTO Users (name, email, password, role_id, active)
        VALUES (?, ?, ?, ?, ?)
    """, (name, email, password, role_id, active))
    
    conn.commit()
    conn.close()
    print(f"[OK insert_user] Usuario '{name}' insertado con role_id '{role_id}'.")

def get_users(search=None, role=None, active=None):
    conn = get_sql_connection()
    cursor = conn.cursor()
    
    query = """
        SELECT u.user_id, u.name, u.email, r.name AS role_name, u.active
        FROM Users u
        JOIN Roles r ON u.role_id = r.role_id
        WHERE 1=1
    """
    params = []
    
    if search:
        query += " AND (u.name LIKE ? OR u.email LIKE ? OR r.name LIKE ?)"
        search_param = f"%{search}%"
        params.extend([search_param] * 3)
    
    if role:
        query += " AND r.name = ?"
        params.append(role)
    if active:
        query += " AND u.active = ?"
        params.append(active)
    
    query += " ORDER BY u.name"
    
    cursor.execute(query, params)
    rows = cursor.fetchall()
    conn.close()
    return [{"user_id": str(r[0]), "name": r[1], "email": r[2], "role": r[3], "active": r[4]} for r in rows]

def update_user(user_id, name=None, email=None, role_id=None, active=None):
    conn = get_sql_connection()
    cursor = conn.cursor()
    conn.autocommit = False
    try:
        cursor.execute("SELECT user_id FROM Users WHERE user_id = ?", (user_id,))
        if not cursor.fetchone():
            raise ValueError("Usuario no encontrado.")
        
        updates = []
        params = []
        if name:
            updates.append("name = ?")
            params.append(name)
        if email:
            updates.append("email = ?")
            params.append(email)
        if role_id:
            updates.append("role_id = ?")
            params.append(role_id)
        if active is not None:
            updates.append("active = ?")
            params.append(active)
        
        if not updates:
            conn.commit()
            conn.close()
            return
        
        query = f"UPDATE Users SET {', '.join(updates)} WHERE user_id = ?"
        params.append(user_id)
        
        cursor.execute(query, params)
        conn.commit()
        print(f"[OK] Usuario {user_id} actualizado.")
    except Exception as e:
        conn.rollback()
        raise e
    finally:
        conn.close()

def delete_user(user_id):
    conn = get_sql_connection()
    cursor = conn.cursor()
    conn.autocommit = False
    try:
        cursor.execute("DELETE FROM Users WHERE user_id = ?", (user_id,))
        if cursor.rowcount == 0:
            raise ValueError("Usuario no encontrado.")
        conn.commit()
        print(f"[OK] Usuario {user_id} eliminado.")
    except Exception as e:
        conn.rollback()
        raise e
    finally:
        conn.close()

def change_user_password(user_id, new_password):
    conn = get_sql_connection()
    cursor = conn.cursor()
    conn.autocommit = False
    try:
        cursor.execute("SELECT user_id FROM Users WHERE user_id = ?", (user_id,))
        if not cursor.fetchone():
            raise ValueError("Usuario no encontrado.")
        
        hashed_password = generate_password_hash(new_password)
        cursor.execute("UPDATE Users SET password = ? WHERE user_id = ?", (hashed_password, user_id))
        conn.commit()
        print(f"[OK] Contraseña de usuario {user_id} cambiada.")
    except Exception as e:
        conn.rollback()
        raise e
    finally:
        conn.close()

def insert_producer(name, phone, address, cod_producer):
    conn = get_sql_connection()
    cursor = conn.cursor()
    conn.autocommit = False
    try:
        producer_id = str(uuid.uuid4())
        cursor.execute("""
            INSERT INTO Producers (producer_id, name, phone, address, cod_producer)
            VALUES (?, ?, ?, ?, ?)
        """, (producer_id, name, phone, address, cod_producer))
        conn.commit()
        print(f"[OK] Productor '{name}' insertado con ID '{producer_id}'.")
        return producer_id
    except Exception as e:
        conn.rollback()
        raise e
    finally:
        conn.close()

def get_producer_by_cod_or_name(cod_or_name):
    conn = get_sql_connection()
    cursor = conn.cursor()
    cursor.execute("""
        SELECT producer_id, name, phone, address, cod_producer 
        FROM Producers 
        WHERE cod_producer = ? OR name = ?
    """, (cod_or_name, cod_or_name))
    row = cursor.fetchone()
    conn.close()
    if row:
        return {
            "producer_id": str(row[0]),
            "name": row[1],
            "phone": row[2],
            "address": row[3],
            "cod_producer": row[4]
        }
    return None

def get_or_insert_producer(name, phone, address, cod_producer):
    existing = get_producer_by_cod_or_name(cod_producer)
    if existing:
        return existing["producer_id"]
    # Si no existe, inserta
    conn = get_sql_connection()
    cursor = conn.cursor()
    conn.autocommit = False
    try:
        producer_id = str(uuid.uuid4())
        cursor.execute("""
            INSERT INTO Producers (producer_id, name, phone, address, cod_producer)
            VALUES (?, ?, ?, ?, ?)
        """, (producer_id, name, phone, address, cod_producer))
        conn.commit()
        return producer_id
    except Exception as e:
        conn.rollback()
        raise e
    finally:
        conn.close()

def start_wizard_session(user_id):
    conn = get_sql_connection()
    cursor = conn.cursor()
    conn.autocommit = False
    try:
        session_id = str(uuid.uuid4())
        cursor.execute("""
            INSERT INTO WizardSessions (session_id, user_id)
            VALUES (?, ?)
        """, (session_id, user_id))
        conn.commit()
        print(f"[OK] Sesión wizard '{session_id}' iniciada para user '{user_id}'.")
        return session_id
    except Exception as e:
        conn.rollback()
        raise e
    finally:
        conn.close()

def update_wizard_session(session_id, current_step=None, producer_id=None, lot_id=None, sample_id=None, report_id=None):
    conn = get_sql_connection()
    cursor = conn.cursor()
    conn.autocommit = False
    try:
        updates = []
        params = []
        if current_step is not None:
            updates.append("current_step = ?")
            params.append(current_step)
        if producer_id:
            updates.append("producer_id = ?")
            params.append(producer_id)
        if lot_id:
            updates.append("lot_id = ?")
            params.append(lot_id)
        if sample_id:
            updates.append("sample_id = ?")
            params.append(sample_id)
        if report_id:
            updates.append("report_id = ?")
            params.append(report_id)
        updates.append("updated_at = GETDATE()")
        query = f"UPDATE WizardSessions SET {', '.join(updates)} WHERE session_id = ?"
        params.append(session_id)
        cursor.execute(query, params)
        conn.commit()
        print(f"[OK] Sesión '{session_id}' actualizada.")
    except Exception as e:
        conn.rollback()
        raise e
    finally:
        conn.close()

def get_wizard_session(session_id):
    conn = get_sql_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM WizardSessions WHERE session_id = ?", (session_id,))
    row = cursor.fetchone()
    conn.close()
    if row:
        return {
            "session_id": str(row[0]),
            "user_id": str(row[1]),
            "current_step": row[2],
            "producer_id": str(row[3]) if row[3] else None,
            "lot_id": str(row[4]) if row[4] else None,
            "sample_id": str(row[5]) if row[5] else None,
            "report_id": str(row[6]) if row[6] else None,
            "created_at": row[7],
            "updated_at": row[8]
        }
    return None

def rollback_wizard_session(session_id):
    conn = get_sql_connection()
    cursor = conn.cursor()
    conn.autocommit = False
    try:
        session = get_wizard_session(session_id)
        if session:
            if session["report_id"]:
                cursor.execute("DELETE FROM Reports WHERE report_id = ?", (session["report_id"],))
            if session["sample_id"]:
                cursor.execute("DELETE FROM Samples WHERE sample_id = ?", (session["sample_id"],))
            if session["lot_id"]:
                cursor.execute("DELETE FROM Lots WHERE lot_id = ?", (session["lot_id"],))
            if session["producer_id"]:
                cursor.execute("DELETE FROM Producers WHERE producer_id = ?", (session["producer_id"],))
            cursor.execute("DELETE FROM WizardSessions WHERE session_id = ?", (session_id,))
        conn.commit()
        print(f"[OK] Rollback completado para sesión '{session_id}'.")
    except Exception as e:
        conn.rollback()
        raise e
    finally:
        conn.close()

def insert_lot(producer_id, species, variety, category, reception, created_by=None):
    conn = get_sql_connection()
    cursor = conn.cursor()
    conn.autocommit = False
    try:
        lot_id = str(uuid.uuid4())
        cursor.execute("""
            INSERT INTO Lots (lot_id, producer, species, variety, category, reception, created_by, producer_id)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (lot_id, "", species, variety, category, reception, created_by, producer_id))
        conn.commit()
        print(f"[OK] Lote registrado con producer_id '{producer_id}'.")
        return lot_id
    except Exception as e:
        conn.rollback()
        raise e
    finally:
        conn.close()

def get_lots(user_id=None, search=None, producer=None, species=None, variety=None, category=None, reception_from=None, reception_to=None):
    conn = get_sql_connection()
    cursor = conn.cursor()
    
    query = """
        SELECT 
            l.lot_id, 
            l.producer, 
            l.species, 
            l.variety, 
            l.category, 
            l.reception, 
            u.name AS created_by_name
        FROM Lots l
        LEFT JOIN Users u ON l.created_by = u.user_id
        WHERE 1=1
    """
    params = []
    if user_id:
        query += " AND l.created_by = ?"
        params.append(user_id)
    
    if search:
        query += " AND (l.producer LIKE ? OR l.species LIKE ? OR l.variety LIKE ? OR l.category LIKE ?)"
        search_param = f"%{search}%"
        params.extend([search_param] * 4)
    
    if producer:
        query += " AND l.producer LIKE ?"
        params.append(f"%{producer}%")
    
    if species:
        query += " AND l.species LIKE ?"
        params.append(f"%{species}%")
    
    if variety:
        query += " AND l.variety LIKE ?"
        params.append(f"%{variety}%")
    
    if category:
        query += " AND l.category LIKE ?"
        params.append(f"%{category}%")
        
    if reception_from:
        query += " AND l.reception >= ?"
        params.append(reception_from)
    
    if reception_to:
        query += " AND l.reception <= ?"
        params.append(reception_to)
    
    query += " ORDER BY l.reception DESC"
    
    cursor.execute(query, params)
    rows = cursor.fetchall()
    conn.close()
    
    return [
        {
            "lot_id": str(r[0]),
            "producer": r[1],
            "species": r[2],
            "variety": r[3],
            "category": r[4],
            "reception": r[5],
            "created_by": r[6] or "Desconocido"
        }
        for r in rows
    ]    

def delete_lot(lot_id, user_id):
    conn = get_sql_connection()
    cursor = conn.cursor()
    conn.autocommit = False
    try:
        cursor.execute("SELECT created_by FROM Lots WHERE lot_id = ?", (lot_id,))
        row = cursor.fetchone()
        if not row or str(row[0]) != user_id:
            raise ValueError("Lote no encontrado o no autorizado para eliminar.")
        
        cursor.execute("DELETE FROM Lots WHERE lot_id = ?", (lot_id,))
        conn.commit()
        print(f"[OK] Lote {lot_id} eliminado.")
    except Exception as e:
        conn.rollback()
        raise e
    finally:
        conn.close()

def insert_sample(lot_id, producer_id, sample_date, analyst, observations):
    conn = get_sql_connection()
    cursor = conn.cursor()
    conn.autocommit = False
    try:
        sample_id = str(uuid.uuid4())
        cursor.execute("""
            INSERT INTO Samples (sample_id, lot_id, sample_date, analyst, observations, producer_id)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (sample_id, lot_id, sample_date, analyst, observations, producer_id))
        conn.commit()
        print(f"[OK] Muestra registrada con producer_id '{producer_id}'.")
        return sample_id
    except Exception as e:
        conn.rollback()
        raise e
    finally:
        conn.close()

def get_samples(user_id=None, search=None, lot_name=None, sample_date_from=None, sample_date_to=None, analyst=None):
    conn = get_sql_connection()
    cursor = conn.cursor()
    
    query = """
        SELECT 
            s.sample_id,
            s.sample_date,
            s.analyst,
            s.observations,
            CONCAT(l.producer, ' - ', l.variety) AS lot_name,
            l.created_by
        FROM Samples s
        LEFT JOIN Lots l ON s.lot_id = l.lot_id
        WHERE 1=1
    """
    params = []
    
    if user_id:
        query += " AND l.created_by = ?"
        params.append(user_id)
    if search:
        query += " AND (s.analyst LIKE ? OR s.observations LIKE ? OR l.producer LIKE ? OR l.variety LIKE ?)"
        search_param = f"%{search}%"
        params.extend([search_param] * 4)
    
    if lot_name:
        query += " AND CONCAT(l.producer, ' - ', l.variety) LIKE ?"
        params.append(f"%{lot_name}%")
    
    if sample_date_from:
        query += " AND s.sample_date >= ?"
        params.append(sample_date_from)
    
    if sample_date_to:
        query += " AND s.sample_date <= ?"
        params.append(sample_date_to)
    
    if analyst:
        query += " AND s.analyst LIKE ?"
        params.append(f"%{analyst}%")
    
    query += " ORDER BY s.sample_date DESC"
    
    cursor.execute(query, params)
    rows = cursor.fetchall()
    conn.close()
    return [
        {
            "sample_id": str(r[0]),
            "sample_date": r[1].strftime("%Y-%m-%d") if r[1] else "",
            "analyst": r[2],
            "observations": r[3],
            "lot_name": r[4] or "Sin lote",
            "created_by": str(r[5]) if r[5] else None
        }
        for r in rows
    ]

def delete_sample(sample_id, user_id):
    conn = get_sql_connection()
    cursor = conn.cursor()
    conn.autocommit = False
    try:
        cursor.execute("""
            SELECT l.created_by FROM Samples s
            LEFT JOIN Lots l ON s.lot_id = l.lot_id
            WHERE s.sample_id = ?
        """, (sample_id,))
        row = cursor.fetchone()
        if not row or str(row[0]) != user_id:
                        raise ValueError("Muestra no encontrada o no autorizada para eliminar.")
        
        cursor.execute("DELETE FROM Samples WHERE sample_id = ?", (sample_id,))
        conn.commit()
        print(f"[OK] Muestra {sample_id} eliminada.")
    except Exception as e:
        conn.rollback()
        raise e
    finally:
        conn.close()

def insert_report_group(sample_id, generated_by, predicted_class, probability, features, observations):
    conn = get_sql_connection()
    cursor = conn.cursor()
    conn.autocommit = False
    try:
        report_key = str(uuid.uuid4())
        report_id = str(uuid.uuid4())
        generated_at = datetime.now()
        
        morphological_state = features.get("MorphologicalState", "Unknown")
        pureza_fisica = "Buena" if morphological_state == "Good" else "Media"
        defect_percentage = features.get("Damage_ratio", 0) * 100
        
        query = """
            INSERT INTO Reports (
                report_key, report_id, sample_id, generated_at, generated_by,
                predicted_class, probability, pureza_fisica, defect_percentage, observations
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """
        cursor.execute(query, (
            report_key, report_id, sample_id, generated_at, generated_by,
            predicted_class, probability, pureza_fisica, defect_percentage, observations
        ))
        conn.commit()
        print(f"[SQL] Reporte {report_key} guardado correctamente.")
        return report_id
    except Exception as e:
        conn.rollback()
        raise e
    finally:
        conn.close()

def get_reports_sql(limit=10):
    conn = get_sql_connection()
    c = conn.cursor()
    c.execute(f"SELECT TOP {limit} * FROM Reports ORDER BY generated_at DESC")
    rows = [dict(zip([col[0] for col in c.description], row)) for row in c.fetchall()]
    conn.close()
    return rows

def get_producers(search=None):
    conn = get_sql_connection()
    cursor = conn.cursor()
    query = "SELECT producer_id, name, cod_producer, phone, address FROM Producers"
    params = []
    if search:
        query += " WHERE name LIKE ? OR cod_producer LIKE ?"
        params = [f"%{search}%", f"%{search}%"]
    cursor.execute(query, params)
    cols = [c[0] for c in cursor.description]
    rows = [dict(zip(cols, r)) for r in cursor.fetchall()]
    conn.close()
    return rows

def update_producer(producer_id, name, cod_producer, phone, address):
    conn = get_sql_connection()
    cursor = conn.cursor()
    cursor.execute(
        "UPDATE Producers SET name = ?, cod_producer = ?, phone = ?, address = ? WHERE producer_id = ?",
        (name, cod_producer, phone, address, producer_id)
    )
    conn.commit()
    conn.close()


def delete_producer(producer_id):
    conn = get_sql_connection()
    cursor = conn.cursor()
    try:
        cursor.execute("DELETE FROM Producers WHERE producer_id = ?", (producer_id,))
        conn.commit()
    except Exception:
        conn.rollback()
        conn.close()
        raise ValueError("No se puede eliminar: el productor tiene lotes asociados.")
    conn.close()