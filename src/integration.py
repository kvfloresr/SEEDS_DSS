from src.infrastructure.sql_db import get_sql_connection


def verify_sample_integrity(sample_guid):
    conn = get_sql_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT COUNT(*) FROM Samples WHERE sample_id = ?", (sample_guid,))
    sql_exists = cursor.fetchone()[0] > 0
    conn.close()


    return {"SQL": sql_exists}
