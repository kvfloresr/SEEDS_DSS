from infraestructure.api.sql_db import insert_role, insert_user, get_users
from integration import verify_sample_integrity

def main():
    print("=== PRUEBA INCREMENTO 1 ===")

    insert_user("Karen Flores", "karenvfloresr@gmail.com", "Kvfr2509", role_name="Administrador")


    print("\nUsuarios actuales:")
    print(get_users())

    

if __name__ == "__main__":
    main()
