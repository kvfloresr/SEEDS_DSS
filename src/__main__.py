import uuid
from flask import Flask, render_template, request, redirect, url_for, session, flash
from .infrastructure.sql_db import insert_user, get_users, get_roles, insert_lot, get_lots, insert_sample, get_samples

app = Flask(__name__)
app.secret_key = "sEEDS"

@app.route("/")
def home():
    return redirect(url_for("login"))

@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        email = request.form["email"]
        users = get_users()
        user = next((u for u in users if u["email"] == email), None)
        if user:
            session["user_id"] = user["user_id"]
            session["name"] = user["name"]
            session["role_name"] = user["role"]
            flash("Inicio de sesión exitoso.", "success")
            return redirect(url_for("dashboard"))
        else:
            flash("Usuario no encontrado.", "danger")
    return render_template("login.html")

@app.route("/dashboard")
def dashboard():
    if "user_id" not in session:
        return redirect(url_for("login"))
    from .infrastructure.sql_db import get_reports_sql
    reports = get_reports_sql()
    return render_template("dashboard.html", name=session["name"], reports=reports)


@app.route("/logout")
def logout():
    session.clear()
    flash("Sesión cerrada.", "info")
    return redirect(url_for("login"))

@app.route("/admin", methods=["GET", "POST"])
def admin_panel():
    if session.get("role_name") != "Administrador":
        flash("Acceso restringido.", "danger")
        return redirect(url_for("dashboard"))
    if request.method == "POST":
        try:
            insert_user(
                request.form["name"],
                request.form["email"],
                request.form["password"],
                request.form["role_id"]
            )
            flash("Usuario creado correctamente.", "success")
        except Exception as e:
            flash(f"Error: {e}", "danger")
    return render_template("admin_panel.html", users=get_users(), roles=get_roles())


@app.route("/lotes", methods=["GET", "POST"])
def lotes():
    if request.method == "POST":
        producer = request.form["producer"]
        species = request.form["species"]
        variety = request.form["variety"]
        category = request.form["category"]
        reception = request.form["reception"]

        created_by = session.get("user_id")

        if not created_by:
            created_by = str(uuid.uuid4())

        try:
            insert_lot(producer, species, variety, category, reception, created_by)
            flash("Lote registrado correctamente.", "success")
        except Exception as e:
            flash(f"Error al registrar lote: {e}", "danger")

        return redirect(url_for("lotes"))

    lots = get_lots()
    return render_template("lotes.html", lots=lots)


@app.route("/muestras", methods=["GET", "POST"])
def muestras():
    if request.method == "POST":
        lot_id = request.form["lot_id"]
        sample_date = request.form["sample_date"]
        analyst = request.form["analyst"]
        observations = request.form["observations"]

        try:
            insert_sample(lot_id, sample_date, analyst, observations)
            flash("Muestra registrada correctamente.", "success")
        except Exception as e:
            flash(f"Error al registrar muestra: {e}", "danger")

        return redirect(url_for("muestras"))

    lots = get_lots()
    samples = get_samples()
    return render_template("samples.html", lots=lots, samples=samples)


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)

