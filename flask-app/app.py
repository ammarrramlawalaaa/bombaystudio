import os
import sqlite3
import pickle
import io
from functools import wraps

import face_recognition
import cv2
import numpy as np
from PIL import Image
from flask import (
    Flask, request, redirect, url_for,
    render_template, flash, send_from_directory,
    session, abort
)
from werkzeug.utils import secure_filename
from werkzeug.security import generate_password_hash, check_password_hash

app = Flask(__name__)
app.secret_key = os.environ.get("SESSION_SECRET", "dev-secret-change-me")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_FOLDER = os.path.join(BASE_DIR, "uploads")
DB_PATH = os.path.join(BASE_DIR, "faces.db")
ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "gif", "webp"}

os.makedirs(UPLOAD_FOLDER, exist_ok=True)


# ─── Database ────────────────────────────────────────────────────────────────

def get_db():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    with get_db() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS faces (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                photo_filename TEXT NOT NULL,
                encoding BLOB NOT NULL
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                email TEXT NOT NULL UNIQUE,
                password_hash TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS admins (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT NOT NULL UNIQUE,
                password_hash TEXT NOT NULL
            )
        """)
        # Create default admin if none exists
        existing = conn.execute("SELECT id FROM admins LIMIT 1").fetchone()
        if not existing:
            conn.execute(
                "INSERT INTO admins (username, password_hash) VALUES (?, ?)",
                ("admin", generate_password_hash("admin123"))
            )
        conn.commit()


# ─── Auth helpers ─────────────────────────────────────────────────────────────

def login_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        if not session.get("user_id"):
            flash("Please log in to access that page.", "warning")
            return redirect(url_for("login", next=request.path))
        return f(*args, **kwargs)
    return decorated


def admin_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        if not session.get("admin_id"):
            flash("Admin access required.", "danger")
            return redirect(url_for("admin_login"))
        return f(*args, **kwargs)
    return decorated


# ─── Image helpers ────────────────────────────────────────────────────────────

def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def load_image_rgb(filepath):
    img = cv2.imread(filepath)
    if img is None:
        pil_img = Image.open(filepath).convert("RGB")
        return np.array(pil_img)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def extract_and_store_faces(filename):
    filepath = os.path.join(UPLOAD_FOLDER, filename)
    img = load_image_rgb(filepath)
    encodings = face_recognition.face_encodings(img)
    with get_db() as conn:
        # Remove old encodings for this file first (re-upload case)
        conn.execute("DELETE FROM faces WHERE photo_filename = ?", (filename,))
        for enc in encodings:
            conn.execute(
                "INSERT INTO faces (photo_filename, encoding) VALUES (?, ?)",
                (filename, pickle.dumps(enc))
            )
        conn.commit()
    return len(encodings)


def find_matching_photos(selfie_bytes):
    nparr = np.frombuffer(selfie_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img is None:
        pil_img = Image.open(io.BytesIO(selfie_bytes)).convert("RGB")
        img = np.array(pil_img)
    else:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    selfie_encodings = face_recognition.face_encodings(img)
    if not selfie_encodings:
        return None

    selfie_enc = selfie_encodings[0]

    with get_db() as conn:
        rows = conn.execute("SELECT photo_filename, encoding FROM faces").fetchall()

    matched = set()
    for row in rows:
        stored_enc = pickle.loads(row["encoding"])
        if face_recognition.compare_faces([stored_enc], selfie_enc, tolerance=0.6)[0]:
            matched.add(row["photo_filename"])

    return sorted(matched)


# ─── User Auth Routes ─────────────────────────────────────────────────────────

@app.route("/register", methods=["GET", "POST"])
def register():
    if session.get("user_id"):
        return redirect(url_for("guest"))
    if request.method == "POST":
        name = request.form.get("name", "").strip()
        email = request.form.get("email", "").strip().lower()
        password = request.form.get("password", "")
        confirm = request.form.get("confirm_password", "")

        if not name or not email or not password:
            flash("All fields are required.", "danger")
            return redirect(request.url)
        if password != confirm:
            flash("Passwords do not match.", "danger")
            return redirect(request.url)
        if len(password) < 6:
            flash("Password must be at least 6 characters.", "danger")
            return redirect(request.url)

        try:
            with get_db() as conn:
                conn.execute(
                    "INSERT INTO users (name, email, password_hash) VALUES (?, ?, ?)",
                    (name, email, generate_password_hash(password))
                )
                conn.commit()
            flash("Account created! Please log in.", "success")
            return redirect(url_for("login"))
        except sqlite3.IntegrityError:
            flash("An account with that email already exists.", "danger")
            return redirect(request.url)

    return render_template("register.html")


@app.route("/login", methods=["GET", "POST"])
def login():
    if session.get("user_id"):
        return redirect(url_for("guest"))
    if request.method == "POST":
        email = request.form.get("email", "").strip().lower()
        password = request.form.get("password", "")
        with get_db() as conn:
            user = conn.execute(
                "SELECT * FROM users WHERE email = ?", (email,)
            ).fetchone()
        if user and check_password_hash(user["password_hash"], password):
            session.clear()
            session["user_id"] = user["id"]
            session["user_name"] = user["name"]
            session["user_email"] = user["email"]
            flash(f"Welcome back, {user['name']}!", "success")
            next_page = request.args.get("next") or url_for("guest")
            return redirect(next_page)
        flash("Invalid email or password.", "danger")
    return render_template("login.html")


@app.route("/logout")
def logout():
    session.clear()
    flash("You have been logged out.", "info")
    return redirect(url_for("index"))


# ─── Admin Auth Routes ────────────────────────────────────────────────────────

@app.route("/admin/login", methods=["GET", "POST"])
def admin_login():
    if session.get("admin_id"):
        return redirect(url_for("admin"))
    if request.method == "POST":
        username = request.form.get("username", "").strip()
        password = request.form.get("password", "")
        with get_db() as conn:
            admin = conn.execute(
                "SELECT * FROM admins WHERE username = ?", (username,)
            ).fetchone()
        if admin and check_password_hash(admin["password_hash"], password):
            session.clear()
            session["admin_id"] = admin["id"]
            session["admin_username"] = admin["username"]
            flash("Welcome, Admin!", "success")
            return redirect(url_for("admin"))
        flash("Invalid admin credentials.", "danger")
    return render_template("admin_login.html")


@app.route("/admin/logout")
def admin_logout():
    session.clear()
    flash("Admin logged out.", "info")
    return redirect(url_for("index"))


# ─── Admin Panel ──────────────────────────────────────────────────────────────

@app.route("/admin", methods=["GET", "POST"])
@admin_required
def admin():
    if request.method == "POST":
        files = request.files.getlist("photos")
        if not files or all(f.filename == "" for f in files):
            flash("No files selected.", "danger")
            return redirect(request.url)

        uploaded = 0
        total_faces = 0
        errors = []

        for f in files:
            if not f or not f.filename:
                continue
            if not allowed_file(f.filename):
                errors.append(f"{f.filename}: unsupported file type")
                continue
            filename = secure_filename(f.filename)
            save_path = os.path.join(UPLOAD_FOLDER, filename)
            f.save(save_path)
            try:
                face_count = extract_and_store_faces(filename)
                total_faces += face_count
                uploaded += 1
            except Exception as e:
                errors.append(f"{filename}: {str(e)}")

        if uploaded:
            flash(
                f"Uploaded {uploaded} photo(s) — {total_faces} face(s) indexed.",
                "success"
            )
        for err in errors:
            flash(err, "warning")
        return redirect(url_for("admin"))

    with get_db() as conn:
        photos = conn.execute(
            "SELECT photo_filename, COUNT(*) as face_count "
            "FROM faces GROUP BY photo_filename ORDER BY photo_filename"
        ).fetchall()
        user_count = conn.execute("SELECT COUNT(*) FROM users").fetchone()[0]

    return render_template("admin.html", photos=photos, user_count=user_count)


@app.route("/admin/delete/<filename>", methods=["POST"])
@admin_required
def delete_photo(filename):
    filename = secure_filename(filename)
    with get_db() as conn:
        conn.execute("DELETE FROM faces WHERE photo_filename = ?", (filename,))
        conn.commit()
    filepath = os.path.join(UPLOAD_FOLDER, filename)
    if os.path.exists(filepath):
        os.remove(filepath)
    flash(f"Deleted {filename}.", "success")
    return redirect(url_for("admin"))


@app.route("/admin/change-password", methods=["GET", "POST"])
@admin_required
def admin_change_password():
    if request.method == "POST":
        current = request.form.get("current_password", "")
        new_pw = request.form.get("new_password", "")
        confirm = request.form.get("confirm_password", "")
        with get_db() as conn:
            admin = conn.execute(
                "SELECT * FROM admins WHERE id = ?", (session["admin_id"],)
            ).fetchone()
        if not check_password_hash(admin["password_hash"], current):
            flash("Current password is incorrect.", "danger")
            return redirect(request.url)
        if new_pw != confirm:
            flash("New passwords do not match.", "danger")
            return redirect(request.url)
        if len(new_pw) < 6:
            flash("Password must be at least 6 characters.", "danger")
            return redirect(request.url)
        with get_db() as conn:
            conn.execute(
                "UPDATE admins SET password_hash = ? WHERE id = ?",
                (generate_password_hash(new_pw), session["admin_id"])
            )
            conn.commit()
        flash("Password changed successfully.", "success")
        return redirect(url_for("admin"))
    return render_template("admin_change_password.html")


# ─── Guest (User) Routes ──────────────────────────────────────────────────────

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/find-my-photos", methods=["GET", "POST"])
@login_required
def guest():
    if request.method == "POST":
        selfie = request.files.get("selfie")
        if not selfie or not selfie.filename:
            flash("Please upload a selfie.", "danger")
            return redirect(request.url)
        if not allowed_file(selfie.filename):
            flash("Unsupported image format.", "danger")
            return redirect(request.url)

        selfie_bytes = selfie.read()
        try:
            matched = find_matching_photos(selfie_bytes)
        except Exception as e:
            flash(f"Error processing selfie: {str(e)}", "danger")
            return redirect(request.url)

        if matched is None:
            flash("No face detected in your selfie. Try a clearer, well-lit photo.", "warning")
            return redirect(request.url)

        # Store matched photos in session so the upload route can verify access
        session["matched_photos"] = matched

        return render_template(
            "results.html",
            name=session["user_name"],
            photos=matched,
            count=len(matched)
        )

    return render_template("guest.html",
                           name=session.get("user_name", ""),
                           email=session.get("user_email", ""))


# ─── Secure file serving ──────────────────────────────────────────────────────

@app.route("/uploads/<filename>")
def uploaded_file(filename):
    filename = secure_filename(filename)
    # Admins can view all photos
    if session.get("admin_id"):
        return send_from_directory(UPLOAD_FOLDER, filename)
    # Users can only view photos they matched
    matched = session.get("matched_photos", [])
    if filename in matched:
        return send_from_directory(UPLOAD_FOLDER, filename)
    abort(403)


if __name__ == "__main__":
    init_db()
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
