import os
import sqlite3
import pickle
import io

import face_recognition
import cv2
import numpy as np
from PIL import Image
from flask import (
    Flask, request, redirect, url_for,
    render_template, flash, send_from_directory, jsonify
)
from werkzeug.utils import secure_filename

app = Flask(__name__)
app.secret_key = os.environ.get("SESSION_SECRET", "dev-secret-change-me")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_FOLDER = os.path.join(BASE_DIR, "uploads")
DB_PATH = os.path.join(BASE_DIR, "faces.db")

ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "gif", "webp"}

os.makedirs(UPLOAD_FOLDER, exist_ok=True)


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
        conn.commit()


def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def load_image_for_face_recognition(filepath):
    img = cv2.imread(filepath)
    if img is None:
        pil_img = Image.open(filepath).convert("RGB")
        img = np.array(pil_img)
    else:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img


def extract_and_store_faces(filename):
    filepath = os.path.join(UPLOAD_FOLDER, filename)
    img = load_image_for_face_recognition(filepath)
    encodings = face_recognition.face_encodings(img)
    count = 0
    with get_db() as conn:
        for enc in encodings:
            blob = pickle.dumps(enc)
            conn.execute(
                "INSERT INTO faces (photo_filename, encoding) VALUES (?, ?)",
                (filename, blob)
            )
            count += 1
        conn.commit()
    return count


def find_matching_photos(selfie_file_storage):
    img_bytes = selfie_file_storage.read()
    nparr = np.frombuffer(img_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img is None:
        pil_img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        img = np.array(pil_img)
    else:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    selfie_encodings = face_recognition.face_encodings(img)
    if not selfie_encodings:
        return None, []

    selfie_enc = selfie_encodings[0]

    with get_db() as conn:
        rows = conn.execute("SELECT photo_filename, encoding FROM faces").fetchall()

    matched_photos = set()
    for row in rows:
        stored_enc = pickle.loads(row["encoding"])
        results = face_recognition.compare_faces([stored_enc], selfie_enc, tolerance=0.6)
        if results[0]:
            matched_photos.add(row["photo_filename"])

    return selfie_enc, sorted(matched_photos)


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/admin", methods=["GET", "POST"])
def admin():
    if request.method == "POST":
        if "photos" not in request.files:
            flash("No files selected.", "danger")
            return redirect(request.url)

        files = request.files.getlist("photos")
        uploaded = 0
        total_faces = 0
        errors = []

        for f in files:
            if f and f.filename and allowed_file(f.filename):
                filename = secure_filename(f.filename)
                save_path = os.path.join(UPLOAD_FOLDER, filename)
                f.save(save_path)
                try:
                    face_count = extract_and_store_faces(filename)
                    total_faces += face_count
                    uploaded += 1
                except Exception as e:
                    errors.append(f"{filename}: {str(e)}")
            else:
                if f.filename:
                    errors.append(f"{f.filename}: unsupported file type")

        if uploaded:
            flash(
                f"Successfully uploaded {uploaded} photo(s) and indexed {total_faces} face(s).",
                "success"
            )
        for err in errors:
            flash(err, "warning")

        return redirect(url_for("admin"))

    with get_db() as conn:
        photos = conn.execute(
            "SELECT photo_filename, COUNT(*) as face_count FROM faces GROUP BY photo_filename ORDER BY photo_filename"
        ).fetchall()

    return render_template("admin.html", photos=photos)


@app.route("/admin/delete/<filename>", methods=["POST"])
def delete_photo(filename):
    filename = secure_filename(filename)
    with get_db() as conn:
        conn.execute("DELETE FROM faces WHERE photo_filename = ?", (filename,))
        conn.commit()
    filepath = os.path.join(UPLOAD_FOLDER, filename)
    if os.path.exists(filepath):
        os.remove(filepath)
    flash(f"Deleted {filename} and its face data.", "success")
    return redirect(url_for("admin"))


@app.route("/guest", methods=["GET", "POST"])
def guest():
    if request.method == "POST":
        name = request.form.get("name", "").strip()
        email = request.form.get("email", "").strip()
        selfie = request.files.get("selfie")

        if not name or not email or not selfie or not selfie.filename:
            flash("Please fill in all fields and upload a selfie.", "danger")
            return redirect(request.url)

        if not allowed_file(selfie.filename):
            flash("Unsupported image format for selfie.", "danger")
            return redirect(request.url)

        try:
            _, matched = find_matching_photos(selfie)
        except Exception as e:
            flash(f"Error processing selfie: {str(e)}", "danger")
            return redirect(request.url)

        if matched is None:
            flash("No face detected in your selfie. Please try a clearer photo.", "warning")
            return redirect(request.url)

        return render_template(
            "results.html",
            name=name,
            email=email,
            photos=matched,
            count=len(matched)
        )

    return render_template("guest.html")


@app.route("/uploads/<filename>")
def uploaded_file(filename):
    return send_from_directory(UPLOAD_FOLDER, filename)


if __name__ == "__main__":
    init_db()
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
