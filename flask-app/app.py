import os
import io
import pickle
import random
import sqlite3
import string
import threading
import uuid
from functools import wraps

import cv2
import face_recognition
import numpy as np
from flask import (
    Flask, abort, jsonify, redirect, render_template,
    request, send_file, send_from_directory, session, url_for, flash
)
from PIL import Image, ImageDraw, ImageFont
from werkzeug.security import check_password_hash, generate_password_hash
from werkzeug.utils import secure_filename

app = Flask(__name__)
app.secret_key = os.environ.get("SESSION_SECRET", "dev-secret-change-me")

BASE_DIR      = os.path.dirname(os.path.abspath(__file__))
UPLOAD_FOLDER = os.path.join(BASE_DIR, "uploads")
DB_PATH       = os.path.join(BASE_DIR, "faces.db")

ALLOWED_IMAGE_EXT = {"png", "jpg", "jpeg", "gif", "webp"}
ALLOWED_VIDEO_EXT = {"mp4", "mov"}
ALLOWED_EXTENSIONS = ALLOWED_IMAGE_EXT | ALLOWED_VIDEO_EXT

FONT_PATH = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# ─── Background job tracker ────────────────────────────────────────────────────
_jobs: dict = {}
_jobs_lock = threading.Lock()


# ══════════════════════════════════════════════════════════════════════════════
# DATABASE
# ══════════════════════════════════════════════════════════════════════════════

def get_db():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def _add_column_if_missing(conn, table, column, definition):
    try:
        conn.execute(f"ALTER TABLE {table} ADD COLUMN {column} {definition}")
    except sqlite3.OperationalError:
        pass


def init_db():
    with get_db() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS admins (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT NOT NULL UNIQUE,
                password_hash TEXT NOT NULL
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
            CREATE TABLE IF NOT EXISTS events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                description TEXT DEFAULT '',
                code TEXT NOT NULL UNIQUE,
                album_min INTEGER DEFAULT 0,
                album_max INTEGER DEFAULT 0,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS faces (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                photo_filename TEXT NOT NULL,
                encoding BLOB NOT NULL,
                event_id INTEGER REFERENCES events(id),
                media_type TEXT DEFAULT 'photo',
                frame_time REAL DEFAULT 0
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS album_selections (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL,
                event_id INTEGER NOT NULL,
                photo_filename TEXT NOT NULL,
                selected_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(user_id, event_id, photo_filename)
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS watermark_settings (
                id INTEGER PRIMARY KEY,
                enabled INTEGER DEFAULT 0,
                text TEXT DEFAULT 'Bombay Studio',
                font_size INTEGER DEFAULT 48,
                opacity INTEGER DEFAULT 60,
                position TEXT DEFAULT 'bottom-right',
                color TEXT DEFAULT '#ffffff'
            )
        """)
        # Migrations for existing DBs
        for col, defn in [
            ("event_id",   "INTEGER REFERENCES events(id)"),
            ("media_type", "TEXT DEFAULT 'photo'"),
            ("frame_time", "REAL DEFAULT 0"),
        ]:
            _add_column_if_missing(conn, "faces", col, defn)
        for col, defn in [
            ("album_min", "INTEGER DEFAULT 0"),
            ("album_max", "INTEGER DEFAULT 0"),
        ]:
            _add_column_if_missing(conn, "events", col, defn)

        # Default admin
        if not conn.execute("SELECT id FROM admins LIMIT 1").fetchone():
            conn.execute(
                "INSERT INTO admins (username, password_hash) VALUES (?, ?)",
                ("admin", generate_password_hash("admin123"))
            )
        # Default watermark row
        if not conn.execute("SELECT id FROM watermark_settings LIMIT 1").fetchone():
            conn.execute("INSERT INTO watermark_settings (id) VALUES (1)")
        conn.commit()


def get_watermark():
    with get_db() as conn:
        row = conn.execute("SELECT * FROM watermark_settings WHERE id=1").fetchone()
    return dict(row) if row else {}


# ══════════════════════════════════════════════════════════════════════════════
# WATERMARK
# ══════════════════════════════════════════════════════════════════════════════

def _get_font(size: int):
    try:
        return ImageFont.truetype(FONT_PATH, size)
    except Exception:
        return ImageFont.load_default()


def _resize_to_1080p(img: Image.Image) -> Image.Image:
    """Fit image into an exact 1920×1080 canvas (letterboxed, black bars).
    Every output is exactly 1920×1080 so the watermark always lands in the
    same spot regardless of the original photo's aspect ratio or resolution.
    """
    TARGET_W, TARGET_H = 1920, 1080
    w, h = img.size
    ratio = min(TARGET_W / w, TARGET_H / h)
    new_w = int(w * ratio)
    new_h = int(h * ratio)
    resized = img.resize((new_w, new_h), Image.LANCZOS)
    canvas = Image.new("RGB", (TARGET_W, TARGET_H), (0, 0, 0))
    offset_x = (TARGET_W - new_w) // 2
    offset_y = (TARGET_H - new_h) // 2
    canvas.paste(resized, (offset_x, offset_y))
    return canvas


def apply_watermark(pil_img: Image.Image, wm: dict) -> Image.Image:
    """Overlay text watermark on a 1080p-sized PIL image and return a new RGB image.
    Uses outlined text (contrasting stroke around each letter) so the chosen
    colour is always readable on any photo background.
    """
    img = pil_img.convert("RGBA")
    w, h = img.size

    text      = (wm.get("text") or "Bombay Studio").strip() or "Bombay Studio"
    fsize     = max(10, min(200, int(wm.get("font_size", 48))))
    opacity   = max(0, min(100, int(wm.get("opacity", 60))))
    position  = wm.get("position", "bottom-right")
    color_hex = (wm.get("color") or "#ffffff").lstrip("#").ljust(6, "0")

    r = int(color_hex[0:2], 16)
    g = int(color_hex[2:4], 16)
    b = int(color_hex[4:6], 16)
    a = int(255 * opacity / 100)

    overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
    draw    = ImageDraw.Draw(overlay)
    font    = _get_font(fsize)

    bbox = draw.textbbox((0, 0), text, font=font)
    tw = bbox[2] - bbox[0]
    th = bbox[3] - bbox[1]
    margin = max(16, fsize // 2)

    pos_map = {
        "top-left":      (margin, margin),
        "top-center":    ((w - tw) // 2, margin),
        "top-right":     (w - tw - margin, margin),
        "center-left":   (margin, (h - th) // 2),
        "center":        ((w - tw) // 2, (h - th) // 2),
        "center-right":  (w - tw - margin, (h - th) // 2),
        "bottom-left":   (margin, h - th - margin),
        "bottom-center": ((w - tw) // 2, h - th - margin),
        "bottom-right":  (w - tw - margin, h - th - margin),
    }
    x, y = pos_map.get(position, pos_map["bottom-right"])

    # ── Contrasting outline so text is legible on ANY background ──────────
    # Luminance of chosen colour → pick opposite for the outline stroke
    lum = 0.299 * r + 0.587 * g + 0.114 * b
    if lum < 128:
        # Dark text → white outline
        oc = (255, 255, 255, a)
    else:
        # Light text → dark outline
        oc = (0, 0, 0, a)

    stroke_r = max(1, fsize // 20)   # outline thickness scales with font
    for dx in range(-stroke_r, stroke_r + 1):
        for dy in range(-stroke_r, stroke_r + 1):
            if dx == 0 and dy == 0:
                continue
            draw.text((x + dx, y + dy), text, font=font, fill=oc)

    # ── Main text in user's chosen colour ─────────────────────────────────
    draw.text((x, y), text, font=font, fill=(r, g, b, a))

    return Image.alpha_composite(img, overlay).convert("RGB")


def serve_image_with_watermark(filepath: str, wm: dict):
    """Resize to 1080p, apply watermark if enabled, return no-cache response."""
    img = Image.open(filepath).convert("RGB")
    # Always normalise to 1080p for consistent watermark size and faster delivery
    img = _resize_to_1080p(img)
    if wm.get("enabled"):
        img = apply_watermark(img, wm)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=85)   # compressed for fast delivery
    buf.seek(0)
    resp = send_file(buf, mimetype="image/jpeg")
    # No-cache: prevent browsers from reusing admin's (unwatermarked) copies
    resp.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
    resp.headers["Pragma"]        = "no-cache"
    resp.headers["Expires"]       = "0"
    return resp


# ══════════════════════════════════════════════════════════════════════════════
# AUTH DECORATORS
# ══════════════════════════════════════════════════════════════════════════════

def login_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        if not session.get("user_id"):
            flash("Please log in first.", "warning")
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


# ══════════════════════════════════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS

def is_video(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_VIDEO_EXT

def load_image_rgb(filepath):
    img = cv2.imread(filepath)
    if img is None:
        return np.array(Image.open(filepath).convert("RGB"))
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

def generate_event_code():
    with get_db() as conn:
        for _ in range(20):
            code = "".join(random.choices(string.digits, k=5))
            if not conn.execute("SELECT id FROM events WHERE code=?", (code,)).fetchone():
                return code
    return str(random.randint(10000, 99999))


# ══════════════════════════════════════════════════════════════════════════════
# FACE INDEXING
# ══════════════════════════════════════════════════════════════════════════════

def _file_already_indexed(filename, event_id, conn):
    return conn.execute(
        "SELECT id FROM faces WHERE photo_filename=? AND event_id=? LIMIT 1",
        (filename, event_id)
    ).fetchone() is not None


def extract_and_store_faces(filename, event_id, force=False):
    filepath = os.path.join(UPLOAD_FOLDER, filename)
    with get_db() as conn:
        if not force and _file_already_indexed(filename, event_id, conn):
            return 0
        conn.execute("DELETE FROM faces WHERE photo_filename=? AND event_id=?", (filename, event_id))
        img = load_image_rgb(filepath)
        encodings = face_recognition.face_encodings(img)
        for enc in encodings:
            conn.execute(
                "INSERT INTO faces (photo_filename, encoding, event_id, media_type, frame_time) VALUES (?,?,?,'photo',0)",
                (filename, pickle.dumps(enc), event_id)
            )
        conn.commit()
    return len(encodings)


def extract_faces_from_video(filename, event_id, force=False):
    filepath = os.path.join(UPLOAD_FOLDER, filename)
    with get_db() as conn:
        if not force and _file_already_indexed(filename, event_id, conn):
            return 0
        conn.execute("DELETE FROM faces WHERE photo_filename=? AND event_id=?", (filename, event_id))
        conn.commit()

    cap = cv2.VideoCapture(filepath)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    frame_interval = max(1, int(fps))
    frame_idx = total_faces = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx % frame_interval == 0:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            encodings = face_recognition.face_encodings(rgb)
            t = round(frame_idx / fps, 2)
            if encodings:
                with get_db() as conn:
                    for enc in encodings:
                        conn.execute(
                            "INSERT INTO faces (photo_filename, encoding, event_id, media_type, frame_time) VALUES (?,?,?,'video',?)",
                            (filename, pickle.dumps(enc), event_id, t)
                        )
                    conn.commit()
                total_faces += len(encodings)
        frame_idx += 1

    cap.release()
    return total_faces


def _run_indexing_job(job_id, file_list, event_id):
    with app.app_context():
        total = len(file_list)
        done = 0
        errors = []
        for filename, _path in file_list:
            try:
                if is_video(filename):
                    extract_faces_from_video(filename, event_id)
                else:
                    extract_and_store_faces(filename, event_id)
            except Exception as exc:
                errors.append(f"{filename}: {exc}")
            done += 1
            with _jobs_lock:
                _jobs[job_id]["done"] = done
                _jobs[job_id]["errors"] = list(errors)
        with _jobs_lock:
            _jobs[job_id]["status"] = "complete"


# ══════════════════════════════════════════════════════════════════════════════
# FACE SEARCH
# ══════════════════════════════════════════════════════════════════════════════

def _decode_selfie(selfie_bytes):
    nparr = np.frombuffer(selfie_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img is None:
        img = np.array(Image.open(io.BytesIO(selfie_bytes)).convert("RGB"))
    else:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img


def find_matching_photos(selfie_bytes, event_id):
    img = _decode_selfie(selfie_bytes)
    encs = face_recognition.face_encodings(img)
    if not encs:
        return None
    selfie_enc = encs[0]
    BATCH = 2000
    matched = set()
    offset = 0
    with get_db() as conn:
        total = conn.execute("SELECT COUNT(*) FROM faces WHERE event_id=?", (event_id,)).fetchone()[0]
    while offset < total:
        with get_db() as conn:
            rows = conn.execute(
                "SELECT photo_filename, encoding FROM faces WHERE event_id=? LIMIT ? OFFSET ?",
                (event_id, BATCH, offset)
            ).fetchall()
        if not rows:
            break
        filenames  = [r["photo_filename"] for r in rows]
        enc_matrix = np.array([pickle.loads(r["encoding"]) for r in rows], dtype=np.float64)
        distances  = face_recognition.face_distance(enc_matrix, selfie_enc)
        for fname, dist in zip(filenames, distances):
            if dist <= 0.6:
                matched.add(fname)
        offset += BATCH
    return sorted(matched)


# ══════════════════════════════════════════════════════════════════════════════
# USER AUTH
# ══════════════════════════════════════════════════════════════════════════════

@app.route("/register", methods=["GET", "POST"])
def register():
    if session.get("user_id"):
        return redirect(url_for("guest"))
    if request.method == "POST":
        name     = request.form.get("name", "").strip()
        email    = request.form.get("email", "").strip().lower()
        password = request.form.get("password", "")
        confirm  = request.form.get("confirm_password", "")
        if not name or not email or not password:
            flash("All fields are required.", "danger"); return redirect(request.url)
        if password != confirm:
            flash("Passwords do not match.", "danger"); return redirect(request.url)
        if len(password) < 6:
            flash("Minimum 6 characters.", "danger"); return redirect(request.url)
        try:
            with get_db() as conn:
                conn.execute("INSERT INTO users (name, email, password_hash) VALUES (?,?,?)",
                             (name, email, generate_password_hash(password)))
                conn.commit()
            flash("Account created! Please log in.", "success")
            return redirect(url_for("login"))
        except sqlite3.IntegrityError:
            flash("Email already registered.", "danger"); return redirect(request.url)
    return render_template("register.html")


@app.route("/login", methods=["GET", "POST"])
def login():
    if session.get("user_id"):
        return redirect(url_for("guest"))
    if request.method == "POST":
        email    = request.form.get("email", "").strip().lower()
        password = request.form.get("password", "")
        with get_db() as conn:
            user = conn.execute("SELECT * FROM users WHERE email=?", (email,)).fetchone()
        if user and check_password_hash(user["password_hash"], password):
            session.clear()
            session.update(user_id=user["id"], user_name=user["name"], user_email=user["email"])
            flash(f"Welcome back, {user['name']}!", "success")
            return redirect(request.args.get("next") or url_for("guest"))
        flash("Invalid email or password.", "danger")
    return render_template("login.html")


@app.route("/logout")
def logout():
    session.clear()
    flash("Logged out.", "info")
    return redirect(url_for("index"))


# ══════════════════════════════════════════════════════════════════════════════
# ADMIN AUTH
# ══════════════════════════════════════════════════════════════════════════════

@app.route("/admin/login", methods=["GET", "POST"])
def admin_login():
    if session.get("admin_id"):
        return redirect(url_for("admin"))
    if request.method == "POST":
        username = request.form.get("username", "").strip()
        password = request.form.get("password", "")
        with get_db() as conn:
            adm = conn.execute("SELECT * FROM admins WHERE username=?", (username,)).fetchone()
        if adm and check_password_hash(adm["password_hash"], password):
            session.clear()
            session.update(admin_id=adm["id"], admin_username=adm["username"])
            flash("Welcome, Admin!", "success")
            return redirect(url_for("admin"))
        flash("Invalid credentials.", "danger")
    return render_template("admin_login.html")


@app.route("/admin/logout")
def admin_logout():
    session.clear()
    flash("Admin logged out.", "info")
    return redirect(url_for("index"))


# ══════════════════════════════════════════════════════════════════════════════
# ADMIN PANEL
# ══════════════════════════════════════════════════════════════════════════════

@app.route("/admin", methods=["GET", "POST"])
@admin_required
def admin():
    if request.method == "POST":
        event_id = request.form.get("event_id", type=int)
        if not event_id:
            flash("Select an event first.", "danger")
            return redirect(url_for("admin") + "#upload")
        files = request.files.getlist("photos")
        if not files or all(f.filename == "" for f in files):
            flash("No files selected.", "danger")
            return redirect(url_for("admin") + "#upload")

        saved, skipped, bad_ext = [], [], []
        for f in files:
            if not f or not f.filename:
                continue
            if not allowed_file(f.filename):
                bad_ext.append(f.filename); continue
            filename  = secure_filename(f.filename)
            save_path = os.path.join(UPLOAD_FOLDER, filename)
            f.save(save_path)
            with get_db() as conn:
                if _file_already_indexed(filename, event_id, conn):
                    skipped.append(filename); continue
            saved.append((filename, save_path))

        for n in bad_ext:
            flash(f"{n}: unsupported type.", "warning")
        if skipped:
            flash(f"{len(skipped)} already indexed — skipped.", "info")
        if not saved:
            flash("Nothing new to index.", "info")
            return redirect(url_for("admin"))

        job_id = str(uuid.uuid4())
        with _jobs_lock:
            _jobs[job_id] = {"total": len(saved), "done": 0, "errors": [], "status": "running"}
        session["current_job_id"] = job_id
        threading.Thread(target=_run_indexing_job, args=(job_id, saved, event_id), daemon=True).start()
        flash(f"Indexing {len(saved)} file(s) in background.", "info")
        return redirect(url_for("admin"))

    with get_db() as conn:
        events = conn.execute("SELECT * FROM events ORDER BY created_at DESC").fetchall()
        photos = conn.execute(
            "SELECT f.photo_filename, f.event_id, f.media_type, "
            "COUNT(*) as face_count, e.name as event_name, e.code as event_code "
            "FROM faces f LEFT JOIN events e ON f.event_id=e.id "
            "GROUP BY f.photo_filename, f.event_id ORDER BY e.name, f.photo_filename"
        ).fetchall()
        users = conn.execute("SELECT id, name, email, created_at FROM users ORDER BY created_at DESC").fetchall()
        # Album selections grouped
        selections = conn.execute("""
            SELECT a.user_id, a.event_id, a.photo_filename, a.selected_at,
                   u.name as user_name, u.email as user_email,
                   e.name as event_name, e.code as event_code
            FROM album_selections a
            JOIN users u ON a.user_id=u.id
            JOIN events e ON a.event_id=e.id
            ORDER BY e.name, u.name, a.selected_at
        """).fetchall()

    wm = get_watermark()
    job_id = session.get("current_job_id")
    current_job = None
    if job_id:
        with _jobs_lock:
            current_job = dict(_jobs.get(job_id, {}))

    return render_template("admin.html",
        events=events, photos=photos, users=users,
        selections=selections, wm=wm,
        current_job=current_job, job_id=job_id)


@app.route("/admin/status")
@admin_required
def admin_status():
    job_id = request.args.get("job_id") or session.get("current_job_id")
    if not job_id:
        return jsonify({"status": "idle"})
    with _jobs_lock:
        job = dict(_jobs.get(job_id, {}))
    if not job:
        return jsonify({"status": "idle"})
    total   = job.get("total", 1)
    done    = job.get("done", 0)
    return jsonify({
        "status":  job.get("status", "idle"),
        "total":   total, "done": done,
        "percent": int(100 * done / total) if total else 0,
        "errors":  job.get("errors", []),
    })


# ─── Events CRUD ──────────────────────────────────────────────────────────────

@app.route("/admin/events/create", methods=["POST"])
@admin_required
def create_event():
    name      = request.form.get("name", "").strip()
    desc      = request.form.get("description", "").strip()
    album_min = request.form.get("album_min", 0, type=int)
    album_max = request.form.get("album_max", 0, type=int)
    if not name:
        flash("Event name is required.", "danger")
        return redirect(url_for("admin") + "#events")
    code = generate_event_code()
    with get_db() as conn:
        conn.execute(
            "INSERT INTO events (name, description, code, album_min, album_max) VALUES (?,?,?,?,?)",
            (name, desc, code, album_min, album_max)
        )
        conn.commit()
    flash(f"Event '{name}' created! Code: {code}", "success")
    return redirect(url_for("admin") + "#events")


@app.route("/admin/events/<int:event_id>/update", methods=["POST"])
@admin_required
def update_event(event_id):
    album_min = request.form.get("album_min", 0, type=int)
    album_max = request.form.get("album_max", 0, type=int)
    with get_db() as conn:
        conn.execute(
            "UPDATE events SET album_min=?, album_max=? WHERE id=?",
            (album_min, album_max, event_id)
        )
        conn.commit()
    flash("Album limits updated.", "success")
    return redirect(url_for("admin") + "#events")


@app.route("/admin/events/<int:event_id>/delete", methods=["POST"])
@admin_required
def delete_event(event_id):
    with get_db() as conn:
        rows = conn.execute("SELECT DISTINCT photo_filename FROM faces WHERE event_id=?", (event_id,)).fetchall()
        conn.execute("DELETE FROM album_selections WHERE event_id=?", (event_id,))
        conn.execute("DELETE FROM faces WHERE event_id=?", (event_id,))
        conn.execute("DELETE FROM events WHERE id=?", (event_id,))
        conn.commit()
    for row in rows:
        fpath = os.path.join(UPLOAD_FOLDER, row["photo_filename"])
        if os.path.exists(fpath):
            os.remove(fpath)
    flash("Event and all its photos deleted.", "success")
    return redirect(url_for("admin") + "#events")


# ─── Photo management ─────────────────────────────────────────────────────────

@app.route("/admin/delete/<filename>", methods=["POST"])
@admin_required
def delete_photo(filename):
    filename = secure_filename(filename)
    event_id = request.form.get("event_id", type=int)
    with get_db() as conn:
        if event_id:
            conn.execute("DELETE FROM faces WHERE photo_filename=? AND event_id=?", (filename, event_id))
        else:
            conn.execute("DELETE FROM faces WHERE photo_filename=?", (filename,))
        conn.commit()
        still_used = conn.execute("SELECT id FROM faces WHERE photo_filename=? LIMIT 1", (filename,)).fetchone()
    if not still_used:
        fpath = os.path.join(UPLOAD_FOLDER, filename)
        if os.path.exists(fpath):
            os.remove(fpath)
    flash(f"Deleted {filename}.", "success")
    return redirect(url_for("admin") + "#gallery")


@app.route("/admin/bulk-delete", methods=["POST"])
@admin_required
def bulk_delete():
    """Delete multiple photos in one shot. Expects JSON body: [{filename, event_id}, ...]"""
    data = request.get_json(silent=True) or []
    if not data:
        return jsonify({"ok": False, "msg": "No items"}), 400

    deleted = 0
    for item in data:
        filename = secure_filename(item.get("filename", ""))
        event_id = item.get("event_id")
        if not filename:
            continue
        with get_db() as conn:
            if event_id:
                conn.execute("DELETE FROM faces WHERE photo_filename=? AND event_id=?", (filename, event_id))
            else:
                conn.execute("DELETE FROM faces WHERE photo_filename=?", (filename,))
            conn.commit()
            still_used = conn.execute("SELECT id FROM faces WHERE photo_filename=? LIMIT 1", (filename,)).fetchone()
        if not still_used:
            fpath = os.path.join(UPLOAD_FOLDER, filename)
            if os.path.exists(fpath):
                os.remove(fpath)
        deleted += 1

    return jsonify({"ok": True, "deleted": deleted})


# ─── Watermark ────────────────────────────────────────────────────────────────

@app.route("/admin/watermark", methods=["POST"])
@admin_required
def save_watermark():
    enabled   = 1 if request.form.get("enabled") else 0
    text      = request.form.get("text", "Bombay Studio").strip() or "Bombay Studio"
    font_size = request.form.get("font_size", 48, type=int)
    opacity   = request.form.get("opacity", 60, type=int)
    position  = request.form.get("position", "bottom-right")
    color     = request.form.get("color", "#ffffff")

    with get_db() as conn:
        conn.execute(
            "UPDATE watermark_settings SET enabled=?, text=?, font_size=?, opacity=?, position=?, color=? WHERE id=1",
            (enabled, text, font_size, opacity, position, color)
        )
        conn.commit()
    flash("Watermark settings saved.", "success")
    return redirect(url_for("admin") + "#watermark")


@app.route("/admin/watermark/preview")
@admin_required
def watermark_preview():
    """Return a 1920x1080 sample image with the current watermark for live preview.
    The canvas matches the exact dimensions guests see so font sizes are accurate.
    """
    wm = get_watermark()
    W, H = 1920, 1080
    # Build gradient via numpy (fast) — left half bright, right half dark
    import numpy as np
    arr = np.zeros((H, W, 3), dtype=np.uint8)
    t = np.linspace(0, 1, H)[:, None]          # (H, 1)
    # Left half: bright grey fading darker
    left_shade = (220 - t * 160).astype(np.uint8)
    arr[:, :W//2, 0] = left_shade
    arr[:, :W//2, 1] = left_shade
    arr[:, :W//2, 2] = np.clip(left_shade.astype(int) + 10, 0, 255).astype(np.uint8)
    # Right half: dark grey
    right_shade = (80 - t * 50).astype(np.uint8)
    arr[:, W//2:, 0] = right_shade
    arr[:, W//2:, 1] = np.clip(right_shade.astype(int) + 5, 0, 255).astype(np.uint8)
    arr[:, W//2:, 2] = np.clip(right_shade.astype(int) + 15, 0, 255).astype(np.uint8)
    sample = Image.fromarray(arr, "RGB")

    img = apply_watermark(sample, wm)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=80)
    buf.seek(0)
    resp = send_file(buf, mimetype="image/jpeg")
    resp.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
    return resp


@app.route("/admin/change-password", methods=["GET", "POST"])
@admin_required
def admin_change_password():
    if request.method == "POST":
        current = request.form.get("current_password", "")
        new_pw  = request.form.get("new_password", "")
        confirm = request.form.get("confirm_password", "")
        with get_db() as conn:
            adm = conn.execute("SELECT * FROM admins WHERE id=?", (session["admin_id"],)).fetchone()
        if not check_password_hash(adm["password_hash"], current):
            flash("Current password incorrect.", "danger"); return redirect(request.url)
        if new_pw != confirm:
            flash("Passwords do not match.", "danger"); return redirect(request.url)
        if len(new_pw) < 6:
            flash("Minimum 6 characters.", "danger"); return redirect(request.url)
        with get_db() as conn:
            conn.execute("UPDATE admins SET password_hash=? WHERE id=?",
                         (generate_password_hash(new_pw), session["admin_id"]))
            conn.commit()
        flash("Password changed.", "success")
        return redirect(url_for("admin"))
    return render_template("admin_change_password.html")


# ══════════════════════════════════════════════════════════════════════════════
# GUEST ROUTES
# ══════════════════════════════════════════════════════════════════════════════

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/find-my-photos", methods=["GET", "POST"])
@login_required
def guest():
    searched = False
    matched  = []
    error_msg = None
    event    = None

    event_code = (
        request.form.get("event_code", "").strip()
        or session.get("guest_event_code", "")
    )

    if event_code:
        with get_db() as conn:
            event = conn.execute("SELECT * FROM events WHERE code=?", (event_code,)).fetchone()
        if not event:
            error_msg = f"No event found with code '{event_code}'."
            event_code = ""
            session.pop("guest_event_code", None)
        else:
            session["guest_event_code"] = event_code

    # Load current album selection for this user+event
    current_album = []
    if event:
        uid = session.get("user_id")
        with get_db() as conn:
            rows = conn.execute(
                "SELECT photo_filename FROM album_selections WHERE user_id=? AND event_id=?",
                (uid, event["id"])
            ).fetchall()
        current_album = [r["photo_filename"] for r in rows]

    if request.method == "POST":
        action = request.form.get("action", "search")

        if action == "set_event":
            pass  # just loading the event

        elif action == "search":
            if not event:
                error_msg = error_msg or "Enter a valid event code first."
            else:
                selfie = request.files.get("selfie")
                if not selfie or not selfie.filename:
                    error_msg = "Please upload a selfie photo."
                elif selfie.filename.rsplit(".", 1)[-1].lower() not in ALLOWED_IMAGE_EXT:
                    error_msg = "Unsupported format. Use JPG, PNG, or WEBP."
                else:
                    try:
                        result = find_matching_photos(selfie.read(), event["id"])
                        if result is None:
                            error_msg = "No face detected. Try a clearer, well-lit photo."
                        else:
                            matched = result
                            searched = True
                            session["matched_photos"] = matched
                    except Exception as exc:
                        error_msg = f"Processing error: {exc}"

    return render_template("find_my_photos.html",
        name=session.get("user_name", ""),
        email=session.get("user_email", ""),
        event=dict(event) if event else None,
        event_code=event_code,
        searched=searched,
        photos=matched,
        count=len(matched),
        error_msg=error_msg,
        current_album=current_album,
    )


@app.route("/album/submit", methods=["POST"])
@login_required
def album_submit():
    """Guest submits their album photo selection."""
    data = request.get_json(silent=True) or {}
    event_id  = data.get("event_id")
    filenames = data.get("filenames", [])
    uid       = session.get("user_id")

    if not event_id or not uid:
        return jsonify({"ok": False, "msg": "Missing data"}), 400

    with get_db() as conn:
        event = conn.execute("SELECT * FROM events WHERE id=?", (event_id,)).fetchone()
        if not event:
            return jsonify({"ok": False, "msg": "Event not found"}), 404

        album_min = event["album_min"] or 0
        album_max = event["album_max"] or 0

        if album_min and len(filenames) < album_min:
            return jsonify({"ok": False, "msg": f"Select at least {album_min} photos."}), 400
        if album_max and len(filenames) > album_max:
            return jsonify({"ok": False, "msg": f"Maximum {album_max} photos allowed."}), 400

        # Replace previous selection
        conn.execute("DELETE FROM album_selections WHERE user_id=? AND event_id=?", (uid, event_id))
        for fname in filenames:
            fname = secure_filename(fname)
            if fname:
                conn.execute(
                    "INSERT OR IGNORE INTO album_selections (user_id, event_id, photo_filename) VALUES (?,?,?)",
                    (uid, event_id, fname)
                )
        conn.commit()

    return jsonify({"ok": True, "count": len(filenames)})


# ─── Secure file serving ───────────────────────────────────────────────────────

@app.route("/uploads/<filename>")
def uploaded_file(filename):
    filename = secure_filename(filename)
    filepath = os.path.join(UPLOAD_FOLDER, filename)
    if not os.path.exists(filepath):
        abort(404)

    if session.get("admin_id"):
        return send_from_directory(UPLOAD_FOLDER, filename)

    matched = session.get("matched_photos", [])
    # Also allow if it's in the user's album selection
    uid = session.get("user_id")
    in_album = False
    if uid:
        with get_db() as conn:
            in_album = conn.execute(
                "SELECT id FROM album_selections WHERE user_id=? AND photo_filename=? LIMIT 1",
                (uid, filename)
            ).fetchone() is not None

    if filename not in matched and not in_album:
        abort(403)

    # Apply watermark for guests
    wm = get_watermark()
    ext = filename.rsplit(".", 1)[-1].lower()
    if ext in ALLOWED_IMAGE_EXT and ext not in {"gif"}:
        return serve_image_with_watermark(filepath, wm)
    return send_from_directory(UPLOAD_FOLDER, filename)


if __name__ == "__main__":
    init_db()
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
