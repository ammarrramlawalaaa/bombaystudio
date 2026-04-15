import os
import io
import pickle
import random
import sqlite3
import string
import threading
import uuid
import gc
from functools import wraps

import cv2
import face_recognition
import numpy as np
from flask import (
    Flask, abort, jsonify, redirect, render_template,
    request, send_from_directory, send_file, session, url_for, flash
)
from PIL import Image, ImageDraw, ImageFont
from werkzeug.security import check_password_hash, generate_password_hash
from werkzeug.utils import secure_filename

app = Flask(__name__)
app.secret_key = os.environ.get("SESSION_SECRET", "dev-secret-change-me")

BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
UPLOAD_FOLDER = os.path.join(BASE_DIR, "uploads")
DB_PATH     = os.path.join(BASE_DIR, "faces.db")

ALLOWED_IMAGE_EXT = {"png", "jpg", "jpeg", "gif", "webp"}
ALLOWED_VIDEO_EXT = {"mp4", "mov"}
ALLOWED_EXTENSIONS = ALLOWED_IMAGE_EXT | ALLOWED_VIDEO_EXT

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# ─── Background job tracker ───────────────────────────────────────────────────
_jobs: dict = {}          # job_id → {total, done, errors, status}
_jobs_lock = threading.Lock()

# ─── Watermark settings (default values) ───────────────────────────────────────
_watermark_settings: dict = {
    "enabled": True,
    "text": "PREVIEW ONLY",
    "text_size": 60,
    "placement": "diagonal",  # diagonal, center, top_left, bottom_right
    "opacity": 0.3,
}


# ─── Database ─────────────────────────────────────────────────────────────────

def get_db():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def _add_column_if_missing(conn, table, column, definition):
    try:
        conn.execute(f"ALTER TABLE {table} ADD COLUMN {column} {definition}")
    except sqlite3.OperationalError:
        pass  # column already exists


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
        # Create watermark settings table
        conn.execute("""
            CREATE TABLE IF NOT EXISTS watermark_settings (
                id INTEGER PRIMARY KEY CHECK (id = 1),
                enabled BOOLEAN DEFAULT 1,
                text TEXT DEFAULT 'PREVIEW ONLY',
                text_size INTEGER DEFAULT 60,
                placement TEXT DEFAULT 'diagonal',
                opacity REAL DEFAULT 0.3
            )
        """)
        
        # Initialize watermark settings if not exists
        if not conn.execute("SELECT id FROM watermark_settings LIMIT 1").fetchone():
            conn.execute(
                "INSERT INTO watermark_settings (id, enabled, text, text_size, placement, opacity) "
                "VALUES (1, 1, 'PREVIEW ONLY', 60, 'diagonal', 0.3)"
            )
        else:
            # Load into memory
            row = conn.execute("SELECT * FROM watermark_settings WHERE id=1").fetchone()
            if row:
                global _watermark_settings
                _watermark_settings.update({
                    "enabled": bool(row["enabled"]),
                    "text": row["text"],
                    "text_size": row["text_size"],
                    "placement": row["placement"],
                    "opacity": row["opacity"]
                })

        # Default admin
        if not conn.execute("SELECT id FROM admins LIMIT 1").fetchone():
            conn.execute(
                "INSERT INTO admins (username, password_hash) VALUES (?, ?)",
                ("admin", generate_password_hash("admin123"))
            )
        conn.commit()


# ─── Auth decorators ───────────────────────────────────────────────────────────

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


# ─── Helpers ──────────────────────────────────────────────────────────────────

def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def is_video(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_VIDEO_EXT


def _resize_image_if_needed(img, max_dimension=1000):
    """
    Resize image to max_dimension if it exceeds that size in width or height.
    Maintains aspect ratio using OpenCV.
    """
    if img is None:
        return img
    height, width = img.shape[:2]
    if width > max_dimension or height > max_dimension:
        scale = min(max_dimension / width, max_dimension / height)
        new_width = int(width * scale)
        new_height = int(height * scale)
        img = cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_AREA)
    return img


def load_image_rgb(filepath):
    img = cv2.imread(filepath)
    if img is None:
        pil_img = Image.open(filepath).convert("RGB")
        img = np.array(pil_img)
    else:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # Resize if image exceeds 1000px in any dimension
    img = _resize_image_if_needed(img, max_dimension=1000)
    return img


def generate_event_code():
    """Generate a unique 5-digit alphanumeric event code."""
    with get_db() as conn:
        for _ in range(20):
            code = "".join(random.choices(string.digits, k=5))
            exists = conn.execute("SELECT id FROM events WHERE code=?", (code,)).fetchone()
            if not exists:
                return code
    return str(random.randint(10000, 99999))


def generate_watermarked_preview(original_path, output_path, max_width=800):
    """
    Generate a watermarked preview from original high-res photo.
    - Resizes to max_width while maintaining aspect ratio
    - Overlays semi-transparent watermark text based on settings
    - Saves as JPEG to output_path
    Returns True if successful, False otherwise.
    """
    try:
        if not os.path.exists(original_path):
            return False
        
        # Open and resize image
        img = Image.open(original_path)
        if img.mode != "RGB":
            img = img.convert("RGB")
        
        # Resize to max width
        aspect_ratio = img.height / img.width
        new_width = min(max_width, img.width)
        new_height = int(new_width * aspect_ratio)
        img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)
        
        # Check if watermarking is enabled
        if not _watermark_settings.get("enabled", True):
            img.save(output_path, "JPEG", quality=85)
            return True
        
        # Get watermark settings
        watermark_text = _watermark_settings.get("text", "PREVIEW ONLY")
        text_size = _watermark_settings.get("text_size", 60)
        placement = _watermark_settings.get("placement", "diagonal")
        opacity = _watermark_settings.get("opacity", 0.3)
        
        # Create watermark layer
        watermark = Image.new("RGBA", img.size, (255, 255, 255, 0))
        draw = ImageDraw.Draw(watermark)
        
        # Try to use a decent font, fallback to default
        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf", text_size)
        except:
            font = ImageFont.load_default()
        
        # Calculate text size and position
        bbox = draw.textbbox((0, 0), watermark_text, font=font)
        text_width = bbox[2] - bbox[0]
        text_height = bbox[3] - bbox[1]
        
        # Alpha value for watermark
        alpha = int(255 * opacity)
        
        # Place watermark based on setting
        if placement == "diagonal":
            x = (img.width - text_width) // 2
            y = (img.height - text_height) // 2
        elif placement == "center":
            x = (img.width - text_width) // 2
            y = (img.height - text_height) // 2
        elif placement == "top_left":
            x = 10
            y = 10
        elif placement == "bottom_right":
            x = img.width - text_width - 10
            y = img.height - text_height - 10
        else:
            x = (img.width - text_width) // 2
            y = (img.height - text_height) // 2
        
        draw.text((x, y), watermark_text, font=font, fill=(255, 255, 255, alpha))
        
        # Add copyright mark at bottom right corner
        copyright_text = "© Bombay Studio"
        copyright_size = max(12, int(text_size * 0.25))  # Small font, 25% of main watermark
        try:
            copyright_font = ImageFont.truetype("/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf", copyright_size)
        except:
            copyright_font = ImageFont.load_default()
        
        # Calculate copyright text size
        copy_bbox = draw.textbbox((0, 0), copyright_text, font=copyright_font)
        copy_width = copy_bbox[2] - copy_bbox[0]
        copy_height = copy_bbox[3] - copy_bbox[1]
        
        # Place copyright at bottom right with padding
        copy_x = img.width - copy_width - 8
        copy_y = img.height - copy_height - 8
        copyright_alpha = int(255 * (opacity * 0.7))  # Slightly more transparent than main watermark
        
        draw.text((copy_x, copy_y), copyright_text, font=copyright_font, fill=(255, 255, 255, copyright_alpha))
        
        # Composite watermark onto image
        img = Image.alpha_composite(img.convert("RGBA"), watermark).convert("RGB")
        
        # Save watermarked preview
        img.save(output_path, "JPEG", quality=85)
        return True
        
    except Exception as exc:
        print(f"Error generating watermark for {original_path}: {exc}")
        return False


# ─── Face indexing ─────────────────────────────────────────────────────────────

def _file_already_indexed(filename, event_id, conn):
    """Return True if this filename is already in the faces table for the event."""
    row = conn.execute(
        "SELECT id FROM faces WHERE photo_filename=? AND event_id=? LIMIT 1",
        (filename, event_id)
    ).fetchone()
    return row is not None


def extract_and_store_faces(filename, event_id, force=False):
    """Index a photo. Skips if already indexed (unless force=True)."""
    filepath = os.path.join(UPLOAD_FOLDER, filename)
    with get_db() as conn:
        if not force and _file_already_indexed(filename, event_id, conn):
            return 0  # already done — skip
        conn.execute(
            "DELETE FROM faces WHERE photo_filename=? AND event_id=?",
            (filename, event_id)
        )
        img = load_image_rgb(filepath)
        encodings = face_recognition.face_encodings(img)
        for enc in encodings:
            conn.execute(
                "INSERT INTO faces (photo_filename, encoding, event_id, media_type, frame_time) "
                "VALUES (?, ?, ?, 'photo', 0)",
                (filename, pickle.dumps(enc), event_id)
            )
        conn.commit()
    return len(encodings)


def extract_faces_from_video(filename, event_id, force=False):
    """Sample 1 frame per second from a video and index faces."""
    filepath = os.path.join(UPLOAD_FOLDER, filename)
    with get_db() as conn:
        if not force and _file_already_indexed(filename, event_id, conn):
            return 0
        conn.execute(
            "DELETE FROM faces WHERE photo_filename=? AND event_id=?",
            (filename, event_id)
        )
        conn.commit()

    cap = cv2.VideoCapture(filepath)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    frame_interval = max(1, int(fps))
    frame_idx = 0
    total_faces = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx % frame_interval == 0:
            # Resize frame if it exceeds 1000px in any dimension
            frame = _resize_image_if_needed(frame, max_dimension=1000)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            encodings = face_recognition.face_encodings(rgb)
            t = frame_idx / fps
            if encodings:
                with get_db() as conn:
                    for enc in encodings:
                        conn.execute(
                            "INSERT INTO faces (photo_filename, encoding, event_id, media_type, frame_time) "
                            "VALUES (?, ?, ?, 'video', ?)",
                            (filename, pickle.dumps(enc), event_id, round(t, 2))
                        )
                    conn.commit()
                total_faces += len(encodings)
        frame_idx += 1

    cap.release()
    return total_faces


# ─── Background indexing thread ────────────────────────────────────────────────

def _run_indexing_job(job_id: str, file_list: list, event_id: int):
    """file_list: [(filename, filepath), …] — files already saved to disk.
    
    Processes files with:
    - Duplicate prevention: skips already-indexed files
    - Progress tracking: updates job status every file
    - RAM management: calls gc.collect() every 50 photos
    - Error handling: continues on individual file errors
    """
    with app.app_context():
        import psutil
        import os as os_module
        
        total = len(file_list)
        done = 0
        errors = []
        skipped = 0
        processed = 0
        
        # Get process for memory monitoring
        try:
            process = psutil.Process(os_module.getpid())
        except:
            process = None

        for i, (filename, _filepath) in enumerate(file_list, start=1):
            try:
                # Check if file is already indexed (duplicate prevention)
                with get_db() as conn:
                    if _file_already_indexed(filename, event_id, conn):
                        skipped += 1
                        done += 1
                        with _jobs_lock:
                            _jobs[job_id]["done"] = done
                            _jobs[job_id]["skipped"] = skipped
                        continue
                
                # Generate watermarked preview for images
                if not is_video(filename):
                    watermarked_name = filename.rsplit(".", 1)[0] + "_watermarked.jpg"
                    watermarked_path = os.path.join(UPLOAD_FOLDER, watermarked_name)
                    original_path = os.path.join(UPLOAD_FOLDER, filename)
                    generate_watermarked_preview(original_path, watermarked_path, max_width=800)
                
                # Process the file for face indexing
                if is_video(filename):
                    extract_faces_from_video(filename, event_id)
                else:
                    extract_and_store_faces(filename, event_id)
                processed += 1
                
            except Exception as exc:
                errors.append(f"{filename}: {str(exc)[:100]}")

            done += 1
            
            # Update progress
            with _jobs_lock:
                _jobs[job_id]["done"] = done
                _jobs[job_id]["processed"] = processed
                _jobs[job_id]["skipped"] = skipped
                _jobs[job_id]["errors"] = list(errors[-10:])  # Keep last 10 errors
                
                # Add memory stats
                if process:
                    try:
                        mem_info = process.memory_info()
                        _jobs[job_id]["memory_mb"] = mem_info.rss / 1024 / 1024
                    except:
                        pass
            
            # Garbage collection every 50 photos to prevent memory leak
            if i % 50 == 0:
                gc.collect()

        with _jobs_lock:
            _jobs[job_id]["status"] = "complete"
            _jobs[job_id]["done"] = total


# ─── Vectorised face search ────────────────────────────────────────────────────

def _decode_selfie(selfie_bytes: bytes):
    nparr = np.frombuffer(selfie_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img is None:
        img = np.array(Image.open(io.BytesIO(selfie_bytes)).convert("RGB"))
    else:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # Resize if image exceeds 1000px in any dimension
    img = _resize_image_if_needed(img, max_dimension=1000)
    return img


def find_matching_photos(selfie_bytes: bytes, event_id: int):
    """
    Vectorised search: loads all 128-D encodings into a single numpy matrix,
    computes distances with face_recognition.face_distance in one pass.
    Returns sorted list of matching filenames, or None if no face in selfie.
    """
    img = _decode_selfie(selfie_bytes)
    selfie_encs = face_recognition.face_encodings(img)
    if not selfie_encs:
        return None
    selfie_enc = selfie_encs[0]

    # Load encodings in batches to handle massive galleries without OOM
    BATCH = 2000
    matched: set = set()
    offset = 0

    with get_db() as conn:
        total = conn.execute(
            "SELECT COUNT(*) FROM faces WHERE event_id=?", (event_id,)
        ).fetchone()[0]

    while offset < total:
        with get_db() as conn:
            rows = conn.execute(
                "SELECT photo_filename, encoding FROM faces WHERE event_id=? LIMIT ? OFFSET ?",
                (event_id, BATCH, offset)
            ).fetchall()

        if not rows:
            break

        filenames   = [r["photo_filename"] for r in rows]
        enc_matrix  = np.array([pickle.loads(r["encoding"]) for r in rows], dtype=np.float64)

        # face_distance = numpy.linalg.norm under the hood — vectorised
        distances = face_recognition.face_distance(enc_matrix, selfie_enc)
        for fname, dist in zip(filenames, distances):
            if dist <= 0.6:
                matched.add(fname)

        offset += BATCH

    return sorted(matched)


# ════════════════════════════════════════════════════════════════════════════════
# ─── USER AUTH ────────────────────────────────────────────────────────────────
# ════════════════════════════════════════════════════════════════════════════════

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
        email    = request.form.get("email", "").strip().lower()
        password = request.form.get("password", "")
        with get_db() as conn:
            user = conn.execute(
                "SELECT * FROM users WHERE email=?", (email,)
            ).fetchone()
        if user and check_password_hash(user["password_hash"], password):
            session.clear()
            session["user_id"]    = user["id"]
            session["user_name"]  = user["name"]
            session["user_email"] = user["email"]
            flash(f"Welcome back, {user['name']}!", "success")
            return redirect(request.args.get("next") or url_for("guest"))
        flash("Invalid email or password.", "danger")
    return render_template("login.html")


@app.route("/logout")
def logout():
    session.clear()
    flash("You have been logged out.", "info")
    return redirect(url_for("index"))


# ════════════════════════════════════════════════════════════════════════════════
# ─── ADMIN AUTH ───────────────────────────────────────────────────────────────
# ════════════════════════════════════════════════════════════════════════════════

@app.route("/admin/login", methods=["GET", "POST"])
def admin_login():
    if session.get("admin_id"):
        return redirect(url_for("admin"))
    if request.method == "POST":
        username = request.form.get("username", "").strip()
        password = request.form.get("password", "")
        with get_db() as conn:
            adm = conn.execute(
                "SELECT * FROM admins WHERE username=?", (username,)
            ).fetchone()
        if adm and check_password_hash(adm["password_hash"], password):
            session.clear()
            session["admin_id"]       = adm["id"]
            session["admin_username"] = adm["username"]
            flash("Welcome, Admin!", "success")
            return redirect(url_for("admin"))
        flash("Invalid admin credentials.", "danger")
    return render_template("admin_login.html")


@app.route("/admin/logout")
def admin_logout():
    session.clear()
    flash("Admin logged out.", "info")
    return redirect(url_for("index"))


# ════════════════════════════════════════════════════════════════════════════════
# ─── ADMIN PANEL ──────────────────────────────────────────────────────────────
# ════════════════════════════════════════════════════════════════════════════════

@app.route("/admin", methods=["GET", "POST"])
@admin_required
def admin():
    if request.method == "POST":
        event_id = request.form.get("event_id", type=int)
        if not event_id:
            flash("Please select an event before uploading.", "danger")
            return redirect(url_for("admin") + "#upload")

        files = request.files.getlist("photos")
        if not files or all(f.filename == "" for f in files):
            flash("No files selected.", "danger")
            return redirect(url_for("admin") + "#upload")

        saved = []
        skipped = []
        bad_ext = []

        for f in files:
            if not f or not f.filename:
                continue
            if not allowed_file(f.filename):
                bad_ext.append(f.filename)
                continue
            filename  = secure_filename(f.filename)
            save_path = os.path.join(UPLOAD_FOLDER, filename)
            f.save(save_path)
            # Check if already indexed for this event
            with get_db() as conn:
                if _file_already_indexed(filename, event_id, conn):
                    skipped.append(filename)
                    continue
            saved.append((filename, save_path))

        for name in bad_ext:
            flash(f"{name}: unsupported type (use jpg/png/mp4/mov).", "warning")
        if skipped:
            flash(f"{len(skipped)} file(s) already indexed — skipped.", "info")

        if not saved:
            flash("Nothing new to index.", "info")
            return redirect(url_for("admin"))

        # Start background thread
        job_id = str(uuid.uuid4())
        with _jobs_lock:
            _jobs[job_id] = {
                "total": len(saved),
                "done": 0,
                "errors": [],
                "status": "running",
                "event_id": event_id,
            }
        session["current_job_id"] = job_id

        t = threading.Thread(
            target=_run_indexing_job,
            args=(job_id, saved, event_id),
            daemon=True
        )
        t.start()

        flash(
            f"Indexing {len(saved)} file(s) in the background. "
            "Watch the progress bar below.",
            "info"
        )
        return redirect(url_for("admin"))

    with get_db() as conn:
        events = conn.execute(
            "SELECT * FROM events ORDER BY created_at DESC"
        ).fetchall()
        photos = conn.execute(
            "SELECT f.photo_filename, f.event_id, f.media_type, "
            "COUNT(*) as face_count, e.name as event_name, e.code as event_code "
            "FROM faces f LEFT JOIN events e ON f.event_id=e.id "
            "GROUP BY f.photo_filename, f.event_id "
            "ORDER BY e.name, f.photo_filename"
        ).fetchall()
        users = conn.execute(
            "SELECT id, name, email, created_at FROM users ORDER BY created_at DESC"
        ).fetchall()

    job_id = session.get("current_job_id")
    current_job = None
    if job_id:
        with _jobs_lock:
            current_job = dict(_jobs.get(job_id, {}))

    return render_template(
        "admin.html",
        events=events,
        photos=photos,
        users=users,
        current_job=current_job,
        job_id=job_id,
    )


@app.route("/admin/status")
@admin_required
def admin_status():
    job_id = request.args.get("job_id") or session.get("current_job_id")
    if not job_id:
        return jsonify({"status": "idle", "percent": 0})
    with _jobs_lock:
        job = dict(_jobs.get(job_id, {}))
    if not job:
        return jsonify({"status": "idle", "percent": 0})
    
    total   = job.get("total", 1)
    done    = job.get("done", 0)
    processed = job.get("processed", 0)
    skipped = job.get("skipped", 0)
    percent = int(100 * done / total) if total else 0
    
    return jsonify({
        "status":    job.get("status", "idle"),
        "total":     total,
        "done":      done,
        "processed": processed,
        "skipped":   skipped,
        "percent":   percent,
        "memory_mb": job.get("memory_mb", 0),
        "errors":    job.get("errors", []),
    })


# ─── Events CRUD ──────────────────────────────────────────────────────────────

@app.route("/admin/events/create", methods=["POST"])
@admin_required
def create_event():
    name = request.form.get("name", "").strip()
    desc = request.form.get("description", "").strip()
    if not name:
        flash("Event name is required.", "danger")
        return redirect(url_for("admin") + "#events")
    code = generate_event_code()
    with get_db() as conn:
        conn.execute(
            "INSERT INTO events (name, description, code) VALUES (?, ?, ?)",
            (name, desc, code)
        )
        conn.commit()
    flash(f"Event '{name}' created! Code: {code}", "success")
    return redirect(url_for("admin") + "#events")


@app.route("/admin/events/<int:event_id>/delete", methods=["POST"])
@admin_required
def delete_event(event_id):
    with get_db() as conn:
        # Get all filenames for this event to delete files
        rows = conn.execute(
            "SELECT DISTINCT photo_filename FROM faces WHERE event_id=?", (event_id,)
        ).fetchall()
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
            conn.execute(
                "DELETE FROM faces WHERE photo_filename=? AND event_id=?",
                (filename, event_id)
            )
        else:
            conn.execute("DELETE FROM faces WHERE photo_filename=?", (filename,))
        conn.commit()
    # Only delete file if no longer referenced by any event
    with get_db() as conn:
        still_used = conn.execute(
            "SELECT id FROM faces WHERE photo_filename=? LIMIT 1", (filename,)
        ).fetchone()
    if not still_used:
        fpath = os.path.join(UPLOAD_FOLDER, filename)
        if os.path.exists(fpath):
            os.remove(fpath)
    flash(f"Deleted {filename}.", "success")
    return redirect(url_for("admin") + "#gallery")


@app.route("/admin/change-password", methods=["GET", "POST"])
@admin_required
def admin_change_password():
    if request.method == "POST":
        current = request.form.get("current_password", "")
        new_pw  = request.form.get("new_password", "")
        confirm = request.form.get("confirm_password", "")
        with get_db() as conn:
            adm = conn.execute(
                "SELECT * FROM admins WHERE id=?", (session["admin_id"],)
            ).fetchone()
        if not check_password_hash(adm["password_hash"], current):
            flash("Current password is incorrect.", "danger")
            return redirect(request.url)
        if new_pw != confirm:
            flash("Passwords do not match.", "danger")
            return redirect(request.url)
        if len(new_pw) < 6:
            flash("Minimum 6 characters.", "danger")
            return redirect(request.url)
        with get_db() as conn:
            conn.execute(
                "UPDATE admins SET password_hash=? WHERE id=?",
                (generate_password_hash(new_pw), session["admin_id"])
            )
            conn.commit()
        flash("Password changed.", "success")
        return redirect(url_for("admin"))
    return render_template("admin_change_password.html")


# ════════════════════════════════════════════════════════════════════════════════
# ─── GUEST ROUTES ─────────────────────────────────────────────────────────────
# ════════════════════════════════════════════════════════════════════════════════

@app.route("/validate-selfie", methods=["POST"])
@login_required
def validate_selfie():
    """
    Validate selfie image for face quality.
    Expects JSON with 'image_data' (base64 or raw bytes).
    Returns: {valid: bool, message: str, preview: base64}
    """
    try:
        data = request.get_json() or {}
        image_data = data.get("image_data", "")

        if not image_data:
            return jsonify({"valid": False, "message": "No image provided"}), 400

        # Handle base64 data URL
        if image_data.startswith("data:image"):
            image_data = image_data.split("base64,")[-1]

        # Decode base64 to image
        try:
            img_bytes = np.frombuffer(__import__("base64").b64decode(image_data), np.uint8)
            img = cv2.imdecode(img_bytes, cv2.IMREAD_COLOR)
        except Exception:
            return jsonify({"valid": False, "message": "Invalid image format"}), 400

        if img is None:
            return jsonify({"valid": False, "message": "Could not read image"}), 400

        # Convert to RGB
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Detect faces
        face_locations = face_recognition.face_locations(img_rgb)
        face_encodings = face_recognition.face_encodings(img_rgb, face_locations)

        # Validation logic
        if not face_locations:
            return jsonify({
                "valid": False,
                "message": "No face detected. Please ensure your face is clearly visible and well-lit."
            }), 200

        if len(face_locations) > 1:
            return jsonify({
                "valid": False,
                "message": f"Multiple faces detected ({len(face_locations)}). Please take a selfie with only your face."
            }), 200

        # Check face size (at least 50x50 pixels to be useful)
        top, right, bottom, left = face_locations[0]
        face_height = bottom - top
        face_width = right - left
        min_face_size = 50

        if face_height < min_face_size or face_width < min_face_size:
            return jsonify({
                "valid": False,
                "message": "Face is too small. Please get closer to the camera."
            }), 200

        # All checks passed
        # Resize to 1000px max for preview/storage (consistent with optimization)
        img_resized = _resize_image_if_needed(img_rgb, max_dimension=1000)

        # Convert to JPEG for preview
        success, buffer = cv2.imencode('.jpg', cv2.cvtColor(img_resized, cv2.COLOR_RGB2BGR))
        if success:
            preview_b64 = __import__("base64").b64encode(buffer).decode("utf-8")
            preview_data_url = f"data:image/jpeg;base64,{preview_b64}"
        else:
            preview_data_url = ""

        return jsonify({
            "valid": True,
            "message": "✓ Face looks good! Ready to search.",
            "preview": preview_data_url,
            "face_count": len(face_locations),
            "face_size": {"width": face_width, "height": face_height}
        }), 200

    except Exception as exc:
        return jsonify({
            "valid": False,
            "message": f"Error validating image: {exc}"
        }), 500


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/find-my-photos", methods=["GET", "POST"])
@login_required
def guest():
    searched  = False
    matched   = []
    error_msg = None
    event     = None

    # Resolve event from session or form
    event_code = (
        request.form.get("event_code", "").strip()
        or session.get("guest_event_code", "")
    )

    if event_code:
        with get_db() as conn:
            event = conn.execute(
                "SELECT * FROM events WHERE code=?", (event_code,)
            ).fetchone()
        if not event:
            error_msg = f"No event found with code '{event_code}'."
            event_code = ""
            session.pop("guest_event_code", None)
        else:
            session["guest_event_code"] = event_code

    if request.method == "POST":
        action = request.form.get("action", "search")

        if action == "set_event":
            # Just setting the event code — render form with event loaded
            pass

        elif action == "search":
            if not event:
                error_msg = error_msg or "Please enter a valid event code first."
            else:
                selfie = request.files.get("selfie")
                if not selfie or not selfie.filename:
                    error_msg = "Please upload a selfie photo."
                elif not (selfie.filename.rsplit(".", 1)[-1].lower() in ALLOWED_IMAGE_EXT):
                    error_msg = "Unsupported image format. Use JPG, PNG, or WEBP."
                else:
                    selfie_bytes = selfie.read()
                    try:
                        result = find_matching_photos(selfie_bytes, event["id"])
                        if result is None:
                            error_msg = "No face detected in your selfie. Try a clearer, well-lit photo."
                        else:
                            matched  = result
                            searched = True
                            session["matched_photos"] = matched
                    except Exception as exc:
                        error_msg = f"Processing error: {exc}"

    return render_template(
        "find_my_photos.html",
        name=session.get("user_name", ""),
        email=session.get("user_email", ""),
        event=event,
        event_code=event_code,
        searched=searched,
        photos=matched,
        count=len(matched),
        error_msg=error_msg,
    )


# ─── Secure file serving ───────────────────────────────────────────────────────

@app.route("/uploads/<filename>")
def uploaded_file(filename):
    filename = secure_filename(filename)
    
    # Admin can access any file (original or preview)
    if session.get("admin_id"):
        return send_from_directory(UPLOAD_FOLDER, filename)
    
    # Guests only get watermarked versions for protection
    matched = session.get("matched_photos", [])

    original_filename = None
    if filename in matched:
        original_filename = filename
    elif filename.endswith("_watermarked.jpg"):
        original_stem = filename.replace("_watermarked.jpg", "")
        for photo in matched:
            if photo.rsplit(".", 1)[0] == original_stem:
                original_filename = photo
                break

    if not original_filename:
        abort(403)

    original_path = os.path.join(UPLOAD_FOLDER, original_filename)
    if not os.path.exists(original_path):
        abort(404)

    # Always refresh guest preview from original to ensure watermark is present.
    # This also repairs old previews that were generated before watermark settings changed.
    watermarked_name = original_filename.rsplit(".", 1)[0] + "_watermarked.jpg"
    watermarked_path = os.path.join(UPLOAD_FOLDER, watermarked_name)
    generated = generate_watermarked_preview(original_path, watermarked_path, max_width=800)
    if not generated or not os.path.exists(watermarked_path):
        abort(500)

    return send_from_directory(UPLOAD_FOLDER, watermarked_name)






# ─── Watermark Management ──────────────────────────────────────────────────────

@app.route("/admin/watermark-settings", methods=["GET", "POST"])
@admin_required
def watermark_settings():
    """Get or update watermark settings."""
    global _watermark_settings
    
    if request.method == "POST":
        enabled = request.form.get("enabled") == "on"
        text = request.form.get("text", "PREVIEW ONLY").strip()
        text_size = int(request.form.get("text_size", 60))
        placement = request.form.get("placement", "diagonal")
        opacity = float(request.form.get("opacity", 0.3))
        
        # Validate inputs
        text_size = max(20, min(120, text_size))  # 20-120px
        opacity = max(0.1, min(0.8, opacity))  # 0.1-0.8
        
        # Update database
        with get_db() as conn:
            conn.execute(
                "UPDATE watermark_settings SET enabled=?, text=?, text_size=?, placement=?, opacity=? WHERE id=1",
                (enabled, text, text_size, placement, opacity)
            )
            conn.commit()
        
        # Update in-memory settings
        _watermark_settings.update({
            "enabled": enabled,
            "text": text,
            "text_size": text_size,
            "placement": placement,
            "opacity": opacity
        })
        
        flash("Watermark settings updated!", "success")
        return redirect(url_for("admin") + "#watermark")
    
    # GET: Return current settings as JSON
    return jsonify(_watermark_settings)


@app.route("/admin/watermark-preview", methods=["POST"])
@admin_required
def watermark_preview():
    """Generate a preview of the watermark on a sample image."""
    try:
        # Settings from request
        enabled = request.form.get("enabled") == "on"
        text = request.form.get("text", "PREVIEW ONLY").strip()
        text_size = int(request.form.get("text_size", 60))
        placement = request.form.get("placement", "diagonal")
        opacity = float(request.form.get("opacity", 0.3))
        
        # Create a sample image
        sample_img = Image.new("RGB", (600, 400), (100, 150, 200))
        
        if not enabled:
            # Return plain image
            buffer = io.BytesIO()
            sample_img.save(buffer, format="JPEG", quality=85)
            buffer.seek(0)
            return send_file(buffer, mimetype="image/jpeg")
        
        # Add watermark to sample
        watermark = Image.new("RGBA", sample_img.size, (255, 255, 255, 0))
        draw = ImageDraw.Draw(watermark)
        
        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf", text_size)
        except:
            font = ImageFont.load_default()
        
        bbox = draw.textbbox((0, 0), text, font=font)
        text_width = bbox[2] - bbox[0]
        text_height = bbox[3] - bbox[1]
        alpha = int(255 * opacity)
        
        if placement == "diagonal":
            x = (sample_img.width - text_width) // 2
            y = (sample_img.height - text_height) // 2
        elif placement == "center":
            x = (sample_img.width - text_width) // 2
            y = (sample_img.height - text_height) // 2
        elif placement == "top_left":
            x = 10
            y = 10
        elif placement == "bottom_right":
            x = sample_img.width - text_width - 10
            y = sample_img.height - text_height - 10
        else:
            x = (sample_img.width - text_width) // 2
            y = (sample_img.height - text_height) // 2
        
        draw.text((x, y), text, font=font, fill=(255, 255, 255, alpha))
        
        # Add copyright mark at bottom right corner
        copyright_text = "© Bombay Studio"
        copyright_size = max(12, int(text_size * 0.25))
        try:
            copyright_font = ImageFont.truetype("/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf", copyright_size)
        except:
            copyright_font = ImageFont.load_default()
        
        copy_bbox = draw.textbbox((0, 0), copyright_text, font=copyright_font)
        copy_width = copy_bbox[2] - copy_bbox[0]
        copy_height = copy_bbox[3] - copy_bbox[1]
        copy_x = sample_img.width - copy_width - 8
        copy_y = sample_img.height - copy_height - 8
        copyright_alpha = int(255 * (opacity * 0.7))
        
        draw.text((copy_x, copy_y), copyright_text, font=copyright_font, fill=(255, 255, 255, copyright_alpha))
        
        sample_img = Image.alpha_composite(sample_img.convert("RGBA"), watermark).convert("RGB")
        
        buffer = io.BytesIO()
        sample_img.save(buffer, format="JPEG", quality=85)
        buffer.seek(0)
        return send_file(buffer, mimetype="image/jpeg")
        
    except Exception as exc:
        print(f"Error generating preview: {exc}")
        abort(500)


# ─── Run ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    init_db()
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
