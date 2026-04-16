import os
import io
import pickle
import random
import sqlite3
import string
import threading
import uuid
import socket
from collections import deque
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

from engine import ProcessorConfig, UniversalProcessor, create_highlight_preview
from storage import StorageConfig, StorageManager
from watcher import FTPFolderWatcher, start_watcher_thread

try:
    import importlib

    SocketIO = importlib.import_module("flask_socketio").SocketIO
except Exception:  # pragma: no cover - optional dependency
    class SocketIO:  # minimal fallback
        def __init__(self, app, **kwargs):
            self._app = app

        def emit(self, *args, **kwargs):
            return None

        def run(self, app, host="0.0.0.0", port=5000, debug=False):
            app.run(host=host, port=port, debug=debug)

app = Flask(__name__)
app.secret_key = os.environ.get("SESSION_SECRET", "dev-secret-change-me")
socketio = SocketIO(app, async_mode="threading", cors_allowed_origins="*")

BASE_DIR      = os.path.dirname(os.path.abspath(__file__))
UPLOAD_FOLDER = os.path.join(BASE_DIR, "static", "uploads")
DB_PATH       = os.path.join(BASE_DIR, "faces.db")
VERIFICATION_CACHE_DIR = os.path.join(BASE_DIR, "cache", "verification")

ALLOWED_IMAGE_EXT = {"png", "jpg", "jpeg", "gif", "webp"}
ALLOWED_VIDEO_EXT = {"mp4", "mov"}
ALLOWED_EXTENSIONS = ALLOWED_IMAGE_EXT | ALLOWED_VIDEO_EXT
MAX_FACE_WORK_EDGE = int(os.environ.get("MAX_FACE_WORK_EDGE", "1600"))
VIDEO_SAMPLE_FPS = float(os.environ.get("VIDEO_SAMPLE_FPS", "1.0"))
VIDEO_DB_FLUSH_ROWS = int(os.environ.get("VIDEO_DB_FLUSH_ROWS", "128"))
MAX_UPLOAD_BATCH_FILES = int(os.environ.get("MAX_UPLOAD_BATCH_FILES", "150"))
MAX_UPLOAD_BATCH_BYTES = int(os.environ.get("MAX_UPLOAD_BATCH_BYTES", str(1024 * 1024 * 1024)))

FONT_PATH = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(VERIFICATION_CACHE_DIR, exist_ok=True)

# ─── Background job tracker ────────────────────────────────────────────────────
_jobs: dict = {}
_jobs_lock = threading.Lock()
_runtime_services_lock = threading.Lock()
_index_queue = deque()
_index_worker_running = False
_index_job_id = None


DEFAULT_RUNTIME_SETTINGS = {
    "processing_mode": "SERVER",
    "storage_mode": "LOCAL_DISK",
}


def get_runtime_settings() -> dict:
    settings = dict(DEFAULT_RUNTIME_SETTINGS)
    with get_db() as conn:
        rows = conn.execute(
            "SELECT key, value FROM app_settings WHERE key IN ('processing_mode', 'storage_mode')"
        ).fetchall()
    for row in rows:
        settings[row["key"]] = row["value"]
    settings["processing_mode"] = settings["processing_mode"].upper()
    settings["storage_mode"] = settings["storage_mode"].upper()
    return settings


def set_runtime_settings(processing_mode: str, storage_mode: str) -> dict:
    processing_mode = (processing_mode or "SERVER").upper()
    storage_mode = (storage_mode or "LOCAL_DISK").upper()
    with get_db() as conn:
        conn.execute(
            "INSERT INTO app_settings (key, value) VALUES ('processing_mode', ?) "
            "ON CONFLICT(key) DO UPDATE SET value=excluded.value",
            (processing_mode,),
        )
        conn.execute(
            "INSERT INTO app_settings (key, value) VALUES ('storage_mode', ?) "
            "ON CONFLICT(key) DO UPDATE SET value=excluded.value",
            (storage_mode,),
        )
        conn.commit()
    refresh_runtime_services()
    return get_runtime_settings()


_processor = None
_storage = None


def _build_processor(settings: dict) -> UniversalProcessor:
    return UniversalProcessor(
        ProcessorConfig(
            mode=settings["processing_mode"],
            manifest_dir=os.environ.get("LOCAL_MANIFEST_DIR"),
            rekognition_region=os.environ.get("AWS_REGION", "us-east-1"),
        )
    )


def _build_storage(settings: dict) -> StorageManager:
    return StorageManager(
        StorageConfig(
            mode=settings["storage_mode"],
            local_root=UPLOAD_FOLDER,
            bucket=os.environ.get("S3_BUCKET"),
            endpoint_url=os.environ.get("S3_ENDPOINT_URL"),
            region=os.environ.get("AWS_REGION", "auto"),
            access_key_id=os.environ.get("S3_ACCESS_KEY_ID") or os.environ.get("AWS_ACCESS_KEY_ID"),
            secret_access_key=os.environ.get("S3_SECRET_ACCESS_KEY") or os.environ.get("AWS_SECRET_ACCESS_KEY"),
        )
    )


def refresh_runtime_services() -> None:
    global _processor, _storage
    settings = get_runtime_settings()
    with _runtime_services_lock:
        _processor = _build_processor(settings)
        try:
            _storage = _build_storage(settings)
        except RuntimeError as exc:
            # Avoid crashing the app if S3 mode was selected but boto3 is unavailable.
            if settings.get("storage_mode") == "S3_COMPATIBLE" and "boto3" in str(exc).lower():
                fallback_settings = dict(settings)
                fallback_settings["storage_mode"] = "LOCAL_DISK"
                with get_db() as conn:
                    conn.execute(
                        "INSERT INTO app_settings (key, value) VALUES ('storage_mode', ?) "
                        "ON CONFLICT(key) DO UPDATE SET value=excluded.value",
                        ("LOCAL_DISK",),
                    )
                    conn.commit()
                _storage = _build_storage(fallback_settings)
                print("[startup] storage_mode fell back to LOCAL_DISK because boto3 is not installed")
            else:
                raise


def get_processor() -> UniversalProcessor:
    if _processor is None:
        refresh_runtime_services()
    return _processor


def get_storage() -> StorageManager:
    if _storage is None:
        refresh_runtime_services()
    return _storage


# ══════════════════════════════════════════════════════════════════════════════
# DATABASE
# ══════════════════════════════════════════════════════════════════════════════

def get_db():
    conn = sqlite3.connect(DB_PATH, timeout=30)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    conn.execute("PRAGMA temp_store=MEMORY")
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
                frame_time REAL DEFAULT 0,
                top INTEGER,
                right INTEGER,
                bottom INTEGER,
                left INTEGER,
                verified INTEGER DEFAULT 0,
                verified_user_id INTEGER,
                verified_at TIMESTAMP
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
        conn.execute("""
            CREATE TABLE IF NOT EXISTS app_settings (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS verification_feedback (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL,
                face_id INTEGER NOT NULL,
                event_id INTEGER NOT NULL,
                decision TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(user_id, face_id)
            )
        """)
        # Migrations for existing DBs
        for col, defn in [
            ("event_id",   "INTEGER REFERENCES events(id)"),
            ("media_type", "TEXT DEFAULT 'photo'"),
            ("frame_time", "REAL DEFAULT 0"),
            ("top", "INTEGER"),
            ("right", "INTEGER"),
            ("bottom", "INTEGER"),
            ("left", "INTEGER"),
            ("verified", "INTEGER DEFAULT 0"),
            ("verified_user_id", "INTEGER"),
            ("verified_at", "TIMESTAMP"),
        ]:
            _add_column_if_missing(conn, "faces", col, defn)
        for col, defn in [
            ("album_min", "INTEGER DEFAULT 0"),
            ("album_max", "INTEGER DEFAULT 0"),
        ]:
            _add_column_if_missing(conn, "events", col, defn)

        # Indexes for large-scale indexing and search workloads.
        conn.execute("CREATE INDEX IF NOT EXISTS idx_faces_event_id ON faces(event_id)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_faces_event_file ON faces(event_id, photo_filename)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_faces_event_id_id ON faces(event_id, id)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_faces_verified_user ON faces(verified_user_id)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_album_user_event ON album_selections(user_id, event_id)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_feedback_user_event ON verification_feedback(user_id, event_id)")

        # Default admin
        if not conn.execute("SELECT id FROM admins LIMIT 1").fetchone():
            conn.execute(
                "INSERT INTO admins (username, password_hash) VALUES (?, ?)",
                ("admin", generate_password_hash("admin123"))
            )
        # Default watermark row
        if not conn.execute("SELECT id FROM watermark_settings LIMIT 1").fetchone():
            conn.execute("INSERT INTO watermark_settings (id) VALUES (1)")
        for key, value in DEFAULT_RUNTIME_SETTINGS.items():
            conn.execute(
                "INSERT OR IGNORE INTO app_settings (key, value) VALUES (?, ?)",
                (key, value),
            )
        conn.commit()


def get_watermark():
    with get_db() as conn:
        row = conn.execute("SELECT * FROM watermark_settings WHERE id=1").fetchone()
    if not row:
        return {
            "id": 1,
            "enabled": 0,
            "text": "Bombay Studio",
            "font_size": 48,
            "opacity": 60,
            "position": "bottom-right",
            "color": "#ffffff",
        }
    return dict(row)


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


def _downscale_for_face_work(rgb: np.ndarray, max_edge: int = MAX_FACE_WORK_EDGE) -> np.ndarray:
    """Resize large inputs before face encoding to cut CPU load significantly."""
    if rgb is None or rgb.size == 0:
        return rgb
    h, w = rgb.shape[:2]
    longest = max(h, w)
    if longest <= max_edge:
        return rgb
    scale = max_edge / float(longest)
    new_w = max(1, int(w * scale))
    new_h = max(1, int(h * scale))
    return cv2.resize(rgb, (new_w, new_h), interpolation=cv2.INTER_AREA)


def load_image_rgb(filepath):
    img = cv2.imread(filepath)
    if img is None:
        rgb = np.array(Image.open(filepath).convert("RGB"))
    else:
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return _downscale_for_face_work(rgb)

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
        locations = face_recognition.face_locations(img)
        encodings = face_recognition.face_encodings(img, known_face_locations=locations)
        if encodings:
            rows = [
                (filename, pickle.dumps(enc), event_id, top, right, bottom, left)
                for enc, (top, right, bottom, left) in zip(encodings, locations)
            ]
            conn.executemany(
                "INSERT INTO faces (photo_filename, encoding, event_id, media_type, frame_time, top, right, bottom, left) VALUES (?,?,?,'photo',0,?,?,?,?)",
                rows
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
    frame_interval = max(1, int(fps / max(0.1, VIDEO_SAMPLE_FPS)))
    frame_idx = total_faces = 0
    pending_rows = []
    conn = get_db()

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if frame_idx % frame_interval == 0:
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                rgb = _downscale_for_face_work(rgb)
                locations = face_recognition.face_locations(rgb)
                encodings = face_recognition.face_encodings(rgb, known_face_locations=locations)
                t = round(frame_idx / fps, 2)
                if encodings:
                    pending_rows.extend(
                        (filename, pickle.dumps(enc), event_id, t, top, right, bottom, left)
                        for enc, (top, right, bottom, left) in zip(encodings, locations)
                    )
                    total_faces += len(encodings)
                    if len(pending_rows) >= VIDEO_DB_FLUSH_ROWS:
                        conn.executemany(
                            "INSERT INTO faces (photo_filename, encoding, event_id, media_type, frame_time, top, right, bottom, left) VALUES (?,?,?,'video',?,?,?,?,?)",
                            pending_rows,
                        )
                        conn.commit()
                        pending_rows.clear()
            frame_idx += 1

        if pending_rows:
            conn.executemany(
                "INSERT INTO faces (photo_filename, encoding, event_id, media_type, frame_time, top, right, bottom, left) VALUES (?,?,?,'video',?,?,?,?,?)",
                pending_rows,
            )
            conn.commit()
    finally:
        conn.close()
        cap.release()

    return total_faces


def _emit_job_progress(job_id: str, done: int, total: int, status: str, errors: list, current_file: str = ""):
    payload = {
        "job_id": job_id,
        "done": done,
        "total": total,
        "status": status,
        "percent": int(100 * done / total) if total else 0,
        "errors": errors,
        "current_file": current_file,
    }
    socketio.emit("index_progress", payload, namespace="/admin")


def _run_indexing_job(job_id, file_list, event_id):
    with app.app_context():
        settings = get_runtime_settings()
        processing_mode = settings["processing_mode"]
        total = len(file_list)
        done = 0
        errors = []
        _emit_job_progress(job_id, done, total, "running", errors)
        for filename, _path in file_list:
            try:
                if processing_mode == "LOCAL":
                    # LOCAL lane expects desktop-side manifest ingestion.
                    get_processor().process_image(_path)
                elif processing_mode == "API":
                    # API lane calls Rekognition (analysis metadata) and does not write local encodings.
                    get_processor().process_image(_path)
                else:
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
            _emit_job_progress(job_id, done, total, "running", errors, current_file=filename)
        with _jobs_lock:
            _jobs[job_id]["status"] = "complete"
        _emit_job_progress(job_id, done, total, "complete", errors)


def _run_index_worker(job_id: str):
    global _index_worker_running
    with app.app_context():
        while True:
            with _jobs_lock:
                if not _index_queue:
                    if job_id in _jobs:
                        _jobs[job_id]["status"] = "complete"
                        done = _jobs[job_id].get("done", 0)
                        total = _jobs[job_id].get("total", 0)
                        errors = list(_jobs[job_id].get("errors", []))
                    else:
                        done, total, errors = 0, 0, []
                    _index_worker_running = False
                    _emit_job_progress(job_id, done, total, "complete", errors)
                    return
                filename, file_path, event_id = _index_queue.popleft()

            try:
                settings = get_runtime_settings()
                processing_mode = settings["processing_mode"]
                if processing_mode == "LOCAL":
                    get_processor().process_image(file_path)
                elif processing_mode == "API":
                    get_processor().process_image(file_path)
                else:
                    if is_video(filename):
                        extract_faces_from_video(filename, event_id)
                    else:
                        extract_and_store_faces(filename, event_id)
                err = None
            except Exception as exc:
                err = f"{filename}: {exc}"

            with _jobs_lock:
                job = _jobs.get(job_id)
                if not job:
                    continue
                if err:
                    job.setdefault("errors", []).append(err)
                job["done"] = int(job.get("done", 0)) + 1
                done = job["done"]
                total = int(job.get("total", 0))
                errors = list(job.get("errors", []))

            _emit_job_progress(job_id, done, total, "running", errors, current_file=filename)


def _enqueue_indexing_files(file_list, event_id) -> str:
    global _index_worker_running, _index_job_id
    start_worker = False
    with _jobs_lock:
        if not _index_job_id or _index_job_id not in _jobs or _jobs[_index_job_id].get("status") == "complete":
            _index_job_id = str(uuid.uuid4())
            _jobs[_index_job_id] = {"total": 0, "done": 0, "errors": [], "status": "running"}

        job = _jobs[_index_job_id]
        for filename, save_path in file_list:
            _index_queue.append((filename, save_path, event_id))
        job["total"] = int(job.get("total", 0)) + len(file_list)
        job["status"] = "running"

        if not _index_worker_running:
            _index_worker_running = True
            start_worker = True

        job_id = _index_job_id
        done = int(job.get("done", 0))
        total = int(job.get("total", 0))
        errors = list(job.get("errors", []))

    _emit_job_progress(job_id, done, total, "running", errors)
    if start_worker:
        threading.Thread(target=_run_index_worker, args=(job_id,), daemon=True).start()
    return job_id


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


def _validate_selfie_quality(selfie_bytes):
    """Validate selfie quality: single clear face with acceptable lighting."""
    try:
        rgb = _decode_selfie(selfie_bytes)
    except Exception:
        return False, "Could not read image. Please retake the selfie."

    if rgb is None or rgb.size == 0:
        return False, "Could not read image. Please retake the selfie."

    # Lighting check (mean grayscale brightness)
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    brightness = float(np.mean(gray))
    if brightness < 55:
        return False, "Image is too dark. Please move to better light."
    if brightness > 235:
        return False, "Image is too bright. Please avoid harsh light."

    # Blur check using variance of Laplacian
    blur_score = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    if blur_score < 90:
        return False, "Image is blurry. Hold still and retake."

    face_locations = face_recognition.face_locations(rgb)
    if len(face_locations) == 0:
        return False, "No face detected. Center your face and retake."
    if len(face_locations) > 1:
        return False, "Multiple faces detected. Only one face should be visible."

    top, right, bottom, left = face_locations[0]
    face_area = max(1, (right - left) * (bottom - top))
    img_area = max(1, rgb.shape[0] * rgb.shape[1])
    face_ratio = face_area / img_area
    if face_ratio < 0.035:
        return False, "Face is too small. Move closer to the camera."

    return True, "Selfie accepted."


def find_matching_photos(selfie_bytes, event_id, user_id=None):
    img = _decode_selfie(selfie_bytes)
    img = _downscale_for_face_work(img)
    
    # Keeping the upgraded AI vision (Upsample & Jitter) so accuracy stays high
    locations = face_recognition.face_locations(img, number_of_times_to_upsample=1, model="hog")
    encs = face_recognition.face_encodings(img, known_face_locations=locations, num_jitters=2)
    
    if not encs:
        return None
    selfie_enc = encs[0]
    BATCH = 2000
    matched = set()

    last_id = 0
    while True:
        with get_db() as conn:
            rows = conn.execute(
                "SELECT id, photo_filename, encoding "
                "FROM faces WHERE event_id=? AND id>? ORDER BY id LIMIT ?",
                (event_id, last_id, BATCH),
            ).fetchall()
        if not rows:
            break
            
        filenames  = [r["photo_filename"] for r in rows]
        enc_matrix = np.array([pickle.loads(r["encoding"]) for r in rows], dtype=np.float64)
        distances  = face_recognition.face_distance(enc_matrix, selfie_enc)
        
        for row, fname, dist in zip(rows, filenames, distances):
            # ---------------------------------------------------------
            # FULL AUTO MATH: Anything under 0.50 is automatically added to the gallery.
            # If you find it is missing too many faces tomorrow, change this to 0.55.
            # If it is adding the wrong people, change it to 0.45.
            # ---------------------------------------------------------
            if float(dist) <= 0.50:
                matched.add(fname)
                
        last_id = rows[-1]["id"]

    # Tell the frontend that verification is permanently disabled
    return {
        "matched": sorted(matched),
        "needs_verification": False,
        "verification_candidates": [],
    }

@app.route("/validate-selfie", methods=["POST"])
@login_required
def validate_selfie():
    selfie = request.files.get("selfie")
    if not selfie or not selfie.filename:
        return jsonify({"ok": False, "error": "Please capture or upload a selfie."}), 400

    ext = selfie.filename.rsplit(".", 1)[-1].lower() if "." in selfie.filename else "jpg"
    if ext not in ALLOWED_IMAGE_EXT:
        return jsonify({"ok": False, "error": "Unsupported image format."}), 400

    ok, message = _validate_selfie_quality(selfie.read())
    return jsonify({"ok": ok, "error": message if not ok else None, "message": message})


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

@app.route("/admin", methods=["GET", "POST"], strict_slashes=False)
@admin_required
def admin():
    if request.method == "POST":
        action = request.form.get("action", "upload")
        if action == "save_runtime_settings":
            processing_mode = request.form.get("processing_mode", "SERVER")
            storage_mode = request.form.get("storage_mode", "LOCAL_DISK")
            settings = set_runtime_settings(processing_mode, storage_mode)
            flash(
                f"Settings saved. Processing={settings['processing_mode']}, Storage={settings['storage_mode']}",
                "success",
            )
            return redirect(url_for("admin") + "#upload")

        event_id = request.form.get("event_id", type=int)
        if not event_id:
            error_msg = "Select an event first."
            if request.headers.get('X-Requested-With') == 'XMLHttpRequest':
                return jsonify({"error": error_msg}), 400
            flash(error_msg, "danger")
            return redirect(url_for("admin") + "#upload")
            
        files = request.files.getlist("photos")
        if not files or all(f.filename == "" for f in files):
            error_msg = "No files selected."
            if request.headers.get('X-Requested-With') == 'XMLHttpRequest':
                return jsonify({"error": error_msg}), 400
            flash(error_msg, "danger")
            return redirect(url_for("admin") + "#upload")

        non_empty_files = [f for f in files if f and f.filename]
        if len(non_empty_files) > MAX_UPLOAD_BATCH_FILES:
            error_msg = (
                f"Too many files in one request ({len(non_empty_files)}). "
                f"Please upload up to {MAX_UPLOAD_BATCH_FILES} files per batch."
            )
            if request.headers.get('X-Requested-With') == 'XMLHttpRequest':
                return jsonify({"error": error_msg}), 413
            flash(error_msg, "warning")
            return redirect(url_for("admin") + "#upload")

        req_size = int(request.content_length or 0)
        if req_size and req_size > MAX_UPLOAD_BATCH_BYTES:
            max_mb = MAX_UPLOAD_BATCH_BYTES // (1024 * 1024)
            got_mb = req_size // (1024 * 1024)
            error_msg = (
                f"Upload batch too large ({got_mb}MB). Keep each batch under about {max_mb}MB."
            )
            if request.headers.get('X-Requested-With') == 'XMLHttpRequest':
                return jsonify({"error": error_msg}), 413
            flash(error_msg, "warning")
            return redirect(url_for("admin") + "#upload")

        saved, skipped, bad_ext = [], [], []
        storage = get_storage()
        for f in files:
            if not f or not f.filename:
                continue
            if not allowed_file(f.filename):
                bad_ext.append(f.filename); continue
            filename  = secure_filename(f.filename)
            try:
                object_key, save_path = storage.save_upload(f, filename)
                filename = object_key
            except Exception as exc:
                bad_ext.append(f"{filename} (store error: {exc})")
                continue
            with get_db() as conn:
                if _file_already_indexed(filename, event_id, conn):
                    skipped.append(filename); continue
            saved.append((filename, save_path))

        # Check if this is an AJAX request
        is_ajax = request.headers.get('X-Requested-With') == 'XMLHttpRequest'
        
        for n in bad_ext:
            if not is_ajax:
                flash(f"{n}: unsupported type.", "warning")
        if skipped:
            if not is_ajax:
                flash(f"{len(skipped)} already indexed — skipped.", "info")
        if not saved:
            if is_ajax:
                # FIX: Return 200 OK. An empty batch (duplicates) is a success, not a fatal error!
                return jsonify({
                    "success": True,
                    "job_id": None,
                    "files_queued": 0,
                    "files_skipped": len(skipped),
                    "files_invalid": len(bad_ext)
                }), 200
                
            flash("Nothing new to index.", "info")
            return redirect(url_for("admin"))
            
        job_id = _enqueue_indexing_files(saved, event_id)
        session["current_job_id"] = job_id
        
        if is_ajax:
            # Return JSON for AJAX requests
            return jsonify({
                "success": True,
                "job_id": job_id,
                "files_queued": len(saved),
                "files_skipped": len(skipped),
                "files_invalid": len(bad_ext)
            }), 200
        
        # Return redirect for regular form submissions
        flash(f"Queued {len(saved)} file(s). Upload can continue while indexing runs in background.", "info")
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
    runtime_settings = get_runtime_settings()
    job_id = session.get("current_job_id")
    current_job = None
    if job_id:
        with _jobs_lock:
            current_job = dict(_jobs.get(job_id, {}))

    return render_template("admin.html",
        events=events, photos=photos, users=users,
        selections=selections, wm=wm,
        current_job=current_job, job_id=job_id,
        runtime_settings=runtime_settings,
        upload_max_batch_files=MAX_UPLOAD_BATCH_FILES,
        upload_max_batch_bytes=MAX_UPLOAD_BATCH_BYTES)


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
            """
            INSERT INTO watermark_settings (id, enabled, text, font_size, opacity, position, color)
            VALUES (1, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(id) DO UPDATE SET
                enabled=excluded.enabled,
                text=excluded.text,
                font_size=excluded.font_size,
                opacity=excluded.opacity,
                position=excluded.position,
                color=excluded.color
            """,
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
    needs_verification = False
    verification_candidates = []

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
                        result = find_matching_photos(selfie.read(), event["id"], session.get("user_id"))
                        if result is None:
                            error_msg = "No face detected. Try a clearer, well-lit photo."
                        else:
                            matched = result.get("matched", [])
                            needs_verification = bool(result.get("needs_verification"))
                            verification_candidates = result.get("verification_candidates", [])
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
        needs_verification=needs_verification,
        verification_candidates=verification_candidates,
    )


@app.route("/verification-preview/<filename>")
@login_required
def verification_preview(filename):
    return send_from_directory(VERIFICATION_CACHE_DIR, secure_filename(filename))


@app.route("/verify-match", methods=["POST"])
@login_required
def verify_match():
    data = request.get_json(silent=True) or {}
    face_id = data.get("face_id")
    event_id = data.get("event_id")
    photo_filename = secure_filename((data.get("photo_filename") or "").strip())
    decision = (data.get("decision") or "yes").strip().lower()

    try:
        face_id = int(face_id)
        event_id = int(event_id)
    except Exception:
        return jsonify({"ok": False, "msg": "Invalid verification payload."}), 400

    if decision not in {"yes", "skip"}:
        return jsonify({"ok": False, "msg": "Invalid decision."}), 400

    uid = session.get("user_id")
    if not uid or not photo_filename:
        return jsonify({"ok": False, "msg": "Missing user or photo."}), 400

    with get_db() as conn:
        row = conn.execute(
            "SELECT id, photo_filename, event_id FROM faces WHERE id=? AND event_id=? AND photo_filename=? LIMIT 1",
            (face_id, event_id, photo_filename),
        ).fetchone()
        if not row:
            return jsonify({"ok": False, "msg": "Face record not found."}), 404

        if decision == "yes":
            conn.execute(
                "UPDATE faces SET verified=1, verified_user_id=?, verified_at=CURRENT_TIMESTAMP WHERE id=?",
                (uid, face_id),
            )
            ## FIX: Removed the automatic album insertion. It now goes to the normal gallery.

        conn.execute(
            "INSERT INTO verification_feedback (user_id, face_id, event_id, decision) VALUES (?,?,?,?) "
            "ON CONFLICT(user_id, face_id) DO UPDATE SET decision=excluded.decision, created_at=CURRENT_TIMESTAMP",
            (uid, face_id, event_id, decision),
        )
        conn.commit()

    if decision == "yes":
        matched = set(session.get("matched_photos", []))
        matched.add(photo_filename)
        session["matched_photos"] = sorted(matched)

    return jsonify({"ok": True, "photo_filename": photo_filename, "decision": decision})


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
    storage = get_storage()
    file_exists_locally = os.path.exists(filepath)

    # 1. Authorization Check
    is_admin = bool(session.get("admin_id"))
    is_guest = bool(session.get("user_id"))

    if not is_admin and not is_guest:
        abort(403)

    if not is_admin:
        # Guest specific album check
        matched = session.get("matched_photos", [])
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

    # 2. Handle Cloud Storage Fallback
    if not file_exists_locally:
        if storage.mode == "S3_COMPATIBLE":
            try:
                return redirect(storage.generate_private_url(filename, expires_seconds=300))
            except Exception:
                abort(404)
        abort(404)

    # 3. Dynamic Generation (Thumbnails & 1080p Previews)
    ext = filename.rsplit(".", 1)[-1].lower()
    target_path = filepath

    req_thumb = request.args.get("thumb") == "1"
    req_preview = request.args.get("preview") == "1"

    # Intercept for Thumbnails (600px) OR Previews (1080p)
    if (req_thumb or req_preview) and ext in ALLOWED_IMAGE_EXT and ext not in {"gif"}:
        proxy_dir = os.path.join(UPLOAD_FOLDER, ".thumbnails" if req_thumb else ".previews")
        os.makedirs(proxy_dir, exist_ok=True)
        proxy_path = os.path.join(proxy_dir, filename)

        # Build the proxy image if it doesn't exist yet
        if not os.path.exists(proxy_path):
            try:
                from PIL import Image
                with Image.open(filepath) as img:
                    max_size = (600, 600) if req_thumb else (1920, 1080)
                    img.thumbnail(max_size)
                    img.convert("RGB").save(proxy_path, "JPEG", quality=85)
            except Exception:
                pass

        if os.path.exists(proxy_path):
            target_path = proxy_path

    # 4. Serve the file (Apply watermark ONLY if it is a guest)
    if not is_admin:
        wm = get_watermark()
        if ext in ALLOWED_IMAGE_EXT and ext not in {"gif"}:
            return serve_image_with_watermark(target_path, wm)

    return send_file(target_path)

    # Enforce guest authorization before serving local or signed cloud URL.
    matched = session.get("matched_photos", [])
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

    if not file_exists_locally and storage.mode == "S3_COMPATIBLE":
        try:
            return redirect(storage.generate_private_url(filename, expires_seconds=300))
        except Exception:
            abort(404)

    if not file_exists_locally:
        abort(404)

    # Apply watermark for guests
    wm = get_watermark()
    ext = filename.rsplit(".", 1)[-1].lower()
    if ext in ALLOWED_IMAGE_EXT and ext not in {"gif"}:
        return serve_image_with_watermark(filepath, wm)
    return send_from_directory(UPLOAD_FOLDER, filename)


def _handle_ftp_new_image(path: str):
    """Background callback for FTP watcher lane."""
    try:
        filename = secure_filename(os.path.basename(path))
        target_path = os.path.join(UPLOAD_FOLDER, filename)
        if os.path.abspath(path) != os.path.abspath(target_path):
            with open(path, "rb") as src, open(target_path, "wb") as dst:
                dst.write(src.read())

        # Trigger lane processing. Event assignment is optional via env.
        event_id = int(os.environ.get("FTP_DEFAULT_EVENT_ID", "0") or 0)
        settings = get_runtime_settings()
        if settings["processing_mode"] == "SERVER" and event_id:
            extract_and_store_faces(filename, event_id)
        else:
            get_processor().process_image(target_path)

        if get_storage().mode == "S3_COMPATIBLE":
            get_storage().store_file(target_path, filename)
    except Exception as exc:
        print(f"FTP watcher error for {path}: {exc}")


if __name__ == "__main__":
    init_db()
    refresh_runtime_services()

    # If PORT is not explicitly provided and 5000 is occupied,
    # auto-pick a free local port so `python3 app.py` still boots.
    requested_port = int(os.environ.get("PORT", 5000))
    port = requested_port
    if "PORT" not in os.environ:
        probe = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        probe.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            probe.bind(("0.0.0.0", requested_port))
        except OSError:
            probe.close()
            probe = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            probe.bind(("0.0.0.0", 0))
            port = probe.getsockname()[1]
            print(f"Port {requested_port} is busy, using {port} instead.")
        finally:
            probe.close()

    if os.environ.get("FTP_WATCH_ENABLED", "0") == "1":
        watch_dir = os.environ.get("FTP_WATCH_DIR", os.path.join(BASE_DIR, "ftp_drop"))
        watcher = FTPFolderWatcher(watch_dir=watch_dir, on_new_image=_handle_ftp_new_image)
        start_watcher_thread(watcher)

    socketio.run(app, host="0.0.0.0", port=port, debug=False)
