git rm -r --cached faces.db
git rm -r --cached uploads/ayscale brightness)
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
    encs = face_recognition.face_encodings(img)
    if not encs:
        return None
    selfie_enc = encs[0]
    BATCH = 2000
    matched = set()
    candidate_by_photo = {}

    handled_face_ids = set()
    if user_id:
        with get_db() as conn:
            handled_rows = conn.execute(
                "SELECT face_id FROM verification_feedback WHERE user_id=? AND event_id=?",
                (user_id, event_id),
            ).fetchall()
        handled_face_ids = {int(r["face_id"]) for r in handled_rows}

    last_id = 0
    while True:
        with get_db() as conn:
            rows = conn.execute(
                "SELECT id, photo_filename, encoding, top, right, bottom, left, verified_user_id "
                "FROM faces WHERE event_id=? AND id>? ORDER BY id LIMIT ?",
                (event_id, last_id, BATCH),
            ).fetchall()
        if not rows:
            break
        filenames  = [r["photo_filename"] for r in rows]
        enc_matrix = np.array([pickle.loads(r["encoding"]) for r in rows], dtype=np.float64)
        distances  = face_recognition.face_distance(enc_matrix, selfie_enc)
        for row, fname, dist in zip(rows, filenames, distances):
            face_id = int(row["id"])
            if user_id and (face_id in handled_face_ids or int(row["verified_user_id"] or 0) == int(user_id)):
                continue

            dist = float(dist)
            if dist <= 0.45:
                matched.add(fname)
                continue
            if dist <= 0.60:
                top = int(row["top"] or 0)
                right = int(row["right"] or 0)
                bottom = int(row["bottom"] or 0)
                left = int(row["left"] or 0)
                coords = (top, right, bottom, left)
                try:
                    preview_filename = create_highlight_preview(
                        os.path.join(UPLOAD_FOLDER, fname),
                        coords,
                        cache_dir=VERIFICATION_CACHE_DIR,
                        key=f"{event_id}_{row['id']}",
                    )
                except Exception:
                    preview_filename = None

                if not preview_filename:
                    continue

                existing = candidate_by_photo.get(fname)
                if existing is None or dist < existing["distance"]:
                    candidate_by_photo[fname] = {
                        "face_id": face_id,
                        "photo_filename": fname,
                        "distance": dist,
                        "event_id": int(event_id),
                        "preview_filename": preview_filename,
                    }
        last_id = rows[-1]["id"]

    verification_candidates = sorted(candidate_by_photo.values(), key=lambda c: c["distance"])[:8]
    return {
        "matched": sorted(matched),
        "needs_verification": bool(verification_candidates),
        "verification_candidates": verification_candidates,
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

@app.route("/admin", methods=["GET", "POST"])
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
            error_msg = "Nothing new to index."
            if is_ajax:
                return jsonify({"error": error_msg}), 400
            flash(error_msg, "info")
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
            conn.execute(
                "INSERT OR IGNORE INTO album_selections (user_id, event_id, photo_filename) VALUES (?,?,?)",
                (uid, event_id, photo_filename),
            )

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

    if session.get("admin_id"):
        if os.path.exists(filepath):
            return send_from_directory(UPLOAD_FOLDER, filename)
        storage = get_storage()
        if storage.mode == "S3_COMPATIBLE":
            try:
                return redirect(storage.generate_private_url(filename, expires_seconds=300))
            except Exception:
                abort(404)
        abort(404)

    if not session.get("user_id"):
        abort(403)

    storage = get_storage()
    file_exists_locally = os.path.exists(filepath)

    if not file_exists_locally and storage.mode != "S3_COMPATIBLE":
        abort(404)

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
