"""Match Tag — a football event tagger with automatic video analysis.

Run it with:
    python match_tag/app.py
and open http://localhost:5055
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

from flask import Flask, jsonify, render_template, request, send_from_directory, url_for
from flask_cors import CORS

HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parent

# The vision engine lives beside the web app rather than inside it, so that it
# can also be driven from the command line and from the training scripts.
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from analysis import analysis as analysis_bp   # noqa: E402
from api import api as api_bp                  # noqa: E402
from jobs import JobRunner, sweep_uploads      # noqa: E402
from models import db, ensure_schema           # noqa: E402

# Uploaded match video, 4 GB ceiling. A full 90 minutes at broadcast quality
# fits comfortably; anything larger is a mistake rather than a match.
MAX_UPLOAD_BYTES = 4 * 1024 * 1024 * 1024

# The public demo has no vision engine, no GPU and a small disk, so it offers
# the tagger and nothing that writes at scale. Two megabytes is far below any
# real video and still leaves room for a mis-click to arrive and be refused
# with a readable message rather than a dropped connection.
DEMO_UPLOAD_BYTES = 2 * 1024 * 1024


def public_demo() -> bool:
    return os.getenv("PUBLIC_DEMO") == "1"


def create_app(config: dict | None = None) -> Flask:
    app = Flask(__name__, static_folder="static", template_folder="templates")

    demo = public_demo()

    database_url = os.getenv("DATABASE_URL") or f"sqlite:///{HERE / 'instance' / 'football1.db'}"
    app.config.update(
        SQLALCHEMY_DATABASE_URI=database_url,
        SQLALCHEMY_TRACK_MODIFICATIONS=False,
        MAX_CONTENT_LENGTH=DEMO_UPLOAD_BYTES if demo else MAX_UPLOAD_BYTES,
        UPLOAD_DIR=os.getenv("UPLOAD_DIR", str(PROJECT_ROOT / "uploads")),
        JSON_SORT_KEYS=False,
    )
    if config:
        app.config.update(config)

    # Never let the browser hold a stale copy. A cached app.js served
    # alongside a freshly rendered template is not hypothetical: it took the
    # whole interface down once. The new script looked for an element the
    # cached page did not have, threw during start-up, and everything after
    # that line — the pitch markings, the player list, the outcome menu —
    # silently never ran. The page looked broken in a way unrelated to the
    # change that caused it.
    #
    # Zero here means "revalidate", not "re-download": the browser sends an
    # If-Modified-Since and gets a 304 for anything unchanged. Stamping a
    # version onto the URL was the other option and covers less — the ES
    # modules import each other by relative path, so only the entry point
    # would ever be versioned and a stale pitch.js could still pair with a
    # fresh app.js.
    app.config["SEND_FILE_MAX_AGE_DEFAULT"] = 0

    (HERE / "instance").mkdir(exist_ok=True)
    Path(app.config["UPLOAD_DIR"]).mkdir(parents=True, exist_ok=True)

    db.init_app(app)
    CORS(app, resources={r"/api/*": {"origins": os.getenv("CORS_ORIGINS", "*")}})

    app.register_blueprint(api_bp)
    app.register_blueprint(analysis_bp)

    with app.app_context():
        added = ensure_schema()
        if added:
            app.logger.info("database schema updated: %s", ", ".join(added))

    app.extensions["job_runner"] = JobRunner(app)

    if demo:
        # Refused here rather than inside the view, because by the time
        # start_analysis() decides the engine is missing it has already saved
        # the upload to disk. A 4 GB ceiling and a public URL is a way for a
        # stranger to fill the server's disk with videos nothing will ever
        # read. before_request runs before the body is touched.
        @app.before_request
        def demo_guard():
            path = request.path.rstrip("/")
            if request.method != "GET" and path.startswith("/api/analysis"):
                return jsonify({
                    "error": "Automatic analysis is disabled on the public demo. "
                             "Run it locally."
                }), 403
            if request.method == "DELETE" and path.startswith("/api/matches"):
                return jsonify({
                    "error": "Deleting matches is disabled on the public demo. "
                             "Everyone shares this database."
                }), 403
            return None

    swept = sweep_uploads(Path(app.config["UPLOAD_DIR"]))
    if swept:
        app.logger.info("removed %d upload folder(s) older than a week", swept)

    @app.get("/")
    def index():
        return render_template("index.html")

    @app.get("/health")
    def health():
        return jsonify({"status": "ok", "version": "2.0.0"})

    @app.errorhandler(413)
    def too_large(_):
        # Read the limit back off the config rather than the constant: the
        # demo lowers it, and a message naming 4 GB to somebody refused at
        # 2 MB sends them looking for a problem that is not there.
        limit = app.config["MAX_CONTENT_LENGTH"]
        if limit >= 1024 ** 3:
            size = f"{limit / 1024 ** 3:.0f} GB"
        else:
            size = f"{limit / 1024 ** 2:.0f} MB"
        return jsonify({"error": f"That file is over the {size} upload limit."}), 413

    @app.errorhandler(404)
    def not_found(_):
        return jsonify({"error": "Not found"}), 404

    return app


app = create_app()


if __name__ == "__main__":
    # Not 5000: macOS binds that to the AirPlay receiver by default, and the
    # resulting "Address already in use" is a confusing first experience.
    port = int(os.getenv("PORT", 5055))
    # threaded=True matters: the analysis worker and the browser's progress
    # polling have to make headway at the same time.
    app.run(host="0.0.0.0", port=port, threaded=True, debug=os.getenv("FLASK_DEBUG") == "1")
