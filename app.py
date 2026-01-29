"""
Johnson's Relative Weights — Web Application
Flask application for calculating Johnson's Relative Weights with multiple imputation methods.
"""

import os
import uuid
import threading
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, Any

from flask import Flask, request, render_template, redirect, url_for, send_file, jsonify
from werkzeug.utils import secure_filename

# Local modules
from weights_handler import WeightsCalculationHandler
from spss_handlers import (
    validate_spss_file_fast,
    read_spss_sample,
    SPSSReadError,
)
from file_handlers.repair_handler import SPSSFileRepairHandler


VERSION = "2.0"


def create_app() -> Flask:
    """Create and configure the Flask application."""
    app = Flask(__name__)

    base_dir = os.path.dirname(os.path.abspath(__file__))
    uploads_dir = os.path.join(base_dir, "temp")
    results_dir = os.path.join(base_dir, "results")
    os.makedirs(uploads_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)
    sample_path = os.path.join(base_dir, "sample_data.sav")

    # Configuration for deployment
    # Koyeb sets PORT env variable automatically
    app.config["APP_HOST"] = os.environ.get("APP_HOST", "0.0.0.0")
    app.config["APP_PORT"] = int(os.environ.get("PORT", os.environ.get("APP_PORT", "8000")))

    # Job management
    executor = ThreadPoolExecutor(max_workers=max(os.cpu_count() or 2, 2))
    jobs: Dict[str, Dict[str, Any]] = {}
    jobs_lock = threading.Lock()

    def register_job(status: str, payload: Dict[str, Any] = None) -> str:
        """Register a new job and return its ID."""
        job_id = uuid.uuid4().hex
        with jobs_lock:
            jobs[job_id] = {
                "status": status,
                "error": None,
                "result": None,
                "payload": payload or {},
            }
        return job_id

    def set_job_status(job_id: str, status: str, error: str = None, result: Dict[str, Any] = None) -> None:
        """Update job status."""
        with jobs_lock:
            job = jobs.get(job_id)
            if not job:
                return
            job["status"] = status
            job["error"] = error
            job["result"] = result

    # Routes
    @app.route("/")
    def index():
        """Main page with file upload form."""
        return render_template("index.html", version=VERSION, sample_available=os.path.exists(sample_path))

    @app.route("/health")
    def health():
        """Health check endpoint for Koyeb."""
        return jsonify({"status": "healthy", "version": VERSION})

    @app.route("/upload", methods=["POST"])
    def upload():
        """Handle file upload and validation."""
        if "file" not in request.files:
            return render_template("index.html", error="Файл не выбран", version=VERSION), 400
        
        f = request.files["file"]
        if not f.filename:
            return render_template("index.html", error="Файл не выбран", version=VERSION), 400
        if not f.filename.lower().endswith(".sav"):
            return render_template("index.html", error="Нужен SPSS (.sav) файл", version=VERSION), 400

        filename = secure_filename(f.filename)
        save_path = os.path.join(uploads_dir, filename)
        f.save(save_path)

        # Fast validation and optional repair
        is_valid, message, file_info = validate_spss_file_fast(save_path)
        needs_repair = bool(file_info and file_info.get("needs_repair", False))

        df = None
        meta = None
        try:
            if needs_repair or not is_valid:
                repair_handler = SPSSFileRepairHandler(save_path)
                df, meta, attempts = repair_handler.attempt_repair()
                if df is None:
                    return render_template(
                        "index.html",
                        error="Не удалось прочитать файл. Проверьте формат и кодировку.",
                        version=VERSION,
                    ), 400
            else:
                # Read sample to infer vars quickly
                df, meta = read_spss_sample(save_path, 500)
        except SPSSReadError as e:
            return render_template("index.html", error=e.get_user_message(), version=VERSION), 400
        except Exception as e:
            return render_template("index.html", error=str(e), version=VERSION), 400

        # Infer variables
        numeric_vars = df.select_dtypes(include=["int64", "float64", "int32", "float32"]).columns.tolist()
        all_vars = df.columns.tolist()

        # Filter zero-variance
        zero_var = [c for c in numeric_vars if df[c].nunique() <= 1]
        numeric_vars = [c for c in numeric_vars if c not in zero_var]

        return render_template(
            "select_vars.html",
            file_path=save_path,
            numeric_vars=numeric_vars,
            all_vars=all_vars,
            numeric_vars_csv=", ".join(numeric_vars),
            all_vars_csv=", ".join(all_vars),
            version=VERSION,
        )

    @app.route("/start", methods=["POST"])
    def start_job():
        """Start the calculation job."""
        file_path = request.form.get("file_path")
        dependent_vars = [v.strip() for v in (request.form.get("dependent_vars") or "").split(",") if v.strip()]
        independent_vars = [v.strip() for v in (request.form.get("independent_vars") or "").split(",") if v.strip()]
        subgroups_raw = request.form.get("subgroups") or ""
        subgroups = [v.strip() for v in subgroups_raw.split(",") if v.strip()] if subgroups_raw else None

        if not (file_path and os.path.exists(file_path)):
            return render_template("index.html", error="Файл не найден. Загрузите заново.", version=VERSION), 400
        
        if not dependent_vars or len(independent_vars) < 2:
            return render_template(
                "select_vars.html",
                file_path=file_path,
                numeric_vars=[],
                all_vars=[],
                error="Нужно выбрать минимум 1 зависимую и 2 независимых переменных",
                version=VERSION,
            ), 400

        handler = WeightsCalculationHandler(base_dir=os.path.dirname(file_path))

        # Pre-validate parameters
        is_valid, error_msg = handler.validate_analysis_parameters(
            input_file=file_path,
            dependent_vars=dependent_vars,
            independent_vars=independent_vars,
            subgroups=subgroups,
        )
        if not is_valid:
            # Re-render selection with error
            try:
                df, _ = read_spss_sample(file_path, 500)
                numeric_vars = df.select_dtypes(include=["int64", "float64", "int32", "float32"]).columns.tolist()
                all_vars = df.columns.tolist()
            except Exception:
                numeric_vars, all_vars = [], []
            return render_template(
                "select_vars.html",
                file_path=file_path,
                numeric_vars=numeric_vars,
                all_vars=all_vars,
                numeric_vars_csv=", ".join(numeric_vars),
                all_vars_csv=", ".join(all_vars),
                error=error_msg,
                version=VERSION,
            ), 400

        job_id = register_job("queued", {
            "file_path": file_path,
            "dependent_vars": dependent_vars,
            "independent_vars": independent_vars,
            "subgroups": subgroups,
        })

        def run_calc(jid: str):
            """Background calculation task."""
            set_job_status(jid, "running")
            try:
                analysis_type = "total" if not subgroups else "group"
                result = handler.calculate_weights(
                    input_file=file_path,
                    dependent_vars=dependent_vars,
                    independent_vars=independent_vars,
                    analysis_type=analysis_type,
                    subgroups=subgroups,
                )
                if result.get("status") == "success":
                    xlsx_path = result.get("results")
                    csv_path = xlsx_path.replace(".xlsx", ".csv") if xlsx_path else None
                    payload = {
                        "xlsx": xlsx_path if (xlsx_path and os.path.exists(xlsx_path)) else None,
                        "csv": csv_path if (csv_path and os.path.exists(csv_path)) else None,
                    }
                    set_job_status(jid, "completed", result=payload)
                else:
                    set_job_status(jid, "failed", error=result.get("message") or "Calculation failed")
            except Exception as e:
                set_job_status(jid, "failed", error=str(e))
            finally:
                # Cleanup uploaded file
                try:
                    if os.path.exists(file_path):
                        os.unlink(file_path)
                except Exception:
                    pass

        executor.submit(run_calc, job_id)
        return redirect(url_for("job_status_page", job_id=job_id))

    @app.route("/status/<job_id>")
    def job_status_page(job_id: str):
        """Show job status page."""
        with jobs_lock:
            job = jobs.get(job_id)
        if not job:
            return render_template("status.html", job_id=job_id, status="not_found", version=VERSION), 404
        return render_template("status.html", job_id=job_id, status=job["status"], error=job.get("error"), version=VERSION)

    @app.route("/api/status/<job_id>")
    def api_status(job_id: str):
        """API endpoint for job status (used by polling)."""
        with jobs_lock:
            job = jobs.get(job_id)
        if not job:
            return jsonify({"status": "not_found"}), 404
        data = {k: v for k, v in job.items() if k != "payload"}
        return jsonify(data)

    @app.route("/download/<job_id>/<fmt>")
    def download(job_id: str, fmt: str):
        """Download results file."""
        with jobs_lock:
            job = jobs.get(job_id)
        if not job:
            return "Job not found", 404
        if job["status"] != "completed" or not job.get("result"):
            return "Job not completed", 400
        path = job["result"].get(fmt)
        if not path or not os.path.exists(path):
            return "File not available", 404
        filename = os.path.basename(path)
        return send_file(path, as_attachment=True, download_name=filename)

    @app.route("/sample/download")
    def sample_download():
        """Download sample data file."""
        if not os.path.exists(sample_path):
            return "Sample file not found", 404
        return send_file(sample_path, as_attachment=True, download_name="sample_data.sav")

    @app.route("/upload-sample", methods=["GET"])
    def upload_sample():
        """Use sample data file for demo."""
        import shutil
        if not os.path.exists(sample_path):
            return render_template("index.html", error="Демо-файл не найден", version=VERSION, sample_available=False), 404
        
        # Copy sample into uploads
        dest = os.path.join(uploads_dir, "sample_data.sav")
        try:
            shutil.copyfile(sample_path, dest)
        except Exception as e:
            return render_template("index.html", error=str(e), version=VERSION, sample_available=True), 500

        # Fast validation and read
        is_valid, message, file_info = validate_spss_file_fast(dest)
        needs_repair = bool(file_info and file_info.get("needs_repair", False))

        try:
            if needs_repair or not is_valid:
                repair_handler = SPSSFileRepairHandler(dest)
                df, meta, attempts = repair_handler.attempt_repair()
                if df is None:
                    return render_template("index.html", error="Не удалось прочитать демо-файл", version=VERSION, sample_available=True), 400
            else:
                df, meta = read_spss_sample(dest, 500)
        except SPSSReadError as e:
            return render_template("index.html", error=e.get_user_message(), version=VERSION, sample_available=True), 400
        except Exception as e:
            return render_template("index.html", error=str(e), version=VERSION, sample_available=True), 400

        numeric_vars = df.select_dtypes(include=["int64", "float64", "int32", "float32"]).columns.tolist()
        zero_var = [c for c in numeric_vars if df[c].nunique() <= 1]
        numeric_vars = [c for c in numeric_vars if c not in zero_var]
        all_vars = df.columns.tolist()

        return render_template(
            "select_vars.html",
            file_path=dest,
            numeric_vars=numeric_vars,
            all_vars=all_vars,
            numeric_vars_csv=", ".join(numeric_vars),
            all_vars_csv=", ".join(all_vars),
            version=VERSION,
        )

    @app.route("/about")
    def about():
        """API info endpoint."""
        return jsonify({"name": "Johnson's Relative Weights", "version": VERSION})

    # Documentation routes
    def _render_markdown(md_text: str, title: str) -> str:
        """Render markdown to HTML with fallback."""
        try:
            import markdown as md
            html_body = md.markdown(md_text, extensions=["extra", "tables"], output_format="html5")
        except Exception:
            # Basic fallback
            import html
            html_body = f"<pre>{html.escape(md_text)}</pre>"
        
        return (
            "<!doctype html>\n<html lang=\"ru\">\n<head>\n"
            "<meta charset=\"utf-8\" />\n<meta name=\"viewport\" content=\"width=device-width, initial-scale=1\" />\n"
            f"<title>{title}</title>\n<link rel=\"stylesheet\" href=\"/static/style.css\" />\n"
            "</head><body>\n<div class=\"container\">\n<div class=\"card\">\n"
            "<div class=\"docs-nav\"><a class=\"btn\" href=\"/\">← На главную</a></div>\n"
            + html_body + "\n</div>\n</div>\n</body></html>"
        )

    @app.route("/docs/multiple-imputations")
    def docs_multiple_imputations():
        """Documentation: Multiple Imputations."""
        path = os.path.join(base_dir, "Multiple Imputations Readme.md")
        if not os.path.exists(path):
            return "File not found", 404
        with open(path, "r", encoding="utf-8") as f:
            md_text = f.read()
        return _render_markdown(md_text, title="О методах импутации")

    @app.route("/docs/readme")
    def docs_readme():
        """Documentation: README."""
        path = os.path.join(base_dir, "README.md")
        if not os.path.exists(path):
            return "File not found", 404
        with open(path, "r", encoding="utf-8") as f:
            md_text = f.read()
        return _render_markdown(md_text, title="О приложении")

    return app


# Application entry point
app = create_app()

if __name__ == "__main__":
    host = app.config.get("APP_HOST", "0.0.0.0")
    port = app.config.get("APP_PORT", 8000)
    print(f"Starting server on {host}:{port}")
    app.run(host=host, port=port, debug=False)
