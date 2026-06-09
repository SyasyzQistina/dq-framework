from __future__ import annotations
import os
import pathlib
from flask import Flask, render_template, request, redirect, url_for, send_from_directory
from werkzeug.utils import secure_filename
from src.profiling.pipeline import run
from src.reporting.generator import generate

app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = "datasets/uploads"
app.config["MAX_CONTENT_LENGTH"] = 50 * 1024 * 1024

ALLOWED_EXTENSIONS = {"csv", "xlsx", "json"}

os.makedirs("datasets/uploads", exist_ok=True)
os.makedirs("outputs/reports", exist_ok=True)


def allowed_file(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route("/", methods=["GET"])
def index():
    return render_template("index.html", error=None)


@app.route("/analyse", methods=["POST"])
def analyse():
    if "file" not in request.files:
        return render_template("index.html", error="No file selected.")
    file = request.files["file"]
    if file.filename == "":
        return render_template("index.html", error="No file selected.")
    if not allowed_file(file.filename):
        return render_template("index.html", error="Only CSV, Excel or JSON files allowed.")
    filename = secure_filename(file.filename)
    upload_path = pathlib.Path(app.config["UPLOAD_FOLDER"]) / filename
    file.save(str(upload_path))
    try:
        profile = run(str(upload_path))
        generate(profile, "outputs/reports")
        return redirect(url_for("result", name=profile.dataset_name))
    except Exception as e:
        return render_template("index.html", error=f"Error processing file: {str(e)}")


@app.route("/result/<name>")
def result(name):
    report_path = pathlib.Path("outputs/reports") / f"{name}_report.html"
    if not report_path.exists():
        return render_template("index.html", error="Report not found.")
    content = report_path.read_text(encoding="utf-8")
    return content


@app.route("/outputs/reports/charts/<filename>")
def charts(filename):
    return send_from_directory("outputs/reports/charts", filename)
@app.route("/download/<name>")
def download(name):
    return send_from_directory(
        "outputs/reports",
        f"{name}_report.html",
        as_attachment=True
    )

if __name__ == "__main__":
    app.run(debug=True)