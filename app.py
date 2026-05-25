import json
import math
import os
from pathlib import Path

from flask import Flask, flash, redirect, render_template, request, send_from_directory, url_for
from werkzeug.utils import secure_filename

from modules.document_export import export_to_docx
from modules.file_utils import (
    allowed_file,
    create_folders,
    generate_result_filename,
    generate_unique_filename,
)
from modules.image_preprocessing import preprocess_image
from modules.ocr_model import OCRModel
from modules.text_postprocessing import TextPostProcessor, is_garbage_ocr


BASE_DIR = Path(__file__).resolve().parent
UPLOAD_FOLDER = "uploads"
PROCESSED_FOLDER = "processed"
RESULTS_FOLDER = "results"
ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg"}
PROCESSED_DIR = BASE_DIR / PROCESSED_FOLDER


app = Flask(__name__)
app.config["SECRET_KEY"] = os.environ.get("FLASK_SECRET_KEY", "change-this-secret-key")
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["PROCESSED_FOLDER"] = PROCESSED_FOLDER
app.config["RESULTS_FOLDER"] = RESULTS_FOLDER
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024
app.config["OCR_DEBUG"] = os.environ.get("OCR_DEBUG", "1").lower() not in {"0", "false", "no"}

create_folders(BASE_DIR, [UPLOAD_FOLDER, PROCESSED_FOLDER, RESULTS_FOLDER])

# Loading here keeps one model instance for all requests handled by this process.
ocr_model = OCRModel()
text_postprocessor = TextPostProcessor()


def _processed_relative_path(file_path):
    if not file_path:
        return None

    return Path(file_path).resolve().relative_to(PROCESSED_DIR.resolve()).as_posix()


def _recognize_and_render(original_filename, processing_result):
    recognized_text = ocr_model.recognize_groups(processing_result["ocr_groups"])
    corrected_text = (
        recognized_text
        if is_garbage_ocr(recognized_text)
        else text_postprocessor.correct(recognized_text)
    )
    line_items = [
        {
            "filename": _processed_relative_path(line["line_path"]),
            "fragment_filenames": [
                _processed_relative_path(fragment_path)
                for fragment_path in line["fragment_paths"]
            ],
        }
        for line in processing_result["lines"]
    ]

    return render_template(
        "result.html",
        original_filename=original_filename,
        cropped_filename=_processed_relative_path(processing_result["cropped_path"]),
        ocr_ready_filename=_processed_relative_path(processing_result["ocr_ready_path"]),
        line_items=line_items,
        mask_filename=_processed_relative_path(processing_result["mask_path"]),
        annotated_filename=_processed_relative_path(processing_result["annotated_path"]),
        scale_factor=processing_result["scale_factor"],
        segmentation_mode=processing_result["segmentation_mode"],
        manual_cut_positions=processing_result["manual_cut_positions"],
        recognized_text=recognized_text,
        corrected_text=corrected_text,
    )


def _get_existing_upload(filename):
    secure_name = secure_filename(filename)

    if secure_name != filename or not allowed_file(secure_name, ALLOWED_EXTENSIONS):
        raise ValueError("Не удалось открыть исходное изображение.")

    image_path = BASE_DIR / UPLOAD_FOLDER / secure_name

    if not image_path.is_file():
        raise ValueError("Исходное изображение больше не найдено.")

    return image_path


def _parse_manual_cuts(raw_value):
    try:
        values = json.loads(raw_value or "[]")
    except json.JSONDecodeError as exc:
        raise ValueError("Некорректные границы строк.") from exc

    if not isinstance(values, list) or len(values) > 50:
        raise ValueError("Некорректные границы строк.")

    positions = []
    for value in values:
        try:
            position = float(value)
        except (TypeError, ValueError) as exc:
            raise ValueError("Некорректные границы строк.") from exc

        if math.isfinite(position) and 0.01 < position < 0.99:
            positions.append(round(position, 6))

    return sorted(set(positions))


@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")


@app.route("/recognize", methods=["POST"])
def recognize():
    file = request.files.get("image")

    if file is None or file.filename == "":
        flash("Выберите изображение для распознавания.", "error")
        return redirect(url_for("index"))

    if not allowed_file(file.filename, ALLOWED_EXTENSIONS):
        flash("Неподдерживаемый формат файла. Загрузите PNG, JPG или JPEG.", "error")
        return redirect(url_for("index"))

    original_secure_name = secure_filename(file.filename)
    if not original_secure_name or not Path(original_secure_name).suffix:
        original_secure_name = f"image{Path(file.filename).suffix.lower()}"

    unique_filename = generate_unique_filename(original_secure_name)
    upload_path = BASE_DIR / UPLOAD_FOLDER / unique_filename

    try:
        file.save(upload_path)
        processing_result = preprocess_image(
            str(upload_path),
            str(PROCESSED_DIR),
            debug=app.config["OCR_DEBUG"],
        )
        return _recognize_and_render(unique_filename, processing_result)
    except Exception as exc:
        flash(f"Ошибка распознавания: {exc}", "error")
        return redirect(url_for("index"))


@app.route("/recognize/manual", methods=["POST"])
def recognize_manual():
    try:
        original_filename = request.form.get("original_filename", "")
        image_path = _get_existing_upload(original_filename)
        manual_cuts = _parse_manual_cuts(request.form.get("manual_cuts", "[]"))
        processing_result = preprocess_image(
            str(image_path),
            str(PROCESSED_DIR),
            debug=app.config["OCR_DEBUG"],
            manual_cuts=manual_cuts,
        )
        return _recognize_and_render(original_filename, processing_result)
    except Exception as exc:
        flash(f"Ошибка ручного распознавания: {exc}", "error")
        return redirect(url_for("index"))


@app.route("/export", methods=["POST"])
def export():
    text = request.form.get("recognized_text", "").strip()

    if not text:
        flash("Текст для экспорта пустой. Добавьте или исправьте текст перед скачиванием.", "error")
        return redirect(url_for("index"))

    result_filename = generate_result_filename(".docx")
    output_path = BASE_DIR / RESULTS_FOLDER / result_filename

    try:
        export_to_docx(text, str(output_path))
    except Exception as exc:
        flash(f"Ошибка экспорта DOCX: {exc}", "error")
        return redirect(url_for("index"))

    return send_from_directory(
        BASE_DIR / RESULTS_FOLDER,
        result_filename,
        as_attachment=True,
        download_name=result_filename,
    )


@app.route("/uploads/<path:filename>")
def uploaded_file(filename):
    return send_from_directory(BASE_DIR / UPLOAD_FOLDER, filename)


@app.route("/processed/<path:filename>")
def processed_file(filename):
    return send_from_directory(PROCESSED_DIR, filename)


@app.errorhandler(413)
def request_entity_too_large(_error):
    flash("Файл слишком большой. Максимальный размер: 16 МБ.", "error")
    return redirect(url_for("index"))


if __name__ == "__main__":
    app_host = os.environ.get("FLASK_HOST", "127.0.0.1")
    app_port = int(os.environ.get("PORT", "5000"))
    debug_enabled = os.environ.get("FLASK_DEBUG", "1").lower() not in {"0", "false", "no"}
    app.run(host=app_host, port=app_port, debug=debug_enabled, use_reloader=False)
