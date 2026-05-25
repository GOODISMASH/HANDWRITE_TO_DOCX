from pathlib import Path
from uuid import uuid4


def allowed_file(filename, allowed_extensions):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in allowed_extensions


def create_folders(base_dir=None, folders=None):
    root = Path(base_dir) if base_dir else Path.cwd()
    target_folders = folders or ["uploads", "processed", "results"]

    for folder in target_folders:
        (root / folder).mkdir(parents=True, exist_ok=True)


def generate_unique_filename(original_filename):
    suffix = Path(original_filename).suffix.lower()
    stem = Path(original_filename).stem or "image"
    return f"{stem}_{uuid4().hex}{suffix}"


def generate_result_filename(extension=".docx"):
    normalized_extension = extension if extension.startswith(".") else f".{extension}"
    return f"recognized_text_{uuid4().hex}{normalized_extension}"
