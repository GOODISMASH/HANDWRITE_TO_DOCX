import os

import torch
from PIL import Image, UnidentifiedImageError
from transformers import TrOCRProcessor, VisionEncoderDecoderModel


DEFAULT_MODEL_NAME = "kazars24/trocr-base-handwritten-ru"


class OCRModel:
    def __init__(self, model_name=None):
        self.model_name = model_name or os.environ.get("OCR_MODEL_NAME", DEFAULT_MODEL_NAME)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        try:
            self.batch_size = max(1, int(os.environ.get("OCR_BATCH_SIZE", "4")))
        except ValueError:
            self.batch_size = 4
        try:
            self.num_beams = max(1, int(os.environ.get("OCR_NUM_BEAMS", "1")))
        except ValueError:
            self.num_beams = 1

        self.processor = TrOCRProcessor.from_pretrained(self.model_name)
        self.model = VisionEncoderDecoderModel.from_pretrained(self.model_name)
        self.model.to(self.device)
        self.model.eval()

    def recognize(self, image_path):
        return self.recognize_batch([image_path])[0]

    def recognize_batch(self, image_paths):
        try:
            texts = []

            for start in range(0, len(image_paths), self.batch_size):
                chunk_paths = image_paths[start : start + self.batch_size]
                images = []

                try:
                    for image_path in chunk_paths:
                        with Image.open(image_path) as image:
                            images.append(image.convert("RGB"))

                    pixel_values = self.processor(images=images, return_tensors="pt").pixel_values.to(self.device)

                    generation_options = {
                        "max_length": 128,
                        "num_beams": self.num_beams,
                    }
                    if self.num_beams > 1:
                        generation_options["early_stopping"] = True

                    with torch.no_grad():
                        generated_ids = self.model.generate(pixel_values, **generation_options)

                    decoded = self.processor.batch_decode(generated_ids, skip_special_tokens=True)
                    texts.extend(text.strip() for text in decoded)
                finally:
                    for image in images:
                        image.close()

            return texts
        except (UnidentifiedImageError, OSError) as exc:
            raise ValueError("Не удалось открыть фрагмент изображения для OCR.") from exc
        except RuntimeError as exc:
            raise RuntimeError("Ошибка выполнения OCR-модели.") from exc
        except Exception as exc:
            raise RuntimeError("Не удалось распознать фрагменты текста.") from exc

    def recognize_groups(self, line_groups):
        flat_paths = [fragment_path for group in line_groups for fragment_path in group]
        texts = self.recognize_batch(flat_paths)
        recognized_lines = []
        offset = 0

        for fragment_paths in line_groups:
            line_texts = texts[offset : offset + len(fragment_paths)]
            line_parts = [text for text in line_texts if text]
            offset += len(fragment_paths)

            if line_parts:
                recognized_lines.append(" ".join(line_parts))

        return "\n".join(recognized_lines)

    def recognize_lines(self, line_paths):
        return self.recognize_groups([[line_path] for line_path in line_paths])
