from pathlib import Path
from uuid import uuid4

import cv2
import numpy as np


def ensure_dir(path):
    Path(path).mkdir(parents=True, exist_ok=True)


def _save_image(path, image):
    ensure_dir(Path(path).parent)

    if not cv2.imwrite(str(path), image):
        raise ValueError("Не удалось сохранить обработанное изображение.")


def _load_image_bgr(input_path):
    image = cv2.imread(input_path)

    if image is None:
        raise ValueError("Не удалось загрузить изображение.")

    return image


def _upscale_small_image(image):
    height, width = image.shape[:2]

    if width >= 1000:
        return image, 1.0

    scale_factor = 4.0 if width < 600 else 3.0
    enlarged = cv2.resize(
        image,
        (int(width * scale_factor), int(height * scale_factor)),
        interpolation=cv2.INTER_CUBIC,
    )
    return enlarged, scale_factor


def _remove_small_components(mask):
    component_count, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    cleaned = np.zeros_like(mask)
    min_area = max(12, min(40, int(mask.shape[0] * mask.shape[1] * 0.000003)))

    for label in range(1, component_count):
        area = stats[label, cv2.CC_STAT_AREA]
        width = stats[label, cv2.CC_STAT_WIDTH]
        height = stats[label, cv2.CC_STAT_HEIGHT]

        if area >= min_area and width > 1 and height > 1:
            cleaned[labels == label] = 255

    return cleaned


def _grid_color_candidates(image_bgr):
    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    hue, saturation, value = cv2.split(hsv)

    colored_grid = (
        (hue >= 70)
        & (hue <= 140)
        & (saturation >= 5)
        & (value >= 185)
    )
    saturated_turquoise_grid = (
        (hue >= 75)
        & (hue <= 105)
        & (saturation >= 35)
        & (value >= 80)
    )
    return ((colored_grid | saturated_turquoise_grid).astype(np.uint8)) * 255


def _build_grid_mask(image_bgr):
    """
    Locate bright colored notebook lines. Ink may share a hue with the grid,
    therefore brightness and long-line geometry protect dark pen strokes.
    """
    candidates = _grid_color_candidates(image_bgr)

    # Retain only long axes so bright anti-aliased edges of blue handwriting
    # are not whitened together with the notebook grid.
    horizontal = cv2.morphologyEx(
        candidates,
        cv2.MORPH_OPEN,
        cv2.getStructuringElement(cv2.MORPH_RECT, (max(35, image_bgr.shape[1] // 12), 1)),
    )
    vertical = cv2.morphologyEx(
        candidates,
        cv2.MORPH_OPEN,
        cv2.getStructuringElement(cv2.MORPH_RECT, (1, max(35, image_bgr.shape[0] // 12))),
    )
    grid_mask = cv2.bitwise_or(horizontal, vertical)
    return cv2.dilate(grid_mask, cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2)))


def _remove_axis_lines(mask):
    """Discard remaining long straight grid strokes from a detection mask."""
    horizontal_length = max(50, mask.shape[1] // 5)
    vertical_length = max(40, mask.shape[0] // 5)
    horizontal = cv2.morphologyEx(
        mask,
        cv2.MORPH_OPEN,
        cv2.getStructuringElement(cv2.MORPH_RECT, (horizontal_length, 1)),
    )
    vertical = cv2.morphologyEx(
        mask,
        cv2.MORPH_OPEN,
        cv2.getStructuringElement(cv2.MORPH_RECT, (1, vertical_length)),
    )
    lines = cv2.bitwise_or(horizontal, vertical)
    lines = cv2.dilate(lines, cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2)))
    return cv2.subtract(mask, lines)


def build_ink_mask(image_bgr):
    """
    Build a mask of handwriting only: text = 255, page and grid = 0.

    A plain grayscale/Otsu threshold is intentionally not used for segmentation,
    because it can classify squared-paper lines as handwriting.
    """
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    grid_mask = _build_grid_mask(image_bgr)
    dark_ink = (gray < 165).astype(np.uint8) * 255
    ink_mask = cv2.bitwise_and(dark_ink, cv2.bitwise_not(grid_mask))
    ink_mask = cv2.morphologyEx(
        ink_mask,
        cv2.MORPH_OPEN,
        cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2)),
        iterations=1,
    )
    ink_mask = cv2.dilate(
        ink_mask,
        cv2.getStructuringElement(cv2.MORPH_RECT, (2, 1)),
        iterations=1,
    )
    return _remove_small_components(_remove_axis_lines(ink_mask))


def remove_grid_for_ocr(image_bgr):
    """Whiten detected colored grid pixels while preserving dark handwriting."""
    result = image_bgr.copy()
    grid_candidates = _build_grid_mask(image_bgr)
    protected_ink = cv2.dilate(
        build_ink_mask(image_bgr),
        cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)),
        iterations=1,
    )
    removable_grid = cv2.bitwise_and(grid_candidates, cv2.bitwise_not(protected_ink))
    result[removable_grid > 0] = [255, 255, 255]
    return result


def is_valid_text_fragment(mask_fragment):
    """Return True only when a mask fragment contains plausible handwriting."""
    height, width = mask_fragment.shape[:2]

    if height < 18 or width < 30:
        return False

    ink_pixels = cv2.countNonZero(mask_fragment)
    if ink_pixels < 80:
        return False

    density = ink_pixels / (height * width)
    if density < 0.003:
        return False

    component_count, _, stats, _ = cv2.connectedComponentsWithStats(mask_fragment, connectivity=8)
    valid_components = 0

    for label in range(1, component_count):
        _, _, component_width, component_height, area = stats[label]

        if area < 10:
            continue
        if component_width > component_height * 12 and component_height <= 4:
            continue
        if component_height > component_width * 12 and component_width <= 4:
            continue

        valid_components += 1

    return valid_components >= 1


def _find_content_bbox(mask, margin=35):
    connection_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 5))
    expanded = cv2.dilate(mask, connection_kernel, iterations=1)
    coordinates = cv2.findNonZero(expanded)

    if coordinates is None:
        return None

    x, y, width, height = cv2.boundingRect(coordinates)
    image_height, image_width = mask.shape[:2]
    return (
        max(x - margin, 0),
        max(y - margin, 0),
        min(x + width + margin, image_width),
        min(y + height + margin, image_height),
    )


def _smooth_projection(values, window_size=13):
    window_size = min(window_size, len(values))
    if window_size <= 1:
        return values.astype(float)
    kernel = np.ones(window_size, dtype=float) / window_size
    return np.convolve(values.astype(float), kernel, mode="same")


def _cluster_valleys(candidates, minimum_distance):
    clusters = []

    for candidate in sorted(candidates, key=lambda item: item[0]):
        if not clusters or candidate[0] - clusters[-1][-1][0] > minimum_distance:
            clusters.append([candidate])
        else:
            clusters[-1].append(candidate)

    return [min(cluster, key=lambda item: item[1])[0] for cluster in clusters]


def _find_line_ranges(mask):
    """
    Split at projection valleys. Descenders and notebook residue can cross a
    line gap, so the splitter does not require fully empty horizontal rows.
    """
    row_projection = np.count_nonzero(mask, axis=1)
    active_rows = np.where(row_projection > 0)[0]

    if len(active_rows) == 0:
        return []

    top = int(active_rows[0])
    bottom = int(active_rows[-1])
    smooth = _smooth_projection(row_projection, window_size=13)
    search_window = max(24, min(50, (bottom - top) // 8))
    candidates = []

    for row in range(top + 8, bottom - 8):
        if smooth[row] > smooth[row - 1] or smooth[row] > smooth[row + 1]:
            continue

        left_peak = smooth[max(top, row - search_window) : row].max(initial=0)
        right_peak = smooth[row + 1 : min(bottom + 1, row + search_window + 1)].max(initial=0)
        lower_peak = min(left_peak, right_peak)

        if lower_peak > 0 and smooth[row] / lower_peak <= 0.45:
            candidates.append((row, smooth[row] / lower_peak))

    component_count, _, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    heights = [
        stats[label, cv2.CC_STAT_HEIGHT]
        for label in range(1, component_count)
        if stats[label, cv2.CC_STAT_AREA] >= 12
    ]
    median_height = float(np.median(heights)) if heights else 20.0
    minimum_distance = max(20, int(median_height * 2.0))
    separators = _cluster_valleys(candidates, minimum_distance)

    boundaries = [top] + separators + [bottom + 1]
    ranges = []
    padding = max(8, int(median_height * 0.7))

    for index in range(len(boundaries) - 1):
        start = max(boundaries[index] - (padding if index == 0 else 0), 0)
        end = min(boundaries[index + 1] + (padding if index == len(boundaries) - 2 else 0), mask.shape[0])
        if end - start >= max(10, int(median_height * 0.6)):
            ranges.append((start, end))

    return ranges


def _get_horizontal_extent(mask, padding=25):
    columns = np.where(np.count_nonzero(mask, axis=0) > 0)[0]

    if len(columns) == 0:
        return None

    return max(int(columns[0]) - padding, 0), min(int(columns[-1]) + padding + 1, mask.shape[1])


def _group_active_columns(mask, max_gap):
    active_columns = np.where(np.count_nonzero(mask, axis=0) > 0)[0]

    if len(active_columns) == 0:
        return []

    groups = []
    start = previous = int(active_columns[0])

    for column in active_columns[1:]:
        column = int(column)
        if column - previous > max_gap:
            groups.append((start, previous + 1))
            start = column
        previous = column

    groups.append((start, previous + 1))
    return groups


def _split_wide_line(line_mask, max_ratio=10.0):
    height, width = line_mask.shape[:2]

    if height == 0 or width / height <= max_ratio:
        return [(0, width)]

    word_gap = max(16, int(height * 0.24))
    word_groups = _group_active_columns(line_mask, max_gap=word_gap)

    if len(word_groups) < 2:
        return [(0, width)]

    fragments = []
    start, end = word_groups[0]

    for next_start, next_end in word_groups[1:]:
        combined_ratio = (next_end - start) / height
        if combined_ratio > max_ratio and end > start:
            fragments.append((start, end))
            start, end = next_start, next_end
        else:
            end = next_end

    fragments.append((start, end))
    return fragments


def _create_line_item(cropped_ocr_bgr, cropped_mask, output_dir, base_name, line_number, y1, y2):
    line_mask = cropped_mask[y1:y2, :]
    horizontal_extent = _get_horizontal_extent(line_mask)

    if horizontal_extent is None:
        return None

    x1, x2 = horizontal_extent
    line_image = cropped_ocr_bgr[y1:y2, x1:x2]
    line_mask = line_mask[:, x1:x2]

    if not is_valid_text_fragment(line_mask):
        return None

    line_path = Path(output_dir) / f"{base_name}_line_{line_number}.png"
    _save_image(line_path, line_image)
    fragment_paths = []
    fragment_boxes = []

    for part_number, (part_x1, part_x2) in enumerate(_split_wide_line(line_mask), start=1):
        padding = max(12, int(line_image.shape[0] * 0.18))
        padded_x1 = max(part_x1 - padding, 0)
        padded_x2 = min(part_x2 + padding, line_image.shape[1])
        fragment_mask = line_mask[:, padded_x1:padded_x2]

        if not is_valid_text_fragment(fragment_mask):
            continue

        fragment = line_image[:, padded_x1:padded_x2]
        fragment_path = Path(output_dir) / f"{base_name}_line_{line_number}_part_{part_number}.png"
        _save_image(fragment_path, fragment)
        fragment_paths.append(str(fragment_path))
        fragment_boxes.append((x1 + padded_x1, y1, x1 + padded_x2, y2))

    if not fragment_paths:
        fragment_paths = [str(line_path)]
        fragment_boxes = [(x1, y1, x2, y2)]

    return {
        "line_path": str(line_path),
        "fragment_paths": fragment_paths,
        "line_box": (x1, y1, x2, y2),
        "fragment_boxes": fragment_boxes,
    }


def _extract_lines(cropped_ocr_bgr, cropped_mask, output_dir, base_name):
    ensure_dir(output_dir)
    lines = []

    for line_number, (y1, y2) in enumerate(_find_line_ranges(cropped_mask), start=1):
        line = _create_line_item(
            cropped_ocr_bgr,
            cropped_mask,
            output_dir,
            base_name,
            line_number,
            y1,
            y2,
        )
        if line:
            lines.append(line)

    return lines


def _extract_manual_lines(cropped_ocr_bgr, cropped_mask, output_dir, base_name, cut_positions):
    """Build text rows from user-provided horizontal separators in normalized Y units."""
    ensure_dir(output_dir)
    image_height = cropped_mask.shape[0]
    cut_rows = sorted(
        {
            min(max(int(round(float(position) * image_height)), 1), image_height - 1)
            for position in cut_positions
            if 0 < float(position) < 1
        }
    )
    boundaries = [0] + cut_rows + [image_height]
    lines = []
    padding = max(8, int(image_height * 0.012))

    for start, end in zip(boundaries, boundaries[1:]):
        band_mask = cropped_mask[start:end, :]
        active_rows = np.where(np.count_nonzero(band_mask, axis=1) > 0)[0]

        if len(active_rows) == 0:
            continue

        y1 = max(start + int(active_rows[0]) - padding, start)
        y2 = min(start + int(active_rows[-1]) + padding + 1, end)
        line = _create_line_item(
            cropped_ocr_bgr,
            cropped_mask,
            output_dir,
            base_name,
            len(lines) + 1,
            y1,
            y2,
        )
        if line:
            lines.append(line)

    return lines


def _create_annotated_preview(cropped_bgr, lines):
    annotated = cropped_bgr.copy()

    for line_number, line in enumerate(lines, start=1):
        x1, y1, x2, y2 = line["line_box"]
        cv2.rectangle(annotated, (x1, y1), (x2, y2), (54, 201, 156), 2)
        cv2.putText(
            annotated,
            str(line_number),
            (x1 + 4, max(18, y1 + 20)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (54, 201, 156),
            2,
            cv2.LINE_AA,
        )

        if len(line["fragment_boxes"]) > 1:
            for fragment_box in line["fragment_boxes"]:
                fx1, fy1, fx2, fy2 = fragment_box
                cv2.rectangle(annotated, (fx1, fy1), (fx2, fy2), (61, 149, 255), 2)

    return annotated


def preprocess_image(input_path, processed_folder, debug=True, manual_cuts=None):
    """
    Enlarge a small page, find text and save unbinarized RGB-compatible crops.

    Thresholded masks are diagnostic/detection artifacts only and are never
    supplied to the OCR model.
    """
    ensure_dir(processed_folder)
    image = _load_image_bgr(input_path)
    working_image, scale_factor = _upscale_small_image(image)
    base_name = Path(input_path).stem
    processed_path = Path(processed_folder)

    scaled_path = processed_path / f"{base_name}_scaled.png"
    _save_image(scaled_path, working_image)

    ink_mask = build_ink_mask(working_image)
    ocr_image = remove_grid_for_ocr(working_image)
    bounding_box = _find_content_bbox(ink_mask, margin=max(25, int(35 * scale_factor)))

    if bounding_box is None:
        raise ValueError("На изображении не удалось найти текст.")

    x1, y1, x2, y2 = bounding_box
    cropped = working_image[y1:y2, x1:x2]
    cropped_ocr = ocr_image[y1:y2, x1:x2]
    cropped_mask = ink_mask[y1:y2, x1:x2]

    cropped_path = processed_path / f"{base_name}_cropped.png"
    _save_image(cropped_path, cropped)
    ocr_ready_path = processed_path / f"{base_name}_ocr_ready.png"
    _save_image(ocr_ready_path, cropped_ocr)

    if manual_cuts is None:
        artifact_base_name = base_name
        lines_folder = processed_path / f"{base_name}_lines"
        lines = _extract_lines(cropped_ocr, cropped_mask, lines_folder, artifact_base_name)
        segmentation_mode = "auto"
    else:
        run_id = uuid4().hex[:10]
        artifact_base_name = f"{base_name}_manual_{run_id}"
        lines_folder = processed_path / f"{artifact_base_name}_lines"
        lines = _extract_manual_lines(
            cropped_ocr,
            cropped_mask,
            lines_folder,
            artifact_base_name,
            manual_cuts,
        )
        segmentation_mode = "manual"

    if not lines:
        raise ValueError("Не удалось выделить отдельные строки текста.")

    mask_path = None
    annotated_path = None

    if debug:
        mask_path = processed_path / f"{base_name}_ink_mask.png"
        _save_image(mask_path, cropped_mask)
        annotated_path = processed_path / f"{artifact_base_name}_detected_lines.png"
        _save_image(annotated_path, _create_annotated_preview(cropped_ocr, lines))

    if manual_cuts is None:
        manual_cut_positions = [
            (lines[index]["line_box"][3] + lines[index + 1]["line_box"][1])
            / (2 * cropped_mask.shape[0])
            for index in range(len(lines) - 1)
        ]
    else:
        manual_cut_positions = sorted(
            float(position) for position in manual_cuts if 0 < float(position) < 1
        )

    return {
        "scaled_path": str(scaled_path),
        "cropped_path": str(cropped_path),
        "ocr_ready_path": str(ocr_ready_path),
        "mask_path": str(mask_path) if mask_path else None,
        "annotated_path": str(annotated_path) if annotated_path else None,
        "lines": lines,
        "ocr_groups": [line["fragment_paths"] for line in lines],
        "scale_factor": scale_factor,
        "manual_cut_positions": manual_cut_positions,
        "segmentation_mode": segmentation_mode,
    }
