#!/usr/bin/env python3
"""Detect table regions in document images and export debug artifacts."""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np


def _ensure_dir(path: Path) -> Path:
    # Create output folders without failing if they already exist.
    path.mkdir(parents=True, exist_ok=True)
    return path


def _load_image(image_path: Path) -> np.ndarray:
    # Load the image in BGR format (OpenCV default).
    image = cv2.imread(str(image_path))
    if image is None:
        raise SystemExit(f"Failed to read image: {image_path}")
    return image


def _adaptive_threshold(gray: np.ndarray, block_size: int, c_value: int) -> np.ndarray:
    # Adaptive thresholding handles uneven lighting and document scans.
    return cv2.adaptiveThreshold(
        gray,
        255,
        cv2.ADAPTIVE_THRESH_MEAN_C,
        cv2.THRESH_BINARY_INV,
        block_size,
        c_value,
    )


def _extract_lines(binary: np.ndarray, kernel_scale: int) -> tuple[np.ndarray, np.ndarray]:
    height, width = binary.shape
    # Scale kernels based on image size for better generalization.
    horiz_len = max(10, width // kernel_scale)
    vert_len = max(10, height // kernel_scale)

    # Use rectangular kernels to isolate horizontal and vertical lines.
    horiz_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (horiz_len, 1))
    vert_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, vert_len))

    horizontal = cv2.morphologyEx(binary, cv2.MORPH_OPEN, horiz_kernel, iterations=1)
    vertical = cv2.morphologyEx(binary, cv2.MORPH_OPEN, vert_kernel, iterations=1)
    return horizontal, vertical


def _combine_lines(horizontal: np.ndarray, vertical: np.ndarray, dilate_iters: int) -> np.ndarray:
    # Combine line maps and slightly dilate to close gaps.
    combined = cv2.addWeighted(horizontal, 0.5, vertical, 0.5, 0.0)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    combined = cv2.dilate(combined, kernel, iterations=dilate_iters)
    return combined


def _find_table_boxes(
    mask: np.ndarray,
    min_area: int,
    min_width: int,
    min_height: int,
    min_aspect: float,
    max_aspect: float,
) -> list[dict]:
    # Find outer contours that resemble table boundaries.
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boxes = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if area < min_area:
            continue
        x, y, w, h = cv2.boundingRect(contour)
        if w < min_width or h < min_height:
            continue
        aspect = w / float(h)
        if aspect < min_aspect or aspect > max_aspect:
            continue
        boxes.append(
            {
                "x": int(x),
                "y": int(y),
                "w": int(w),
                "h": int(h),
                "area": float(area),
                "aspect": float(aspect),
            }
        )
    boxes.sort(key=lambda b: b["area"], reverse=True)
    return boxes


def _draw_boxes(image: np.ndarray, boxes: list[dict]) -> np.ndarray:
    overlay = image.copy()
    # Draw labeled rectangles on top of the original image.
    for idx, box in enumerate(boxes, start=1):
        x, y, w, h = box["x"], box["y"], box["w"], box["h"]
        cv2.rectangle(overlay, (x, y), (x + w, y + h), (0, 153, 255), 2)
        cv2.putText(
            overlay,
            f"table_{idx}",
            (x, max(0, y - 6)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 153, 255),
            1,
            cv2.LINE_AA,
        )
    return overlay


def _save_debug_images(
    out_dir: Path,
    gray: np.ndarray,
    binary: np.ndarray,
    horizontal: np.ndarray,
    vertical: np.ndarray,
    mask: np.ndarray,
    contour_vis: np.ndarray,
    overlay: np.ndarray,
) -> None:
    # Write intermediate outputs to inspect the pipeline visually.
    cv2.imwrite(str(out_dir / "01_gray.png"), gray)
    cv2.imwrite(str(out_dir / "02_thresh.png"), binary)
    cv2.imwrite(str(out_dir / "03_horizontal.png"), horizontal)
    cv2.imwrite(str(out_dir / "04_vertical.png"), vertical)
    cv2.imwrite(str(out_dir / "05_mask.png"), mask)
    cv2.imwrite(str(out_dir / "06_contours.png"), contour_vis)
    cv2.imwrite(str(out_dir / "07_overlay.png"), overlay)


def main() -> None:
    parser = argparse.ArgumentParser(description="Detect tables in a document image.")
    parser.add_argument("--image", required=True, help="Path to input image (PNG/JPG).")
    parser.add_argument("--out-dir", default="outputs", help="Output directory.")
    parser.add_argument("--block-size", type=int, default=25, help="Adaptive threshold block size (odd).")
    parser.add_argument("--c-value", type=int, default=15, help="Adaptive threshold C value.")
    parser.add_argument("--kernel-scale", type=int, default=30, help="Kernel scale for line extraction.")
    parser.add_argument("--dilate-iters", type=int, default=2, help="Dilate iterations for line mask.")
    parser.add_argument("--min-area-ratio", type=float, default=0.01, help="Min area as ratio of image area.")
    parser.add_argument("--min-width", type=int, default=120, help="Min table width in pixels.")
    parser.add_argument("--min-height", type=int, default=80, help="Min table height in pixels.")
    parser.add_argument("--min-aspect", type=float, default=0.3, help="Min aspect ratio (w/h).")
    parser.add_argument("--max-aspect", type=float, default=6.0, help="Max aspect ratio (w/h).")
    args = parser.parse_args()

    if args.block_size % 2 == 0:
        raise SystemExit("--block-size must be odd")

    image_path = Path(args.image)
    out_dir = _ensure_dir(Path(args.out_dir))

    # Load + preprocess.
    image = _load_image(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    binary = _adaptive_threshold(gray, args.block_size, args.c_value)

    # Line extraction + mask creation.
    horizontal, vertical = _extract_lines(binary, args.kernel_scale)
    mask = _combine_lines(horizontal, vertical, args.dilate_iters)

    img_area = image.shape[0] * image.shape[1]
    # Enforce a minimum box size based on the image area.
    min_area = max(int(img_area * args.min_area_ratio), 500)
    boxes = _find_table_boxes(
        mask,
        min_area=min_area,
        min_width=args.min_width,
        min_height=args.min_height,
        min_aspect=args.min_aspect,
        max_aspect=args.max_aspect,
    )

    # Visualize boxes before labeling.
    contour_vis = image.copy()
    for box in boxes:
        x, y, w, h = box["x"], box["y"], box["w"], box["h"]
        cv2.rectangle(contour_vis, (x, y), (x + w, y + h), (0, 255, 0), 2)

    overlay = _draw_boxes(image, boxes)

    _save_debug_images(out_dir, gray, binary, horizontal, vertical, mask, contour_vis, overlay)

    json_path = out_dir / "tables.json"
    with json_path.open("w", encoding="utf-8") as f:
        json.dump({"image": str(image_path), "tables": boxes}, f, indent=2)

    print("Table detection complete.")
    print(f"- Debug images: {out_dir}")
    print(f"- Boxes JSON: {json_path}")
    print(f"- Tables found: {len(boxes)}")


if __name__ == "__main__":
    main()
