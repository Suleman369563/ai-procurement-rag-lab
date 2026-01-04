#!/usr/bin/env python3
"""Generate synthetic CSV, image, and PDF artifacts for the lab."""
import argparse
import csv
import random
from datetime import date, timedelta
from pathlib import Path

import cv2
import numpy as np
from reportlab.lib.pagesizes import letter
from reportlab.lib.utils import ImageReader
from reportlab.pdfgen import canvas


# Ensure output directories exist before writing files.
def _ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def _generate_csv(out_path: Path, num_records: int, seed: int) -> None:
    """Generate a synthetic procurement CSV dataset."""
    rng = random.Random(seed)
    # Controlled vocab to keep data realistic and repeatable.
    vendors = [
        "Nordic Supply Co",
        "Alpha Logistics",
        "Greenfield Industrials",
        "Rhein Components",
        "Helios Parts",
        "Bergmann Tools",
    ]
    categories = ["Fasteners", "Electrical", "Packaging", "Safety", "Mechanical"]
    items = [
        "M8 Bolts",
        "Copper Wire",
        "Cardboard Boxes",
        "Safety Gloves",
        "Hydraulic Hose",
        "Steel Brackets",
        "Circuit Breaker",
        "Shrink Wrap",
    ]
    statuses = ["requested", "approved", "ordered", "delivered", "delayed"]
    base_date = date(2024, 1, 1)

    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        # CSV header defines the schema we will use downstream.
        writer.writerow(
            [
                "po_id",
                "vendor",
                "item",
                "category",
                "quantity",
                "unit_price",
                "total_price",
                "currency",
                "request_date",
                "delivery_date",
                "status",
            ]
        )
        for i in range(1, num_records + 1):
            # Create consistent but varied records.
            quantity = rng.randint(1, 50)
            unit_price = round(rng.uniform(5.0, 250.0), 2)
            total_price = round(quantity * unit_price, 2)
            request_date = base_date + timedelta(days=rng.randint(0, 365))
            delivery_date = request_date + timedelta(days=rng.randint(1, 30))
            writer.writerow(
                [
                    f"PO-2024-{i:04d}",
                    rng.choice(vendors),
                    rng.choice(items),
                    rng.choice(categories),
                    quantity,
                    f"{unit_price:.2f}",
                    f"{total_price:.2f}",
                    "EUR",
                    request_date.isoformat(),
                    delivery_date.isoformat(),
                    rng.choice(statuses),
                ]
            )


def _draw_table_image(out_path: Path, rows: int, cols: int, seed: int) -> None:
    """Draw a synthetic table image to stress-test table detection."""
    rng = random.Random(seed)
    headers = ["Item", "Qty", "Unit", "Total", "Vendor", "Status"]
    cell_w = 160
    cell_h = 40
    margin = 40

    width = margin * 2 + cell_w * cols
    height = margin * 2 + cell_h * rows
    # Start with a white canvas.
    img = np.full((height, width, 3), 255, dtype=np.uint8)

    # Draw table grid lines.
    for r in range(rows + 1):
        y = margin + r * cell_h
        cv2.line(img, (margin, y), (margin + cols * cell_w, y), (0, 0, 0), 2)
    for c in range(cols + 1):
        x = margin + c * cell_w
        cv2.line(img, (x, margin), (x, margin + rows * cell_h), (0, 0, 0), 2)

    def write_cell(text: str, row: int, col: int) -> None:
        x = margin + col * cell_w + 8
        y = margin + row * cell_h + 25
        cv2.putText(
            img,
            text,
            (x, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 0, 0),
            1,
            cv2.LINE_AA,
        )

    # Header row.
    for c in range(cols):
        header = headers[c] if c < len(headers) else f"Col {c + 1}"
        write_cell(header, 0, c)

    sample_items = ["Bolts", "Wire", "Boxes", "Gloves", "Hose", "Brackets"]
    sample_vendors = ["Nordic", "Alpha", "Rhein", "Helios"]
    # Body rows.
    for r in range(1, rows):
        for c in range(cols):
            if c == 0:
                text = rng.choice(sample_items)
            elif c == 1:
                text = str(rng.randint(1, 50))
            elif c == 2:
                text = f"{rng.randint(5, 200)} EUR"
            elif c == 3:
                text = f"{rng.randint(100, 5000)} EUR"
            elif c == 4:
                text = rng.choice(sample_vendors)
            else:
                text = rng.choice(["ok", "hold", "review"])
            write_cell(text, r, c)

    cv2.imwrite(str(out_path), img)


def _image_to_pdf(image_path: Path, pdf_path: Path) -> None:
    """Embed the generated image into a single-page PDF."""
    img = ImageReader(str(image_path))
    img_w, img_h = img.getSize()
    page_w, page_h = letter
    margin = 36
    scale = min((page_w - 2 * margin) / img_w, (page_h - 2 * margin) / img_h)
    draw_w = img_w * scale
    draw_h = img_h * scale
    x = (page_w - draw_w) / 2
    y = (page_h - draw_h) / 2

    c = canvas.Canvas(str(pdf_path), pagesize=letter)
    c.drawImage(str(image_path), x, y, width=draw_w, height=draw_h)
    c.showPage()
    c.save()


def _log(message: str, quiet: bool) -> None:
    """Print step-by-step progress unless quiet mode is enabled."""
    if not quiet:
        print(message)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate synthetic procurement data.")
    parser.add_argument("--out-dir", default="data", help="Output directory.")
    parser.add_argument("--rows", type=int, default=8, help="Table rows (including header).")
    parser.add_argument("--cols", type=int, default=5, help="Table columns.")
    parser.add_argument("--records", type=int, default=120, help="CSV record count.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--no-pdf", action="store_true", help="Skip PDF generation.")
    parser.add_argument("--quiet", action="store_true", help="Disable step-by-step output.")
    args = parser.parse_args()

    # Basic sanity check for the table layout.
    if args.rows < 2 or args.cols < 2:
        raise SystemExit("rows and cols must be >= 2")

    # Resolve output file locations.
    out_dir = _ensure_dir(Path(args.out_dir))
    csv_path = out_dir / "synthetic_purchases.csv"
    img_path = out_dir / "synthetic_table.png"
    pdf_path = out_dir / "synthetic_table.pdf"

    _log(f"Step 1/3: Generating CSV ({args.records} records)...", args.quiet)
    _generate_csv(csv_path, args.records, args.seed)
    _log(f"Step 2/3: Drawing table image ({args.rows}x{args.cols})...", args.quiet)
    _draw_table_image(img_path, args.rows, args.cols, args.seed)
    if not args.no_pdf:
        _log("Step 3/3: Converting image to PDF...", args.quiet)
        _image_to_pdf(img_path, pdf_path)

    print("Generated:")
    print(f"- {csv_path}")
    print(f"- {img_path}")
    if not args.no_pdf:
        print(f"- {pdf_path}")


if __name__ == "__main__":
    main()
