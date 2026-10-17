"""Render a short paper-preview MP4 from real project imagery and frozen data."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import statistics
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import imageio_ffmpeg
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageOps

ROOT = Path(__file__).resolve().parents[1]
VIDEO_DIR = Path(__file__).resolve().parent
DESIGN_SIZE = (1920, 1080)

NAVY = "#071521"
NAVY_2 = "#10283A"
INK = "#14212B"
MUTED = "#62717D"
PAPER = "#F7F8F6"
WHITE = "#FFFFFF"
BLUE = "#0072B2"
CYAN = "#34C8DF"
OPTICAL = "#55585C"
TESSERA = "#E69F00"
LAVA = "#E56B3F"
GREEN = "#26866A"
GRID = "#D9DFE3"
PALE_BLUE = "#EAF5FA"
PALE_ORANGE = "#FFF3D8"


@dataclass(frozen=True)
class Scene:
    key: str
    duration_s: float
    caption: str
    image: Image.Image


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data",
        type=Path,
        default=VIDEO_DIR / "preview_data.json",
        help="Editable video copy, values, and asset paths.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=VIDEO_DIR / "output",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Render a 1280x720, 15 fps review copy.",
    )
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Validate all sources and numerical invariants without rendering.",
    )
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def resolve_repo_path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"Expected an object in {path}")
    return payload


def load_heldout_rows(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8", newline="") as stream:
        for raw in csv.DictReader(stream):
            rows.append(
                {
                    "slot": int(raw["slot"]),
                    "event": raw["event"],
                    "event_year": int(raw["event_year"]),
                    "alphaearth": float(raw["rho_alphaearth"]),
                    "optical": float(raw["rho_optical_oracle"]),
                    "tessera": (
                        float(raw["rho_tessera"])
                        if raw["rho_tessera"].strip()
                        else math.nan
                    ),
                    "tessera_status": raw["tessera_status"],
                }
            )
    return rows


def validate_inputs(
    data_path: Path,
    payload: dict[str, Any],
) -> tuple[list[dict[str, Any]], list[Path]]:
    if payload.get("schema") != "reo2_paper_preview_v1":
        raise ValueError("Unexpected preview-data schema")
    provenance = payload["provenance"]
    heldout_path = resolve_repo_path(provenance["heldout_event_data"])
    ledger_path = resolve_repo_path(provenance["frozen_summary"])
    visual_manifest = resolve_repo_path(provenance["visual_manifest"])
    required = [
        data_path,
        heldout_path,
        ledger_path,
        visual_manifest,
        Path(__file__).resolve(),
        VIDEO_DIR / "requirements.txt",
    ]

    examples = payload["examples"]
    if {item["key"] for item in examples} != {"woolsey", "kilauea"}:
        raise ValueError("V1 requires the Woolsey and Kīlauea examples")
    for example in examples:
        for key in ("before", "after", "reference_image", "ecs_image"):
            required.append(resolve_repo_path(example[key]))

    missing = [path for path in required if not path.is_file()]
    if missing:
        formatted = "\n".join(f"  - {path}" for path in missing)
        raise FileNotFoundError(f"Missing video source files:\n{formatted}")

    for path in required:
        if path.suffix.lower() in {".png", ".jpg", ".jpeg"}:
            with Image.open(path) as image:
                image.verify()

    rows = load_heldout_rows(heldout_path)
    if len(rows) != 10 or [row["slot"] for row in rows] != list(range(1, 11)):
        raise ValueError("Held-out CSV must preserve frozen slots 1-10")
    stats = payload["paper_statistics"]
    observed_median = statistics.median(row["alphaearth"] for row in rows)
    if not math.isclose(
        observed_median,
        float(stats["alphaearth_median_rho"]),
        abs_tol=1e-12,
    ):
        raise ValueError("Held-out AlphaEarth median does not match frozen CSV")
    evaluable_tessera = sum(math.isfinite(row["tessera"]) for row in rows)
    if evaluable_tessera != int(stats["tessera_events"]):
        raise ValueError("TESSERA event count does not match frozen CSV")

    woolsey = next(item for item in examples if item["key"] == "woolsey")
    woolsey_row = next(row for row in rows if row["event"] == "WOOLSEY")
    if not math.isclose(
        float(woolsey["rho"]),
        float(woolsey_row["alphaearth"]),
        abs_tol=1e-12,
    ):
        raise ValueError("Woolsey example coefficient differs from frozen CSV")

    for prefix in ("alphaearth", "tessera"):
        mean = float(stats[f"{prefix}_contrast_mean"])
        low, high = (float(value) for value in stats[f"{prefix}_contrast_ci95"])
        if not low < mean < high or not low < 0.0 < high:
            raise ValueError(f"{prefix} interval invariant failed")

    families = payload["development_families"]
    if sum(int(item["events"]) for item in families) != int(
        stats["development_events"]
    ):
        raise ValueError("Development family counts do not sum to eight")
    for family in families:
        low = float(family["min_rho"])
        median = float(family["median_rho"])
        high = float(family["max_rho"])
        if not low <= median <= high:
            raise ValueError(f"Invalid development range for {family['label']}")
    return rows, required


def font_candidates(*, bold: bool) -> list[str]:
    if bold:
        return [
            "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
            "/Library/Fonts/Arial Bold.ttf",
            "DejaVuSans-Bold.ttf",
            "Arial Bold.ttf",
        ]
    return [
        "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/Library/Fonts/Arial.ttf",
        "DejaVuSans.ttf",
        "Arial.ttf",
    ]


_FONT_CACHE: dict[tuple[int, bool], ImageFont.FreeTypeFont | ImageFont.ImageFont] = {}


def get_font(
    size: int, *, bold: bool = False
) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    key = (size, bold)
    if key in _FONT_CACHE:
        return _FONT_CACHE[key]
    for candidate in font_candidates(bold=bold):
        try:
            loaded = ImageFont.truetype(candidate, size=size)
            _FONT_CACHE[key] = loaded
            return loaded
        except OSError:
            continue
    fallback = ImageFont.load_default(size=size)
    _FONT_CACHE[key] = fallback
    return fallback


def hex_to_rgb(value: str) -> tuple[int, int, int]:
    stripped = value.lstrip("#")
    return tuple(int(stripped[index : index + 2], 16) for index in (0, 2, 4))


def gradient_background(
    top: str = NAVY,
    bottom: str = NAVY_2,
) -> Image.Image:
    width, height = DESIGN_SIZE
    first = np.asarray(hex_to_rgb(top), dtype=float)
    second = np.asarray(hex_to_rgb(bottom), dtype=float)
    mix = np.linspace(0.0, 1.0, height, dtype=float)[:, None, None]
    row = first[None, None, :] * (1.0 - mix) + second[None, None, :] * mix
    pixels = np.repeat(row, width, axis=1).astype(np.uint8)
    return Image.fromarray(pixels, mode="RGB")


def text_width(
    draw: ImageDraw.ImageDraw, text: str, font: ImageFont.ImageFont
) -> float:
    return float(draw.textlength(text, font=font))


def wrap_lines(
    draw: ImageDraw.ImageDraw,
    text: str,
    font: ImageFont.ImageFont,
    max_width: int,
) -> list[str]:
    output: list[str] = []
    for paragraph in text.splitlines() or [""]:
        if not paragraph:
            output.append("")
            continue
        words = paragraph.split()
        line = words[0]
        for word in words[1:]:
            candidate = f"{line} {word}"
            if text_width(draw, candidate, font) <= max_width:
                line = candidate
            else:
                output.append(line)
                line = word
        output.append(line)
    return output


def draw_wrapped(
    draw: ImageDraw.ImageDraw,
    xy: tuple[int, int],
    text: str,
    *,
    font: ImageFont.ImageFont,
    fill: str,
    max_width: int,
    line_gap: int = 8,
    anchor: str = "la",
) -> int:
    x, y = xy
    lines = wrap_lines(draw, text, font, max_width)
    bbox = draw.textbbox((0, 0), "Ag", font=font)
    line_height = bbox[3] - bbox[1]
    for line in lines:
        if anchor == "ma":
            draw.text((x, y), line, font=font, fill=fill, anchor="ma")
        else:
            draw.text((x, y), line, font=font, fill=fill)
        y += line_height + line_gap
    return y


def draw_rounded_box(
    image: Image.Image,
    box: tuple[int, int, int, int],
    *,
    fill: str,
    outline: str | None = None,
    radius: int = 28,
    width: int = 2,
    shadow: bool = False,
) -> None:
    draw = ImageDraw.Draw(image)
    if shadow:
        x0, y0, x1, y1 = box
        draw.rounded_rectangle(
            (x0 + 8, y0 + 12, x1 + 8, y1 + 12),
            radius=radius,
            fill="#030A0F",
        )
    draw.rounded_rectangle(
        box,
        radius=radius,
        fill=fill,
        outline=outline,
        width=width,
    )


_IMAGE_CACHE: dict[Path, Image.Image] = {}


def load_image(path: Path) -> Image.Image:
    if path not in _IMAGE_CACHE:
        _IMAGE_CACHE[path] = Image.open(path).convert("RGB")
    return _IMAGE_CACHE[path]


def paste_rounded_image(
    canvas: Image.Image,
    source: Path,
    box: tuple[int, int, int, int],
    *,
    radius: int = 24,
    contain: bool = False,
    background: str = "#E8EAEC",
) -> None:
    x0, y0, x1, y1 = box
    size = (x1 - x0, y1 - y0)
    original = load_image(source)
    if contain:
        fitted = ImageOps.contain(original, size, Image.Resampling.LANCZOS)
        card = Image.new("RGB", size, background)
        offset = ((size[0] - fitted.width) // 2, (size[1] - fitted.height) // 2)
        card.paste(fitted, offset)
    else:
        card = ImageOps.fit(
            original,
            size,
            Image.Resampling.LANCZOS,
            centering=(0.5, 0.5),
        )
    mask = Image.new("L", size, 0)
    ImageDraw.Draw(mask).rounded_rectangle((0, 0, *size), radius=radius, fill=255)
    canvas.paste(card, (x0, y0), mask)


def draw_image_card(
    canvas: Image.Image,
    source: Path,
    box: tuple[int, int, int, int],
    label: str,
    *,
    contain: bool = False,
    label_fill: str = INK,
) -> None:
    x0, y0, x1, y1 = box
    draw_rounded_box(canvas, box, fill="#FFFFFF", radius=24, shadow=True)
    paste_rounded_image(
        canvas,
        source,
        (x0 + 5, y0 + 5, x1 - 5, y1 - 54),
        radius=20,
        contain=contain,
    )
    draw = ImageDraw.Draw(canvas)
    draw.text(
        ((x0 + x1) // 2, y1 - 27),
        label,
        font=get_font(24, bold=True),
        fill=label_fill,
        anchor="mm",
    )


def draw_kicker(draw: ImageDraw.ImageDraw, text: str, *, dark: bool = True) -> None:
    draw.text(
        (72, 54),
        text.upper(),
        font=get_font(22, bold=True),
        fill=CYAN if dark else BLUE,
    )


def draw_title(
    draw: ImageDraw.ImageDraw,
    text: str,
    *,
    y: int = 88,
    dark: bool = True,
    size: int = 52,
    max_width: int = 1700,
) -> int:
    return draw_wrapped(
        draw,
        (72, y),
        text,
        font=get_font(size, bold=True),
        fill=WHITE if dark else INK,
        max_width=max_width,
        line_gap=7,
    )


def draw_footer(
    draw: ImageDraw.ImageDraw,
    text: str,
    *,
    dark: bool = True,
) -> None:
    draw.text(
        (72, 1040),
        text,
        font=get_font(18),
        fill="#ADC0CC" if dark else MUTED,
        anchor="lm",
    )


def draw_badge(
    image: Image.Image,
    box: tuple[int, int, int, int],
    title: str,
    detail: str,
    *,
    accent: str = CYAN,
) -> None:
    draw_rounded_box(
        image,
        box,
        fill="#102C3E",
        outline="#28526A",
        radius=24,
    )
    draw = ImageDraw.Draw(image)
    x0, y0, _, _ = box
    draw.text((x0 + 26, y0 + 22), title, font=get_font(28, bold=True), fill=accent)
    draw.text((x0 + 26, y0 + 61), detail, font=get_font(20), fill="#D4E2EA")


def draw_hex_motif(image: Image.Image) -> None:
    draw = ImageDraw.Draw(image, "RGBA")
    radius = 52
    for row in range(6):
        for column in range(7):
            cx = 1480 + column * 78 + (row % 2) * 39
            cy = 115 + row * 69
            points = [
                (
                    cx + radius * math.cos(math.radians(60 * index)),
                    cy + radius * math.sin(math.radians(60 * index)),
                )
                for index in range(6)
            ]
            alpha = max(12, 56 - row * 5 - column * 3)
            draw.polygon(points, outline=(52, 200, 223, alpha), width=2)


def scene_title(payload: dict[str, Any]) -> Image.Image:
    image = gradient_background("#06131F", "#123047")
    draw_hex_motif(image)
    draw = ImageDraw.Draw(image)
    draw.text(
        (78, 70),
        "NEURIPS REO-2 • PAPER PREVIEW",
        font=get_font(23, bold=True),
        fill=CYAN,
    )
    draw_wrapped(
        draw,
        (78, 188),
        payload["title"],
        font=get_font(82, bold=True),
        fill=WHITE,
        max_width=1170,
        line_gap=10,
    )
    draw_wrapped(
        draw,
        (82, 380),
        payload["subtitle"],
        font=get_font(38),
        fill="#C9D9E3",
        max_width=1120,
    )
    draw_badge(image, (82, 560, 545, 662), "10 held-out fires", "Outcome-blind test")
    draw_badge(
        image, (570, 560, 1033, 662), "8 development cases", "Lava • forest • drought"
    )
    draw_badge(
        image,
        (1058, 560, 1521, 662),
        "One reusable score",
        "No hazard-specific training",
        accent="#FFD166",
    )
    draw_wrapped(
        draw,
        (82, 757),
        "Do annual Earth embeddings recover real disturbance patterns, "
        "and do they add value beyond specialized optical indices?",
        font=get_font(33, bold=True),
        fill=WHITE,
        max_width=1420,
        line_gap=10,
    )
    draw.text(
        (82, 970),
        payload["repository"],
        font=get_font(22),
        fill="#AFC4D0",
    )
    return image


def scene_annual_question(payload: dict[str, Any]) -> Image.Image:
    example = next(item for item in payload["examples"] if item["key"] == "woolsey")
    image = gradient_background()
    draw = ImageDraw.Draw(image)
    draw_kicker(draw, "The basic question")
    draw_title(draw, "What changed between two annual snapshots?")
    draw_image_card(
        image,
        resolve_repo_path(example["before"]),
        (72, 190, 838, 755),
        "2017 • before",
    )
    draw_image_card(
        image,
        resolve_repo_path(example["after"]),
        (1082, 190, 1848, 755),
        "2019 • after",
    )
    draw.line((875, 450, 1045, 450), fill=CYAN, width=8)
    draw.polygon([(1045, 450), (1010, 430), (1010, 470)], fill=CYAN)
    draw.text(
        (960, 390),
        "2018\nhazard",
        font=get_font(26, bold=True),
        fill=WHITE,
        anchor="mm",
        align="center",
        spacing=5,
    )
    draw_rounded_box(
        image,
        (240, 820, 1680, 967),
        fill="#102C3E",
        outline="#28526A",
        radius=28,
    )
    draw_wrapped(
        draw,
        (960, 852),
        "AlphaEarth stores 64 numbers for every 10 m pixel each year. "
        "We ask whether their change follows the mapped impact.",
        font=get_font(30, bold=True),
        fill=WHITE,
        max_width=1320,
        line_gap=8,
        anchor="ma",
    )
    draw_footer(draw, "Real Sentinel-2 context imagery • Woolsey Fire, California")
    return image


def draw_vector_bars(
    draw: ImageDraw.ImageDraw,
    origin: tuple[int, int],
    values: list[float],
    *,
    color: str,
) -> None:
    x, y = origin
    for index, value in enumerate(values):
        height = int(180 * value)
        draw.rounded_rectangle(
            (x + index * 29, y + 180 - height, x + index * 29 + 17, y + 180),
            radius=7,
            fill=color,
        )


def draw_hex_grid(
    draw: ImageDraw.ImageDraw,
    origin: tuple[int, int],
    *,
    color: str,
) -> None:
    x0, y0 = origin
    radius = 39
    for row in range(4):
        for column in range(5):
            cx = x0 + column * 66 + (row % 2) * 33
            cy = y0 + row * 57
            points = [
                (
                    cx + radius * math.cos(math.radians(60 * index)),
                    cy + radius * math.sin(math.radians(60 * index)),
                )
                for index in range(6)
            ]
            fill = color if (row + column) % 4 == 0 else "#DDE9EE"
            draw.polygon(points, fill=fill, outline="#91A8B5", width=2)


def scene_method(payload: dict[str, Any]) -> Image.Image:
    stats = payload["paper_statistics"]
    image = Image.new("RGB", DESIGN_SIZE, PAPER)
    draw = ImageDraw.Draw(image)
    draw_kicker(draw, "One score, no labels", dark=False)
    draw_title(
        draw,
        "Embedding change squared (ECS)",
        dark=False,
    )
    boxes = [(74, 215, 590, 790), (702, 215, 1218, 790), (1330, 215, 1846, 790)]
    accents = [BLUE, LAVA, GREEN]
    titles = ["1. Annual embedding", "2. Compare endpoints", "3. Average to H3"]
    for box, accent, title in zip(boxes, accents, titles, strict=True):
        draw_rounded_box(
            image,
            box,
            fill=WHITE,
            outline=accent,
            radius=28,
            width=3,
            shadow=True,
        )
        draw.text(
            (box[0] + 28, box[1] + 28),
            title,
            font=get_font(31, bold=True),
            fill=INK,
        )
    draw_vector_bars(
        draw,
        (145, 405),
        [0.35, 0.72, 0.52, 0.88, 0.44, 0.62, 0.26, 0.77, 0.58, 0.91],
        color=BLUE,
    )
    draw.text(
        (332, 665),
        f"{stats['alphaearth_dimensions']} dimensions\nper {stats['alphaearth_pixel_m']} m pixel, per year",
        font=get_font(27, bold=True),
        fill=MUTED,
        anchor="mm",
        align="center",
        spacing=7,
    )
    draw.text(
        (960, 440),
        "ECS = ‖u post − u pre‖²",
        font=get_font(39, bold=True),
        fill=LAVA,
        anchor="mm",
    )
    draw.text(
        (960, 540),
        "Large value = large\nrepresentation change",
        font=get_font(28),
        fill=MUTED,
        anchor="mm",
        align="center",
        spacing=8,
    )
    draw_hex_grid(draw, (1420, 395), color="#68C3A7")
    draw.text(
        (1588, 681),
        "H3-r8 cells\nfor the fire study",
        font=get_font(27, bold=True),
        fill=MUTED,
        anchor="mm",
        align="center",
        spacing=7,
    )
    draw_rounded_box(
        image,
        (260, 855, 1660, 968),
        fill=PALE_BLUE,
        outline="#BBDCEB",
        radius=24,
    )
    draw.text(
        (960, 911),
        "The same unsupervised score is used for fire, lava, forest loss, and drought.",
        font=get_font(31, bold=True),
        fill=INK,
        anchor="mm",
    )
    draw_footer(
        draw,
        "Primary event bracket: full-calendar Y−1 → Y+1 • No disturbance labels enter ECS",
        dark=False,
    )
    return image


def scene_example(payload: dict[str, Any], key: str) -> Image.Image:
    example = next(item for item in payload["examples"] if item["key"] == key)
    is_fire = key == "woolsey"
    accent = BLUE if is_fire else LAVA
    image = gradient_background()
    draw = ImageDraw.Draw(image)
    draw_kicker(draw, "Real example • same 2017 → 2019 bracket")
    draw_title(
        draw,
        f"{example['label']}: embedding change follows the mapped impact",
        size=47,
        max_width=1770,
    )
    cards = [
        (
            resolve_repo_path(example["before"]),
            (70, 180, 680, 510),
            f"{example['pre_year']} • before",
            False,
        ),
        (
            resolve_repo_path(example["after"]),
            (720, 180, 1330, 510),
            f"{example['post_year']} • after",
            False,
        ),
        (
            resolve_repo_path(example["reference_image"]),
            (70, 550, 680, 940),
            example["reference"],
            True,
        ),
        (
            resolve_repo_path(example["ecs_image"]),
            (720, 550, 1330, 940),
            "AlphaEarth ECS",
            True,
        ),
    ]
    for source, box, label, contain in cards:
        draw_image_card(
            image,
            source,
            box,
            label,
            contain=contain,
            label_fill=INK,
        )

    draw_rounded_box(
        image,
        (1380, 190, 1848, 940),
        fill="#102C3E",
        outline=accent,
        radius=30,
        width=3,
    )
    draw.text(
        (1614, 265),
        "Within-event\nSpearman association",
        font=get_font(27, bold=True),
        fill="#CFE0E9",
        anchor="mm",
        align="center",
        spacing=8,
    )
    draw.text(
        (1614, 438),
        f"ρ = {float(example['rho']):.3f}",
        font=get_font(66, bold=True),
        fill=accent,
        anchor="mm",
    )
    draw.text(
        (1614, 535),
        f"{example['h3_cells']} matched H3-r{example['h3_resolution']} cells",
        font=get_font(25, bold=True),
        fill=WHITE,
        anchor="mm",
    )
    message = (
        "Severity ordering is recovered inside the official fire perimeter."
        if is_fire
        else "The same generic score follows a mapped lava footprint."
    )
    draw_wrapped(
        draw,
        (1614, 640),
        message,
        font=get_font(27),
        fill="#D5E2E9",
        max_width=390,
        line_gap=9,
        anchor="ma",
    )
    draw_wrapped(
        draw,
        (1614, 820),
        "Association is not detection, causation, or ground truth.",
        font=get_font(21, bold=True),
        fill="#FFD166",
        max_width=390,
        line_gap=7,
        anchor="ma",
    )
    draw_footer(
        draw,
        "Actual project imagery and maps • Colors normalized per panel; ρ uses raw matched values",
    )
    return image


def value_to_x(value: float, left: int, right: int, low: float, high: float) -> int:
    return int(left + (value - low) / (high - low) * (right - left))


def draw_triangle(
    draw: ImageDraw.ImageDraw,
    center: tuple[int, int],
    *,
    radius: int,
    fill: str,
) -> None:
    x, y = center
    draw.polygon(
        [(x, y - radius), (x - radius, y + radius), (x + radius, y + radius)],
        fill=fill,
    )


def scene_cohort(payload: dict[str, Any], rows: list[dict[str, Any]]) -> Image.Image:
    stats = payload["paper_statistics"]
    image = Image.new("RGB", DESIGN_SIZE, PAPER)
    draw = ImageDraw.Draw(image)
    draw_kicker(draw, "Outcome-blind held-out test", dark=False)
    draw_title(
        draw,
        "Ten fires were frozen before AlphaEarth extraction",
        dark=False,
        size=48,
    )
    chart = (60, 170, 1495, 965)
    draw_rounded_box(
        image, chart, fill=WHITE, outline="#D5DBDF", radius=26, shadow=True
    )
    plot_left, plot_right = 350, 1430
    low, high = -0.45, 1.0
    row_top, row_step = 252, 62

    for value in (-0.4, 0.0, 0.4, 0.8, 1.0):
        x = value_to_x(value, plot_left, plot_right, low, high)
        draw.line((x, 225, x, 855), fill=GRID, width=2)
        draw.text(
            (x, 885),
            f"{value:.1f}",
            font=get_font(20),
            fill=MUTED,
            anchor="ma",
        )
    zero_x = value_to_x(0.0, plot_left, plot_right, low, high)
    draw.line((zero_x, 225, zero_x, 855), fill="#9BA5AB", width=3)

    for index, row in enumerate(rows):
        y = row_top + index * row_step
        if index % 2 == 0:
            draw.rectangle((82, y - 25, 1465, y + 26), fill="#F4F6F7")
        draw.text(
            (320, y),
            row["event"],
            font=get_font(20, bold=True),
            fill=INK,
            anchor="rm",
        )
        values = [float(row["alphaearth"]), float(row["optical"])]
        if math.isfinite(float(row["tessera"])):
            values.append(float(row["tessera"]))
        draw.line(
            (
                value_to_x(min(values), plot_left, plot_right, low, high),
                y,
                value_to_x(max(values), plot_left, plot_right, low, high),
                y,
            ),
            fill="#B7C0C5",
            width=3,
        )
        alpha_x = value_to_x(float(row["alphaearth"]), plot_left, plot_right, low, high)
        optical_x = value_to_x(float(row["optical"]), plot_left, plot_right, low, high)
        draw.ellipse((alpha_x - 8, y - 8, alpha_x + 8, y + 8), fill=BLUE)
        draw.rectangle(
            (optical_x - 7, y - 7, optical_x + 7, y + 7),
            fill=OPTICAL,
        )
        if math.isfinite(float(row["tessera"])):
            tessera_x = value_to_x(
                float(row["tessera"]),
                plot_left,
                plot_right,
                low,
                high,
            )
            draw_triangle(draw, (tessera_x, y), radius=9, fill=TESSERA)
        else:
            draw.text(
                (plot_left + 10, y),
                "TESSERA missing",
                font=get_font(16),
                fill=TESSERA,
                anchor="lm",
            )

    draw.text(
        ((plot_left + plot_right) // 2, 930),
        "Within-perimeter Spearman association, ρ",
        font=get_font(23, bold=True),
        fill=INK,
        anchor="mm",
    )
    legend_y = 196
    draw.ellipse((570, legend_y - 7, 584, legend_y + 7), fill=BLUE)
    draw.text((596, legend_y), "AlphaEarth", font=get_font(18), fill=INK, anchor="lm")
    draw.rectangle((760, legend_y - 7, 774, legend_y + 7), fill=OPTICAL)
    draw.text((786, legend_y), "Optical", font=get_font(18), fill=INK, anchor="lm")
    draw_triangle(draw, (925, legend_y), radius=8, fill=TESSERA)
    draw.text((945, legend_y), "TESSERA", font=get_font(18), fill=INK, anchor="lm")

    draw_rounded_box(
        image,
        (1535, 178, 1860, 963),
        fill=NAVY_2,
        outline="#2E546B",
        radius=28,
    )
    draw.text(
        (1698, 255),
        "AlphaEarth",
        font=get_font(27, bold=True),
        fill=CYAN,
        anchor="mm",
    )
    draw.text(
        (1698, 357),
        f"{float(stats['alphaearth_median_rho']):.3f}",
        font=get_font(64, bold=True),
        fill=WHITE,
        anchor="mm",
    )
    draw.text(
        (1698, 410),
        "median event ρ",
        font=get_font(22),
        fill="#C7D7E0",
        anchor="mm",
    )
    draw.line((1580, 470, 1815, 470), fill="#35586B", width=2)
    draw.text(
        (1698, 545),
        "9 / 10",
        font=get_font(49, bold=True),
        fill="#7BDCB5",
        anchor="mm",
    )
    draw.text(
        (1698, 595),
        "positive associations",
        font=get_font(20),
        fill="#C7D7E0",
        anchor="mm",
    )
    draw.line((1580, 655, 1815, 655), fill="#35586B", width=2)
    draw.text(
        (1698, 730),
        f"{int(stats['heldout_h3_cells']):,}",
        font=get_font(47, bold=True),
        fill="#FFD166",
        anchor="mm",
    )
    draw.text(
        (1698, 782),
        "frozen H3-r8 cells",
        font=get_font(20),
        fill="#C7D7E0",
        anchor="mm",
    )
    draw_wrapped(
        draw,
        (1698, 862),
        "Events, not cells, are the inferential units.",
        font=get_font(20, bold=True),
        fill=WHITE,
        max_width=255,
        line_gap=6,
        anchor="ma",
    )
    draw_footer(
        draw,
        "Real frozen coefficients • Blue: AlphaEarth • Gray: best dNBR/dNDVI • Orange: TESSERA",
        dark=False,
    )
    return image


def scene_incremental_value(payload: dict[str, Any]) -> Image.Image:
    stats = payload["paper_statistics"]
    image = gradient_background("#071521", "#173348")
    draw = ImageDraw.Draw(image)
    draw_kicker(draw, "The stricter question")
    draw_title(
        draw,
        "Does embedding change consistently beat specialized optical change?",
        size=48,
    )
    plot_box = (80, 230, 1320, 865)
    draw_rounded_box(
        image,
        plot_box,
        fill="#F8FAFA",
        outline="#C8D3D9",
        radius=30,
        shadow=True,
    )
    axis_left, axis_right = 265, 1240
    low, high = -0.45, 0.15
    axis_y = 760
    draw.line((axis_left, axis_y, axis_right, axis_y), fill="#9EA8AE", width=3)
    for value in (-0.4, -0.2, 0.0, 0.1):
        x = value_to_x(value, axis_left, axis_right, low, high)
        draw.line((x, 310, x, axis_y + 10), fill=GRID, width=2)
        draw.text(
            (x, 800),
            f"{value:.1f}",
            font=get_font(21),
            fill=MUTED,
            anchor="ma",
        )
    zero_x = value_to_x(0.0, axis_left, axis_right, low, high)
    draw.line((zero_x, 300, zero_x, axis_y), fill="#7A858C", width=4)
    draw.text(
        (zero_x, 270),
        "equal performance",
        font=get_font(18, bold=True),
        fill=MUTED,
        anchor="ma",
    )

    rows = [
        (
            "AlphaEarth − optical",
            float(stats["alphaearth_contrast_mean"]),
            tuple(float(value) for value in stats["alphaearth_contrast_ci95"]),
            BLUE,
            430,
        ),
        (
            "TESSERA − optical",
            float(stats["tessera_contrast_mean"]),
            tuple(float(value) for value in stats["tessera_contrast_ci95"]),
            TESSERA,
            610,
        ),
    ]
    for label, mean, interval, color, y in rows:
        draw.text(
            (105, y),
            label,
            font=get_font(24, bold=True),
            fill=INK,
            anchor="lm",
        )
        left_x = value_to_x(interval[0], axis_left, axis_right, low, high)
        right_x = value_to_x(interval[1], axis_left, axis_right, low, high)
        mean_x = value_to_x(mean, axis_left, axis_right, low, high)
        draw.line((left_x, y, right_x, y), fill=color, width=12)
        draw.line((left_x, y - 17, left_x, y + 17), fill=color, width=5)
        draw.line((right_x, y - 17, right_x, y + 17), fill=color, width=5)
        draw.ellipse((mean_x - 14, y - 14, mean_x + 14, y + 14), fill=color)
        draw.text(
            (mean_x, y - 48),
            f"{mean:+.3f}",
            font=get_font(22, bold=True),
            fill=color,
            anchor="mm",
        )
        draw.text(
            ((left_x + right_x) // 2, y + 42),
            f"95% interval [{interval[0]:+.3f}, {interval[1]:+.3f}]",
            font=get_font(18),
            fill=MUTED,
            anchor="mm",
        )

    draw.text(
        ((axis_left + axis_right) // 2, 842),
        "Mean event-level difference in Spearman association",
        font=get_font(22, bold=True),
        fill=INK,
        anchor="mm",
    )
    draw_rounded_box(
        image,
        (1380, 255, 1845, 840),
        fill="#102C3E",
        outline="#39637A",
        radius=30,
    )
    draw.text(
        (1612, 360),
        "Both intervals\ncross zero",
        font=get_font(44, bold=True),
        fill=WHITE,
        anchor="mm",
        align="center",
        spacing=9,
    )
    draw.text(
        (1612, 530),
        "No consistent\nstandalone advantage",
        font=get_font(30, bold=True),
        fill="#FFD166",
        anchor="mm",
        align="center",
        spacing=8,
    )
    draw_wrapped(
        draw,
        (1612, 665),
        "Embedding change is informative, but the task-aligned optical baseline "
        "is often equally strong or stronger.",
        font=get_font(23),
        fill="#D4E2E9",
        max_width=365,
        line_gap=8,
        anchor="ma",
    )
    draw_footer(
        draw,
        "Prespecified event bootstrap • Honest negative result: fidelity does not imply added value",
    )
    return image


def scene_development(payload: dict[str, Any]) -> Image.Image:
    stats = payload["paper_statistics"]
    image = Image.new("RGB", DESIGN_SIZE, PAPER)
    draw = ImageDraw.Draw(image)
    draw_kicker(draw, "Development audit", dark=False)
    draw_title(
        draw,
        "The same score was reused across three disturbance families",
        dark=False,
        size=48,
    )
    draw_rounded_box(
        image,
        (1420, 55, 1835, 130),
        fill="#E3F5EE",
        outline="#9DD6C0",
        radius=24,
    )
    draw.text(
        (1628, 93),
        f"{stats['development_positive_events']} / {stats['development_events']} associations positive",
        font=get_font(24, bold=True),
        fill=GREEN,
        anchor="mm",
    )
    families = payload["development_families"]
    boxes = [(72, 235, 592, 820), (700, 235, 1220, 820), (1328, 235, 1848, 820)]
    colors = [LAVA, GREEN, BLUE]
    for family, box, color in zip(families, boxes, colors, strict=True):
        draw_rounded_box(
            image,
            box,
            fill=WHITE,
            outline=color,
            radius=30,
            width=3,
            shadow=True,
        )
        center_x = (box[0] + box[2]) // 2
        draw.text(
            (center_x, box[1] + 65),
            family["label"],
            font=get_font(33, bold=True),
            fill=INK,
            anchor="mm",
        )
        draw.text(
            (center_x, box[1] + 120),
            f"{family['events']} predefined cases",
            font=get_font(22),
            fill=MUTED,
            anchor="mm",
        )
        draw.text(
            (center_x, box[1] + 240),
            f"median ρ = {float(family['median_rho']):.3f}",
            font=get_font(38, bold=True),
            fill=color,
            anchor="mm",
        )
        line_left, line_right = box[0] + 70, box[2] - 70
        y = box[1] + 365
        draw.line((line_left, y, line_right, y), fill="#CAD2D7", width=8)
        range_left = value_to_x(
            float(family["min_rho"]),
            line_left,
            line_right,
            0.0,
            0.8,
        )
        range_right = value_to_x(
            float(family["max_rho"]),
            line_left,
            line_right,
            0.0,
            0.8,
        )
        median_x = value_to_x(
            float(family["median_rho"]),
            line_left,
            line_right,
            0.0,
            0.8,
        )
        draw.line((range_left, y, range_right, y), fill=color, width=12)
        draw.ellipse((median_x - 14, y - 14, median_x + 14, y + 14), fill=color)
        draw.text(
            (center_x, y + 55),
            f"range {float(family['min_rho']):.3f} to {float(family['max_rho']):.3f}",
            font=get_font(21),
            fill=MUTED,
            anchor="mm",
        )
        draw_wrapped(
            draw,
            (center_x, box[1] + 485),
            "Positive association with the mapped reference.",
            font=get_font(23, bold=True),
            fill=INK,
            max_width=400,
            line_gap=7,
            anchor="ma",
        )
    draw_rounded_box(
        image,
        (275, 880, 1645, 975),
        fill=PALE_ORANGE,
        outline="#F3D596",
        radius=24,
    )
    draw.text(
        (960, 927),
        "Promising portability, but development evidence is not cross-hazard proof.",
        font=get_font(29, bold=True),
        fill=INK,
        anchor="mm",
    )
    draw_footer(
        draw,
        "USGS/CEMS lava • Hansen forest loss • U.S. Drought Monitor",
        dark=False,
    )
    return image


def scene_takeaway(payload: dict[str, Any]) -> Image.Image:
    image = gradient_background("#06131F", "#15364B")
    draw_hex_motif(image)
    draw = ImageDraw.Draw(image)
    draw_kicker(draw, "Bottom line")
    draw_title(draw, "A reusable change signal, with clear limits", size=58)
    statements = [
        (
            "1",
            "Annual embeddings often recover mapped spatial ordering.",
            CYAN,
        ),
        (
            "2",
            "Specialized optical indices remain highly competitive.",
            "#FFD166",
        ),
        (
            "3",
            "Change signal ≠ detector, causation, or ground truth.",
            "#FF9F7A",
        ),
    ]
    y_positions = [280, 490, 700]
    for (number, text, color), y in zip(statements, y_positions, strict=True):
        draw_rounded_box(
            image,
            (115, y, 1650, y + 150),
            fill="#102C3E",
            outline="#31556A",
            radius=30,
        )
        draw.ellipse((150, y + 35, 230, y + 115), fill=color)
        draw.text(
            (190, y + 75),
            number,
            font=get_font(37, bold=True),
            fill=NAVY,
            anchor="mm",
        )
        draw_wrapped(
            draw,
            (275, y + 46),
            text,
            font=get_font(34, bold=True),
            fill=WHITE,
            max_width=1305,
            line_gap=8,
        )
    draw.text(
        (960, 955),
        payload["repository"],
        font=get_font(27, bold=True),
        fill=CYAN,
        anchor="mm",
    )
    draw_footer(draw, "Reusable Change Signal • NeurIPS REO-2 paper preview")
    return image


def build_scenes(payload: dict[str, Any], rows: list[dict[str, Any]]) -> list[Scene]:
    return [
        Scene(
            "title",
            4.5,
            "Can one reusable representation describe different kinds of "
            "Earth-surface change?",
            scene_title(payload),
        ),
        Scene(
            "annual-question",
            7.0,
            "AlphaEarth stores a 64-dimensional annual embedding for every "
            "10-meter pixel. We compare years bracketing a hazard.",
            scene_annual_question(payload),
        ),
        Scene(
            "method",
            7.0,
            "We square the distance between pre- and post-event embeddings, "
            "then average that generic score within H3 cells.",
            scene_method(payload),
        ),
        Scene(
            "woolsey",
            9.5,
            "For the Woolsey Fire, AlphaEarth change follows MTBS burn severity "
            "with a within-event Spearman correlation of 0.757.",
            scene_example(payload, "woolsey"),
        ),
        Scene(
            "kilauea",
            9.5,
            "For the Kīlauea eruption, the same unsupervised score follows the "
            "USGS lava map with a correlation of 0.774.",
            scene_example(payload, "kilauea"),
        ),
        Scene(
            "heldout-cohort",
            11.5,
            "The ten-fire cohort was frozen before AlphaEarth extraction. "
            "Nine associations were positive, with median rho 0.712.",
            scene_cohort(payload, rows),
        ),
        Scene(
            "incremental-value",
            8.5,
            "But neither AlphaEarth nor TESSERA consistently beat the "
            "task-aligned optical comparator. Both confidence intervals cross zero.",
            scene_incremental_value(payload),
        ),
        Scene(
            "development-audit",
            7.5,
            "Across eight development cases, all associations were positive, "
            "although strength varied across lava, forest loss, and drought.",
            scene_development(payload),
        ),
        Scene(
            "takeaway",
            6.5,
            "Annual embeddings provide a portable change representation, not "
            "a replacement for specialized indices or an operational detector.",
            scene_takeaway(payload),
        ),
    ]


def format_srt_time(seconds: float) -> str:
    milliseconds = round(seconds * 1000)
    hours, remainder = divmod(milliseconds, 3_600_000)
    minutes, remainder = divmod(remainder, 60_000)
    whole_seconds, milliseconds = divmod(remainder, 1000)
    return f"{hours:02d}:{minutes:02d}:{whole_seconds:02d},{milliseconds:03d}"


def write_srt(path: Path, scenes: list[Scene]) -> None:
    lines: list[str] = []
    start = 0.0
    for index, scene in enumerate(scenes, start=1):
        end = start + scene.duration_s
        lines.extend(
            [
                str(index),
                f"{format_srt_time(start)} --> {format_srt_time(end)}",
                scene.caption,
                "",
            ]
        )
        start = end
    path.write_text("\n".join(lines), encoding="utf-8")


def write_storyboard(path: Path, scenes: list[Scene]) -> None:
    thumb_size = (576, 324)
    margin = 32
    label_height = 44
    board = Image.new(
        "RGB",
        (
            margin + 3 * (thumb_size[0] + margin),
            margin + 3 * (thumb_size[1] + label_height + margin),
        ),
        "#E5E8EA",
    )
    draw = ImageDraw.Draw(board)
    for index, scene in enumerate(scenes):
        row, column = divmod(index, 3)
        x = margin + column * (thumb_size[0] + margin)
        y = margin + row * (thumb_size[1] + label_height + margin)
        thumbnail = scene.image.resize(thumb_size, Image.Resampling.LANCZOS)
        board.paste(thumbnail, (x, y))
        draw.text(
            (x, y + thumb_size[1] + 9),
            f"{index + 1}. {scene.key} • {scene.duration_s:g}s",
            font=get_font(19, bold=True),
            fill=INK,
        )
    board.save(path, quality=92, optimize=True)


def resize_frame(image: Image.Image, output_size: tuple[int, int]) -> Image.Image:
    if image.size == output_size:
        return image
    return image.resize(output_size, Image.Resampling.LANCZOS)


def render_video(
    path: Path,
    scenes: list[Scene],
    *,
    output_size: tuple[int, int],
    fps: int,
) -> dict[str, Any]:
    transition_frames = max(1, round(0.55 * fps))
    frame_counts = [round(scene.duration_s * fps) for scene in scenes]
    total_frames = sum(frame_counts)
    writer = imageio_ffmpeg.write_frames(
        str(path),
        output_size,
        fps=fps,
        codec="libx264",
        pix_fmt_in="rgb24",
        pix_fmt_out="yuv420p",
        quality=7,
        macro_block_size=2,
        ffmpeg_log_level="warning",
        output_params=["-movflags", "+faststart"],
    )
    writer.send(None)
    try:
        prior: Image.Image | None = None
        emitted = 0
        for scene_index, (scene, count) in enumerate(
            zip(scenes, frame_counts, strict=True)
        ):
            current = resize_frame(scene.image, output_size)
            previous = resize_frame(prior, output_size) if prior is not None else None
            for local_frame in range(count):
                if previous is not None and local_frame < transition_frames:
                    fraction = (local_frame + 1) / transition_frames
                    eased = fraction * fraction * (3.0 - 2.0 * fraction)
                    frame = Image.blend(previous, current, eased)
                else:
                    frame = current
                if scene_index == 0 and local_frame < transition_frames:
                    fraction = (local_frame + 1) / transition_frames
                    frame = Image.blend(
                        Image.new("RGB", output_size, "#000000"), frame, fraction
                    )
                if (
                    scene_index == len(scenes) - 1
                    and local_frame >= count - transition_frames
                ):
                    fraction = (count - local_frame - 1) / transition_frames
                    frame = Image.blend(
                        Image.new("RGB", output_size, "#000000"),
                        frame,
                        max(0.0, fraction),
                    )
                writer.send(np.asarray(frame, dtype=np.uint8).tobytes())
                emitted += 1
            prior = scene.image
    finally:
        writer.close()
    encoded_frames, encoded_duration = imageio_ffmpeg.count_frames_and_secs(str(path))
    if abs(encoded_frames - total_frames) > 1:
        raise RuntimeError(
            f"Encoded frame count {encoded_frames} differs from expected {total_frames}"
        )
    if not math.isclose(encoded_duration, total_frames / fps, abs_tol=0.1):
        raise RuntimeError(
            f"Encoded duration {encoded_duration} differs from expected {total_frames / fps}"
        )
    return {
        "fps": fps,
        "width": output_size[0],
        "height": output_size[1],
        "frame_count": encoded_frames,
        "duration_s": encoded_duration,
    }


def main() -> int:
    args = parse_args()
    data_path = args.data.expanduser().resolve()
    payload = load_json(data_path)
    rows, source_paths = validate_inputs(data_path, payload)
    print(f"Validated {len(source_paths)} source files and {len(rows)} frozen fires.")
    if args.validate_only:
        return 0

    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    scenes = build_scenes(payload, rows)
    stem = "paper_preview_v1_quick" if args.quick else "paper_preview_v1"
    video_path = output_dir / f"{stem}.mp4"
    storyboard_path = output_dir / f"{stem}_storyboard.jpg"
    poster_path = output_dir / f"{stem}_poster.jpg"
    srt_path = output_dir / f"{stem}.srt"
    manifest_path = output_dir / f"{stem}_manifest.json"

    write_storyboard(storyboard_path, scenes)
    scenes[0].image.save(poster_path, quality=94, optimize=True)
    write_srt(srt_path, scenes)
    render = render_video(
        video_path,
        scenes,
        output_size=(1280, 720) if args.quick else DESIGN_SIZE,
        fps=15 if args.quick else 30,
    )
    manifest: dict[str, Any] = {
        "schema": "reo2_paper_preview_render_v1",
        "created_at_utc": datetime.now(timezone.utc)
        .isoformat(timespec="seconds")
        .replace("+00:00", "Z"),
        "video": {
            **render,
            "path": str(video_path.relative_to(ROOT)),
            "bytes": video_path.stat().st_size,
            "sha256": sha256_file(video_path),
            "codec": "H.264",
            "audio": False,
        },
        "scenes": [
            {
                "key": scene.key,
                "duration_s": scene.duration_s,
                "caption": scene.caption,
            }
            for scene in scenes
        ],
        "sources": [
            {
                "path": str(path.relative_to(ROOT)),
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
            for path in sorted(set(source_paths))
        ],
        "artifacts": {
            "storyboard": str(storyboard_path.relative_to(ROOT)),
            "poster": str(poster_path.relative_to(ROOT)),
            "captions": str(srt_path.relative_to(ROOT)),
        },
        "rendering": {
            "generated_imagery": False,
            "source_imagery": "existing project imagery and mapped-reference artifacts",
            "encoder": Path(imageio_ffmpeg.get_ffmpeg_exe()).name,
        },
    }
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "video": str(video_path),
                "duration_s": render["duration_s"],
                "storyboard": str(storyboard_path),
                "manifest": str(manifest_path),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
