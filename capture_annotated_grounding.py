from __future__ import annotations

import argparse
import time
from datetime import datetime
from pathlib import Path

import cv2
import pyautogui

from main import DesktopGrounder, TARGET_LABEL


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Capture an annotated screenshot with detected Notepad icon coordinates."
    )
    parser.add_argument(
        "--tag",
        default="capture",
        help="Tag used in output filename, e.g., top_left, center, bottom_right.",
    )
    parser.add_argument(
        "--output-dir",
        default="docs/screenshots",
        help="Directory where annotated screenshots are saved.",
    )
    return parser.parse_args()


def draw_annotation(image, x: int, y: int, w: int, h: int, text: str):
    annotated = image.copy()

    # Bounding box around estimated icon area.
    cv2.rectangle(annotated, (x, y), (x + w, y + h), (40, 220, 40), 2)

    cx = x + w // 2
    cy = y + h // 2
    cv2.circle(annotated, (cx, cy), 5, (0, 0, 255), -1)

    label_bg_tl = (max(0, x - 2), max(0, y - 32))
    label_bg_br = (min(annotated.shape[1], x + 440), y)
    cv2.rectangle(annotated, label_bg_tl, label_bg_br, (0, 0, 0), -1)
    cv2.putText(
        annotated,
        text,
        (x + 4, y - 10),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )

    return annotated


def main() -> int:
    args = parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    pyautogui.FAILSAFE = True
    pyautogui.PAUSE = 0.12

    # Ensure desktop is visible to reduce false negatives from overlapping windows.
    pyautogui.hotkey("win", "d")
    time.sleep(0.6)

    grounder = DesktopGrounder(target_label=TARGET_LABEL)
    match = grounder.locate()

    frame = grounder.capture()
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    label = (
        f"target={TARGET_LABEL} text='{match.ocr_text}' x={match.center[0]} y={match.center[1]} "
        f"score={match.match_score:.2f}"
    )

    annotated = draw_annotation(frame, match.x, match.y, match.w, match.h, label)

    output_file = output_dir / f"{args.tag}_{stamp}.png"
    cv2.imwrite(str(output_file), annotated)

    print(f"saved={output_file}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
