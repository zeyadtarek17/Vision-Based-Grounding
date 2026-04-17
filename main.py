from __future__ import annotations

import argparse
import difflib
import functools
import json
import logging
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable

import cv2
import numpy as np
import pyautogui
import pygetwindow as gw
import requests

try:
    import pytesseract
except Exception:  # pragma: no cover - optional runtime dependency resolution
    pytesseract = None


POSTS_URL = "https://jsonplaceholder.typicode.com/posts"
TARGET_LABEL = "Notepad"
EXPECTED_RESOLUTION = (1920, 1080)


class IconNotFoundError(RuntimeError):
    pass


class ApiUnavailableError(RuntimeError):
    pass


def retry(
    attempts: int = 3,
    delay_seconds: float = 1.0,
    backoff: float = 1.0,
    exceptions: tuple[type[BaseException], ...] = (Exception,),
) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            wait = delay_seconds
            last_error: BaseException | None = None
            for attempt in range(1, attempts + 1):
                try:
                    return func(*args, **kwargs)
                except exceptions as exc:
                    last_error = exc
                    logging.warning(
                        "Attempt %s/%s failed in %s: %s",
                        attempt,
                        attempts,
                        func.__name__,
                        exc,
                    )
                    if attempt < attempts:
                        time.sleep(wait)
                        wait *= backoff
            if last_error is not None:
                raise last_error
            raise RuntimeError("Retry failed without an explicit exception")

        return wrapper

    return decorator


def normalize_text(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", value.lower())


@dataclass
class BoundingBox:
    x: int
    y: int
    w: int
    h: int
    score: float = 0.0
    ocr_text: str = ""
    match_score: float = 0.0

    @property
    def center(self) -> tuple[int, int]:
        return (self.x + self.w // 2, self.y + self.h // 2)


class DesktopGrounder:
    def __init__(self, target_label: str) -> None:
        self.target_label = target_label
        self.target_normalized = normalize_text(target_label)
        self.last_known_box: BoundingBox | None = None

    def capture(self, region: tuple[int, int, int, int] | None = None) -> np.ndarray:
        image = pyautogui.screenshot(region=region)
        return cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

    def _detect_marks(self, image: np.ndarray) -> list[BoundingBox]:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blurred, 60, 170)
        edges = cv2.dilate(edges, np.ones((3, 3), dtype=np.uint8), iterations=1)

        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        marks: list[BoundingBox] = []

        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            area = w * h
            if w < 18 or h < 18 or w > 180 or h > 180:
                continue
            if area < 500 or area > 20000:
                continue

            aspect_ratio = w / max(h, 1)
            if aspect_ratio < 0.5 or aspect_ratio > 1.8:
                continue

            contour_area = cv2.contourArea(contour)
            score = float(contour_area / max(area, 1))
            marks.append(BoundingBox(x=x, y=y, w=w, h=h, score=score))

        marks.sort(key=lambda box: (-box.score, box.y, box.x))
        return marks[:200]

    def _label_crop(self, image: np.ndarray, box: BoundingBox) -> np.ndarray:
        x1 = max(box.x - 28, 0)
        x2 = min(box.x + box.w + 28, image.shape[1])
        y1 = min(box.y + box.h, image.shape[0])
        y2 = min(box.y + box.h + 52, image.shape[0])
        return image[y1:y2, x1:x2]

    def _ocr(self, label_image: np.ndarray) -> str:
        if pytesseract is None or label_image.size == 0:
            return ""

        gray = cv2.cvtColor(label_image, cv2.COLOR_BGR2GRAY)
        gray = cv2.resize(gray, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC)
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        config = (
            "--oem 3 --psm 7 "
            "-c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ"
            "abcdefghijklmnopqrstuvwxyz0123456789()_- "
        )
        text = pytesseract.image_to_string(binary, config=config)
        return text.strip()

    def _text_matches_target(self, text: str) -> tuple[bool, float]:
        normalized = normalize_text(text)
        if not normalized:
            return (False, 0.0)

        ratio = difflib.SequenceMatcher(a=self.target_normalized, b=normalized).ratio()
        contains_target = self.target_normalized in normalized
        starts_with_target = normalized.startswith(self.target_normalized)
        match = contains_target or starts_with_target or ratio >= 0.70

        score = ratio
        if contains_target or starts_with_target:
            score += 0.2
        return (match, min(score, 1.0))

    def _ground_marks(
        self, image: np.ndarray, offset_x: int = 0, offset_y: int = 0
    ) -> BoundingBox | None:
        marks = self._detect_marks(image)
        candidates: list[BoundingBox] = []

        for mark in marks:
            label_crop = self._label_crop(image, mark)
            text = self._ocr(label_crop)
            is_match, match_score = self._text_matches_target(text)
            if not is_match:
                continue

            absolute = BoundingBox(
                x=mark.x + offset_x,
                y=mark.y + offset_y,
                w=mark.w,
                h=mark.h,
                score=mark.score,
                ocr_text=text,
                match_score=match_score,
            )
            candidates.append(absolute)

        if not candidates:
            return None

        candidates.sort(key=lambda item: (item.match_score, item.score), reverse=True)
        return candidates[0]

    def _ocr_label_fallback(
        self, image: np.ndarray, offset_x: int = 0, offset_y: int = 0
    ) -> BoundingBox | None:
        if pytesseract is None:
            return None

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        data = pytesseract.image_to_data(
            gray,
            output_type=pytesseract.Output.DICT,
            config="--oem 3 --psm 11",
        )

        candidates: list[BoundingBox] = []
        for i, text in enumerate(data.get("text", [])):
            is_match, match_score = self._text_matches_target(text)
            if not is_match:
                continue

            tx = int(data["left"][i])
            ty = int(data["top"][i])
            tw = int(data["width"][i])
            th = int(data["height"][i])

            # On desktop, icon graphics are usually above the text label.
            icon_w = 52
            icon_h = 52
            ix = max(tx + (tw // 2) - (icon_w // 2), 0)
            iy = max(ty - (icon_h + 12), 0)

            candidates.append(
                BoundingBox(
                    x=ix + offset_x,
                    y=iy + offset_y,
                    w=icon_w,
                    h=icon_h,
                    score=0.1,
                    ocr_text=text,
                    match_score=match_score,
                )
            )

        if not candidates:
            return None

        candidates.sort(key=lambda item: item.match_score, reverse=True)
        return candidates[0]

    def _local_region(self, padding: int = 250) -> tuple[int, int, int, int] | None:
        if self.last_known_box is None:
            return None

        screen_w, screen_h = pyautogui.size()
        x1 = max(self.last_known_box.x - padding, 0)
        y1 = max(self.last_known_box.y - padding, 0)
        x2 = min(self.last_known_box.x + self.last_known_box.w + padding, screen_w)
        y2 = min(self.last_known_box.y + self.last_known_box.h + padding, screen_h)
        return (x1, y1, max(0, x2 - x1), max(0, y2 - y1))

    def locate(self) -> BoundingBox:
        region = self._local_region()
        if region is not None and region[2] > 0 and region[3] > 0:
            local_frame = self.capture(region=region)
            local_match = self._ground_marks(local_frame, offset_x=region[0], offset_y=region[1])
            if local_match is not None:
                self.last_known_box = local_match
                logging.info(
                    "Local search matched %s at (%s, %s)",
                    local_match.ocr_text,
                    local_match.x,
                    local_match.y,
                )
                return local_match

            local_fallback = self._ocr_label_fallback(
                local_frame, offset_x=region[0], offset_y=region[1]
            )
            if local_fallback is not None:
                self.last_known_box = local_fallback
                logging.info(
                    "Local OCR fallback matched %s at (%s, %s)",
                    local_fallback.ocr_text,
                    local_fallback.x,
                    local_fallback.y,
                )
                return local_fallback

        full_frame = self.capture()
        full_match = self._ground_marks(full_frame)
        if full_match is None:
            full_match = self._ocr_label_fallback(full_frame)
        if full_match is None:
            self.last_known_box = None
            raise IconNotFoundError(
                "Unable to locate Notepad icon via Set-of-Mark detection + OCR."
            )

        self.last_known_box = full_match
        logging.info(
            "Global search matched %s at (%s, %s)",
            full_match.ocr_text,
            full_match.x,
            full_match.y,
        )
        return full_match


class PostClient:
    def __init__(self, base_url: str = POSTS_URL) -> None:
        self.base_url = base_url

    def fetch_first_posts(self, limit: int = 10) -> list[dict[str, Any]]:
        try:
            response = requests.get(self.base_url, timeout=15)
            response.raise_for_status()
        except requests.RequestException as exc:
            raise ApiUnavailableError(f"Failed to fetch posts from API: {exc}") from exc

        payload = response.json()
        return self._normalize_posts(payload, limit=limit)

    def load_posts_from_file(self, file_path: Path, limit: int = 10) -> list[dict[str, Any]]:
        try:
            raw = file_path.read_text(encoding="utf-8")
            payload = json.loads(raw)
        except OSError as exc:
            raise ApiUnavailableError(f"Failed to read local posts file {file_path}: {exc}") from exc
        except json.JSONDecodeError as exc:
            raise ApiUnavailableError(f"Invalid JSON in local posts file {file_path}: {exc}") from exc

        return self._normalize_posts(payload, limit=limit)

    @staticmethod
    def _normalize_posts(payload: Any, limit: int = 10) -> list[dict[str, Any]]:
        if not isinstance(payload, list):
            raise ApiUnavailableError("Unexpected API response format. Expected a list of posts.")

        posts = payload[:limit]
        normalized: list[dict[str, Any]] = []
        for post in posts:
            if not isinstance(post, dict):
                continue
            if "id" not in post or "body" not in post:
                continue

            post_id = post["id"]
            body = str(post["body"])
            title = str(post.get("title", f"Post {post_id}"))
            normalized.append({"id": post_id, "title": title, "body": body})

        if not normalized:
            raise ApiUnavailableError("API returned no usable posts.")
        return normalized


class NotepadAutomation:
    def __init__(self, grounder: DesktopGrounder, output_dir: Path, dry_run: bool = False) -> None:
        self.grounder = grounder
        self.output_dir = output_dir
        self.dry_run = dry_run
        self.output_dir.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _activate_notepad_window(timeout_seconds: float = 8.0) -> bool:
        end_time = time.time() + timeout_seconds
        while time.time() < end_time:
            windows = gw.getWindowsWithTitle("Notepad")
            if windows:
                try:
                    window = windows[0]
                    if window.isMinimized:
                        window.restore()
                    window.activate()
                    time.sleep(0.3)
                    return True
                except Exception:
                    pass
            time.sleep(0.2)
        return False

    @retry(attempts=3, delay_seconds=1.0, backoff=1.0, exceptions=(IconNotFoundError,))
    def _locate_with_retry(self) -> BoundingBox:
        return self.grounder.locate()

    def open_notepad_from_desktop(self) -> None:
        pyautogui.hotkey("win", "d")
        time.sleep(0.6)

        box = self._locate_with_retry()
        x, y = box.center

        if self.dry_run:
            logging.info("[DRY-RUN] Located Notepad icon at (%s, %s). Skipping double-click.", x, y)
            return

        pyautogui.doubleClick(x, y, interval=0.20)
        time.sleep(1.0)

        if not self._activate_notepad_window():
            raise RuntimeError("Notepad window did not appear after opening icon.")

    def type_post_content(self, body: str) -> None:
        if self.dry_run:
            logging.info("[DRY-RUN] Skipping typing %s characters.", len(body))
            return

        pyautogui.hotkey("ctrl", "a")
        pyautogui.press("backspace")
        pyautogui.write(body, interval=0.01)

    def save_current_post(self, post_id: int) -> Path:
        output_file = self.output_dir / f"post_{post_id}.txt"

        if self.dry_run:
            logging.info("[DRY-RUN] Skipping save. Would write: %s", output_file)
            return output_file

        if output_file.exists():
            try:
                output_file.unlink()
                logging.info("Existing file removed before save: %s", output_file)
            except OSError as exc:
                logging.warning("Could not remove existing file %s: %s", output_file, exc)

        def _attempt_save(use_save_as: bool) -> None:
            if use_save_as:
                pyautogui.hotkey("ctrl", "shift", "s")
            else:
                pyautogui.hotkey("ctrl", "s")

            time.sleep(1.0)
            pyautogui.write(str(output_file), interval=0.01)
            pyautogui.press("enter")
            time.sleep(1.0)

            overwrite_dialogs = gw.getWindowsWithTitle("Confirm Save As")
            if overwrite_dialogs:
                pyautogui.press("left")
                pyautogui.press("enter")
                time.sleep(0.7)

        _attempt_save(use_save_as=False)
        if output_file.exists():
            return output_file

        logging.warning("Primary save did not create file. Retrying with Save As.")
        _attempt_save(use_save_as=True)

        end_time = time.time() + 5.0
        while time.time() < end_time:
            if output_file.exists():
                return output_file
            time.sleep(0.25)

        raise RuntimeError(f"Save failed. File was not created: {output_file}")

    def close_notepad(self) -> None:
        if self.dry_run:
            logging.info("[DRY-RUN] Skipping close action.")
            return

        windows = gw.getWindowsWithTitle("Notepad")
        if not windows:
            return

        try:
            windows[0].activate()
        except Exception:
            pass

        pyautogui.hotkey("alt", "f4")
        time.sleep(0.6)


def resolve_desktop_dir() -> Path:
    candidates: Iterable[Path] = (
        Path.home() / "Desktop",
        Path.home() / "OneDrive" / "Desktop",
    )
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return Path.home()


def configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%H:%M:%S",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Automate Notepad writing with Set-of-Mark grounding + OCR."
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Run detection and workflow logging without clicking, typing, or saving.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=10,
        help="Number of posts to process (default: 10).",
    )
    parser.add_argument(
        "--posts-file",
        type=Path,
        default=None,
        help="Optional local JSON file with posts. If set, skips API fetch.",
    )
    parser.add_argument(
        "--offline",
        action="store_true",
        help="Use local sample posts only (posts.sample.json), never call API.",
    )
    return parser.parse_args()


def run(*, dry_run: bool = False, limit: int = 10, posts_file: Path | None = None, offline: bool = False) -> None:
    configure_logging()
    pyautogui.FAILSAFE = True
    pyautogui.PAUSE = 0.15

    if limit < 1:
        logging.error("Invalid --limit value: %s. Must be >= 1.", limit)
        return

    if dry_run:
        logging.info("Dry-run mode enabled. GUI actions are disabled.")

    if offline and posts_file is not None:
        logging.error("Use either --offline or --posts-file, not both.")
        return

    screen_size = pyautogui.size()
    if (screen_size.width, screen_size.height) != EXPECTED_RESOLUTION:
        logging.warning(
            "Expected screen resolution %sx%s, found %sx%s. Continuing anyway.",
            EXPECTED_RESOLUTION[0],
            EXPECTED_RESOLUTION[1],
            screen_size.width,
            screen_size.height,
        )

    api = PostClient()
    sample_posts_path = Path(__file__).with_name("posts.sample.json")
    try:
        if posts_file is not None:
            posts = api.load_posts_from_file(posts_file, limit=limit)
            logging.info("Loaded %s posts from local file: %s", len(posts), posts_file)
        elif offline:
            posts = api.load_posts_from_file(sample_posts_path, limit=limit)
            logging.info("Offline mode enabled. Loaded %s sample posts.", len(posts))
        else:
            posts = api.fetch_first_posts(limit=limit)
    except ApiUnavailableError as exc:
        if posts_file is None and not offline and sample_posts_path.exists():
            logging.warning("API unavailable (%s). Falling back to local sample posts.", exc)
            try:
                posts = api.load_posts_from_file(sample_posts_path, limit=limit)
                logging.info("Loaded %s sample posts from: %s", len(posts), sample_posts_path)
            except ApiUnavailableError as fallback_exc:
                logging.error("Aborting. Local fallback also failed: %s", fallback_exc)
                return
        else:
            logging.error("Aborting. Could not load posts: %s", exc)
            return

    desktop_dir = resolve_desktop_dir()
    output_dir = desktop_dir / "tjm-project"

    grounder = DesktopGrounder(target_label=TARGET_LABEL)
    automator = NotepadAutomation(grounder=grounder, output_dir=output_dir, dry_run=dry_run)

    for post in posts:
        post_id = int(post["id"])
        title = str(post.get("title", f"Post {post_id}")).strip() or f"Post {post_id}"
        body = str(post["body"])
        content = f"Title: {title}\n\n{body}"

        try:
            logging.info("Processing post id=%s", post_id)
            automator.open_notepad_from_desktop()
            automator.type_post_content(content)
            saved_file = automator.save_current_post(post_id=post_id)
            logging.info("Saved: %s", saved_file)
        except Exception as exc:
            logging.error("Post id=%s failed: %s", post_id, exc)
        finally:
            automator.close_notepad()
            time.sleep(0.5)

    logging.info("Completed automation run.")


if __name__ == "__main__":
    cli_args = parse_args()
    run(
        dry_run=cli_args.dry_run,
        limit=cli_args.limit,
        posts_file=cli_args.posts_file,
        offline=cli_args.offline,
    )
