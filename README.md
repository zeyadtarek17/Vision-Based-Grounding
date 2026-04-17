# Notepad Grounding Automation (uv)

Python desktop automation demo that uses a Set-of-Mark style grounding approach to find the **Notepad** desktop icon and automate writing/saving text files from API posts.

## What It Does

For the first 10 posts from JSONPlaceholder:

1. Capture desktop screenshot.
2. Detect icon-like marks using bounding boxes (OpenCV contour pipeline).
3. OCR each candidate label and semantically match `Notepad` (with fuzzy tolerance).
4. Double-click the icon.
5. Type post content as `Title: {title}\n\n{body}`.
6. Save as `Desktop/tjm-project/post_{id}.txt`.
7. Close Notepad.
8. Re-locate Notepad each iteration.

## Why This Is Grounding (Not Template Matching)

- Template matching is pixel-sensitive and breaks under scale/theme/transparency shifts.
- This implementation uses:
  - **Set-of-Mark detection** (candidate regions)
  - **OCR semantic check** (label text near icon)
  - **Fuzzy text matching** (e.g., `Notepad (2)` still likely matches)

## Requirements

- Windows desktop environment
- Python 3.11+
- Screen resolution target: `1920x1080` (script warns if different)
- Tesseract OCR installed and available on PATH
- uv (recommended) for dependency management

## Install Prerequisites

### 1) Install uv

If `uv --version` fails, install uv first:

```powershell
winget install --id Astral-sh.uv -e
```

If `winget` is not available on your machine:

```powershell
python -m pip install uv
```

Close and reopen terminal, then verify:

```powershell
uv --version
```

### 2) Install Tesseract OCR

Option A (winget):

```powershell
winget install --id UB-Mannheim.TesseractOCR -e
```

Option B (Chocolatey):

```powershell
choco install tesseract
```

Verify:

```powershell
tesseract --version
```

If `tesseract` is not found, add install directory to PATH, commonly:

- `C:\Program Files\Tesseract-OCR`

## Project Setup (uv)

From project root:

```powershell
uv sync
```

Run the automation:

```powershell
uv run python main.py
```

Run interview-safe dry-run mode (no clicks, typing, save, or close actions):

```powershell
uv run python main.py --dry-run
```

Optional: process fewer posts during demos:

```powershell
uv run python main.py --dry-run --limit 3
```

Offline mode (no API calls, uses local sample data in `posts.sample.json`):

```powershell
uv run python main.py --offline --dry-run --limit 3
```

Use a custom local JSON file instead of API:

```powershell
uv run python main.py --posts-file .\posts.sample.json --dry-run --limit 3
```

## Notes on Safety and Reliability

- PyAutoGUI fail-safe is enabled (`move mouse to top-left corner to abort`).
- Notepad icon detection uses retry decorator (3 attempts).
- API failures are handled cleanly with logging and graceful exit.
- Local-search-first optimization reduces repeated full-screen scans.

## Interview Discussion Pointers

- **Method choice:** Grounding + OCR is more robust than static templates.
- **Performance:** Local search around last known icon position before full scan.
- **Failure modes:** Icon occlusion, renamed labels, temporary OCR miss, covered desktop.
- **Scalability:** To automate another app (e.g., Chrome), change target label string.

## Dependencies

Defined in [pyproject.toml](pyproject.toml):

- pyautogui
- opencv-python
- requests
- pygetwindow
- numpy
- pillow
- pytesseract
