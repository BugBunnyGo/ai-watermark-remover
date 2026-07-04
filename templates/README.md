# Watermark Templates

This directory holds **PNG watermark templates** that `WatermarkDetector` matches
against uploaded images via `cv2.matchTemplate`. The detector loads every
`*.png` in this folder at startup (see `detector.py::_load_templates`).

## How to add a real template

1. Crop a representative sample of the watermark from an AI-generated image
   (e.g. the corner "Made with AI" tag).
2. Save it as a **square (or near-square) PNG**, grayscale is fine but colour
   is converted to grayscale at load time.
3. The image should be **semi-transparent / low-contrast** — the detector uses
   normalized cross-correlation and works best on the faint overlays that
   watermark systems apply.
4. Drop the file into this directory with a `.png` extension. Filename
   (without extension) becomes the template's `source` label in
   `WatermarkRegion.source`.

## If the directory is empty (current state)

`WatermarkDetector._load_templates` returns an empty list when no `*.png`
files are present. The detector then falls back to `_heuristic_detect`,
which scans the bottom corners for low-contrast rectangular regions.

The app surfaces a status warning when this happens — see `app.py`. Heuristic
mode is less precise than template matching, so adding even one good template
significantly improves detection.

## Notes

- `.gitkeep` is here so the empty directory is tracked by git.
- Real `.png` templates are intentionally **gitignored** (see repo `.gitignore`
  `*.png`) — keep your template library outside the repo or unignore per-file.