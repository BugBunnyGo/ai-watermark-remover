#!/usr/bin/env python3
"""Web界面 - AI水印识别与去除"""

import gradio as gr
import cv2
import numpy as np
from pathlib import Path
import tempfile

from detector import WatermarkDetector
from remover import WatermarkRemover


# --- Configuration ---------------------------------------------------------

# 10 MB cap on raw upload bytes; Gradio's Image component hands us a numpy
# array, but when called from the /upload endpoint or file-picker path we may
# receive a filepath-like value that we still need to validate.
MAX_UPLOAD_BYTES = 10 * 1024 * 1024

# Pillow / cv2 accept these; anything else is rejected up front.
_ALLOWED_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".webp", ".tif", ".tiff"}


# --- Module-level singletons ----------------------------------------------

_detector: WatermarkDetector | None = None
_remover: WatermarkRemover | None = None
_templates_warning: str | None = None


def _get_detector() -> WatermarkDetector:
    """Lazily construct and cache the detector so templates load exactly once."""
    global _detector, _templates_warning
    if _detector is None:
        _detector = WatermarkDetector()
        if not _detector.templates:
            _templates_warning = (
                "未加载到任何水印模板（templates/ 目录为空），"
                "已切换到启发式检测模式，精度会下降。"
                "请参考 templates/README.md 添加真实模板。"
            )
    return _detector


def _get_remover() -> WatermarkRemover:
    global _remover
    if _remover is None:
        _remover = WatermarkRemover()
    return _remover


# --- Validation helpers ----------------------------------------------------

def _validate_upload(image) -> tuple[bool, str]:
    """Return (ok, message). Message is empty when ok; otherwise a status string."""
    if image is None:
        return False, "请先上传图片"

    # Gradio's gr.Image(type="numpy") yields an ndarray, so size in pixels.
    # If a filepath-like slips through, reject based on extension + on-disk size.
    if isinstance(image, (str, Path)):
        p = Path(image)
        if p.suffix.lower() not in _ALLOWED_EXTS:
            return False, f"不支持的文件类型: {p.suffix or '(无后缀)'}，仅接受图片文件"
        try:
            size = p.stat().st_size
        except OSError:
            return False, f"无法读取上传文件: {p}"
        if size > MAX_UPLOAD_BYTES:
            mb = size / (1024 * 1024)
            return False, f"文件过大 ({mb:.1f} MB)，上限 {MAX_UPLOAD_BYTES // (1024 * 1024)} MB"
        return True, ""

    if isinstance(image, np.ndarray):
        # Sanity-check shape: must be 2-D grayscale or 3-D HWC.
        if image.ndim not in (2, 3):
            return False, f"图片维度异常: shape={image.shape}"
        if image.size == 0:
            return False, "图片为空，请重新上传"
        return True, ""

    return False, f"不支持的上传类型: {type(image).__name__}"


# --- Core processing -------------------------------------------------------

def process_image(image, auto_detect=True, manual_bbox=None):
    """处理图片并返回对比结果"""
    ok, msg = _validate_upload(image)
    if not ok:
        return None, None, msg

    # Gradio传入的是RGB，转为BGR；防御性处理灰度图与cvtColor失败。
    try:
        if isinstance(image, np.ndarray):
            if image.ndim == 2:
                image_bgr = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
            elif image.shape[2] == 4:
                image_bgr = cv2.cvtColor(image, cv2.COLOR_RGBA2BGR)
            else:
                image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        else:
            return None, None, "图片格式无法解析，请重新上传"
    except cv2.error as exc:
        return None, None, f"图片色彩空间转换失败: {exc}"

    detector = _get_detector()
    remover = _get_remover()

    # Surface the empty-templates warning on the first request only.
    warning = _templates_warning
    if warning is not None:
        global_msg = warning
    else:
        global_msg = ""

    if auto_detect:
        regions = detector.detect(image_bgr)
        if regions:
            result = remover.remove(image_bgr, regions)
            # 标注检测结果
            vis = image_bgr.copy()
            for r in regions:
                cv2.rectangle(vis, (r.x, r.y), (r.x + r.w, r.y + r.h), (0, 0, 255), 2)
            vis_rgb = cv2.cvtColor(vis, cv2.COLOR_BGR2RGB)
            result_rgb = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)
            info = f"检测到 {len(regions)} 个水印区域，已自动去除"
            if global_msg:
                info = f"{global_msg}\n{info}"
            return vis_rgb, result_rgb, info
        else:
            info = "未检测到水印，请使用手动模式框选水印区域"
            if global_msg:
                info = f"{global_msg}\n{info}"
            # Show the original RGB image back to the user.
            original_rgb = image if isinstance(image, np.ndarray) else image_bgr[:, :, ::-1]
            return original_rgb, None, info
    else:
        if manual_bbox:
            x, y, w, h = manual_bbox
            result = remover.remove_manual(image_bgr, (x, y, w, h))
            result_rgb = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)
            info = "手动区域已去除"
            if global_msg:
                info = f"{global_msg}\n{info}"
            original_rgb = image if isinstance(image, np.ndarray) else image_bgr[:, :, ::-1]
            return original_rgb, result_rgb, info
        info = "请在下方输入水印区域坐标 (x, y, w, h)"
        if global_msg:
            info = f"{global_msg}\n{info}"
        original_rgb = image if isinstance(image, np.ndarray) else image_bgr[:, :, ::-1]
        return original_rgb, None, info


with gr.Blocks(title="AI水印去除工具") as app:
    gr.Markdown("# AI水印识别与去除工具")
    gr.Markdown("上传图片，自动检测并去除AI生成标识")

    with gr.Row():
        with gr.Column():
            input_img = gr.Image(label="原始图片")
            auto_detect = gr.Checkbox(label="自动检测", value=True)
            bbox_input = gr.Textbox(label="手动区域 (x, y, w, h)", placeholder="例如: 100, 200, 150, 50")
            process_btn = gr.Button("处理", variant="primary")

        with gr.Column():
            detected_img = gr.Image(label="检测结果")
            result_img = gr.Image(label="去除结果")

    status = gr.Textbox(label="状态")

    def on_process(img, auto, bbox_text):
        bbox = None
        if bbox_text.strip():
            try:
                bbox = tuple(int(x.strip()) for x in bbox_text.split(","))
            except ValueError:
                pass
        return process_image(img, auto, bbox)

    process_btn.click(
        on_process,
        inputs=[input_img, auto_detect, bbox_input],
        outputs=[detected_img, result_img, status]
    )


if __name__ == "__main__":
    app.launch(server_name="0.0.0.0", server_port=7860)