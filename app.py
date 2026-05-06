#!/usr/bin/env python3
"""Web界面 - AI水印识别与去除"""

import gradio as gr
import cv2
import numpy as np
from pathlib import Path
import tempfile

from detector import WatermarkDetector
from remover import WatermarkRemover


def process_image(image, auto_detect=True, manual_bbox=None):
    """处理图片并返回对比结果"""
    if image is None:
        return None, None, "请先上传图片"

    # Gradio传入的是RGB，转为BGR
    image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    detector = WatermarkDetector()
    remover = WatermarkRemover()

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
            return vis_rgb, result_rgb, info
        else:
            info = "未检测到水印，请使用手动模式框选水印区域"
            return image, None, info
    else:
        if manual_bbox:
            x, y, w, h = manual_bbox
            result = remover.remove_manual(image_bgr, (x, y, w, h))
            result_rgb = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)
            return image, result_rgb, "手动区域已去除"
        return image, None, "请在下方输入水印区域坐标 (x, y, w, h)"


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
