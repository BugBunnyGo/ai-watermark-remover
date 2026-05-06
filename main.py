#!/usr/bin/env python3
"""CLI入口 - AI水印识别与去除"""

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

from detector import WatermarkDetector
from remover import WatermarkRemover


def visualize_detections(image: np.ndarray, regions) -> np.ndarray:
    """在图片上标注检测到的水印区域"""
    vis = image.copy()
    for region in regions:
        cv2.rectangle(vis, (region.x, region.y),
                      (region.x + region.w, region.y + region.h),
                      (0, 0, 255), 2)
        label = f"{region.source} ({region.confidence:.2f})"
        cv2.putText(vis, label, (region.x, region.y - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
    return vis


def manual_select(image: np.ndarray) -> tuple[int, int, int, int]:
    """手动框选水印区域"""
    clone = image.copy()
    ref_point = []

    def mouse_callback(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            ref_point.clear()
            ref_point.append((x, y))
        elif event == cv2.EVENT_LBUTTONUP:
            ref_point.append((x, y))

    cv2.namedWindow("Select watermark region (drag to select)")
    cv2.setMouseCallback("Select watermark region (drag to select)", mouse_callback)

    print("Drag to select watermark area, then press any key...")
    while True:
        display = clone.copy()
        if len(ref_point) == 2:
            cv2.rectangle(display, ref_point[0], ref_point[1], (0, 255, 0), 2)
        elif len(ref_point) == 1:
            cv2.circle(display, ref_point[0], 3, (0, 255, 0), -1)
        cv2.imshow("Select watermark region (drag to select)", display)
        if len(ref_point) == 2:
            break
        if cv2.waitKey(1) & 0xFF != 255:
            pass

    cv2.destroyAllWindows()

    x1, y1 = ref_point[0]
    x2, y2 = ref_point[1]
    x = min(x1, x2)
    y = min(y1, y2)
    w = abs(x2 - x1)
    h = abs(y2 - y1)
    return x, y, w, h


def main():
    parser = argparse.ArgumentParser(description="AI水印识别与去除工具")
    parser.add_argument("input", help="输入图片路径")
    parser.add_argument("-o", "--output", help="输出图片路径（默认: input_clean.jpg）")
    parser.add_argument("--manual", action="store_true", help="手动框选水印区域")
    parser.add_argument("--train", help="从当前图片提取水印模板并保存")
    parser.add_argument("--show", action="store_true", help="显示检测结果")
    args = parser.parse_args()

    # 读取图片
    image = cv2.imread(args.input)
    if image is None:
        print(f"错误: 无法读取图片 {args.input}")
        sys.exit(1)

    detector = WatermarkDetector()
    remover = WatermarkRemover()

    if args.manual:
        # 手动模式
        bbox = manual_select(image)
        result = remover.remove_manual(image, bbox)
    else:
        # 自动检测模式
        regions = detector.detect(image)
        if not regions:
            print("未检测到水印，请使用 --manual 手动框选")
            if args.show:
                cv2.imshow("No watermark detected", image)
                cv2.waitKey(0)
                cv2.destroyAllWindows()
            sys.exit(0)

        print(f"检测到 {len(regions)} 个水印区域:")
        for r in regions:
            print(f"  - 位置: ({r.x},{r.y}) 大小: {r.w}x{r.h} 置信度: {r.confidence:.2f} 来源: {r.source}")

        if args.show:
            vis = visualize_detections(image, regions)
            cv2.imshow("Watermark detection", vis)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

        result = remover.remove(image, regions)

    # 训练模式：保存当前水印区域为模板
    if args.train:
        for i, region in enumerate(detector.detect(image)):
            roi = image[region.y:region.y+region.h, region.x:region.x+region.w]
            detector.add_template(roi, f"{args.train}_{i}")
            print(f"已保存水印模板: {args.train}_{i}")

    # 输出结果
    output = args.output or str(Path(args.input).stem) + "_clean" + Path(args.input).suffix
    cv2.imwrite(output, result)
    print(f"已保存到: {output}")


if __name__ == "__main__":
    main()
