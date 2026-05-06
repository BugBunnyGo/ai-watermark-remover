"""水印去除模块 - 使用图像修复算法去除水印"""

import cv2
import numpy as np
from detector import WatermarkRegion


class WatermarkRemover:
    def __init__(self, inpaint_radius: int = 5, algorithm: str = "telea"):
        self.inpaint_radius = inpaint_radius
        self.algorithm = cv2.INPAINT_TELEA if algorithm == "telea" else cv2.INPAINT_NS

    def remove(self, image: np.ndarray, regions: list[WatermarkRegion]) -> np.ndarray:
        """对检测到的水印区域进行修复"""
        result = image.copy()
        h, w = image.shape[:2]

        for region in regions:
            # 扩展水印区域边界，确保完整覆盖
            pad = max(region.w, region.h) // 4
            x1 = max(0, region.x - pad)
            y1 = max(0, region.y - pad)
            x2 = min(w, region.x + region.w + pad)
            y2 = min(h, region.y + region.h + pad)

            # 创建掩码
            mask = np.zeros((h, w), dtype=np.uint8)
            mask[y1:y2, x1:x2] = 255

            # 图像修复
            result = cv2.inpaint(result, mask, self.inpaint_radius, self.algorithm)

        return result

    def remove_manual(self, image: np.ndarray, bbox: tuple[int, int, int, int]) -> np.ndarray:
        """手动指定区域去除水印 (x, y, w, h)"""
        x, y, w, h = bbox
        result = image.copy()
        img_h, img_w = image.shape[:2]

        # 边界检查
        x = max(0, x)
        y = max(0, y)
        w = min(w, img_w - x)
        h = min(h, img_h - y)

        if w <= 0 or h <= 0:
            return result

        mask = np.zeros((img_h, img_w), dtype=np.uint8)
        mask[y:y+h, x:x+w] = 255

        return cv2.inpaint(result, mask, self.inpaint_radius, self.algorithm)
