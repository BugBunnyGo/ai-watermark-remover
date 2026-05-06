"""AI水印检测模块 - 通过模板匹配和启发式规则定位水印区域"""

import cv2
import numpy as np
import os
from pathlib import Path
from dataclasses import dataclass
from typing import Optional

TEMPLATES_DIR = Path(__file__).parent / "templates"


@dataclass
class WatermarkRegion:
    """检测到的水印区域"""
    x: int
    y: int
    w: int
    h: int
    confidence: float
    source: str  # "template" 或 "heuristic"


class WatermarkDetector:
    def __init__(self, templates_dir: Optional[Path] = None):
        self.templates_dir = templates_dir or TEMPLATES_DIR
        self.templates = self._load_templates()

    def _load_templates(self) -> list[tuple[str, np.ndarray]]:
        """加载预置水印模板"""
        templates = []
        if not self.templates_dir.exists():
            return templates
        for f in self.templates_dir.glob("*.png"):
            tmpl = cv2.imread(str(f), cv2.IMREAD_GRAYSCALE)
            if tmpl is not None:
                templates.append((f.stem, tmpl))
        return templates

    def detect(self, image: np.ndarray) -> list[WatermarkRegion]:
        """检测图片中的水印区域，返回所有匹配结果"""
        results = []
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image

        # 1. 模板匹配
        results.extend(self._template_match(gray))

        # 2. 启发式检测：角落半透明文本区域
        if not results:
            results.extend(self._heuristic_detect(gray))

        # 3. NMS去重
        results = self._nms(results)
        return results

    def _template_match(self, gray: np.ndarray) -> list[WatermarkRegion]:
        """使用模板匹配定位水印"""
        results = []
        h, w = gray.shape

        for name, template in self.templates:
            th, tw = template.shape
            if th > h or tw > w:
                continue

            # 多尺度匹配
            for scale in [1.0, 0.8, 0.6, 0.5]:
                sw, sh = int(tw * scale), int(th * scale)
                if sw < 10 or sh < 10:
                    continue
                resized = cv2.resize(template, (sw, sh))
                result = cv2.matchTemplate(gray, resized, cv2.TM_CCOEFF_NORMED)
                min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)

                if max_val > 0.5:
                    results.append(WatermarkRegion(
                        x=max_loc[0], y=max_loc[1],
                        w=sw, h=sh,
                        confidence=max_val,
                        source=f"template:{name}"
                    ))

        return results

    def _heuristic_detect(self, gray: np.ndarray) -> list[WatermarkRegion]:
        """启发式检测：AI水印通常在图片底部角落"""
        results = []
        h, w = gray.shape

        # 检查底部区域（最后20%高度）
        bottom_region = gray[int(h * 0.8):, :]
        edge_map = cv2.Canny(bottom_region, 50, 150)

        # 分析左下角和右下角
        for x_start, x_end, label in [
            (0, int(w * 0.35), "left_bottom"),
            (int(w * 0.65), w, "right_bottom"),
        ]:
            roi = edge_map[:, x_start:x_end]
            edge_density = np.count_nonzero(roi) / roi.size

            # 高边缘密度 + 文本特征 = 可能的水印
            if edge_density > 0.02:
                # 找到具体边界
                coords = np.argwhere(roi > 0)
                if len(coords) == 0:
                    continue
                y_min, x_min = coords.min(axis=0)
                y_max, x_max = coords.max(axis=0)

                # 扩展边界
                y_min = max(0, y_min - 5)
                x_min = max(0, x_min - 5)
                y_max = min(roi.shape[0] - 1, y_max + 5)
                x_max = min(roi.shape[1] - 1, x_max + 5)

                results.append(WatermarkRegion(
                    x=x_start + x_min,
                    y=int(h * 0.8) + y_min,
                    w=x_max - x_min + 1,
                    h=y_max - y_min + 1,
                    confidence=min(edge_density * 10, 0.9),
                    source=f"heuristic:{label}"
                ))

        return results

    def _nms(self, regions: list[WatermarkRegion], threshold: float = 0.3) -> list[WatermarkRegion]:
        """非极大值抑制，去除重叠检测框"""
        if not regions:
            return []

        regions.sort(key=lambda r: r.confidence, reverse=True)
        kept = []

        for region in regions:
            keep = True
            for k in kept:
                iou = self._compute_iou(region, k)
                if iou > threshold:
                    keep = False
                    break
            if keep:
                kept.append(region)

        return kept

    def _compute_iou(self, a: WatermarkRegion, b: WatermarkRegion) -> float:
        """计算两个区域的IoU"""
        x1 = max(a.x, b.x)
        y1 = max(a.y, b.y)
        x2 = min(a.x + a.w, b.x + b.w)
        y2 = min(a.y + a.h, b.y + b.h)

        inter_w = max(0, x2 - x1)
        inter_h = max(0, y2 - y1)
        inter = inter_w * inter_h

        area_a = a.w * a.h
        area_b = b.w * b.h
        union = area_a + area_b - inter

        return inter / union if union > 0 else 0

    def add_template(self, image: np.ndarray, name: str):
        """从图片中提取水印区域并保存为新模板"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
        self.templates_dir.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(self.templates_dir / f"{name}.png"), gray)
        self.templates = self._load_templates()
