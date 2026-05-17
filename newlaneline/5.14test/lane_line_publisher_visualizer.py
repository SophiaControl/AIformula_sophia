#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import cv2
import numpy as np


LEFT_INDEX = 0
RIGHT_INDEX = 1
CENTER_INDEX = 2
SIDE_NAMES = {LEFT_INDEX: "left", RIGHT_INDEX: "right", CENTER_INDEX: "center"}

DEFAULT_MASK_DIR = r"C:\Users\17396\Desktop\newlaneline\5.14test\masks"
DEFAULT_OUT_DIR = r"C:\Users\17396\Desktop\newlaneline\lane_line_visualization_output"

@dataclass
class ImageLineFit:
    valid: bool = False
    a: float = 0.0
    b: float = 0.0
    lost_count: int = 0

    def copy(self) -> "ImageLineFit":
        return ImageLineFit(self.valid, self.a, self.b, self.lost_count)


@dataclass
class SideDebugInfo:
    use_tracking_roi: bool = False
    has_line: bool = False
    line: ImageLineFit = field(default_factory=ImageLineFit)
    extracted_pixels: np.ndarray = field(default_factory=lambda: np.empty((0, 2), dtype=np.int32))


@dataclass
class SideResult:
    roi_mode: str = "init"
    status: str = "not_processed"
    pixels: np.ndarray = field(default_factory=lambda: np.empty((0, 2), dtype=np.int32))
    vehicle_points: np.ndarray = field(default_factory=lambda: np.empty((0, 3), dtype=np.float64))
    fitted_points: np.ndarray = field(default_factory=lambda: np.empty((0, 3), dtype=np.float64))
    current_line: ImageLineFit = field(default_factory=ImageLineFit)
    angle_deg: float = 0.0
    lost_count: int = 0


@dataclass
class FrameResult:
    cleaned_mask: np.ndarray
    annotated_mask: np.ndarray
    side_results: Dict[int, SideResult]
    center_points: np.ndarray
    contour_points: Dict[int, np.ndarray]


@dataclass
class Config:
    min_area: int = 100
    min_required_pixels: int = 75
    max_lost_frames: int = 5
    init_top_crop_ratio: float = 0.45
    tolerance: int = 10
    line_update_alpha: float = 0.75
    min_line_angle_deg: float = 12.0
    max_line_angle_deg: float = 168.0
    xmin: float = 1.5
    xmax: float = 10.0
    ymin: float = -3.0
    ymax: float = 3.0
    spacing: float = 0.5
    camera_matrix: Optional[np.ndarray] = None
    vehicle_R_camera: Optional[np.ndarray] = None
    vehicle_t_camera: Optional[np.ndarray] = None

    @property
    def margin_far(self) -> float:
        return max(20.0, 0.6 * float(max(1, self.tolerance)))

    @property
    def margin_near(self) -> float:
        return max(60.0, 2.0 * float(max(1, self.tolerance)))

    @property
    def has_vehicle_projection(self) -> bool:
        return (
            self.camera_matrix is not None
            and self.vehicle_R_camera is not None
            and self.vehicle_t_camera is not None
        )


class LaneLinePublisherEquivalent:
    def __init__(self, config: Config) -> None:
        self.cfg = config
        self.stored_lines = {
            LEFT_INDEX: ImageLineFit(),
            RIGHT_INDEX: ImageLineFit(),
        }
        self.debug_info = {
            LEFT_INDEX: SideDebugInfo(),
            RIGHT_INDEX: SideDebugInfo(),
        }

    def find_lane_lines(self, mask: np.ndarray) -> FrameResult:
        binary_mask = to_binary_mask(mask)
        cleaned_mask = remove_small_components(binary_mask, self.cfg.min_area)

        side_results: Dict[int, SideResult] = {}
        side_output_valid = {LEFT_INDEX: False, RIGHT_INDEX: False}
        contour_points: Dict[int, np.ndarray] = {
            LEFT_INDEX: empty_points3(),
            RIGHT_INDEX: empty_points3(),
            CENTER_INDEX: empty_points3(),
        }

        for side in (LEFT_INDEX, RIGHT_INDEX):
            result = self._process_side(cleaned_mask, side)
            side_results[side] = result
            contour_points[side] = result.vehicle_points
            side_output_valid[side] = result.fitted_points.size > 0

        center_points = empty_points3()
        if side_output_valid[LEFT_INDEX] and side_output_valid[RIGHT_INDEX]:
            center_points = make_center_from_sides(
                side_results[LEFT_INDEX].fitted_points,
                side_results[RIGHT_INDEX].fitted_points,
            )
            contour_points[CENTER_INDEX] = center_points

        annotated_mask = self.make_annotated_mask_image(cleaned_mask)
        return FrameResult(
            cleaned_mask=cleaned_mask,
            annotated_mask=annotated_mask,
            side_results=side_results,
            center_points=center_points,
            contour_points=contour_points,
        )

    def _process_side(self, cleaned_mask: np.ndarray, side: int) -> SideResult:
        stored_line = self.stored_lines[side]
        use_tracking_roi = stored_line.valid and stored_line.lost_count <= self.cfg.max_lost_frames
        extracted_pixels = (
            extract_tracking_pixels(cleaned_mask, stored_line, self.cfg)
            if use_tracking_roi
            else extract_init_pixels(cleaned_mask, side, self.cfg)
        )

        self.debug_info[side] = SideDebugInfo(
            use_tracking_roi=use_tracking_roi,
            has_line=use_tracking_roi,
            line=stored_line.copy(),
            extracted_pixels=extracted_pixels,
        )

        result = SideResult(
            roi_mode="tracking" if use_tracking_roi else "init",
            pixels=extracted_pixels,
        )

        current_line, angle_deg, fit_ok = fit_image_line_uv(extracted_pixels, self.cfg)
        result.current_line = current_line
        result.angle_deg = angle_deg

        reject_horizontal = fit_ok and is_nearly_horizontal(angle_deg, self.cfg)
        reject_jump = fit_ok and has_excessive_jump(
            stored_line, current_line, cleaned_mask.shape[0], cleaned_mask.shape[1], self.cfg
        )

        if not fit_ok:
            result.status = "too_few_or_degenerate_pixels"
            self.register_detection_failure(stored_line)
            result.lost_count = stored_line.lost_count
            return result
        if reject_horizontal:
            result.status = "rejected_horizontal"
            self.register_detection_failure(stored_line)
            result.lost_count = stored_line.lost_count
            return result
        if reject_jump:
            result.status = "rejected_excessive_jump"
            self.register_detection_failure(stored_line)
            result.lost_count = stored_line.lost_count
            return result

        if not self.cfg.has_vehicle_projection:
            result.status = "image_fit_ok_projection_disabled"
            self.update_stored_line(stored_line, current_line)
            result.lost_count = stored_line.lost_count
            return result

        vehicle_points = pixels_to_vehicle_points(extracted_pixels, self.cfg)
        vehicle_points = crop_vehicle_points(vehicle_points, self.cfg)
        result.vehicle_points = vehicle_points

        if len(vehicle_points) < 3:
            result.status = "too_few_vehicle_points_after_crop"
            self.register_detection_failure(stored_line)
            result.lost_count = stored_line.lost_count
            return result

        fitted_points = fit_quadratic_vehicle_line(vehicle_points, self.cfg)
        if fitted_points.size == 0:
            result.status = "quadratic_fit_failed"
            self.register_detection_failure(stored_line)
            result.lost_count = stored_line.lost_count
            return result

        result.fitted_points = fitted_points
        result.status = "ok"
        self.update_stored_line(stored_line, current_line)
        result.lost_count = stored_line.lost_count
        return result

    def update_stored_line(self, stored_line: ImageLineFit, current_line: ImageLineFit) -> None:
        if not stored_line.valid:
            stored_line.valid = current_line.valid
            stored_line.a = current_line.a
            stored_line.b = current_line.b
        else:
            alpha = self.cfg.line_update_alpha
            stored_line.a = alpha * current_line.a + (1.0 - alpha) * stored_line.a
            stored_line.b = alpha * current_line.b + (1.0 - alpha) * stored_line.b
            stored_line.valid = True
        stored_line.lost_count = 0

    def register_detection_failure(self, stored_line: ImageLineFit) -> None:
        stored_line.lost_count += 1
        if stored_line.lost_count > self.cfg.max_lost_frames:
            stored_line.valid = False

    def make_annotated_mask_image(self, cleaned_mask: np.ndarray) -> np.ndarray:
        annotated = cv2.cvtColor(cleaned_mask, cv2.COLOR_GRAY2BGR)
        for side in (LEFT_INDEX, RIGHT_INDEX):
            info = self.debug_info[side]
            if info.use_tracking_roi and info.has_line:
                draw_tracking_roi_border(annotated, info.line, side, self.cfg)
            else:
                draw_init_roi_border(annotated, side, self.cfg)
            draw_sparse_pixels(annotated, info.extracted_pixels, side)
        return annotated


def empty_points3() -> np.ndarray:
    return np.empty((0, 3), dtype=np.float64)


def clamp_int(value: int, low: int, high: int) -> int:
    return max(low, min(value, high))


def clamp_float(value: float, low: float, high: float) -> float:
    return max(low, min(value, high))


def top_row(rows: int, cfg: Config) -> int:
    return clamp_int(round(rows * cfg.init_top_crop_ratio), 0, max(0, rows - 1))


def margin_at_row(row: np.ndarray | int, rows: int, cfg: Config) -> np.ndarray | float:
    v_top = top_row(rows, cfg)
    v_bottom = max(v_top + 1, rows - 1)
    ratio = (np.asarray(row, dtype=np.float64) - float(v_top)) / float(v_bottom - v_top)
    ratio = np.clip(ratio, 0.0, 1.0)
    return cfg.margin_far + (cfg.margin_near - cfg.margin_far) * ratio


def to_binary_mask(mask: np.ndarray) -> np.ndarray:
    if mask.ndim == 3 and mask.shape[2] == 3:
        gray = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    elif mask.ndim == 3 and mask.shape[2] == 4:
        gray = cv2.cvtColor(mask, cv2.COLOR_BGRA2GRAY)
    else:
        gray = mask

    if gray.dtype != np.uint8:
        gray = gray.astype(np.uint8)
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY)
    return binary


def remove_small_components(binary_mask: np.ndarray, min_area: int) -> np.ndarray:
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary_mask, 8, cv2.CV_32S)
    cleaned = np.zeros(binary_mask.shape, dtype=np.uint8)
    for label in range(1, num_labels):
        area = int(stats[label, cv2.CC_STAT_AREA])
        if area >= min_area:
            cleaned[labels == label] = 255
    return cleaned


def extract_init_pixels(cleaned_mask: np.ndarray, side: int, cfg: Config) -> np.ndarray:
    rows, cols = cleaned_mask.shape[:2]
    v_top = top_row(rows, cfg)
    u_begin = 0 if side == LEFT_INDEX else cols // 2
    u_end = cols // 2 if side == LEFT_INDEX else cols
    roi = cleaned_mask[v_top:rows, u_begin:u_end]
    v_local, u_local = np.nonzero(roi)
    if len(u_local) == 0:
        return np.empty((0, 2), dtype=np.int32)
    return np.column_stack((u_local + u_begin, v_local + v_top)).astype(np.int32)


def extract_tracking_pixels(cleaned_mask: np.ndarray, previous_line: ImageLineFit, cfg: Config) -> np.ndarray:
    rows, _cols = cleaned_mask.shape[:2]
    v_top = top_row(rows, cfg)
    v_local, u = np.nonzero(cleaned_mask[v_top:rows, :])
    if len(u) == 0:
        return np.empty((0, 2), dtype=np.int32)

    v = v_local + v_top
    u_pred = previous_line.a * v.astype(np.float64) + previous_line.b
    margin = margin_at_row(v, rows, cfg)
    keep = (u >= np.floor(u_pred - margin)) & (u <= np.ceil(u_pred + margin))
    if not np.any(keep):
        return np.empty((0, 2), dtype=np.int32)
    return np.column_stack((u[keep], v[keep])).astype(np.int32)


def fit_image_line_uv(pixels: np.ndarray, cfg: Config) -> Tuple[ImageLineFit, float, bool]:
    if len(pixels) < cfg.min_required_pixels:
        return ImageLineFit(), 0.0, False

    u = pixels[:, 0].astype(np.float64)
    v = pixels[:, 1].astype(np.float64)
    mean_v = float(v.mean())
    mean_u = float(u.mean())
    dv = v - mean_v
    du = u - mean_u
    var_v = float(np.dot(dv, dv))
    cov_vu = float(np.dot(dv, du))
    if var_v < 1e-9:
        return ImageLineFit(), 0.0, False

    a = cov_vu / var_v
    b = mean_u - a * mean_v
    angle_deg = math.degrees(math.atan2(1.0, a))
    if angle_deg < 0.0:
        angle_deg += 180.0
    return ImageLineFit(True, a, b, 0), angle_deg, True


def is_nearly_horizontal(angle_deg: float, cfg: Config) -> bool:
    return angle_deg < cfg.min_line_angle_deg or angle_deg > cfg.max_line_angle_deg


def has_excessive_jump(
    previous_line: ImageLineFit,
    current_line: ImageLineFit,
    rows: int,
    cols: int,
    cfg: Config,
) -> bool:
    if not previous_line.valid:
        return False

    v_top = top_row(rows, cfg)
    v_bottom = rows - 1
    prev_top = previous_line.a * float(v_top) + previous_line.b
    curr_top = current_line.a * float(v_top) + current_line.b
    prev_bottom = previous_line.a * float(v_bottom) + previous_line.b
    curr_bottom = current_line.a * float(v_bottom) + current_line.b
    top_jump = abs(curr_top - prev_top)
    bottom_jump = abs(curr_bottom - prev_bottom)
    top_threshold = max(3.0 * cfg.margin_far, 0.30 * float(cols))
    bottom_threshold = max(3.0 * cfg.margin_near, 0.30 * float(cols))
    return top_jump > top_threshold and bottom_jump > bottom_threshold


def pixels_to_vehicle_points(pixels: np.ndarray, cfg: Config) -> np.ndarray:
    if len(pixels) == 0 or not cfg.has_vehicle_projection:
        return empty_points3()

    assert cfg.camera_matrix is not None
    assert cfg.vehicle_R_camera is not None
    assert cfg.vehicle_t_camera is not None

    camera_matrix_inv = np.linalg.inv(cfg.camera_matrix)
    homogeneous = np.column_stack(
        (
            pixels[:, 0].astype(np.float64),
            pixels[:, 1].astype(np.float64),
            np.ones(len(pixels), dtype=np.float64),
        )
    )
    dir_cam = homogeneous @ camera_matrix_inv.T
    dir_vehicle = dir_cam @ cfg.vehicle_R_camera.T
    dz = dir_vehicle[:, 2]
    valid = np.abs(dz) >= 1e-9
    scale = np.empty(len(pixels), dtype=np.float64)
    scale[:] = np.nan
    scale[valid] = -cfg.vehicle_t_camera[2] / dz[valid]
    valid &= np.isfinite(scale) & (scale > 0.0)
    if not np.any(valid):
        return empty_points3()

    hit = cfg.vehicle_t_camera.reshape(1, 3) + dir_vehicle[valid] * scale[valid, None]
    finite = np.isfinite(hit).all(axis=1)
    if not np.any(finite):
        return empty_points3()
    return hit[finite]


def crop_vehicle_points(points: np.ndarray, cfg: Config) -> np.ndarray:
    if len(points) == 0:
        return empty_points3()
    keep = (
        (points[:, 0] >= cfg.xmin)
        & (points[:, 0] <= cfg.xmax)
        & (points[:, 1] >= cfg.ymin)
        & (points[:, 1] <= cfg.ymax)
    )
    return points[keep] if np.any(keep) else empty_points3()


def fit_quadratic_vehicle_line(points: np.ndarray, cfg: Config) -> np.ndarray:
    if len(points) < 3 or cfg.xmax <= cfg.xmin or cfg.spacing <= 0.0:
        return empty_points3()

    x = points[:, 0]
    y = points[:, 1]
    finite = np.isfinite(x) & np.isfinite(y)
    x = x[finite]
    y = y[finite]
    if len(x) < 3:
        return empty_points3()

    design = np.column_stack((np.ones_like(x), x, x * x))
    if np.linalg.matrix_rank(design) < 3:
        return empty_points3()

    coeff, *_ = np.linalg.lstsq(design, y, rcond=None)
    xs: List[float] = []
    cur = cfg.xmin
    while cur <= cfg.xmax + 1e-9:
        xs.append(cur)
        cur += cfg.spacing
    xs_arr = np.array(xs, dtype=np.float64)
    ys_arr = coeff[0] + coeff[1] * xs_arr + coeff[2] * xs_arr * xs_arr
    finite_out = np.isfinite(ys_arr)
    if not np.any(finite_out):
        return empty_points3()
    return np.column_stack((xs_arr[finite_out], ys_arr[finite_out], np.zeros(np.count_nonzero(finite_out))))


def make_center_from_sides(left_points: np.ndarray, right_points: np.ndarray) -> np.ndarray:
    if len(left_points) == 0 or len(right_points) == 0:
        return empty_points3()
    n = min(len(left_points), len(right_points))
    return 0.5 * (left_points[:n] + right_points[:n])


def side_color(side: int) -> Tuple[int, int, int]:
    if side == LEFT_INDEX:
        return (0, 255, 0)
    if side == RIGHT_INDEX:
        return (0, 0, 255)
    return (255, 255, 0)


def draw_init_roi_border(image: np.ndarray, side: int, cfg: Config) -> None:
    rows, cols = image.shape[:2]
    v_top = top_row(rows, cfg)
    u_begin = 0 if side == LEFT_INDEX else cols // 2
    u_end = cols // 2 if side == LEFT_INDEX else cols - 1
    cv2.rectangle(image, (u_begin, v_top), (u_end, rows - 1), side_color(side), 1)


def draw_tracking_roi_border(image: np.ndarray, line: ImageLineFit, side: int, cfg: Config) -> None:
    rows, cols = image.shape[:2]
    v_top = top_row(rows, cfg)
    left_boundary: List[Tuple[int, int]] = []
    right_boundary: List[Tuple[int, int]] = []
    for v in range(v_top, rows):
        u_pred = line.a * float(v) + line.b
        margin = float(margin_at_row(v, rows, cfg))
        u_min = clamp_int(math.floor(u_pred - margin), 0, cols - 1)
        u_max = clamp_int(math.ceil(u_pred + margin), 0, cols - 1)
        left_boundary.append((u_min, v))
        right_boundary.append((u_max, v))

    if len(left_boundary) >= 2:
        cv2.polylines(image, [np.array(left_boundary, dtype=np.int32)], False, side_color(side), 1)
    if len(right_boundary) >= 2:
        cv2.polylines(image, [np.array(right_boundary, dtype=np.int32)], False, side_color(side), 1)


def draw_sparse_pixels(image: np.ndarray, pixels: np.ndarray, side: int) -> None:
    if len(pixels) == 0:
        return
    color = side_color(side)
    step = max(1, len(pixels) // 2000 + 1)
    h, w = image.shape[:2]
    for u, v in pixels[::step]:
        if 0 <= u < w and 0 <= v < h:
            image[v, u] = color


def draw_image_line_fit_panel(result: FrameResult, cfg: Config) -> np.ndarray:
    panel = cv2.cvtColor(result.cleaned_mask, cv2.COLOR_GRAY2BGR)
    rows, cols = panel.shape[:2]
    v_top = top_row(rows, cfg)
    cv2.line(panel, (0, v_top), (cols - 1, v_top), (255, 255, 0), 2)

    for side in (LEFT_INDEX, RIGHT_INDEX):
        side_result = result.side_results[side]
        if not side_result.current_line.valid:
            continue
        line = side_result.current_line
        u1 = int(round(line.a * float(v_top) + line.b))
        u2 = int(round(line.a * float(rows - 1) + line.b))
        p1 = (clamp_int(u1, 0, cols - 1), v_top)
        p2 = (clamp_int(u2, 0, cols - 1), rows - 1)
        cv2.line(panel, p1, p2, side_color(side), 4)

    y0 = 150
    for side in (LEFT_INDEX, RIGHT_INDEX):
        side_result = result.side_results[side]
        text = (
            f"{SIDE_NAMES[side]} {side_result.status} | {side_result.roi_mode} | "
            f"pix {len(side_result.pixels)} | veh {len(side_result.vehicle_points)} | "
            f"angle {side_result.angle_deg:.1f}"
        )
        put_text_with_bg(panel, text, (28, y0), side_color(side), scale=1.0, thickness=2)
        y0 += 42
    return panel


def draw_vehicle_output_panel(result: FrameResult, cfg: Config, size: Tuple[int, int]) -> np.ndarray:
    width, height = size
    panel = np.full((height, width, 3), 248, dtype=np.uint8)

    if not cfg.has_vehicle_projection:
        put_text_with_bg(
            panel,
            "projection disabled: pass calibration or use --profile aiformula-hd1080",
            (42, height // 2),
            (0, 0, 180),
            scale=0.85,
            thickness=2,
            bg=(248, 248, 248),
        )
        return panel

    margin_left = 78
    margin_right = 36
    margin_top = 86
    margin_bottom = 58
    inner_w = width - margin_left - margin_right
    inner_h = height - margin_top - margin_bottom

    def to_px(point: Sequence[float]) -> Tuple[int, int]:
        x, y = float(point[0]), float(point[1])
        px = margin_left + (cfg.ymax - y) / (cfg.ymax - cfg.ymin) * inner_w
        py = height - margin_bottom - (x - cfg.xmin) / (cfg.xmax - cfg.xmin) * inner_h
        return int(round(px)), int(round(py))

    cv2.rectangle(
        panel,
        (margin_left, margin_top),
        (width - margin_right, height - margin_bottom),
        (40, 40, 40),
        2,
    )

    for x in np.arange(math.ceil(cfg.xmin), math.floor(cfg.xmax) + 1, 1.0):
        p1 = to_px((x, cfg.ymin))
        p2 = to_px((x, cfg.ymax))
        cv2.line(panel, p1, p2, (222, 222, 222), 1)
        put_text(panel, f"{x:.0f}", (p2[0] + 4, p2[1] + 4), (90, 90, 90), scale=0.45)

    for y in np.arange(math.ceil(cfg.ymin), math.floor(cfg.ymax) + 1, 1.0):
        p1 = to_px((cfg.xmin, y))
        p2 = to_px((cfg.xmax, y))
        cv2.line(panel, p1, p2, (222, 222, 222), 1)
        put_text(panel, f"{y:.0f}", (p1[0] - 28, p1[1] + 4), (90, 90, 90), scale=0.45)

    put_text(panel, "x forward (m)", (width // 2 - 70, 68), (70, 70, 70), scale=0.55)
    put_text(panel, "y left", (margin_left + 4, height - 18), (70, 70, 70), scale=0.55)
    put_text(panel, "y right", (width - 102, height - 18), (70, 70, 70), scale=0.55)

    for side in (LEFT_INDEX, RIGHT_INDEX):
        raw_points = result.contour_points[side]
        color = side_color(side)
        if len(raw_points) > 0:
            step = max(1, len(raw_points) // 1500 + 1)
            for point in raw_points[::step]:
                cv2.circle(panel, to_px(point), 1, blend_color(color, (255, 255, 255), 0.45), -1, cv2.LINE_AA)

    line_specs = (
        (LEFT_INDEX, result.side_results[LEFT_INDEX].fitted_points),
        (RIGHT_INDEX, result.side_results[RIGHT_INDEX].fitted_points),
        (CENTER_INDEX, result.center_points),
    )
    for side, points in line_specs:
        if len(points) == 0:
            continue
        color = side_color(side)
        px_points = np.array([to_px(point) for point in points], dtype=np.int32)
        if len(px_points) >= 2:
            cv2.polylines(panel, [px_points], False, color, 3, cv2.LINE_AA)
        for point in px_points:
            cv2.circle(panel, tuple(point), 4, color, -1, cv2.LINE_AA)

    legend_y = height - 28
    legend_x = margin_left + 130
    for side in (LEFT_INDEX, RIGHT_INDEX, CENTER_INDEX):
        color = side_color(side)
        cv2.circle(panel, (legend_x, legend_y), 5, color, -1, cv2.LINE_AA)
        put_text(panel, SIDE_NAMES[side], (legend_x + 10, legend_y + 5), (55, 55, 55), scale=0.55)
        legend_x += 116
    return panel


def blend_color(color: Tuple[int, int, int], other: Tuple[int, int, int], alpha: float) -> Tuple[int, int, int]:
    return tuple(int(round(alpha * c + (1.0 - alpha) * o)) for c, o in zip(color, other))


def make_preview_frame(
    source_mask: np.ndarray,
    result: FrameResult,
    cfg: Config,
    frame_name: str,
    panel_size: Tuple[int, int] = (960, 540),
) -> np.ndarray:
    source_bgr = cv2.cvtColor(to_binary_mask(source_mask), cv2.COLOR_GRAY2BGR)
    panels = [
        label_panel(source_bgr, "input mask_image", frame_name, panel_size),
        label_panel(result.annotated_mask, "annotated_mask_image equivalent", "", panel_size),
        label_panel(draw_image_line_fit_panel(result, cfg), "image ROI and fitted u = a*v + b", "", panel_size),
        label_panel(draw_vehicle_output_panel(result, cfg, panel_size), "published lane_lines output", "", panel_size),
    ]
    top = np.hstack((panels[0], panels[1]))
    bottom = np.hstack((panels[2], panels[3]))
    return np.vstack((top, bottom))


def label_panel(image: np.ndarray, title: str, subtitle: str, size: Tuple[int, int]) -> np.ndarray:
    width, height = size
    resized = resize_to_cover(image, width, height)
    overlay = resized.copy()
    cv2.rectangle(overlay, (0, 0), (width, 52), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.55, resized, 0.45, 0, resized)
    put_text(resized, title, (18, 31), (255, 255, 255), scale=0.75, thickness=2)
    if subtitle:
        text_size, _ = cv2.getTextSize(subtitle, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
        put_text(resized, subtitle, (width - text_size[0] - 18, 32), (230, 230, 230), scale=0.6, thickness=1)
    return resized


def resize_to_cover(image: np.ndarray, width: int, height: int) -> np.ndarray:
    h, w = image.shape[:2]
    if w == 0 or h == 0:
        return np.zeros((height, width, 3), dtype=np.uint8)
    scale = max(width / w, height / h)
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))
    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
    x0 = max(0, (new_w - width) // 2)
    y0 = max(0, (new_h - height) // 2)
    return resized[y0 : y0 + height, x0 : x0 + width].copy()


def put_text(
    image: np.ndarray,
    text: str,
    origin: Tuple[int, int],
    color: Tuple[int, int, int],
    scale: float = 0.6,
    thickness: int = 1,
) -> None:
    cv2.putText(image, text, origin, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness, cv2.LINE_AA)


def put_text_with_bg(
    image: np.ndarray,
    text: str,
    origin: Tuple[int, int],
    color: Tuple[int, int, int],
    scale: float = 0.6,
    thickness: int = 1,
    bg: Tuple[int, int, int] = (255, 255, 255),
) -> None:
    size, baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, thickness)
    x, y = origin
    cv2.rectangle(
        image,
        (x - 5, y - size[1] - 5),
        (x + size[0] + 5, y + baseline + 5),
        bg,
        -1,
    )
    put_text(image, text, origin, color, scale, thickness)


def rpy_to_rotation_matrix(roll: float, pitch: float, yaw: float, degrees: bool = True) -> np.ndarray:
    if degrees:
        roll, pitch, yaw = map(math.radians, (roll, pitch, yaw))
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    rx = np.array([[1.0, 0.0, 0.0], [0.0, cr, -sr], [0.0, sr, cr]], dtype=np.float64)
    ry = np.array([[cp, 0.0, sp], [0.0, 1.0, 0.0], [-sp, 0.0, cp]], dtype=np.float64)
    rz = np.array([[cy, -sy, 0.0], [sy, cy, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64)
    return rz @ ry @ rx


def apply_profile(config: Config, profile: str) -> Config:
    if profile == "image-only":
        config.camera_matrix = None
        config.vehicle_R_camera = None
        config.vehicle_t_camera = None
        return config

    if profile == "aiformula-hd1080":
        config.camera_matrix = np.array(
            [
                [763.1748657226562, 0.0, 990.0390625],
                [0.0, 763.1748657226562, 543.4489135742188],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        )
    elif profile == "aiformula-nhd":
        config.camera_matrix = np.array(
            [
                [254.391622, 0.0, 330.013020833],
                [0.0, 254.391622, 181.149637858],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        )
    else:
        raise ValueError(f"unknown profile: {profile}")

    baseline = 0.120005
    config.vehicle_t_camera = np.array([0.055, baseline * 0.5, 0.54], dtype=np.float64)
    config.vehicle_R_camera = rpy_to_rotation_matrix(-89.0, 0.0, -90.0, degrees=True)
    return config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Visualize the lane_line_publisher_rewrite_ordered.cpp pipeline on saved mask images."
    )
    parser.add_argument("--mask-dir", default=DEFAULT_MASK_DIR, help="Directory containing mask images to read.")
    parser.add_argument("--glob", default="*.png", help="Input filename glob inside --mask-dir.")
    parser.add_argument(
        "--out-dir",
        default=DEFAULT_OUT_DIR,
        help="Output directory.",
    )
    parser.add_argument(
        "--profile",
        choices=("aiformula-hd1080", "aiformula-nhd", "image-only"),
        default="aiformula-hd1080",
        help="Calibration profile. Use image-only to skip vehicle-frame point clouds.",
    )
    parser.add_argument("--limit", type=int, default=0, help="Maximum number of images to process. 0 means all.")
    parser.add_argument("--stride", type=int, default=1, help="Process every Nth image.")
    parser.add_argument("--fps", type=float, default=12.0, help="Preview video FPS.")
    parser.add_argument("--no-video", action="store_true", help="Do not write preview.mp4.")
    parser.add_argument("--save-frame-every", type=int, default=100, help="Save one JPEG panel every N frames. 0 disables.")

    parser.add_argument("--min-area", type=int, default=100, help="Connected-component area threshold in pixels.")
    parser.add_argument("--tolerance", type=int, default=10, help="Lane pixel finder tolerance; controls tracking ROI margins.")
    parser.add_argument("--xmin", type=float, default=1.5)
    parser.add_argument("--xmax", type=float, default=10.0)
    parser.add_argument("--ymin", type=float, default=-3.0)
    parser.add_argument("--ymax", type=float, default=3.0)
    parser.add_argument("--spacing", type=float, default=0.5)

    parser.add_argument("--fx", type=float, default=None, help="Override camera focal length x.")
    parser.add_argument("--fy", type=float, default=None, help="Override camera focal length y.")
    parser.add_argument("--cx", type=float, default=None, help="Override camera principal point x.")
    parser.add_argument("--cy", type=float, default=None, help="Override camera principal point y.")
    parser.add_argument("--tx", type=float, default=None, help="Override vehicle_T_camera translation x.")
    parser.add_argument("--ty", type=float, default=None, help="Override vehicle_T_camera translation y.")
    parser.add_argument("--tz", type=float, default=None, help="Override vehicle_T_camera translation z.")
    parser.add_argument("--roll-deg", type=float, default=None, help="Override vehicle_T_camera roll in degrees.")
    parser.add_argument("--pitch-deg", type=float, default=None, help="Override vehicle_T_camera pitch in degrees.")
    parser.add_argument("--yaw-deg", type=float, default=None, help="Override vehicle_T_camera yaw in degrees.")
    return parser.parse_args()


def build_config(args: argparse.Namespace) -> Config:
    cfg = Config(
        min_area=max(1, args.min_area),
        tolerance=max(1, args.tolerance),
        xmin=args.xmin,
        xmax=args.xmax,
        ymin=args.ymin,
        ymax=args.ymax,
        spacing=args.spacing,
    )
    cfg = apply_profile(cfg, args.profile)

    camera_values = (args.fx, args.fy, args.cx, args.cy)
    if any(value is not None for value in camera_values):
        if not all(value is not None for value in camera_values):
            raise ValueError("camera override requires all of --fx --fy --cx --cy")
        cfg.camera_matrix = np.array(
            [[args.fx, 0.0, args.cx], [0.0, args.fy, args.cy], [0.0, 0.0, 1.0]], dtype=np.float64
        )
        if cfg.vehicle_R_camera is None:
            cfg.vehicle_R_camera = rpy_to_rotation_matrix(-89.0, 0.0, -90.0, degrees=True)
        if cfg.vehicle_t_camera is None:
            cfg.vehicle_t_camera = np.array([0.055, 0.0600025, 0.54], dtype=np.float64)

    translation_values = (args.tx, args.ty, args.tz)
    if any(value is not None for value in translation_values):
        if not all(value is not None for value in translation_values):
            raise ValueError("translation override requires all of --tx --ty --tz")
        cfg.vehicle_t_camera = np.array([args.tx, args.ty, args.tz], dtype=np.float64)

    rpy_values = (args.roll_deg, args.pitch_deg, args.yaw_deg)
    if any(value is not None for value in rpy_values):
        if not all(value is not None for value in rpy_values):
            raise ValueError("rotation override requires all of --roll-deg --pitch-deg --yaw-deg")
        cfg.vehicle_R_camera = rpy_to_rotation_matrix(args.roll_deg, args.pitch_deg, args.yaw_deg, degrees=True)

    return cfg


def resolve_output_dir(path_text: str) -> Path:
    cwd = Path.cwd().resolve()
    out_path = Path(path_text)
    if not out_path.is_absolute():
        out_path = cwd / out_path
    out_path = out_path.resolve()
    try:
        out_path.relative_to(cwd)
    except ValueError as exc:
        raise ValueError(f"refusing to create output outside working directory: {out_path}") from exc
    return out_path


def iter_input_files(mask_dir: Path, glob_text: str, stride: int, limit: int) -> List[Path]:
    files = sorted(mask_dir.glob(glob_text))
    if stride > 1:
        files = files[::stride]
    if limit > 0:
        files = files[:limit]
    return files


def write_summary_header(writer: csv.writer) -> None:
    writer.writerow(
        [
            "frame_index",
            "file",
            "side",
            "status",
            "roi_mode",
            "pixels",
            "vehicle_points_after_crop",
            "fitted_lane_points",
            "angle_deg",
            "line_a",
            "line_b",
            "lost_count",
        ]
    )


def write_summary_rows(writer: csv.writer, frame_index: int, file_name: str, result: FrameResult) -> None:
    for side in (LEFT_INDEX, RIGHT_INDEX):
        side_result = result.side_results[side]
        writer.writerow(
            [
                frame_index,
                file_name,
                SIDE_NAMES[side],
                side_result.status,
                side_result.roi_mode,
                len(side_result.pixels),
                len(side_result.vehicle_points),
                len(side_result.fitted_points),
                f"{side_result.angle_deg:.6f}",
                f"{side_result.current_line.a:.12g}",
                f"{side_result.current_line.b:.12g}",
                side_result.lost_count,
            ]
        )
    writer.writerow(
        [
            frame_index,
            file_name,
            "center",
            "ok" if len(result.center_points) else "not_published",
            "from_left_right",
            "",
            "",
            len(result.center_points),
            "",
            "",
            "",
            "",
        ]
    )


def write_lane_points_rows(writer: csv.writer, frame_index: int, file_name: str, result: FrameResult) -> None:
    lane_sets = {
        "left": result.side_results[LEFT_INDEX].fitted_points,
        "right": result.side_results[RIGHT_INDEX].fitted_points,
        "center": result.center_points,
    }
    for side_name, points in lane_sets.items():
        for point_index, point in enumerate(points):
            writer.writerow(
                [
                    frame_index,
                    file_name,
                    side_name,
                    point_index,
                    f"{point[0]:.9f}",
                    f"{point[1]:.9f}",
                    f"{point[2]:.9f}",
                ]
            )


def main() -> int:
    args = parse_args()
    cfg = build_config(args)
    mask_dir = Path(args.mask_dir)
    out_dir = resolve_output_dir(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    sample_dir = out_dir / "sample_frames"
    if args.save_frame_every > 0:
        sample_dir.mkdir(parents=True, exist_ok=True)

    files = iter_input_files(mask_dir, args.glob, max(1, args.stride), max(0, args.limit))
    if not files:
        raise FileNotFoundError(f"no input masks matched {mask_dir / args.glob}")

    processor = LaneLinePublisherEquivalent(cfg)
    video_writer: Optional[cv2.VideoWriter] = None
    video_path = out_dir / "preview.mp4"
    summary_path = out_dir / "summary.csv"
    lane_points_path = out_dir / "lane_lines_points.csv"

    with summary_path.open("w", newline="", encoding="utf-8") as summary_fp, lane_points_path.open(
        "w", newline="", encoding="utf-8"
    ) as points_fp:
        summary_writer = csv.writer(summary_fp)
        points_writer = csv.writer(points_fp)
        write_summary_header(summary_writer)
        points_writer.writerow(["frame_index", "file", "line", "point_index", "x", "y", "z"])

        for frame_index, path in enumerate(files):
            mask = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
            if mask is None:
                print(f"warning: failed to read {path}")
                continue

            result = processor.find_lane_lines(mask)
            write_summary_rows(summary_writer, frame_index, path.name, result)
            write_lane_points_rows(points_writer, frame_index, path.name, result)

            preview_frame = make_preview_frame(mask, result, cfg, path.name)
            if not args.no_video:
                if video_writer is None:
                    height, width = preview_frame.shape[:2]
                    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                    video_writer = cv2.VideoWriter(str(video_path), fourcc, args.fps, (width, height))
                    if not video_writer.isOpened():
                        raise RuntimeError(f"failed to open video writer for {video_path}")
                video_writer.write(preview_frame)

            if args.save_frame_every > 0 and frame_index % args.save_frame_every == 0:
                cv2.imwrite(str(sample_dir / f"{frame_index:06d}_{path.stem}.jpg"), preview_frame)

            if frame_index == len(files) - 1 and args.save_frame_every > 0:
                cv2.imwrite(str(sample_dir / f"{frame_index:06d}_{path.stem}.jpg"), preview_frame)

            if frame_index % 100 == 0 or frame_index == len(files) - 1:
                left = result.side_results[LEFT_INDEX]
                right = result.side_results[RIGHT_INDEX]
                print(
                    f"[{frame_index + 1}/{len(files)}] {path.name}: "
                    f"left={left.status}/{len(left.fitted_points)} "
                    f"right={right.status}/{len(right.fitted_points)} "
                    f"center={len(result.center_points)}"
                )

    if video_writer is not None:
        video_writer.release()

    print(f"wrote {summary_path}")
    print(f"wrote {lane_points_path}")
    if not args.no_video:
        print(f"wrote {video_path}")
    if args.save_frame_every > 0:
        print(f"wrote sample frames under {sample_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
