#!/usr/bin/env python3
import time

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
from typing import List

import numpy as np
import cv2

# TensorRT runtime
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit  # noqa: F401


def letterbox_bgr(image, new_shape=640, color=(114, 114, 114)):
    shape = image.shape[:2]  # (h, w)
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    new_unpad = (int(round(shape[1] * r)), int(round(shape[0] * r)))  # (w, h)

    img = cv2.resize(image, new_unpad, interpolation=cv2.INTER_LINEAR)

    dw = (new_shape[1] - new_unpad[0]) / 2
    dh = (new_shape[0] - new_unpad[1]) / 2

    top = int(round(dh - 0.1))
    bottom = int(round(dh + 0.1))
    left = int(round(dw - 0.1))
    right = int(round(dw + 0.1))

    img = cv2.copyMakeBorder(
        img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color
    )
    ratio = (r, r)
    return img, ratio, top, bottom, left, right


class TrtRunner:
    def __init__(self, engine_path: str, logger_level=trt.Logger.WARNING):
        logger = trt.Logger(logger_level)
        with open(engine_path, "rb") as f, trt.Runtime(logger) as runtime:
            engine = runtime.deserialize_cuda_engine(f.read())
        if engine is None:
            raise RuntimeError(f"Failed to deserialize TensorRT engine: {engine_path}")

        context = engine.create_execution_context()
        if context is None:
            raise RuntimeError("Failed to create TensorRT execution context")

        self.engine = engine
        self.context = context
        self.stream = cuda.Stream()

        self.bindings = [0] * engine.num_bindings
        self.hmem = [None] * engine.num_bindings
        self.dmem = [None] * engine.num_bindings

        self.input_ids = []
        self.output_ids = []

        for i in range(engine.num_bindings):
            dtype = trt.nptype(engine.get_binding_dtype(i))
            shape = tuple(engine.get_binding_shape(i))
            size = int(np.prod(shape))

            host = cuda.pagelocked_empty(size, dtype=dtype)
            dev = cuda.mem_alloc(host.nbytes)

            self.hmem[i] = host
            self.dmem[i] = dev
            self.bindings[i] = int(dev)

            if engine.binding_is_input(i):
                self.input_ids.append(i)
            else:
                self.output_ids.append(i)

        if len(self.input_ids) != 1:
            raise RuntimeError(f"Expected 1 input binding, got {len(self.input_ids)}")

        self.in_id = self.input_ids[0]
        self.in_shape = tuple(engine.get_binding_shape(self.in_id))
        self.in_dtype = trt.nptype(self.engine.get_binding_dtype(self.in_id))

    def infer(self, x: np.ndarray) -> List[np.ndarray]:
        if x.dtype != self.in_dtype:
            raise ValueError(f"Input dtype mismatch: got {x.dtype}, want {self.in_dtype}")
        if tuple(x.shape) != self.in_shape:
            raise ValueError(f"Input shape mismatch: got {x.shape}, want {self.in_shape}")
        if not x.flags["C_CONTIGUOUS"]:
            x = np.ascontiguousarray(x)

        np.copyto(self.hmem[self.in_id], x.ravel())
        cuda.memcpy_htod_async(self.dmem[self.in_id], self.hmem[self.in_id], self.stream)

        ok = self.context.execute_async_v2(self.bindings, self.stream.handle)
        if not ok:
            raise RuntimeError("TensorRT execute_async_v2 failed")

        for oid in self.output_ids:
            cuda.memcpy_dtoh_async(self.hmem[oid], self.dmem[oid], self.stream)

        self.stream.synchronize()

        outs: list[np.ndarray] = []
        for oid in self.output_ids:
            oshape = tuple(self.engine.get_binding_shape(oid))
            odtype = trt.nptype(self.engine.get_binding_dtype(oid))
            outs.append(np.array(self.hmem[oid], dtype=odtype).reshape(oshape))
        return outs


class RoadDetector(Node):
    def __init__(self):
        super().__init__("road_detector")
        self.cv_bridge = CvBridge()
        buffer_size = 10

        try:
            cv2.setNumThreads(1)
        except Exception:
            pass

        engine_path, ll_thresh, publish_annotated, drop_every_n = self.get_params()

        self.ll_thresh = float(ll_thresh)
        self.publish_annotated = bool(publish_annotated)
        self.drop_every_n = int(drop_every_n)
        self._frame_cnt = 0

        # ROI params
        self.roi_rect_ratio = float(self.declare_parameter("roi_rect_ratio", 0.3).value)
        self.roi_trap_ratio = float(self.declare_parameter("roi_trap_ratio", 0.18).value)
        self.roi_trap_left = float(self.declare_parameter("roi_trap_left", 0.38).value)
        self.roi_trap_right = float(self.declare_parameter("roi_trap_right", 0.51).value)

        self.trt = TrtRunner(engine_path)

        self.lane_mask_image_pub = self.create_publisher(Image, "pub_mask_image", buffer_size)
        self.annotated_mask_image_pub = self.create_publisher(Image, "pub_annotated_mask_image", buffer_size)
        self.lane_mask_roi_pub = self.create_publisher(Image, "pub_mask_image_roi", buffer_size)

        self.image_sub = self.create_subscription(Image, "sub_image", self.image_callback, buffer_size)
        self._last_log_t = time.time()

    def get_params(self):
        self.declare_parameter(
            "engine_path",
            "/home/nvidia/workspace/ros2_ws/src/aiformula/perception/road_detector/weights/yolopv2_fp16.engine",
        )
        self.declare_parameter("ll_threshold", 0.5)
        self.declare_parameter("publish_annotated", True)
        self.declare_parameter("drop_every_n", 0)

        engine_path = self.get_parameter("engine_path").get_parameter_value().string_value
        ll_thresh = self.get_parameter("ll_threshold").get_parameter_value().double_value
        publish_annotated = self.get_parameter("publish_annotated").get_parameter_value().bool_value
        drop_every_n = self.get_parameter("drop_every_n").get_parameter_value().integer_value
        return engine_path, ll_thresh, publish_annotated, drop_every_n

    def build_roi_mask_original(self, h: int, w: int) -> np.ndarray:
        """
        ✅ 你要求的 ROI：
          - 底部矩形：高度 = 0.2*h，宽度全宽
          - 矩形上方紧贴一个梯形：高度 = 0.3*h（注意是“梯形自身高度”），不与矩形重叠
              梯形底边：全宽
              梯形顶边：x in [0.35*w, 0.65*w]
        """
        rect_ratio = float(self.roi_rect_ratio)   # 0.2
        trap_h_ratio = float(self.roi_trap_ratio) # 0.3 (height of trapezoid itself)
        trap_left = float(self.roi_trap_left)     # 0.35
        trap_right = float(self.roi_trap_right)   # 0.65

        roi = np.zeros((h, w), dtype=np.uint8)

        # rectangle (bottom)
        rect_h = int(round(rect_ratio * h))
        rect_h = int(np.clip(rect_h, 0, h))
        rect_y_top = h - rect_h
        rect_y_top = int(np.clip(rect_y_top, 0, h))
        roi[rect_y_top:h, 0:w] = 1

        # trapezoid (above rectangle), height = 0.3*h, touching rectangle top, no overlap
        trap_h = int(round(trap_h_ratio * h))
        trap_h = int(np.clip(trap_h, 0, h))

        trap_y_bottom = rect_y_top - 1
        trap_y_top = trap_y_bottom - trap_h + 1

        # clamp
        trap_y_top = int(np.clip(trap_y_top, 0, h - 1))
        trap_y_bottom = int(np.clip(trap_y_bottom, -1, h - 1))

        if trap_y_bottom >= 0 and trap_y_top <= trap_y_bottom:
            tlx = int(round(trap_left * (w - 1)))
            trx = int(round(trap_right * (w - 1)))
            tlx = int(np.clip(tlx, 0, w - 1))
            trx = int(np.clip(trx, 0, w - 1))

            pts = np.array([[
                (tlx, trap_y_top),
                (trx, trap_y_top),
                (w - 1, trap_y_bottom),
                (0, trap_y_bottom),
            ]], dtype=np.int32)
            cv2.fillPoly(roi, pts, 1)

        return roi

    def image_callback(self, msg):
        self._frame_cnt += 1
        if self.drop_every_n and (self._frame_cnt % self.drop_every_n != 0):
            return

        try:
            img = self.cv_bridge.imgmsg_to_cv2(msg, "bgr8")
        except CvBridgeError as e:
            self.get_logger().error(f"CvBridgeError: {e}")
            return

        pad_img, ratio, top, bottom, left, right = letterbox_bgr(img, new_shape=640)
        pad_img = cv2.cvtColor(pad_img, cv2.COLOR_BGR2RGB)

        x = pad_img.astype(np.float32) / 255.0
        x = np.transpose(x, (2, 0, 1))[None, ...]
        x = np.ascontiguousarray(x, dtype=np.float32)

        outs = self.trt.infer(x)
        if len(outs) < 8:
            self.get_logger().error(f"Unexpected TRT outputs count: {len(outs)}")
            return

        ll_out = None
        for o in outs:
            if o.ndim == 4:
                n, c, h, w = o.shape
                if c == 1 and h >= 200 and w >= 200:
                    ll_out = o
                    break

        if ll_out is None:
            self.get_logger().error(f"Failed to find lane line output. Shapes: {[o.shape for o in outs]}")
            return

        H = ll_out.shape[2]
        W = ll_out.shape[3]
        y1 = H - bottom
        x1 = W - right
        if top >= y1 or left >= x1:
            self.get_logger().error(
                f"Invalid crop: top={top}, bottom={bottom}, left={left}, right={right}, H={H}, W={W}"
            )
            return

        # no sigmoid
        ll_full = ll_out[0, 0, :, :]  # (H,W)
        ll_crop = ll_full[top:y1, left:x1]

        # inverse letterbox mapping (keep your original behavior)
        h0, w0 = img.shape[:2]
        r = ratio[0]
        h_r = int(round(h0 * r))
        w_r = int(round(w0 * r))

        ll_r = cv2.resize(ll_crop, (w_r, h_r), interpolation=cv2.INTER_LINEAR)

        ll_full_orig = np.zeros((h0, w0), dtype=np.float32)
        ll_full_orig[:h_r, :w_r] = ll_r

        ll_mask_full = (ll_full_orig > self.ll_thresh).astype(np.uint8)

        roi_orig = self.build_roi_mask_original(h0, w0)
        ll_mask_roi = (ll_mask_full & roi_orig).astype(np.uint8)

        now = time.time()
        if now - self._last_log_t > 1.0:
            self._last_log_t = now
            self.get_logger().info(
                f"mask_full mean={ll_mask_full.mean():.3f}, mask_roi mean={ll_mask_roi.mean():.3f}"
            )

        try:
            mask_full_msg = self.cv_bridge.cv2_to_imgmsg(ll_mask_full * 255, encoding="mono8")
            mask_full_msg.header = msg.header
            self.lane_mask_image_pub.publish(mask_full_msg)
        except CvBridgeError as e:
            self.get_logger().error(f"CvBridgeError (full mask publish): {e}")
            return

        try:
            mask_roi_msg = self.cv_bridge.cv2_to_imgmsg(ll_mask_roi * 255, encoding="mono8")
            mask_roi_msg.header = msg.header
            self.lane_mask_roi_pub.publish(mask_roi_msg)
        except CvBridgeError as e:
            self.get_logger().error(f"CvBridgeError (roi mask publish): {e}")
            return

        if self.publish_annotated:
            ann = img.copy()
            ann[ll_mask_full == 1] = (0, 255, 0)
            try:
                ann_msg = self.cv_bridge.cv2_to_imgmsg(ann, encoding="bgr8")
                ann_msg.header = msg.header
                self.annotated_mask_image_pub.publish(ann_msg)
            except CvBridgeError as e:
                self.get_logger().error(f"CvBridgeError (annotated publish): {e}")


def main():
    rclpy.init()
    node = RoadDetector()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()

