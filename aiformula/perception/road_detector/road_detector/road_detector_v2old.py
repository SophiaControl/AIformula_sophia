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
    """
    Resize + pad to square (new_shape,new_shape) like YOLO letterbox.

    Returns:
      img: padded BGR image (new_shape,new_shape,3)
      ratio: scale ratio (r, r)
      top,bottom,left,right: padding sizes
    """
    shape = image.shape[:2]  # (h, w)
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    new_unpad = (int(round(shape[1] * r)), int(round(shape[0] * r)))  # (w, h)

    img = cv2.resize(image, new_unpad, interpolation=cv2.INTER_LINEAR)

    dw = (new_shape[1] - new_unpad[0]) / 2  # width padding
    dh = (new_shape[0] - new_unpad[1]) / 2  # height padding

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
    """Static-shape TensorRT runner for explicit-batch engines."""

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
        self.in_dtype = trt.nptype(engine.get_binding_dtype(self.in_id))

    def infer(self, x: np.ndarray) -> List[np.ndarray]:
        """
        x: float32 NCHW contiguous, must match engine input shape exactly.
        returns: list of outputs in binding order (output bindings only).
        """
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

        # Reduce OpenCV CPU thread pressure
        try:
            cv2.setNumThreads(1)
        except Exception:
            pass

        # ---------------- Params ----------------
        engine_path, ll_thresh, publish_annotated, drop_every_n = self.get_params()

        self.ll_thresh = float(ll_thresh)
        self.publish_annotated = bool(publish_annotated)
        self.drop_every_n = int(drop_every_n)  # 0=不丢帧；2=每2帧处理1帧（丢1帧）
        self._frame_cnt = 0

        # -------------- TensorRT Engine --------------
        self.trt = TrtRunner(engine_path)
        self.get_logger().info(f"Loaded TensorRT engine: {engine_path}")
        self.get_logger().info(f"Engine input shape: {self.trt.in_shape}, dtype: {self.trt.in_dtype}")
        self.get_logger().info(f"ll_threshold: {self.ll_thresh}")
        self.get_logger().info(f"publish_annotated: {self.publish_annotated}, drop_every_n: {self.drop_every_n}")

        # -------------- ROS pubs/sub --------------
        self.annotated_mask_image_pub = self.create_publisher(
            Image, "pub_annotated_mask_image", buffer_size
        )
        self.lane_mask_image_pub = self.create_publisher(
            Image, "pub_mask_image", buffer_size
        )

        self.image_sub = self.create_subscription(
            Image, "sub_image", self.image_callback, buffer_size
        )

        # Debug log throttling (1s)
        self._last_log_t = time.time()

    def get_params(self):
        # 基本参数
        self.declare_parameter(
            "engine_path",
            "/home/nvidia/workspace/ros2_ws/src/aiformula/perception/road_detector/weights/yolopv2_fp16.engine",
        )
        self.declare_parameter("ll_threshold", 0.0)

        # 性能相关参数
        self.declare_parameter("publish_annotated", True)  # False 可减少 CPU
        self.declare_parameter("drop_every_n", 0)  # 0=不丢帧；2=每2帧处理1帧

        engine_path = self.get_parameter("engine_path").get_parameter_value().string_value
        ll_thresh = self.get_parameter("ll_threshold").get_parameter_value().double_value
        publish_annotated = self.get_parameter("publish_annotated").get_parameter_value().bool_value
        drop_every_n = self.get_parameter("drop_every_n").get_parameter_value().integer_value

        return engine_path, ll_thresh, publish_annotated, drop_every_n

    def image_callback(self, msg):
        # optional drop frames
        self._frame_cnt += 1
        if self.drop_every_n and (self._frame_cnt % self.drop_every_n != 0):
            return

        try:
            img = self.cv_bridge.imgmsg_to_cv2(msg, "bgr8")
        except CvBridgeError as e:
            self.get_logger().error(f"CvBridgeError: {e}")
            return

        # Letterbox to 640
        pad_img, ratio, top, bottom, left, right = letterbox_bgr(img, new_shape=640)

        # BGR -> RGB (多数模型训练用 RGB)
        pad_img = cv2.cvtColor(pad_img, cv2.COLOR_BGR2RGB)
        

        # Preprocess -> float32 [1,3,640,640] contiguous
        x = pad_img.astype(np.float32) / 255.0
        x = np.transpose(x, (2, 0, 1))[None, ...]  # 1x3x640x640
        x = np.ascontiguousarray(x, dtype=np.float32)

        # TensorRT inference
        outs = self.trt.infer(x)

        # Expected ONNX outputs order:
        # ['p3','p4','p5','772','773','774','da','ll']
        if len(outs) < 8:
            self.get_logger().error(f"Unexpected TRT outputs count: {len(outs)}")
            return

        # Find outputs by shape, not by index
        ll_out = None
        da_out = None

        for o in outs:
            if o.ndim == 4:
                n, c, h, w = o.shape
        # lane line: [1,1,H,W]
                if c == 1 and h >= 200 and w >= 200:
                    ll_out = o
        # drivable area: [1,2,H,W]
                elif c == 2 and h >= 200 and w >= 200:
                    da_out = o

        if ll_out is None:
            self.get_logger().error(
                f"Failed to find lane line output. Shapes: {[o.shape for o in outs]}"
            )
            return


        # # Crop padding area back to unpadded
        # H = ll_out.shape[2]
        # W = ll_out.shape[3]
        # y1 = H - bottom
        # x1 = W - right
        # if top >= y1 or left >= x1:
        #     self.get_logger().error(
        #         f"Invalid crop: top={top}, bottom={bottom}, left={left}, right={right}, H={H}, W={W}"
        #     )
        #     return

        # #这一段 可能 会有缩进问题，如果出现了来这检查
        # # ll_crop is score/logit map (no sigmoid)
        # ll_crop = ll_out[0, 0, top:y1, left:x1]
        # self.get_logger().info(
        #     f"ll_crop stats(logit?): min={ll_crop.min():.3f} "
        #     f"max={ll_crop.max():.3f} mean={ll_crop.mean():.3f}"
        # )
        # ll_score = ll_crop.astype(np.float32)
        H, W = ll_out.shape[2], ll_out.shape[3]

        # 关键：把 640 输入尺度的 padding 映射到 ll_out 尺度
        sy = H / 640.0
        sx = W / 640.0
        top_o    = int(round(top * sy))
        bottom_o = int(round(bottom * sy))
        left_o   = int(round(left * sx))
        right_o  = int(round(right * sx))

        y1 = H - bottom_o
        x1 = W - right_o
        if top_o >= y1 or left_o >= x1:
            self.get_logger().error(f"Invalid crop: top_o={top_o}, bottom_o={bottom_o}, left_o={left_o}, right_o={right_o}, H={H}, W={W}")
            return

        ll_score = ll_out[0, 0, top_o:y1, left_o:x1].astype(np.float32)
        self.get_logger().info(f"pad_img={pad_img.shape}, ll_out={ll_out.shape}, top/bottom/left/right={top}/{bottom}/{left}/{right}")



        # ---------- inverse letterbox mapping (NO sigmoid) ----------
        # h0, w0 = img.shape[:2]

        # # ratio might be (rw, rh) or a single float; handle both safely
        # if isinstance(ratio, (tuple, list, np.ndarray)):
        #     rw = float(ratio[0])
        #     rh = float(ratio[1]) if len(ratio) > 1 else float(ratio[0])
        # else:
        #     rw = rh = float(ratio)

        # h_r = int(round(h0 * rh))
        # w_r = int(round(w0 * rw))

        # # resize score/logit map back to resized-image space
        # ll_score_r = cv2.resize(
        #     ll_score,
        #     (w_r, h_r),
        #     interpolation=cv2.INTER_LINEAR
        # )

        # # paste into original-image canvas
        # ll_score_full = np.zeros((h0, w0), dtype=np.float32)
        # ll_score_full[:h_r, :w_r] = ll_score_r

        # # threshold LAST on score/logit (choose thresh accordingly)
        # ll_mask = (ll_score_full > self.ll_thresh).astype(np.uint8)
        h0, w0 = img.shape[:2]

        # 关键：crop 后就是“有效区域(缩放后)”，直接拉回原图即可
        ll_score_full = cv2.resize(ll_score, (w0, h0), interpolation=cv2.INTER_LINEAR)

        ll_mask = (ll_score_full > self.ll_thresh).astype(np.uint8)



        #到这为止

        # Throttled debug log (once per ~1s)
        now = time.time()
        if now - self._last_log_t > 1.0:
            self._last_log_t = now
            self.get_logger().info(
                f"mask stats: min={ll_mask.min()} max={ll_mask.max()} mean={ll_mask.mean():.3f}"
            )

        # Publish lane mask image (mono8)
        try:
            mask_msg = self.cv_bridge.cv2_to_imgmsg(ll_mask * 255, encoding="mono8")
            mask_msg.header = msg.header
            self.lane_mask_image_pub.publish(mask_msg)
        except CvBridgeError as e:
            self.get_logger().error(f"CvBridgeError (mask publish): {e}")
            return

        # Publish annotated mask image if enabled
        if self.publish_annotated:
            ann = img.copy()
            # overlay: green where mask==1
            ann[ll_mask == 1] = (0, 255, 0)
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
