#!/usr/bin/env python
from __future__ import annotations

"""
这个节点是 motor_controller 的运行时代码。

简单说，它做的事是：

1. 订阅传统控制器给的 base cmd_vel，也就是原本想发给车的 [v, omega]。
2. 订阅车辆当前观测到的响应：车体前向速度 meas_v 和 yaw rate meas_omega。
3. 用最近一段历史，让模型估计当前车辆的动态 ax+b 关系。
4. 反过来算出 corrected cmd_vel。
5. 把 corrected cmd_vel 转成左右轮 RPM。
6. 把左右轮 RPM 编码成 8-byte CAN payload 发出去。

这里的 ax+b 不是一条固定直线，而是每个 runtime step 都会变：

    observed_response = a_t * sent_command + b_t

也就是：

    meas_v     = a_v     * cmd_v     + b_v
    meas_omega = a_omega * cmd_omega + b_omega

模型输出的是动态的 [a_v, a_omega, b_v, b_omega]。
真正发给车之前，我们会反解：

    cmd_send = (cmd_base - b) / a

在传统控制器输出后面加一层补偿。
"""

import csv
from collections import deque
from pathlib import Path
from typing import List

import numpy as np
import rclpy
import torch
from ament_index_python.packages import get_package_share_directory
from can_msgs.msg import Frame
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from rclpy.node import Node
from torch import nn


class AffineCommandCorrectionModel(nn.Module):
    """
    这个网络只负责估计动态的 a 和 b。

    输入：
    - history_norm: 过去 20 步左右的低维历史
      [cmd_v, cmd_omega, meas_v, meas_omega]
    - command_norm: 当前 base command
      [cmd_v, cmd_omega]

    输出：
    - [a_v, a_omega, b_v, b_omega]

    不直接输出 RPM，也不直接输出 CAN。
    当前车大概满足什么样的 ax+b 响应关系。
    """

    def __init__(
        self,
        history_dim: int,
        hidden_size: int,
        gru_layers: int,
        dropout: float,
        gain_span: float,
        max_bias_v: float,
        max_bias_omega: float,
    ) -> None:
        super().__init__()
        self.gain_span = float(gain_span)
        self.max_bias_v = float(max_bias_v)
        self.max_bias_omega = float(max_bias_omega)
        self.gru = nn.GRU(
            input_size=history_dim,
            hidden_size=hidden_size,
            num_layers=gru_layers,
            batch_first=True,
            dropout=dropout if gru_layers > 1 else 0.0,
        )
        self.head = nn.Sequential(
            nn.Linear(hidden_size + 2, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 4),
        )

    def forward(self, history_norm: torch.Tensor, command_norm: torch.Tensor) -> torch.Tensor:
        # GRU 先看最近一段历史，这辆车最近是怎么响应 command 的”。
        _, hidden = self.gru(history_norm)
        context = hidden[-1]

        # 当前 command 也要给模型看，因为 correction 通常和当前 v/omega 大小有关。
        raw = self.head(torch.cat([context, command_norm], dim=-1))

        # gain 以 1 为中心：a=1 表示理想 baseline，command 和响应一致。
        a_v = 1.0 + self.gain_span * torch.tanh(raw[:, 0])
        a_omega = 1.0 + self.gain_span * torch.tanh(raw[:, 1])

        # bias 以 0 为中心：b=0 表示没有固定偏差。
        b_v = self.max_bias_v * torch.tanh(raw[:, 2])
        b_omega = self.max_bias_omega * torch.tanh(raw[:, 3])
        return torch.stack([a_v, a_omega, b_v, b_omega], dim=-1)


class MotorController(Node):
    def __init__(self):
        super().__init__("motor_controller")

        # 基本参数：轮距、轮径、齿比。
        self.declare_parameter("publish_timer_loop_duration", 0.01)
        self.declare_parameter("wheel.tread", 0.60)
        self.declare_parameter("wheel.diameter", 0.254)
        self.declare_parameter("wheel.gear_ratio", 1.1)
        self.declare_parameter("model_path", "")

        # 这两个 topic 是模型需要的车辆响应观测。
        # velocity_body 用来拿车体坐标系前向速度 meas_v。
        # odom 用来拿 yaw rate meas_omega。
        self.declare_parameter("velocity_body_topic", "/aiformula_sensing/vectornav/velocity_body")
        self.declare_parameter("odom_topic", "/aiformula_sensing/gyro_odometry_publisher/odom")

        # 刚启动时历史还不够，不马上用模型修正，先直接发 base command。
        self.declare_parameter("min_history_samples", 3)

        # 修正后的 command 做限幅，避免模型输出异常时直接打爆电机。
        self.declare_parameter("max_corrected_v", 3.0)
        self.declare_parameter("max_corrected_omega", 1.0)

        # debug CSV 会记录每次模型估计的 a,b，以及修正前后的 command。
        self.declare_parameter("debug_compare_enabled", True)
        self.declare_parameter("debug_csv_path", "affine_command_correction_debug.csv")

        publish_timer_loop_duration = self.get_parameter("publish_timer_loop_duration").get_parameter_value().double_value
        self.tread = float(self.get_parameter("wheel.tread").value)
        self.diameter = float(self.get_parameter("wheel.diameter").value)
        self.gear_ratio = float(self.get_parameter("wheel.gear_ratio").value)
        self.velocity_body_topic = str(self.get_parameter("velocity_body_topic").value)
        self.odom_topic = str(self.get_parameter("odom_topic").value)
        self.min_history_samples = int(self.get_parameter("min_history_samples").value)
        self.max_corrected_v = float(self.get_parameter("max_corrected_v").value)
        self.max_corrected_omega = float(self.get_parameter("max_corrected_omega").value)
        self.debug_compare_enabled = bool(self.get_parameter("debug_compare_enabled").value)
        self.debug_csv_path = self.resolve_debug_csv_path(str(self.get_parameter("debug_csv_path").value))
        self.debug_csv_file = None
        self.debug_csv_writer = None
        self.debug_sample_index = 0

        (
            self.model,
            self.history_steps,
            self.feature_columns,
            self.command_columns,
            self.hist_mean,
            self.hist_std,
            self.cmd_mean,
            self.cmd_std,
        ) = self.load_model()

        expected_features = ["cmd_v", "cmd_omega", "meas_v", "meas_omega"]
        expected_commands = ["cmd_v", "cmd_omega"]
        if self.feature_columns != expected_features:
            raise RuntimeError(f"Expected affine history features {expected_features}, got {self.feature_columns}")
        if self.command_columns != expected_commands:
            raise RuntimeError(f"Expected affine command columns {expected_commands}, got {self.command_columns}")

        self.history: deque[list[float]] = deque(maxlen=self.history_steps)

        # latest_meas_* 保存最近一次传感器观测。
        # command callback 来的时候，就用这些值拼历史。
        self.latest_meas_v: float | None = None
        self.latest_meas_omega: float | None = None

        # history 里记录的是“实际发出去的 command”和“车辆观测响应”。
        # 所以上一次发出去的 corrected command 也要记住。
        self.last_sent_v: float | None = None
        self.last_sent_omega: float | None = None

        buffer_size = 10

        # 订阅 1：传统控制器 / gamepad 给的 base cmd_vel。
        # launch 里会把 sub_speed_command remap 到实际 speed command topic。
        self.twist_sub = self.create_subscription(Twist, "sub_speed_command", self.twist_callback, buffer_size)

        # 订阅 2：车体前向速度。训练时 meas_v 用的就是 vn_body_vx。
        self.velocity_sub = self.create_subscription(Odometry, self.velocity_body_topic, self.velocity_body_callback, buffer_size)

        # 订阅 3：yaw rate。训练时 meas_omega 用的是 odom_omega_z。
        self.odom_sub = self.create_subscription(Odometry, self.odom_topic, self.odom_callback, buffer_size)

        # 输出：CAN frame。payload 还是 8 bytes，不改底层 CAN 格式。
        self.can_pub = self.create_publisher(Frame, "pub_can", buffer_size)
        self.publish_timer = self.create_timer(publish_timer_loop_duration, self.publish_canframe_callback)

        self.frame_msg = Frame()
        self.frame_msg.header.frame_id = "can0"
        self.frame_msg.id = 0x210
        self.frame_msg.dlc = 8

        # CAN data 一共 8 bytes：
        # bytes 0-3: right wheel RPM, int32 little-endian
        # bytes 4-7: left wheel RPM, int32 little-endian
        self.frame_msg.data = self.toCanCmd(0) + self.toCanCmd(0)

        self.open_debug_csv()
        self.log_info("Loaded affine command-correction motor controller model.")
        self.log_info(f"Subscribing velocity-body response from: {self.velocity_body_topic}")
        self.log_info(f"Subscribing odometry yaw-rate response from: {self.odom_topic}")

    def destroy_node(self):
        if self.debug_csv_file is not None:
            self.debug_csv_file.flush()
            self.debug_csv_file.close()
            self.debug_csv_file = None
        return super().destroy_node()

    def resolve_debug_csv_path(self, configured: str) -> Path:
        path = Path(configured or "affine_command_correction_debug.csv").expanduser()
        if not path.is_absolute():
            path = self.model_controller_dir() / path
        path.parent.mkdir(parents=True, exist_ok=True)
        return path

    def model_controller_dir(self) -> Path:
        return Path(get_package_share_directory("motor_controller")) / "model_controller"

    def log_info(self, message: str) -> None:
        self.get_logger().info(str(message))

    def open_debug_csv(self) -> None:
        if not self.debug_compare_enabled:
            return
        self.debug_csv_file = self.debug_csv_path.open("w", newline="", encoding="utf-8")
        self.debug_csv_writer = csv.DictWriter(
            self.debug_csv_file,
            fieldnames=[
                "sample",
                "node_time_sec",
                "base_v",
                "base_omega",
                "meas_v",
                "meas_omega",
                "history_len",
                "a_v",
                "a_omega",
                "b_v",
                "b_omega",
                "corrected_v",
                "corrected_omega",
                "delta_v",
                "delta_omega",
                "left_rpm",
                "right_rpm",
                "can_first_right_rpm",
                "can_second_left_rpm",
            ],
        )
        self.debug_csv_writer.writeheader()
        self.debug_csv_file.flush()
        self.log_info(f"Writing affine correction debug CSV: {self.debug_csv_path}")

    def resolve_model_path(self) -> Path:
        configured = str(self.get_parameter("model_path").value or "").strip()
        if configured:
            path = Path(configured).expanduser()
            if path.exists():
                return path

        # 默认加载已经复制到 model_controller 文件夹里的新 affine 权重。
        return self.model_controller_dir() / "affine_command_correction.pt"

    def load_model(self):
        # checkpoint 里除了模型参数，也保存了训练时的 mean/std 和列名。
        # runtime 必须用同一套 normalization，不然模型输入尺度会错。
        model_path = self.resolve_model_path()
        if not model_path.exists():
            raise FileNotFoundError(f"Affine command-correction weight not found: {model_path}")
        try:
            checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)
        except TypeError:
            checkpoint = torch.load(model_path, map_location="cpu")

        config = checkpoint.get("config", {})
        feature_columns = list(checkpoint["feature_cols"])
        command_columns = list(checkpoint["command_cols"])
        history_steps = int(config.get("history_steps", 20))
        model = AffineCommandCorrectionModel(
            history_dim=len(feature_columns),
            hidden_size=int(config.get("hidden_size", 32)),
            gru_layers=int(config.get("gru_layers", 1)),
            dropout=float(config.get("dropout", 0.0)),
            gain_span=float(config.get("gain_span", 0.75)),
            max_bias_v=float(config.get("max_bias_v", 0.75)),
            max_bias_omega=float(config.get("max_bias_omega", 0.35)),
        )
        model.load_state_dict(checkpoint["model"])
        model.eval()

        hist_mean = np.asarray(checkpoint["hist_mean"], dtype=np.float32)
        hist_std = np.asarray(checkpoint["hist_std"], dtype=np.float32)
        cmd_mean = np.asarray(checkpoint["cmd_mean"], dtype=np.float32)
        cmd_std = np.asarray(checkpoint["cmd_std"], dtype=np.float32)
        return model, history_steps, feature_columns, command_columns, hist_mean, hist_std, cmd_mean, cmd_std

    def velocity_body_callback(self, msg: Odometry) -> None:
        # velocity_body 的 linear.x 是车体坐标系前向速度，对应训练里的 meas_v。
        self.latest_meas_v = float(msg.twist.twist.linear.x)

    def odom_callback(self, msg: Odometry) -> None:
        # odom 的 angular.z 是 yaw rate，对应训练里的 meas_omega。
        self.latest_meas_omega = float(msg.twist.twist.angular.z)

    def twist_callback(self, msg: Twist) -> None:
        # base command 是传统控制器原本想发的命令。
        # 模型不会替代它，只在它后面加一层 correction。
        base_v = float(msg.linear.x)
        base_omega = float(msg.angular.z)

        # 先把“上一次实际发出的 command + 最新观测响应”放入历史。
        self.append_latest_response_to_history(base_v, base_omega)

        # 用动态 ax+b 反解 corrected command。
        corrected_v, corrected_omega, params = self.correct_command(base_v, base_omega)

        # corrected cmd_vel 仍然走传统轮速/RPM转换，再编码成 CAN。
        left_rpm, right_rpm = self.cmd_to_motor_rpms(corrected_v, corrected_omega)
        cmd_right = self.toCanCmd(right_rpm)
        cmd_left = self.toCanCmd(left_rpm)
        self.frame_msg.data = cmd_right + cmd_left

        self.last_sent_v = corrected_v
        self.last_sent_omega = corrected_omega
        self.log_debug_compare(
            base_v=base_v,
            base_omega=base_omega,
            corrected_v=corrected_v,
            corrected_omega=corrected_omega,
            params=params,
            left_rpm=left_rpm,
            right_rpm=right_rpm,
        )

    def append_latest_response_to_history(self, fallback_v: float, fallback_omega: float) -> None:
        # 如果传感器还没来，就先用 base command 当 fallback。
        # 这只影响启动初期，历史够了以后就用真实订阅到的 response。
        meas_v = self.latest_meas_v if self.latest_meas_v is not None else fallback_v
        meas_omega = self.latest_meas_omega if self.latest_meas_omega is not None else fallback_omega

        # history 里的 command 应该是上一次真正发给车的 corrected command。
        cmd_v = self.last_sent_v if self.last_sent_v is not None else fallback_v
        cmd_omega = self.last_sent_omega if self.last_sent_omega is not None else fallback_omega
        self.history.append([float(cmd_v), float(cmd_omega), float(meas_v), float(meas_omega)])

    def history_sequence(self, base_v: float, base_omega: float) -> np.ndarray:
        # 模型固定需要 history_steps 长度的序列。
        # 如果刚启动历史不够，就用最早的样本补齐前面。
        fallback_meas_v = self.latest_meas_v if self.latest_meas_v is not None else base_v
        fallback_meas_omega = self.latest_meas_omega if self.latest_meas_omega is not None else base_omega
        fallback = [base_v, base_omega, fallback_meas_v, fallback_meas_omega]
        values = list(self.history)
        if not values:
            values = [fallback] * self.history_steps
        elif len(values) < self.history_steps:
            values = [values[0]] * (self.history_steps - len(values)) + values
        else:
            values = values[-self.history_steps :]
        return np.asarray(values, dtype=np.float32)

    def correct_command(self, base_v: float, base_omega: float) -> tuple[float, float, np.ndarray]:
        # 冷启动保护：历史太短时，先不相信模型，直接发 base command。
        if len(self.history) < self.min_history_samples:
            return base_v, base_omega, np.asarray([1.0, 1.0, 0.0, 0.0], dtype=np.float32)

        history_raw = self.history_sequence(base_v, base_omega)
        command_raw = np.asarray([base_v, base_omega], dtype=np.float32)
        history_norm = (history_raw - self.hist_mean) / np.clip(self.hist_std, 1e-8, None)
        command_norm = (command_raw - self.cmd_mean) / np.clip(self.cmd_std, 1e-8, None)

        with torch.no_grad():
            params = self.model(
                torch.as_tensor(history_norm[None, :, :], dtype=torch.float32),
                torch.as_tensor(command_norm[None, :], dtype=torch.float32),
            ).cpu().numpy()[0]

        # 模型输出当前动态的 ax+b:
        #   meas_v     = a_v     * cmd_v     + b_v
        #   meas_omega = a_omega * cmd_omega + b_omega
        a_v = max(float(params[0]), 1.0e-4)
        a_omega = max(float(params[1]), 1.0e-4)
        b_v = float(params[2])
        b_omega = float(params[3])

        # runtime 不是直接用 ax+b 做预测，而是反解：
        #   corrected_cmd = (desired_response - b) / a
        # 这里 desired_response 就是 base command 希望车辆实现的响应。
        corrected_v = (base_v - b_v) / a_v
        corrected_omega = (base_omega - b_omega) / a_omega

        # 最后再限幅，保证发给底层的 command 在可控范围里。
        corrected_v = float(np.clip(corrected_v, -self.max_corrected_v, self.max_corrected_v))
        corrected_omega = float(np.clip(corrected_omega, -self.max_corrected_omega, self.max_corrected_omega))
        return corrected_v, corrected_omega, params

    def cmd_to_motor_rpms(self, linear_velocity: float, angular_velocity: float) -> tuple[float, float]:
        # 这一段还是传统 diff-drive 的 cmd_vel -> 左右轮 RPM 转换。
        # 学习模型只修正 cmd_vel，不直接改 CAN 编码规则。
        wheel_angular_velocities = np.array(
            [
                (linear_velocity / (self.diameter * 0.5)) + (self.tread / self.diameter) * angular_velocity,
                (linear_velocity / (self.diameter * 0.5)) - (self.tread / self.diameter) * angular_velocity,
            ],
            dtype=float,
        )
        a_left = 0.834
        w0_left = 2.76
        a_right = 0.844
        w0_right = 2.81
        w_left_cmd = np.sign(wheel_angular_velocities[0]) * (abs(wheel_angular_velocities[0]) / a_left + w0_left)
        w_right_cmd = np.sign(wheel_angular_velocities[1]) * (abs(wheel_angular_velocities[1]) / a_right + w0_right)
        rpm_left = (w_left_cmd * (60.0 / (2.0 * np.pi))) * self.gear_ratio
        rpm_right = (w_right_cmd * (60.0 / (2.0 * np.pi))) * self.gear_ratio
        return float(rpm_left), float(rpm_right)

    def log_debug_compare(
        self,
        *,
        base_v: float,
        base_omega: float,
        corrected_v: float,
        corrected_omega: float,
        params: np.ndarray,
        left_rpm: float,
        right_rpm: float,
    ) -> None:
        if self.debug_csv_writer is None or self.debug_csv_file is None:
            return
        now = self.get_clock().now().nanoseconds * 1e-9
        self.debug_csv_writer.writerow(
            {
                "sample": self.debug_sample_index,
                "node_time_sec": now,
                "base_v": base_v,
                "base_omega": base_omega,
                "meas_v": self.latest_meas_v,
                "meas_omega": self.latest_meas_omega,
                "history_len": len(self.history),
                "a_v": float(params[0]),
                "a_omega": float(params[1]),
                "b_v": float(params[2]),
                "b_omega": float(params[3]),
                "corrected_v": corrected_v,
                "corrected_omega": corrected_omega,
                "delta_v": corrected_v - base_v,
                "delta_omega": corrected_omega - base_omega,
                "left_rpm": left_rpm,
                "right_rpm": right_rpm,
                "can_first_right_rpm": right_rpm,
                "can_second_left_rpm": left_rpm,
            }
        )
        self.debug_sample_index += 1
        self.debug_csv_file.flush()

    def publish_canframe_callback(self):
        # 定时发布最近一次算好的 CAN frame。
        # command callback 没来的时候，frame 会保持上一帧。
        self.can_pub.publish(self.frame_msg)

    @staticmethod
    def toCanCmd(rpm: float) -> List[int]:
        # 一个 RPM 编成 4 bytes signed int32 little-endian。
        # 两个轮子拼起来就是 8 bytes CAN payload。
        rounded = int(round(rpm))
        return list(rounded.to_bytes(4, "little", signed=True))


def main(args=None):
    rclpy.init(args=args)
    motor_controller = MotorController()
    rclpy.spin(motor_controller)
    motor_controller.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
