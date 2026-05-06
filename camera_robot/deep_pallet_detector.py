#!/usr/bin/env python3
"""
深度学习托盘检测节点 (RGBD)
修复：
  1. rotation 输出为 rot6d [B,N,6]，用 Gram-Schmidt 转 3x3
  2. 深度图解码改用 passthrough 编码，避免 buffer 错误
  3. 取置信度最高 anchor 而非固定 index=0
  4. 每5帧推理一次，RGB/深度图严格用同一对消息
"""

import sys
import os
import rclpy
from rclpy.node import Node
import cv2
import numpy as np
import torch
import message_filters
from cv_bridge import CvBridge
from sensor_msgs.msg import Image
from geometry_msgs.msg import PoseStamped
from scipy.spatial.transform import Rotation

# 将训练库加入 Python 路径
# sys.path.insert(0, os.path.join(
#     os.path.dirname(__file__), 'efficientpose_lib'))
from camera_robot.efficientpose_lib.models.efficientpose import EfficientPose
from camera_robot.efficientpose_lib.config import Config


# ═══════════════════════════════════════════════════════════
#  rot6d → 旋转矩阵（Gram-Schmidt 正交化）
# ═══════════════════════════════════════════════════════════

def rot6d_to_matrix(rot6d: np.ndarray) -> np.ndarray:
    """
    rot6d: shape (6,)  — 模型 RotationHead 的 Tanh 输出
    返回:  shape (3,3) — 正交旋转矩阵
    参考: Zhou et al. "On the Continuity of Rotation Representations in Neural Networks"
    """
    a1 = rot6d[:3]
    a2 = rot6d[3:6]
    b1 = a1 / (np.linalg.norm(a1) + 1e-8)
    b2 = a2 - np.dot(b1, a2) * b1
    b2 = b2 / (np.linalg.norm(b2) + 1e-8)
    b3 = np.cross(b1, b2)
    return np.stack([b1, b2, b3], axis=1)   # shape (3,3)


# ═══════════════════════════════════════════════════════════
#  绘制工具函数
# ═══════════════════════════════════════════════════════════

def rotation_to_euler_deg(R):
    """旋转矩阵 → ZYX 欧拉角（度），返回 (roll, pitch, yaw)"""
    r = Rotation.from_matrix(R)
    rpy = r.as_euler('zyx', degrees=True)
    return rpy[2], rpy[1], rpy[0]


def project_axis(R, t, K, length=0.05):
    """将 3D 坐标轴投影到图像平面"""
    fx, fy, cx, cy = K

    def proj(pt3d):
        if pt3d[2] < 1e-6:
            return None
        x = fx * pt3d[0] / pt3d[2] + cx
        y = fy * pt3d[1] / pt3d[2] + cy
        return (int(x), int(y))

    origin = t
    x_end  = t + R[:, 0] * length
    y_end  = t + R[:, 1] * length
    z_end  = t + R[:, 2] * length
    return proj(origin), proj(x_end), proj(y_end), proj(z_end)


def draw_pose_annotation(img, bbox, rotation, translation, K):
    x1, y1, x2, y2 = bbox
    roll, pitch, yaw = rotation_to_euler_deg(rotation)
    dist = float(np.linalg.norm(translation))
    tx, ty, tz = translation
    h_img, w_img = img.shape[:2]

    # ① 检测框
    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
    corner_len = 12
    for cx_, cy_, dx, dy in [
            (x1, y1,  1,  1), (x2, y1, -1,  1),
            (x1, y2,  1, -1), (x2, y2, -1, -1)]:
        cv2.line(img, (cx_, cy_),
                 (cx_ + dx * corner_len, cy_), (0, 255, 128), 3)
        cv2.line(img, (cx_, cy_),
                 (cx_, cy_ + dy * corner_len), (0, 255, 128), 3)

    label = f'Pallet  {dist:.2f} m'
    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
    lx = max(x1, 0)
    ly = max(y1 - th - 10, th + 4)
    cv2.rectangle(img, (lx, ly - th - 4), (lx + tw + 8, ly + 2), (0, 200, 0), -1)
    cv2.putText(img, label, (lx + 4, ly - 2),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 1, cv2.LINE_AA)

    # ② 坐标轴
    if tz > 0.05:
        try:
            pts = project_axis(rotation, translation, K, length=0.06)
            if None not in pts:
                origin_2d, x_end_2d, y_end_2d, z_end_2d = pts

                def in_bounds(p):
                    return 0 <= p[0] < w_img and 0 <= p[1] < h_img

                if in_bounds(origin_2d):
                    for end_pt, color, name in [
                            (x_end_2d, (0, 0, 255), 'X'),
                            (y_end_2d, (0, 255, 0), 'Y'),
                            (z_end_2d, (255, 0, 0), 'Z')]:
                        if in_bounds(end_pt):
                            cv2.arrowedLine(img, origin_2d, end_pt,
                                            color, 2, tipLength=0.3)
                            cv2.putText(img, name,
                                        (end_pt[0] + 3, end_pt[1]),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.4,
                                        color, 1, cv2.LINE_AA)
        except Exception:
            pass

    # ③ 信息面板
    panel_lines = [
        '-- Pallet Pose --',
        f'Dist : {dist:.3f} m',
        f'Tx   : {tx:+.3f} m',
        f'Ty   : {ty:+.3f} m',
        f'Tz   : {tz:+.3f} m',
        f'Roll : {roll:+.1f} deg',
        f'Pitch: {pitch:+.1f} deg',
        f'Yaw  : {yaw:+.1f} deg',
    ]
    line_h, pad, panel_w = 18, 8, 185
    panel_h = len(panel_lines) * line_h + pad * 2
    px, py = w_img - panel_w - 8, 8

    overlay = img.copy()
    cv2.rectangle(overlay,
                  (px - pad, py), (px + panel_w, py + panel_h),
                  (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.65, img, 0.35, 0, img)

    for i, line in enumerate(panel_lines):
        if i == 0:
            color = (180, 255, 180)
        elif any(k in line for k in ['Tx', 'Ty', 'Tz', 'Dist']):
            color = (100, 220, 255)
        elif any(k in line for k in ['Roll', 'Pitch', 'Yaw']):
            color = (255, 200, 100)
        else:
            color = (220, 220, 220)
        cv2.putText(img, line,
                    (px, py + pad + i * line_h + line_h - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)

    return img


# ═══════════════════════════════════════════════════════════
#  ROS2 节点
# ═══════════════════════════════════════════════════════════

class DeepPalletDetector(Node):
    def __init__(self):
        super().__init__('deep_pallet_detector')

        # ── 参数声明 ──────────────────────────────────────
        self.declare_parameter('model_path',
            os.path.join(os.path.dirname(__file__),
                         'resource', 'weights', 'best_model_phi0.pth'))
        self.declare_parameter('phi', 0)
        self.declare_parameter('confidence_threshold', 0.7)
        self.declare_parameter('device', 'cuda')
        self.declare_parameter('use_depth', True)
        self.declare_parameter('infer_every_n_frames', 1)   # ← 新增参数
        self.declare_parameter('fx', 554.0)
        self.declare_parameter('fy', 554.0)
        self.declare_parameter('cx', 320.0)
        self.declare_parameter('cy', 240.0)

        model_path            = self.get_parameter('model_path').value
        phi                   = self.get_parameter('phi').value
        self.conf_thresh      = self.get_parameter('confidence_threshold').value
        device_str            = self.get_parameter('device').value
        self.use_depth        = self.get_parameter('use_depth').value
        self.infer_every_n    = self.get_parameter('infer_every_n_frames').value
        self.K = (
            self.get_parameter('fx').value,
            self.get_parameter('fy').value,
            self.get_parameter('cx').value,
            self.get_parameter('cy').value,
        )

        # ── 帧计数器 & 缓存上一次推理结果 ────────────────
        self._frame_count   = 0          # 收到的同步帧总数
        self._last_annotated = None      # 上次推理的标注图（numpy BGR）
        self._last_pose      = None      # 上次推理的 PoseStamped

        # ── 加载模型 ──────────────────────────────────────
        self.device = torch.device(device_str)
        cfg = Config()
        cfg.phi = phi
        self.image_size = cfg.compound_coef[phi]['resolution']  # phi=0 → 512

        self.model = EfficientPose(phi=phi, num_classes=1)
        self.model.load_state_dict(
            torch.load(model_path, map_location=self.device))
        self.model.to(self.device)
        self.model.eval()
        self.get_logger().info(f'模型加载完成: {model_path}')

        self.mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        self.std  = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        self.bridge = CvBridge()

        # ── 话题订阅 ──────────────────────────────────────
        if self.use_depth:
            sub_rgb = message_filters.Subscriber(
                self, Image, '/camera_robot/rgbd_camera/image_raw')
            sub_depth = message_filters.Subscriber(
                self, Image, '/camera_robot/rgbd_camera/depth/image_raw')
            self.sync = message_filters.ApproximateTimeSynchronizer(
                [sub_rgb, sub_depth], queue_size=10, slop=0.05)
            self.sync.registerCallback(self.rgbd_callback)
            self.get_logger().info('已启用 RGBD 模式')
        else:
            self.create_subscription(
                Image, '/camera_robot/rgbd_camera/image_raw',
                self.rgb_only_callback, 10)
            self.get_logger().info('已启用纯 RGB 模式')

        self.pub_img  = self.create_publisher(
            Image, '/camera_robot/annotated_image', 10)
        self.pub_pose = self.create_publisher(
            PoseStamped, '/camera_robot/pallet_pose', 10)

        self.get_logger().info(
            f'每 {self.infer_every_n} 帧推理一次，'
            f'中间帧复用上一次结果')

    # ── 深度图解码（兼容 32FC1 / 16UC1 / passthrough） ──
    def decode_depth(self, depth_msg: Image) -> np.ndarray:
        enc = depth_msg.encoding.lower()
        if enc == '32fc1':
            return self.bridge.imgmsg_to_cv2(depth_msg, '32FC1')
        if enc == '16uc1':
            raw = self.bridge.imgmsg_to_cv2(depth_msg, '16UC1')
            return raw.astype(np.float32) / 1000.0
        try:
            raw = self.bridge.imgmsg_to_cv2(depth_msg, 'passthrough')
            if raw.dtype == np.uint16:
                return raw.astype(np.float32) / 1000.0
            return raw.astype(np.float32)
        except Exception:
            h = depth_msg.height
            w = depth_msg.width
            data = np.frombuffer(depth_msg.data, dtype=np.float32)
            if data.size == h * w:
                return data.reshape(h, w)
            data16 = np.frombuffer(depth_msg.data, dtype=np.uint16)
            if data16.size == h * w:
                return data16.reshape(h, w).astype(np.float32) / 1000.0
            self.get_logger().warn(
                f'无法解析深度图 encoding={depth_msg.encoding}，使用零深度')
            return np.zeros((depth_msg.height, depth_msg.width), dtype=np.float32)

    # ── 预处理：纯 RGB ────────────────────────────────────
    def preprocess_rgb(self, bgr):
        self.orig_h, self.orig_w = bgr.shape[:2]
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        resized = cv2.resize(rgb, (self.image_size, self.image_size))
        t = torch.from_numpy(resized).permute(2, 0, 1).float() / 255.0
        t = (t - self.mean) / self.std
        return t.unsqueeze(0).to(self.device)

    # ── 预处理：RGBD 4 通道 ───────────────────────────────
    def preprocess_rgbd(self, bgr, depth_img: np.ndarray):
        self.orig_h, self.orig_w = bgr.shape[:2]
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        rgb_r = cv2.resize(rgb, (self.image_size, self.image_size))

        # --- 深度处理：单位 m，裁剪 0.1~8.0m，与 DatasetCollector/pallet_data_collector 一致 ---
        depth = depth_img.astype(np.float32)
        depth = np.nan_to_num(depth, nan=0.0, posinf=0.0, neginf=0.0)
        depth[(depth < 0.1) | (depth > 8.0)] = 0.0          # ← 上限改 8.0

        depth_r = cv2.resize(depth, (self.image_size, self.image_size),
                            interpolation=cv2.INTER_NEAREST)

        depth_blur = cv2.GaussianBlur(depth_r, (3, 3), 0)
        depth_r[depth_r == 0] = depth_blur[depth_r == 0]

        # 训练时如果是 depth_norm = depth / 8.0，这里保持一致
        depth_norm = depth_r / 8.0                          # ← 归一化改 8.0
        depth_norm[depth_r == 0] = 0.0

        t_rgb = torch.from_numpy(rgb_r).permute(2, 0, 1).float() / 255.0
        t_rgb = (t_rgb - self.mean) / self.std
        t_d   = torch.from_numpy(depth_norm).unsqueeze(0).float()

        return torch.cat([t_rgb, t_d], dim=0).unsqueeze(0).to(self.device)

    # ── RGBD 回调 ─────────────────────────────────────────
    def rgbd_callback(self, rgb_msg: Image, depth_msg: Image):
        """
        ApproximateTimeSynchronizer 已保证 rgb_msg 和 depth_msg
        是时间戳最接近的一对，天然对应。
        每 infer_every_n 帧做一次推理，其余帧复用缓存结果。
        """
        self._frame_count += 1

        bgr = self.bridge.imgmsg_to_cv2(rgb_msg, 'bgr8')

        if self._frame_count % self.infer_every_n == 0:
            # ── 推理帧：用同步好的 RGB + 深度 ──────────
            depth  = self.decode_depth(depth_msg)
            tensor = self.preprocess_rgbd(bgr, depth)
            self._run_inference(bgr, tensor, rgb_msg)
        else:
            # ── 跳过帧：复用上次结果，重新发布 ──────────
            self._republish_cached(bgr, rgb_msg)

    # ── RGB only 回调 ─────────────────────────────────────
    def rgb_only_callback(self, msg: Image):
        self._frame_count += 1
        bgr = self.bridge.imgmsg_to_cv2(msg, 'bgr8')

        if self._frame_count % self.infer_every_n == 0:
            tensor = self.preprocess_rgb(bgr)
            self._run_inference(bgr, tensor, msg)
        else:
            self._republish_cached(bgr, msg)

    # ── 复用缓存结果 ──────────────────────────────────────
    def _republish_cached(self, bgr, header_msg: Image):
        """跳过帧：若有缓存则发布上次标注图 + pose，否则发原图"""
        if self._last_annotated is not None:
            out_msg = self.bridge.cv2_to_imgmsg(self._last_annotated, 'bgr8')
            out_msg.header = header_msg.header   # 时间戳更新为当前帧
            self.pub_img.publish(out_msg)

            if self._last_pose is not None:
                pose = self._last_pose
                pose.header = header_msg.header
                self.pub_pose.publish(pose)
        else:
            # 推理前的最初几帧，直接发原图，不发 pose
            out_msg = self.bridge.cv2_to_imgmsg(bgr, 'bgr8')
            out_msg.header = header_msg.header
            self.pub_img.publish(out_msg)

    # ── 推理 + 绘制 + 发布 ───────────────────────────────
    def _run_inference(self, bgr, tensor, header_msg: Image):
        with torch.no_grad():
            outputs = self.model(tensor)

        class_scores = outputs['class'][0, :, 0].cpu().numpy()
        best_idx     = int(np.argmax(class_scores))
        best_score   = float(class_scores[best_idx])

        if best_score < self.conf_thresh:
            out_msg = self.bridge.cv2_to_imgmsg(bgr, 'bgr8')
            out_msg.header = header_msg.header
            self.pub_img.publish(out_msg)
            self._last_annotated = bgr
            self._last_pose      = None
            return

        rot6d       = outputs['rotation'][0, best_idx].cpu().numpy()
        rotation    = rot6d_to_matrix(rot6d)
        translation = outputs['translation'][0, best_idx].cpu().numpy()

        raw_bbox = outputs['bbox'][0, best_idx].cpu().numpy()
        sx, sy = self.orig_w, self.orig_h
        bbox = (
            int(np.clip(raw_bbox[0] * sx, 0, sx - 1)),
            int(np.clip(raw_bbox[1] * sy, 0, sy - 1)),
            int(np.clip(raw_bbox[2] * sx, 0, sx - 1)),
            int(np.clip(raw_bbox[3] * sy, 0, sy - 1)),
        )

        annotated = draw_pose_annotation(
            bgr.copy(), bbox, rotation, translation, self.K)

        out_msg = self.bridge.cv2_to_imgmsg(annotated, 'bgr8')
        out_msg.header = header_msg.header
        self.pub_img.publish(out_msg)

        pose = PoseStamped()
        pose.header          = header_msg.header
        pose.header.frame_id = 'camera_link'
        pose.pose.position.x = float(translation[0])
        pose.pose.position.y = float(translation[1])
        pose.pose.position.z = float(translation[2])
        q = Rotation.from_matrix(rotation).as_quat()
        pose.pose.orientation.x = float(q[0])
        pose.pose.orientation.y = float(q[1])
        pose.pose.orientation.z = float(q[2])
        pose.pose.orientation.w = float(q[3])
        self.pub_pose.publish(pose)

        # 缓存本次结果
        self._last_annotated = annotated
        self._last_pose      = pose


def main(args=None):
    rclpy.init(args=args)
    node = DeepPalletDetector()
    rclpy.spin(node)
    rclpy.shutdown()


if __name__ == '__main__':
    main()