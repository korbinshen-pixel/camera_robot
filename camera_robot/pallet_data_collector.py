#!/usr/bin/env python3
"""
pallet_data_collector_node.py

无 GUI 数据采集节点：
  - 订阅 RGB + 深度图（同步）
  - 提供 /save_pallet_sample 服务触发保存
  - 保存 RGB PNG + 深度 PNG(16位, 单位mm) + 深度可视化 PNG + meta 信息

深度图保存方式已改为与 DatasetCollector 一致：
  depth_png: uint16 PNG, 单位 mm
读取方式：
  depth_m = cv2.imread(path, cv2.IMREAD_ANYDEPTH).astype(np.float32) / 1000.0
"""

import os
import datetime
import threading

import rclpy
from rclpy.node import Node
import message_filters
from sensor_msgs.msg import Image
from std_srvs.srv import Trigger
from cv_bridge import CvBridge
import cv2
import numpy as np


def decode_depth(bridge: CvBridge, depth_msg: Image) -> np.ndarray:
    """返回 float32 深度图，单位米。兼容 32FC1 / 16UC1 / passthrough。"""
    enc = depth_msg.encoding.lower()

    if enc == '32fc1':
        return bridge.imgmsg_to_cv2(depth_msg, '32FC1').astype(np.float32)

    if enc == '16uc1':
        raw = bridge.imgmsg_to_cv2(depth_msg, '16UC1')
        return raw.astype(np.float32) / 1000.0

    try:
        raw = bridge.imgmsg_to_cv2(depth_msg, 'passthrough')
        if raw.dtype == np.uint16:
            return raw.astype(np.float32) / 1000.0
        return raw.astype(np.float32)
    except Exception:
        h, w = depth_msg.height, depth_msg.width
        buf = bytes(depth_msg.data)

        data = np.frombuffer(buf, dtype=np.float32)
        if data.size == h * w:
            return data.reshape(h, w)

        data16 = np.frombuffer(buf, dtype=np.uint16)
        if data16.size == h * w:
            return data16.reshape(h, w).astype(np.float32) / 1000.0

        return np.zeros((h, w), dtype=np.float32)


def depth_to_colormap(depth: np.ndarray,
                      min_m: float = 0.1,
                      max_m: float = 8.0) -> np.ndarray:
    """将深度图（米）转为 BGR 伪彩色图，便于人眼查看。"""
    valid = (depth >= min_m) & (depth <= max_m)
    norm = np.zeros_like(depth, dtype=np.float32)
    norm[valid] = (depth[valid] - min_m) / (max_m - min_m)
    norm_u8 = (np.clip(norm, 0, 1) * 255).astype(np.uint8)
    color = cv2.applyColorMap(norm_u8, cv2.COLORMAP_TURBO)
    color[~valid] = 0
    return color


def save_depth_png_mm(depth_m: np.ndarray, path: str):
    """
    将 float32 深度图（单位 m）保存为 16位 PNG（单位 mm）。
    与 DatasetCollector 的保存方式一致。
    """
    depth = np.array(depth_m, dtype=np.float32)
    depth[~np.isfinite(depth)] = 0.0
    depth[depth < 0.0] = 0.0

    depth_mm = depth * 1000.0
    depth_mm = np.clip(depth_mm, 0.0, 65535.0)
    depth_u16 = depth_mm.astype(np.uint16)

    ok = cv2.imwrite(path, depth_u16)
    if not ok:
        raise IOError(f'Failed to save depth PNG: {path}')


class PalletDataCollector(Node):
    def __init__(self):
        super().__init__('pallet_data_collector')

        # 参数
        self.declare_parameter('rgb_topic', '/camera_robot/rgb/image_raw')
        self.declare_parameter('depth_topic', '/camera_robot/depth/image_raw')
        self.declare_parameter('save_dir', os.path.expanduser('~/pallet_test_samples'))
        self.declare_parameter('depth_min_m', 0.1)
        self.declare_parameter('depth_max_m', 8.0)

        self.rgb_topic = self.get_parameter('rgb_topic').value
        self.depth_topic = self.get_parameter('depth_topic').value
        self.save_dir = self.get_parameter('save_dir').value
        self.depth_min = float(self.get_parameter('depth_min_m').value)
        self.depth_max = float(self.get_parameter('depth_max_m').value)

        os.makedirs(self.save_dir, exist_ok=True)
        self.bridge = CvBridge()

        # 状态
        self._lock = threading.Lock()
        self._latest_bgr = None
        self._latest_depth = None
        self._latest_rgb_stamp = None
        self._save_count = 0

        # 订阅与同步
        sub_rgb = message_filters.Subscriber(self, Image, self.rgb_topic)
        sub_dep = message_filters.Subscriber(self, Image, self.depth_topic)
        self._sync = message_filters.ApproximateTimeSynchronizer(
            [sub_rgb, sub_dep], queue_size=10, slop=0.05)
        self._sync.registerCallback(self._sync_callback)

        # 服务：触发保存
        self._srv = self.create_service(
            Trigger, 'save_pallet_sample', self._handle_save_request)

        self.get_logger().info(
            f'\n  RGB  topic : {self.rgb_topic}'
            f'\n  Depth topic: {self.depth_topic}'
            f'\n  Save dir   : {self.save_dir}'
            f'\n  Depth clip : [{self.depth_min}, {self.depth_max}] m'
            f'\n------------------------------------------------'
            f'\n  调用服务保存一帧:'
            f'\n    ros2 service call /save_pallet_sample std_srvs/srv/Trigger "{{}}"'
        )

    def _sync_callback(self, rgb_msg: Image, depth_msg: Image):
        """同步回调：更新最新 BGR + depth."""
        bgr = self.bridge.imgmsg_to_cv2(rgb_msg, 'bgr8')
        depth = decode_depth(self.bridge, depth_msg)

        depth = np.array(depth, dtype=np.float32)
        depth[~np.isfinite(depth)] = 0.0
        depth[(depth < self.depth_min) | (depth > self.depth_max)] = 0.0

        with self._lock:
            self._latest_bgr = bgr
            self._latest_depth = depth
            self._latest_rgb_stamp = rgb_msg.header.stamp

    def _handle_save_request(self, request, response):
        """服务回调：保存当前最新 RGB + depth 一帧。"""
        with self._lock:
            if self._latest_bgr is None or self._latest_depth is None:
                response.success = False
                response.message = '尚未接收到任何 RGB/Depth 帧'
                return response

            bgr = self._latest_bgr.copy()
            depth = self._latest_depth.copy()
            stamp = self._latest_rgb_stamp

        self._save_count += 1
        idx = self._save_count
        prefix = os.path.join(self.save_dir, f'sample_{idx:04d}')

        # 1) RGB PNG
        rgb_path = f'{prefix}_rgb.png'
        cv2.imwrite(rgb_path, bgr)

        # 2) 深度 NPY（保留，方便数值检查）
        npy_path = f'{prefix}_depth.npy'
        np.save(npy_path, depth)

        # 3) 深度 PNG（16位, 单位mm）—— 与 DatasetCollector 一致
        depth_png_path = f'{prefix}_depth_vis.png'
        save_depth_png_mm(depth, depth_png_path)

        # 4) 深度可视化彩色图（仅查看用）
        depth_color_path = f'{prefix}_depth_color.png'
        depth_color = depth_to_colormap(depth, min_m=self.depth_min, max_m=self.depth_max)
        cv2.imwrite(depth_color_path, depth_color)

        # 5) meta 信息
        now = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        meta_path = f'{prefix}_meta.txt'

        valid = depth[depth > 0]
        depth_min_val = float(valid.min()) if valid.size > 0 else 0.0
        depth_max_val = float(valid.max()) if valid.size > 0 else 0.0

        with open(meta_path, 'w') as f:
            f.write(
                f'index            : {idx}\n'
                f'wall_time        : {now}\n'
                f'ros_stamp        : {stamp.sec}.{stamp.nanosec:09d}\n'
                f'rgb_topic        : {self.rgb_topic}\n'
                f'depth_topic      : {self.depth_topic}\n'
                f'image_size       : {bgr.shape[1]}x{bgr.shape[0]}\n'
                f'depth_encoding   : uint16 png in millimeter\n'
                f'depth_load       : cv2.imread(path, cv2.IMREAD_ANYDEPTH).astype(np.float32) / 1000.0\n'
                f'depth_clip_range : {self.depth_min:.3f} ~ {self.depth_max:.3f} m\n'
                f'depth_valid_min  : {depth_min_val:.3f} m\n'
                f'depth_valid_max  : {depth_max_val:.3f} m\n'
            )

        msg = (
            f'[采集 #{idx:04d}] '
            f'rgb={os.path.basename(rgb_path)} '
            f'depth_png={os.path.basename(depth_png_path)} '
            f'depth_npy={os.path.basename(npy_path)} '
            f'color={os.path.basename(depth_color_path)}'
        )
        self.get_logger().info(msg)

        response.success = True
        response.message = msg
        return response


def main(args=None):
    rclpy.init(args=args)
    node = PalletDataCollector()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()