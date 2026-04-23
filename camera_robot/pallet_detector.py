#!/usr/bin/env python3
"""
pallet_detector.py
完整托盘检测节点：使用 RGBD 深度相机的彩色图 + 深度图联合检测。
- 订阅 /camera_robot/rgb/image_raw      (彩色图)
- 订阅 /camera_robot/depth/image_raw    (深度图, 32FC1, 单位: 米)
- 订阅 /camera_robot/rgb/camera_info    (相机内参)
- 发布 /camera_robot/annotated_image    (标注结果图)
- 发布 /camera_robot/pallet_pose        (托盘位姿 PoseStamped)
"""

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo
from geometry_msgs.msg import PoseStamped
from cv_bridge import CvBridge
import cv2
import numpy as np
import math


def rotation_matrix_to_quaternion(rmat):
    """将 3×3 旋转矩阵转换为四元数 (x, y, z, w)。"""
    trace = rmat[0, 0] + rmat[1, 1] + rmat[2, 2]
    if trace > 0:
        s = 0.5 / math.sqrt(trace + 1.0)
        w = 0.25 / s
        x = (rmat[2, 1] - rmat[1, 2]) * s
        y = (rmat[0, 2] - rmat[2, 0]) * s
        z = (rmat[1, 0] - rmat[0, 1]) * s
    elif rmat[0, 0] > rmat[1, 1] and rmat[0, 0] > rmat[2, 2]:
        s = 2.0 * math.sqrt(1.0 + rmat[0, 0] - rmat[1, 1] - rmat[2, 2])
        w = (rmat[2, 1] - rmat[1, 2]) / s
        x = 0.25 * s
        y = (rmat[0, 1] + rmat[1, 0]) / s
        z = (rmat[0, 2] + rmat[2, 0]) / s
    elif rmat[1, 1] > rmat[2, 2]:
        s = 2.0 * math.sqrt(1.0 + rmat[1, 1] - rmat[0, 0] - rmat[2, 2])
        w = (rmat[0, 2] - rmat[2, 0]) / s
        x = (rmat[0, 1] + rmat[1, 0]) / s
        y = 0.25 * s
        z = (rmat[1, 2] + rmat[2, 1]) / s
    else:
        s = 2.0 * math.sqrt(1.0 + rmat[2, 2] - rmat[0, 0] - rmat[1, 1])
        w = (rmat[1, 0] - rmat[0, 1]) / s
        x = (rmat[0, 2] + rmat[2, 0]) / s
        y = (rmat[1, 2] + rmat[2, 1]) / s
        z = 0.25 * s
    return x, y, z, w


class PalletDetector(Node):
    def __init__(self):
        super().__init__('pallet_detector')

        self.bridge = CvBridge()

        # ---------- 订阅者 ----------
        # 彩色图
        self.rgb_sub = self.create_subscription(
            Image,
            '/camera_robot/rgb/image_raw',
            self.rgb_callback,
            10
        )
        # 深度图（32FC1，单位：米）
        self.depth_sub = self.create_subscription(
            Image,
            '/camera_robot/depth/image_raw',
            self.depth_callback,
            10
        )
        # 相机内参
        self.camera_info_sub = self.create_subscription(
            CameraInfo,
            '/camera_robot/rgb/camera_info',
            self.camera_info_callback,
            10
        )

        # ---------- 发布者 ----------
        self.annotated_pub = self.create_publisher(
            Image,
            '/camera_robot/annotated_image',
            10
        )
        self.pallet_pose_pub = self.create_publisher(
            PoseStamped,
            '/camera_robot/pallet_pose',
            10
        )

        # ---------- 内部状态 ----------
        self.camera_matrix = None
        self.dist_coeffs = None
        self.camera_info_received = False
        self.current_rgb = None
        self.current_depth = None

        # 托盘实际尺寸（单位：米），对应 pallet/model.sdf
        self.pallet_width = 1.2    # X 方向
        self.pallet_height = 0.8   # Y 方向
        # 叉孔（两个叉入口，左右各一）
        self.fork_hole_width = 0.5  # 叉孔 X 长度（近似）
        self.fork_hole_height = 0.15  # 叉孔 Y 宽度

        # 托盘4个角点（托盘坐标系，z=0 平面）
        self.pallet_3d_points = np.array([
            [-self.pallet_width / 2, -self.pallet_height / 2, 0.0],
            [ self.pallet_width / 2, -self.pallet_height / 2, 0.0],
            [ self.pallet_width / 2,  self.pallet_height / 2, 0.0],
            [-self.pallet_width / 2,  self.pallet_height / 2, 0.0],
        ], dtype=np.float32)

        self.display_text = ""
        self.pallet_position = None

        self.get_logger().info("托盘识别节点（深度相机版）已启动！")
        self.get_logger().info("话题监听：")
        self.get_logger().info("  彩色图：/camera_robot/rgb/image_raw")
        self.get_logger().info("  深度图：/camera_robot/depth/image_raw")

    # ==================== 回调 ====================

    def camera_info_callback(self, msg):
        if not self.camera_info_received:
            self.camera_matrix = np.array([
                [msg.k[0], msg.k[1], msg.k[2]],
                [msg.k[3], msg.k[4], msg.k[5]],
                [msg.k[6], msg.k[7], msg.k[8]]
            ], dtype=np.float32)
            self.dist_coeffs = np.array(msg.d, dtype=np.float32)
            self.camera_info_received = True
            self.get_logger().info(
                f"相机内参已接收：fx={msg.k[0]:.1f}, fy={msg.k[4]:.1f}, "
                f"cx={msg.k[2]:.1f}, cy={msg.k[5]:.1f}"
            )

    def rgb_callback(self, msg):
        try:
            self.current_rgb = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            if self.current_rgb is not None and self.camera_info_received:
                self.detect_and_publish()
        except Exception as e:
            self.get_logger().error(f"彩色图转换错误: {e}")

    def depth_callback(self, msg):
        try:
            # 深度图为 32FC1，单位米；Gazebo 可能发布 16UC1（毫米），做兼容处理
            if msg.encoding == '16UC1':
                depth_raw = self.bridge.imgmsg_to_cv2(msg, desired_encoding='16UC1')
                self.current_depth = depth_raw.astype(np.float32) / 1000.0
            else:
                self.current_depth = self.bridge.imgmsg_to_cv2(
                    msg, desired_encoding='32FC1'
                )
        except Exception as e:
            self.get_logger().warning(f"深度图转换错误: {e}")
            self.current_depth = None

    # ==================== 检测主流程 ====================

    def detect_and_publish(self):
        rgb = self.current_rgb
        depth = self.current_depth  # 可为 None，此时退化为纯视觉

        # 1. 预处理 → 提取候选区域
        binary = self.preprocess(rgb)
        candidates = self.find_candidates(binary)

        # 2. 配对叉孔
        pair = self.pair_fork_holes(candidates)

        annotated = rgb.copy()

        if pair:
            hole1, hole2 = pair

            # 3a. 若有深度图，用深度直接获取 3D 位置
            if depth is not None:
                pose_result = self.estimate_pose_from_depth(hole1, hole2, depth)
            else:
                # 3b. 退化：solvePnP（无深度）
                pose_result = self.estimate_pose_pnp(hole1, hole2)

            if pose_result:
                self.draw_detection(annotated, hole1, hole2, pose_result)
                self.publish_pallet_pose(pose_result)
                tvec = pose_result['translation']
                euler = pose_result.get('euler_deg', (0, 0, 0))
                self.display_text = (
                    f"Position: ({tvec[0]:.2f}, {tvec[1]:.2f}, {tvec[2]:.2f}) m  "
                    f"RPY: ({euler[0]:.1f}, {euler[1]:.1f}, {euler[2]:.1f}) deg"
                )
                self.pallet_position = tvec
            else:
                self.display_text = "叉孔已配对，位姿估计失败"
                self.pallet_position = None
        else:
            # 显示所有候选区域
            for h in candidates:
                x, y, w, hh = h['bbox']
                cv2.rectangle(annotated, (x, y), (x + w, y + hh), (0, 200, 0), 1)
            self.display_text = f"候选区域 {len(candidates)} 个，未成功配对"
            self.pallet_position = None

        self.add_overlay(annotated)
        self.publish_annotated(annotated)

    # ==================== 图像处理 ====================

    def preprocess(self, image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        binary = cv2.adaptiveThreshold(
            blurred, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV, 11, 2
        )
        return binary

    def find_candidates(self, binary):
        kernel = np.ones((5, 5), np.uint8)
        cleaned = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
        cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_OPEN, kernel)
        contours, _ = cv2.findContours(
            cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        candidates = []
        for c in contours:
            area = cv2.contourArea(c)
            x, y, w, h = cv2.boundingRect(c)
            if area < 300 or w < 10 or h < 10:
                continue
            aspect = w / h if h > 0 else 0
            if 1.0 < aspect < 6.0:
                extent = area / (w * h) if w * h > 0 else 0
                if extent > 0.5:
                    candidates.append({
                        'contour': c,
                        'bbox': (x, y, w, h),
                        'center': (x + w // 2, y + h // 2),
                        'area': area,
                    })
        return candidates

    def pair_fork_holes(self, candidates):
        if len(candidates) < 2:
            return None
        best_pair = None
        best_score = float('inf')
        for i in range(len(candidates)):
            for j in range(i + 1, len(candidates)):
                hi, hj = candidates[i], candidates[j]
                y_diff = abs(hi['center'][1] - hj['center'][1])
                x_diff = abs(hi['center'][0] - hj['center'][0])
                area_diff = abs(hi['area'] - hj['area'])
                if y_diff < 60 and x_diff > 50:
                    score = y_diff + area_diff / (hi['area'] + 1)
                    if score < best_score:
                        best_score = score
                        best_pair = (hi, hj)
        return best_pair

    # ==================== 位姿估计 ====================

    def estimate_pose_from_depth(self, hole1, hole2, depth):
        """利用深度图获取叉孔中心的 3D 坐标，直接计算托盘位置。"""
        cx_param = self.camera_matrix[0, 2]
        cy_param = self.camera_matrix[1, 2]
        fx = self.camera_matrix[0, 0]
        fy = self.camera_matrix[1, 1]

        h, w = depth.shape[:2]

        def get_3d(px, py):
            # 取 5×5 邻域中值深度，过滤噪声
            r0, r1 = max(0, py - 2), min(h, py + 3)
            c0, c1 = max(0, px - 2), min(w, px + 3)
            patch = depth[r0:r1, c0:c1]
            valid = patch[np.isfinite(patch) & (patch > 0.01)]
            if valid.size == 0:
                return None
            z = float(np.median(valid))
            x3d = (px - cx_param) * z / fx
            y3d = (py - cy_param) * z / fy
            return np.array([x3d, y3d, z])

        p1 = get_3d(*hole1['center'])
        p2 = get_3d(*hole2['center'])

        if p1 is None or p2 is None:
            return self.estimate_pose_pnp(hole1, hole2)

        # 托盘中心 = 两叉孔中心的中点
        center_3d = (p1 + p2) / 2.0

        # 估计偏航角（两叉孔连线方向）
        direction = p2 - p1
        yaw = math.atan2(direction[1], direction[0])

        rmat = np.array([
            [math.cos(yaw), -math.sin(yaw), 0],
            [math.sin(yaw),  math.cos(yaw), 0],
            [0,              0,             1],
        ], dtype=np.float64)

        euler_deg = (0.0, 0.0, math.degrees(yaw))

        return {
            'translation': center_3d,
            'rotation_matrix': rmat,
            'euler_deg': euler_deg,
            'source': 'depth',
        }

    def estimate_pose_pnp(self, hole1, hole2):
        """无深度图时退化为 solvePnP 估计位姿。"""
        if self.camera_matrix is None:
            return None

        if hole1['center'][0] < hole2['center'][0]:
            left, right = hole1, hole2
        else:
            left, right = hole2, hole1

        lx, ly, lw, lh = left['bbox']
        rx, ry, rw, rh = right['bbox']

        image_pts = np.array([
            [lx,       ly      ],
            [lx + lw,  ly      ],
            [rx,       ry      ],
            [rx + rw,  ry      ],
        ], dtype=np.float32)

        object_pts = np.array([
            [-self.fork_hole_width / 2,  self.fork_hole_height / 2, 0.0],
            [ self.fork_hole_width / 2,  self.fork_hole_height / 2, 0.0],
            [-self.fork_hole_width / 2, -self.fork_hole_height / 2, 0.0],
            [ self.fork_hole_width / 2, -self.fork_hole_height / 2, 0.0],
        ], dtype=np.float32)

        try:
            ok, rvec, tvec = cv2.solvePnP(
                object_pts, image_pts,
                self.camera_matrix, self.dist_coeffs
            )
            if not ok:
                return None
            rmat, _ = cv2.Rodrigues(rvec)
            # 欧拉角
            sy = math.sqrt(rmat[0, 0] ** 2 + rmat[1, 0] ** 2)
            if sy > 1e-6:
                rx_a = math.atan2(rmat[2, 1], rmat[2, 2])
                ry_a = math.atan2(-rmat[2, 0], sy)
                rz_a = math.atan2(rmat[1, 0], rmat[0, 0])
            else:
                rx_a = math.atan2(-rmat[1, 2], rmat[1, 1])
                ry_a = math.atan2(-rmat[2, 0], sy)
                rz_a = 0.0
            euler_deg = (
                math.degrees(rx_a),
                math.degrees(ry_a),
                math.degrees(rz_a),
            )
            return {
                'translation': tvec.flatten(),
                'rotation_matrix': rmat,
                'rvec': rvec,
                'euler_deg': euler_deg,
                'source': 'pnp',
            }
        except Exception as e:
            self.get_logger().error(f"solvePnP 错误: {e}")
            return None

    # ==================== 可视化 ====================

    def draw_detection(self, image, hole1, hole2, pose_result):
        for hole in [hole1, hole2]:
            x, y, w, h = hole['bbox']
            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.circle(image, hole['center'], 5, (255, 0, 0), -1)

        # 绘制两叉孔连线
        cv2.line(image, hole1['center'], hole2['center'], (0, 200, 255), 2)

        # 若有相机内参，投影托盘边框
        if (self.camera_matrix is not None
                and pose_result.get('source') == 'pnp'
                and 'rvec' in pose_result):
            pts, _ = cv2.projectPoints(
                self.pallet_3d_points,
                pose_result['rvec'],
                pose_result['translation'].reshape(3, 1),
                self.camera_matrix,
                self.dist_coeffs
            )
            pts = pts.reshape(-1, 2).astype(int)
            for i in range(4):
                cv2.line(image, tuple(pts[i]), tuple(pts[(i + 1) % 4]),
                         (0, 0, 255), 2)

    def add_overlay(self, image):
        overlay = image.copy()
        cv2.rectangle(overlay, (8, 8), (580, 130), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.6, image, 0.4, 0, image)

        cv2.putText(image, "Pallet Detection (RGBD)",
                    (18, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        y0 = 65
        for line in self.display_text.split('  '):
            cv2.putText(image, line.strip(), (18, y0),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            y0 += 24

        color = (0, 255, 0) if self.pallet_position is not None else (0, 0, 255)
        status = "DETECTED" if self.pallet_position is not None else "SEARCHING"
        cv2.putText(image, f"Status: {status}", (18, 125),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    # ==================== 发布 ====================

    def publish_pallet_pose(self, pose_result):
        msg = PoseStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = "camera_link"

        tvec = pose_result['translation']
        msg.pose.position.x = float(tvec[0])
        msg.pose.position.y = float(tvec[1])
        msg.pose.position.z = float(tvec[2])

        rmat = pose_result['rotation_matrix']
        qx, qy, qz, qw = rotation_matrix_to_quaternion(rmat)
        msg.pose.orientation.x = qx
        msg.pose.orientation.y = qy
        msg.pose.orientation.z = qz
        msg.pose.orientation.w = qw

        self.pallet_pose_pub.publish(msg)

    def publish_annotated(self, image):
        try:
            msg = self.bridge.cv2_to_imgmsg(image, "bgr8")
            msg.header.stamp = self.get_clock().now().to_msg()
            msg.header.frame_id = "camera_link"
            self.annotated_pub.publish(msg)
        except Exception as e:
            self.get_logger().error(f"发布标注图像错误: {e}")


def main(args=None):
    rclpy.init(args=args)
    node = PalletDetector()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
