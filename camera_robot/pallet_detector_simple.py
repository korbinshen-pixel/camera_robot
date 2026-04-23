#!/usr/bin/env python3
"""
pallet_detector_simple.py
简化托盘检测节点：HSV 颜色过滤 + 深度图测距。
- 订阅 /camera_robot/rgb/image_raw      (彩色图)
- 订阅 /camera_robot/depth/image_raw    (深度图, 32FC1, 单位: 米)
- 发布 /camera_robot/annotated_image    (标注结果图)
"""

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import numpy as np


class SimplePalletDetector(Node):
    def __init__(self):
        super().__init__('simple_pallet_detector')

        self.bridge = CvBridge()

        # ---------- 订阅者 ----------
        self.rgb_sub = self.create_subscription(
            Image,
            '/camera_robot/rgb/image_raw',
            self.rgb_callback,
            10
        )
        self.depth_sub = self.create_subscription(
            Image,
            '/camera_robot/depth/image_raw',
            self.depth_callback,
            10
        )

        # ---------- 发布者 ----------
        self.annotated_pub = self.create_publisher(
            Image,
            '/camera_robot/annotated_image',
            10
        )

        # ---------- 内部状态 ----------
        self.current_depth = None
        # 焦距估计（像素），在没有相机内参时使用
        # Gazebo 默认：horizontal_fov=1.047 rad，width=640 → fx ≈ 640/(2*tan(0.5235))≈ 554
        self.focal_length = 554.0
        # 托盘实际宽度（米），对应 pallet/model.sdf 的 1.2m
        self.pallet_real_width = 1.2

        self.get_logger().info("简化托盘识别节点（深度辅助）已启动！")
        self.get_logger().info("  彩色图：/camera_robot/rgb/image_raw")
        self.get_logger().info("  深度图：/camera_robot/depth/image_raw")

    # ==================== 回调 ====================

    def depth_callback(self, msg):
        try:
            if msg.encoding == '16UC1':
                raw = self.bridge.imgmsg_to_cv2(msg, desired_encoding='16UC1')
                self.current_depth = raw.astype(np.float32) / 1000.0
            else:
                self.current_depth = self.bridge.imgmsg_to_cv2(
                    msg, desired_encoding='32FC1'
                )
        except Exception as e:
            self.get_logger().warning(f"深度图转换错误: {e}")
            self.current_depth = None

    def rgb_callback(self, msg):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            annotated, info = self.detect(cv_image)
            self.publish_annotated(annotated)
            if info:
                self.get_logger().info(info)
        except Exception as e:
            self.get_logger().error(f"图像处理错误: {e}")

    # ==================== 检测 ====================

    def detect(self, image):
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        # 木色 HSV 范围（棕黄色托盘）
        lower_wood = np.array([8,  40,  60])
        upper_wood = np.array([35, 255, 255])
        mask = cv2.inRange(hsv, lower_wood, upper_wood)

        # 形态学去噪
        kernel = np.ones((7, 7), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  kernel)

        contours, _ = cv2.findContours(
            mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        annotated = image.copy()
        info_text = None

        if contours:
            largest = max(contours, key=cv2.contourArea)
            x, y, w, h = cv2.boundingRect(largest)

            if w > 40 and h > 40:
                cx = x + w // 2
                cy = y + h // 2
                img_cx = image.shape[1] // 2
                img_cy = image.shape[0] // 2

                # ---------- 距离估计 ----------
                if self.current_depth is not None:
                    # 取检测框中心区域的中值深度（更鲁棒）
                    dh, dw = self.current_depth.shape[:2]
                    r0 = max(0, cy - h // 6)
                    r1 = min(dh, cy + h // 6)
                    c0 = max(0, cx - w // 6)
                    c1 = min(dw, cx + w // 6)
                    patch = self.current_depth[r0:r1, c0:c1]
                    valid = patch[np.isfinite(patch) & (patch > 0.05)]
                    if valid.size > 0:
                        distance = float(np.median(valid))
                        depth_source = "depth"
                    else:
                        distance = self._estimate_by_size(w)
                        depth_source = "size(fallback)"
                else:
                    distance = self._estimate_by_size(w)
                    depth_source = "size"

                # 水平/垂直偏移（简单针孔模型）
                x_offset = (cx - img_cx) * distance / self.focal_length
                y_offset = (cy - img_cy) * distance / self.focal_length

                # ---------- 绘制 ----------
                cv2.rectangle(annotated, (x, y), (x + w, y + h), (0, 0, 255), 3)
                cv2.circle(annotated, (cx, cy), 6, (255, 0, 0), -1)
                cv2.line(annotated, (img_cx, img_cy), (cx, cy), (200, 200, 0), 1)

                cv2.putText(annotated,
                            f"Dist: {distance:.2f}m [{depth_source}]",
                            (x, max(y - 12, 20)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2)
                cv2.putText(annotated,
                            f"Offset: ({x_offset:.2f}, {y_offset:.2f}) m",
                            (x, y + h + 22),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2)

                info_text = (
                    f"Pallet detected! "
                    f"Dist={distance:.2f}m, "
                    f"X={x_offset:.2f}m, Y={y_offset:.2f}m "
                    f"[{depth_source}]"
                )

        # 状态文字
        color  = (0, 255, 0) if info_text else (0, 0, 255)
        status = "DETECTED"  if info_text else "SEARCHING"
        cv2.putText(annotated, f"Status: {status}", (18, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

        return annotated, info_text

    def _estimate_by_size(self, pixel_width):
        """根据目标像素宽度和已知实际宽度估计距离。"""
        if pixel_width > 0:
            return (self.pallet_real_width * self.focal_length) / pixel_width
        return 0.0

    # ==================== 发布 ====================

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
    node = SimplePalletDetector()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
