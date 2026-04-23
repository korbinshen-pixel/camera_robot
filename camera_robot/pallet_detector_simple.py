#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import numpy as np
import math

class SimplePalletDetector(Node):
    def __init__(self):
        super().__init__('simple_pallet_detector')
        
        # 创建图像转换器
        self.bridge = CvBridge()
        
        # 订阅RGB图像
        self.rgb_subscription = self.create_subscription(
            Image,
            '/camera_robot/rgbd_camera/image_raw',
            self.image_callback,
            10
        )
        
        # 发布标记后的图像
        self.annotated_publisher = self.create_publisher(
            Image,
            '/camera_robot/annotated_image',
            10
        )
        
        # 托盘参数
        self.pallet_width_pixels = 100  # 估计的托盘像素宽度
        self.focal_length = 500  # 估计的焦距（像素）
        
        self.get_logger().info("简化托盘识别节点已启动!")
    
    def image_callback(self, msg):
        """处理图像回调"""
        try:
            # 转换为OpenCV格式
            cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            
            # 检测托盘
            annotated_image, position_text = self.detect_pallet_simple(cv_image)
            
            # 发布标注后的图像
            self.publish_annotated_image(annotated_image)
            
            if position_text:
                self.get_logger().info(position_text)
                
        except Exception as e:
            self.get_logger().error(f"图像处理错误: {e}")
    
    def detect_pallet_simple(self, image):
        """简化的托盘检测"""
        # 转换为HSV颜色空间
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        
        # 定义木色（托盘颜色）的HSV范围
        lower_wood = np.array([10, 50, 50])
        upper_wood = np.array([30, 255, 255])
        
        # 创建掩码
        mask = cv2.inRange(hsv, lower_wood, upper_wood)
        
        # 形态学操作
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        
        # 查找轮廓
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # 复制图像用于标注
        annotated = image.copy()
        position_text = None
        
        if contours:
            # 找到最大的轮廓（假设是托盘）
            largest_contour = max(contours, key=cv2.contourArea)
            
            # 计算边界框
            x, y, w, h = cv2.boundingRect(largest_contour)
            
            # 过滤太小或太大的区域
            if w > 50 and h > 50:
                # 绘制红色边界框
                cv2.rectangle(annotated, (x, y), (x+w, y+h), (0, 0, 255), 3)
                
                # 计算托盘中心
                center_x = x + w // 2
                center_y = y + h // 2
                
                # 绘制中心点
                cv2.circle(annotated, (center_x, center_y), 5, (255, 0, 0), -1)
                
                # 估计距离（基于透视原理）
                # 假设已知托盘实际宽度为0.3m
                distance = (0.3 * self.focal_length) / w if w > 0 else 0
                
                # 计算相对于相机的近似位置
                # 假设图像中心为相机光心
                image_center_x = image.shape[1] // 2
                image_center_y = image.shape[0] // 2
                
                x_offset = (center_x - image_center_x) * distance / self.focal_length
                y_offset = (center_y - image_center_y) * distance / self.focal_length
                
                # 准备位置文本
                position_text = (
                    f"Pallet detected! "
                    f"Distance: {distance:.2f}m, "
                    f"X offset: {x_offset:.2f}m, "
                    f"Y offset: {y_offset:.2f}m"
                )
                
                # 在图像上添加文本
                cv2.putText(annotated, f"Distance: {distance:.2f}m", 
                           (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                cv2.putText(annotated, f"Position: ({x_offset:.2f}, {y_offset:.2f})", 
                           (x, y+h+25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        
        # 添加检测状态
        status = "DETECTED" if position_text else "SEARCHING"
        color = (0, 255, 0) if position_text else (0, 0, 255)
        cv2.putText(annotated, f"Status: {status}", (20, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        
        return annotated, position_text
    
    def publish_annotated_image(self, image):
        """发布标注后的图像"""
        try:
            msg = self.bridge.cv2_to_imgmsg(image, "bgr8")
            self.annotated_publisher.publish(msg)
        except Exception as e:
            self.get_logger().error(f"发布图像错误: {e}")

def main(args=None):
    rclpy.init(args=args)
    
    detector = SimplePalletDetector()
    
    try:
        rclpy.spin(detector)
    except KeyboardInterrupt:
        pass
    finally:
        detector.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()