#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo
from geometry_msgs.msg import PoseStamped
from cv_bridge import CvBridge
import cv2
import numpy as np
import math

class PalletDetector(Node):
    def __init__(self):
        super().__init__('pallet_detector')
        
        # 创建图像转换器
        self.bridge = CvBridge()
        
        # 订阅RGB图像
        self.rgb_subscription = self.create_subscription(
            Image,
            '/camera_robot/rgbd_camera/image_raw',
            self.rgb_callback,
            10
        )
        
        # 订阅深度图像
        self.depth_subscription = self.create_subscription(
            Image,
            '/camera_robot/rgb/image_raw',  # 注意：需要修改为实际的深度话题
            self.depth_callback,
            10
        )
        
        # 订阅相机信息（用于内参）
        self.camera_info_subscription = self.create_subscription(
            CameraInfo,
            '/camera_robot/rgb/camera_info',
            self.camera_info_callback,
            10
        )
        
        # 发布标记后的图像
        self.annotated_publisher = self.create_publisher(
            Image,
            '/camera_robot/annotated_image',
            10
        )
        
        # 发布托盘位姿
        self.pallet_pose_publisher = self.create_publisher(
            PoseStamped,
            '/camera_robot/pallet_pose',
            10
        )
        
        # 初始化变量
        self.camera_matrix = None
        self.dist_coeffs = None
        self.current_rgb = None
        self.current_depth = None
        self.camera_info_received = False
        
        # 托盘尺寸（单位：米）
        self.pallet_width = 0.3  # 托盘宽度
        self.pallet_height = 0.3  # 托盘高度
        self.fork_hole_width = 0.125  # 叉孔宽度
        self.fork_hole_height = 0.05  # 叉孔高度
        
        # 托盘3D角点（在托盘坐标系中）
        # 定义托盘边界框的3D点
        self.pallet_3d_points = np.array([
            # 底部四个角点
            [-self.pallet_width/2, -self.pallet_height/2, 0.0],  # 左下后
            [self.pallet_width/2, -self.pallet_height/2, 0.0],   # 右下后
            [self.pallet_width/2, self.pallet_height/2, 0.0],    # 右前上
            [-self.pallet_width/2, self.pallet_height/2, 0.0],   # 左前上
        ], dtype=np.float32)
        
        # 用于可视化显示
        self.display_text = ""
        self.pallet_position = None
        self.pallet_orientation = None
        
        self.get_logger().info("托盘识别节点已启动!")
        self.get_logger().info("等待相机数据...")
    
    def camera_info_callback(self, msg):
        """获取相机内参"""
        if not self.camera_info_received:
            # 构造相机内参矩阵
            self.camera_matrix = np.array([
                [msg.k[0], msg.k[1], msg.k[2]],
                [msg.k[3], msg.k[4], msg.k[5]],
                [msg.k[6], msg.k[7], msg.k[8]]
            ], dtype=np.float32)
            
            # 畸变系数
            self.dist_coeffs = np.array(msg.d, dtype=np.float32)
            
            self.camera_info_received = True
            self.get_logger().info("相机内参已接收")
    
    def rgb_callback(self, msg):
        """处理RGB图像"""
        try:
            # 转换为OpenCV格式
            cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            self.current_rgb = cv_image
            
            # 如果深度图可用，进行处理
            if self.current_rgb is not None and self.camera_info_received:
                self.detect_pallet()
                
        except Exception as e:
            self.get_logger().error(f"RGB图像转换错误: {e}")
    
    def depth_callback(self, msg):
        """处理深度图像"""
        try:
            # 注意：根据Gazebo设置，深度图像可能是32FC1格式
            depth_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='32FC1')
            self.current_depth = depth_image
            
        except Exception as e:
            self.get_logger().warning(f"深度图像转换错误: {e}")
            # 如果没有深度图，使用默认深度
            self.current_depth = None
    
    def preprocess_image(self, image):
        """图像预处理"""
        # 转换为灰度图
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # 高斯模糊
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # 自适应阈值
        binary = cv2.adaptiveThreshold(
            blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV, 11, 2
        )
        
        return binary
    
    def detect_fork_holes(self, binary_image):
        """检测叉孔"""
        # 形态学操作去除小噪点
        kernel = np.ones((5, 5), np.uint8)
        cleaned = cv2.morphologyEx(binary_image, cv2.MORPH_CLOSE, kernel)
        cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_OPEN, kernel)
        
        # 查找轮廓
        contours, _ = cv2.findContours(cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # 过滤轮廓
        fork_holes = []
        for contour in contours:
            # 计算轮廓面积
            area = cv2.contourArea(contour)
            
            # 计算轮廓的边界矩形
            x, y, w, h = cv2.boundingRect(contour)
            
            # 根据托盘叉孔的特征进行过滤
            # 叉孔应该是矩形，长宽比约0.3/0.125=2.4
            aspect_ratio = w / h if h > 0 else 0
            
            # 筛选条件（根据图像调整）
            if area > 500 and 1.5 < aspect_ratio < 4.0:
                # 计算矩形度
                rect_area = w * h
                extent = area / rect_area if rect_area > 0 else 0
                
                if extent > 0.6:  # 矩形度较高
                    fork_holes.append({
                        'contour': contour,
                        'bbox': (x, y, w, h),
                        'center': (x + w//2, y + h//2),
                        'area': area
                    })
        
        return fork_holes
    
    def pair_fork_holes(self, fork_holes):
        """配对叉孔（两个叉孔应该大致水平对齐）"""
        if len(fork_holes) < 2:
            return None
        
        # 按y坐标（垂直位置）排序
        fork_holes.sort(key=lambda h: h['center'][1])
        
        # 找到y坐标相近的叉孔对
        pairs = []
        for i in range(len(fork_holes)):
            for j in range(i+1, len(fork_holes)):
                hi = fork_holes[i]
                hj = fork_holes[j]
                
                # 检查y坐标是否相近（在同一水平线上）
                y_diff = abs(hi['center'][1] - hj['center'][1])
                
                # 检查x坐标是否有足够距离（叉孔间的距离）
                x_diff = abs(hi['center'][0] - hj['center'][0])
                
                # 筛选条件：y坐标相近，x坐标有一定距离
                if y_diff < 50 and x_diff > 100:
                    pairs.append((hi, hj))
        
        # 返回最佳匹配对（面积相近的）
        if pairs:
            # 选择面积最相近的一对
            best_pair = min(pairs, key=lambda p: abs(p[0]['area'] - p[1]['area']))
            return best_pair
        
        return None
    
    def estimate_pallet_pose(self, fork_hole_pair):
        """估计托盘位姿"""
        if fork_hole_pair is None or self.camera_matrix is None:
            return None
        
        hole1, hole2 = fork_hole_pair
        
        # 获取叉孔的中心点
        x1, y1 = hole1['center']
        x2, y2 = hole2['center']
        
        # 排序：左叉孔和右叉孔
        if x1 < x2:
            left_hole = hole1
            right_hole = hole2
        else:
            left_hole = hole2
            right_hole = hole1
        
        # 提取叉孔的4个角点（用于PnP求解）
        # 简单起见，使用叉孔的边界框角点
        lx, ly, lw, lh = left_hole['bbox']
        rx, ry, rw, rh = right_hole['bbox']
        
        # 2D图像点（叉孔的四个角点）
        image_points = np.array([
            [lx, ly],           # 左叉孔左上角
            [lx + lw, ly],      # 左叉孔右上角
            [rx, ry],           # 右叉孔左上角
            [rx + rw, ry],      # 右叉孔右上角
        ], dtype=np.float32)
        
        # 对应的3D点（在托盘坐标系中）
        # 假设叉孔在托盘底部，z=0
        object_points = np.array([
            [-self.fork_hole_width/2, self.pallet_height/2 - self.fork_hole_height, 0.0],  # 左叉孔左上
            [self.fork_hole_width/2, self.pallet_height/2 - self.fork_hole_height, 0.0],   # 左叉孔右上
            [-self.fork_hole_width/2, -self.pallet_height/2 + self.fork_hole_height, 0.0], # 右叉孔左上
            [self.fork_hole_width/2, -self.pallet_height/2 + self.fork_hole_height, 0.0],  # 右叉孔右上
        ], dtype=np.float32)
        
        try:
            # 使用solvePnP求解位姿
            success, rvec, tvec = cv2.solvePnP(
                object_points,
                image_points,
                self.camera_matrix,
                self.dist_coeffs
            )
            
            if success:
                # 将旋转向量转换为旋转矩阵
                rmat, _ = cv2.Rodrigues(rvec)
                
                return {
                    'translation': tvec.flatten(),
                    'rotation_matrix': rmat,
                    'rotation_vector': rvec.flatten(),
                    'image_points': image_points,
                    'object_points': object_points
                }
        
        except Exception as e:
            self.get_logger().error(f"PnP求解错误: {e}")
        
        return None
    
    def calculate_pallet_bounding_box(self, pose_result):
        """计算托盘的2D边界框"""
        if pose_result is None:
            return None
        
        # 投影3D点到图像平面
        image_points, _ = cv2.projectPoints(
            self.pallet_3d_points,
            pose_result['rotation_vector'],
            pose_result['translation'],
            self.camera_matrix,
            self.dist_coeffs
        )
        
        # 转换为整数坐标
        image_points = image_points.reshape(-1, 2).astype(int)
        
        return image_points
    
    def rotation_matrix_to_euler(self, rmat):
        """旋转矩阵转换为欧拉角"""
        # 从旋转矩阵提取欧拉角（绕ZYX顺序）
        sy = math.sqrt(rmat[0, 0] ** 2 + rmat[1, 0] ** 2)
        
        singular = sy < 1e-6
        
        if not singular:
            x = math.atan2(rmat[2, 1], rmat[2, 2])
            y = math.atan2(-rmat[2, 0], sy)
            z = math.atan2(rmat[1, 0], rmat[0, 0])
        else:
            x = math.atan2(-rmat[1, 2], rmat[1, 1])
            y = math.atan2(-rmat[2, 0], sy)
            z = 0
        
        return np.array([x, y, z])
    
    def detect_pallet(self):
        """主检测函数"""
        if self.current_rgb is None:
            return
        
        # 图像预处理
        binary = self.preprocess_image(self.current_rgb)
        
        # 检测叉孔
        fork_holes = self.detect_fork_holes(binary)
        
        # 配对叉孔
        fork_hole_pair = self.pair_fork_holes(fork_holes)
        
        # 复制图像用于标注
        annotated_image = self.current_rgb.copy()
        
        if fork_hole_pair:
            # 估计位姿
            pose_result = self.estimate_pallet_pose(fork_hole_pair)
            
            if pose_result:
                # 计算边界框
                bbox_points = self.calculate_pallet_bounding_box(pose_result)
                
                if bbox_points is not None:
                    # 绘制边界框
                    color = (0, 0, 255)  # 红色
                    thickness = 2
                    
                    # 连接边界框的点
                    for i in range(4):
                        cv2.line(annotated_image,
                                tuple(bbox_points[i]),
                                tuple(bbox_points[(i+1)%4]),
                                color, thickness)
                    
                    # 绘制叉孔检测结果
                    hole1, hole2 = fork_hole_pair
                    for hole in [hole1, hole2]:
                        x, y, w, h = hole['bbox']
                        cv2.rectangle(annotated_image, (x, y), (x+w, y+h), (0, 255, 0), 2)
                        cv2.circle(annotated_image, hole['center'], 5, (255, 0, 0), -1)
                    
                    # 提取位置和姿态信息
                    tvec = pose_result['translation']
                    rmat = pose_result['rotation_matrix']
                    
                    # 计算欧拉角
                    euler_angles = self.rotation_matrix_to_euler(rmat)
                    
                    # 存储结果
                    self.pallet_position = tvec
                    self.pallet_orientation = euler_angles
                    
                    # 准备显示文本
                    self.display_text = (
                        f"Position: ({tvec[0]:.2f}, {tvec[1]:.2f}, {tvec[2]:.2f}) m\n"
                        f"Orientation: ({math.degrees(euler_angles[0]):.1f}, "
                        f"{math.degrees(euler_angles[1]):.1f}, "
                        f"{math.degrees(euler_angles[2]):.1f}) deg"
                    )
                    
                    # 发布托盘位姿
                    self.publish_pallet_pose(tvec, rmat)
                    
                else:
                    self.display_text = "无法计算边界框"
            else:
                self.display_text = "无法估计位姿"
        else:
            # 显示检测到的叉孔
            for hole in fork_holes:
                x, y, w, h = hole['bbox']
                cv2.rectangle(annotated_image, (x, y), (x+w, y+h), (0, 255, 0), 2)
                cv2.circle(annotated_image, hole['center'], 5, (255, 0, 0), -1)
            
            self.display_text = f"检测到 {len(fork_holes)} 个候选区域"
            self.pallet_position = None
            self.pallet_orientation = None
        
        # 添加文本到图像
        self.add_text_overlay(annotated_image)
        
        # 发布标注后的图像
        self.publish_annotated_image(annotated_image)
    
    def add_text_overlay(self, image):
        """在图像上添加文本覆盖层"""
        # 添加半透明背景
        overlay = image.copy()
        cv2.rectangle(overlay, (10, 10), (500, 120), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, image, 0.3, 0, image)
        
        # 添加标题
        cv2.putText(image, "Pallet Detection Result", (20, 40),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        
        # 添加检测结果
        if self.display_text:
            y_offset = 70
            for line in self.display_text.split('\n'):
                cv2.putText(image, line, (20, y_offset),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                y_offset += 25
        
        # 添加状态指示器
        status_color = (0, 255, 0) if self.pallet_position is not None else (0, 0, 255)
        status_text = "DETECTED" if self.pallet_position is not None else "SEARCHING"
        cv2.putText(image, f"Status: {status_text}", (20, 160),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, status_color, 2)
    
    def publish_pallet_pose(self, translation, rotation_matrix):
        """发布托盘位姿"""
        pose_msg = PoseStamped()
        
        # 设置头信息
        pose_msg.header.stamp = self.get_clock().now().to_msg()
        pose_msg.header.frame_id = "camera_link"
        
        # 设置位置
        pose_msg.pose.position.x = float(translation[0])
        pose_msg.pose.position.y = float(translation[1])
        pose_msg.pose.position.z = float(translation[2])
        
        # 将旋转矩阵转换为四元数
        # 这里简化处理，实际应根据旋转矩阵计算四元数
        pose_msg.pose.orientation.x = 0.0
        pose_msg.pose.orientation.y = 0.0
        pose_msg.pose.orientation.z = 0.0
        pose_msg.pose.orientation.w = 1.0
        
        self.pallet_pose_publisher.publish(pose_msg)
    
    def publish_annotated_image(self, image):
        """发布标注后的图像"""
        try:
            msg = self.bridge.cv2_to_imgmsg(image, "bgr8")
            msg.header.stamp = self.get_clock().now().to_msg()
            msg.header.frame_id = "camera_link"
            self.annotated_publisher.publish(msg)
        except Exception as e:
            self.get_logger().error(f"发布图像错误: {e}")

def main(args=None):
    rclpy.init(args=args)
    
    detector = PalletDetector()
    
    try:
        rclpy.spin(detector)
    except KeyboardInterrupt:
        pass
    finally:
        detector.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()