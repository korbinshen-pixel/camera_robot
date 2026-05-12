#!/usr/bin/env python3
import math

import rclpy
from rclpy.node import Node
from rclpy.duration import Duration
from rclpy.time import Time

from geometry_msgs.msg import Twist, PoseStamped
from std_msgs.msg import Bool
from visualization_msgs.msg import Marker
from tf2_ros import (
    Buffer,
    TransformListener,
    LookupException,
    ConnectivityException,
    ExtrapolationException,
)

from nav2_simple_commander.robot_navigator import BasicNavigator, TaskResult
from gazebo_msgs.msg import ModelStates


def quat_to_yaw(q) -> float:
    siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
    cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
    return math.atan2(siny_cosp, cosy_cosp)


def yaw_to_quat(yaw: float):
    from geometry_msgs.msg import Quaternion
    q = Quaternion()
    q.x = 0.0
    q.y = 0.0
    q.z = math.sin(yaw / 2.0)
    q.w = math.cos(yaw / 2.0)
    return q


class PalletPickupMission(Node):

    STATE_WAIT_NAV      = 'WAIT_NAV'
    STATE_SEARCH        = 'SEARCH'          # 原地搜索 + 第一次命中
    STATE_STOP_SPIN     = 'STOP_SPIN'       # 停车 + 第一次 10 帧平均
    STATE_NAV_APPROACH  = 'NAV_APPROACH'    # 纯导航，不再用检测
    STATE_FINAL_PICKUP  = 'FINAL_PICKUP'    # 终点处二次 10 帧检查 + 直行
    STATE_DONE          = 'DONE'

    def __init__(self):
        super().__init__('pallet_pickup_mission')

        # 是否自动开始任务（Nav2 ready 后立即开始搜索）
        self.declare_parameter('auto_start', False)
        self.auto_start = self.get_parameter('auto_start').value

        # 参数
        self.declare_parameter('search_angular_z', 0.30)   # 保留，但 SEARCH 已不再直接用
        self.declare_parameter('goal_offset_m', 2.00)
        self.declare_parameter('final_forward_speed', 0.08)
        self.declare_parameter('final_forward_time', 6.0)
        self.declare_parameter('stop_spin_duration', 0.8)  # 这版不再用，可以留着
        self.declare_parameter('lock_confirm_count', 5)    # 这版不再用，可以留着
        self.declare_parameter('pallet_min_dist', 0.4)
        self.declare_parameter('pallet_max_dist', 3.5)
        self.declare_parameter('nav_timeout_sec', 60.0)
        self.declare_parameter('pallet_topic', '/camera_robot/pallet_pose')
        self.declare_parameter('cmd_vel_topic', '/camera_robot/cmd_vel')
        self.declare_parameter('map_frame', 'map')

        self.angular_z_search  = self.get_parameter('search_angular_z').value
        self.goal_offset       = self.get_parameter('goal_offset_m').value
        self.final_speed       = self.get_parameter('final_forward_speed').value
        self.final_time        = self.get_parameter('final_forward_time').value
        self.stop_spin_dur     = self.get_parameter('stop_spin_duration').value
        self.lock_count_thresh = self.get_parameter('lock_confirm_count').value
        self.pallet_min_dist   = self.get_parameter('pallet_min_dist').value
        self.pallet_max_dist   = self.get_parameter('pallet_max_dist').value
        self.nav_timeout       = self.get_parameter('nav_timeout_sec').value
        self.map_frame         = self.get_parameter('map_frame').value

        # 状态
        self.state = self.STATE_WAIT_NAV
        self.state_entry_time = self.get_clock().now()

        self.latest_pallet = None
        self.lock_count = 0                  # 不再作为关键逻辑，只用于 log
        self.locked_pallet_map = None
        self.nav_task = None
        self.final_start_time = None

        # 两段 10 帧采样缓冲
        self.first_samples = []              # 第一次命中后用于平均（相机系）
        self.second_samples = []             # 到达目标后用于验证（相机系）
        self.samples_needed = 20
        self._stop_settle_start = None
        self._extra_spin_done = False

        # 任务开始标志（默认为 auto_start）
        self.start_mission = self.auto_start

        # 搜索旋转相关：Nav2 Spin 分段旋转
        self.search_spin_task_active = False
        self.search_spin_angle = math.radians(25.0)   # 每次搜索旋转 25°
        self.search_spin_timeout = 8                  # 每次 spin 的最大允许时间（秒，int）

        # 发布 / 订阅
        self.cmd_pub = self.create_publisher(
            Twist,
            self.get_parameter('cmd_vel_topic').value,
            10
        )

        self.pose_sub = self.create_subscription(
            PoseStamped,
            self.get_parameter('pallet_topic').value,
            self._pallet_callback,
            10
        )

        # 订阅 start_mission 信号
        self.start_sub = self.create_subscription(
            Bool,
            'start_mission',
            self._start_cb,
            10
        )

        self.pallet_marker_pub = self.create_publisher(
            Marker,
            '/camera_robot/pallet_marker_est',
            10
        )
        self.goal_marker_pub = self.create_publisher(
            Marker,
            '/camera_robot/pallet_goal_marker',
            10
        )
        # 订阅 Gazebo 模型状态
        self.gazebo_sub = self.create_subscription(
            ModelStates,
            '/gazebo/model_states',
            self._gazebo_model_states_cb,
            10
        )
        self._gazebo_pallet_pose = None
        self.PALLET_MODEL_NAME = 'pallet'  # ← 改成你在 Gazebo 里的模型名

        # TF / Nav2
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        self.navigator = BasicNavigator()
        self.nav_ready = False
        self.nav_wait_logged = False

        # 控制循环
        self.control_timer = self.create_timer(0.1, self._control_loop)

        self.get_logger().info(
            '\n'
            '  ╔══════════════════════════════════════╗\n'
            '  ║    Pallet Pickup Mission  Started    ║\n'
            '  ╚══════════════════════════════════════╝\n'
            f'  搜索角速度(参数) : {self.angular_z_search:.2f} rad/s\n'
            f'  搜索spin角度     : {math.degrees(self.search_spin_angle):.1f}°/次\n'
            f'  停车偏移         : {self.goal_offset:.2f} m\n'
            f'  终段速度         : {self.final_speed:.2f} m/s × {self.final_time:.1f} s\n'
            f'  话题订阅         : {self.get_parameter("pallet_topic").value}\n'
            f'  导航全局框架     : {self.map_frame}\n'
            f'  auto_start       : {self.auto_start}\n'
            '  手动开始话题     : /start_mission (std_msgs/Bool, data: true)\n'
        )

    # ---------------- 基础工具 ----------------

    def _start_cb(self, msg: Bool):
        if msg.data:
            if not self.start_mission:
                self.get_logger().info('[MISSION] 收到 start_mission=True，允许开始搜索托盘')
            self.start_mission = True

    def _pallet_callback(self, msg: PoseStamped):
        tx = msg.pose.position.x
        ty = msg.pose.position.y
        tz = msg.pose.position.z
        dist = tz

        if not (self.pallet_min_dist <= dist <= self.pallet_max_dist):
            return

        self.latest_pallet = msg

        # 搜索阶段：只是做计数统计
        if self.state == self.STATE_SEARCH:
            self.lock_count += 1

        # 第一次停车采样：只在 STOP_SPIN 状态收集 10 帧
        if self.state == self.STATE_STOP_SPIN:
            if len(self.first_samples) < self.samples_needed:
                self.first_samples.append(msg)

        # 到达目标点后再次采样 10 帧，用 second_samples
        if self.state == self.STATE_FINAL_PICKUP and self.final_start_time is None:
            if len(self.second_samples) < self.samples_needed:
                self.second_samples.append(msg)

    def _stop(self):
        self.cmd_pub.publish(Twist())

    def _forward(self):
        t = Twist()
        t.linear.x = self.final_speed
        self.cmd_pub.publish(t)

    def _elapsed(self) -> float:
        return (self.get_clock().now() - self.state_entry_time).nanoseconds / 1e9

    def _enter_state(self, new_state: str):
        self.get_logger().info(f'[状态] {self.state} → {new_state}')
        self.state = new_state
        self.state_entry_time = self.get_clock().now()
        # 进入 SEARCH 时统一清掉所有残留任务 / 计时
        if new_state == self.STATE_SEARCH:
            try:
                self.navigator.cancelTask()
            except Exception:
                pass
            self.search_spin_task_active = False
            self.final_start_time = None
            self.second_samples = []

        if new_state == self.STATE_STOP_SPIN:
            self._stop_settle_start = None
            self._settle_flushed = False
            self.first_samples = []

    # Nav2 Spin 搜索控制
    def _start_nav_spin_search(self):
        try:
            self.navigator.spin(
                spin_dist=self.search_spin_angle,
                time_allowance=int(self.search_spin_timeout)
            )
            self.search_spin_task_active = True
            self.get_logger().info(
                f'[搜索] 调用 Nav2 Spin，旋转 {math.degrees(self.search_spin_angle):.1f}°'
            )
        except Exception as e:
            self.get_logger().error(f'[搜索] Nav2 Spin 启动失败: {e}')
            self.search_spin_task_active = False

    # ---------------- Nav2 就绪检查 ----------------

    def _check_nav_ready(self):
        if self.nav_ready:
            return True

        try:
            self.navigator.waitUntilNav2Active()
        except Exception as e:
            if not self.nav_wait_logged:
                self.get_logger().info(f'[NAV] 等待 Nav2 active 中: {e}')
                self.nav_wait_logged = True
            return False

        try:
            self.tf_buffer.lookup_transform(
                self.map_frame,
                'base_link',
                Time(),
                Duration(seconds=0.5)
            )
            self.nav_ready = True
            self.get_logger().info('[NAV] Nav2 + AMCL 已就绪 ✓')
            return True
        except (LookupException, ConnectivityException, ExtrapolationException) as e:
            if not self.nav_wait_logged:
                self.get_logger().info(f'[NAV] Nav2 active，但 TF 还没 ready: {e}')
                self.nav_wait_logged = True
            return False

    # ---------------- 坐标 / 目标计算 ----------------

    def _gazebo_model_states_cb(self, msg: ModelStates):
        if self.PALLET_MODEL_NAME in msg.name:
            idx = msg.name.index(self.PALLET_MODEL_NAME)
            pose = PoseStamped()
            pose.header.frame_id = 'map'
            pose.header.stamp = self.get_clock().now().to_msg()
            pose.pose = msg.pose[idx]   # 直接拿 position + orientation
            self._gazebo_pallet_pose = pose

    def _estimate_pallet_in_map(self, avg_cam: PoseStamped) -> PoseStamped:
        """
        使用 Tx/Tz + 机器人位姿估计托盘在 map 下的平面位置。
        假设 camera_link 与 base_link 无旋转（SDF 中 rpy=0），
        相机系与 base_link 系对齐：x 前、y 左、z 上。
        avg_cam.header.frame_id 应该是 camera_link。
        """
        try:
            tf_rb = self.tf_buffer.lookup_transform(
                self.map_frame,
                'base_link',
                Time(),
                Duration(seconds=0.5)
            )
        except (LookupException, ConnectivityException, ExtrapolationException) as e:
            self.get_logger().warn(f'[TF] 获取 base_link 位姿失败: {e}')
            return None

        bx = tf_rb.transform.translation.x
        by = tf_rb.transform.translation.y
        q_rb = tf_rb.transform.rotation
        robot_yaw = quat_to_yaw(q_rb)

        # 你的定义：Tx 左右（负左正右），Tz 前后（正前）
        tx = avg_cam.pose.position.x
        tz = avg_cam.pose.position.z

        # 在 base_link 系里，“前”是 x，“左”是 y
        cos_r = math.cos(robot_yaw)
        sin_r = math.sin(robot_yaw)

        # 前方向量 (cos_r, sin_r)
        # 左方向量 (-sin_r, cos_r)
        dx = tz * cos_r + tx * ( sin_r)
        dy = tz * sin_r + tx * (-cos_r)

        px = bx + dx
        py = by + dy

        pallet_map = PoseStamped()
        pallet_map.header.frame_id = self.map_frame
        pallet_map.header.stamp = self.get_clock().now().to_msg()
        pallet_map.pose.position.x = px
        pallet_map.pose.position.y = py
        pallet_map.pose.position.z = 0.0

        # yaw 先带上网络的托盘 yaw（后面算方向要用）
        pallet_map.pose.orientation = avg_cam.pose.orientation
        return pallet_map

    def _compute_nav_goal(self, pallet_map: PoseStamped) -> PoseStamped:
        """
        托盘正反面对称，计算正面和背面两个候选目标点，选距离机器人更近的那个。
        目标点 = 托盘中心沿正面/背面方向偏移 goal_offset；
        目标朝向 = 从目标点看向托盘中心。
        """
        px = pallet_map.pose.position.x
        py = pallet_map.pose.position.y

        # 当前机器人 yaw（map 下）
        try:
            tf_rb = self.tf_buffer.lookup_transform(
                self.map_frame,
                'base_link',
                Time(),
                Duration(seconds=0.5)
            )
            q_rb = tf_rb.transform.rotation
            robot_yaw = quat_to_yaw(q_rb)
            robot_x = tf_rb.transform.translation.x
            robot_y = tf_rb.transform.translation.y
        except (LookupException, ConnectivityException, ExtrapolationException) as e:
            self.get_logger().warn(f'[TF] 获取 base_link yaw 失败: {e}')
            robot_yaw = 0.0
            robot_x, robot_y = 0.0, 0.0

        # 网络给的托盘 yaw（camera/base_link 系）
        pallet_yaw_cam = quat_to_yaw(pallet_map.pose.orientation)

        # 托盘正前方向（base_link 系）
        forward_angle_base = pallet_yaw_cam + math.radians(180.0)

        # base_link → map 旋转
        cos_r = math.cos(robot_yaw)
        sin_r = math.sin(robot_yaw)
        nx_base = math.cos(forward_angle_base)
        ny_base = math.sin(forward_angle_base)
        nx_world = nx_base * cos_r - ny_base * sin_r
        ny_world = nx_base * sin_r + ny_base * cos_r

        # 归一化
        norm = math.hypot(nx_world, ny_world)
        if norm < 1e-6:
            nx_world, ny_world = 1.0, 0.0
        else:
            nx_world /= norm
            ny_world /= norm

        # 正面和背面两个候选目标点
        candidates = []
        for sign, label in [(+1.0, '正面'), (-1.0, '背面')]:
            gx = px + self.goal_offset * sign * nx_world
            gy = py + self.goal_offset * sign * ny_world
            yaw_to_pallet = math.atan2(py - gy, px - gx)

            goal = PoseStamped()
            goal.header.frame_id = self.map_frame
            goal.header.stamp = self.get_clock().now().to_msg()
            goal.pose.position.x = gx
            goal.pose.position.y = gy
            goal.pose.position.z = 0.0
            goal.pose.orientation = yaw_to_quat(yaw_to_pallet)
            candidates.append((goal, label))

        # 选距离机器人更近的候选点
        def dist_to_robot(item):
            g, _ = item
            return math.hypot(g.pose.position.x - robot_x,
                            g.pose.position.y - robot_y)

        chosen_goal, chosen_label = min(candidates, key=dist_to_robot)

        d0 = dist_to_robot(candidates[0])
        d1 = dist_to_robot(candidates[1])
        self.get_logger().info(
            f'[NAV] 选择{chosen_label} | '
            f'正面距离: {d0:.2f}m, 背面距离: {d1:.2f}m | '
            f'目标点: ({chosen_goal.pose.position.x:.2f}, {chosen_goal.pose.position.y:.2f}), '
            f'朝向: {math.degrees(quat_to_yaw(chosen_goal.pose.orientation)):.1f}°'
        )
        return chosen_goal


    def _publish_pallet_marker(self, pallet_map: PoseStamped):
        marker = Marker()
        marker.header.frame_id = self.map_frame
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.ns = 'pallet_est'
        marker.id = 0
        marker.type = Marker.SPHERE
        marker.action = Marker.ADD

        marker.pose.position.x = pallet_map.pose.position.x
        marker.pose.position.y = pallet_map.pose.position.y
        marker.pose.position.z = 0.05
        marker.pose.orientation.w = 1.0

        marker.scale.x = 0.20
        marker.scale.y = 0.20
        marker.scale.z = 0.20

        marker.color.a = 1.0
        marker.color.r = 1.0
        marker.color.g = 0.3
        marker.color.b = 0.0

        marker.lifetime.sec = 0
        self.pallet_marker_pub.publish(marker)

    def _publish_goal_marker(self, goal: PoseStamped):
        # 目标点球
        marker = Marker()
        marker.header.frame_id = self.map_frame
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.ns = 'nav_goal'
        marker.id = 0
        marker.type = Marker.SPHERE
        marker.action = Marker.ADD

        marker.pose.position.x = goal.pose.position.x
        marker.pose.position.y = goal.pose.position.y
        marker.pose.position.z = 0.05
        marker.pose.orientation.w = 1.0

        marker.scale.x = 0.18
        marker.scale.y = 0.18
        marker.scale.z = 0.18

        marker.color.a = 1.0
        marker.color.r = 0.0
        marker.color.g = 0.4
        marker.color.b = 1.0

        marker.lifetime.sec = 0
        self.goal_marker_pub.publish(marker)

        # 目标点朝向箭头
        arrow = Marker()
        arrow.header.frame_id = self.map_frame
        arrow.header.stamp = self.get_clock().now().to_msg()
        arrow.ns = 'nav_goal_arrow'
        arrow.id = 1
        arrow.type = Marker.ARROW
        arrow.action = Marker.ADD
        arrow.pose = goal.pose

        arrow.scale.x = 0.35   # 箭头长度
        arrow.scale.y = 0.06   # 箭头宽
        arrow.scale.z = 0.06

        arrow.color.a = 1.0
        arrow.color.r = 0.0
        arrow.color.g = 0.8
        arrow.color.b = 1.0

        arrow.lifetime.sec = 0
        self.goal_marker_pub.publish(arrow)

    def _average_poses(self, poses):
        """对 camera_link 下多帧位姿做简单位置平均，方向先用第一帧的朝向。"""
        if not poses:
            return None

        sx = sy = sz = 0.0
        for p in poses:
            sx += p.pose.position.x
            sy += p.pose.position.y
            sz += p.pose.position.z
        n = float(len(poses))

        avg = PoseStamped()
        avg.header = poses[0].header
        avg.pose.position.x = sx / n
        avg.pose.position.y = sy / n
        avg.pose.position.z = sz / n
        # 简化：直接用第一帧的 orientation（托盘 yaw）
        avg.pose.orientation = poses[0].pose.orientation
        return avg

    # ---------------- 主状态机 ----------------

    def _control_loop(self):
        # 1. 等 Nav2 / AMCL ready
        if self.state == self.STATE_WAIT_NAV:
            self._stop()
            if not self._check_nav_ready():
                return

            # Nav2 ready 后，如果没 start_mission，就一直静止
            if not self.start_mission:
                self.get_logger().info(
                    '[NAV] Nav2 + AMCL 就绪，等待 /start_mission:=true 才开始搜索...',
                    throttle_duration_sec=5.0
                )
                return

            # Nav2 ready 且允许开始任务，进入搜索
            self._enter_state(self.STATE_SEARCH)
            return

        # 2. 搜索托盘：调用 Nav2 Spin 分段原地旋转，第一次看到就停
        if self.state == self.STATE_SEARCH:
            if self.latest_pallet is not None:
                # 先补转 5°
                if not self._extra_spin_done:
                    if not self.search_spin_task_active:
                        if self.search_spin_task_active:
                            try:
                                self.navigator.cancelTask()
                            except Exception:
                                pass
                        try:
                            self.navigator.spin(
                                spin_dist=math.radians(10.0),
                                time_allowance=5
                            )
                            self.search_spin_task_active = True
                            self.get_logger().info('[搜索] 检测到托盘，补转 10°...')
                        except Exception as e:
                            self.get_logger().error(f'[搜索] 补转失败: {e}，直接停车')
                            self._extra_spin_done = True
                        return

                    if self.navigator.isTaskComplete():
                        self.search_spin_task_active = False
                        self._extra_spin_done = True
                        self.get_logger().info('[搜索] 补转 5° 完成，进入停车采样')
                    return

                # 补转完成，进入停车采样
                self._stop()
                self.get_logger().info('[搜索] 停止并开始采样...')
                self._enter_state(self.STATE_STOP_SPIN)
                return

            # 没检测到托盘，正常 Spin 搜索
            if not self.search_spin_task_active:
                self._start_nav_spin_search()
                return

            if self.navigator.isTaskComplete():
                result = self.navigator.getResult()
                self.search_spin_task_active = False
                self.get_logger().info(f'[搜索] 本轮 Nav2 Spin 完成，result={result}')
            return

        # 3. 停车 + 第一次 10 帧平均，算导航目标
        if self.state == self.STATE_STOP_SPIN:
            self._stop()

            # Step 1: 等停稳 1 秒
            if self._stop_settle_start is None:
                self._stop_settle_start = self.get_clock().now()
                self.get_logger().info('[停车] 等待车体停稳 1 秒...')
                return

            elapsed = (self.get_clock().now() - self._stop_settle_start).nanoseconds / 1e9
            if elapsed < 1.0:
                return

            # Step 2: 1 秒刚到时，清掉停稳前积累的脏数据，重新开始采样
            if not hasattr(self, '_settle_flushed') or not self._settle_flushed:
                self.first_samples = []
                self._settle_flushed = True
                self.get_logger().info('[停车] 停稳完成，开始采样 10 帧...')
                return

            # Step 3: 等够 10 帧
            if len(self.first_samples) < self.samples_needed:
                self.get_logger().info(
                    f'[采样1] 已采样 {len(self.first_samples)}/{self.samples_needed} 帧...',
                    throttle_duration_sec=1.0
                )
                return

            # Step 4: 采样完成，估计位置
            avg_cam = self._average_poses(self.first_samples)
            pallet_map = self._estimate_pallet_in_map(avg_cam)
            if pallet_map is None:
                self.get_logger().warn('[采样1] 估计托盘 map 位姿失败，回到 SEARCH 重新来过')
                self._reset_search()
                return

            self.locked_pallet_map = pallet_map
            self._publish_pallet_marker(pallet_map)

            self.get_logger().info(
                f'[采样1] 托盘平均位置(map): '
                f'({pallet_map.pose.position.x:.2f}, {pallet_map.pose.position.y:.2f})'
            )

            goal = self._compute_nav_goal(self.locked_pallet_map)
            self._publish_goal_marker(goal)
            self.navigator.goToPose(goal)
            self._enter_state(self.STATE_NAV_APPROACH)
            return

        # 4. 导航途中：完全不再依赖检测结果
        if self.state == self.STATE_NAV_APPROACH:
            if self.navigator.isTaskComplete():
                result = self.navigator.getResult()
                if result == TaskResult.SUCCEEDED:
                    self.get_logger().info('[NAV] 已到达托盘前方目标点 ✓')
                    # 到达后开始第二次 10 帧采样
                    self.second_samples = []
                    self.final_start_time = None  # 作为“是否已经开始直行”的标志
                    self._enter_state(self.STATE_FINAL_PICKUP)
                else:
                    self.get_logger().error(f'[NAV] 导航失败 (result={result})，重新搜索...')
                    self._reset_search()
                return

            feedback = self.navigator.getFeedback()
            if feedback:
                try:
                    dist_rem = feedback.distance_remaining
                    nav_time = Duration.from_msg(
                        feedback.navigation_time
                    ).nanoseconds / 1e9
                    self.get_logger().info(
                        f'[NAV] 导航中... 剩余: {dist_rem:.2f} m  已用: {nav_time:.1f} s',
                        throttle_duration_sec=2.0
                    )
                    if nav_time > self.nav_timeout:
                        self.get_logger().error('[NAV] 导航超时，重新搜索')
                        self.navigator.cancelTask()
                        self._reset_search()
                except Exception:
                    pass
            return

        # 5. 终点处：二次 10 帧采样，确认在正前方一定范围内，再直行
        if self.state == self.STATE_FINAL_PICKUP:
            # final_start_time 为空 → 还在做第二次采样
            if self.final_start_time is None:
                self._stop()
                if len(self.second_samples) < self.samples_needed:
                    self.get_logger().info(
                        f'[采样2] 已采样 {len(self.second_samples)}/{self.samples_needed} 帧...',
                        throttle_duration_sec=1.0
                    )
                    return

                # 第二次采样完成，检查托盘是否在正前方一定范围内（仍在相机系下用 Tx/Tz 判）
                avg_cam = self._average_poses(self.second_samples)
                if avg_cam is None:
                    self.get_logger().warn('[采样2] 平均失败，回到 SEARCH')
                    self._reset_search()
                    return

                # 角度检测：托盘 yaw 在相机系下，正对时约为 -90°
                # 允许偏差范围，超出说明托盘侧对着机器人
                pallet_yaw_cam = quat_to_yaw(avg_cam.pose.orientation)
                # 归一化到 [-π, π]
                # 正对托盘时 yaw ≈ ±90°（±π/2），允许 ±30° 偏差
                angle_to_front = abs(pallet_yaw_cam) 
                angle_thresh = math.radians(10.0)  # 可调

                # 简单判定：z 前后，x 左右
                x = avg_cam.pose.position.x
                z = avg_cam.pose.position.z
                lateral_thresh = 0.2  # 侧向误差阈值，可做成参数

                if not (self.pallet_min_dist <= z <= self.pallet_max_dist) or abs(x) > lateral_thresh:
                    self.get_logger().warn(
                        f'[采样2] 托盘不在正前方范围内(z={z:.2f}, x={x:.2f})，回到 SEARCH'
                    )
                    self._reset_search()
                    return

                if angle_to_front > angle_thresh:
                    self.get_logger().warn(
                        f'[采样2] 托盘角度偏差过大({math.degrees(angle_to_front):.1f}° > '
                        f'{math.degrees(angle_thresh):.1f}°)，回到 SEARCH'
                    )
                    self._reset_search()
                    return


                self.get_logger().info(
                    f'[采样2] 托盘在正前方范围内，开始直行靠近(z={z:.2f}, x={x:.2f})'
                )
                # 设置直行计时起点
                self.final_start_time = self.get_clock().now()
                return

            # final_start_time 已经设置 → 正在直行阶段
            elapsed = (self.get_clock().now() - self.final_start_time).nanoseconds / 1e9
            if elapsed < self.final_time:
                self._forward()
                self.get_logger().info(
                    f'[拾取] 直线前进 {elapsed:.1f}/{self.final_time:.1f} s',
                    throttle_duration_sec=1.0
                )
            else:
                self._stop()
                self._enter_state(self.STATE_DONE)
                self.get_logger().info(
                    '\n  ╔══════════════════════════════════╗\n'
                    '  ║   拾取任务完成！Mission Complete ║\n'
                    '  ╚══════════════════════════════════╝'
                )
            return

        if self.state == self.STATE_DONE:
            self._stop()

    def _reset_search(self):
        # 重置时也要确保 Nav2 任务被取消
        try:
            self.navigator.cancelTask()
        except Exception:
            pass
        self.search_spin_task_active = False
        self.lock_count = 0
        self.latest_pallet = None
        self.locked_pallet_map = None
        self.first_samples = []
        self.second_samples = []
        self.final_start_time = None
        self._settle_flushed = False
        self._stop_settle_start = None 
        self._enter_state(self.STATE_SEARCH)

    def destroy_node(self):
        self._stop()
        try:
            self.navigator.cancelTask()
        except Exception:
            pass
        self.search_spin_task_active = False
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = PalletPickupMission()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info('Keyboard interrupt, shutting down...')
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == '__main__':
    main()