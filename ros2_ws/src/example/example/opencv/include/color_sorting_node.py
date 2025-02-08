#!/usr/bin/env python3
# encoding: utf-8
# 颜色分拣
import os
import cv2
import yaml
import math
import time
import queue
import rclpy
import signal
import threading
import numpy as np
from sdk import common
from rclpy.node import Node
from std_srvs.srv import Trigger
from kinematics_msgs.srv import SetRobotPose
from sensor_msgs.msg import Image, CameraInfo
from rclpy.executors import MultiThreadedExecutor
from interfaces.msg import ColorsInfo, ColorDetect
from servo_controller_msgs.msg import ServosPosition
from rclpy.callback_groups import ReentrantCallbackGroup
from kinematics.kinematics_control import set_pose_target
from interfaces.srv import SetColorDetectParam, SetCircleROI
from servo_controller.bus_servo_control import set_servo_position

TARGET_POSITION = {
    "red": (0.055, 0.245, 0.015),
    "green": (-0.01, 0.245, 0.015),
    "blue": (-0.085, 0.245, 0.015),
    "tag_1": (-0.075, 0.175, 0.015),
    "tag_2": (-0.005, 0.175, 0.015),
    "tag_3": (0.065, 0.175, 0.015)
}



class ColorSortingNode(Node):
    hand2cam_tf_matrix = [
    [0.0, 0.0, 1.0, -0.101],
    [-1.0, 0.0, 0.0, 0.011],
    [0.0, -1.0, 0.0, 0.045],
    [0.0, 0.0, 0.0, 1.0]
]
    def __init__(self, name):
        rclpy.init()
        super().__init__(name, allow_undeclared_parameters=True, automatically_declare_parameters_from_overrides=True)
        self.name = name
        self.running = True
        self.start_place = False
        self.start_move = False
        self.start_count = False
        self.pick_pitch = 80
        self.language = os.environ['ASR_LANGUAGE']
        self.config_file = 'transform.yaml'
        self.config_path = "/home/jetson/repos/ros2_ws/src/app/config/"
        self.pose_t = None
        self.start = False
        self.start_pick = False
        self.target_color = ''
        self.color = ''
        self.count = 0
        self.image_queue = queue.Queue(maxsize=2)
        signal.signal(signal.SIGINT, self.shutdown)
        with open(self.config_path + self.config_file, 'r') as f:
            config = yaml.safe_load(f)

            # 转换为 numpy 数组
            extristric = np.array(config['extristric'])
            white_area_center = np.array(config['white_area_pose_world'])
        self.white_area_center = white_area_center
        tvec = extristric[:1]  # 平移向量
        rmat = extristric[1:]  # 旋转矩阵
        tvec, rmat = common.extristric_plane_shift(np.array(tvec).reshape((3, 1)), np.array(rmat), 0.030)
        self.extristric = tvec, rmat
        self.create_subscription(ColorsInfo, '/color_detect/color_info', self.get_color_callback, 1)        

        self.camera_info_sub = self.create_subscription(CameraInfo, '/depth_cam/depth/camera_info', self.camera_info_callback, 1)
        self.create_subscription(Image, '/color_detect/image_result', self.image_callback, 1)
        timer_cb_group = ReentrantCallbackGroup()
        self.create_service(Trigger, '~/start', self.start_srv_callback, callback_group=timer_cb_group) # 进入玩法
        self.create_service(Trigger, '~/stop', self.stop_srv_callback, callback_group=timer_cb_group) # 退出玩法


        self.joints_pub = self.create_publisher(ServosPosition, '/servo_controller', 1)
        
        #等待服务启动
        self.client = self.create_client(Trigger, '/controller_manager/init_finish')
        self.client.wait_for_service()
        self.client = self.create_client(Trigger, '/kinematics/init_finish')
        self.client.wait_for_service()
        self.kinematics_client = self.create_client(SetRobotPose, '/kinematics/set_pose_target')
        self.kinematics_client.wait_for_service()

        self.set_color_client = self.create_client(SetColorDetectParam, '/color_detect/set_param', callback_group=timer_cb_group)
        self.set_roi_client = self.create_client(SetCircleROI, '/color_detect/set_circle_roi', callback_group=timer_cb_group)
        self.set_color_client.wait_for_service()
        self.set_roi_client.wait_for_service()
        self.debug = self.get_parameter('debug').value

        self.timer = self.create_timer(0.0, self.init_process, callback_group=timer_cb_group)

    def init_process(self):
        self.timer.cancel()
        
        set_servo_position(self.joints_pub, 1.0, ((1, 500), (2, 560), (3, 130), (4, 115), (5, 500), (10, 200)))  # 设置机械臂初始位置
        time.sleep(1)
        if self.get_parameter('start').value:
            self.start_srv_callback(Trigger.Request(), Trigger.Response())

        threading.Thread(target=self.main, daemon=True).start()
        self.create_service(Trigger, '~/init_finish', self.get_node_state)
        self.get_logger().info('\033[1;32m%s\033[0m' % 'start')

    def get_node_state(self, request, response):
        response.success = True
        return response

    def shutdown(self, signum, frame):
        self.running = False

    def send_request(self, client, msg):
        future = client.call_async(msg)
        while rclpy.ok():
            if future.done() and future.result():
                return future.result()
    def camera_info_callback(self, msg):
        self.K = np.matrix(msg.k).reshape(1, -1, 3)

    def start_srv_callback(self, request, response):
        self.get_logger().info('\033[1;32m%s\033[0m' % "start color sorting")

        msg = SetColorDetectParam.Request()
        msg_red = ColorDetect()
        msg_red.color_name = 'red'
        msg_red.detect_type = 'rect'
        msg_green = ColorDetect()
        msg_green.color_name = 'green'
        msg_green.detect_type = 'rect'
        msg_blue = ColorDetect()
        msg_blue.color_name = 'blue'
        msg_blue.detect_type = 'rect'
        msg.data = [msg_red, msg_green, msg_blue]
        res = self.send_request(self.set_color_client, msg)
        if res.success:
            self.get_logger().info('\033[1;32m%s\033[0m' % 'set color success')
        else:
            self.get_logger().info('\033[1;32m%s\033[0m' % 'set color fail')
        self.start = True
        response.success = True
        response.message = "start"
        return response
    
    def stop_srv_callback(self, request, response):
        self.get_logger().info('\033[1;32m%s\033[0m' % "stop color sorting")
        self.start = False
        # self.start_move = True
        res = self.send_request(self.set_color_client, SetColorDetectParam.Request())
        if res.success:
            self.get_logger().info('\033[1;32m%s\033[0m' % 'set color success')
        else:
            self.get_logger().info('\033[1;32m%s\033[0m' % 'set color fail')
        response.success = True
        response.message = "stop"
        return response   

    def get_color_callback(self, msg):
        data = msg.data
        if data == []:
                self.color = ''

        else:
            if not self.start_move:
                for i in data:
                    x = data[0].x
                    y = data[0].y
                    center = (int(x), int(y))
                    self.angle = data[0].angle
                    self.color = data[0].color

                    projection_matrix = np.row_stack((np.column_stack((self.extristric[1], self.extristric[0])), np.array([[0, 0, 0, 1]])))
                    world_pose = common.pixels_to_world([center, ], self.K, projection_matrix)[0]  # 像素坐标相对于识别区域中心的相对坐标(pixel coordinates relative to the center of the recognition area)
                    world_pose[1] = -world_pose[1]
                    world_pose[2] = 0.02
                    world_pose = np.matmul(self.white_area_center, common.xyz_euler_to_mat(world_pose, (0, 0, 0)))  # 转换到相机相对坐标(convert to the camera relative coordinates)
                    world_pose[2] = 0.02
                    pose_t, _ = common.mat_to_xyz_euler(world_pose)
                    config_data = common.get_yaml_data("/home/jetson/repos/ros2_ws/src/app/config/positions.yaml")
                    offset = tuple(config_data['color_sortting']['offset'])
                    scale = tuple(config_data['color_sortting']['scale'])
                    for i in range(3):

                        pose_t[i] = pose_t[i] + offset[i]
                        pose_t[i] = pose_t[i] * scale[i]
                    self.pose_t = pose_t
    
    def pick(self, pose_t, angle):
        if self.start_move:
            
            pose_t[2] += 0.03
            msg = set_pose_target(pose_t, self.pick_pitch,  [-90.0, 90.0], 1.0)
            res = self.send_request(self.kinematics_client, msg)
            servo_data = res.pulse  
            angle = angle % 90
            angle = angle - 90 if angle > 45 else (angle + 90 if angle < -45 else angle)
                
            angle = 500 + int(1000 * (angle + res.rpy[-1]) / 240)

            # 驱动舵机
            set_servo_position(self.joints_pub, 1.0, ((1, servo_data[0]), (2, servo_data[1]), (3, servo_data[2]), (4, servo_data[3])))
            time.sleep(1)
            
            set_servo_position(self.joints_pub, 1.0, ((5, angle),))
            time.sleep(1)

            pose_t[2] -= 0.03
            msg = set_pose_target(pose_t, self.pick_pitch,  [-90.0, 90.0], 1.0)
            res = self.send_request(self.kinematics_client, msg)
            servo_data = res.pulse 

            set_servo_position(self.joints_pub, 1.0, ((1, servo_data[0]), (2, servo_data[1]), (3, servo_data[2]), (4, servo_data[3])))
            time.sleep(1)
            set_servo_position(self.joints_pub, 1.0, ((10, 600),))
            time.sleep(1)


            pose_t[2] += 0.08
            msg = set_pose_target(pose_t, self.pick_pitch,  [-180.0, 180.0], 1.0)
            res = self.send_request(self.kinematics_client, msg)
            servo_data = res.pulse  

            set_servo_position(self.joints_pub, 1.0, ((2, servo_data[1]), (3, servo_data[2]), (4, servo_data[3])))
            time.sleep(1)
            set_servo_position(self.joints_pub, 1.5, ((1, 500), (2, 610), (3, 70), (4, 140), (5, 500)))  # 设置机械臂初始位置
            time.sleep(1.5)
            self.start_move = False
            self.start_place = True
            threading.Thread(target=self.place, args=(self.target_color,),daemon=True).start()

    def place(self, target_color):
        
        if self.start_place:
            for k, v in TARGET_POSITION.items():
                if target_color in k:
                    position = v
                    break
            position = list(position)

       
            position[2] += 0.03
            msg = set_pose_target(position, self.pick_pitch,  [-90.0, 90.0], 1.0)
            res = self.send_request(self.kinematics_client, msg)
            if res.pulse : # 可以达到
                servo_data = res.pulse  
          
                # 驱动舵机
                set_servo_position(self.joints_pub, 1.0, ((1, servo_data[0]),))
                time.sleep(1)
                set_servo_position(self.joints_pub, 1.0, ((2, servo_data[1]), (3, servo_data[2]), (4, servo_data[3])))
                time.sleep(1)

            position[2] -= 0.03
            msg = set_pose_target(position, self.pick_pitch,  [-90.0, 90.0], 1.0)
            res = self.send_request(self.kinematics_client, msg)
            if res.pulse : # 可以达到
                servo_data = res.pulse 
                set_servo_position(self.joints_pub, 2.0, ((1, servo_data[0]), (2, servo_data[1]), (3, servo_data[2]), (4, servo_data[3])))
                time.sleep(2)

                set_servo_position(self.joints_pub, 1.0, ((10, 200),))
                time.sleep(1)
    
                
                set_servo_position(self.joints_pub, 1.0, ( (2, 610), (3, 70), (4, 140), (5, 500), (10,200)))  
                time.sleep(1)
                set_servo_position(self.joints_pub, 1.0, ((1, 500), ))  
                time.sleep(1)

                self.count = 0
                self.start_place = False
                self.start_count = False
            else:
                self.get_logger().info('\033[1;31m%s\033[0m' % '目标位置无法到达')
                time.sleep(0.01)
    def image_callback(self, ros_image):
        rgb_image = np.ndarray(shape=(ros_image.height, ros_image.width, 3), dtype=np.uint8,
                               buffer=ros_image.data)  # 原始 RGB 画面

        if self.image_queue.full():
            # 如果队列已满，丢弃最旧的图像
            self.image_queue.get()
            # 将图像放入队列
        self.image_queue.put(rgb_image)

    def main(self):
        count = 0
        while self.running:
            try:
                image = self.image_queue.get(block=True, timeout=1)
            except queue.Empty:
                if not self.running:
                    break
                else:
                    continue
            if self.color in ['red', 'green', 'blue'] and not self.start_move and not self.start_count and self.start:
                self.count += 1
                if self.count > 50:
                    self.count = 0
                    self.target_color = self.color
                    self.start_pick = True
                    self.start_srv_callback(Trigger.Request(), Trigger.Response())
                    self.debug = False
                    self.start_count = True
                    self.start_move = True
                
                threading.Thread(target=self.pick, args=(self.pose_t, self.angle), daemon=True).start()
            else:
                count = 0
            if image is not None:
                cv2.imshow('image', image)
                key = cv2.waitKey(1)
                if key == ord('q') or key == 27:  # 按q或者esc退出
                    self.running = False
        rclpy.shutdown()



def main():
    node = ColorSortingNode('color_sorting')
    executor = MultiThreadedExecutor()
    executor.add_node(node)
    executor.spin()
    node.destroy_node()
 
if __name__ == "__main__":
    main()

