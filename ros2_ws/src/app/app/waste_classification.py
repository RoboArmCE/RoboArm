#!/usr/bin/env python3
# encoding: utf-8
# 垃圾分类
import os
import cv2
import yaml
import time
import math
import queue
import rclpy
import signal
import threading
import numpy as np
from sdk import common
from rclpy.node import Node
from app.common import Heart
from cv_bridge import CvBridge
from std_srvs.srv import Trigger, SetBool
from interfaces.msg import ObjectsInfo
from xf_mic_asr_offline import voice_play
from sensor_msgs.msg import Image, CameraInfo
from rclpy.executors import MultiThreadedExecutor
from servo_controller_msgs.msg import ServosPosition
from rclpy.callback_groups import ReentrantCallbackGroup
from kinematics.kinematics_control import set_pose_target
from kinematics_msgs.srv import GetRobotPose, SetRobotPose
from servo_controller.bus_servo_control import set_servo_position

WASTE_CLASSES = {
    'food_waste': ('BananaPeel', 'BrokenBones', 'Ketchup'),
    'hazardous_waste': ('Marker', 'OralLiquidBottle', 'StorageBattery'),
    'recyclable_waste': ('PlasticBottle', 'Toothbrush', 'Umbrella'),
    'residual_waste': ('Plate', 'CigaretteEnd', 'DisposableChopsticks'),
}


class WasteClassificationNode(Node):
    hand2cam_tf_matrix = [
    [0.0, 0.0, 1.0, -0.101],
    [-1.0, 0.0, 0.0, 0.011],
    [0.0, -1.0, 0.0, 0.045],
    [0.0, 0.0, 0.0, 1.0]
]

    def __init__(self, name):
        rclpy.init()
        super().__init__(name, allow_undeclared_parameters=True, automatically_declare_parameters_from_overrides=True)
        self.running = True
        self.center = None
        self.count = 0
        self.stop = False
        self.class_name = None
        self.start_place = False
        self.start_move = True
        self.enable_transport = False
        self.waste_category = None
        self.pick_pitch = 80
        self.current_class_name = None
        self.config_file = 'transform.yaml'
        self.config_path = "/home/enrique/repos/ros2_ws/src/app/config/"

        with open(self.config_path + self.config_file, 'r') as f:
            config = yaml.safe_load(f)

            # 转换为 numpy 数组
            self.extristric = np.array(config['extristric'])
            self.white_area_pose_world = np.array(config['white_area_pose_world'])

        rmat = self.extristric[1:]  # 旋转矩阵
        tvec = self.extristric[:1]  # 平移向量
        rmat = np.array(rmat).reshape(3, 3)
        tvec, rmat = common.extristric_plane_shift(np.array(tvec).reshape((3, 1)), np.array(rmat), 0.030)
        self.extristric = tvec, rmat
        white_area_center = self.white_area_pose_world 
        self.white_area_center = white_area_center
        self.previous_pose = None  # 上一次检测到的位置


        signal.signal(signal.SIGINT, self.shutdown)
        self.bridge = CvBridge()
        self.raw_image_queue = queue.Queue(maxsize=2)
        self.image_queue = queue.Queue(maxsize=2)
        self.joints_pub = self.create_publisher(ServosPosition, '/servo_controller', 1)

        self.timer_cb_group = ReentrantCallbackGroup()

        self.enter_srv = self.create_service(Trigger, '~/enter', self.enter_srv_callback)
        self.exit_srv = self.create_service(Trigger, '~/exit', self.exit_srv_callback)
        self.enable_srv = self.create_service(SetBool, '~/enable_transport', self.start_srv_callback)

        self.result_publisher = self.create_publisher(Image, '~/image_result',  1)
        self.start_yolov8_client = self.create_client(Trigger, '/yolov8/start', callback_group=self.timer_cb_group)
        self.start_yolov8_client.wait_for_service()
        self.stop_yolov8_client = self.create_client(Trigger, '/yolov8/stop', callback_group=self.timer_cb_group)
        self.stop_yolov8_client.wait_for_service()

        Heart(self, '~/heartbeat', 5, lambda _: self.exit_srv_callback(request=Trigger.Request(), response=Trigger.Response()))  # 心跳包(heartbeat package)

        
        #等待服务启动
        self.client = self.create_client(Trigger, '/controller_manager/init_finish')
        self.client.wait_for_service()
        self.client = self.create_client(Trigger, '/kinematics/init_finish')
        self.client.wait_for_service()
        self.kinematics_client = self.create_client(SetRobotPose, '/kinematics/set_pose_target')
        self.kinematics_client.wait_for_service()

        if self.get_parameter('start').value: 
            threading.Thread(target=self.enter_srv_callback,args=(Trigger.Request(), Trigger.Response()), daemon=True).start()
            req = SetBool.Request()
            req.data = True 
            res = SetBool.Response()
            self.start_srv_callback(req, res)

    def go_home(self):
        self.get_logger().info('\033[1;32m%s\033[0m' % "go home")
        if self.waste_category == "recyclable_waste":
            t = 2.0
        elif self.waste_category == "hazardous_waste":
            t = 1.7
        elif self.waste_category == "food_waste":
            t = 1.4
        elif self.waste_category == "residual_waste":
            t = 1.0
        else :
            t = 1.0
        set_servo_position(self.joints_pub, 0.5, ((10, 200), ))
        time.sleep(0.5)
        set_servo_position(self.joints_pub, 1.0, ( (2, 560), (3, 130), (4, 115), (5, 500)))  # 设置机械臂初始位置
        time.sleep(1)
        set_servo_position(self.joints_pub, t, ((1, 500), ))
        time.sleep(t)


    def init_process(self):
        self.timer.cancel()

        set_servo_position(self.joints_pub, 1.0, ((1, 500), (2, 560), (3, 130), (4, 115), (5, 500), (10, 200)))  # 设置机械臂初始位置
        time.sleep(1)

        threading.Thread(target=self.main, daemon=True).start()
        self.create_service(Trigger, '~/init_finish', self.get_node_state)
        self.get_logger().info('\033[1;32m%s\033[0m' % 'start')

    def get_node_state(self, request, response):
        response.success = True
        return response

    def camera_info_callback(self, msg):
        self.K = np.matrix(msg.k).reshape(1, -1, 3)


    def send_request(self, client, msg):
        future = client.call_async(msg)
        while rclpy.ok():
            if future.done() and future.result():
                return future.result()

    def enter_srv_callback(self, request, response):
        self.get_logger().info('\033[1;32m%s\033[0m' % "start garbage classification")

        self.timer = self.create_timer(0.0, self.init_process, callback_group=self.timer_cb_group)
        self.send_request(self.start_yolov8_client, Trigger.Request())
        self.raw_image_sub = self.create_subscription(Image, '/depth_cam/rgb/image_raw', self.raw_image_callback, 1)  # 摄像头订阅(subscribe to the camera)
        self.camera_info_sub = self.create_subscription(CameraInfo, '/depth_cam/depth/camera_info', self.camera_info_callback, 1)
        self.image_sub = self.create_subscription(Image, '/yolov8/object_image', self.image_callback, 1)
        self.object_sub = self.create_subscription(ObjectsInfo, '/yolov8/object_detect', self.get_object_callback, 1)
        response.success = True
        response.message = "start"
        return response
        self.stop = False

    def exit_srv_callback(self, request, response):
        self.get_logger().info('\033[1;32m%s\033[0m' % "stop garbage classification")
        self.send_request(self.stop_yolov8_client, Trigger.Request())
        try:
            if self.raw_image_sub is not None:
                self.destroy_subscription(self.raw_image_sub)
                self.destroy_subscription(self.image_sub)
                self.destroy_subscription(self.object_sub)
                self.destroy_subscription(self.camera_info_sub)
                self.raw_image_sub = None
                self.image_sub = None
                self.object_sub = None
                self.camera_info_sub = None
        except Exception as e:
            self.get_logger().error(str(e))
        self.stop = True
        self.start_move = True
        response.success = True
        response.message = "stop"
        return response
        

    def start_srv_callback(self, request, response):
        if request.data:

            self.get_logger().info('\033[1;32m%s\033[0m' % "start waste classification")
            self.start_move = False
            self.stop = False
            response.success = True
            response.message = "start"
            return response
        else:
            self.get_logger().info('\033[1;32m%s\033[0m' % "stop waste classification")
            self.start_move = True
            self.stop = True
            response.success = False
            response.message = "stop"
            return response

    def shutdown(self, signum, frame):
        self.running = False

    def raw_image_callback(self, ros_image):
        cv_image = self.bridge.imgmsg_to_cv2(ros_image, "bgr8")
        bgr_image = np.array(cv_image, dtype=np.uint8)
        if self.raw_image_queue.full():
            # 如果队列已满，丢弃最旧的图像
            self.raw_image_queue.get()
        # 将图像放入队列
        self.raw_image_queue.put(bgr_image)

    def image_callback(self, ros_image):
        cv_image = self.bridge.imgmsg_to_cv2(ros_image, "bgr8")
        bgr_image = np.array(cv_image, dtype=np.uint8)
        if self.image_queue.full():
            # 如果队列已满，丢弃最旧的图像
            self.image_queue.get()
        # 将图像放入队列
        self.image_queue.put(bgr_image)

    def pick(self, pose_t, angle):
        waste_category = None
        while self.start_move:
            time.sleep(0.2)
            for k, v in WASTE_CLASSES.items():
                if self.current_class_name in v:
                    waste_category = k
                    break
            self.class_name = None
            pose_t[2] += 0.03
            msg = set_pose_target(pose_t, self.pick_pitch,  [-90.0, 90.0], 1.0)
            res = self.send_request(self.kinematics_client, msg)
            servo_data = res.pulse  
          
            angle = 500 + int(1000 * (angle / 240))
            # 驱动舵机
            set_servo_position(self.joints_pub, 1.0, ((1, servo_data[0]), ))
            time.sleep(1)

            set_servo_position(self.joints_pub, 1.0, ((2, servo_data[1]), (3, servo_data[2]), (4, servo_data[3])))
            time.sleep(1)
            
            set_servo_position(self.joints_pub, 1.0, ((5, angle),))
            time.sleep(1)

            if self.stop:
                self.go_home()
                # self.start_move = True
                break
            pose_t[2] -= 0.03
            msg = set_pose_target(pose_t, self.pick_pitch,  [-90.0, 90.0], 1.0)
            res = self.send_request(self.kinematics_client, msg)

            servo_data = res.pulse 

            set_servo_position(self.joints_pub, 1.0, ((1, servo_data[0]), (2, servo_data[1]), (3, servo_data[2]), (4, servo_data[3])))
            time.sleep(1)
            set_servo_position(self.joints_pub, 1.0, ((10, 550),))
            time.sleep(1.0)


            if self.stop:
                self.go_home()
                # self.start_move = True
                break
            
            pose_t[2] += 0.08
            msg = set_pose_target(pose_t, self.pick_pitch,  [-180.0, 180.0], 1.0)
            res = self.send_request(self.kinematics_client, msg)
            servo_data = res.pulse  

            set_servo_position(self.joints_pub, 1.0, ((2, servo_data[1]), (3, servo_data[2]), (4, servo_data[3])))
            time.sleep(1)
            
            if self.stop:
                self.go_home()
                # self.start_move = True
                break
            set_servo_position(self.joints_pub, 1.0, ((1, 500), (2, 610), (3, 70), (4, 140), (5, 500)))  # 设置机械臂初始位置
            time.sleep(1.5)
            if self.stop:
                self.go_home()
                # self.start_move = True
                break
            self.start_place = True
            threading.Thread(target=self.place, args=(waste_category,),daemon=True).start()
            break

    def place(self, waste_category):
        
        while self.start_place :
            self.waste_category = waste_category
            if waste_category == 'residual_waste':
                target = 'target_1'
            elif waste_category == 'food_waste':
                target = 'target_2'
            elif waste_category == 'hazardous_waste':
                target = 'target_3'
            elif waste_category == 'recyclable_waste':
                target = 'target_4'
            config_data = common.get_yaml_data("/home/enrique/repos/ros2_ws/src/app/config/positions.yaml")
            position = list(config_data['waste_classification'][target])
            position[2] += 0.03
            msg = set_pose_target(position, self.pick_pitch,  [-90.0, 90.0], 1.0)
            res = self.send_request(self.kinematics_client, msg)
            servo_data = res.pulse  
          
            # 驱动舵机
            set_servo_position(self.joints_pub, 1.5, ((1, servo_data[0]),))
            time.sleep(1.5)

            set_servo_position(self.joints_pub, 1.0, ((2, servo_data[1]), (3, servo_data[2]), (4, servo_data[3])))
            time.sleep(1)
    
            if self.stop:
                self.go_home()
                self.class_name = None
                self.start_move = True
                self.start_place = False

                break
            position[2] -= 0.03
            msg = set_pose_target(position, self.pick_pitch,  [-90.0, 90.0], 1.0)
            res = self.send_request(self.kinematics_client, msg)
            servo_data = res.pulse 
            set_servo_position(self.joints_pub, 1.0, ((1, servo_data[0]), (2, servo_data[1]), (3, servo_data[2]), (4, servo_data[3])))
            time.sleep(1)
                    
            angle = 500 + int(1000 * ((90 + res.rpy[-1]) / 240)) 
            set_servo_position(self.joints_pub, 1.0, ((5, angle),))
            time.sleep(1)

            set_servo_position(self.joints_pub, 1.0, ((10, 200),))
            time.sleep(1)
            if self.stop:
                self.go_home()
                self.class_name = None
                self.start_move = True
                self.start_place = False
                break
            self.go_home()

            self.class_name = None
            self.start_move = False
            # self.stop = False
            self.start_place = False
            break

    def main(self):
        count = 0
        while self.running:
            try:
                image = self.image_queue.get(block=True, timeout=1)
                self.image = self.image_queue.get(block=True, timeout=1)
            except queue.Empty:
                if not self.running:
                    break
                else:
                    continue
            if self.class_name is not None and not self.start_move  :
                self.count += 1
                if self.count > 15:
                    self.count = 0
                    self.current_class_name = self.class_name
                    self.start_move = True
                    self.enable_transport = True

                self.result_publisher.publish(self.bridge.cv2_to_imgmsg(image, "bgr8"))
            else:
                self.count = 0
                image = self.raw_image_queue.get()
                self.result_publisher.publish(self.bridge.cv2_to_imgmsg(image, "bgr8"))
            if image is not None and self.get_parameter('display').value:
                cv2.imshow('image', image)
                key = cv2.waitKey(1)


    def get_object_callback(self, msg):
        objects = msg.objects
        if objects == []:
            self.center = None
            self.class_name = None
        else:
            for i in objects:
                center = (int(i.box[0]), int(i.box[1]))
                self.class_name = i.class_name
                r = i.angle
                r = r % 90 # 将旋转角限制到 ±45°(limit the rotation angle to ±45°)
                angle = r - 90 if r > 45 else (r + 90 if r < -45 else r)
                projection_matrix = np.row_stack((np.column_stack((self.extristric[1], self.extristric[0])), np.array([[0, 0, 0, 1]])))
                world_pose = common.pixels_to_world([center], self.K, projection_matrix)[0]  # 像素坐标相对于识别区域中心的相对坐标(pixel coordinates relative to the center of the recognition area)
                world_pose[1] = -world_pose[1]
                world_pose[2] = 0.04
                world_pose = np.matmul(self.white_area_center, common.xyz_euler_to_mat(world_pose, (0, 0, 0)))  # 转换到相机相对坐标(convert to the camera relative coordinates)
                world_pose[2] = 0.04
                pose_t, _ = common.mat_to_xyz_euler(world_pose)
                pose_t[2] = 0.015
                config_data = common.get_yaml_data("/home/enrique/repos/ros2_ws/src/app/config/positions.yaml")
                offset = tuple(config_data['waste_classification']['offset'])
                scale = tuple(config_data['waste_classification']['scale'])
                for i in range(3):
                    pose_t[i] = pose_t[i] + offset[i]
                    pose_t[i] = pose_t[i] * scale[i]

                pose_t[2] += (math.sqrt(pose_t[1] ** 2 + pose_t[0] ** 2) - 0.15) / 0.20 * 0.020
                if self.enable_transport:
                    threading.Thread(target=self.pick, args=(pose_t, angle), daemon=True).start()
                    self.enable_transport = False

def main():
    node = WasteClassificationNode('waste_classification')
    executor = MultiThreadedExecutor()
    executor.add_node(node)
    executor.spin()
    node.destroy_node()
    rclpy.shutdown()

if __name__ == "__main__":
    main()

