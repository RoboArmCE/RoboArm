#!/usr/bin/env python3
# coding: utf8
#标签码垛


import os
import cv2
import yaml
import time
import math
import queue
import rclpy
import threading
import numpy as np
from sdk import common, fps
from rclpy.node import Node
from app.common import Heart
from cv_bridge import CvBridge
from dt_apriltags import Detector
from std_srvs.srv import Trigger, SetBool
from sensor_msgs.msg import Image, CameraInfo
from rclpy.executors import MultiThreadedExecutor
from interfaces.srv import SetStringBool
from servo_controller_msgs.msg import ServosPosition
from rclpy.callback_groups import ReentrantCallbackGroup
from kinematics.kinematics_control import set_pose_target
from kinematics_msgs.srv import GetRobotPose, SetRobotPose
from kinematics.kinematics_control import set_joint_value_target
from servo_controller.bus_servo_control import set_servo_position



class TagStackup(Node):
    hand2cam_tf_matrix = [
    [0.0, 0.0, 1.0, -0.101],
    [-1.0, 0.0, 0.0, 0.011],
    [0.0, -1.0, 0.0, 0.045],
    [0.0, 0.0, 0.0, 1.0]
]
    def __init__(self, name):
        rclpy.init()
        super().__init__(name, allow_undeclared_parameters=True, automatically_declare_parameters_from_overrides=True)
        self.tag_size = 0.025
        self.endpoint = None
        self.start_count = False
        self.start_move = False
        self.stop_thread = False
        self.enable_stackup = False
        self.last_position = None
        self.stackup_step = 99
        self.status = 1
        self.target_position = None
        self.target_count = 0
        self.finish_percent = 0
        self.pick_pitch = 80
        self.entered = False
        self.err_msg = None
        self.thread = None
        self.stop = False
        self.thread_started = False 

        self.image_queue = queue.Queue(maxsize=2)
        self.K = None
        self.D = None

        self.lock = threading.RLock()
        self.fps = fps.FPS()  # 帧率统计器(frame rate counter)
        self.bridge = CvBridge()  # 用于ROS Image消息与OpenCV图像之间的转换

        self.at_detector = Detector(searchpath=['apriltags'], 
                                    families='tag36h11',
                                    nthreads=4,
                                    quad_decimate=1.0,
                                    quad_sigma=0.0,
                                    refine_edges=1,
                                    decode_sharpening=0.25,
                                    debug=0)

        # services and topics
        self.image_sub = None
        self.camera_info_sub = None

        self.joints_pub = self.create_publisher(ServosPosition, '/servo_controller', 1)
        self.result_publisher = self.create_publisher(Image, '~/image_result',  1)
        self.enter_srv = self.create_service(Trigger, '~/enter', self.enter_srv_callback)
        self.exit_srv = self.create_service(Trigger, '~/exit', self.exit_srv_callback)
        self.enable_stack_up_srv = self.create_service(SetBool, '~/enable_stackup', self.enable_stackup_srv_callback)

        timer_cb_group = ReentrantCallbackGroup()
        self.get_current_pose_client = self.create_client(GetRobotPose, '/kinematics/get_current_pose', callback_group=timer_cb_group)
        self.get_current_pose_client.wait_for_service()
        self.kinematics_client = self.create_client(SetRobotPose, '/kinematics/set_pose_target')
        self.kinematics_client.wait_for_service()

        Heart(self, '~/heartbeat', 5, lambda _: self.exit_srv_callback(request=Trigger.Request(), response=Trigger.Response()))  # 心跳包(heartbeat package)

    def get_endpoint(self):

        endpoint = self.send_request(self.get_current_pose_client, GetRobotPose.Request())
        pose_t = endpoint.pose.position
        pose_r = endpoint.pose.orientation

        self.endpoint = common.xyz_quat_to_mat([pose_t.x, pose_t.y, pose_t.z], [pose_r.w, pose_r.x, pose_r.y, pose_r.z])

    def send_request(self, client, msg):
        future = client.call_async(msg)
        while rclpy.ok():
            if future.done() and future.result():
                return future.result()

    def enter_srv_callback(self, request, response):
        self.get_logger().info('\033[1;32m%s\033[0m' % "Loading tag stackup")
        self.image_sub = self.create_subscription(Image, '/depth_cam/rgb/image_raw', self.image_callback, 1)
        self.camera_info_sub = self.create_subscription(CameraInfo, '/depth_cam/depth/camera_info', self.camera_info_callback, 1)
        with self.lock:
            if self.entered:
                response.success = True
                response.message = "start"
            self.thread_started = False 
            self.entered = True
            self.stop = False
            self.enable_stackup = False
            # self.go_home()
            self.thread = threading.Thread(target=self.go_home).start()
            return response

    def camera_info_callback(self, msg):
        with self.lock:
            K = np.matrix(msg.k).reshape(1, -1, 3)
            D = np.array(msg.d)
            new_K, roi = cv2.getOptimalNewCameraMatrix(K, D, (640, 480), 0, (640, 480))
            self.K, self.D = new_K, np.zeros((5, 1))


    def exit_srv_callback(self, request, response):
        self.get_logger().info('\033[1;32m%s\033[0m' % "exit tag stackup")
        try:
            if self.image_sub is not None:
                self.destroy_subscription(self.image_sub)
                self.destroy_subscription(self.camera_info_sub)
                self.image_sub = None
                self.camera_info_sub = None
        except Exception as e:
            self.get_logger().error(str(e))
        with self.lock:
            self.entered = False
            self.stop = True
            self.enable_stackup = False
            response.success = True
            response.message = "stop"
            return response

    def enable_stackup_srv_callback(self, request, response):
        with self.lock:
            if request.data:
                self.enable_stackup = True
                self.target_count = 0
                self.last_position = None
                self.get_logger().info('\033[1;32m%s\033[0m' % "start tag stackup")
                self.get_endpoint()
                self.stackup_step = 99
                self.stop = False
                threading.Thread(target=self.start_set_target).start()

            else:
                self.enable_stackup = False
                self.stop = True
                self.get_logger().info('\033[1;32m%s\033[0m' % "stop tag stackup")
        response.success = True
        response.message = "start"
        return response

    def start_set_target(self):
        self.go_left()
        self.get_endpoint()
        # time.sleep(0.5)
        self.stackup_step = 10

    def set_target(self):
        set_servo_position(self.joints_pub, 1.0, ( (1, 500),(2, 560), (3, 130), (4, 115), (5, 500), (10, 200))) 
        time.sleep(1.0)
        # self.go_home()
        self.get_endpoint()
        self.stackup_step = 0

    def go_home(self):
        set_servo_position(self.joints_pub, 1.0, ( (2, 560), (3, 130), (4, 115), (5, 500), (10, 200)))  
        time.sleep(1.0)
        set_servo_position(self.joints_pub, 1.0, ((1, 500), ))
        time.sleep(1.0)


    def go_left(self):
        set_servo_position(self.joints_pub, 1.0, ((1, 875), (2, 610), (3, 70), (4, 140), (5, 500), (10, 200)))  
        time.sleep(1.0)


    def pick(self, pose_t, angle):
        while self.start_move :
            
            self.start_count = True
            pose_t[2] += 0.03
            msg = set_pose_target(pose_t, self.pick_pitch,  [-90.0, 90.0], 1.0)
            res = self.send_request(self.kinematics_client, msg)
            servo_data = res.pulse  
            angle = angle % 90
            angle = angle - 90 if angle > 45 else (angle + 90 if angle < -45 else angle)
                
            angle = 500 + int(1000 * (angle + res.rpy[-1]) / 240)

            # 驱动舵机

            set_servo_position(self.joints_pub, 1.0, ((1, servo_data[0]), ))
            time.sleep(1)

            set_servo_position(self.joints_pub, 1.0, ( (2, servo_data[1]), (3, servo_data[2]), (4, servo_data[3])))
            time.sleep(1)
            
            set_servo_position(self.joints_pub, 1.0, ((5, angle),))
            time.sleep(1)
            if self.stop:
                self.go_home()
                self.start_move = False
                break
            pose_t[2] -= 0.03
            msg = set_pose_target(pose_t, self.pick_pitch,  [-90.0, 90.0], 1.0)
            res = self.send_request(self.kinematics_client, msg)
            servo_data = res.pulse 

            set_servo_position(self.joints_pub, 1.0, ((1, servo_data[0]), (2, servo_data[1]), (3, servo_data[2]), (4, servo_data[3])))
            time.sleep(1)
            set_servo_position(self.joints_pub, 1.0, ((10, 600),))
            time.sleep(1)

            if self.stop:
                self.go_home()
                self.start_move = False
                break
            
            pose_t[2] += 0.08
            msg = set_pose_target(pose_t, self.pick_pitch,  [-180.0, 180.0], 1.0)
            res = self.send_request(self.kinematics_client, msg)
            servo_data = res.pulse  

            set_servo_position(self.joints_pub, 1.0, ((2, servo_data[1]), (3, servo_data[2]), (4, servo_data[3])))
            time.sleep(1)
            
            if self.stop:
                self.go_home()
                self.start_move = False
                self.start_count = False
                break

            set_servo_position(self.joints_pub, 1.0, ((1, 500), (2, 610), (3, 70), (4, 140), (5, 500)))  # 设置机械臂初始位置
            time.sleep(1.5)
            if self.stop:
                self.go_home()
                self.start_move = False
                break
            self.start_move = False
            self.start_place = True
            threading.Thread(target=self.place,daemon=True).start()
            break
    def place(self):
        
        while self.start_place :
            self.config_data = common.get_yaml_data("/home/jetson/repos/ros2_ws/src/app/config/positions.yaml")

            position =  list(self.config_data['tag_stackup']['target_1'])
            position[2] = position[2] + self.target_position[0][2]

            position[2] += 0.03
            msg = set_pose_target(position, self.pick_pitch,  [-90.0, 90.0], 1.0)
            res = self.send_request(self.kinematics_client, msg)
            servo_data = res.pulse  
          
            # 驱动舵机
            set_servo_position(self.joints_pub, 1.0, ((1, servo_data[0]),))
            time.sleep(1)

            set_servo_position(self.joints_pub, 1.0, ((2, servo_data[1]), (3, servo_data[2]), (4, servo_data[3])))
            time.sleep(1)

            if self.stop:
                self.go_home()
                self.target_position = None
                self.start_place = False
                self.start_count = False
                break

            position[2] -= 0.03
            msg = set_pose_target(position, self.pick_pitch,  [-90.0, 90.0], 1.0)
            res = self.send_request(self.kinematics_client, msg)
            servo_data = res.pulse 

            angle = 500 + int(1000 * (-90 + res.rpy[-1]) / 240)
            set_servo_position(self.joints_pub, 2.0, ((1, servo_data[0]), (2, servo_data[1]), (3, servo_data[2]), (4, servo_data[3])))
            time.sleep(2)
            
            set_servo_position(self.joints_pub, 1.0, ((5, angle),))
            time.sleep(1)

            set_servo_position(self.joints_pub, 0.5, ((10, 200),))
            time.sleep(0.5)
                
            if self.stop:
                self.go_home()
                self.target_position = None
                self.start_place = False
                self.start_count = False
                break
            threading.Thread(target=self.start_set_target).start()
            self.target_position = None
            self.start_place = False
            self.start_count = False 
            break

    def image_callback(self, ros_image):
        # 将ros格式图像转换为opencv格式(convert the ros format image to opencv format)
        cv_image = self.bridge.imgmsg_to_cv2(ros_image, "rgb8")
        rgb_image = np.array(cv_image, dtype=np.uint8)
        if not self.thread_started:
            # 只在第一次调用时启动线程
            threading.Thread(target=self.image_processing, args=(ros_image,), daemon=True).start()
            self.thread_started = True 

        if self.image_queue.full():
            # # 如果队列已满，丢弃最旧的图像
            self.image_queue.get()
        # # 将图像放入队列
        self.image_queue.put(rgb_image)

    def image_processing(self, ros_image):

        while True:
            rgb_image = self.image_queue.get()
            result_image = np.copy(rgb_image)
            if (self.stackup_step == 0 or self.stackup_step == 10) and self.K is not None and self.D is not None and self.hand2cam_tf_matrix is not None and self.endpoint is not None  :
                tags = self.at_detector.detect(cv2.cvtColor(rgb_image, cv2.COLOR_RGB2GRAY), True, (self.K[0,0], self.K[1,1], self.K[0,2], self.K[1,2]), self.tag_size)
                if len(tags) > 0:
                    common.draw_tags(result_image, tags, corners_color=(0, 0, 255), center_color=(0, 255, 0))
                    pose_end = np.matmul(self.hand2cam_tf_matrix, common.xyz_rot_to_mat(tags[0].pose_t, tags[0].pose_R))  # 转换的末端相对坐标(relative coordinates of the converted end)
                    pose_world = np.matmul(self.endpoint, pose_end)  # 转换到机械臂世界坐标(convert to the robotic arm world coordinates)
                    pose_world_T, pose_world_euler = common.mat_to_xyz_euler(pose_world, degrees=True)
                    cv2.putText(result_image, "{:.3f} {:.3f}".format(pose_world_T[0], pose_world_T[1]), (int(tags[0].center[0]-50), int(tags[0].center[1]+22)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
                    r = pose_world_euler[2] % 90  # 将旋转角限制到 ±45°(limit the rotation angle to ±45°)
                    r = r - 90 if r > 45 else (r + 90 if r < -45 else r)
                    pose_world_euler[-1] = r
                    if  self.last_position is None or common.distance(self.last_position[0], pose_world_T) > 0.005 or self.start_count :  # 前后距离大于 5mm 就是结果不可信(if the distance forward or backward is greater than 5mm, the result is considered unreliable)
                        self.target_count = 0
                    else:
                        self.target_count += 1
                    self.last_position = pose_world_T, pose_world_euler
                    if self.target_count > 15:
                        self.target_count = 0
                        if self.stackup_step == 0 and self.enable_stackup and tags[0].tag_id != 100:
                            config_data = common.get_yaml_data("/home/jetson/repos/ros2_ws/src/app/config/positions.yaml")
                            offset = tuple(config_data['tag_stackup']['offset'])
                            scale = tuple(config_data['tag_stackup']['scale'])
                            pose_world_T[2] = 0.015
                            for i in range(3):
                                pose_world_T[i] = pose_world_T[i]  + offset[i]
                                pose_world_T[i] = pose_world_T[i]  * scale[i]

                            pose_world_T[2] += (math.sqrt(pose_world_T[1] ** 2 + pose_world_T[0] ** 2) - 0.21) / 0.20 * 0.030
                            self.status = 1
                            threading.Thread(target=self.pick, args=(pose_world_T, pose_world_euler[2]), daemon=True).start()
                        elif self.stackup_step == 10:
                            self.get_logger().info('\033[1;32m%s\033[0m' % "set target")
                            self.target_position = pose_world_T, pose_world_euler
                            # self.get_logger().info('\033[1;32m%s\033[0m' % 'target position: %s' % str(self.target_position))
                            if pose_world_T[-1] > 0.07 and self.enable_stackup:
                                self.err_msg = "Too high, please remove some blocks first!!!"
                            else:
                                self.err_msg = None
                        
                                self.start_move = True
                                threading.Thread(target=self.set_target).start()
                else:
                    self.target_count = 0
            if self.err_msg is not None:
                self.get_logger().error(self.err_msg)
                err_msg = self.err_msg.split(';')
                for i, m in enumerate(err_msg):
                    cv2.putText(result_image, m, (10, 150 + (i * 30)), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 7)
                    cv2.putText(result_image, m, (10, 150 + (i * 30)), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)

            # result_image = cv2.cvtColor(result_image, cv2.COLOR_RGB2BGR)
            self.result_publisher.publish(self.bridge.cv2_to_imgmsg(result_image, "rgb8"))


def main():
    node = TagStackup('tag_stackup')
    executor = MultiThreadedExecutor()
    executor.add_node(node)
    executor.spin()
    node.destroy_node()
    rclpy.shutdown()

if __name__ == "__main__":
    main()


