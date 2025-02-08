
#!/usr/bin/env python3
# coding: utf8
#目标分拣

import os
import cv2
import yaml
import time
import math
import queue
import rclpy
import threading
import numpy as np
from rclpy.node import Node
from sdk import common, fps
from app.common import Heart
from cv_bridge import CvBridge
from dt_apriltags import Detector
from std_srvs.srv import Trigger, SetBool
from servo_controller_msgs.msg import Grasp 
from sensor_msgs.msg import Image, CameraInfo
from rclpy.executors import MultiThreadedExecutor
from interfaces.srv import SetStringBool
from servo_controller_msgs.msg import ServosPosition
from rclpy.callback_groups import ReentrantCallbackGroup
from kinematics.kinematics_control import set_pose_target
from kinematics_msgs.srv import GetRobotPose, SetRobotPose
from servo_controller.bus_servo_control import set_servo_position



class ObjectSorttingNode(Node):
    hand2cam_tf_matrix = [
    [0.0, 0.0, 1.0, -0.101],
    [-1.0, 0.0, 0.0, 0.011],
    [0.0, -1.0, 0.0, 0.045],
    [0.0, 0.0, 0.0, 1.0]
]
    def __init__(self, name):
        rclpy.init()
        super().__init__(name, allow_undeclared_parameters=True, automatically_declare_parameters_from_overrides=True)

        self.K = None
        self.D = None

        self.camera_pose = None
        
        self.stop_thread = False
        self.stop = False

        self.start_move = False
        self.start_count = False
        self.target_position_count = 0

        self.lock = threading.RLock()
        self.fps = fps.FPS()    # 帧率统计器(frame rate counter)
        self.bridge = CvBridge()  # 用于ROS Image消息与OpenCV图像之间的转换
        self.thread = None
        self.enable_sortting = False
        self.imgpts = None
        self.tag_size = 0.025
        self.entered = False
        self.center_imgpts = None
        self.roi = None
        self.pick_pitch = 80
        self.moving_step = 0
        self.status = 1
        self.count = 0
        self.white_area_width = 0.175
        self.white_area_height = 0.135
        self.size = {'width': 640, 'height':480}
        self.config_file = 'transform.yaml'
        self.config_path = "/home/jetson/repos/RoboArm/ros2_ws/src/app/config/"
        self.data = common.get_yaml_data("/home/jetson/repos/RoboArm/ros2_ws/src/app/config/lab_config.yaml")  
        self.lab_data = self.data['/**']['ros__parameters'] 
        # self.lab_data = common.get_yaml_data("/home/jetson/repos/RoboArm/software/lab_tool/lab_config.yaml")
        # min_area < 物体的颜色面积 < max_area(min_area < color area of object < max_area)
        self.min_area = 500
        self.max_area = 7000 

        self.target = None
        self.target_labels = {
            "red": False,
            "green": False,
            "blue": False,
            "tag_1": False,
            "tag_2": False,
            "tag_3": False,
        }

        self.at_detector = Detector(searchpath=['apriltags'],
                       families='tag36h11',
                       nthreads=4,
                       quad_decimate=1.0,
                       quad_sigma=0.0,
                       refine_edges=1,
                       decode_sharpening=0.25,
                       debug=0)

        # sub
        self.image_sub = None
        self.camera_info_sub = None
        self.endpoint_info_sub = None
        self.joints_pub = self.create_publisher(ServosPosition, '/servo_controller', 1)
        self.grasp_publisher = self.create_publisher(Grasp, '/grasp', 1)
        self.result_publisher = self.create_publisher(Image, '~/image_result',  1)
        self.image_sub = self.create_subscription(Image, '/depth_cam/rgb/image_raw', self.image_callback, 1)
        self.camera_info_sub = self.create_subscription(CameraInfo, '/depth_cam/depth/camera_info', self.camera_info_callback, 1)

        # services and topics
        self.enter_srv = self.create_service(Trigger, '~/enter', self.enter_srv_callback)
        self.exit_srv = self.create_service(Trigger, '~/exit', self.exit_srv_callback)
        self.enable_sortting_srv = self.create_service(SetBool, '~/enable_sortting', self.enable_sortting_srv_callback)
        self.set_target_srv = self.create_service(SetStringBool, '~/set_color_target', self.set_color_target_srv_callback)
        self.set_target_srv = self.create_service(SetStringBool, '~/set_tag_target', self.set_tag_target_srv_callback)
        timer_cb_group = ReentrantCallbackGroup()
        self.get_current_pose_client = self.create_client(GetRobotPose, '/kinematics/get_current_pose', callback_group=timer_cb_group)
        self.get_current_pose_client.wait_for_service()
        self.kinematics_client = self.create_client(SetRobotPose, '/kinematics/set_pose_target')
        self.kinematics_client.wait_for_service()
        Heart(self, '~/heartbeat', 5, lambda _: self.exit_srv_callback(request=Trigger.Request(), response=Trigger.Response()))  # 心跳包(heartbeat package)
        
        if self.get_parameter('start').value: 
            threading.Thread(target=self.enter_srv_callback,args=(Trigger.Request(), Trigger.Response()), daemon=True).start()
            req = SetBool.Request()
            req.data = True 
            res = SetBool.Response()
            self.enable_sortting_srv_callback(req, res)

            req_1 = SetStringBool.Request()
            req_1.data_bool = True
            req_1.data_str = 'red'
            res_1 = SetBool.Response()
            self.set_color_target_srv_callback(req_1, res_1)

    def load_params_from_yaml(yaml_path):
        with open(yaml_path, 'r') as f:
            return yaml.safe_load(f)

    def go_home(self):
        if self.target is not None and self.target[0] in ["bule", "tag_1"]:
            t = 1.6
        elif self.target is not None and self.target[0] in ["green", "tag_2"]:
            t = 1.3
        elif self.target is not None and self.target[0] in ["red", "tag_3"]:
            t = 1.0
        else :
            t = 1.0
        time.sleep(0.5)
        set_servo_position(self.joints_pub, 1.0, ( (2, 560), (3, 130), (4, 115), (5, 500), (10, 200)))  # 设置机械臂初始位置
        time.sleep(1)
        set_servo_position(self.joints_pub, t, ((1, 500), ))
        time.sleep(t)


    def send_request(self, client, msg):
        future = client.call_async(msg)
        while rclpy.ok():
            if future.done() and future.result():
                return future.result()

    def camera_info_callback(self, msg):
        with self.lock:
            K = np.matrix(msg.k).reshape(1, -1, 3)
            D = np.array(msg.d)
            new_K, roi = cv2.getOptimalNewCameraMatrix(K, D, (640, 480), 0, (640, 480))
            self.K, self.D = np.matrix(new_K), np.zeros((5, 1))


    def enter_srv_callback(self, request, response):
        self.get_logger().info('\033[1;32m%s\033[0m' % "Loading object sortting")
        if self.entered:        
            response.success = True
            response.message = "start"
            return response
        self.entered = True
        self.stop = False

        for k, v in self.target_labels.items():
            self.target_labels[k] = False
        self.enable_sortting = False
        set_servo_position(self.joints_pub, 1.0, ( (1, 500),(2, 560), (3, 130), (4, 115), (5, 500), (10, 200)))  # 设置机械臂初始位置
        time.sleep(0.5)
        with open(self.config_path + self.config_file, 'r') as f:
            config = yaml.safe_load(f)

            # 转换为 numpy 数组
            extristric = np.array(config['extristric'])
            white_area_pose_cam = np.array(config['white_area_pose_cam'])
            white_area_pose_world = np.array(config['white_area_pose_world'])

        # 识别区域的四个角的世界坐标(the world coordinates of the four corners of the recognition area)

        white_area_center = white_area_pose_world.reshape(4, 4)
        self.white_area_center = white_area_center
        white_area_cam = white_area_pose_cam.reshape(4, 4)
        euler_matrix = common.xyz_euler_to_mat((self.white_area_height / 2, self.white_area_width / 2 + 0.0, 0.0), (0, 0, 0))
        white_area_lt = np.matmul(white_area_center, common.xyz_euler_to_mat((self.white_area_height / 2, self.white_area_width / 2 + 0.0, 0.0), (0, 0, 0)))
        white_area_lb = np.matmul(white_area_center, common.xyz_euler_to_mat((-self.white_area_height / 2, self.white_area_width / 2 + 0.0, 0.0), (0, 0, 0)))
        white_area_rb = np.matmul(white_area_center, common.xyz_euler_to_mat((-self.white_area_height / 2, -self.white_area_width / 2 -0.0, 0.0), (0, 0, 0)))
        white_area_rt = np.matmul(white_area_center, common.xyz_euler_to_mat((self.white_area_height / 2, -self.white_area_width / 2 -0.0, 0.0), (0, 0, 0)))


        endpoint = self.send_request(self.get_current_pose_client, GetRobotPose.Request())
        pose_t = endpoint.pose.position
        pose_r = endpoint.pose.orientation

        self.endpoint = common.xyz_quat_to_mat([pose_t.x, pose_t.y, pose_t.z], [pose_r.w, pose_r.x, pose_r.y, pose_r.z])
        corners_cam =  np.matmul(np.linalg.inv(np.matmul(self.endpoint, self.hand2cam_tf_matrix)), [white_area_lt, white_area_lb, white_area_rb, white_area_rt, white_area_center])
        corners_cam = np.matmul(np.linalg.inv(white_area_cam), corners_cam)
        corners_cam = corners_cam[:, :3, 3:].reshape((-1, 3))
        tvec = extristric[:1]  # 取第一行
        rmat = extristric[1:]  # 取后面三行
        while self.K is None or self.D is None:
            time.sleep(0.5)

        with self.lock:
            center_imgpts, jac = cv2.projectPoints(corners_cam[-1:], np.array(rmat), np.array(tvec), self.K, self.D)
            self.center_imgpts = np.int32(center_imgpts).reshape(2)

            tvec, rmat = common.extristric_plane_shift(np.array(tvec).reshape((3, 1)), np.array(rmat), 0.030)
            self.extristric = tvec, rmat
            imgpts, jac = cv2.projectPoints(corners_cam[:-1], np.array(rmat), np.array(tvec), self.K, self.D)
            self.imgpts = np.int32(imgpts).reshape(-1, 2)

            # 裁切出ROI区域(crop RIO region)
            x_min = min(self.imgpts, key=lambda p: p[0])[0] # x轴最小值(the minimum value of X-axis)
            x_max = max(self.imgpts, key=lambda p: p[0])[0] # x轴最大值(the maximum value of X-axis)
            y_min = min(self.imgpts, key=lambda p: p[1])[1] # y轴最小值(the minimum value of Y-axis)
            y_max = max(self.imgpts, key=lambda p: p[1])[1] # y轴最大值(the maximum value of Y-axis)
            roi = np.maximum(np.array([y_min, y_max, x_min, x_max]), 0)
            self.roi = roi
                
        response.success = True
        response.message = "start"
        return response
     
    def exit_srv_callback(self, request, response):
        self.get_logger().info('\033[1;32m%s\033[0m' % "exit  object sortting")
        with self.lock:
            self.entered = False

            self.stop = True
            self.enable_sortting = False
        response.success = True
        response.message = "start"
        return response
    
    def enable_sortting_srv_callback(self, request, response):
        with self.lock:
            if request.data:
                self.get_logger().info('\033[1;32m%s\033[0m' % 'enable  object sortting')
                self.enable_sortting = True
                self.stop = False
            else:
                self.get_logger().info('\033[1;32m%s\033[0m' % 'exit  object sortting')
                self.stop = True
                self.enable_sortting = False
        response.success = True
        response.message = "start"
        return response

        
    def set_color_target_srv_callback(self, request, response):
        self.get_logger().info('\033[1;32mset color target %s %s\033[0m' % (str(request.data_str), str(request.data_bool)))
        if request.data_str in self.target_labels:
            self.target_labels[request.data_str] = request.data_bool

        response.success = True
        response.message = "start"
        return response

    
    def set_tag_target_srv_callback(self, request, response):
        self.get_logger().info('\033[1;32mset tag target %s %s\033[0m' % (str(request.data_str), str(request.data_bool)))
        if "tag_" + request.data_str in self.target_labels:
            self.target_labels['tag_' + request.data_str] = request.data_bool

        response.success = True
        response.message = "start"
        return response

    def point_remapped(self, point, now, new, data_type=float):
        """
        将一个点的坐标从一个图片尺寸映射的新的图片上(map the coordinate of one point from a picture to a new picture of different size)
        :param point: 点的坐标(coordinate of point)
        :param now: 现在图片的尺寸(size of current picture)
        :param new: 新的图片尺寸(new picture size)
        :return: 新的点坐标(new point coordinate)
        """
        x, y = point
        now_w, now_h = now
        new_w, new_h = new
        new_x = x * new_w / now_w
        new_y = y * new_h / now_h
        return data_type(new_x), data_type(new_y)

    def adaptive_threshold(self, gray_image):
        # 用自适应阈值先进行分割, 过滤掉侧面(Segment using adaptive threshold, and filter out the side view)
        # cv2.ADAPTIVE_THRESH_MEAN_C： 邻域所有像素点的权重值是一致的(all neighboring pixel values have equal weights)
        # cv2.ADAPTIVE_THRESH_GAUSSIAN _C ： 与邻域各个像素点到中心点的距离有关，通过高斯方程得到各个点的权重值(the wights if each point are related to the distance between each neighboring pixel and the center pixel, and are calculated using the Gaussian function)
        binary = cv2.adaptiveThreshold(gray_image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 41, 7)
        return binary
    
    def canny_proc(self, bgr_image):
        # 边缘提取，用来进一步分割(当两个相同颜色物体靠在一起时，只能靠边缘去区分)(Edge detection is used for further segmentation (when two objects of the same color are adjacent, only edges can be used to distinguish them))
        mask = cv2.Canny(bgr_image, 9, 41, 9, L2gradient=True)
        mask = 255 - cv2.dilate(mask, cv2.getStructuringElement(cv2.MORPH_RECT, (11, 11)))  # 加粗边界，黑白反转(thicken the edge and invert black and white)
        return mask

    def get_top_surface(self, rgb_image):
        # 为了只提取物体最上层表面(to extract only the top surface of the object)
        # 将图像转换到HSV颜色空间(convert the image to the HSV color space)
        image_scale = cv2.convertScaleAbs(rgb_image, alpha=2.5, beta=0)
        image_gray = cv2.cvtColor(image_scale, cv2.COLOR_RGB2GRAY)
        image_mb = cv2.medianBlur(image_gray, 3)  # 中值滤波(median filtering)
        image_gs = cv2.GaussianBlur(image_mb, (5, 5), 5)  # 高斯模糊去噪(Gaussian blur for noise reduction)
        binary = self.adaptive_threshold(image_gs)  # 阈值自适应(adaptive thresholding)
        mask = self.canny_proc(image_gs)  # 边缘检测(edge detection)
        mask1 = cv2.bitwise_and(binary, mask)  # 合并两个提取出来的图像，保留他们共有的地方(merge two extracted images and retain their common areas)
        roi_image_mask = cv2.bitwise_and(rgb_image, rgb_image, mask=mask1)  # 和原图做遮罩，保留需要识别的区域(mask the original image to retain the area to be recognized)
        return roi_image_mask
    
 
    def pick(self, pose_t, angle):
        while self.start_move:

            pose_t[2] += 0.03
            msg = set_pose_target(pose_t, self.pick_pitch,  [-180.0, 180.0], 1.0)
            res = self.send_request(self.kinematics_client, msg)
            servo_data = res.pulse  
            angle = angle % 90
            angle = angle - 90 if angle > 45 else (angle + 90 if angle < -45 else angle)
                
            # angle =  int(1000 * (angle + res.rpy[-2]) / 240)
            angle = 500 + int(1000 * (angle + res.rpy[-1]) / 240)

            # 驱动舵机
            set_servo_position(self.joints_pub, 1.0, ((1, servo_data[0]), ))
            time.sleep(1)
            set_servo_position(self.joints_pub, 1.0, ((2, servo_data[1]), (3, servo_data[2]), (4, servo_data[3])))
            time.sleep(1)
            
            set_servo_position(self.joints_pub, 1.0, ((5, angle),))
            time.sleep(1)

            if self.stop:
                self.go_home()
                self.start_move = False
                self.start_count = False
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
                self.start_count = False
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
                self.start_count = False
                break

            self.start_move = False
            self.start_place = True
            threading.Thread(target=self.place,daemon=True).start()
            break

    def place(self):
        
        while self.start_place :
            if 'tag' in self.target[0]:
                target = 'target_' + self.target[0][-1]
                config_data = common.get_yaml_data("/home/jetson/repos/RoboArm/ros2_ws/src/app/config/positions.yaml")
                position = list(config_data['tag_sortting'][target])
            else:
                if self.target[0] == 'red':
                    target_name = 'target_1'
                elif self.target[0] == 'green':
                    target_name = 'target_2'
                else:
                    target_name = 'target_3'
                config_data = common.get_yaml_data("/home/jetson/repos/RoboArm/ros2_ws/src/app/config/positions.yaml")

                position = list(config_data['color_sortting'][target_name])

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
                self.start_count = False
                break

            position[2] -= 0.03
            msg = set_pose_target(position, self.pick_pitch,  [-90.0, 90.0], 1.0)
            res = self.send_request(self.kinematics_client, msg)
            servo_data = res.pulse 
            angle = 500 + int(1000 * (-90 + res.rpy[-1]) / 240)

            set_servo_position(self.joints_pub, 1.0, ((1, servo_data[0]), (2, servo_data[1]), (3, servo_data[2]), (4, servo_data[3])))
            time.sleep(1)
            set_servo_position(self.joints_pub, 1.0, ((5, angle),))
            time.sleep(1)
            set_servo_position(self.joints_pub, 1.0, ((10, 200),))
            time.sleep(1)
    
            self.go_home()
            time.sleep(1)

            self.start_place = False
            self.start_count = False
            break

    def image_callback(self, ros_image):
        # 将ros格式图像转换为opencv格式(convert the image from ros format to opencv format)
        cv_image = self.bridge.imgmsg_to_cv2(ros_image, "rgb8")
        rgb_image = np.array(cv_image, dtype=np.uint8)
        result_image = np.copy(rgb_image)

        # 绘制识别区域(draw recognition region)
        # if self.imgpts is not None:
            # cv2.drawContours(result_image, [self.imgpts], -1, (255, 255, 0), 2, cv2.LINE_AA) # 绘制矩形(draw rectangle)
            # for p in self.imgpts:
                # cv2.circle(result_image, tuple(p), 8, (255, 0, 0), -1)
            # pass

        if self.center_imgpts is not None:

            cv2.line(result_image, (self.center_imgpts[0]-10, self.center_imgpts[1]), (self.center_imgpts[0]+10, self.center_imgpts[1]), (255, 255, 0), 2)
            cv2.line(result_image, (self.center_imgpts[0], self.center_imgpts[1]-10), (self.center_imgpts[0], self.center_imgpts[1]+10), (255, 255, 0), 2)
        # 生成识别区域的遮罩(generate the mask of recognition region)
        target_list = []
        index = 0
        if self.roi is not None and not self.start_count:
            roi_area_mask = np.zeros(shape=(ros_image.height, ros_image.width, 1), dtype=np.uint8)
            roi_area_mask = cv2.drawContours(roi_area_mask, [self.imgpts], -1, 255, cv2.FILLED)
            rgb_image = cv2.bitwise_and(rgb_image, rgb_image, mask=roi_area_mask)  # 和原图做遮罩，保留需要识别的区域(create a mask based on the original image to retain the region to be recognized)
            roi_img = rgb_image[self.roi[0]:self.roi[1], self.roi[2]:self.roi[3]]
            roi_img = self.get_top_surface(roi_img)
            image_lab = cv2.cvtColor(roi_img, cv2.COLOR_RGB2LAB) # 转换到 LAB 空间(convert to LAB space)
            img_h, img_w = rgb_image.shape[:2]

            for i in ['red', 'green', 'blue']:
                mask = cv2.inRange(image_lab, tuple(self.lab_data['color_range_list'][i]['min']), tuple(self.lab_data['color_range_list'][i]['max']))  # 二值化

                # 平滑边缘，去除小块，合并靠近的块(Smooth edges, remove small blocks, and merge adjacent blocks)
                eroded = cv2.erode(mask, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)))
                dilated = cv2.dilate(eroded, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)))
                contours = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)[-2]  # 找出所有轮廓(find all contours)
                contours_area = map(lambda c: (math.fabs(cv2.contourArea(c)), c), contours)  # 计算轮廓面积(calculate contour area)
                contours = map(lambda a_c: a_c[1], filter(lambda a: self.min_area <= a[0] <= self.max_area, contours_area))
                for c in contours:
                    rect = cv2.minAreaRect(c)  # 获取最小外接矩形(obtain the minimum bounding rectangle)
                    (center_x, center_y), _ = cv2.minEnclosingCircle(c)
                    center_x, center_y = self.roi[2] + center_x, self.roi[0] + center_y
                    cv2.circle(result_image, (int(center_x), int(center_y)), 8, (0, 0, 0), -1)
                    corners = list(map(lambda p: (self.roi[2] + p[0], self.roi[0] + p[1]), cv2.boxPoints(rect))) # 获取最小外接矩形的四个角点, 转换回原始图的坐标(obtain the four corner points of the minimum rectangle and convert to the coordinates of the original image)
                    cv2.drawContours(result_image, [np.intp(corners)], -1, (0, 255, 255), 2, cv2.LINE_AA)  # 绘制矩形轮廓(draw rectangle contour)
                    index += 1 # 序号递增(incremental numbering)
                    angle = int(round(rect[2]))  # 矩形角度(rectangle angle)
                    target_list.append([i, index, (center_x, center_y), angle])

            tags = self.at_detector.detect(cv2.cvtColor(rgb_image, cv2.COLOR_RGB2GRAY), True, (self.K[0,0], self.K[1,1], self.K[0,2], self.K[1,2]), self.tag_size)
            if len(tags) > 0:
                common.draw_tags(result_image, tags, corners_color=(0, 0, 255), center_color=(0, 255, 0))
                for tag in tags:
                    if 'tag_%d'%tag.tag_id in self.target_labels:
                        index += 1
                        target_list.append(['tag_%d'%tag.tag_id, index, tag])

        if self.enable_sortting :
            for target in target_list:
                if self.target_labels[target[0]]:
                    if not 'tag' in target[0]:
                        # 颜色处理(color processing)
                        if self.target is not None :
                            if self.target[0] == target[0]:
                                self.count += 1
                            else:
                                self.count = 0
                        self.target = target
                        # self.get_logger().info("target[2] "+ str(target[2]))
                        if self.count > 10:
                            self.count = 0
                            projection_matrix = np.row_stack((np.column_stack((self.extristric[1], self.extristric[0])), np.array([[0, 0, 0, 1]])))
                            world_pose = common.pixels_to_world([target[2]], self.K, projection_matrix)[0]
                            world_pose[1] = -world_pose[1]
                            world_pose[2] = 0.030
                            world_pose = np.matmul(self.white_area_center, common.xyz_euler_to_mat(world_pose, (0, 0, 0)))
                            world_pose[2] = 0.030
                            pose_t, _ = common.mat_to_xyz_euler(world_pose)
                            pose_t[2] = 0.010
                            config_data = common.get_yaml_data("/home/jetson/repos/RoboArm/ros2_ws/src/app/config/positions.yaml")
                            offset = tuple(config_data['color_sortting']['offset'])
                            scale = tuple(config_data['color_sortting']['scale'])
                            # self.get_logger().info("pose_t: "+ str(pose_t))
                            for i in range(3):

                                pose_t[i] = pose_t[i] + offset[i]
                                pose_t[i] = pose_t[i] * scale[i]
                            # 高度补偿，越远重力下垂越严重补偿下垂高度(height compensation, the farther the gravity droop is more severe, compensating for droop height)
                            pose_t[2] += (math.sqrt(pose_t[1] ** 2 + pose_t[0] ** 2) - 0.15) / 0.20 * 0.025
                            self.target = target
                            self.start_count = True
                            self.start_move = True


                            threading.Thread(target=self.pick, args=(pose_t, target[3]), daemon=True).start()

                    else:
                        # 标签处理(tag processing)
                        if self.target is not None and not self.start_move:
                            if self.target[0] == target[0]:
                                self.count += 1
                            else:
                                self.count = 0

                        self.target = target
                        tag = target[-1]
                        pose_end = np.matmul(self.hand2cam_tf_matrix,common.xyz_rot_to_mat(tag.pose_t, tag.pose_R)) # 转换的末端相对坐标(relative coordinate of the converted end-effector)
                        pose_world = np.matmul(self.endpoint, pose_end) # 转换到机械臂世界坐标(convert to the world coordinate of the robotic arm)
                        pose_world_T, pose_world_euler = common.mat_to_xyz_euler(pose_world, degrees=True)
                        cv2.putText(result_image, "{:.3f} {:.3f}".format(pose_world_T[0], pose_world_T[1]), (int(tag.center[0]-50), int(tag.center[1]+22)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
                        if self.count > 5 :
                            self.count = 0
                            pose_world_T[2] = 0.015
                            for i in range(3):
                                config_data = common.get_yaml_data("/home/jetson/repos/RoboArm/ros2_ws/src/app/config/positions.yaml")
                                offset = tuple(config_data['tag_sortting']['offset'])
                                scale = tuple(config_data['tag_sortting']['scale'])
                                for i in range(3):
                                    pose_world_T[i] = pose_world_T[i] + offset[i]
                                    pose_world_T[i] = pose_world_T[i] * scale[i]
                            r = pose_world_euler[2] % 90
                            r = r - 90 if r > 45 else (r + 90 if r < -45 else r)
                            pose_world_euler[2] = -r
                            self.status = 1
                            self.stop_thread = False
                            self.start_count = True
                            self.start_move = True
                            threading.Thread(target=self.pick, args=(pose_world_T, pose_world_euler[2]), daemon=True).start()
                    break

        self.result_publisher.publish(self.bridge.cv2_to_imgmsg(result_image, "rgb8"))
        if result_image is not None and self.get_parameter('display').value:
            result_image = cv2.cvtColor(result_image, cv2.COLOR_BGR2RGB)
            cv2.imshow('result_image', result_image)
            key = cv2.waitKey(1)

def main():
    node = ObjectSorttingNode('object_sortting')
    executor = MultiThreadedExecutor()
    executor.add_node(node)
    executor.spin()
    node.destroy_node()
    rclpy.shutdown()

if __name__ == "__main__":
    main()



