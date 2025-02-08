#!/usr/bin/env python3
# encoding: utf-8
import rclpy
import time
from rclpy.node import Node
from ros_robot_controller import ros_robot_controller_node

class CheckServoConnection(Node):
    
    def __init__(self, name='check_servo_connection'):
        super().__init__(name)
        self.RRC = ros_robot_controller_node.RosRobotController('ros_robot_controller')
        self.servo_ids_to_check = [1, 2, 3, 4, 5, 10]
        self.check_servo_connections()


    def check_servo_connections(self):
        for i in self.servo_ids_to_check:
            try:
                servo_id = self.RRC.board.bus_servo_read_id(i)
                if servo_id is not None:
                    self.get_logger().info(f"Detected serial servo ID: {servo_id[0]}")
                else:
                    msg = f"Could not detect serial servo ID: {i} on the bus!!! Node is shutting down."
                    self.handle_servo_not_found(msg)
            except Exception as e:
                self.get_logger().error(f"Error checking servo ID {i}: {e}")

 
    def handle_servo_not_found(self, msg):
        while True:
            try:
                self.RRC.board.set_buzzer(2000, 0.05, 0.01, 1)
                time.sleep(0.2)
                self.RRC.get_logger().error(msg)  
            except Exception as e:
                self.get_logger().error(f"Error handling servo not found: {e}")
                break  
 
        self.request_shutdown()
 
def main(args=None):
    rclpy.init(args=args)
    node = CheckServoConnection()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()
 
if __name__ == '__main__':
    main()

    
