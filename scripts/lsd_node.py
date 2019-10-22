#!/usr/bin/env python

import cv2
import rospy
import sys
from sensor_msgs.msg import Image
from lsd_ros.msg import Lines, Line
import numpy as np
from cv_bridge import CvBridge


class LSD():
    def __init__(self):
        self.INPUT_IMAGE = rospy.get_param(
            '~input_image', "/head_mount_kinect/hd/image_color_rect_desktop")
        # self.INPUT_IMAGE = rospy.get_param(
        #     '~input_image', "/image_publisher_1571650592823369089/image_raw")
        self.bridge = CvBridge()
        self.pub_img = rospy.Publisher("/lsd_detected_line",
                                       Image,
                                       queue_size=1)
        self.pub_lines = rospy.Publisher("/lines",
                                         Lines,
                                         queue_size=1)
        self.subscribe()
        self.lsd = cv2.createLineSegmentDetector()
        self.v0 = np.array([0, 1.0])

    def subscribe(self):
        self.image_sub = rospy.Subscriber(self.INPUT_IMAGE,
                                          Image,
                                          self.callback,
                                          queue_size=1)

    def callback(self, msg):
        rospy.loginfo("lsd called")
        img = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (5, 5), 5)
        lines = self.lsd.detect(gray)[0]
        selected_lines = []
        lines_msg = Lines()
        line_msg = Line()
        lines_msg.header = msg.header
        for line in lines:
            x1, y1, x2, y2 = map(int, line[0])
            v = np.array([x2-x1, y2 - y1])
            ve = v / np.linalg.norm(v)
            dot = np.dot(self.v0, ve)
            if ((x2-x1)**2 + (y2-y1)**2 > 1000 and np.abs(dot) > 0.9):
                img = cv2.line(img, (x1, y1), (x2, y2), (0, 0, 255), 3)
                selected_lines.append([x1, y1, x2, y2])
                line_msg.x1 = x1
                line_msg.x2 = x2
                line_msg.y1 = y1
                line_msg.y2 = y2
        lines_msg.lines.append(line_msg)
        msg_out = self.bridge.cv2_to_imgmsg(img, "bgr8")
        msg_out.header = msg.header
        self.pub_img.publish(msg_out)
        self.pub_lines.publish(lines_msg)


def main(args):
    rospy.init_node("lsd", anonymous=False)
    lsd_instance = LSD()
    rospy.spin()


if __name__ == '__main__':
    main(sys.argv)

