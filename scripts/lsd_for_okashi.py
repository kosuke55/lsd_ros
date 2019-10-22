#!/usr/bin/env python

import cv2
import rospy
import sys
import message_filters
from sensor_msgs.msg import Image
from lsd_ros.msg import Lines, Line
import numpy as np
from cv_bridge import CvBridge


class LSD():
    def __init__(self):
        self.INPUT_IMAGE = rospy.get_param(
            '~input_image', "/head_mount_kinect/hd/image_color_rect_desktop")
        self.INPUT_MASK = rospy.get_param(
            '~input_mask', "/erode_mask_image_table/output")
        self.USE_MASK = rospy.get_param(
            '~use_mask', True)
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
        if(self.USE_MASK):
            rospy.loginfo("use mask")
            sub_img = message_filters.Subscriber(
                self.INPUT_IMAGE, Image,
                queue_size=1)
            sub_mask = message_filters.Subscriber(
                self.INPUT_MASK, Image,
                queue_size=1)
            self.subs = [sub_img, sub_mask]

            sync = message_filters.ApproximateTimeSynchronizer(
                fs=self.subs, queue_size=100, slop=1)
            sync.registerCallback(self.callback_with_mask)

        else:
            rospy.loginfo("not use mask")
            self.image_sub = rospy.Subscriber(self.INPUT_IMAGE,
                                              Image,
                                              self.callback,
                                              queue_size=1)

    def callback_with_mask(self, msg, mask_msg):
        rospy.loginfo("lsd called")

        mask = self.bridge.imgmsg_to_cv2(mask_msg, "passthrough")
        print(mask.shape)

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
            if ((x2-x1)**2 + (y2-y1)**2 > 1000 and
                    np.abs(dot) > 0.9 and
                    mask[y1, x1] == 255 and
                    mask[y2, x2] == 255):
                img = cv2.line(img, (x1, y1), (x2, y2), (0, 0, 255), 2)
                selected_lines.append([x1, y1, x2, y2])
        selected_lines = np.array(selected_lines)
        max_idx = (selected_lines[:, 0] + selected_lines[:, 3]).argmax()
        min_idx = (selected_lines[:, 0] + selected_lines[:, 3]).argmin()
        img = cv2.line(img,
                       (selected_lines[min_idx, 0],
                        selected_lines[min_idx, 1]),
                       (selected_lines[min_idx, 2],
                        selected_lines[min_idx, 3]),
                       (0, 255, 0), 2)
        img = cv2.line(img,
                       (selected_lines[max_idx, 0],
                        selected_lines[max_idx, 1]),
                       (selected_lines[max_idx, 2],
                        selected_lines[max_idx, 3]),
                       (0, 255, 0), 2)
        selected_lines = np.delete(selected_lines, [min_idx, max_idx], 0)
        for selected_line in selected_lines:
            line_msg.x1 = selected_line[0]
            line_msg.x2 = selected_line[1]
            line_msg.y1 = selected_line[2]
            line_msg.y2 = selected_line[3]
            lines_msg.lines.append(line_msg)
        msg_out = self.bridge.cv2_to_imgmsg(img, "bgr8")
        msg_out.header = msg.header
        self.pub_img.publish(msg_out)
        self.pub_lines.publish(lines_msg)

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
                img = cv2.line(img, (x1, y1), (x2, y2), (0, 0, 255), 2)
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

