#!/usr/bin/env python
# license removed for brevity
import rospy
from std_msgs.msg import String
import cv2
import numpy as np

# Capturing video through webcam
camera = cv2.VideoCapture(0)

num_frames = 10

def talker():
    pub = rospy.Publisher('detector', String, queue_size=10)
    rospy.init_node('red_detector', anonymous=True)
    rate = rospy.Rate(20) # 10hz
    while not rospy.is_shutdown():
        
        num_positives = 0
        for i in range(num_frames):
            
            # Reading the video from the
            # webcam in image frames
            _, imageFrame = camera.read()

            # Convert the imageFrame in
            # BGR(RGB color space) to
            # HSV(hue-saturation-value)
            # color space
            hsvFrame = cv2.cvtColor(imageFrame, cv2.COLOR_BGR2HSV)

            # Set range for red color and
            # define mask
            red_lower = np.array([0, 50, 111], np.uint8)
            red_upper = np.array([10, 255, 255], np.uint8)
            red_mask = cv2.inRange(hsvFrame, red_lower, red_upper)

            # Morphological Transform, Dilation
            # for each color and bitwise_and operator
            # between imageFrame and mask determines
            # to detect only that particular color
            kernel = np.ones((5, 5), "uint8")
            
            # For red color
            red_mask = cv2.dilate(red_mask, kernel)
            res_red = cv2.bitwise_and(imageFrame, imageFrame,
                                    mask = red_mask)
            
            # Creating contour to track red color
            contours, hierarchy = cv2.findContours(red_mask,
                                                cv2.RETR_TREE,
                                                cv2.CHAIN_APPROX_SIMPLE)
            
            for pic, contour in enumerate(contours):
                area = cv2.contourArea(contour)
                if(area > 300):
                    num_positives += 1

        if num_positives >= int(0.8 * num_frames):
            detect = True
        else:
            detect = False

        print(num_positives)
        detect_str = f"{detect}"
        if not detect:
            rospy.loginfo(detect_str)
        pub.publish(detect_str)
        rate.sleep()

if __name__ == '__main__':
    try:
        talker()
    except rospy.ROSInterruptException:
        pass