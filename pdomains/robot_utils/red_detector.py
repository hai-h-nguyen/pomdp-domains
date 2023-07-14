# Python code for Multiple Color Detection


import numpy as np
import cv2


def is_detect_red(camera, num_frames=5):

    # Start a while loop
    for _ in range(num_frames):
        
        # Reading the video from the
        # webcam in image frames
        ret, imageFrame = camera.read()

        # Convert the imageFrame in
        # BGR(RGB color space) to
        # HSV(hue-saturation-value)
        # color space
        if ret:
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
                    return True

    return False

if __name__ == "__main__":
    # Capturing video through webcam
    camera = cv2.VideoCapture(0)
    print(is_detect_red(camera))