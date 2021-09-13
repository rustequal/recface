#!/usr/bin/env python

import os
import cv2

IMAGE_WIDTH, IMAGE_HEIGHT, CAMERA_FPS = 1280, 720, 30
WINDOW_NAME = 'Camera'
DISPLAY = ':0'


# The main working function
def main_loop(cap):
    while True:
        _, img = cap.read()
        cv2.imshow(WINDOW_NAME, img)
        key = cv2.waitKey(1)
        if key == 27:
            break


# Creating and positioning the GUI window
def open_window(width, height):
    cv2.namedWindow(WINDOW_NAME,
                    cv2.WINDOW_NORMAL | cv2.WINDOW_GUI_NORMAL)
    cv2.resizeWindow(WINDOW_NAME, width, height)
    cv2.moveWindow(WINDOW_NAME, 0, 0)
    cv2.setWindowProperty(WINDOW_NAME, cv2.WND_PROP_FULLSCREEN,
                          cv2.WINDOW_FULLSCREEN)
    cv2.setWindowTitle(WINDOW_NAME, 'Simple Camera')


# Opening the camera device
def open_cam_onboard(width, height, fps):
    gst_str = ('nvarguscamerasrc ! '
               'video/x-raw(memory:NVMM), '
               'width=(int){0}, height=(int){1}, '
               'format=(string)NV12, framerate=(fraction){2}/1 ! '
               'nvvidconv flip-method=0 ! '
               'video/x-raw, width=(int){0}, height=(int){1}, '
               'format=(string)BGRx ! '
               'videoconvert ! appsink').format(width, height, fps)
    return cv2.VideoCapture(gst_str, cv2.CAP_GSTREAMER)


# The main function
def main():
    os.environ["DISPLAY"] = DISPLAY
    open_window(IMAGE_WIDTH, IMAGE_HEIGHT)
    cap = open_cam_onboard(IMAGE_WIDTH, IMAGE_HEIGHT, CAMERA_FPS)
    if cap.isOpened():
        main_loop(cap)
    else:
        print('Failed to open camera device!')
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()