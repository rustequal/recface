import logging

WINDOW_NAME = 'RecFace'
FD_TYPE = 'trt_ssd' # 'cv2_haar', 'dlib_frontal', 'dlib_human', 'trt_mtcnn', 'trt_ssd', 'cv2_ssd'
FD_MINSIZE = 200
FD_X_SCALE = 4
FD_Y_SCALE = 4
FD_SQUARE = True
MODEL_DATA = 'data/95_dlib_continued_MobileFaceNetV2_ArcFace_planH_batch256_model.uff'
#MODEL_DATA = 'data/95_dlib_continued_MobileFaceNetV2_ArcFace_planH_batch256_model.pb'
MODEL_IMG_SIZE = 112
MODEL_INPUT_NAME = 'x'
MODEL_OUTPUT_NAME = 'Identity'
USE_GPU = True
TRT_MAX_WORKSPACE_SIZE = 536870912
TRT_DIR = '/usr/lib/python3.6/dist-packages'
CLF_PKL = 'data/classifier.pkl'
CLF_MEDIAN = 0.6
DRAW_FONT = '/usr/share/fonts/truetype/freefont/FreeSansBold.ttf'
DISPLAY = ':0'
IMAGE_WIDTH, IMAGE_HEIGHT = 1280, 720
CAMERA_FPS = 30
TITLER_TEXT = 'Welcome, {}!'
TITLER_TOP, TITLER_BOTTOM, TITLER_LEFT, TITLER_RIGHT = 600, 700, 20, 1260
TITLER_FRAMES, TITLER_TOTAL, TITLER_FONT_SIZE = 50, 80, 64
FPS_SHOW = True
FPS_DEPTH = 8
GPIO_PIN = 24  # BCM pin 24, BOARD pin 18
LOG_FILE = '/var/log/recface.log'
LOG_LEVEL = logging.INFO


if '_ssd' in FD_TYPE:
    FD_X_SCALE = IMAGE_WIDTH / 300
    FD_Y_SCALE = IMAGE_HEIGHT / 300