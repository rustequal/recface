#!/usr/bin/env python

import os
import sys
import time
import cv2
import numpy as np
import pickle
import signal
import logging
import RPi.GPIO as GPIO
from PIL import Image, ImageDraw, ImageFont
import recface_config as config

# Loading only the libraries that are needed, in order to save memory
IS_TRT = '.uff' in config.MODEL_DATA
if IS_TRT or config.FD_TYPE == 'trt_ssd':
    sys.path.append(config.TRT_DIR)
    import pycuda.driver as cuda
    import pycuda.autoinit
    import tensorrt as trt
    TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
    trt.init_libnvinfer_plugins(TRT_LOGGER, '')

if 'dlib' in config.FD_TYPE:
    import dlib
elif config.FD_TYPE == 'trt_mtcnn':
    from mtcnn.mtcnn import TrtMtcnn


# Loading font
def load_font(path, size):
    try:
        font = ImageFont.truetype(path, size)
    except Exception as argument:
        logging.error('Please check the parameter "DRAW_FONT" in config file!')
        exception_exit(argument)
    return font


# Cleanup before exiting the script
def cleanup():
    logging.info('Stopping Face Recognition service')
    GPIO.cleanup()
    cv2.destroyAllWindows()


# Exiting the script by error
def exception_exit(argument, stderr=False):
    msg = '. '.join([line for line in str(argument).split('\n') if line])
    if stderr:
        sys.stderr.write(msg + '\n')
    else:
        logging.error(msg)
    cleanup()
    sys.exit(1)


# Process signal handling
def signal_handler(signum, frame):
    cleanup()
    sys.exit(0)


# Class for creating title on a video
class Titler:
    def __init__(self, helper):
        self.active, self.hidden, self.helper = {}, {}, helper
        self.top, self.bottom = config.TITLER_TOP, config.TITLER_BOTTOM
        self.left, self.right = config.TITLER_LEFT, config.TITLER_RIGHT
        self.shape = (self.right - self.left, self.bottom - self.top)
        self.text_pos = (self.shape[0] // 2, self.shape[1] * 0.72)
        self.font = load_font(config.DRAW_FONT, config.TITLER_FONT_SIZE)
        self.frames, self.total = config.TITLER_FRAMES, config.TITLER_TOTAL
        self.curr_title, self.curr_img = None, None
    
    def create_img_title(self, title):
        img = Image.new('RGB', self.shape)
        draw = ImageDraw.Draw(img)
        draw.text(self.text_pos, config.TITLER_TEXT.format(title),
                  fill=(255, 255, 255), anchor='ms', font=self.font,
                  align='center')
        self.curr_img = np.array(img, dtype=np.uint16)
        self.curr_title = title

    def draw_title(self, img, title, frame):
        img_title = img[self.top:self.bottom,
                        self.left:self.right] // 2
        if frame > 0 and frame < self.frames - 1:
            if self.curr_title != title:
                self.create_img_title(title)
            img_title = np.clip(img_title + self.curr_img, None, 255)
        img[self.top:self.bottom, self.left:self.right] = img_title
    
    def draw_frame(self, img):
        self.helper(self.active)
        if self.active:
            key = list(self.active.keys())[0]
            title = self.active[key]
            self.draw_title(img, title['person'], title['frame'])
            if title['frame'] < self.frames:
                title['frame'] += 1
            else:
                self.hidden[key] = self.active.pop(key)

        hidden_keys = list(self.hidden.keys())
        for key in hidden_keys:
            title = self.hidden[key]
            title['frame'] += 1
            if title['frame'] >= self.total:
                self.hidden.pop(key)
    
    def show_title(self, index, person):
        if index not in self.active and index not in self.hidden:
            self.active[index] = {'person': person, 'frame': 0}
            logging.debug('The person "{}" is recognized'.format(person))


# Classifier loader class
class Classifier:
    def __init__(self, path):
        self.path, self.frame = path, 0
        logging.info('Loading classifier from "{}"'.format(self.path))
        self.load()
    
    def get_attrs(self):
        return (os.path.getsize(self.path), os.path.getmtime(self.path))
    
    def load(self):
        try:
            with open(self.path, 'rb') as f:
                (self.le, self.clf) = pickle.load(f)
            self.attrs = self.get_attrs()
            logging.info('The classifier was loaded successfully for '
                         '{} classes'.format(len(self.le.classes_)))
        except Exception as argument:
            exception_exit(argument)

    def reload(self):
        self.frame += 1
        if self.frame % 1800 == 0:
            self.frame = 0
            if self.attrs != self.get_attrs():
                logging.info('Reloading classifier from "{}"'
                             .format(self.path))
                self.load()


# FPS counter class
class FPSCounter:
    def __init__(self):
        self.delta = np.zeros(config.FPS_DEPTH)
        self.start = time.time()
    
    def update(self, img):
        self.delta = np.roll(self.delta, 1)
        end_time = time.time()
        self.delta[0] = end_time - self.start
        self.start = end_time
        delta2 = np.sum(np.reshape(self.delta, (len(self.delta)//2, 2)),
                        axis=-1)
        delta2 = np.fmax(delta2, 2 / config.CAMERA_FPS)
        if config.FPS_SHOW:
            cv2.putText(img, f"FPS: {np.mean(2 / delta2):0.1f}",
                        (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    

# Image coordinate transformations
def _rect_to_css(rect):
    return rect.top(), rect.right(), rect.bottom(), rect.left()


# Trimming bounding box with image borders
def _trim_css_to_bounds(css, image_shape):
    return (max(css[0], 0), min(css[1], image_shape[1]),
            min(css[2], image_shape[0]), max(css[3], 0))


# Inference of the face detector CV2 haarcascade
def fd_haar_inference(fd, img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = fd['model'].detectMultiScale(img, 1.2, 6, minSize=fd['min_size'])
    return [[y, x + w, y + h, x] for (x, y, w, h) in faces]


# Checking the size of a specific face
def is_face_min_size(face, min_size):
    return face[2] - face[0] > min_size or face[1] - face[3] > min_size


# Inference of the face detector Dlib human 
def fd_dlib_human_inference(fd, img):
    faces = fd['model'](cv2.cvtColor(img, cv2.COLOR_BGR2RGB), 1)
    faces = [_trim_css_to_bounds(_rect_to_css(face.rect), img.shape) \
            for face in faces]
    return [face for face in faces if is_face_min_size(face, fd['min_size'])]


# Inference of the face detector Dlib frontal 
def fd_dlib_frontal_inference(fd, img):
    faces = fd['model'](cv2.cvtColor(img, cv2.COLOR_BGR2RGB), 1)
    faces = [_trim_css_to_bounds(_rect_to_css(face), img.shape) \
            for face in faces]
    return [face for face in faces if is_face_min_size(face, fd['min_size'])]


# Inference of the face detector MTCNN 
def fd_mtcnn_inference(fd, img):
    faces, _ = fd['model'].detect(img, minsize=fd['min_size'])
    return [[y1, x2, y2, x1] \
            for (x1, y1, x2, y2, p) in faces if p > 0.5]


# Inference of the face detector ResnetSSD TensorRT
def fd_trt_ssd_inference(fd, img):
    size = fd['size']
    blob = cv2.dnn.blobFromImage(img, 1.0, (size, size), (104.0, 177.0, 123.0))
    np.copyto(fd['inputs'][0].host, blob.ravel())
    [faces] = trt_do_inference(
            fd['context'], bindings=fd['bindings'], inputs=fd['inputs'],
            outputs=fd['outputs'], stream=fd['stream'])
    faces = np.reshape(faces, (-1, 7))
    faces = [[y1 * size, x2 * size, y2 * size, x1 * size] \
            for (_, _, p, x1, y1, x2, y2) in faces if p > 0.5]
    return [face for face in faces if is_face_min_size(face, fd['min_size'])]


# Inference of the face detector ResnetSSD CV2
def fd_cv2_ssd_inference(fd, img):
    size = fd['size']
    blob = cv2.dnn.blobFromImage(img, 1.0, (size, size), (104.0, 177.0, 123.0))
    fd['model'].setInput(blob)
    faces = fd['model'].forward().squeeze()
    faces = [[y1 * size, x2 * size, y2 * size, x1 * size] \
            for (_, _, p, x1, y1, x2, y2) in faces if p > 0.5]
    return [face for face in faces if is_face_min_size(face, fd['min_size'])]


# Getting the coordinates of the faces in the image
def get_face_locations(fd, img):
    try:
        faces = fd['inference'](fd, img)
    except Exception as argument:
        logging.error('Please check the parameter "FD_TYPE" '
                      'in config file!')
        exception_exit(argument)
    return [[top, right, bottom, left, 0, None, ''] \
            for (top, right, bottom, left) in faces]


# Restoring the size of the bounding box with a face
def scale_box(top, right, bottom, left):
    top *= config.FD_Y_SCALE
    right *= config.FD_X_SCALE
    bottom *= config.FD_Y_SCALE
    left *= config.FD_X_SCALE

    if config.FD_SQUARE:
        height, width = bottom - top, right - left
        margin = abs(height - width) / 4
        if height > width:
            top += margin
            right += margin
            bottom -= margin
            left -= margin
        else:
            top -= margin
            right -= margin
            bottom += margin
            left += margin
            
    return (int(top), int(right), int(bottom), int(left))


# L2 normalization
def l2_norm(x, axis=1):
    norm = np.linalg.norm(x, axis=axis, keepdims=True)
    return x / norm


# Changing the lock state
def gpio_helper(active):
    value = GPIO.HIGH if active else GPIO.LOW
    GPIO.output(config.GPIO_PIN, value)
        

# A small class for organizing variables
class HostDeviceMem(object):
    def __init__(self, host_mem, device_mem):
        self.host = host_mem
        self.device = device_mem

    def __str__(self):
        return "Host:\n" + str(self.host) + "\nDevice:\n" + str(self.device)

    def __repr__(self):
        return self.__str__()


# Allocating GPU memory for TensorRT
def trt_allocate_buffers(engine):
    stream, inputs, outputs, bindings = cuda.Stream(), [], [], []
    for binding in engine:
        size = trt.volume(engine.get_binding_shape(binding)) \
                * engine.max_batch_size
        dtype = trt.nptype(engine.get_binding_dtype(binding))
        host_mem = cuda.pagelocked_empty(size, dtype)
        device_mem = cuda.mem_alloc(host_mem.nbytes)
        bindings.append(int(device_mem))
        if engine.binding_is_input(binding):
            inputs.append(HostDeviceMem(host_mem, device_mem))
        else:
            outputs.append(HostDeviceMem(host_mem, device_mem))
    return inputs, outputs, bindings, stream


# TensorRT inference function
def trt_do_inference(context, bindings, inputs, outputs, stream, batch_size=1):
    [cuda.memcpy_htod_async(inp.device, inp.host, stream) for inp in inputs]
    context.execute_async(batch_size=batch_size, bindings=bindings,
                          stream_handle=stream.handle)
    [cuda.memcpy_dtoh_async(out.host, out.device, stream) for out in outputs]
    stream.synchronize()
    return [out.host for out in outputs]


# Building the UFF TensorRT engine
def trt_uff_build_engine(model_file):
    with trt.Builder(TRT_LOGGER) as builder, \
            builder.create_network() as network, trt.UffParser() as parser:
        builder.max_workspace_size = config.TRT_MAX_WORKSPACE_SIZE
        size = config.MODEL_IMG_SIZE
        parser.register_input(config.MODEL_INPUT_NAME, (3, size, size))
        parser.register_output(config.MODEL_OUTPUT_NAME)
        parser.parse(model_file, network)
        return builder.build_cuda_engine(network)


# Building the Caffe TensorRT engine
def trt_caffe_build_engine(model_file, proto_file):
    with trt.Builder(TRT_LOGGER) as builder, \
            builder.create_network() as network, trt.CaffeParser() as parser:
        builder.max_workspace_size = config.TRT_MAX_WORKSPACE_SIZE
        parser.parse(deploy=proto_file, model=model_file, network=network,
                     dtype=trt.float32)
        last_layer = network.get_layer(network.num_layers - 1)
        network.mark_output(last_layer.get_output(0))
        return builder.build_cuda_engine(network)


# The main working function
def work_loop(cap, fd, model, classifier):

    def cv2_inference(data):
        data = np.transpose(data, (2, 0, 1))[None, ...]
        model.setInput(data)
        return model.forward()
    
    def trt_inference(data):
        data = np.transpose(data, (2, 0, 1)).ravel()
        np.copyto(inputs[0].host, data)
        [output] = trt_do_inference(
                context, bindings=bindings, inputs=inputs,
                outputs=outputs, stream=stream)
        return output[None, ...]
    
    def main_loop(inference):
        logging.info('Running the main loop ...')
        face_locations, face_img, fps = [], None, FPSCounter()
        titler = Titler(gpio_helper)
        fx_scale = 1 / config.FD_X_SCALE
        fy_scale = 1 / config.FD_Y_SCALE
        font = load_font(config.DRAW_FONT, 24)
        process_this_frame = True
        while True:
            _, img = cap.read()
            if process_this_frame:
                img_fd = cv2.resize(img, (0, 0), fx=fx_scale, fy=fy_scale)
                new_face_locations = get_face_locations(fd, img_fd)
                new_img = img.copy()
            else:
                face_locations = new_face_locations
                face_img = new_img

            for i, (top, right, bottom, left, confidence, index, person) in \
                    enumerate(face_locations):
                top, right, bottom, left = scale_box(top, right, bottom, left)
                if not process_this_frame:
                    fr_img = face_img[max(0, top):max(0, bottom),
                                      max(0, left):max(0, right)]
                    fr_img = cv2.resize(fr_img, (config.MODEL_IMG_SIZE,
                                                 config.MODEL_IMG_SIZE),
                                        interpolation=cv2.INTER_CUBIC)
                    fr_img = (fr_img - 127.5) / 127.5
                    output = inference(fr_img)
                    emb = l2_norm(output)
                    predictions = classifier.clf.predict_proba(emb).ravel()
                    index = np.argmax(predictions)
                    person = classifier.le.inverse_transform(
                            index.reshape(-1))[0]
                    confidence = predictions[index]
                    face_locations[i][4] = confidence
                    face_locations[i][5] = index
                    face_locations[i][6] = person

                color = (0, 255, 0) if confidence > config.CLF_MEDIAN \
                        else (0, 0, 255)
                cv2.rectangle(img, (left, top), (right, bottom), color, 2)
                if confidence > config.CLF_MEDIAN:
                    top_l = np.clip(bottom, 0, img.shape[0])
                    right_l = np.clip(right + 1, 0, img.shape[1])
                    bottom_l = np.clip(bottom + 35, 0, img.shape[0])
                    left_l = np.clip(left - 1, 0, img.shape[1])
                    label_img = Image.new(
                            "RGB", [right_l - left_l, bottom_l - top_l], color)
                    if min(label_img.size) > 0:
                        draw = ImageDraw.Draw(label_img)
                        draw.text((10, 5),
                                person + ' ' + str(round(confidence, 4)),
                                font=font, fill=(255, 255, 255))
                        img[top_l:bottom_l, left_l:right_l] = cv2.cvtColor(
                                np.array(label_img), cv2.COLOR_RGB2BGR)
                    titler.show_title(index, person)

            titler.draw_frame(img)
            classifier.reload()
            fps.update(img)
            cv2.imshow(config.WINDOW_NAME, img)
            key = cv2.waitKey(1)
            if key == 27:
                break
            process_this_frame = not process_this_frame


    if IS_TRT:
        inputs, outputs, bindings, stream = trt_allocate_buffers(model)
        with model.create_execution_context() as context:
            main_loop(trt_inference)
    else:
        main_loop(cv2_inference)
    

# Loading the Face Detector
def load_face_detector(fd_type):
    
    def log_engine(engine, version, device):
        info = ('Using {} engine {} ({}) for face detection'
                .format(engine, version, device))
        logging.info(info)
    
    logging.info('Loading "{}" face detector model'.format(fd_type))
    fd_ext = {}
    min_scale = min(config.FD_X_SCALE, config.FD_Y_SCALE)
    min_size = config.FD_MINSIZE // min_scale
    if fd_type == 'cv2_haar':
        log_engine('OpenCV', cv2.__version__, 'CPU')
        path = 'data/haarcascade_frontalface_default.xml'
        model = cv2.CascadeClassifier(path)
        inference = fd_haar_inference
        min_size = (min_size,) * 2
    elif fd_type == 'dlib_human':
        device = 'GPU' if dlib.DLIB_USE_CUDA else 'CPU'
        log_engine('Dlib', dlib.__version__, device)
        path = 'data/mmod_human_face_detector.dat'
        model = dlib.cnn_face_detection_model_v1(path)
        inference = fd_dlib_human_inference
    elif fd_type == 'dlib_frontal':
        log_engine('Dlib', dlib.__version__, 'CPU')
        model = dlib.get_frontal_face_detector()
        inference = fd_dlib_frontal_inference
    elif fd_type == 'trt_mtcnn':
        log_engine('TensorRT', 'TrtMtcnn', 'GPU')
        model = TrtMtcnn('mtcnn')
        inference = fd_mtcnn_inference
    elif fd_type == 'trt_ssd':
        log_engine('TensorRT', trt.__version__, 'GPU')
        model_file = 'data/Res10_300x300_SSD_iter_140000.caffemodel'
        proto_file = 'data/Res10_300x300_SSD_iter_140000_trt.prototxt'
        model = trt_caffe_build_engine(model_file, proto_file)
        inputs, outputs, bindings, stream = trt_allocate_buffers(model)
        context = model.create_execution_context()
        fd_ext = {'size': 300, 'inputs': inputs, 'outputs': outputs,
                  'bindings': bindings, 'stream': stream, 'context': context}
        inference = fd_trt_ssd_inference
    elif fd_type == 'cv2_ssd':
        device = 'GPU' if config.USE_GPU else 'CPU'
        log_engine('OpenCV', cv2.__version__, device)
        model_file = 'data/Res10_300x300_SSD_iter_140000.caffemodel'
        proto_file = 'data/Res10_300x300_SSD_iter_140000.prototxt'
        model = cv2.dnn.readNetFromCaffe(proto_file, model_file)
        if config.USE_GPU:
            model.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
            model.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
        fd_ext = {'size': 300}
        inference = fd_cv2_ssd_inference
    else:
        exception_exit('Unknown face detector! Please check the parameter '
                       '"FD_TYPE" in config file!')
    fd = {'model': model, 'inference': inference, 'min_size': min_size}
    fd.update(fd_ext)
    return fd


# Loading a face recognition model
def load_model(path):

    def log_engine(engine, version, device):
        info = ('Using {} engine {} ({}) for face recognition'
                .format(engine, version, device))
        logging.info(info)

    logging.info('Loading model from "{}"'.format(path))
    try:
        if IS_TRT:
            log_engine('TensorRT', trt.__version__, 'GPU')
            model = trt_uff_build_engine(path)
        else:
            device = 'GPU' if config.USE_GPU else 'CPU'
            log_engine('OpenCV', cv2.__version__, device)
            model = cv2.dnn.readNetFromTensorflow(path)
            if config.USE_GPU:
                model.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
                model.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
    except Exception as argument:
        exception_exit(argument)
    return model


# Creating and positioning the GUI window
def open_window(width, height):
    logging.info('Setup the GUI window')
    cv2.namedWindow(config.WINDOW_NAME,
                    cv2.WINDOW_NORMAL | cv2.WINDOW_GUI_NORMAL)
    cv2.resizeWindow(config.WINDOW_NAME, width, height)
    cv2.moveWindow(config.WINDOW_NAME, 0, 0)
    cv2.setWindowProperty(config.WINDOW_NAME, cv2.WND_PROP_FULLSCREEN,
                          cv2.WINDOW_FULLSCREEN)
    cv2.setWindowTitle(config.WINDOW_NAME, 'RecFace')


# Opening the camera device
def open_cam_onboard(width, height, fps):
    logging.info('Opening the camera device')
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
    os.environ["DISPLAY"] = config.DISPLAY
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    GPIO.setmode(GPIO.BCM)
    GPIO.setup(config.GPIO_PIN, GPIO.OUT, initial=GPIO.LOW)
    os.chdir(os.path.dirname(os.path.realpath(__file__)))
    try:
        logging.basicConfig(filename=config.LOG_FILE,
                            format='{asctime} {levelname} {message}',
                            datefmt='%Y-%m-%d %H:%M:%S', style='{',
                            level=config.LOG_LEVEL)
    except Exception as argument:
        exception_exit(argument, stderr=True)

    logging.info('Starting Face Recognition service')
    fd = load_face_detector(config.FD_TYPE)
    model = load_model(config.MODEL_DATA)
    classifier = Classifier(config.CLF_PKL)
    open_window(config.IMAGE_WIDTH, config.IMAGE_HEIGHT)
    cap = open_cam_onboard(config.IMAGE_WIDTH, config.IMAGE_HEIGHT,
                           config.CAMERA_FPS)
    if cap.isOpened():
        logging.info('Camera device was opened successfully')
    else:
        exception_exit('Failed to open camera device!')
    work_loop(cap, fd, model, classifier)
    cleanup()


if __name__ == '__main__':
    main()