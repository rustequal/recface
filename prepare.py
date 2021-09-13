#!/usr/bin/env python

import os
import sys
import glob
import cv2
import pickle
import numpy as np
import prepare_config as config

CURRENT_DIR = os.getcwd()
IS_TRT = '.uff' in config.MODEL_DATA

# Loading only the libraries that are needed, in order to save memory
if len(sys.argv) > 1:
    if sys.argv[1] == 'align':
        if 'dlib' in config.FD_TYPE:
            import dlib
        elif config.FD_TYPE == 'trt_mtcnn':
            from mtcnn.mtcnn import TrtMtcnn
        
    elif sys.argv[1] == 'train':
        from sklearn.preprocessing import LabelEncoder
        from sklearn.svm import SVC
        if IS_TRT:
            sys.path.append(config.TRT_DIR)
            import pycuda.driver as cuda
            import pycuda.autoinit
            import tensorrt as trt
            TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
            trt.init_libnvinfer_plugins(TRT_LOGGER, '')            

    
# Exiting the script by error
def exception_exit(argument, stderr=False):
    msg = '. '.join([line for line in str(argument).split('\n') if line])
    if stderr:
        sys.stderr.write(msg + '\n')
    else:
        print(msg)
    os.chdir(CURRENT_DIR)
    sys.exit(1)


# Image coordinate transformations
def _rect_to_css(rect):
    return rect.top(), rect.right(), rect.bottom(), rect.left()


# Trimming bounding box with image borders
def _trim_css_to_bounds(css, image_shape):
    return (max(css[0], 0), min(css[1], image_shape[1]),
            min(css[2], image_shape[0]), max(css[3], 0))


# Inference of the face detector Dlib human 
def fd_dlib_human_inference(fd, img):
    faces = fd['model'](cv2.cvtColor(img, cv2.COLOR_BGR2RGB), 1)
    faces = [_trim_css_to_bounds(_rect_to_css(face.rect), img.shape) \
            for face in faces]
    return faces


# Inference of the face detector Dlib frontal 
def fd_dlib_frontal_inference(fd, img):
    faces = fd['model'](cv2.cvtColor(img, cv2.COLOR_BGR2RGB), 1)
    faces = [_trim_css_to_bounds(_rect_to_css(face), img.shape) \
            for face in faces]
    return faces


# Inference of the face detector MTCNN 
def fd_mtcnn_inference(fd, img):
    faces, _ = fd['model'].detect(img)
    return [[y1, x2, y2, x1] \
            for (x1, y1, x2, y2, p) in faces if p > 0.5]


# Getting the coordinates of the faces in the image
def get_face_locations(fd, img):
    try:
        faces = fd['inference'](fd, img)
    except Exception as argument:
        print('Please check the parameter "FD_TYPE" in config file!')
        exception_exit(argument)
    return [[top, right, bottom, left, 0, None, ''] \
            for (top, right, bottom, left) in faces]


# Restoring the size of the bounding box with a face
def scale_box(top, right, bottom, left, scale):
    top /= scale
    right /= scale
    bottom /= scale
    left /= scale

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


# Loading the Face Detector
def load_face_detector(fd_type):
    
    def log_engine(engine, version, device):
        info = ('Using {} engine {} ({}) for face detection'
                .format(engine, version, device))
        print(info)

    print('Loading "{}" face detector model'.format(fd_type))
    if fd_type == 'dlib_human':
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
    else:
        exception_exit('Unknown face detector! Please check the parameter '
                       '"FD_TYPE" in config file!\n')
    fd = {'model': model, 'inference': inference}
    return fd


# Loading a face recognition model
def load_model(path):

    def log_engine(engine, version, device):
        info = ('Using {} engine {} ({}) for face recognition'
                .format(engine, version, device))
        print(info)

    ext = ()
    print('Loading model from "{}"'.format(path))
    try:
        if IS_TRT:
            log_engine('TensorRT', trt.__version__, 'GPU')
            model = trt_uff_build_engine(path)
            inputs, outputs, bindings, stream = trt_allocate_buffers(model)
            context = model.create_execution_context()
            ext = (inputs, outputs, bindings, stream, context)
        else:
            device = 'GPU' if config.USE_GPU else 'CPU'
            log_engine('OpenCV', cv2.__version__, device)
            model = cv2.dnn.readNetFromTensorflow(path)
            if config.USE_GPU:
                model.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
                model.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
    except Exception as argument:
        exception_exit(argument)
    return (model, ext)


# Image decoding function
def decode_image(data):
    return cv2.imdecode(data, cv2.IMREAD_COLOR)


# Loading an image
def load_image(img_path):
    img = decode_image(np.fromfile(img_path, dtype=np.uint8))
    if img is None:
        raise Exception('Image loading error:', img_path)
    return img


# Data normalization for a neural network
def norm_binary(data):
    norm_fwd = lambda x: (x - 127.5) / 127.5
    return norm_fwd(data.astype(np.float32))


# Getting a list of labels of a dataset
def get_dataset_labels(path, verbose=True):
    labels = sorted([p for p in os.listdir(path)
                     if os.path.isdir(os.path.join(path, p))])
    if verbose:
        print('\"{}\" labels found in directory \"{}\"'
              .format(len(labels), path))
    return labels


# Function for getting embeddings from the directory with images
def get_dir_embeddings(model, ext, path, label):
    
    def cv2_inference(data):
        data = np.transpose(data, (2, 0, 1))[None, ...]
        model.setInput(data)
        return model.forward()
    
    def trt_inference(data):
        (inputs, outputs, bindings, stream, context) = ext
        data = np.transpose(data, (2, 0, 1)).ravel()
        np.copyto(inputs[0].host, data)
        [output] = trt_do_inference(
                context, bindings=bindings, inputs=inputs,
                outputs=outputs, stream=stream)
        return output[None, ...]
    
    inference = trt_inference if IS_TRT else cv2_inference
    img_paths = glob.glob(os.path.join(path, label, '*.jpg'))
    embeds, labels = [], []
    for img_path in img_paths:
        basename = os.path.basename(img_path)
        if config.IMG_ALIGNED_PREF not in basename:
            print('Image \"{}\" is not aligned'.format(img_path))
            exception_exit('You need to align the image dataset!\n',
                           stderr=True)
        else:
            img = load_image(img_path)
            img = cv2.resize(img, (config.MODEL_IMG_SIZE,
                                   config.MODEL_IMG_SIZE),
                             interpolation=cv2.INTER_CUBIC)
            output = inference(norm_binary(img))
            embeds.append(l2_norm(output)[0])
    labels += [label, ] * len(img_paths)
    return (embeds, labels)


# Function for getting embeddings from a dataset
def get_dataset_embeddings(model, ext, path):
    ds_labels = get_dataset_labels(path)
    embeddings, label_list = [], []
    for label in ds_labels:
        embeds, labels = get_dir_embeddings(model, ext, path, label)
        embeddings.append(embeds)
        label_list += labels

    embeddings = np.concatenate(embeddings)
    return (embeddings, label_list)


# Classifier Training
def train_images(model, ext, path):
    embeddings, label_list = get_dataset_embeddings(model, ext, path)
    le = LabelEncoder().fit(label_list)
    label_idx = le.transform(label_list)
    num_classes = len(le.classes_)
    print('Training the classifier for {} classes and {} images.'
          .format(num_classes, len(label_idx)))
    clf = SVC(C=1, kernel='linear', probability=True)
    clf.fit(embeddings, label_idx)
    print('Saving the classifier to a file:', config.CLF_PKL)
    with open(config.CLF_PKL, 'wb') as f:
        pickle.dump((le, clf), f)


# Image alignment
def align_images(fd, path):
    
    def resize_img(img):
        height, width, _ = img.shape
        max_size, scale = max(height, width), 1
        if max_size > config.FD_MAXSIZE:
            scale = config.FD_MAXSIZE / max_size
            img = cv2.resize(img, (0, 0), fx=scale, fy=scale)
        return (img, scale)

    ds_labels = get_dataset_labels(path)
    for label in ds_labels:
        img_paths = glob.glob(os.path.join(path, label, '*.jpg'))
        for img_path in img_paths:
            basename = os.path.basename(img_path)
            if config.IMG_ALIGNED_PREF not in basename:
                img = load_image(img_path)
                img_fd, scale = resize_img(img)
                face_locations = get_face_locations(fd, img_fd)
                if len(face_locations) == 0:
                    exception_exit('ERROR! Faces not found in the image '
                                   '\"{}\"!\n'.format(img_path), stderr=True)
                elif len(face_locations) > 1:
                    print('WARNING! The image \"{}\" contains more than one '
                          'face!'.format(img_path))
                top, right, bottom, left, _, _, _ = face_locations[0]
                top, right, bottom, left = scale_box(top, right, bottom, left,
                                                     scale)
                img = img[max(0, top):max(0, bottom),
                          max(0, left):max(0, right)]
                img = cv2.resize(img, (config.MODEL_IMG_SIZE,
                                       config.MODEL_IMG_SIZE),
                                 interpolation = cv2.INTER_CUBIC)
                
                filename, fileext = os.path.splitext(img_path)
                new_path = filename + '_' + config.IMG_ALIGNED_PREF + fileext
                _, buff = cv2.imencode('.jpg', img)
                buff.tofile(new_path)
                os.remove(img_path)
                
                print('Image \"{}\" is aligned'.format(img_path))
                
    print('Image alignment was completed successfully.')
                

# Description of the script
def show_description():
    print('This script is needed to prepare images with faces\n'
          'and create a calssifier file.\nCommands:\n'
          '\talign [directory] -\tAlign images with faces in the selected\n'
          '\t\t\t\tdirectory\n'
          '\ttrain [directory] -\tTrain the classifier on images '
          'in the selected\n\t\t\t\tdirectory\n'
          'Examples:\n\tprepare.py align images/train\n'
          '\tprepare.py train images/train\n')


# The main function
def main():
    if len(sys.argv) == 1:
        show_description()
    elif len(sys.argv) != 3:
        show_description()
        exception_exit('ERROR! Wrong number of parameters!\n', stderr=True)
    else:
        if sys.argv[1] == 'align':
            os.chdir(os.path.dirname(os.path.realpath(__file__)))
            fd = load_face_detector(config.FD_TYPE)
            os.chdir(CURRENT_DIR)
            align_images(fd, sys.argv[2])
        elif sys.argv[1] == 'train':
            os.chdir(os.path.dirname(os.path.realpath(__file__)))
            config.CLF_PKL = os.path.abspath(config.CLF_PKL)
            model, ext = load_model(config.MODEL_DATA)
            os.chdir(CURRENT_DIR)
            train_images(model, ext, sys.argv[2])
        else:
            show_description()
            exception_exit('ERROR! Wrong command!\n', stderr=True)


if __name__ == '__main__':
    main()