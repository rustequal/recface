The information on this page describes the process of training the model. If you are not going to train the model, then you do not need the scenarios described here. For training, it is recommended to have a high-performance PC with a powerful GPU. I trained the model using the "Google Colab Pro" service. If you use a NVIDIA V100 GPU, then the training time of one epoch is about 95 minutes. For training I used the Tensorflow 2.4 framework.

## Preparing the python environment

- I recommend using [Miniconda](https://docs.conda.io/en/latest/miniconda.html) to create a development environment.
- Install "Miniconda".
- Create a new python environment:
  <pre>
  <b>(base) user@dev:~$ conda create -n recface python=3.7</b>
  <b>(base) user@dev:~$ conda activate recface</b>
  </pre>
- Install the necessary packages:
  <pre>
  <b>(recface) user@dev:~$ conda install nodejs</b>
  <b>(recface) user@dev:~$ pip install scikit-learn matplotlib opencv-contrib-python</b>
  <b>(recface) user@dev:~$ pip install jupyterlab</b>
  <b>(recface) user@dev:~$ pip install ipywidgets</b>
  <b>(recface) user@dev:~$ pip install tensorflow==2.4.1 tensorboard==2.4.1</b>
  <b>(recface) user@dev:~$ pip install bcolz tqdm</b>
  <b>(recface) user@dev:~$ pip install tensorflow-addons</b>
  <b>(recface) user@dev:~$ pip install cmake</b>
  <b>(recface) user@dev:~$ pip install dlib</b>
  <b>(recface) user@dev:~$ pip install torch==1.7.1+cu110 torchvision==0.8.2+cu110 torchaudio==0.7.2 -f https://download.pytorch.org/whl/torch_stable.html</b>
  <b>(recface) user@dev:~$ pip install facenet-pytorch</b>
  <b>(recface) user@dev:~$ pip install mxnet</b>
  </pre>
  > For "torch" packages, select the package names that correspond to your version of CUDA. The "torch" and "facenet-pytorch" packages are needed to align images with faces using the MTCNN algorithm, if you need it.
- <details>
  <summary>List of packages after installation</summary>
  <pre>
  <b>(recface) user@dev:~$ pip list</b>
  Package                Version
  ---------------------- -------------------
  absl-py                0.13.0
  anyio                  3.3.0
  argcomplete            1.12.3
  argon2-cffi            21.1.0
  astunparse             1.6.3
  attrs                  21.2.0
  Babel                  2.9.1
  backcall               0.2.0
  bcolz                  1.2.1
  bleach                 4.1.0
  cachetools             4.2.2
  certifi                2021.5.30
  cffi                   1.14.6
  charset-normalizer     2.0.4
  cmake                  3.21.2
  cycler                 0.10.0
  debugpy                1.4.1
  decorator              5.0.9
  defusedxml             0.7.1
  dlib                   19.22.1
  entrypoints            0.3
  facenet-pytorch        2.5.2
  flatbuffers            1.12
  gast                   0.3.3
  google-auth            1.35.0
  google-auth-oauthlib   0.4.6
  google-pasta           0.2.0
  graphviz               0.8.4
  grpcio                 1.32.0
  h5py                   2.10.0
  idna                   3.2
  importlib-metadata     4.8.1
  ipykernel              6.3.1
  ipython                7.27.0
  ipython-genutils       0.2.0
  ipywidgets             7.6.4
  jedi                   0.18.0
  Jinja2                 3.0.1
  joblib                 1.0.1
  json5                  0.9.6
  jsonschema             3.2.0
  jupyter-client         7.0.2
  jupyter-core           4.7.1
  jupyter-server         1.10.2
  jupyterlab             3.1.10
  jupyterlab-pygments    0.1.2
  jupyterlab-server      2.7.2
  jupyterlab-widgets     1.0.1
  Keras-Preprocessing    1.1.2
  kiwisolver             1.3.2
  Markdown               3.3.4
  MarkupSafe             2.0.1
  matplotlib             3.4.3
  matplotlib-inline      0.1.2
  mistune                0.8.4
  mxnet                  1.8.0.post0
  nbclassic              0.3.1
  nbclient               0.5.4
  nbconvert              6.1.0
  nbformat               5.1.3
  nest-asyncio           1.5.1
  notebook               6.4.3
  numpy                  1.19.5
  oauthlib               3.1.1
  opencv-contrib-python  4.5.3.56
  opt-einsum             3.3.0
  packaging              21.0
  pandocfilters          1.4.3
  parso                  0.8.2
  pexpect                4.8.0
  pickleshare            0.7.5
  Pillow                 8.3.1
  pip                    21.0.1
  prometheus-client      0.11.0
  prompt-toolkit         3.0.20
  protobuf               3.17.3
  ptyprocess             0.7.0
  pyasn1                 0.4.8
  pyasn1-modules         0.2.8
  pycparser              2.20
  Pygments               2.10.0
  pyparsing              2.4.7
  pyrsistent             0.18.0
  python-dateutil        2.8.2
  pytz                   2021.1
  pyzmq                  22.2.1
  requests               2.26.0
  requests-oauthlib      1.3.0
  requests-unixsocket    0.2.0
  rsa                    4.7.2
  scikit-learn           0.24.2
  scipy                  1.7.1
  Send2Trash             1.8.0
  setuptools             52.0.0.post20210125
  six                    1.15.0
  sniffio                1.2.0
  tensorboard            2.4.1
  tensorboard-plugin-wit 1.8.0
  tensorflow             2.4.1
  tensorflow-addons      0.14.0
  tensorflow-estimator   2.4.0
  termcolor              1.1.0
  terminado              0.11.1
  testpath               0.5.0
  threadpoolctl          2.2.0
  torch                  1.7.1+cu110
  torchaudio             0.7.2
  torchvision            0.8.2+cu110
  tornado                6.1
  tqdm                   4.62.2
  traitlets              5.1.0
  typeguard              2.12.1
  typing-extensions      3.7.4.3
  urllib3                1.26.6
  wcwidth                0.2.5
  webencodings           0.5.1
  websocket-client       1.2.1
  Werkzeug               2.0.1
  wheel                  0.37.0
  widgetsnbextension     3.5.1
  wrapt                  1.12.1
  zipp                   3.5.0
  </pre>
  </details>
- Clone the repository:
  <pre>
  <b>(recface) user@dev:~$ git clone https://github.com/rustequal/recface.git</b>
  <b>(recface) user@dev:~$ cd recface/train/</b>
  </pre>
- Download the [MS1M-ArcFace (85K ids/5.8M images)](https://github.com/deepinsight/insightface/wiki/Dataset-Zoo) ZIP-archive (12 GB).
- Unpack the "faces_emore.zip" archive into "recface/train" directory.
  <pre>
  <b>(recface) user@dev:~/recface/train$ unzip faces_emore.zip</b>
  </pre>
- Move the binary datasets to "recface/train" directory:
  <pre>
  <b>(recface) user@dev:~/recface/train$ mv faces_emore/*.bin .</b>
  </pre>
- Create the "10-tensorflow-gpu-mem.py" startup file:
  <pre>
  <b>(recface) user@dev:~/recface/train$ mkdir -p ~/.ipython/profile_default/startup</b>
  </pre>
  > Create and save the contents of the file:
  <pre>
  <b>(recface) user@dev:~/recface/train$ vi ~/.ipython/profile_default/startup/10-tensorflow-gpu-mem.py</b>
  <i>import os
  os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'

  import tensorflow as tf
  physical_devices = tf.config.list_physical_devices('GPU')
  tf.config.experimental.set_memory_growth(physical_devices[0], True)
  tf.config.experimental.enable_tensor_float_32_execution(False)</i>
  </pre>
  > The last line is needed in case of using NVIDIA GPU based on Ampere chip and newer to disable the "TensorFloat-32" feature.
- Download the file [mmod_human_face_detector.dat](https://github.com/davisking/dlib-models) to "recface/train" directory (you need to unpack the "mmod_human_face_detector.dat.bz2" archive):
  <pre>
  <b>(recface) user@dev:~/recface/train$ bzip2 -d mmod_human_face_detector.dat.bz2</b>
  </pre>
- Prepare the model weights for testing:
  <pre>
  <b>(recface) user@dev:~/recface/train$ unzip weights/95_Default_MobileFaceNetV2_ArcFace_planF_batch256_model.zip</b>
  <b>(recface) user@dev:~/recface/train$ unzip -o weights/95_dlib_continued_MobileFaceNetV2_ArcFace_planH_batch256_model.zip</b>
  </pre>
- Run the "Jupyter Lab":
  <pre>
  <b>(recface) user@dev:~/recface/train$ jupyter lab</b>
  </pre>

## Creating the training dataset "ms1mv2_bin_dlib_continued.tfrecord"

- Use the script [rec2jpg.py](/train/tools/rec2jpg.py) to convert dataset to JPG format
  <pre>
  <b>(recface) user@dev:~/recface/train$ python ./tools/rec2jpg.py faces_emore faces_emore_jpg</b>
  </pre>
- Run the "JupyterLab" session, navigate the "File Browser" to directory "recface/train".
- Duplicate the "recface_95_dlib_continued_MobileFaceNetV2_ArcFace_planH_batch256_EN.ipynb" notebook.
- Rename the duplicated notebook to "ms1mv2_bin.ipynb".
- Open the duplicated notebook "ms1mv2_bin.ipynb".
- Clear all outputs in this new notebook.
- Change the configuration parameters in cell 8:
  ```
  source_dataset = {
      'path': 'faces_emore_jpg',
      'align': True,
      'align_error_continue': True,
      'align_error_delete': False,

  cfg = {
      'train_build_dataset': 'ms1mv2_bin_dlib_continued.tfrecord',
      'test_build_path': '',
      'test_build_name': 'friends_16_test_dlib.bin',
      'test_build_prefix': 'test_data',
      'test_split': 0,
      'test_samples': 0,
      'test_max_pairs': 3000,
      'face_detector_type': 'dlib', # 'MTCNN' or 'dlib'
      'face_detector_prob': 0.4,
      'face_detector_square': False,
      'binary_img': True,
  ```
- Run the cells: 2, 7, 8, 10, 11, 12
- The dataset file "ms1mv2_bin_dlib_continued.tfrecord" must be created in the current directory. Check the dataset parameters:
  <pre>
  <b>(recface) user@dev:~/recface/train$ cat ms1mv2_bin_dlib_continued.tfrecord.txt</b>
  {'num_samples': 5822653, 'num_classes': 85742}
  </pre>
- Delete the directory "faces_emore_jpg":
  <pre>
  <b>(recface) user@dev:~/recface/train$ rm -r faces_emore_jpg</b>
  </pre>
> If you want to create the training dataset "ms1mv2_bin_mtcnn_sq_continued.tfrecord" then use the same guide, but change the parameters of cell 8:
> <pre>
> 'train_build_dataset': 'ms1mv2_bin_mtcnn_sq_continued.tfrecord',
> 'face_detector_type': 'MTCNN',
> 'face_detector_square': True,
> </pre>

## Creating the source dataset "friends_16"

Prepare your own 16-person friend faces dataset. The rules:

- <b>These persons should NOT be in the training dataset "MS1M-ArcFace"</b>
- The dataset should be a directory containing 2 subdirectories "train" and "test" corresponding to two subsets that will be used for training and testing purposes.
- The "train" and "test" directories must have an identical structure. They should contain 16 subdirectories of persons in each - 16 subdirectories of persons in the "train" directory and 16 subdirectories in the "test" directory.
- Each person subdirectory should contain the images with one person with a clearly visible face.
- The recommended number of images is 20 per person in the "train" subset and 10 per person in the "test" subset. The images must be unique for the two subsets.
- Name the dataset directory "friends_16".

## Creating the "friends_16_test_dlib.bin" and "friends_16_test_mtcnn_sq.bin" datasets

- Prepare the working directories of the datasets:
  <pre>
  <b>(recface) user@dev:~/recface/train$ cp -r friends_16 friends_16_dlib</b>
  <b>(recface) user@dev:~/recface/train$ cp -r friends_16 friends_16_mtcnn_sq</b>
  </pre>
- Run the "JupyterLab" session, navigate the "File Browser" to directory "recface/train".
- Duplicate the "recface_95_dlib_continued_MobileFaceNetV2_ArcFace_planH_batch256_EN.ipynb" notebook.
- Rename the duplicated notebook to "friends_16.ipynb".
- Open the duplicated notebook "friends_16.ipynb".
- Clear all outputs in this new notebook.
- Change the configuration parameters in cell 8:
  ```
  source_dataset = {
      'path': 'friends_16_dlib/test',
      'align': True,
      'align_error_continue': False,
      'align_error_delete': False,

  cfg = {
      'train_build_dataset': 'friends_16.tfrecord',
      'test_build_path': './',
      'test_build_name': 'friends_16_test_dlib.bin',
      'test_build_prefix': 'test_data',
      'test_split': 1,
      'test_samples': 0,
      'test_max_pairs': 3000,
      'face_detector_type': 'dlib', # 'MTCNN' or 'dlib'
      'face_detector_prob': 0.4,
      'face_detector_square': False,
      'binary_img': True,
  ```
- Run the cells: 2, 7, 8, 10, 11, 12
  > If errors occurred during the alignment of images, then you need to find other images so that the face detector can detect the face without errors. Or, reduce the resolution of the images in case of an out of memory error.
- The dataset "friends_16_test_dlib.bin" with 720 positive and 720 negative pairs must be created in the current directory.
- Change the configuration parameters in cell 8:
  ```
  source_dataset = {
      'path': 'friends_16_mtcnn_sq/test',
      'align': True,
      'align_error_continue': False,
      'align_error_delete': False,

  cfg = {
      'train_build_dataset': 'friends_16.tfrecord',
      'test_build_path': './',
      'test_build_name': 'friends_16_test_mtcnn_sq.bin',
      'test_build_prefix': 'test_data',
      'test_split': 1,
      'test_samples': 0,
      'test_max_pairs': 3000,
      'face_detector_type': 'MTCNN', # 'MTCNN' or 'dlib'
      'face_detector_prob': 0.4,
      'face_detector_square': True,
      'binary_img': True,
  ```
- Restart the kernel and run the cells: 2, 7, 8, 10, 11, 12
  > If errors occurred during the alignment of images, then you need to find other images so that the face detector can detect the face without errors. Or, reduce the resolution of the images in case of an out of memory error.
- The dataset "friends_16_test_mtcnn_sq.bin" with 720 positive and 720 negative pairs must be created in the current directory.

## Creating aligned versions of the "lfw.bin" and "agedb_30.bin" datasets

- Duplicate the original datasets:
  <pre>
  <b>(recface) user@dev:~/recface/train$ cp lfw.bin lfw_dlib.bin</b>
  <b>(recface) user@dev:~/recface/train$ cp lfw.bin lfw_mtcnn_sq.bin</b>
  <b>(recface) user@dev:~/recface/train$ cp agedb_30.bin agedb_30_dlib.bin</b>
  <b>(recface) user@dev:~/recface/train$ cp agedb_30.bin agedb_30_mtcnn_sq.bin</b>
  </pre>
- Run the "JupyterLab" session, navigate the "File Browser" to directory "recface/train".
- Duplicate the "recface_95_dlib_continued_MobileFaceNetV2_ArcFace_planH_batch256_EN.ipynb" notebook.
- Rename the duplicated notebook to "align_bin.ipynb".
- Open the duplicated notebook "align_bin.ipynb".
- Clear all outputs in this new notebook.
- Change the configuration parameters in cell 8 (set "align" to True):
  ```
  test_dlib_binary_datasets = [
      {'name': 'Friends 16',
       'path': 'friends_16_test_dlib.bin'},
      {'name': 'LFW',
       'path': 'lfw_dlib.bin',
       'align': True,
       'align_error_continue': False,
       'align_error_delete': False},
      {'name': 'AgeDB30',
       'path': 'agedb_30_dlib.bin',
       'align': True,
       'align_error_continue': False,
       'align_error_delete': False,

  cfg = {
      'face_detector_type': 'dlib', # 'MTCNN' or 'dlib'
      'face_detector_prob': 0.4,
      'face_detector_square': False,
      'binary_img': True,

      'test_datasets': test_dlib_binary_datasets,
  ```
- Run the cells: 2, 7, 8, 10, 11, 20, 25
  > Do not rerun the notebook with the same parameters, as this will lead to the re-alignment of the datasets and their damage.
- The datasets "lfw_dlib.bin" and "agedb_30_dlib.bin" will be aligned.
- Change the configuration parameters in cell 8 (set "align" to True):
  ```
  test_mtcnn_sq_binary_datasets = [
      {'name': 'Friends 16',
       'path': 'friends_16_test_mtcnn_sq.bin'},
      {'name': 'LFW',
       'path': 'lfw_mtcnn_sq.bin',
       'align': True,
       'align_error_continue': False,
       'align_error_delete': False,
       'align_manual': {
           '11217': [20, 13, 109, 111]}},
      {'name': 'AgeDB30',
       'path': 'agedb_30_mtcnn_sq.bin',
       'align': True,
       'align_error_continue': False,
       'align_error_delete': False,

  cfg = {
      'face_detector_type': 'MTCNN', # 'MTCNN' or 'dlib'
      'face_detector_prob': 0.4,
      'face_detector_square': True,
      'binary_img': True,

      'test_datasets': test_mtcnn_sq_binary_datasets,
  ```
- Restart the kernel and run the cells: 2, 7, 8, 10, 11, 20, 25
  > Do not rerun the notebook with the same parameters, as this will lead to the re-alignment of the datasets and their damage.
- The datasets "lfw_mtcnn_sq.bin" and "agedb_30_mtcnn_sq.bin" will be aligned.

> Face detectors work differently on different operating systems (Windows, Ubuntu, Google Colab). In addition, face detectors work differently when using a GPU or CPU. Therefore, the accuracy of the model when testing with datasets aligned on different operating systems may vary slightly. I aligned the test datasets using the platform: Ubuntu 20.04 (x86_64) + CUDA 11.0 + NVIDIA RTX 3060 GPU.

## Testing the model using aligned datasets

- Run the "JupyterLab" session, navigate the "File Browser" to directory "recface/train".
- Duplicate the "recface_95_dlib_continued_MobileFaceNetV2_ArcFace_planH_batch256_EN.ipynb" notebook.
- Rename the duplicated notebook to "testing_datasets.ipynb".
- Open the duplicated notebook "testing_datasets.ipynb".
- Clear all outputs in this new notebook.
- Run the cells: 2, 7, 8, 10, 11, 20, 25
> If you want to test the model using unaligned datasets then use the same guide, but duplicate the "recface_95_Default_MobileFaceNetV2_ArcFace_planF_batch256_EN.ipynb" notebook.

## Creating and testing a classifier

- Create the "Others" directory.
- Fill the "Others" directory with images of persons other than 16 friends. These images will be needed for testing the classifier in order to determine persons other than those in the classifier database. I used 3200 images of other people.
- Copy the "Others" directory with images to each of the subsets:
  <pre>
  <b>(recface) user@dev:~/recface/train$ cp -r Others friends_16_dlib/test/</b>
  <b>(recface) user@dev:~/recface/train$ cp -r Others friends_16_mtcnn_sq/test/</b>
  </pre>
- Run the "JupyterLab" session, navigate the "File Browser" to directory "recface/train".
- Duplicate the "recface_95_dlib_continued_MobileFaceNetV2_ArcFace_planH_batch256_EN.ipynb" notebook.
- Rename the duplicated notebook to "testing_clf.ipynb".
- Open the duplicated notebook "testing_clf.ipynb".
- Clear all outputs in this new notebook.
- Change the configuration parameters in cell 8:
  ```
  clf_train_dataset = {
      'path': r'friends_16_dlib/train',
      'align': True}

  clf_test_dataset = {
      'path': r'friends_16_dlib/test',
      'align': True}
      
  cfg = {
      'face_detector_type': 'dlib', # 'MTCNN' or 'dlib'
      'face_detector_prob': 0.4,
      'face_detector_square': False,
  ```
- Run the cells: 2, 7, 8, 10, 11, 20, 26, 27, 28, 29
  > If errors occurred during the alignment of images, then you need to find other images so that the face detector can detect the face without errors. Or, reduce the resolution of the images in case of an out of memory error.
- A classifier will be created and tested for the "friends_16" dataset aligned with the "dlib" face detector.
- Change the configuration parameters in cell 8:
  ```
  clf_train_dataset = {
      'path': r'friends_16_mtcnn_sq/train',
      'align': True}

  clf_test_dataset = {
      'path': r'friends_16_mtcnn_sq/test',
      'align': True}
      
  cfg = {
      'face_detector_type': 'MTCNN', # 'MTCNN' or 'dlib'
      'face_detector_prob': 0.4,
      'face_detector_square': True,
  ```
- Restart the kernel and run the cells: 2, 7, 8, 10, 11, 20, 26, 27, 28, 29
  > If errors occurred during the alignment of images, then you need to find other images so that the face detector can detect the face without errors. Or, reduce the resolution of the images in case of an out of memory error.
- A classifier will be created and tested for the "friends_16" dataset aligned with the "MTCNN" face detector.

## Training the model

- To train the model, you will need a training dataset, aligned datasets, aligned directories with images for classifier validation, so it's recommended to run all the previous scenarios in order to prepare the data.
- Run the "JupyterLab" session, navigate the "File Browser" to directory "recface/train".
- Duplicate the "recface_95_dlib_continued_MobileFaceNetV2_ArcFace_planH_batch256_EN.ipynb" notebook.
- Rename the duplicated notebook to "training.ipynb".
- Open the duplicated notebook "training.ipynb".
- Clear all outputs in this new notebook.
- Change the project name in cell 2.
- Change the configuration parameters in cell 8:
  ```
  clf_datasets = [
      {'name': 'Classifier',
       'clf_type': 'svc_lin',
       'train_path': r'friends_16_dlib/train',
       'test_path': r'friends_16_dlib/test',
       'align': True}]

  clf_datasets_mixed = [
      {'name': 'Classifier',
       'clf_type': 'svc_lin',
       'train_path': r'friends_16_dlib/train',
       'test_path': r'friends_16_dlib/test',
       'align': True},
      {'name': 'Clf mtcnn',
       'clf_type': 'svc_lin',
       'train_path': r'friends_16_mtcnn_sq/train',
       'test_path': r'friends_16_mtcnn_sq/test',
       'align': True}]
  
  train_plan = { # (epochs, lr, val_save, val_per_epoch)
      'plan': ((8, 0.01, False, 1), (6, 0.005, False, 1), (8, 0.0025, True, 1),
               (10, 1e-3, True, 4), (10, 1e-4, True, 4), (10, 1e-5, True, 4)),
      'train_dataset': 'ms1mv2_bin_dlib_continued.tfrecord',
      'val_datasets': val_datasets,
      'clf_datasets': clf_datasets,
      'train_helper': copy_to_google_drive,
      'train_head': 'ArcFaceHead', # 'ArcFaceHead', 'CosFaceHead',
                                   # 'AdaCosHead', 'SphereFaceHead'
      'margin': 0.5, # 0.5 for ArcFace, 0.35 for CosFace, 1.35 for SphereFace
      'batch_size': 256,
      'train_crop': True,
      'pretrained': False,
      'num_classes': 85742,
      'num_samples': 5822653,     
      'logist_scale': 64} # for AdaCosHead this is "max_scale"
  
  cfg = {
      'net_type': 'MobileFaceNetV2', # 'ResNet34', 'ResNet34R', 'ResNet50'
          # 'ResNet101', 'MobileNetV2', 'MobileFaceNetV2',
          # 'Seesaw_shuffleFaceNet'
      'input_size': 112,
      'embd_shape': 512,

      'normalization': '-1:1', # '0:1', '-1:1', '-0.99:0.99', '0:255'
      'w_decay': 5e-4, #5e-4, 1e-6, 1e-8
  ```
- Change the variable "train_plan" in cell 8, if necessary. In particular, you can adjust the batch size, the training plan, the validation plan, and other parameters. You can change this parameters after each stage of training. You can also change the value of "w_decay" after each stage.
- Run the cells: 1, 2, 7, 8, 10, 11, 14, 15, 16 + training cells
> The Transfer Learning process is very similar to training, but it uses the "learn_plan" variable in cell 8. And instead of cells 14, 15, 16, you need to run cells 17, 18, 19 + learning cells.

## Serializing and saving the model

- After completing the training process, you need to save the model weights in production format. Run the cell 30 for this.
- The created ZIP-archive with the model weights can be copied to the production platform.
