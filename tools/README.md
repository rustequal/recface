This directory contains additional tools for preparing and processing data.<br><br>
**convert_uff.py** - Converts a Tensorflow frozen graph model into a TensorRT UFF format. You need to install the "Tensorflow" framework for the script to work, as described in the [Installation](/README.md#installation) section of the guide.<br>
<pre>
usage: convert_uff.py [-h] frz_path uff_path<br>
positional arguments:
  frz_path    specify the frozen model path
  uff_path    specify the UFF model path<br>
optional arguments:
  -h, --help  show this help message and exit
</pre>
Example of using the script:

- Connect to Nano via SSH. Login as "recface".
<pre>
<b>recface@jetson:~$ source /opt/venv/tensorflow/bin/activate</b>
<b>(tensorflow) recface@jetson:~$ cd /opt/recface/tools/</b>
<b>(tensorflow) recface@jetson:/opt/recface/tools$ ./convert_uff.py ../data/test.pb ../data/test.uff</b>
2021-08-20 23:18:20.290884: I tensorflow/stream_executor/platform/default/dso_loader.cc:49] Successfully opened dynamic library libcudart.so.10.2
WARNING:tensorflow:From /usr/lib/python3.6/dist-packages/uff/converters/tensorflow/conversion_helpers.py:274: The name tf.gfile.GFile is deprecated. Please use tf.io.gfile.GFile instead.<br>
NOTE: UFF has been tested with TensorFlow 1.15.0.
WARNING: The version of TensorFlow installed on this system is not guaranteed to work with UFF.
UFF Version 0.6.9
=== Automatically deduced input nodes ===
[name: "x"
op: "Placeholder"
attr {
  key: "dtype"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "shape"
  value {
    shape {
      dim {
        size: -1
      }
      dim {
        size: 112
      }
      dim {
        size: 112
      }
      dim {
        size: 3
      }
    }
  }
}
]
=========================================<br>
Using output node Identity
Converting to UFF graph
DEBUG: convert reshape to flatten node
DEBUG [/usr/lib/python3.6/dist-packages/uff/converters/tensorflow/converter.py:143] Marking ['Identity'] as outputs
No. nodes: 412
UFF Output written to ../data/test.uff
</pre>
