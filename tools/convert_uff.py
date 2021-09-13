#!/usr/bin/env python

import sys
import logging
import argparse
logging.disable(logging.WARNING)
# The "uff 0.6.9" module works only with "tensorflow 1.15.0"
UFF_DIR = '/usr/lib/python3.6/dist-packages'
sys.path.append(UFF_DIR)
import uff
logging.disable(logging.NOTSET)


def create_parser():
    parser = argparse.ArgumentParser(
        description=('Converts a Tensorflow frozen graph model '
                     'into a TensorRT UFF format'))
    parser.add_argument('frz_path', type=str,
                        help='specify the frozen model path')
    parser.add_argument('uff_path', type=str,
                        help='specify the UFF model path')
    return parser


def convert(frz_path, uff_path):
    uff.from_tensorflow_frozen_model(
        frozen_file=frz_path, output_nodes=["Identity"],
        output_filename=uff_path, debug_mode=False)


def main():
    parser = create_parser()
    if len(sys.argv) < 2:
        parser.print_help()
        sys.exit(0)
    args = parser.parse_args()        
    convert(args.frz_path, args.uff_path)


if __name__ == '__main__':
    main()