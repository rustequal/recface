#!/usr/bin/env python

import os
import sys
import argparse
import mxnet as mx


def create_parser():
    parser = argparse.ArgumentParser(
        description=('Converts a dataset in MXNET RecordIO format '
                     'into a JPG format'))
    parser.add_argument('rec_path', type=str,
                        help='specify the MXNET RecordIO dataset path')
    parser.add_argument('jpg_path', type=str,
                        help='specify the JPG dataset path')
    return parser


def convert(rec_path, jpg_path):
    path_imgrec = os.path.join(rec_path, 'train.rec')
    path_imgidx = os.path.join(rec_path, 'train.idx')
    imgrec = mx.recordio.MXIndexedRecordIO(path_imgidx, path_imgrec, 'r')
    if not os.path.exists(jpg_path):
        os.makedirs(jpg_path)

    s = imgrec.read_idx(0)
    header, _ = mx.recordio.unpack(s)
    assert header.flag > 0
    print('header0 label', header.label)
    header0 = (int(header.label[0]), int(header.label[1]))
    seq_identity = range(int(header.label[0]), int(header.label[1]))
    ds_id, pp = 0, 0
    for identity in seq_identity:
        id_dir = os.path.join(jpg_path, "%d_%d" % (ds_id, identity))
        if not os.path.exists(id_dir):
            os.makedirs(id_dir)
        pp += 1
        if pp % 10000 == 0:
            print('processing id', pp)
        s = imgrec.read_idx(identity)
        header, _ = mx.recordio.unpack(s)
        imgid = 0
        for _idx in range(int(header.label[0]), int(header.label[1])):
            s = imgrec.read_idx(_idx)
            _header, _img = mx.recordio.unpack(s)
            image_path = os.path.join(id_dir, "%d.jpg" % imgid)
            f = open(image_path, 'wb')
            f.write(_img)
            f.close()    
            imgid += 1


def main():
    parser = create_parser()
    if len(sys.argv) < 2:
        parser.print_help()
        sys.exit(0)
    args = parser.parse_args()        
    convert(args.rec_path, args.jpg_path)
    print('Process completed successfully.')


if __name__ == '__main__':
    main()