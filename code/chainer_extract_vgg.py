#!/usr/bin/env python

from __future__ import division, print_function

import os
import re
import sys
import argparse
import cv2
import pickle
import numpy as np
import h5py

import chainer
from chainer.links import caffe
from chainer import cuda

"""
Resize and crop an image to 224x224 (some part of sourcecode from chainer_imagenet_tools/inspect_caffenet.py)
Extract features of an image frame using caffe pretrained model and chainer
"""


def mismatch(error_message):
    print('An error occurred in loading a property of model.')
    print('Probably there is a mismatch between the versions of Chainer.')
    print('Remove the pickle file and try again.')
    print(error_message)
    sys.exit(1)


def chainer_extract_features(input_folder, batchsize, layer='fc7'):
    i = 0
    z = xp.zeros((len(frames), 4096), dtype=np.float32)
    x_batch = np.ndarray((batchsize, 3, in_size, in_size), dtype=np.float32)
    num_frame = len(os.listdir(input_folder))
    for step in range(num_frame):
        step2 = 1 + (step * 8)
        image_path = os.path.join(input_folder, "%06d.png" % step2)
        print(image_path)
        image = cv2.imread(image_path)
        height, width, depth = image.shape
        new_height = output_side_length
        new_width = output_side_length
        if height > width:
            new_height = output_side_length * height // width
        else:
            new_width = output_side_length * width // height
        resized_img = cv2.resize(image, (new_width, new_height))
        height_offset = (new_height - output_side_length) // 2
        width_offset = (new_width - output_side_length) // 2
        image = resized_img[height_offset:height_offset + output_side_length,
                            width_offset:width_offset + output_side_length]

        image = image.transpose(2, 0, 1)
        image = image[:, start:stop, start:stop].astype(np.float32)
        image -= mean_image

        x_batch[i] = image
        i += 1
        if i == batchsize:
            x_data = xp.asarray(x_batch)
            x = chainer.Variable(x_data)
            try:
                y, = func(inputs={'data': x}, outputs=[layer])
            except AttributeError as e:
                mismatch(str(e))
            z[step - batchsize + 1:step + 1] = y.data
            # print(y.data)
            i = 0
    if not i == 0:
        x_data = xp.asarray(x_batch[0:i])
        x = chainer.Variable(x_data)
        try:
            y, = func(inputs={'data': x}, outputs=[layer])
        except AttributeError as e:
            mismatch(str(e))
        z[len(frames) - i:len(frames)] = y.data

    return z


if __name__ == "__main__":

    description = \
        "Extract features for images using chainer and caffe pretrained models"

    argparser = argparse.ArgumentParser(description=description)
    argparser.add_argument("input_folder",
                           type=str,
                           help="input frame folder")
    argparser.add_argument("output_filename",
                           type=str,
                           help="output file name")
    argparser.add_argument("--mode", default='all')
    argparser.add_argument("--mean",
                           default='ilsvrc_2012_mean.npy')
    argparser.add_argument('--model',
                           help='path to model file',
                           default='VGG_ILSVRC_16_layers.caffemodel',
                           type=str)
    argparser.add_argument('--batchsize',
                           help='batch size',
                           default=50,
                           type=int)
    argparser.add_argument('--layer',
                           help='layer name: fc6, fc7(default), fc8',
                           default='fc7',
                           type=str,
                           choices=['fc6', 'fc7', 'fc8'])
    argparser.add_argument('--gpu',
                           '-g',
                           type=int,
                           default=-1,
                           help='Zero-origin GPU ID (nevative value indicates CPU)')

    args = argparser.parse_args()
    if args.gpu >= 0:
        cuda.check_cuda_available()
    xp = cuda.cupy if args.gpu >= 0 else np

    caffe_model = args.model
    func = caffe.CaffeFunction(caffe_model)
    if args.gpu >= 0:
        cuda.get_device(args.gpu).use()
        func.to_gpu()

    in_size = 224
    mean_image = np.load(args.mean)
    cropwidth = 256 - in_size
    start = cropwidth // 2
    stop = start + in_size
    mean_image = mean_image[:, start:stop, start:stop].copy()
    target_shape = (256, 256)
    output_side_length = 256

    features_path = args.output_filename
    if args.mode == 'all':
        videos = os.listdir(args.input_folder)
        fw = h5py.File(features_path + '.h5', mode='w')

        for video in videos:
            print(video)
            path = os.path.join(args.input_folder, video)
            frames = os.listdir(path)
            y = chainer_extract_features(path, args.batchsize, args.layer)
            y_cpu = cuda.to_cpu(y)
            fw.create_dataset(video, data=y_cpu)
        fw.close()

    elif args.mode == 'append':
        videos = os.listdir(args.input_folder)
        fw = h5py.File(features_path + '.h5', mode='a')

        for video in videos:
            print(video)
            path = os.path.join(args.input_folder, video)
            frames = os.listdir(path)
            y = chainer_extract_features(path, args.batchsize, args.layer)
            y_cpu = cuda.to_cpu(y)
            fw.create_dataset(video, data=y_cpu)
        fw.close()
