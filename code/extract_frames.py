#!/usr/bin/env python

from __future__ import division, print_function

import os
import sys
import argparse
import cv2
import numpy as np
import utils

try:
    xrange
except NameError:
    xrange = range


def extract_frames(path, stride=1):
    print(path)
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        print("Error: Failed to open %s" % path)
        sys.exit(-1)

    try:
        FRAME_COUNT = cv2.CAP_PROP_FRAME_COUNT
        FRAME_HEIGHT = cv2.CAP_PROP_FRAME_HEIGHT
        FRAME_WIDTH = cv2.CAP_PROP_FRAME_WIDTH
    except AttributeError:
        FRAME_COUNT = cv2.cv.CV_CAP_PROP_FRAME_COUNT
        FRAME_HEIGHT = cv2.cv.CV_CAP_PROP_FRAME_HEIGHT
        FRAME_WIDTH = cv2.cv.CV_CAP_PROP_FRAME_WIDTH

    number_of_frames = int(cap.get(FRAME_COUNT))
    length2 = number_of_frames // stride
    height = int(cap.get(FRAME_HEIGHT))
    width = int(cap.get(FRAME_WIDTH))
    frames = np.zeros((length2, height, width, 3), dtype=np.uint8)
    for frame_i in xrange(length2):
        _, image = cap.read()
        frames[frame_i] = image[:, :, :]
        for i in xrange(1, stride):
            _, image = cap.read()
    print(len(frames))
    return frames


if __name__ == "__main__":

    description = \
        "Extract features for video frames using chainer and caffe pretrained models"

    argparser = argparse.ArgumentParser(description=description)
    argparser.add_argument("movie_folder",
                           type=str,
                           help="input folder")
    argparser.add_argument("output_folder",
                           type=str,
                           help="output folder")
    argparser.add_argument('--keys_file',
                           type=str,
                           default='keys.txt',
                           help='keep records of processed files')
    argparser.add_argument('--frame_list',
                           type=str,
                           default='frame_list.txt',
                           help='list of extracted image frames')
    argparser.add_argument('--stride',
                           type=int,
                           default=8)
    args = argparser.parse_args()

    videos_path = args.movie_folder
    keys_file = args.keys_file
    output_path = args.output_folder

    utils.create_file_if_not_exist(keys_file)
    keys = open(keys_file, "a")
    a = utils.read_keys(keys_file)

    utils.create_file_if_not_exist(args.frame_list)
    frame_list = open(args.frame_list, "a")

    old_keys = [row for row in a]
    # print(old_keys)
    video_names = os.listdir(videos_path)
    for video_name in video_names:
        key = os.path.join(output_path, video_name)
        if not video_name.startswith('.') and key not in old_keys:
            frames = extract_frames(os.path.join(videos_path, video_name), args.stride)
            utils.create_folder_if_not_exist(key)
            step = 1
            for i, frame in enumerate(frames):
                print(key, len(frame))
                m = cv2.imwrite(os.path.join(key, "%06d.png" % step), frame)
                print(m)
                step += args.stride
                frame_list.write("%s %s\n" % (key, "%06d.png" % step))

            keys.write("%s %d\n" % (key, len(frames)))
    keys.flush()
    keys.close()
    frame_list.flush()
    frame_list.close()
