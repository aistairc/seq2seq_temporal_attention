import os
import argparse
import utils
import h5py
import numpy as np

# from c3d_read_binary_blob import read_binary_blob


def find_diff(input1, input2):
    list1 = []
    lines1 = utils.lines_list(input1)
    for line1 in lines1:
        head, filename = os.path.split(line1[0])
        if filename not in list1:
            list1.append(filename)

    list2 = []
    lines2 = utils.lines_list(input2)
    for line2 in lines2:
        head, filename = os.path.split(line2[0])
        if filename not in list1:
            list2.append(filename)

    return list2


def merge_feature_resnet(input_path, output_file):
    fw = h5py.File(output_file, 'w')
    dirs = os.listdir(input_path)
    for d in dirs:
        files = os.listdir(input_path + '/' + d)
        numfiles = len(files)
        framenum = 1
        feat = []
        for i in range(numfiles):
            print '%d Process features for %s' % (i, framenum)
            f1 = h5py.File('{}/{}.h5'.format(input_path + '/' + d,
                                             '%06.d' % framenum),
                           'r')
            f1 = np.array(f1['feat'])
            feat.append(f1)
            framenum += 8
        fw.create_dataset(d, data=feat)
    fw.close()
    print 'concatenated features and saved'


def merge_feature_c3d(input_path, output_file):
    fw = h5py.File(output_file, 'w')
    dirs = os.listdir(input_path)
    for d in dirs:
        files = os.listdir(input_path + '/' + d)
        numfiles = len(files)/4
        framenum = 1
        feat = []
        for i in range(numfiles):
            print '%d Process features for %s' % (i, framenum)

            c3d_blob = read_binary_blob('{}/{}.fc6-1'.format(input_path + '/' + d,
                                                             '%06.d' % framenum))
            assert(c3d_blob[2] == 1)
            f1 = np.array(c3d_blob[1].data.reshape(-1))
            feat.append(f1)
            framenum += 8
        fw.create_dataset(d, data=feat)
    fw.close()
    print 'concatenated features and saved'


def merge_feature_file(input1, input2, output, list1, list2):
    f1 = h5py.File(input1, 'r')
    f2 = h5py.File(input2, 'r')
    f3 = h5py.File(output, 'w')

    lines = utils.lines_list(list1)
    for line in lines:
        head, filename = os.path.split(line[0])
        f3.create_dataset(filename, data=f1[filename])

    lines = utils.lines_list(list2)
    for line in lines:
        head, filename = os.path.split(line[0])
        print filename
        f3.create_dataset(filename, data=f2[filename])

    f1.close()
    f2.close()
    f3.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', help='find_diff, merge features c3d')
    parser.add_argument('--input1', default='/home/ubuntu/git/data/Y2T/resnet', help='input#1')
    parser.add_argument('--input2', help='input#2')
    parser.add_argument('--output', default='/home/ubuntu/git/data/Y2T/resnet.h5', help='output')
    parser.add_argument('--list1')
    parser.add_argument('--list2')

    args = parser.parse_args()

    if args.mode == 'find_diff':
        diff = find_diff(args.input1, args.input2)
        with open('download.txt', 'w') as f:
            for d in diff:
                video = d.split('_DVS')[0]
                f.write('wget --user courvila_contact --password 59db938f6d http://lisaweb.iro.umontreal.ca/transfert/lisa/users/courvila/data/lisatmp2/torabi/DVDtranscription/' + video + '/video/' + d + '\n')

    elif args.mode == 'merge_feature_c3d':
        merge_feature_c3d(args.input1, args.output)

    elif args.mode == 'merge_feature_resnet':
        merge_feature_resnet(args.input1, args.output)

    elif args.mode == 'merge_feature_file':
        merge_feature_file(args.input1, args.input2, args.output, args.list1, args.list2)
