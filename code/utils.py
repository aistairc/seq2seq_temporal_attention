import os


def read_keys(filename):
    lines = open(filename).readlines()
    keys = [line.strip().split()[0] for line in lines]
    return keys


def lines_list(filename):
    with open(filename) as f:
        for line in f:
            yield line.strip().split()


def read_mapping(filename):
    lines = open(filename).readlines()
    keys = [line.strip().split()[0] for line in lines]
    values = [line.strip().split()[1:] for line in lines]
    return {k: v for k, v in zip(keys, values)}


def read_mapping_inv(filename):
    lines = open(filename).readlines()
    values = [line.strip().split()[0] for line in lines]
    keys = [int(line.strip().split()[1]) for line in lines]
    return {k: v for k, v in zip(keys, values)}


def create_file_if_not_exist(filename):
    try:
        fp = open(filename)
    except IOError:
        # If not exists, create the file
        fp = open(filename, 'w+')
    fp.close()


def create_folder_if_not_exist(folderpath):
    if not os.path.exists(folderpath):
        os.makedirs(folderpath)
