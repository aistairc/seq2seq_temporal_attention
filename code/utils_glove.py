import os
import json
import argparse
import string
import utils
import h5py
import chainer
from chainer import functions, links, optimizers, cuda, serializers
from chainer.functions import *
import numpy as np

def read_word_vector(word, mapping):	
	if word not in mapping:
		return [1]*300
	else:
		return	mapping[word]

def read_whole_file(glovefile):
	return utils.read_mapping(glovefile)


if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	
	# input json
	parser.add_argument('mode')
	parser.add_argument('--glovefile', default='/Users/natsuda/Desktop/glove.840B.300d.txt')
	parser.add_argument('--gpu', default=-1)
  
	args = parser.parse_args()

	if args.mode == 'get_vector':
		read_word_vector(args.glovefile)
		
