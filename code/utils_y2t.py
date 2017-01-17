"""
Input converter for Youtube2Text dataset
"""

import os
import json
import argparse
import string
import utils
import h5py

def prepro_caption(sentence):
	#print sentence.encode('utf8')
	txt = str(sentence.encode('utf8')).lower().translate(None, string.punctuation)
	return txt
	  
def get_dataset(input_txt, split):
	lines = utils.lines_list(input_txt)
	captions = [{'caption': prepro_caption(' '.join(line[1:])), 'video_id': line[0]} for line in lines]

	return captions

def map_feature(input_h5, output_h5, input_map):
	mapping = utils.read_mapping(input_map)
	fw = h5py.File(output_h5, 'w')
	with h5py.File(input_h5, 'r') as f:
		for m in mapping:
			print m, mapping[m]
			fw.create_dataset(mapping[m][0], data=f[m + '.avi'])
	fw.close()

def create_vocab_file(input_txt, output_json):
	vocab = utils.read_keys(input_txt)
	with open(output_json, 'w') as f:
		json.dump(vocab, f)
	
if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	
	# input json
	parser.add_argument('mode')
	parser.add_argument('--input', help='input file')
	parser.add_argument('--output', help='output file')
	parser.add_argument('--mapfile', default='../../../data/Y2T/subhashini/youtube_video_to_id_mapping.txt')
  
	args = parser.parse_args()

	if args.mode == 'get_dataset':
		dataset = get_dataset(args.input)
		with open(args.output,'w') as output_file:
			json.dump(dataset, output_file)
	elif args.mode == 'map_feature':
		map_feature(args.input, args.output, args.mapfile)
	elif args.mode == 'create_vocab_file':
		create_vocab_file(args.input, args.output)



