"""
Input converter for MVAD dataset
"""

import os
import json
import argparse
import string
import utils
#import h5py
#import nltk
#from nltk.tokenize import WordPunctTokenizer

def prepro_caption(sentence):
	print sentence
	txt = WordPunctTokenizer().tokenize(sentence)
	return (' '.join(txt)).encode('utf-8')
	  
def gen_dataset(video_list, input_txt):
	lines1 = utils.lines_list(video_list)
	lines2 = utils.lines_list(input_txt)
	captions = [{'caption': prepro_caption(' '.join(line2)), 'video_id': os.path.split(line1[0])[1]} for line1, line2 in zip(lines1,lines2)]
	#print captions
	return captions

def get_dataset(input_json):
	captions = json.load(open(input_json, 'r'))
	return captions

def create_vocab_file(input_txt, output_json):
	lines = utils.lines_list(input_txt)
	vocab =[]
	for line in lines:
		for word in line:
			if word not in vocab:
				vocab.append(word)

	with open(output_json, 'w') as f:
		json.dump(vocab, f)

def gen_subtitle(srt_file, input_json, output_file):
	lines = open(srt_file).readlines()
	captions = json.load(open(input_json, 'r'))
	
	cap = {}
	for caption in captions:
		cap[caption['image_id']] = caption['caption']
	i = 0
	for line in lines:
		if i == 0:
			avi = line
			output = open(output_file+'/'+avi.strip()+'.srt', 'w')
			output.write("1\n")
		elif i == 1:
			output.write("00:00:00,00 --> 00:00:10,50\n")
		elif i == 2:
			line = cap[avi.strip() + '.avi']
			output.write(line.strip() + "\n")
		elif i == 3: 
			output.close()
		i += 1
		if i == 4:
			i = 0



if __name__ == "__main__":
	
	parser = argparse.ArgumentParser()
	
	parser.add_argument('mode')
	parser.add_argument('--video_list')
	parser.add_argument('--input_txt', help='input file')
	parser.add_argument('--input_json', help='input file')
	parser.add_argument('--output', help='output file')
	
	args = parser.parse_args()

	if args.mode == 'gen_dataset':
		dataset = gen_dataset(args.video_list ,args.input_txt)
		with open(args.output,'w') as output_file:
		 	json.dump(dataset, output_file)
	elif args.mode == 'create_vocab_file':
		create_vocab_file(args.input_txt, args.output)
	elif args.mode == 'gen_subtitle':
		gen_subtitle(args.input_txt, args.input_json, args.output)




