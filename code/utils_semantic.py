import sys
import os
import json
import numpy as np
from chainer import cuda
import h5py
import random
random.seed(4)
import utils_glove

import chainer
from chainer import functions, links, optimizers, cuda, serializers
from chainer.functions import *

def replace_unknown_words(vocab, caption):
	for i in xrange(len(caption)):
		if not caption[i] in vocab:
			caption[i] = '~'
	return caption

def gen_batch(train_data, source, batch_size, vocab, xp, words):
	vid_batch = []
	caption_batch = []
	word_batch = []
	random.shuffle(train_data)
	for line in train_data:		
		caption_batch.append(replace_unknown_words(vocab,line['caption'].strip().split()))

		with h5py.File(source, 'r') as f:
			feat = xp.array(f[line['video_id']], dtype=xp.float32)
		with h5py.File(words, 'r') as w:
			word = np.array(w[line['video_id']])

		vid_batch.append(feat)
		word_batch.append(create_word_pool(word, vocab))

		if(len(caption_batch) == batch_size):
			yield vid_batch,caption_batch, word_batch
			caption_batch = []
			vid_batch = []
			word_batch = []

def gen_batch_val(val_data, source, batch_size, vocab, xp, words):
	vid_batch = []
	caption_batch = []
	id_batch = []
	word_batch = []
	for line in val_data:
		caption_batch.append(replace_unknown_words(vocab,line['caption'].strip().split()))

		with h5py.File(source, 'r') as f:
			feat = xp.array(f[line['video_id']], dtype=xp.float32)
		with h5py.File(words, 'r') as w:
			word = np.array(w[line['video_id']])

		vid_batch.append(feat)
		word_batch.append(create_word_pool(word, vocab))
		id_batch.append(line['video_id'])
		
		if(len(caption_batch) == batch_size):
			yield vid_batch,caption_batch,id_batch, word_batch
			caption_batch = []
			vid_batch = []
			id_batch = []
			word_batch = []

	if not len(vid_batch) == 0:
		yield vid_batch,caption_batch,id_batch, word_batch

def gen_batch_test(test_data, source, batch_size, vocab, xp, words):
	vid_batch = []
	caption_batch = []
	id_batch = []
	vids = []
	word_batch = []
	for line in test_data:
		if not line['video_id'] in vids:
			caption_batch.append(replace_unknown_words(vocab,line['caption'].strip().split()))

			with h5py.File(source, 'r') as f:
				feat = xp.array(f[line['video_id']], dtype=xp.float32)
			with h5py.File(words, 'r') as w:
				word = np.array(w[line['video_id']])

			vid_batch.append(feat)
			word_batch.append(create_word_pool(word, vocab))
			id_batch.append(line['video_id'])
			vids.append(line['video_id'])
			
			if(len(caption_batch) == batch_size):
				yield vid_batch,caption_batch,id_batch, word_batch
				caption_batch = []
				vid_batch = []
				id_batch = []
				word_batch = []

	if not len(vid_batch) == 0:
		yield vid_batch,caption_batch,id_batch, word_batch

def gen_vocab_dict(train_data, mapping):
	"""
	generate dict from training set's caption file (txt)
	format of txt file:
	id1 video_id1 caption1 
	id2 video_id2 caption2
	...

	"""
	embeddings =[]
	vocab = {}
	inv_vocab = {}
	vocab[';'] = len(vocab)
	inv_vocab[len(vocab)-1] = ';'
	embeddings.append(utils_glove.read_word_vector(';', mapping))
	for line in train_data:
		words = line['caption'].strip().split()		
		for word in words:
			if word not in vocab:
				vocab[word] = len(vocab)
				inv_vocab[len(vocab)-1]=word
				embeddings.append(utils_glove.read_word_vector(word, mapping))
		#print words	
	vocab[':'] = len(vocab)
	inv_vocab[len(vocab)-1] = ':'
	embeddings.append(utils_glove.read_word_vector(':', mapping))
	vocab['.'] = len(vocab)
	inv_vocab[len(vocab)-1] = '.'
	embeddings.append(utils_glove.read_word_vector('.', mapping))
	vocab['~'] = len(vocab)
	inv_vocab[len(vocab)-1] = '~'
	embeddings.append(utils_glove.read_word_vector('~', mapping))
	vocab_size = len(vocab)
	
	#with open('vocab_natsuda.json', 'w') as f:
		#json.dump(vocab , f)
	#print 'embdding shape:', embeddings.shape
	return vocab, inv_vocab, embeddings

def read_vocab(vocab_file, mapping):
	embeddings = []
	vocab = {}
	inv_vocab = {}
	words = json.load(open(vocab_file, 'r'))
	vocab[';'] = len(vocab)
	inv_vocab[len(vocab)-1] = ';'
	embeddings.append(utils_glove.read_word_vector(';', mapping))
	for word in words:
		if word not in vocab:
			vocab[word] = len(vocab)
			inv_vocab[len(vocab)-1]=word
			embeddings.append(utils_glove.read_word_vector(word, mapping))
		#print words	
	#print words	
	vocab[':'] = len(vocab)
	inv_vocab[len(vocab)-1] = ':'
	embeddings.append(utils_glove.read_word_vector(':', mapping))
	vocab['.'] = len(vocab)
	inv_vocab[len(vocab)-1] = '.'
	embeddings.append(utils_glove.read_word_vector('.', mapping))
	vocab['~'] = len(vocab)
	inv_vocab[len(vocab)-1] = '~'
	embeddings.append(utils_glove.read_word_vector('~', mapping))
	vocab_size = len(vocab)	

	#print 'embedding shape:', embeddings.shape
	return vocab, inv_vocab, embeddings
		
def read_file(filename, xp):
	lines = open(filename).readlines()
	lines = [map(float, line.strip().split()) for line in lines]
	lines = xp.asarray(lines, dtype=xp.float32)
	return lines

def create_word_pool(word, vocab, num_concepts=20):	
	embeddings_pool = [vocab['~']]*num_concepts
	functional_words = ['a', 'on', 'of', 'the', 'in', 'with', 'and', 'is', 'to', 'an', 'two', 'at', 'next', 'are']
	k = 0
	for i in range(len(word)):
		for j in range(len(word[0])):	
			if (vocab[word[i][j]] not in embeddings_pool) and (word[i][j] not in functional_words):
				#print word[i][j]
				if k == num_concepts:
					return embeddings_pool
				embeddings_pool[k] = vocab[word[i][j]]
				k += 1
	return embeddings_pool

def add_concepts_to_dict(h5_file, mapping, vocab, inv_vocab, embeddings):
	with h5py.File(h5_file, 'r') as ws:
		for word in ws:

			for i in range(len(ws[word])):
				print ws[word][i]
				for j in range(len(ws[word][i])):

					if ws[word][i][j] not in vocab:
						embeddings.append(utils_glove.read_word_vector(ws[word][i][j], mapping))
						vocab[ws[word][i][j]] = len(vocab)
						inv_vocab[len(vocab)-1]=ws[word][i][j]
						#print ws[word][i][j]
	return vocab, inv_vocab, embeddings

