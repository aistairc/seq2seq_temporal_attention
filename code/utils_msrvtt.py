"""
Read caption from a json file and output txt files
"""

import os
import json
import argparse
import string

def prepro_caption(sentence):
  #print sentence.encode('utf8')
  txt = str(sentence.encode('utf8')).lower().translate(None, string.punctuation)
  return txt
      
def get_dataset(input_json, split):
  info = json.load(open(input_json, 'r'))    
  videos = [v['video_id'] for v in info['videos'] if v['split'] == split]
  sentences = info['sentences']
  captions = [{'caption': prepro_caption(sentence['caption']), 'video_id': sentence['video_id']} for sentence in sentences if sentence['video_id'] in videos]

  return captions
    
if __name__ == "__main__":
  parser = argparse.ArgumentParser()
    
  # input json
  parser.add_argument('--input_json', default='data/msrvtt/train_val_videodatainfo.json', help='input json file to process into text file')
  parser.add_argument('--output_txt', default='data/msrvtt/train.txt', help='output txt file')
  parser.add_argument('--split', default='train', choices=['train','validate'])
  
  args = parser.parse_args()

  dataset = get_dataset(args.input_json, args.split)
  output_file = open(args.output_txt,'w')
  for i,v in enumerate(dataset):
    output_file.write(v['video_id'] + '\t' + v['caption'] +'\n')

