#!/bin/bash

set -e

# Submodule
# coco-caption https://github.com/tylin/coco-caption
if [ -z "$(find coco-caption -type f)" ]
then
    git submodule init
    git submodule update
fi

cd "data"
# Mean image
# https://github.com/BVLC/caffe/blob/master/python/caffe/imagenet/ilsvrc_2012_mean.npy
mean='https://github.com/BVLC/caffe/raw/master/python/caffe/imagenet/ilsvrc_2012_mean.npy'
wget --no-clobber $mean

# VGG-16 pretrained caffe model
# https://gist.github.com/ksimonyan/211839e770f7b538e2d8
vgg='http://www.robots.ox.ac.uk/~vgg/software/very_deep/caffe/VGG_ILSVRC_16_layers.caffemodel'
wget --no-clobber $vgg

# Model
# Our models
# + `bilinear_att.model`
# + `concat_att.model`
# + `dot_att.model`
# + `none_att.model`
s3_dir="https://s3-ap-northeast-1.amazonaws.com/plu-aist/movie/Microsoft/model/temporal_attention"
for f in "bilinear" "concat" "dot" "none"
do
    wget --no-clobber "${s3_dir}/${f}_att.model"
done

# Venugopalan et al. (2015)
# http://www.cs.utexas.edu/users/ml/papers/venugopalan.naacl15.pdf
# https://www.cs.utexas.edu/~vsub/naacl15_project.html
# + `sents_train_lc_nopunc.txt`
# + `sents_val_lc_nopunc.txt`
# + `sents_test_lc_nopunc.txt`
venugopalan_et_al_2015='https://www.dropbox.com/sh/4ecwl7zdha60xqo/AAC_TAsR7SkEYhkSdAFKcBlMa?dl=1'
dest='naacl15_translating_videos_processed_data.zip'
wget --no-clobber $venugopalan_et_al_2015 -O $dest
if [ -f "Y2T/sents_test_lc_nopunc.txt" ]
then
    echo "$dest has been already extracted."
else
    unzip $dest -d "Y2T"
fi

# Video
# *playing wool ball with my cat : )* by ppsupp
# Available under the Creative Commons Attribution license
video='https://www.youtube.com/watch?v=JyT9qPb5Fe0'
video_dir="mp4"
if [ ! -d "$video_dir" ]
then
    mkdir "$video_dir"
fi
cd "$video_dir"
if [ ! -f "test.mp4" ]
then
    youtube-dl $video -f mp4 -o original.mp4
    # an example of extracting
    ffmpeg -ss 8 -t 5 -i original.mp4 "test.mp4"
    rm original.mp4
fi
