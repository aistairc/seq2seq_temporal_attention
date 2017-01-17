#!/bin/bash

set -e

gpu_id="-1"
video_dir="data/mp4"
frame_dir="frames"
h5_filename="features"
align="dot"
model="data/${align}_att.model"
dest="result"
python_dir="code"
caffemodel="data/VGG_ILSVRC_16_layers.caffemodel"
mean="data/ilsvrc_2012_mean.npy"

error_style="\e[1;31m"
reset="\e[0;0m"

while [[ $# -gt 1 ]]
do
    key="$1"
    case $key in
        -g|--gpu)
        gpu_id="$2"
        shift
        ;;
        *)
        ;;
    esac
    shift
done

# Remove temporary files
rm -rf frames
for f in "frame_list.txt" "keys.txt" "features.h5"
do
    if [ -f "$f" ]
    then
        rm "$f"
    fi
done

if [ ! -d "$video_dir" ]
then
    error_message="Directory ${video_dir} does not exist."
    echo -e "${error_style}${error_message}${reset}"
    exit 1
fi

if [ "$gpu_id" = "-1" ]
then
    echo "Running on CPU"
else
    echo "Running on GPU ${gpu_id}"
fi

echo "Dividing each video into frames."
python "$python_dir/extract_frames.py" "$video_dir" "$frame_dir"

echo "Extracting features and save them into an h5 file."
python "$python_dir/chainer_extract_vgg.py" \
    --gpu "$gpu_id" \
    --batchsize 10 \
    --model "$caffemodel" \
    --mean "$mean" \
    "$frame_dir" "$h5_filename"

echo "Predict captions and write them in a text file."
python "$python_dir/chainer_seq2seq_att.py" \
    --batchsize 20 \
    --train "data/Y2T/sents_train_lc_nopunc.txt" \
    --val "data/Y2T/sents_val_lc_nopunc.txt" \
    --test "data/Y2T/sents_test_lc_nopunc.txt" \
    --vocab "data/Y2T/vocab.json" \
    --gpu "$gpu_id" \
    --mode "predict" \
    --align "$align" \
    --model "$model" \
    --feature "${h5_filename}.h5" \
    "$dest"
