Sequence-to-sequence Model with Temporal Attention
==================================================

__seq2seq\_temporal\_attention__ is a tool for __automatic video captioning__.
This is an implementation of
*Generating Video Description using Sequence-to-sequence Model with Temporal Attention*
([PDF](http://aclweb.org/anthology/C16-1005)).

## Requirements (Linux/Mac)

+ [Python 2 or Python 3](https://www.python.org/)   
    To train a model, Python 2 is required.
+ [OpenCV](http://opencv.org/)  
    __Make sure that modules for video are included__.
    If you encounter an error while extracting frames,
    perhaps you can get helpful information from here:
    [OpenCV video capture from file fails on Linux](https://github.com/ContinuumIO/anaconda-issues/issues/121).
+ [Chainer](http://chainer.org/)  
+ [youtube-dl](https://github.com/rg3/youtube-dl/)

For requirements for Windows, read [docs/requirements-windows.md](docs/requirements-windows.md).

## Examples

### Captioning a video

To test out the tool, run `example.sh`.
It gives a caption for an excerpt of the video
titled [*playing wool ball with my cat : )*](https://www.youtube.com/watch?v=JyT9qPb5Fe0).
Our models were trained on [Microsoft Video Description Dataset](http://www.cs.utexas.edu/users/ml/papers/chen.acl11.pdf).

![cat video thumbnail](docs/images/example.jpg)

```bash
git clone git@github.com:aistairc/seq2seq_temporal_attention.git --recursive
./download.sh
./example.sh --gpu GPU_ID  # It will generate a caption *a cat is playing with a toy*
```

__Note__: In most cases, setting `GPU_ID` to `0` will work.
If you want to run it without GPU, set the parameter to `-1`.

### Training

This is an example command to train.

```bash
cd code
python chainer_seq2seq_att.py \
    --mode train \
    --gpu GPU_ID \
    --batchsize 40 \
    --dropout 0.3 \
    --align ('dot'|'bilinear'|'concat'|'none') \
    --feature feature_file_name \
    output_folder
```

### Test

There are two ways for test, `test` and `test-batch`.
The latter runs much faster, but it does not use beam search.
Be careful to specify which alignment model you want to use. It has to match your pre-trained model, in order to make it work correctly.

```bash
cd code
python chainer_seq2seq_att.py \
    --mode ('test'|'test-batch') \
    --gpu GPU_ID \
    --model path_to_model_file \
    --align ('dot'|'bilinear'|'concat'|'none') \
    --feature feature_file_name \
    output_folder
```
