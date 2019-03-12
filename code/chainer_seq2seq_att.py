from __future__ import print_function

import time
import os
import numpy as np
import math as mth
from copy import deepcopy
import json
import h5py

from argparse import ArgumentParser

import chainer
from chainer import cuda, optimizers, serializers
from chainer.functions.activation.softmax import softmax
from chainer.functions.loss.softmax_cross_entropy import softmax_cross_entropy

import eval_coco
import S2S_att
import utils_seq2seq
import utils_coco
import utils_y2t as utils_y2t
from seq2seq_params import Params

try:
    xrange
except NameError:
    xrange = range


utils_corpus = utils_y2t


class _DefaultPaths:

    def __init__(self):

        data_dir = os.path.join('..', 'data', 'Y2T')

        self.feature = os.path.join(data_dir, 'y2t_vgg16.h5')
        self.train = os.path.join(data_dir, 'sents_train_lc_nopunc.txt')
        self.val = os.path.join(data_dir, 'sents_val_lc_nopunc.txt')
        self.test = os.path.join(data_dir, 'sents_test_lc_nopunc.txt')
        self.vocab = os.path.join(data_dir, 'vocab.json')
        self.coco_val_ref = os.path.join(data_dir, 'val_coco_y2t.json')
        self.coco_test_ref = os.path.join(data_dir, 'test_coco_y2t.json')


def parse_args():

    default_paths = _DefaultPaths()
    default_params = Params()

    parser = ArgumentParser()
    parser.add_argument('output',
                        help='[out] output folder')
    parser.add_argument('--mode',
                        help='[mode] train, test, test-batch')
    parser.add_argument('--feature',
                        default=default_paths.feature,
                        help='[in] features of movie frame features')
    parser.add_argument('--train',
                        default=default_paths.train,
                        help='[in] a file containing file names and gold sentences of training data')
    parser.add_argument('--val',
                        default=default_paths.val,
                        help='[in] a file containing file namesand gold sentences of validation data')
    parser.add_argument('--test',
                        default=default_paths.test,
                        help='[in] a file containg file names and gold sentences of test data')
    parser.add_argument('--vocab',
                        default=default_paths.vocab,
                        help='[in] vocab file name (json)')
    parser.add_argument('--cocoval',
                        default=default_paths.coco_val_ref,
                        help='[in] coco ref file name (val)')
    parser.add_argument('--cocotest',
                        default=default_paths.coco_test_ref,
                        help='[in] coco ref file name (test)')
    parser.add_argument('--align',
                        default='dot',
                        help='[alignment function] dot, bilinear, concat')
    parser.add_argument('--model',
                        help='[in] model file for test mode')
    parser.add_argument('--epoch',
                        default=default_params.epoch,
                        metavar='INT',
                        type=int,
                        help='number of training epoch (default: %(default)d)')
    parser.add_argument('--batchsize',
                        default=default_params.batch_size,
                        metavar='INT',
                        type=int,
                        help='mini batch size (default: %(default)d)')
    parser.add_argument('--gpu',
                        default=default_params.gpu,
                        metavar='INT',
                        type=int,
                        help='gpu device number (default: %(default)d)')
    parser.add_argument('--beamsize',
                        default=default_params.beam_size,
                        metavar='INT',
                        type=int)
    parser.add_argument('--store',
                        default=default_params.store_state,
                        metavar='INT',
                        type=int)
    parser.add_argument('--initmodel',
                        '-m',
                        default='',
                        help='Initialize the model from given file')
    parser.add_argument('--resume',
                        '-r',
                        default='',
                        help='Resume the optimization from snapshot')
    parser.add_argument('--lr',
                        default=default_params.lr,
                        type=float,
                        help='Learning rate')
    parser.add_argument('--dropout',
                        default=None,
                        type=float,
                        help='dropout ratio')
    parser.add_argument('--inputsize',
                        default=default_params.input_size,
                        type=int,
                        help='size of input vector')
    parser.add_argument('--hiddensize',
                        default=default_params.hidden_size,
                        type=int,
                        help='size of hidden unit')
    parser.add_argument('--embedsize',
                        default=default_params.embed_size,
                        type=int,
                        help='size of embedded vector')
    parser.add_argument('--batchsizeval',
                        default=default_params.batch_size_val,
                        type=int,
                        help='batch size for evaluation (validation set)')
    parser.add_argument('--numfoldenc',
                        default=default_params.numfold_enc,
                        type=int,
                        help='number of LSTM units in encoding stage')
    parser.add_argument('--numfolddec',
                        default=default_params.numfold_dec,
                        type=int,
                        help='number of LSTM units in decoding stage')
    parser.add_argument('--weightdecay',
                        default=None,
                        type=float,
                        help='weight decay rate')
    parser.add_argument('--gradclip',
                        default=None,
                        type=float,
                        help='Gradient threshold to (hard)clip')

    args = parser.parse_args()

    return args


def fill_batch_to_fixed_size(params, vocab, src, target):

    batch_size = len(src)
    src_new = xp.zeros((batch_size, params.numfold_enc, params.input_size),
                       dtype=xp.float32)
    target_new = xp.zeros((batch_size, params.numfold_dec), dtype=xp.int32)
    inter = xp.zeros((batch_size, params.numfold_dec), dtype=xp.int32)

    for i in range(batch_size):
        numpad_enc = params.numfold_enc - len(src[i])
        src_len = len(src[i])
        if numpad_enc < 0:
            src_len = src_len + numpad_enc
            numpad_enc = 0
        for j in range(src_len):
            src_new[i][j + numpad_enc] = src[i][j]

        numpad_dec = params.numfold_dec - len(target[i]) - 1
        length = len(target[i])
        if numpad_dec >= 0:
            inter[i][0] = vocab['BOS']

            for j in range(length):
                inter[i][1 + j] = vocab[target[i][j]]
                target_new[i][j] = vocab[target[i][j]]

            target_new[i][length] = vocab['EOS']
            for j in range(numpad_dec):
                inter[i][1 + length + j] = vocab['']
                target_new[i][1 + length + j] = vocab['']
        else:
            inter[i][0] = vocab['BOS']
            for j in range(length + numpad_dec):
                inter[i][1 + j] = vocab[target[i][j]]
                target_new[i][j] = vocab[target[i][j]]

            target_new[i][length + numpad_dec] = \
                vocab[target[i][params.numfold_dec - 1]]

    return src_new, inter, target_new


def initial_state(xp, batch_size, hidden_size):

    c1 = chainer.Variable(xp.zeros((batch_size, hidden_size), dtype=xp.float32))
    c2 = chainer.Variable(xp.zeros((batch_size, hidden_size), dtype=xp.float32))
    h1 = chainer.Variable(xp.zeros((batch_size, hidden_size), dtype=xp.float32))
    h2 = chainer.Variable(xp.zeros((batch_size, hidden_size), dtype=xp.float32))

    return {'c1': c1, 'h1': h1, 'c2': c2, 'h2': h2}


def predict(model, params, vocab, inv_vocab, src, batch_size, beam_size=1):

    state = initial_state(xp, batch_size, params.hidden_size)

    model.reset()

    for n in range(len(src[0])):
        xb = chainer.Variable(src[0][n].reshape((1, params.input_size)))
        ib = chainer.Variable(xp.array([vocab['']], dtype=xp.int32))
        state = model.encode(xb, ib, state)

    final_sentences = [([], None, None) for i in xrange(beam_size)]
    sentence_candidate = [([], None, None) for i in xrange(beam_size)]
    sentence_candidate_tmp = [([], None, None) for i in xrange(beam_size * beam_size)]
    success = 0
    depth = 0

    sentence_candidate[0] = (['BOS'], state, 0)
    k = 1

    while success < beam_size and depth < 20:
        frame = chainer.Variable(xp.zeros((1, params.input_size), dtype=xp.float32))
        j = 0
        for i in xrange(k):
            sentence_tuple = sentence_candidate[i]
            cur_sentence = sentence_tuple[0]
            cur_index = sentence_tuple[0][-1]
            cur_state = sentence_tuple[1]
            cur_log_likely = sentence_tuple[2]

            prev_word = chainer.Variable(xp.array([vocab[cur_index]], dtype=xp.int32))
            y, state = model.decode(frame, prev_word, cur_state, batch_size, xp)
            y_np = cuda.to_cpu(softmax(y).data)
            top_indexes = (y_np[0]).argsort(0)[::-1][:beam_size]
            for index in np.nditer(top_indexes):
                index = int(index)
                probability = y_np[0][index]
                next_sentence = deepcopy(cur_sentence)
                next_sentence.append(inv_vocab[index])
                log_likely = mth.log(probability)
                next_log_likely = cur_log_likely + log_likely
                sentence_candidate_tmp[j] = (next_sentence, state, next_log_likely)
                j += 1

        prob_np_array = np.array([sentence_tuple[2] for sentence_tuple in sentence_candidate_tmp[:j]])
        top_candidates_indexes = (prob_np_array).argsort()[::-1][:beam_size]
        k = 0
        for i in top_candidates_indexes:
            sentence_tuple = sentence_candidate_tmp[i]
            word = sentence_tuple[0][-1]
            if word == 'EOS':
                final_sentence = sentence_tuple[0]
                final_likely = sentence_tuple[2]
                final_probability = mth.exp(final_likely / len(final_sentence))
                final_sentences[success] = (final_sentence, final_probability, final_likely)
                success += 1
                if success == beam_size:
                    break
            else:
                sentence_candidate[k] = sentence_tuple
                k += 1

        depth += 1

    candidates = []
    for sentence_tuple in final_sentences:
        sentence = [word for word in sentence_tuple[0]][1: -1]
        final_probability = sentence_tuple[1]
        a_candidate = {'sentence': sentence, 'probability': final_probability}
        candidates.append(a_candidate)
    scores = [caption['probability'] for caption in candidates]
    argmax = np.argmax(scores)
    top_caption = candidates[argmax]['sentence']

    sentence = ''
    for word in top_caption:
        sentence += word + ' '

    return sentence.strip()


def forward(model, params, vocab, inv_vocab, src, target, mode, batch_size):

    state = initial_state(xp, batch_size, params.hidden_size)

    model.reset()

    src_b, inter_b, target_b = fill_batch_to_fixed_size(params, vocab, src, target)
    loss = xp.zeros((), dtype=xp.float32)
    dropout_ratio = params.dropout if isinstance(params.dropout, float) else 0.0
    if mode == 'training':
        chainer.global_config['train'] = True
    else:
        chainer.global_config['train'] = False
    for n in range(params.numfold_enc):
        xb = chainer.Variable(src_b[:, n])
        ib = chainer.Variable(xp.zeros((batch_size), dtype=xp.int32))

        if mode == 'training':
            state = model.encode(xb, ib, state,
                                 dropout_flag=isinstance(dropout_ratio, float),
                                 dropout_ratio=dropout_ratio)
        else:
            state = model.encode(xb, ib, state)

    if mode == 'training' or mode == 'validating':

        for n in range(params.numfold_dec):
            xb = chainer.Variable(xp.zeros((batch_size, params.input_size), dtype=xp.float32))
            ib = chainer.Variable(xp.array([inter_b[i][n] for i in range(batch_size)], dtype=xp.int32))
            y, state = model.decode(xb, ib, state, batch_size, xp)
            t = chainer.Variable(xp.array([target_b[i][n] for i in range(batch_size)], dtype=xp.int32))
            loss = loss + softmax_cross_entropy(y, t)
        return loss

    # mode == 'test-on-train':
    else:

        words = [vocab['BOS']] * batch_size
        output = [''] * batch_size
        for n in range(params.numfold_dec):
            xb = chainer.Variable(xp.zeros((batch_size, params.input_size), dtype=xp.float32))
            ib = chainer.Variable(xp.array(words, dtype=xp.int32))
            y, state = model.decode(xb, ib, state, batch_size, xp)
            words = y.data.argmax(1)
            for i in range(batch_size):
                if output[i].endswith('EOS'):
                    continue
                output[i] = output[i] + ' ' + inv_vocab[int(words[i])]

        for i in range(batch_size):
            output[i] = output[i].replace('EOS', '').strip()
        return output


def get_current_score(model, vocab, inv_vocab, val_data, source, batch_size_val, eval_filename):
    batch_test = utils_seq2seq.gen_batch_test(val_data, source, batch_size_val, vocab, xp)
    caption_out = []
    for vid_batch_test, caption_batch_test, id_batch_test in batch_test:
        batch_size_val2 = len(vid_batch_test)
        output_test = forward(model, params, vocab, inv_vocab, vid_batch_test, caption_batch_test, 'test-on-train', batch_size_val2)
        for ii in range(batch_size_val2):
            caption_out.append({'image_id': id_batch_test[ii], 'caption': output_test[ii]})

    with open(eval_filename, mode='w') as f:
        json.dump(caption_out, f)
    score = eval_coco.eval_coco(coco_ref_filename, eval_filename)
    return score


def get_accum_loss(model, vocab, inv_vocab, params, val_data, source, batch_size_val):
    accum_loss_val = 0
    batch_index_val = 0
    batch_val = utils_seq2seq.gen_batch_val(val_data, source, batch_size_val, vocab, xp)
    for vid_batch_val, caption_batch_val, id_batch_val in batch_val:
        batch_size_val2 = len(vid_batch_val)
        loss_val = forward(model, params, vocab, inv_vocab, vid_batch_val, caption_batch_val, 'validating', batch_size_val2)
        accum_loss_val += loss_val.data * batch_size_val2
        batch_index_val += batch_size_val2
    return accum_loss_val / batch_index_val


def train(model, train_data, val_data, vocab, inv_vocab, params, score_json):

    opt = optimizers.Adam(alpha=args.lr)
    opt.setup(model)
    if isinstance(params.weight_decay, float):
        print('Add weight decay ({})'.format(params.weight_decay))
        opt.add_hook(chainer.optimizer.WeightDecay(params.weight_decay))
    if isinstance(params.grad_clip, float):
        print('Add hard gradient clip ({})'.format(params.grad_clip))
        opt.add_hook(chainer.optimizer.GradientHardClipping(-1.0 * params.grad_clip,
                                                            params.grad_clip))

    print('minibatch size: %d %d' % (args.batchsize, args.batchsizeval))
    if args.resume:
        print('Load optimizer state from {}'.format(args.resume))
        serializers.load_hdf5(args.resume, opt)
    if args.initmodel:
        print('Load model from {}'.format(args.initmodel))
        serializers.load_hdf5(args.initmodel, model)
        print('Testing: model ', args.initmodel)
        eval_filename = eval_filename_prefix + '_pretrained'
        score = get_current_score(model, vocab, inv_vocab, val_data, args.feature, args.batchsizeval, eval_filename)
        score_json.append({'model': model_path,
                           'lr': args.lr,
                           'iter': -1,
                           'epoch': -1,
                           'score': score})
        with open(score_filename, mode='w') as f:
            json.dump(score_json, f)
        print(score)

    iternum = 0
    max_score = 0
    for i in range(args.epoch):
        start_time = time.time()
        print('Training: epoch#', i)
        batch = utils_seq2seq.gen_batch(train_data, args.feature, args.batchsize, vocab, xp)
        batch_index = 0
        accum_loss = 0

        prefix = model_path + '/%03.d' % i
        for vid_batch, caption_batch in batch:
            print('Training: epoch# %d iteration# %d' % (i, batch_index))
            model.zerograds()
            loss = forward(model, params, vocab, inv_vocab, vid_batch, caption_batch, 'training', args.batchsize)
            loss.backward()
            opt.update()
            accum_loss += loss.data

            if (i > 1 and iternum % 100 == 0) or (iternum % 100 == 0):
                print('Testing: iteration#', iternum)
                eval_filename = eval_filename_prefix + str(iternum)
                score = get_current_score(model, vocab, inv_vocab, val_data, args.feature, args.batchsizeval, eval_filename)
                if score['total'] > max_score:
                    serializers.save_hdf5(prefix + '.model.iter' + str(iternum), model)
                    serializers.save_hdf5(prefix + '.opt.iter' + str(iternum), opt)
                    max_score = score['total']
                print(score, max_score)
                score_json.append({'model': model_path, 'lr': args.lr, 'iter': iternum, 'epoch': i, 'score': score, 'max_score': max_score})
                with open(score_filename, 'w') as f:
                    json.dump(score_json, f)

            batch_index += 1
            iternum += 1

        elapsed = time.time() - start_time
        accum_loss_val = get_accum_loss(model, vocab, inv_vocab, params, val_data, args.feature, args.batchsizeval)
        output_file = open(loss_filename, mode='a')
        output_file.write(str(i) + '\t' + str(elapsed) + '\t' + str(accum_loss / batch_index) + '\t' + str(accum_loss_val) + '\n')
        output_file.close()
        if i == 0 or (i + 1) % args.store == 0:
            serializers.save_hdf5(prefix + '.model', model)
            serializers.save_hdf5(prefix + '.opt', opt)


def test(model, test_data, vocab, inv_vocab, modelfile_to_load, params):

    print('Testing ...')
    print('Beam size: {}'.format(params.beam_size))
    print('print output to file:', out_test_filename)
    serializers.load_hdf5(modelfile_to_load, model)
    batch_test = utils_seq2seq.gen_batch_test(test_data, args.feature, 1, vocab, xp)
    output_file = open(out_test_filename, mode='w')
    for vid_batch, caption_batch, id_batch in batch_test:
        output = predict(model, params, vocab, inv_vocab, vid_batch,
                         batch_size=1, beam_size=params.beam_size)
        print('%s %s' % (id_batch[0], output))
        output_file.write(id_batch[0] + '\t' + output + '\n')
    output_file.close()
    utils_coco.convert(out_test_filename, eval_test_filename)
    eval_coco.eval_coco(args.cocotest, eval_test_filename)


def test_batch(model, test_data, vocab, inv_vocab, modelfile_to_load):

    print('Testing (beam size = 1)...')
    print('print output to file: {}'.format(out_test_filename))
    serializers.load_hdf5(modelfile_to_load, model)
    batch_test = \
        utils_seq2seq.gen_batch_test(test_data, args.feature, params.batch_size_val, vocab, xp)
    caption_out = []
    output_file = open(out_test_filename, mode='w')
    for vid_batch_test, caption_batch_test, id_batch_test in batch_test:
        output_test = forward(model, params, vocab, inv_vocab,
                              vid_batch_test, caption_batch_test,
                              'test-on-train', args.batchsizeval)
        for ii in range(args.batchsizeval):
            caption_out.append({'image_id': id_batch_test[ii],
                                'caption': output_test[ii]})
            print('%s %s' % (id_batch_test[ii], output_test[ii]))
            output_file.write(id_batch_test[ii] + '\t' + output_test[ii] + '\n')
    output_file.close()
    with open(eval_test_filename, mode='w') as f:
        json.dump(caption_out, f)
    eval_coco.eval_coco(args.cocotest, eval_test_filename)


def eval(args, params):

    train_data = utils_corpus.get_dataset(args.train, 'train')
    val_data = utils_corpus.get_dataset(args.val, 'val')
    test_data = utils_corpus.get_dataset(args.test, 'test')

    vocab, inv_vocab = utils_seq2seq.read_vocab(args.vocab)
    num_vocab = len(vocab)

    print('training data size: {}'.format(len(train_data)))
    print('validating data size: {}'.format(len(val_data)))
    print('test data size: {}'.format(len(test_data)))
    print('no. vocabs in training data: {}'.format(len(vocab)))

    model = S2S_att.S2S_att(params.input_size,
                            num_vocab,
                            params.embed_size,
                            params.hidden_size,
                            args.align)

    if params.gpu >= 0:
        model.to_gpu()

    score_json = [vars(args)]
    with open(score_filename, mode='w') as f:
        json.dump(score_json, f)

    if args.mode == 'train':
        train(model, train_data, val_data, vocab, inv_vocab, params, score_json)

    elif args.mode == 'test':
        test(model, test_data, vocab, inv_vocab,
             modelfile_to_load=args.model,
             params=params)

    elif args.mode == 'test-batch':
        test_batch(model, test_data, vocab, inv_vocab,
                   modelfile_to_load=args.model)

    else:
        modelfile_to_load = args.model
        serializers.load_hdf5(modelfile_to_load, model)

        with h5py.File(args.feature, mode='r') as f:
            features = []
            for k in f.keys():
                features.append(xp.array(f[k], dtype=xp.float32))

        output = predict(model, params, vocab, inv_vocab, features,
                         batch_size=1, beam_size=params.beam_size)

        print(output)


if __name__ == '__main__':

    args = parse_args()

    score_filename = args.output + '/score.json'
    loss_filename = args.output + '/loss_epoch.txt'
    out_test_filename = args.output + '/out_test.txt'
    eval_test_filename = out_test_filename + '.json'
    output_path = args.output + '/output/'
    eval_filename_prefix = output_path + '/out_iter'
    model_path = args.output + '/model/'
    coco_ref_filename = args.cocoval

    if not os.path.exists(args.output):
        os.makedirs(args.output)
    if not os.path.exists(model_path):
        os.makedirs(model_path)
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    if args.gpu >= 0:
        cuda.check_cuda_available()
    xp = cuda.cupy if args.gpu >= 0 else np
    if args.gpu >= 0:
        cuda.get_device(args.gpu).use()

    params = Params(embed_size=args.embedsize,
                    hidden_size=args.hiddensize,
                    input_size=args.inputsize,
                    epoch=args.epoch,
                    batch_size=args.batchsize,
                    beam_size=args.beamsize,
                    gpu=args.gpu,
                    store_state=args.store,
                    lr=args.lr,
                    dropout=args.dropout,
                    batch_size_val=args.batchsizeval,
                    numfold_enc=args.numfoldenc,
                    numfold_dec=args.numfolddec,
                    weight_decay=args.weightdecay,
                    grad_clip=args.gradclip)

    eval(args, params)
