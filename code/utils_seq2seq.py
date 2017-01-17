import json
import h5py
import random
random.seed(4)


def replace_unknown_words(vocab, caption):
    for i in xrange(len(caption)):
        if not caption[i] in vocab:
            caption[i] = 'unk'
    return caption


def gen_batch(train_data, source, batch_size, vocab, xp):
    vid_batch = []
    caption_batch = []
    random.shuffle(train_data)
    for line in train_data:
        caption_batch.append(replace_unknown_words(vocab, line['caption'].strip().split()))

        with h5py.File(source, 'r') as f:
            feat = xp.array(f[line['video_id']], dtype=xp.float32)

        vid_batch.append(feat)

        if(len(caption_batch) == batch_size):
            yield vid_batch, caption_batch
            caption_batch = []
            vid_batch = []


def gen_batch_val(val_data, source, batch_size, vocab, xp):
    vid_batch = []
    caption_batch = []
    id_batch = []
    for line in val_data:
        caption_batch.append(replace_unknown_words(vocab, line['caption'].strip().split()))

        with h5py.File(source, 'r') as f:
            feat = xp.array(f[line['video_id']], dtype=xp.float32)

        vid_batch.append(feat)
        id_batch.append(line['video_id'])

        if(len(caption_batch) == batch_size):
            yield vid_batch, caption_batch, id_batch
            caption_batch = []
            vid_batch = []
            id_batch = []

    if not len(vid_batch) == 0:
        yield vid_batch, caption_batch, id_batch


def gen_batch_test(test_data, source, batch_size, vocab, xp):
    vid_batch = []
    caption_batch = []
    id_batch = []
    vids = []
    for line in test_data:
        if not line['video_id'] in vids:
            caption_batch.append(replace_unknown_words(vocab, line['caption'].strip().split()))

            with h5py.File(source, 'r') as f:
                feat = xp.array(f[line['video_id']], dtype=xp.float32)

            vid_batch.append(feat)
            id_batch.append(line['video_id'])
            vids.append(line['video_id'])

            if(len(caption_batch) == batch_size):
                yield vid_batch, caption_batch, id_batch
                caption_batch = []
                vid_batch = []
                id_batch = []

    if not len(vid_batch) == 0:
        yield vid_batch, caption_batch, id_batch


def gen_vocab_dict(train_data):
    """
    generate dict from training set's caption file (txt)
    format of txt file:
    id1 video_id1 caption1
    id2 video_id2 caption2
    ...

    """
    vocab = {}
    inv_vocab = {}
    vocab[''] = len(vocab)
    inv_vocab[len(vocab) - 1] = ''
    for line in train_data:
        words = line['caption'].strip().split()
        for word in words:
            if word not in vocab:
                vocab[word] = len(vocab)
                inv_vocab[len(vocab) - 1] = word

    vocab['BOS'] = len(vocab)
    inv_vocab[len(vocab) - 1] = 'BOS'
    vocab['EOS'] = len(vocab)
    inv_vocab[len(vocab) - 1] = 'EOS'
    # vocab_size = len(vocab)
    # with open('vocab_natsuda.json', 'w') as f:
    #     json.dump(vocab , f)
    return vocab, inv_vocab


def read_vocab(vocab_file):
    vocab = {}
    inv_vocab = {}
    words = json.load(open(vocab_file, 'r'))
    vocab[''] = len(vocab)
    inv_vocab[len(vocab) - 1] = ''

    for word in words:
        if word not in vocab:
            vocab[word] = len(vocab)
            inv_vocab[len(vocab) - 1] = word

    vocab['BOS'] = len(vocab)
    inv_vocab[len(vocab) - 1] = 'BOS'
    vocab['EOS'] = len(vocab)
    inv_vocab[len(vocab) - 1] = 'EOS'
    # vocab_size = len(vocab)

    return vocab, inv_vocab


def read_file(filename, xp):
    lines = open(filename).readlines()
    lines = [map(float, line.strip().split()) for line in lines]
    lines = xp.asarray(lines, dtype=xp.float32)
    return lines
