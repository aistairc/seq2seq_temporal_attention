import chainer

from chainer.links import Linear, Bilinear
from chainer.functions import array, batch_matmul
from chainer.functions.activation.lstm import lstm
from chainer.functions.activation.tanh import tanh
from chainer.functions.math.exponential import exp
from chainer.functions.noise.dropout import dropout


class EncoderDecoder(chainer.Chain):

    def __init__(self, input_size, vocab_size, embed_size, hidden_size, glove_embed_size):
        super(EncoderDecoder, self).__init__(
            xi1=Linear(input_size, embed_size),
            ih1=Linear(embed_size, 4 * hidden_size),
            hh1=Linear(hidden_size, 4 * hidden_size),
            xi2=Linear(glove_embed_size, embed_size),
            ih2=Linear(hidden_size + embed_size, 4 * hidden_size),
            ah2=Linear(hidden_size, 4 * hidden_size),
            hh2=Linear(hidden_size, 4 * hidden_size),
            hy=Linear(hidden_size, vocab_size),
        )

    def encode(self, frame, prev_word, state, dropout_flag, dropout_ratio):
        i1 = self.xi1(dropout(frame, dropout_ratio, dropout_flag))
        c1, h1 = lstm(state['c1'], self.ih1(i1) + self.hh1(state['h1']))
        i2 = self.xi2(prev_word)
        concat = array.concat.concat((i2, h1))
        c2, h2 = lstm(state['c2'], self.ih2(concat) + self.hh2(state['h2']))
        state = {'c1': c1, 'h1': h1, 'c2': c2, 'h2': h2}
        return state

    def decode(self, frame, prev_word, state, context):
        i1 = self.xi1(frame)
        c1, h1 = lstm(state['c1'], self.ih1(i1) + self.hh1(state['h1']))
        i2 = self.xi2(prev_word)
        concat = array.concat.concat((i2, h1))
        c2, h2 = lstm(state['c2'], self.ih2(concat) + self.hh2(state['h2']) + self.ah2(context))
        y = self.hy(h2)
        state = {'c1': c1, 'h1': h1, 'c2': c2, 'h2': h2}
        return y, state

    def decode_nonatt(self, frame, prev_word, state):
        i1 = self.xi1(frame)
        c1, h1 = lstm(state['c1'], self.ih1(i1) + self.hh1(state['h1']))
        i2 = self.xi2(prev_word)
        concat = array.concat.concat((i2, h1))
        c2, h2 = lstm(state['c2'], self.ih2(concat) + self.hh2(state['h2']))
        y = self.hy(h2)
        state = {'c1': c1, 'h1': h1, 'c2': c2, 'h2': h2}
        return y, state

    def arrange_concepts(self, concepts_batch, xp, alignment, embeddings):
        a_list = []
        num_concepts = len(concepts_batch[0])
        batch_size = len(concepts_batch)
        concepts_batch = xp.array(concepts_batch, dtype=xp.int32)
        if not alignment == 'none':
            for i in range(num_concepts):
                column = concepts_batch[:, i]
                column_emb = xp.take(embeddings, xp.reshape(column, (batch_size)), axis=0)
                a_list.append(self.xi2(chainer.Variable(column_emb)))
        return a_list


class Attention_dot(chainer.Chain):

    def __init__(self, hidden_size):
        super(Attention_dot, self).__init__()
        self.hidden_size = hidden_size

    def __call__(self, a_list, state, batch_size, xp):
        e_list = []
        sum_e = xp.zeros((batch_size, 1), dtype=xp.float32)
        for a in a_list:
            w = array.reshape(batch_matmul(state['h2'], a, transa=True), (batch_size, 1))
            e = exp(tanh(w))
            e_list.append(e)
            sum_e = sum_e + e

        context = xp.zeros((batch_size, self.hidden_size), dtype=xp.float32)

        for a, e in zip(a_list, e_list):
            e /= sum_e
            context = context + array.reshape(batch_matmul(a, e), (batch_size, self.hidden_size))
        return context


class Attention_bilinear(chainer.Chain):

    def __init__(self, hidden_size):
        super(Attention_bilinear, self).__init__(
            aw=Bilinear(hidden_size, hidden_size, 1),
        )
        self.hidden_size = hidden_size

    def __call__(self, a_list, state, batch_size, xp):
        e_list = []
        sum_e = xp.zeros((batch_size, 1), dtype=xp.float32)
        for a in a_list:
            w = tanh(self.aw(a, state['h2']))
            e = exp(w)
            e_list.append(e)
            sum_e = sum_e + e

        context = xp.zeros((batch_size, self.hidden_size), dtype=xp.float32)
        for a, e in zip(a_list, e_list):
            e /= sum_e
            context = context + array.reshape(batch_matmul(a, e), (batch_size, self.hidden_size))
        return context


class S2S_att(chainer.Chain):

    def __init__(self, input_size, vocab_size, embed_size, hidden_size, align, embeddings):
        glove_embed_size = 300
        self.a_list = []
        self.alignment = align
        self.embeddings = embeddings
        self.input_size = input_size
        self.hidden_size = hidden_size
        if align == 'dot':
            super(S2S_att, self).__init__(
                encdec=EncoderDecoder(input_size, vocab_size, embed_size, hidden_size, glove_embed_size),
                att=Attention_dot(hidden_size),
            )
        elif align == 'bilinear':
            super(S2S_att, self).__init__(
                encdec=EncoderDecoder(input_size, vocab_size, embed_size, hidden_size, glove_embed_size),
                att=Attention_bilinear(hidden_size),
            )
        elif align == 'none':
            super(S2S_att, self).__init__(
                encdec=EncoderDecoder(input_size, vocab_size, embed_size, hidden_size, glove_embed_size),
            )

    def reset(self):
        self.zerograds()
        self.a_list = []

    def encode(self, frame, prev_word, state, dropout_flag=False, dropout_ratio=0.0):
        state = self.encdec.encode(frame, prev_word, state, dropout_flag, dropout_ratio)
        return state

    def decode(self, frame, prev_word, state, batch_size, xp):
        if self.alignment == 'none':
            y, state = self.encdec.decode_nonatt(frame, prev_word, state)
        else:
            context = self.att(self.a_list, state, batch_size, xp)
            y, state = self.encdec.decode(frame, prev_word, state, context)
        return y, state

    def arrange_concepts(self, concepts_batch, xp):
        self.a_list = self.encdec.arrange_concepts(concepts_batch, xp, self.alignment, self.embeddings)
