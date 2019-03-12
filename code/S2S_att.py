import chainer

from chainer.links import Linear, Bilinear, EmbedID
from chainer.functions import array, batch_matmul, reshape
from chainer.functions.activation.lstm import lstm
from chainer.functions.activation.tanh import tanh
from chainer.functions.math.exponential import exp
from chainer.functions.noise.dropout import dropout


class Encoder(chainer.Chain):

    def __init__(self, input_size, vocab_size, embed_size, hidden_size):
        super(Encoder, self).__init__(
            xi1=Linear(input_size, embed_size),
            ih1=Linear(embed_size, 4 * hidden_size),
            hh1=Linear(hidden_size, 4 * hidden_size),
            xi2=EmbedID(vocab_size, embed_size),
            ih2=Linear(hidden_size + embed_size, 4 * hidden_size),
            hh2=Linear(hidden_size, 4 * hidden_size)
        )

    def __call__(self, frame, prev_word, state, dropout_flag, dropout_ratio):
        i1 = self.xi1(dropout(frame, dropout_ratio))
        c1, h1 = lstm(state['c1'], self.ih1(i1) + self.hh1(state['h1']))
        i2 = self.xi2(prev_word)
        concat = array.concat.concat((i2, h1))
        c2, h2 = lstm(state['c2'], self.ih2(concat) + self.hh2(state['h2']))
        state = {'c1': c1, 'h1': h1, 'c2': c2, 'h2': h2}
        return state


class Decoder(chainer.Chain):

    def __init__(self, input_size, vocab_size, embed_size, hidden_size):
        super(Decoder, self).__init__(
            xi1=Linear(input_size, embed_size),
            ih1=Linear(embed_size, 4 * hidden_size),
            hh1=Linear(hidden_size, 4 * hidden_size),
            xi2=EmbedID(vocab_size, embed_size),
            ih2=Linear(hidden_size + embed_size, 4 * hidden_size),
            ah2=Linear(hidden_size, 4 * hidden_size),
            hh2=Linear(hidden_size, 4 * hidden_size),
            hy=Linear(hidden_size, vocab_size),
        )

    def decode_att(self, frame, prev_word, state, context):
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


class Attention_dot(chainer.Chain):

    def __init__(self, hidden_size):
        super(Attention_dot, self).__init__()
        self.hidden_size = hidden_size

    def __call__(self, a_list, state, batch_size, xp):
        e_list = []
        sum_e = xp.zeros((batch_size, 1), dtype=xp.float32)
        for a in a_list:
            w = reshape(batch_matmul(state['h2'], a, transa=True), (batch_size, 1))
            w.data = xp.clip(w.data, -40, 40)
            e = exp(w)
            e_list.append(e)
            sum_e = sum_e + e

        context = xp.zeros((batch_size, self.hidden_size), dtype=xp.float32)

        for a, e in zip(a_list, e_list):
            e /= sum_e
            context = context + reshape(batch_matmul(a, e), (batch_size, self.hidden_size))
        return context, e_list, sum_e


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
            w = self.aw(a, state['h2'])
            w.data = xp.clip(w.data, -20, 20)
            e = exp(w)
            e_list.append(e)
            sum_e = sum_e + e

        context = xp.zeros((batch_size, self.hidden_size), dtype=xp.float32)
        for a, e in zip(a_list, e_list):
            e /= sum_e
            context = context + reshape(batch_matmul(a, e), (batch_size, self.hidden_size))
        return context, e_list, sum_e


class Attention_concat(chainer.Chain):

    def __init__(self, hidden_size):
        super(Attention_concat, self).__init__(
            av=Linear(2 * hidden_size, 2 * hidden_size),
            vw=Linear(2 * hidden_size, 1),
        )
        self.hidden_size = hidden_size

    def __call__(self, a_list, state, batch_size, xp):
        e_list = []
        sum_e = xp.zeros((batch_size, 1), dtype=xp.float32)
        for a in a_list:
            v = tanh(self.av(array.concat.concat((a, state['h2']), axis=1)))
            w = self.vw(v)
            e = exp(w)
            e_list.append(e)
            sum_e = sum_e + e

        context = xp.zeros((batch_size, self.hidden_size), dtype=xp.float32)
        for a, e in zip(a_list, e_list):
            e /= sum_e
            context = context + reshape(batch_matmul(a, e), (batch_size, self.hidden_size))
        return context, e_list, sum_e


class S2S_att(chainer.Chain):

    def __init__(self, input_size, vocab_size, embed_size, hidden_size, align):
        self.a_list = []
        self.align = align
        # self.input_size = input_size
        # self.hidden_size = hidden_size
        if align == 'dot':
            super(S2S_att, self).__init__(
                enc=Encoder(input_size, vocab_size, embed_size, hidden_size),
                att=Attention_dot(hidden_size),
                dec=Decoder(input_size, vocab_size, embed_size, hidden_size),
            )
        elif align == 'bilinear':
            super(S2S_att, self).__init__(
                enc=Encoder(input_size, vocab_size, embed_size, hidden_size),
                att=Attention_bilinear(hidden_size),
                dec=Decoder(input_size, vocab_size, embed_size, hidden_size),
            )
        elif align == 'concat':
            super(S2S_att, self).__init__(
                enc=Encoder(input_size, vocab_size, embed_size, hidden_size),
                att=Attention_concat(hidden_size),
                dec=Decoder(input_size, vocab_size, embed_size, hidden_size),
            )
        elif align == 'none':
            super(S2S_att, self).__init__(
                enc=Encoder(input_size, vocab_size, embed_size, hidden_size),
                dec=Decoder(input_size, vocab_size, embed_size, hidden_size),
            )

    def reset(self):
        self.zerograds()
        self.a_list = []

    def encode(self, frame, prev_word, state, dropout_flag=False, dropout_ratio=0.0):
        state = self.enc(frame, prev_word, state, dropout_flag, dropout_ratio)
        self.a_list.append(state['h1'])
        return state

    def decode(self, frame, prev_word, state, batch_size, xp):
        if self.align == 'none':
            y, state = self.dec.decode_nonatt(frame, prev_word, state)
            return y, state
        else:
            context, e_list, sum_e = self.att(self.a_list, state, batch_size, xp)
            y, state = self.dec.decode_att(frame, prev_word, state, context)
            return y, state

    def decode_and_print_weight(self, frame, prev_word, state, batch_size, xp):
        context, e_list, sum_e = self.att(self.a_list, state, batch_size, xp)
        y, state = self.dec.decode_att(frame, prev_word, state, context)
        return y, state, e_list, sum_e
