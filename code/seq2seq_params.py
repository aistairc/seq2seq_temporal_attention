class Params:

    def __init__(self,
                 embed_size=512,
                 hidden_size=300,
                 input_size=4096,
                 epoch=50,
                 batch_size=40,
                 beam_size=20,
                 gpu=-1,
                 store_state=1,
                 lr=0.0001,
                 dropout=0.0,
                 batch_size_val=50,
                 numfold_enc=60,
                 numfold_dec=20,
                 weight_decay=None,
                 grad_clip=None):

        self.embed_size = embed_size
        self.hidden_size = hidden_size
        self.input_size = input_size
        self.epoch = epoch
        self.batch_size = batch_size
        self.gpu = gpu
        self.beam_size = beam_size
        self.store_state = store_state
        self.lr = lr
        self.dropout = dropout
        self.batch_size_val = batch_size_val
        self.numfold_enc = numfold_enc
        self.numfold_dec = numfold_dec
        self.weight_decay = weight_decay
        self.grad_clip = grad_clip
