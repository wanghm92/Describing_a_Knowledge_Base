class Config(object):
    def __init__(self):
        self.cell = "GRU"
        self.emsize = 256
        self.pemsize = 5
        self.nlayers = 1
        self.lr = 0.001
        self.epochs = 60
        self.batch_size = 64
        self.dropout = 0
        self.directions = 2
        self.max_grad_norm = 10
        self.max_len = 100
        self.unk_gen = False

class ConfigWikibio(object):
    def __init__(self):
        self.cell = "LSTM"
        self.emsize = 400
        self.fdsize = 50
        self.hdsize = 500
        # self.emsize = 256
        # self.fdsize = 64
        # self.hdsize = 256
        self.pemsize = 5
        self.nlayers = 1
        self.lr = 0.0003
        self.epochs = 60
        self.batch_size = 32
        self.dropout = 0
        self.directions = 1
        # self.directions = 2
        self.max_grad_norm = 5
        self.max_len = 100
        self.unk_gen = True  # allow generating UNK

class ConfigRotowire_new(object):
    def __init__(self):
        self.cell = "LSTM"
        self.optimizer = "adam"
        self.decay_rate = 0.95
        self.decay_start = 1
        self.emsize = 256
        self.fdsize = 128
        self.rcdsize = 64
        self.hasize = 64
        self.hdsize = 512
        self.attn_size = 0
        self.pemsize = 5
        self.nlayers = 1
        self.lr = 0.0002
        self.epochs = 30
        self.batch_size = 16
        self.valid_batch = 64
        self.dropout = 0.3
        self.directions = 2
        self.enc_otl_dir = 2
        self.max_grad_norm = 5
        self.max_len = 80
        self.max_sum_len = 500
        self.min_len = 35
        self.min_sum_len = 350
        self.unk_gen = True  # allow generating UNK
        self.scheduler = 'exp'
        self.tbptt = 100

class ConfigRotowire(object):
    def __init__(self):
        self.cell = "LSTM"
        self.optimizer = "adam"
        self.decay_rate = 0.95
        self.decay_start = 1
        self.emsize = 600
        self.fdsize = 600
        self.rcdsize = 600
        self.hasize = 600
        self.hdsize = 600
        self.attn_size = 0
        self.pemsize = 5
        self.nlayers = 1
        self.lr = 0.0002
        self.epochs = 30
        self.batch_size = 4
        self.valid_batch = 64
        self.dropout = 0.3
        self.directions = 2
        self.enc_otl_dir = 2
        self.max_grad_norm = 5
        self.max_len = 80
        self.max_sum_len = 500
        self.min_len = 35
        self.min_sum_len = 350
        self.unk_gen = True  # allow generating UNK
        self.scheduler = 'exp'
        self.tbptt = 100

class ConfigSmall(object):
    def __init__(self):
        self.cell = "GRU"
        self.emsize = 256
        self.pemsize = 5
        self.nlayers = 1
        self.lr = 0.001
        self.epochs = 60
        self.batch_size = 32
        self.dropout = 0
        self.directions = 2
        self.max_grad_norm = 10
        self.max_len = 100
        self.unk_gen = False

class ConfigTest(object):
    def __init__(self):
        self.cell = "LSTM"
        self.emsize = 30
        self.pemsize = 5
        self.nlayers = 1
        self.lr = 0.001
        self.epochs = 2
        self.batch_size = 3
        self.dropout = 0
        self.directions = 2
        self.max_grad_norm = 1
        self.testmode = True
        self.max_len = 50
        self.unk_gen = False
