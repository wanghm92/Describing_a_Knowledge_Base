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

class ConfigRotowire(object):
    def __init__(self):
        self.cell = "LSTM"
        self.optimizer = "adam"
        self.decay_rate = 0.9
        self.decay_start = 1
        self.emsize = 256
        self.fdsize = 128
        self.rcdsize = 64
        self.hasize = 64
        self.hdsize = 512
        self.pemsize = 5
        self.nlayers = 1
        self.lr = 0.0002
        self.epochs = 30
        self.batch_size = 4
        self.valid_batch = 64
        self.dropout = 0.5
        self.directions = 1
        self.max_grad_norm = 5
        self.max_len = 80
        self.min_len = 20
        self.unk_gen = True  # allow generating UNK

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
