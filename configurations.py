class Config(object):
    cell = "GRU"
    emsize = 256
    pemsize = 5
    nlayers = 1
    lr = 0.001
    epochs = 60
    batch_size = 64
    dropout = 0
    directions = 2
    max_grad_norm = 10
    max_len = 100
    unk_gen = False

class ConfigWikibio(object):
    cell = "LSTM"
    emsize = 400
    fdsize = 50
    hdsize = 500
    # emsize = 256
    # fdsize = 64
    # hdsize = 256
    pemsize = 5
    nlayers = 1
    lr = 0.0005
    epochs = 60
    batch_size = 32
    dropout = 0
    directions = 1
    # directions = 2
    max_grad_norm = 5
    max_len = 100
    unk_gen = False  # allow generating UNK

class ConfigSmall(object):
    cell = "GRU"
    emsize = 256
    pemsize = 5
    nlayers = 1
    lr = 0.001
    epochs = 60
    batch_size = 32
    dropout = 0
    directions = 2
    max_grad_norm = 10
    max_len = 100
    unk_gen = False

class ConfigTest(object):
    cell = "LSTM"
    emsize = 30
    pemsize = 5
    nlayers = 1
    lr = 0.001
    epochs = 2
    batch_size = 3
    dropout = 0
    directions = 2
    max_grad_norm = 1
    testmode = True
    max_len = 50
    unk_gen = False
