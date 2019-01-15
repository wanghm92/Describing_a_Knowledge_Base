class Config(object):
    cell = "GRU"
    emsize = 256
    pemsize = 5
    nlayers = 1
    lr = 0.001
    epochs = 50
    batch_size = 128
    dropout = 0
    directions = 2
    max_grad_norm = 10
    max_len = 100


class ConfigTest(object):
    cell = "GRU"
    emsize = 30
    pemsize = 30
    nlayers = 1
    lr = 0.001
    epochs = 2
    batch_size = 10
    dropout = 0
    directions = 2
    max_grad_norm = 1
    testmode = True
    max_len = 50