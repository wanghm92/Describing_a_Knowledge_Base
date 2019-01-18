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

class ConfigWikibio(object):
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

class ConfigTest(object):
    cell = "GRU"
    emsize = 30
    pemsize = 5
    nlayers = 1
    lr = 0.001
    epochs = 2
    batch_size = 2
    dropout = 0
    directions = 2
    max_grad_norm = 1
    testmode = True
    max_len = 50