""" A base class for RNN. """
import torch.nn as nn


class BaseRNN(nn.Module):
    def __init__(self, vocab_size, hidden_size, input_dropout_p, dropout_p, n_layers):
        super(BaseRNN, self).__init__()
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.input_dropout_p = input_dropout_p
        self.input_dropout = nn.Dropout(p=input_dropout_p)
        if self.rnn_type == 'lstm':
            self.rnn_cell = nn.LSTM
        elif self.rnn_type == 'gru':
            self.rnn_cell = nn.GRU
        else:
            raise ValueError("Unsupported RNN Cell: {0}".format(self.rnn_type))

        self.dropout_p = dropout_p

    def forward(self, *args, **kwargs):
        raise NotImplementedError()
