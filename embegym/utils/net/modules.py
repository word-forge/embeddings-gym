import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import numpy


def _log_softmax1(x):
    return F.log_softmax(x, dim=1)


def _no_activation(x):
    return x


class CnnLSTMAttention(nn.Module):
    def __init__(self,
                 embeddings_size,
                 out_classes=2,
                 conv_len=3,
                 conv_channels=64,
                 conv_act=F.relu,
                 lstm_hidden_size=64,
                 lstm_num_layers=1,
                 lstm_bidir=True,
                 out_act=_log_softmax1):
        super(CnnLSTMAttention, self).__init__()
        self.word_conv = nn.Conv1d(embeddings_size,
                                   conv_channels,
                                   conv_len,
                                   padding=1)
        self.in_conv_act = conv_act
        self.in_conv_channels = conv_channels
        self.lstm_hidden_size = lstm_hidden_size
        self.lstm_num_layers = lstm_num_layers
        self.lstm_bidir = bool(lstm_bidir)
        self.rnn = nn.LSTM(conv_channels,
                           lstm_hidden_size,
                           lstm_num_layers,
                           bidirectional=lstm_bidir)
        self.lstm_out_size = lstm_hidden_size * (int(lstm_bidir) + 1)
        self.attention = nn.Linear(self.lstm_out_size,
                                   1)
        self.out = nn.Linear(self.lstm_out_size,
                             out_classes)
        self.out_act = out_act

    def make_hidden(self, batch_size):
        hidden_shape = (self.lstm_num_layers * (int(self.lstm_bidir) + 1),
                        batch_size,
                        self.lstm_hidden_size)
        return (Variable(torch.randn(*hidden_shape)),
                Variable(torch.randn(*hidden_shape)))

    def forward(self, x, hidden=None):
        """
        :param x: (BatchSize, EmbeddingsSize, MaxSentenceLength)
        :param hidden: see `make_hidden`
        :return: Probabilities of classes
        """
        x = self.word_conv(x)  # (B, Conv, MS)
        x = x.permute(2, 0, 1)  # (MS, B, Conv)
        if hidden is None:
            hidden = self.make_hidden(x.size()[1])
            if x.data.is_cuda:
                hidden = (hidden[0].cuda(), hidden[1].cuda())
        x, hidden = self.rnn(x, hidden)  # (MS, B, lstm_out)
        att = self.attention(x)  # (MS, B, 1)
        att_norm = F.softmax(att, 2)  # (MS, B, 1)
        x = x * att_norm.expand_as(x)  # (MS, B, lstm_out)
        x = x.sum(0)  # (B, lstm_out)
        x = self.out_act(self.out(x))  # (B, out_classes)
        return x
