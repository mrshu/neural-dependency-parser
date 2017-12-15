import torch
import torch.nn as nn
import torch.optim as optim
from itertools import product
from torch.autograd import Variable
import numpy as np

class DepParser(nn.Module):
    def __init__(self, word_voc_size, pos_embedding_len, labels_size, d_embed,
                 hidden_size, w2i, i2w, t2i, i2t, l2i, i2l):
        super(DepParser, self).__init__()
        self.hidden_size = hidden_size
        self.d_embed = d_embed
        self.labels_size = labels_size

        self.w2i = w2i
        self.i2w = i2w
        self.t2i = t2i
        self.i2t = i2t
        self.l2i = l2i
        self.i2l = i2l

        self.w_embedding = nn.Embedding(word_voc_size, d_embed)
        self.pos_embedding = nn.Embedding(pos_embedding_len, d_embed)

        self.lstm = nn.LSTM(2 * d_embed, hidden_size, 1, bidirectional=True)

        self.fc1 = torch.nn.Linear(2 * hidden_size * 2, hidden_size)
        self.tanh = nn.Tanh()
        self.fc2 = torch.nn.Linear(hidden_size, 1)

        self.mlp_fc1 = torch.nn.Linear(2 * hidden_size * 2, hidden_size)
        self.mlp_fc2 = torch.nn.Linear(hidden_size, labels_size)

        self.root = Variable(torch.zeros((1, 1, hidden_size * 2)),
                             requires_grad=False)

    def forward(self, words, pos, gl):
        e_w = self.w_embedding(words)
        e_p = self.pos_embedding(pos)

        input = torch.cat((e_w, e_p), 1).view(len(words), 1, self.d_embed * 2)

        output, (hidden, cell) = self.lstm(input)

        M = Variable(torch.zeros(len(words) + 1, len(words) + 1))

        # Concatentate the root vector
        output = torch.cat((self.root, output), 0)

        words_comb = list(product(list(range(len(words) + 1)), repeat=2))
        arcs = Variable(torch.zeros(len(words_comb), 2 * 2 * self.hidden_size))

        for i, (w1, w2) in enumerate(words_comb):
            arc = torch.cat((output[w1], output[w2]), 1)
            arcs[i, :] = arc

        out = self.fc1(arcs)
        out = self.tanh(out)
        g = self.fc2(out)

        for i, (w1, w2) in enumerate(words_comb):
            M[w1, w2] = g[i]

        L = Variable(torch.zeros(len(gl), self.labels_size))
        Ls = Variable(torch.zeros(len(gl), 2 * 2 * self.hidden_size))
        for i, (w1, w2, _) in enumerate(gl):
            arc = torch.cat((output[w1], output[w2]), 1)
            Ls[i, :] = arc

        out = self.mlp_fc1(Ls)
        out = self.tanh(out)
        L = self.mlp_fc2(out)

        return M, L
