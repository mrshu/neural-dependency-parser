import torch
import torch.nn as nn
from torch.autograd import Variable

class DepParser(nn.Module):
    def __init__(self, word_embedding, pos_embedding, d_embed, hidden_size):
        super(DepParser, self).__init__()
        self.hidden_size = hidden_size
        self.d_embed = d_embed

        self.w_embedding = nn.Embedding(word_embedding.size(0),
                                        word_embedding.size(1))
        self.w_embedding.weight = nn.Parameter(word_embedding)

        self.pos_embedding = nn.Embedding(pos_embedding.size(0),
                                          pos_embedding.size(1))
        self.pos_embedding.weight = nn.Parameter(pos_embedding)


        self.lstm = nn.LSTM(d_embed, hidden_size, 1)

    def forward(self, words, pos, hidden, cell):
        e_w = self.embedding(words).view(1, 1, -1)
        e_p = self.embedding(pos).view(1, 1, -1)
        output = torch.cat((e_w, e_p), 0)
        output, hidden, cell = self.lstm(output, (hidden, cell))
        return output, hidden, cell
