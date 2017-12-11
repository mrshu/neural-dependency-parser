import torch
import torch.nn as nn
import torch.optim as optim
from itertools import product
from torch.autograd import Variable
import numpy as np

class DepParser(nn.Module):
    def __init__(self, word_voc_size, pos_embedding_len, d_embed, hidden_size):
        super(DepParser, self).__init__()
        self.hidden_size = hidden_size
        self.d_embed = d_embed

        self.w_embedding = nn.Embedding(word_voc_size, d_embed)
        self.pos_embedding = nn.Embedding(pos_embedding_len, d_embed)

        self.lstm = nn.LSTM(2 * d_embed, hidden_size, 1)

        self.fc1 = torch.nn.Linear(2 * hidden_size, hidden_size)
        self.tanh = nn.Tanh()
        self.fc2 = torch.nn.Linear(hidden_size, 1)

        self.root = Variable(torch.zeros((1, 1, hidden_size)),
                             requires_grad=False)

    def forward(self, words, pos):
        e_w = self.w_embedding(words)
        e_p = self.pos_embedding(pos)

        input = torch.cat((e_w, e_p), 1).view(len(words), 1, self.d_embed * 2)

        output, (hidden, cell) = self.lstm(input)

        M = Variable(torch.zeros(len(words) + 1, len(words) + 1))

        # Concatentate the root vector
        output = torch.cat((self.root, output), 0)

        for w1, w2 in product(list(range(len(words) + 1)), repeat=2):
            arc = torch.cat((output[w1], output[w2]), 1)

            out = self.fc1(arc)
            out = self.tanh(out)
            M[w1, w2] = self.fc2(out)

        return M


if __name__ == "__main__":
    # from data_import import read_voc_pos_tags_from_conllu_file
    # voc, pos, s = read_voc_pos_tags_from_conllu_file('./en-ud-dev.conllu.txt')

    from data_import import read_conllu_file
    w2i, i2w, t2i, i2t, l2i, i2l, sentences, index_sentences = read_conllu_file('./en-ud-dev.conllu.txt')

    model = DepParser(len(w2i), len(t2i), 30, 50)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    for i in range(len(sentences)):
        s, M = sentences[i]
        sentence = index_sentences[i]

        words = list(map(lambda x: x[0], sentence))
        pos = list(map(lambda x: x[1], sentence))

        rev_words = list(map(lambda x: i2w[x], words))
        rev_pos = list(map(lambda x: i2t[x], pos))

        # print(words, pos)
        # print(rev_words, rev_pos)

        words = Variable(torch.LongTensor(words))
        pos = Variable(torch.LongTensor(pos))

        optimizer.zero_grad()

        out_M = model(words, pos)
        t_out_M = torch.t(out_M)

        if i > 0 and i % 50 == 0:
            s = nn.Softmax()
            print(torch.t(s(t_out_M)))
            print(M)

        np_targets = np.argmax(M, axis=0)
        targets = Variable(torch.from_numpy(np_targets), requires_grad=False)

        loss = criterion(t_out_M, targets)
        print('{} loss: {}'.format(i, loss.data[0]))
        loss.backward()
        optimizer.step()


