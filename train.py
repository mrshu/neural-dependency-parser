import torch
import torch.nn as nn
import torch.optim as optim
from itertools import product
from torch.autograd import Variable
import numpy as np
import datetime

from model import DepParser
from tensorboardX import SummaryWriter
import sys

if __name__ == "__main__":
    # from data_import import read_voc_pos_tags_from_conllu_file
    # voc, pos, s = read_voc_pos_tags_from_conllu_file('./en-ud-dev.conllu.txt')

    from data_import import read_conllu_file
    (w2i, i2w, t2i, i2t, l2i, i2l, sentences,
     index_sentences, golden_labels) = read_conllu_file(sys.argv[1])

    dt = datetime.datetime.now().isoformat()

    writer = SummaryWriter()

    model = DepParser(len(w2i), len(t2i), len(l2i), 100, 125, w2i, i2w, t2i, i2t,
                      l2i, i2l)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    for i in range(len(sentences) * 3):
        s, M, _ = sentences[i % len(sentences)]
        sentence = index_sentences[i % len(sentences)]

        gl = golden_labels[i % len(sentences)]
        gl_targets = np.array(list(map(lambda x: x[2], gl)))

        words = list(map(lambda x: x[0], sentence))
        pos = list(map(lambda x: x[1], sentence))

        rev_words = list(map(lambda x: i2w[x], words))
        rev_pos = list(map(lambda x: i2t[x], pos))

        # print(words, pos)
        # print(rev_words, rev_pos)

        words = Variable(torch.LongTensor(words))
        pos = Variable(torch.LongTensor(pos))

        optimizer.zero_grad()

        out_M, out_L = model(words, pos, gl)
        t_out_M = torch.t(out_M)

        s = nn.Softmax()

        predicted_M = torch.t(s(t_out_M))
        _, indices = torch.max(s(out_L), 1)
        _, indices_M = torch.max(predicted_M, 0)

        if i > 0 and i % 50 == 0:
            print(predicted_M)
            print(M)

            print(indices.unsqueeze(0))
            print(gl_targets)

        np_targets = np.argmax(M, axis=0)
        targets = Variable(torch.from_numpy(np_targets), requires_grad=False)

        label_targets = Variable(torch.from_numpy(gl_targets),
                                 requires_grad=False)

        loss_matrix = criterion(t_out_M, targets)
        loss_labels = criterion(out_L, label_targets)

        loss = loss_matrix + loss_labels

        print('{} loss: {}'.format(i, loss.data[0]))

        if i > 0 and i % 5000 == 0:
            torch.save(model, 'model_{}_{}_{}_{}'.format(dt, sys.argv[1],
                                                         loss.data[0], i))

        writer.add_scalar('loss', loss.data[0], i)

        lab_acc = (label_targets == indices.unsqueeze(0)).float().mean()
        writer.add_scalar('labels_accuracy', lab_acc.data[0], i)

        M_targets = Variable(torch.from_numpy(np_targets),
                             requires_grad=False)

        matrix_acc = (M_targets == indices_M).float().mean()
        writer.add_scalar('matrix_accuracy', matrix_acc.data[0], i)

        loss.backward()
        optimizer.step()
    torch.save(model, 'model_{}_{}'.format(dt, sys.argv[1]))
