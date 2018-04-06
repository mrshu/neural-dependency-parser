import torch
import torch.nn as nn
import sys
from data_import import read_conllu_file
from torch.autograd import Variable
from mst import mst_one_out_root
from collections import defaultdict

if __name__ == "__main__":
    (w2i, i2w, t2i, i2t, l2i, i2l, sentences,
     index_sentences, golden_labels) = read_conllu_file(sys.argv[1])

    model = torch.load(sys.argv[2])
    s = nn.Softmax()

    model.w2i = defaultdict(lambda: 0, model.w2i)
    model.t2i = defaultdict(lambda: 0, model.t2i)

    for z, (sentence, _, _) in enumerate(sentences):
        input_words = []
        input_pos = []
        for k in sentence:
            input_words.append(model.w2i[sentence[k][0]])
            input_pos.append(model.t2i[sentence[k][1]])

        input_words = Variable(torch.LongTensor(input_words))
        input_pos = Variable(torch.LongTensor(input_pos))

        M, L = model(input_words, input_pos, [])
        t_out_M = torch.t(M)
        predicted_M = torch.t(s(t_out_M))

        graph = {}
        for i in range(M.size(1)):
            sg = {}
            for k in range(M.size(0)):
                sg[k] = predicted_M[i, k].data[0]
                if i == 0:
                    sg[k] = 0
            graph[i] = sg
        out_graph = mst_one_out_root(graph)

        labels_input = []
        to_from_mapping = {}
        for k in out_graph:
            for l in out_graph[k]:
                labels_input.append([k, l, '_'])

        M, L = model(input_words, input_pos, labels_input)
        _, indices = torch.max(s(L), 1)
        new_sentence = {}

        for i in sentence:
            if sentence[i][2] == '_':
                new_sentence[i] = [sentence[i][5],
                                   sentence[i][4],
                                   sentence[i][1],
                                   '_',
                                   '_']
                continue

            for j, (f, t, _) in enumerate(labels_input):
                if int(sentence[i][5]) == t:
                    new_sentence[i] = [sentence[i][5],
                                       sentence[i][4],
                                       sentence[i][1],
                                       f,
                                       model.i2l[indices[j].data[0]]]

        for i in new_sentence:
            string = '{}\t{}\t_\t_\t{}\t_\t{}\t{}\t_\t'.format(*new_sentence[i])
            print(string)
        print()
