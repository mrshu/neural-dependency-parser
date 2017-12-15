import re
from collections import Counter
from collections import defaultdict
import numpy as np

def read_voc_pos_tags_from_conllu_file(filename):
    file = open(filename, 'r', encoding="utf8")
    pos_tags = []
    vocabulary = []
    sentences = []

    text = file.read()

    for sentence in text.split('\n\n'):
        s = {}
        w2i = defaultdict(lambda: len(w2i))
        w2i['0'] = 0
        for line in sentence.split('\n'):
            if line.startswith('#'):
                continue
            if line and line != '\n':
                line_split = line.split('\t')
                # remove sentences which start with integer index with hyphens
                if re.match("\d+[-]\d+", line_split[0]):
                    file.remove(line)
                id = w2i[line_split[0]]

                s[id] = ([line_split[1].lower(),
                          line_split[4],
                          line_split[6],
                          line_split[7]])

                pos_tags.append(line_split[4])
                vocabulary.append(line_split[1].lower())

        golden_labels = []
        M = np.zeros((len(s) + 1, len(s) + 1))
        for i, w in enumerate(s.keys()):
            if s[w][2] == '_':
                continue
            M[w2i[s[w][2]]][i+1] = 1
            golden_labels.append([w2i[s[w][2]], i+1, s[w][3]])
        M[0, 0] = 1
        if s:
            sentences.append([s, M, golden_labels])
    return vocabulary, pos_tags, sentences

def read_conllu_file(filename):

    vocabulary, pos_tags, sentences = read_voc_pos_tags_from_conllu_file(filename)

    vocabulary = set(vocabulary)
    pos_tags = list(set(pos_tags))
    voc_counter = Counter(vocabulary)

    filtered_vocabulary = set()

    labels = set()
    for s in sentences:
        for i, v in s[0].items():
            labels.add(v[3])

    # replace words that occur once with <unk>
    for word in vocabulary:
        if voc_counter[word] > 2:
            filtered_vocabulary.add(word)
        else:
            filtered_vocabulary.add('<unk>')

    voc_counter = Counter(vocabulary)

    w2i = defaultdict(lambda: len(w2i))
    t2i = defaultdict(lambda: len(t2i))
    l2i = defaultdict(lambda: len(l2i))

    for index, word in enumerate(voc_counter):
        w2i[word] = index

    i2w = {v: k for k, v in w2i.items()}

    for index, tag in enumerate(pos_tags):
        t2i[tag] = index

    for index, label in enumerate(labels):
        l2i[label] = index

    i2t = {v: k for k, v in t2i.items()}

    i2l = {v: k for k, v in l2i.items()}

    index_sentences = []
    golden_labels = []
    for (sentence, _, gl) in sentences:
        s = []
        for k, v in sentence.items():
            s.append((w2i[v[0]], t2i[v[1]]))
        l = []
        for f, t, label in gl:
            l.append([f, t, l2i[label]])
        golden_labels.append(l)
        index_sentences.append(s)

    return (dict(w2i), dict(i2w), dict(t2i), dict(i2t), dict(l2i), dict(i2l),
            sentences, index_sentences, golden_labels)
