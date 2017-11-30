import re
from collections import Counter
from collections import defaultdict

file = open('data/en-ud-train.conllu', 'r', encoding="utf8")
pos_tags = []
vocabulary = []

for line in file:
    if line.startswith('#'):
        continue
    if line != '\n':
        line_split = line.split('\t')
        # remove sentences which start with integer index with hyphens
        if re.match("\d+[-]\d+", line_split[0]):
            file.remove(line)
        pos_tags.append(line_split[4])
        vocabulary.append(line_split[1])

pos_tags = list(set(pos_tags))

voc_counter = Counter(vocabulary)

#replace words that occur once with <unk>
for word in vocabulary:
    if voc_counter[word] < 2:
        vocabulary.remove(word)
        vocabulary.append('<unk>')

voc_counter = Counter(vocabulary)

w2i = defaultdict(lambda: len(w2i))
t2i = defaultdict(lambda: len(t2i))
UNK = w2i["<unk>"]
w2i = defaultdict(lambda: UNK, w2i)

for index, word in enumerate(voc_counter):
    w2i[word] = index

i2w = {v: k for k, v in w2i.items()}

for index, tag in enumerate(pos_tags):
    t2i[tag] = index

i2t = {v: k for k, v in t2i.items()}

l2i = {1: 'root', 2: 'dep', 3: 'aux', 4: 'auxpass', 5: 'cop', 6: 'arg', 7: 'agent', 8: 'comp', 9: 'acomp', 10: 'ccomp', 11: 'xcomp', 12: 'obj', 13: 'dobj', 14: 'iobj', 15: 'pobj', 16: 'subj', 17: 'nsubj', 18: 'nsubjpass', 19: 'csubj', 20: 'csubjpass', 21: 'cc', 22: 'conj', 23: 'expl', 24: 'mod', 25: 'amod', 26: 'appos', 27: 'advcl', 28: 'det', 29: 'predet', 30: 'preconj', 31: 'vmod', 32: 'mwe', 33: 'mark', 34: 'advmod', 35: 'neg', 36: 'rcmod', 37: 'quantmod', 38: 'nn', 39: 'npadvmod', 40: 'tmod', 41: 'num', 42: 'number', 43: 'prep', 44: 'poss', 45: 'possessive', 46: 'prt', 47: 'parataxis', 48: 'goeswith', 49: 'punct', 50: 'ref', 51: 'sdep', 52: 'xsubj', 53: 'partmod', 54: 'abbrev', 55: 'attr', 56: 'complm', 57: 'discourse', 58: 'infmod', 59: 'purpcl', 60: 'rel', 61: 'pcomp'}
i2l = {v: k for k, v in l2i.items()}
