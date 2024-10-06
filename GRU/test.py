#

import d2l
from d2l import torch as d2l
#print(d2l.__version__)
batch_size = 32
num_steps = 35
train_iter, vocab = d2l.load_data_voc(batch_size, num_steps)

print(vocab.dataset)
