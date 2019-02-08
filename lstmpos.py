#LSTM unit in pytorch expects only 3D tensors. The dimensions are 
#First dim = sequence
#second dim = instances in the mini batch
#third dim = indices of input
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

torch.manual_seed(1)

#for part of speech tagging, input sentence is w1, w2, w3....E(V) v is vocab
#T is the complete tag set and yi is the tag for word wi

#data preparation 
def prepare_sequence(seq, to_ix):
    idxs = [to_ix[w] for w in seq]
    return torch.tensor(idxs, dtype=torch.long)

training_data = [
    ("The dog ate the apple".split(), ["DET", "NN", "V", "DET", "NN"]),
    ("Everybody read that book".split(), ["NN", "V", "DET", "NN"])
]

#vocab builder
word_to_ix = {}
for sent, tags in training_data:
    for word in sent:
        if word not in word_to_ix:
            word_to_ix[word] = len(word_to_ix)
# print(word_to_ix)
#note that we have two 'the's in word_to_ix because of casing

#testing code to understand input type
#note that tensor.view is equivalent to tensor.reshape
for sentence, tag in training_data:
    x = prepare_sequence(sentence, word_to_ix)
    word_embed = nn.Embedding(len(word_to_ix), 6)
    yo = word_embed(x)
    print(yo)
    print(yo.view(len(sentence), 1, -1))


#label indices
tag_to_ix = {"DET": 0, "NN": 1, "V": 2}

#param settings
EMBEDDING_DIM = 6 #dimension of each word embedding
HIDDEN_DIM = 6 #dimension of h unit

class LSTM_tagger(nn.Module):
    def __init__(self, embed_dim, hidden_dim, vocab_size, tagset_size):
        super().__init__()
        self.hidden_dim = hidden_dim

        #for creating a word vector for each word of constant size (embedding_dim)
        self.word_embed = nn.Embedding(vocab_size, hidden_dim)

        self.lstm = nn.LSTM(embed_dim, hidden_dim) #input size and hidden size

        #output layer has to be explicitely defined as LSTM does not have a sigmoid/softmax layer in it for passing output. It passes out the hidden state.
        self.output_layer = nn.Linear(hidden_dim, tagset_size)

        #initialise the hidden state for input into lstm
        self.hidden = self.init_hidden()
    
    def init_hidden(self):
        #we have to define the hidden state
        #we are returning two tensors, one is h0 and other is c0 
        #dimensions for both are (num_layers * num_directions, batch, hidden_size)
        return (
            torch.zeros(1, 1, self.hidden_dim), 
            torch.zeros(1, 1, self.hidden_dim)
        )

    def forward_prop(self, sentence):
        embeds  = self.word_embed

