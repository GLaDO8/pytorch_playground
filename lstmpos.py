#learning LSTM code for parts-of-speech tagging
#main tutorial - https://pytorch.org/tutorials/beginner/nlp/sequence_models_tutorial.html#sphx-glr-download-beginner-nlp-sequence-models-tutorial-py

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

#training data. note that the first sentence is of length 5 and second is of length 4
training_data = [
    ("The dog ate the apple".split(), ["DET", "NN", "V", "DET", "NN"]),
    ("Everybody read that book".split(), ["NN", "V", "DET", "NN"])
]
#label indices
tag_to_ix = {"DET": 0, "NN": 1, "V": 2}

#vocab builder
word_to_ix = {}
for sent, tags in training_data:
    for word in sent:
        if word not in word_to_ix:
            word_to_ix[word] = len(word_to_ix)
# print(word_to_ix)
#note that we have two 'the's in word_to_ix because of casing

#data preparation 
#prepare_sequence will convert each sequence into a tensor of indices which refer the main vocabulary
def prepare_sequence(seq, to_ix):
    idxs = [to_ix[w] for w in seq]
    return torch.tensor(idxs, dtype=torch.long)

#testing code to understand input type
#note that tensor.view is equivalent to tensor.reshape
# for sentence, tag in training_data:
#     x = prepare_sequence(sentence, word_to_ix)
#     word_embed = nn.Embedding(len(word_to_ix), 6)
#     yo = word_embed(x)
#     print(yo)
#     print(yo.view(len(sentence), 1, -1))

#param settings
EMBEDDING_DIM = 6 #dimension of each word embedding
HIDDEN_DIM = 6 #dimension of h unit

#neural network custom nn.Module
class LSTM_tagger(nn.Module):
    
    def __init__(self, embed_dim, hid_dim, vocab_size, tagset_size):
        super().__init__()
        self.hidden_dim = hid_dim

        #for creating a word vector for each word of constant size (embedding_dim)
        self.word_embed = nn.Embedding(vocab_size, embed_dim)

        self.lstm = nn.LSTM(embed_dim, hid_dim) #input size and hidden size

        #output layer has to be explicitely defined as LSTM does not have a sigmoid/softmax layer in it for passing output. It passes out the hidden state.
        #it takes input(N, *, in_features) and outputs(N, *, out_features)
        self.output_layer = nn.Linear(hid_dim, tagset_size)

        #initialise the hidden state for input into lstm
        self.hidden = self.init_hidden()
    
    def init_hidden(self):
        #we have to define the hidden state
        #we are returning two tensors, one is h0 and other is c0 
        #dimensions for both are (num_layers * num_directions, batch, hidden_size)
        return(
            torch.zeros(1, 1, self.hidden_dim), 
            torch.zeros(1, 1, self.hidden_dim)
        )

    #note that this method is supposed to be named as 'forward'
    def forward(self, sentence):
        #sent_embed is of size (len(sentence), EMBEDDING_DIM)
        sent_embed  = self.word_embed(sentence)
        #sent_embed.view will reshape the embedded vector to (seq_len, batch_size, embedding_size)
        #in our case, the batch size is 1, and the -1 will reshape to 6 as EMBEDDING_DIM = 6
        lstm_out, self.hidden = self.lstm(sent_embed.view(len(sentence), 1, -1), self.hidden)

        #take the output of lstm, throw it in nn.Linear which will map it to tag space.
        output = F.log_softmax(self.output_layer(lstm_out.view(len(sentence), -1)), dim = 1)
        return output


#training
learning_rate = 0.1
model = LSTM_tagger(EMBEDDING_DIM, HIDDEN_DIM, len(word_to_ix), len(tag_to_ix))
loss_func = nn.NLLLoss()
optimizer = optim.SGD(model.parameters(), learning_rate)

for epoch in range(300):
    for sentence, tag in training_data:
        #clear out gradients after each epoch
        model.zero_grad()

        #clear out initial hidden state value before each epoch 
        model.init_hidden()

        #find index array for each sequence and tag 
        indseq = prepare_sequence(sentence, word_to_ix)
        ground_truth_tags = prepare_sequence(tag, tag_to_ix)
        
        #forward propogation
        predictions = model(indseq)

        #loss calculation
        loss = loss_func(predictions, ground_truth_tags)
        print(str(epoch) + " --> " + str(loss))

        #calculate all gradients of loss wrt weights
        loss.backward(retain_graph = True)

        #update all weights
        optimizer.step()

with torch.no_grad():
    indseq = prepare_sequence(training_data[0][0], word_to_ix)

    #in custom nn modules, there is no seperate predict function. we pass the input for which we want the prediction through the model while being nested inside a torch.no_grad() condition. This gives us the output but will not train. 
    predicted_tag = model(indseq)

    #print predictions
    print(predicted_tag)
