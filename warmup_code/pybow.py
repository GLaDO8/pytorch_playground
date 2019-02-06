import torch

data = [
    ("me gusta comer en la cafeteria".split(), "SPANISH"),
    ("Give it to me".split(), "ENGLISH"),
    ("No creo que sea una buena idea".split(), "SPANISH"),
    ("No it is not a good idea to get lost at sea".split(), "ENGLISH")
]

test_data = [
    ("Yo creo que si".split(), "SPANISH"),
    ("it is lost on me".split(), "ENGLISH")
]

word_to_ix = {}
for sent, _ in data + test_data:
    for word in sent:
        if word not in word_to_ix:
            word_to_ix[word] = len(word_to_ix)
# print(word_to_ix)

VOCAB_SIZE = len(word_to_ix)
NUM_LABELS = 2

class bow_classifier(torch.nn.Module):

    def __init__(self, num_labels, vocab_size):
        #calls parent class init
        super().__init__()
        #nn.linear takes input size and output size
        self.linear = torch.nn.Linear(vocab_size, num_labels) 

    def forwardprop(self, bow_vec):
        return torch.nn.LogSoftmax(self.linear(bow_vec), dim = 1)


def make_bow_vector(sentence, word_to_ix):
    vec = torch.zeros(len(word_to_ix))
    for word in sentence:
        vec[word_to_ix[word]] += 1
    return vec.view(1, -1)