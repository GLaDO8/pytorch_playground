import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

torch.manual_seed(1)

word_to_ix = {"hello": 0, "world": 1}
#first argument is the size of the embedded matrix. The second argument is the dimension of each word embedding. 
embeds = nn.Embedding(2, 5)  # 2 words in vocab, 5 dimensional embeddings
lookup_tensor = torch.tensor([word_to_ix["hello"], word_to_ix["world"]], dtype=torch.long)
hello_embed = embeds(lookup_tensor)
print(hello_embed)
# print(embeds)