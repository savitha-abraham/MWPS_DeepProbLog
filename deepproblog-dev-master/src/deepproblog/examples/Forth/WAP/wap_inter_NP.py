from functools import lru_cache

import torch
import torch.nn as nn
import torch.optim as optim
from deepproblog.utils import count_parameters
from torch.autograd import Variable
from problog.logic import unquote, make_safe, term2list
from deepproblog.network import Network
from deepproblog.utils.standard_networks import MLP

vocab = dict()
hidden_size = 512
embed_size = 256

with open('data/vocab_inter.txt') as f:
    for i, word in enumerate(f):
        word = word.strip()
        # vocab[i] = word
        vocab[word] = i

def isDigit(x):
    if x.isdigit():
        return True
    try:
        float(x)
        return True
    except ValueError:
        return False


def tokenize(sentence):
    #print("In tokenize")
    sentence = sentence.split(' ')
    tokens = []
    
    for i, word in enumerate(sentence):
        if word[-1]=='.' or word[-1]==',' or word[-1]=='?':
            word = word[0:-1]
        
        if isDigit(word):
            tokens.append('<NR>')
        elif any(char.isdigit() for char in word):
            numb = word[1:]
            if isDigit(numb):
                tokens.append('<NR>')
                      
        else:
            if word.lower() in vocab:
                tokens.append(word.lower())
            else:
                tokens.append('<UNK>')
    return [vocab[token] for token in tokens]




#Verb classifier
class VC(nn.Module):
    def __init__(self, vocab_size, hidden_size, p_drop=0.0):
        super(VC, self).__init__()
        self.lstm = nn.GRU(embed_size, hidden_size, 1, bidirectional=True, batch_first=True)
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(vocab_size, embed_size)  # , _weight=torch.nn.Parameter(weights))
        self.dropout = nn.Dropout(p_drop)
        self.classifier = MLP(hidden_size *2, 4)
    def forward(self, sentence):
        sentence = make_safe(" ".join(map(lambda x: unquote(str(x)), term2list(sentence))))
        print(sentence)
        #sentence = sentence.value.strip('"')
        x = tokenize(sentence)
        seq_len = len(x)
        x = torch.LongTensor(x).unsqueeze(0)
        x = self.embedding(x)
        x, _ = self.lstm(x)
        x = x.view(seq_len, 2, self.hidden_size)
        x1 = x[-1, 0, ...]
        x2 = x[0, 1, ...]
        WP_repr = torch.cat([x1, x2])
        y = self.classifier(WP_repr)
        
        return y
       
class WC(nn.Module):
    def __init__(self, vocab_size, hidden_size, p_drop=0.0):
        super(WC, self).__init__()
        self.lstm = nn.GRU(embed_size, hidden_size, 1, bidirectional=True, batch_first=True)
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(vocab_size, embed_size)  # , _weight=torch.nn.Parameter(weights))
        self.dropout = nn.Dropout(p_drop)
        self.classifier = MLP(hidden_size * 4, 5)
        

    def forward(self, sentence, I1):
        sentence = make_safe(" ".join(map(lambda x: unquote(str(x)), term2list(sentence))))
        #unquote(str(sentence)).strip('"')
        I1 = int(I1)
        x = tokenize(sentence)
        seq_len = len(x)
        x = torch.LongTensor(x).unsqueeze(0)
        x = self.embedding(x)
        x, _ = self.lstm(x)
        x = x.view(seq_len, 2, self.hidden_size)
        x_i11 = torch.cat([x[-1, 0, ...], x[I1, 0, ...]])
        x_i12 = torch.cat([x[0, 1, ...], x[I1, 1, ...]])
        x_i1 = torch.cat([x_i11, x_i12])
        y = self.classifier(x_i1)  
        
        return y


# class LSTMTagger(nn.Module):

#     def __init__(self, vocab_size, hidden_size, tagset_size):
#         super(LSTMTagger, self).__init__()
#         self.hidden_dim = hidden_size

#         self.word_embeddings = nn.Embedding(vocab_size, embed_size)

#         # The LSTM takes word embeddings as inputs, and outputs hidden states
#         # with dimensionality hidden_dim.
#         self.lstm = nn.LSTM(embed_size, hidden_size)

#         # The linear layer that maps from hidden state space to tag space
#         self.hidden2tag = nn.Linear(hidden_size, tagset_size)

#     def forward(self, sentence):
#         embeds = self.word_embeddings(sentence)
#         lstm_out, _ = self.lstm(embeds.view(len(sentence), 1, -1))
#         tag_space = self.hidden2tag(lstm_out.view(len(sentence), -1))
#         tag_scores = F.log_softmax(tag_space, dim=1)
#         return tag_scores

         
def get_networks(lr, p_drop, n=8):
    #word type classifiers (Classes: Container_Source, Container_Destination, Entity, Attribute, None)
    #num_wordCategories = 5 
    nn_wc= WC(len(vocab), hidden_size, p_drop=p_drop)
    #network1 = MLP(hidden_size * 4, num_wordCategories)
    #Verb classifier (Classes: holds, increase, decrease, transfer)
    #num_verbCategories = 4	
    nn_vc= VC(len(vocab), hidden_size, p_drop=p_drop)
    #network2 = MLP(hidden_size *2, num_verbCategories)
    
    names = ['nn_wc', 'nn_vc']
    networks = [nn_wc, nn_vc]

    networks = [(networks[i], names[i], optim.Adam(networks[i].parameters(), lr=lr)) for i in range(2)]

    return networks
