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

with open('data/vocab.txt') as f:
    for i, word in enumerate(f):
        word = word.strip()
        # vocab[i] = word
        vocab[word] = i

def tokenize_n(sentence, n):
    #print("In tokenize:", sentence, n)
    sentence = str(sentence).split(' ')
    tokens = []
    index =-1
    for i, word in enumerate(sentence):
        if word[-1]=='.' or word[-1]==',' or word[-1]=='?':
            word = word[0:-1]
        if word == n:
            index = i
                      
        if word.lower() in vocab:
            tokens.append(word.lower())
        else:
            tokens.append('<UNK>')
    return [vocab[token] for token in tokens], index


def tokenize_query(sentence):
    sentence = str(sentence).split(' ')
    tokens = []
    for i, word in enumerate(sentence):
        if word[-1]=='.' or word[-1]==',' or word[-1]=='?':
            word = word[0:-1]
        
        
        if word.lower() in vocab:
            tokens.append(word.lower())
        else:
            tokens.append('<UNK>')
        
    return [vocab[token] for token in tokens]




#Schema n0/n1 Category classifier
class NC(nn.Module):
    def __init__(self, vocab_size, hidden_size, p_drop=0.0):
        super(NC, self).__init__()
        self.lstm = nn.GRU(embed_size, hidden_size, 1, bidirectional=True, batch_first=True)
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(vocab_size, embed_size)  # , _weight=torch.nn.Parameter(weights))
        self.dropout = nn.Dropout(p_drop)
        self.classifier = MLP(hidden_size *4, 7)
    def forward(self, sentence, n):
        sentence = unquote(str(sentence))#make_safe(" ".join(map(lambda x: unquote(str(x)), term2list(sentence))))
        n = unquote(str(n))
        #sentence = make_safe(" ".join(map(lambda x: unquote(str(x)), term2list(sentence))))
        #print(sentence)
        #sentence = sentence.value.strip('"')
        x, index = tokenize_n(sentence, n)
        seq_len = len(x)
        x = torch.LongTensor(x).unsqueeze(0)
        x = self.embedding(x)
        x, _ = self.lstm(x)
        x = x.view(seq_len, 2, self.hidden_size)
        
        x_i11 = torch.cat([x[-1, 0, ...], x[index, 0, ...]])
        x_i12 = torch.cat([x[0, 1, ...], x[index, 1, ...]])
        x_i1 = torch.cat([x_i11, x_i12])
        y = self.classifier(x_i1)  
        
        return y

#Schema query Category classifier
class QC(nn.Module):
    def __init__(self, vocab_size, hidden_size, p_drop=0.0):
        super(QC, self).__init__()
        self.lstm = nn.GRU(embed_size, hidden_size, 1, bidirectional=True, batch_first=True)
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(vocab_size, embed_size)  # , _weight=torch.nn.Parameter(weights))
        self.dropout = nn.Dropout(p_drop)
        self.classifier = MLP(hidden_size *2, 7)
    def forward(self, sentence):
        #print("Sentence:",sentence)
        #sentence = make_safe(" ".join(map(lambda x: unquote(str(x)), term2list(sentence))))
        #print(sentence)
        #sentence = sentence.value.strip('"')
        
        sentence = unquote(str(sentence))
        x = tokenize_query(sentence)
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


         
def get_networks(lr, p_drop, n=8):
    #word type classifiers (Classes: Container_Source, Container_Destination, Entity, Attribute, None)
    #num_wordCategories = 5 
    nn_n0= NC(len(vocab), hidden_size, p_drop=p_drop)
    #network1 = MLP(hidden_size * 4, num_wordCategories)
    #Verb classifier (Classes: holds, increase, decrease, transfer)
    #num_verbCategories = 4	
    nn_query= QC(len(vocab), hidden_size, p_drop=p_drop)
    #network2 = MLP(hidden_size *2, num_verbCategories)
    
    names = ['nn_n0', 'nn_query']
    networks = [nn_n0, nn_query]

    networks = [(networks[i], names[i], optim.Adam(networks[i].parameters(), lr=lr)) for i in range(2)]
    return networks