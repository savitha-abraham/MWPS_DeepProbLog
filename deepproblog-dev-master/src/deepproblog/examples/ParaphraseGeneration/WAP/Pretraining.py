#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug  3 13:46:43 2021

@author: savitha
"""
from functools import lru_cache

import torch
import torch.nn as nn
import torch.optim as optim
from deepproblog.utils import count_parameters
from torch.autograd import Variable
from problog.logic import unquote, make_safe, term2list
from deepproblog.network import Network
from deepproblog.utils.standard_networks import MLP
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from ..WAP.components.contextual_embeddings import BertEncoder, RobertaEncoder
from ..WAP.components.encoder import Encoder

vocab = dict()
hidden_size = 384
embed_size = 768
tagset_size = 3
with open('/home/savitha/Documents/MWPS/DPL_New/deepproblog-dev-master/src/deepproblog/examples/Forth/WAP/data/vocab_inter_all.txt') as f:
    for i, word in enumerate(f):
        word = word.strip()
        # vocab[i] = word
        vocab[word] = i

def sort_by_len(seqs, input_len, device=None, dim=1):
	orig_idx = list(range(seqs.size(dim)))
	# pdb.set_trace()

	# Index by which sorting needs to be done
	sorted_idx = sorted(orig_idx, key=lambda k: input_len[k], reverse=True)
	sorted_idx= torch.LongTensor(sorted_idx)
	if device:
		sorted_idx = sorted_idx.to(device)

	sorted_seqs = seqs.index_select(1, sorted_idx)
	sorted_lens=  [input_len[i] for i in sorted_idx]

	# For restoring original order
	orig_idx = sorted(orig_idx, key=lambda k: sorted_idx[k])
	orig_idx = torch.LongTensor(orig_idx)
	if device:
		orig_idx = orig_idx.to(device)
	return sorted_seqs, sorted_lens, orig_idx


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
        if word[-1]=='.' or word[-1]==',' or word[-1]=='?' or word[-1]=='\'':
            word = word[0:-1]
        if word[0]=='\'':
            word = word[1:]
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

class VC(nn.Module):
    def __init__(self, vocab_size, hidden_size, p_drop=0.0):
        super(VC, self).__init__()
        self.lstm = nn.GRU(embed_size, hidden_size, 2, bidirectional=True, batch_first=True)
        self.hidden_size = hidden_size
        self.embedding = RobertaEncoder('roberta-base', torch.device("cpu"))
        #self.embedding = nn.Embedding(vocab_size, embed_size)  # , _weight=torch.nn.Parameter(weights))
        self.dropout = nn.Dropout(0.1)
        self.classifier = MLP(hidden_size *4, 3)
        
    def forward(self, sentence, context):
        #sentence = make_safe(" ".join(map(lambda x: unquote(str(x)), sentence)))
        #context = make_safe(" ".join(map(lambda x: unquote(str(x)), context)))
        sentence = make_safe(" ".join(map(lambda x: unquote(str(x)), term2list(sentence))))
        context = make_safe(" ".join(map(lambda x: unquote(str(x)), term2list(context))))
        context = unquote(context)
        sentence = unquote(sentence)
        #print("VC:")
        #print(sentence)
        #x = tokenize(sentence)
        #cont = tokenize(context)
        #seq_len = len(x)
        #cont_seq_len = len(cont)
        #x = torch.LongTensor(x).unsqueeze(0)
        #cont = torch.LongTensor(cont).unsqueeze(0)
        #x = self.embedding(x)
        embeds_context, seq_len1 = self.embedding([context])
        embeds_sent, seq_len2 = self.embedding([sentence])
        lstm_Sout, _ = self.lstm(embeds_sent)
        lstm_Cout, _ = self.lstm(embeds_context)
        lstm_Sout = lstm_Sout.view(lstm_Sout.shape[1], 2, self.hidden_size)
        lstm_Cout = lstm_Cout.view(lstm_Cout.shape[1], 2, self.hidden_size)
        
        x_i11 = torch.cat([lstm_Sout[-1, 0, ...], lstm_Cout[-1, 0, ...]])
        x_i12 = torch.cat([lstm_Sout[0, 1, ...], lstm_Cout[0, 1, ...]])
        
        #x1 = x[-1, 0, ...]
        #x2 = x[0, 1, ...]
        WP_repr = torch.cat([x_i11, x_i12])
        y = self.classifier(WP_repr)
        
        return y


class WC(nn.Module):
    def __init__(self, vocab_size, hidden_size, tagset_size):
        super(WC, self).__init__()
        self.hidden_dim = hidden_size
        self.emb_size = embed_size
        #self.cell_type = 'lstm'
        self.depth = 1
        #self.bidirectional = True
        self.dropout = 0.1
        #self.word_embeddings = nn.Embedding(vocab_size, embed_size)
        self.word_embeddings = RobertaEncoder('roberta-base', torch.device("cpu"))
        # The LSTM takes word embeddings as inputs, and outputs hidden states
        # with dimensionality hidden_dim.
        self.lstm = nn.LSTM(embed_size, hidden_size,2,bidirectional=True, batch_first=True)
        
        # The linear layer that maps from hidden state space to tag space
        self.classifier = MLP(hidden_size*4, tagset_size)
        
    def forward(self, sentence, context, I1):
        #sentence = make_safe(" ".join(map(lambda x: unquote(str(x)), sentence)))
        #context = make_safe(" ".join(map(lambda x: unquote(str(x)), context)))
        sentence = make_safe(" ".join(map(lambda x: unquote(str(x)), term2list(sentence))))
        context = make_safe(" ".join(map(lambda x: unquote(str(x)), term2list(context))))
        context = unquote(context)
        sentence = unquote(sentence)
        
        #unquote(str(sentence)).strip('"')
        I1 = int(I1)
        #sent = tokenize(sentence)
        #cont = tokenize(context)
        
        #cont_seq_len = len(cont)
        #sent_seq_len = len(sent)

        #Convert to list to tensor and get sequence of embedding - input: NxL-->NxLx256
        #sent = torch.LongTensor(sent).unsqueeze(0)
        #cont = torch.LongTensor(cont).unsqueeze(0)
        
        embeds_context, seq_len1 = self.word_embeddings([context])
        embeds_sent, seq_len2 = self.word_embeddings([sentence])
        #print(embeds_context.shape) ## cont_seq_len x batch size x emb_size(768)
        #print(embeds_sent.shape) ## # sent_seq_len x batch size x emb_size(768)
        #embeds_context = embeds_context.transpose(0,1) 
        #embeds_sent = embeds_sent.transpose(0,1) 
        
        #sorted_sent_seqs, sorted_sent_len, sent_orig_idx = sort_by_len(embeds_sent, seq_len2, torch.device("cpu"))
        #sorted_cont_seqs, sorted_cont_len, cont_orig_idx = sort_by_len(embeds_context, seq_len1, torch.device("cpu"))
        #print(sorted_cont_seqs.shape) ## cont_seq_len x batch size x emb_size(768)
        #print(sorted_sent_seqs.shape) ## # sent_seq_len x batch size x emb_size(768)
        # sorted_sent_seqs: Tensor [max_len x BS x emb1_size]
		# seq_len1: List [BS]
		# sent_orig_idx: Tensor [BS]
        
        
        #embeds_context = self.word_embeddings(cont)
        #embeds_sent = self.word_embeddings(sent)
        
        #Lstm encoding--> NxLx768-->NxLx1024
        #lstm_Cout_f,lstm_Cout_b, _ = self.encoder(embeds_context, sorted_cont_len, cont_orig_idx)
        #lstm_Sout_f,lstm_Sout_b,_ = self.encoder(sorted_sent_seqs, sorted_sent_len, sent_orig_idx)
        lstm_Cout, _ = self.lstm(embeds_context)
        lstm_Sout, _ = self.lstm(embeds_sent)
        
        #Get forward and reverse separately Lx2x512
        lstm_Sout = lstm_Sout.view(lstm_Sout.shape[1], 2, hidden_size)
        lstm_Cout = lstm_Cout.view(lstm_Cout.shape[1], 2, hidden_size)
        
        #lstm_Sout = lstm_Sout.view(sent_seq_len, 2, hidden_size)
        #lstm_Cout = lstm_Cout.view(cont_seq_len, 2, hidden_size)
        
        #shape of lstm_Sout[-1,0,...] is [512]
        #Concatenate fwd emb of each word in sentence with forward contextual emb of ith word in context
        #[1024,1024]
        
        x_i11 = torch.cat([lstm_Sout[-1, 0, ...], lstm_Cout[I1, 0, ...]])
        x_i12 = torch.cat([lstm_Sout[0, 1, ...], lstm_Cout[I1, 1, ...]])
        #shape is 2048
        x_i1 = torch.cat([x_i11, x_i12])
        
        entity_scores = self.classifier(x_i1)
        
        return entity_scores

def get_networks(lr, p_drop, n=8):
    #word type classifiers (Classes: Container_Source, Container_Destination, Entity, Attribute, None)
    #num_wordCategories = 5 
    nn_wc= WC(len(vocab), hidden_size, tagset_size)
    #network1 = MLP(hidden_size * 4, num_wordCategories)
    #Verb classifier (Classes: holds, increase, decrease, transfer)
    #num_verbCategories = 4	
    nn_vc= VC(len(vocab), hidden_size, p_drop=p_drop)
    #network2 = MLP(hidden_size *2, num_verbCategories)
    
    names = ['nn_wc', 'nn_vc']
    networks = [nn_wc, nn_vc]

    networks = [(networks[i], names[i], optim.Adam(networks[i].parameters(), lr=lr)) for i in range(2)]

    return networks
