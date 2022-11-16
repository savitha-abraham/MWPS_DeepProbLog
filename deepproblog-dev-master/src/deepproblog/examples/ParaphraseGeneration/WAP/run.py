import sys
# the mock-0.3.1 dir contains testcase.py, testutils.py & mock.py

sys.path.append('/home/savitha/Documents/MWPS/DPL_New/deepproblog-dev-master/src')

import os
import random
import numpy

#from deepproblog.examples.Forth.WAP.Pretraining import get_networks,VC, WC
from genWAP_Net import get_networks, NC, QC

import torch
import torch.optim as optim
from deepproblog.network import Network
from deepproblog.evaluate import get_confusion_matrix
from deepproblog.examples.Forth import EncodeModule
from deepproblog.dataset import DataLoader, QueryDataset
from deepproblog.model import Model
from deepproblog.engines import ApproximateEngine
from deepproblog.engines import ExactEngine
from deepproblog.heuristics import ucs, geometric_mean
from deepproblog.utils import get_configuration, format_time_precise, config_to_string
from deepproblog.train import train_model
from json import dumps


torch.manual_seed(3)
os.environ['PYTHONHASHSEED'] = str(4)
numpy.random.seed(10)
random.seed(10)

#i = int(sys.argv[1])
##parameters = {'method': ['gm'], 'N': [1,2,3], 'pretrain': [0, 16],'exploration': [False,True], 'run': range(5)}
#parameters = {'method': ['gm'], 'N': [3], 'pretrain': [0], 'exploration': [False,True], 'run': range(5)}
#configuration = get_configuration(parameters, i)
#torch.manual_seed(configuration['run'])

def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    torch.manual_seed(10)
    numpy.random.seed(10)
    random.seed(10)



name = '20Pre-PD_' + format_time_precise()
#train_queries = QueryDataset('data/dpl_data_detection/train.pl')
train_queries = QueryDataset('data/DPL_pairs_PD/train.pl')
dev_queries = QueryDataset('data/DPL_pairs_PD/val.pl')
##test_queries = QueryDataset('data/DPL_pairs_PD/test.pl')

##train_queries = QueryDataset('data/DPL_Type/train.pl')
##dev_queries = QueryDataset('data/DPL_Type/val.pl')
##test_queries = QueryDataset('data/DPL_Type/test.pl')

#test_queries = QueryDataset('data/wap_inter_all_test.pl')
##Get vocabulary u=if pre-trained models used
vocab = dict()
with open('data/vocab.txt') as f:
    for i, word in enumerate(f):
        word = word.strip()
        # vocab[i] = word
        vocab[word] = i

hidden_size = 512
tagset_size = 7

##PRETRAINED NETWORKS
nn_n0= NC(len(vocab), hidden_size, p_drop=0.5)
nn_query= QC(len(vocab), hidden_size, p_drop=0.5)
#Pretraining

nn_n0.load_state_dict(torch.load("snapshot/nn_n0_Pretrained.pth"))
nn_query.load_state_dict(torch.load("snapshot/nn_query_Pretrained.pth"))
names = ['nn_n0', 'nn_query']
networks = [nn_n0,nn_query]
networks = [(networks[i], names[i], optim.Adam(networks[i].parameters(), lr=1e-3)) for i in range(2)]                      

#Load pretrained network
##nn_category = SC(len(vocab), hidden_size, tagset_size)
#Pretraining
##nn_sc.load_state_dict(torch.load("models/1_5_nn_wc_roberta_384_2.pth"))
##names = ['nn_sc']
##networks = [nn_category]
##networks = [(networks[i], names[i], optim.Adam(networks[i].parameters(), lr=1e-3)) for i in range(1)]                      

##No pretraining
##networks = get_networks(0.005, 0.5)

train_networks = [Network(x[0], x[1], x[2]) for x in networks]
#test_networks = [Network(networks[0][0], networks[0][1])]+ [Network(x[0], x[1], k=1) for x in networks[2:]]

##model = Model('predictType.pl', train_networks)
model = Model('detSim.pl', train_networks)

##model = Model('wap_inter_Logic1 (working).pl', train_networks)
model.set_engine(ApproximateEngine(model, 1, geometric_mean, exploration=False))
#model.set_engine(ExactEngine(model), cache=True)

#test_model = Model('wap_inter_Logic.pl', test_networks)
#test_model.set_engine(ApproximateEngine(test_model,1,geometric_mean, exploration=False), cache=False)

#train_obj = train_model(model, DataLoader(train_queries, 2),1, log_iter=1, worker_init_fn=seed_worker,
#                        test=lambda x: [('Accuracy', get_confusion_matrix(test_model, dev_queries).accuracy())],
#                        test_iter=1)


#train_obj = train_model(model, DataLoader(train_queries, 1),1, log_iter=1, worker_init_fn=seed_worker)
train_obj = train_model(model, DataLoader(train_queries, 100), 20, log_iter=100, worker_init_fn=seed_worker,
                        test=lambda x: [('Accuracy', get_confusion_matrix(x, dev_queries).accuracy())],
                        test_iter=1)
model.save_state('snapshot/'+name+'.pth')
print('Model saved..')

#Also save the neural network states separately 
for net in train_networks:
    net1 = net.network_module
    torch.save(net1.state_dict(), "snapshot/{}_{}.pth".format(net.name, 'PD')) 

##train_obj.logger.comment(dumps(model.get_hyperparameters()))
#train.logger.comment('Accuracy {}'.format(evaluate_dataset(model, test, verbose=1).accuracy()))
##train_obj.logger.comment('Accuracy {}'.format(get_confusion_matrix(model, dev_queries, verbose=1).accuracy()))
#train_obj.logger.write_to_file('log/' + name)

#test_model.save('')

##TESTING WITH SAVED MODEL
##model.load_state('snapshot/roberta_wap_384h_2L_E10.pth')
##print('Test Set Performance Normal..')
##acc_test_n = get_confusion_matrix(model, test_queries_n).accuracy()
##print("Accuracy normal:",acc_test_n)

##print('Test Set Performance Compositional..')
##acc_test_c = get_confusion_matrix(model, test_queries_c).accuracy()
##print("Accuracy compositional:",acc_test_c)