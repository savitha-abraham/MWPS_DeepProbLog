import sys
# the mock-0.3.1 dir contains testcase.py, testutils.py & mock.py

sys.path.append('/home/savitha/Documents/MWPS/DPL_New/deepproblog-dev-master/src')

import os
import random
import numpy

from deepproblog.examples.Forth.WAP.Pretraining import get_networks,VC, WC
#from deepproblog.examples.Forth.WAP.wap_inter_NP import get_networks
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

name = 'roberta_wap_384h_2L_' + format_time_precise()
#name = 'nn_embedding'

train_queries = QueryDataset('data/Formatteddata_SingleOpPlusAddSub/wap_inter_all_train.pl')
dev_queries = QueryDataset('data/Formatteddata_SingleOpPlusAddSub/wap_inter_all_dev.pl')

test_queries_c = QueryDataset('data/Formatteddata_SingleOpPlusAddSub/wap_inter_all_test_c.pl')

test_queries_n = QueryDataset('data/Formatteddata_SingleOpPlusAddSub/wap_inter_all_test_n.pl')

#test_queries = QueryDataset('data/wap_inter_all_test.pl')
#Get vocabulary
vocab = dict()
with open('data/vocab_inter_all.txt') as f:
    for i, word in enumerate(f):
        word = word.strip()
        # vocab[i] = word
        vocab[word] = i

hidden_size = 384
tagset_size = 3
#Load pretrained network
nn_wc = WC(len(vocab), hidden_size, tagset_size)
nn_vc = VC(len(vocab), hidden_size, 0.1)

nn_wc.load_state_dict(torch.load("models/1_5_nn_wc_roberta_384_2.pth"))
nn_vc.load_state_dict(torch.load("models/1_5_nn_vc_roberta_384_2.pth"))

#nn_wc.load_state_dict(torch.load("models/1_25_nn_wc.pth"))
#nn_vc.load_state_dict(torch.load("models/1_25_nn_vc.pth"))

names = ['nn_wc', 'nn_vc']
networks = [nn_wc, nn_vc]
networks = [(networks[i], names[i], optim.Adam(networks[i].parameters(), lr=1e-3)) for i in range(2)]                      
#networks = get_networks(0.005, 0.5)
train_networks = [Network(x[0], x[1], x[2]) for x in networks]
#test_networks = [Network(networks[0][0], networks[0][1])]+[Network(networks[1][0], networks[1][1])] + [Network(x[0], x[1], k=1) for x in networks[2:]]

model = Model('wap_inter_Logic1_Simplify.pl', train_networks)
model.set_engine(ApproximateEngine(model, 1, geometric_mean, exploration=False))
#model.set_engine(ExactEngine(model), cache=True)


train_obj = train_model(model, DataLoader(train_queries, 10),20, log_iter=100, worker_init_fn=seed_worker,
                        test=lambda x: [('Accuracy', get_confusion_matrix(x, dev_queries).accuracy())],
                       test_iter=1)
model.save_state('snapshot/'+name+'.pth')
print('Model saved..')
##train_obj.logger.comment(dumps(model.get_hyperparameters()))
##train.logger.comment('Accuracy {}'.format(evaluate_dataset(model, test, verbose=1).accuracy()))
##train_obj.logger.comment('Accuracy {}'.format(get_confusion_matrix(model, dev_queries, verbose=1).accuracy()))
#train_obj.logger.write_to_file('log/' + name)

#test_model.save('')

model.load_state('snapshot/roberta_wap_384h_2L_E10.pth')
model.load_state('snapshot/nn_embedding.pth')

print('Test Set Performance Normal..')
acc_test_n = get_confusion_matrix(model, test_queries_n).accuracy()
print("Accuracy normal:",acc_test_n)

print('Test Set Performance Compositional..')
acc_test_c = get_confusion_matrix(model, test_queries_c).accuracy()
print("Accuracy compositional:",acc_test_c)

#test_model = Model('wap_inter_Logic.pl', test_networks)
#test_model.set_engine(ApproximateEngine(test_model,1,geometric_mean, exploration=False), cache=False)


#train_obj = train_model(model, DataLoader(train_queries, 2),1, log_iter=1, worker_init_fn=seed_worker,
#                        test=lambda x: [('Accuracy', get_confusion_matrix(test_model, dev_queries).accuracy())],
#                        test_iter=1)


#train_obj = train_model(model, DataLoader(train_queries, 1),1, log_iter=1, worker_init_fn=seed_worker)

