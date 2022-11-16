import sys
# the mock-0.3.1 dir contains testcase.py, testutils.py & mock.py

sys.path.append('/home/savitha/Documents/MWPS/DPL_New/deepproblog-dev-master/src')
import time
import torch
from torch.autograd import Variable
from torch.nn import CrossEntropyLoss
from torch.optim import Adam
from torch.utils.data import DataLoader, Subset

from deepproblog.examples.Forth.WAP.Pretraining import get_networks,VC, WC

vocab = dict()
with open('/home/savitha/Documents/MWPS/DPL_New/deepproblog-dev-master/src/deepproblog/examples/Forth/WAP/data/vocab_inter_all.txt') as f:
    for i, word in enumerate(f):
        word = word.strip()
        # vocab[i] = word
        vocab[word] = i

hidden_size = 384
tagset_size = 3
#Load pretrained network



def get_accuracy(model, dataloader: DataLoader):
    total = 0
    correct = 0
    for input_data, gt_labels in dataloader:
        _, predicted = torch.max(model(input_data), 1)
        total += len(gt_labels)
        correct_labels = torch.eq(predicted, gt_labels)
        correct += correct_labels.sum().item()
    return correct / total


accuracies = dict()

def train(label_list, text_list, context_list):
    #label_list = label_list.to(device)
    nn_vc.train()
    total_acc, total_count = 0, len(label_list)
    log_interval = 1
    #start_time = time.time()
    for idx in range(label_list.shape[0]):
        optimizer.zero_grad()
        predicted_label = nn_vc(text_list[idx], context_list[idx])
        predicted_label = predicted_label.unsqueeze(dim=0)
        loss = criterion(predicted_label, label_list[idx].view(1))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(nn_vc.parameters(), 0.1)
        optimizer.step()
        if(predicted_label.argmax(1) == label_list[idx].view(1)):
            total_acc = total_acc+1 
        #total_count += label_list.size(0)
        if idx % log_interval == 0 and idx > 0:
            print('| epoch {:3d} |accuracy {:8.3f}'.format(epoch, total_acc/total_count))
        #     total_acc, total_count = 0, 0
        #     start_time = time.time()

def evaluate(label_list, text_list, context_list):
    nn_vc.eval()
    total_acc, total_count = 0, len(label_list)

    with torch.no_grad():
        for idx in range(label_list.shape[0]):
            predicted_label = nn_vc(text_list[idx], context_list[idx])
            predicted_label = predicted_label.unsqueeze(dim=0)
            loss = criterion(predicted_label, label_list[idx].view(1))
            if(predicted_label.argmax(1) == label_list[idx].view(1)):
                total_acc = total_acc+1 
            
    return total_acc/total_count

def get_instances(dataset):
    file1 = open(dataset, 'r')   
    label_list, text_list, context_list = [], [], []
    for l in file1:
        line = l.split(',')
        label_list.append(int(line[-1]))
        line = l.split(']')
        text = line[0]
        cont = line[1]
        sent = []
        text_w = text.split(',')
        for w in text_w:
            if '[' in w:
                w = w[1:]
            sent.append(w)
        context=[]
        cont_w = cont.split(',')
        for w in cont_w:
            if w=='':
                continue
            if '[' in w:
                w = w[1:]
            context.append(w)
        
        text_list.append(sent)
        context_list.append(context)
    label_list = torch.tensor(label_list, dtype=torch.int64)
    return label_list, text_list, context_list          
    
    
   
nn_vc = VC(len(vocab), hidden_size, p_drop=0.5) #MNIST_Net(with_softmax=False)
label_list, text_list, context_list  = get_instances("pretrain_vc.txt")
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(nn_vc.parameters(), lr=1e-3)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1, gamma=0.1)
total_accu = None
for epoch in range(1, 5):
    epoch_start_time = time.time()
    train(label_list[0:20], text_list[0:20], context_list[0:20])
    accu_val = evaluate(label_list[20:], text_list[20:], context_list[20:])
    if total_accu is not None and total_accu > accu_val:
      scheduler.step()
    else:
       total_accu = accu_val
    print('-' * 59)
    print('| end of epoch {:3d} | time: {:5.2f}s | '
          'valid accuracy {:8.3f} '.format(epoch,
                                           time.time() - epoch_start_time,
                                           accu_val))
    print('-' * 59)
torch.save(nn_vc.state_dict(), "{}_{}.pth".format('1_5', 'nn_vc_roberta_384_2'))
#         dataloader = DataLoader(subset, 4, shuffle=True)
#         dataloader_test = DataLoader(datasets["test"], 4)
#         optimizer = Adam(net.parameters(), lr=1e-3, weight_decay=1e-2)
#         criterion = CrossEntropyLoss()

#         cumulative_loss = 0
#         i = 0

#         for _ in range(4):
#             for epoch in range(10):
#                 for batch in dataloader:
#                     i += 1
#                     data, labels = batch

#                     optimizer.zero_grad()

#                     data = Variable(data)
#                     out = net(data)

#                     loss = criterion(out, labels)
#                     cumulative_loss += float(loss)
#                     loss.backward()
#                     optimizer.step()

#                     if i % 50 == 0:
#                         print("Loss: ", cumulative_loss / 100.0)
#                         cumulative_loss = 0
#             print("Accuracy", get_accuracy(net, dataloader_test))
#         accuracies[(k, nr_examples)] = get_accuracy(net, dataloader_test)
#         torch.save(net.state_dict(), "{}_{}.pth".format(k, nr_examples))

# with open("accuracies.txt", "w") as f:
#     for k in accuracies:
#         f.write("{}\t{}\n".format(k, accuracies[k]))
