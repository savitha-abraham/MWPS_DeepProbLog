#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 20 08:32:32 2021

@author: savitha
"""
import json
import random
#Dataset - SingleOp.json Analysis - schema distribution

#To change everything to lower case
# file1 = open('/home/savitha/Documents/MWPS/DPL_New/deepproblog-dev-master/src/deepproblog/examples/Forth/WAP/data/Formatteddata_SingleOpPlusAddSub/wap_inter_all.txt', 'r')
# f_write = open("/home/savitha/Documents/MWPS/DPL_New/deepproblog-dev-master/src/deepproblog/examples/Forth/WAP/data/Formatteddata_SingleOpPlusAddSub/wap_inter_all_lower.txt", "w")  

# for line in file1:
#     line = line.lower()
#     f_write.write(line)
    
# file1.close()
# f_write.close()

file1 = open('/home/savitha/Documents/MWPS/DPL_New/deepproblog-dev-master/src/deepproblog/examples/Forth/WAP/data/Formatteddata_SingleOpPlusAddSub/wap_inter_all_lower.txt', 'r')   
f_write = open("/home/savitha/Documents/MWPS/DPL_New/deepproblog-dev-master/src/deepproblog/examples/Forth/WAP/data/Formatteddata_SingleOpPlusAddSub/wap_inter_all_updated.txt", "w")  

for line in file1:
    ls = line.split('[[')
    cons = ls[1]
    pos = ls[2]
    c_w = cons.split(',')
    comb_con="["
    for i in c_w:
        if(']]' in i):
            i = i[:-2]
        elif(']' in i):
            i = i[:-1]

        if('[' in i):  
            i=i[1:]
        if(comb_con=='['):
            comb_con = comb_con+i
        else:
            comb_con = comb_con+","+i
    comb_con=comb_con[:-1]+"]"
    
    comb_pos = "["
    c_w = pos.split(',')
    for i in c_w:
        if(']]' in i):
            i = i[:-2]
        elif(']' in i):
            i = i[:-1]

        if('[' in i):  
            i=i[1:]
        if(comb_pos=='['):
            comb_pos = comb_pos+i
        else:
            comb_pos = comb_pos+","+i
    comb_pos=comb_pos[:-1]+"]"
    ls1 = line.split(')')
    updated_line = ls1[0]+","+comb_con+","+comb_pos+")."+"\n"
    f_write.write(updated_line)
file1.close()
f_write.close()
#Create vocab from list of questions
questions_list = []

file1 = open('/home/savitha/Documents/MWPS/DPL_New/deepproblog-dev-master/src/deepproblog/examples/Forth/WAP/data/Formatteddata_SingleOpPlusAddSub/singleOp_all.txt', 'r')
for line in file1:
    ls = line.split('\"')
    questions_list.append(ls[1])
    
    
print(questions_list)
file1.close()
       
#Create vocab.txt from question_list
vocab = []
vocab.append('<UNK>')
vocab.append('<NR>')
maxlen = 0
for q in questions_list:
    num_words = 0
    for i in q.split():
        i.strip()
        
        if '?' in i:
            i = i[0:-1]
        if i[-1]=='.' or i[-1]==',':
            i = i[0:-1]
        num_words=num_words+1
        if i not in vocab:
            #print('Adding '+i.lower()) 
            vocab.append(i.lower())
    if maxlen<num_words:
        maxlen=num_words
print("Max num of words in a problem:")
print(maxlen)
mylist = list(dict.fromkeys(vocab))
#print(mylist)
f_write = open("/home/savitha/Documents/MWPS/DPL_New/deepproblog-dev-master/src/deepproblog/examples/Forth/WAP/data/vocab_inter_all.txt", "w")  
for v in mylist:
    f_write.write(v+'\n')
f_write.close()
