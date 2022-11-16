import re

def isDigit(x):
    if x.isdigit():
        return True
    try:
        float(x)
        return True
    except ValueError:
        return False
# for filename in ["train", "test", "dev"]:
#     out_lines = []
#     with open(filename + ".txt") as f:
#         for line in f:
#             solution, question = line.rstrip().split("\t")
#             solution = int(float(solution))
#             numbers = [int(i) for i in re.findall(" (\d+)", question)]
#             out_lines.append(
#                 'wap("{}",{},{},{},{}).'.format(question, *numbers, solution)
#             )
#     with open("{}.pl".format(filename), "w") as f:
#         f.write("\n".join(out_lines))
file1 = open('SingleOp/seq2seq_test_c.txt', 'r') 
f_write = open("SingleOp/seq2seq_test_cpre.txt", "w")  

for line in file1:
    #print(line)
    ls = line.split('"')
    #print(ls)
    prob = ls[1]
    #print(prob)
    eqn1 = ls[2].split(",")[1]
    #convert to preorder
    #eqn1 = eqn1.strip("\n")
    
    #print(eqn)
    words = prob.split(" ")
    num_var = {}
    i=1
    print(prob)
    for word in words:
        if word[0]=='$':
            word = word[1:]
        if word[-1]=='.' or word[-1]==',' or word[-1]=='?' or word[-1]=='\'':
            word = word[0:-1]
        if isDigit(word):
            var = "n_"+str(i)
            num_var[word]=var
            i=i+1
    for num in num_var:
        prob = re.sub(r'\b'+num+r'\b', num_var[num],prob)
        if num in eqn:
            eqn = re.sub(r'\b'+num+r'\b', num_var[num],eqn)
    instance = prob+'\t'+eqn
    print(instance)
    f_write.write(instance+'\n')
            
    
file1.close()
f_write.close()
    
    