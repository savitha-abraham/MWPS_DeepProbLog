from __future__ import print_function
import sys
sys.path.append('/home/savitha/Documents/MWPS/DPL_New/deepproblog-dev-master/src/deepproblog/examples/ParaphraseGeneration/WAP/next-word-prediction-master')

from problog.logic import unquote, make_safe, list2term, Term, term2list, Constant
from problog.extern import problog_export, problog_export_raw

from next_word_prediction import GPT2


#get_next(["<Start>", "noun"], Container)
#@problog_export("+list", "-str")
def get_next(terms):
    print("I am in get_next:", terms)
    prediction = ""
    text = ""
    pos = ""
    count = 0
    for i in term2list(terms):
        if str(type(i)) == "<class 'bytes'>":
            if count == 0:
                text=text+unquote(str(i.decode('ascii'))) 
            else:
                pos = pos+unquote(str(i.decode('ascii')))
            
        else:
            if count == 0:
                text = text + str(i)
            else:
                pos = pos + str(i)
        count = count+1
    
    gpt2 = GPT2()
    print("Created gpt2 instance..")
    # # Encode a text input
    prediction = gpt2.predict_next(text, pos, 1000)

    print("{} {}".format(text, prediction))
    #prediction = "Neena"
    return Constant(prediction)     



#@problog_export("+list", "-str")
def concat(terms):
    str_cat = ""
    for i in term2list(terms):
        if str(type(i)) == "<class 'bytes'>":
            str_cat=str_cat+unquote(str(i.decode('ascii'))) + " "
            
        else:
            
            str_cat = str_cat+ str(i) +" "
    return Constant(str_cat) 
#make_safe("".join(map(lambda x: unquote(x.decode('ascii')), terms)))


#@problog_export("+list", "-int")
def isEquation(X):
    X = term2list(X)
    x1 = unquote(X[0])
    
    if "=" in x1:
        x_split = x1.split("=")
        if x_split[0] == "X":
            return Constant(1)
        else:
            return Constant(0)  
    else:
        return Constant(0)
   
#@problog_export("+list", "-int")
def equal(X):
    E = term2list(X)
    E1 = unquote(str(E[0]))
    E1 = E1.split("=")[1]
    E2 = unquote(str(E[1]))
    if "+" in E1:
        if "+" in E2:
            return Constant(1)
        else:
            return Constant(0)
    elif "-" in E1:
        if E1==E2:
            return Constant(1)
        else:
            return Constant(0)
                
    elif "*" in E1:
        if "*" in E2:
            return Constant(1)
        else:
            return Constant(0)
    elif "/" in E1:
        if E1==E2:
            return Constant(1)
        else:
            return Constant(0)
    
#@problog_export("+list", "+list", "-int")
def contains(X, String):
    X = term2list(X)
    String = term2list(String)
    n=0
    x1 = unquote(X[0])
    S =  unquote(String[0])
    for i in S:
        if i==x1:
            n=n+1
   
    
    return Constant(n)	

#@problog_export("+list", "-float")
def evaluate(eq):
    eq= term2list(eq)
    equation = unquote(eq[0])
    n = len(equation)
    sign = 1
    coeff = 0
    total = 0
    i = 0
  
    # Traverse the equation
    for j in range(0, n) :
      
        if (equation[j] == '+' or equation[j] == '-') :
          
            if (j > i) :
                total = (total + sign * float(equation[i: j]))
            i = j
          
        # For cases such 
        # as: x, -x, +x
        elif (equation[j] == 'X') :
          
            if ((i == j) or
                equation[j - 1] == '+') :
                coeff += sign
            elif (equation[j - 1] == '-') :
                coeff = coeff - sign
            else :
                coeff = (coeff + sign * float(equation[i: j]))
            i = j + 1
          
        # Flip sign once 
        # '=' is seen
        elif (equation[j] == '=') :
          
            if (j > i) :
                total = (total + sign * float(equation[i: j]))
            sign = -1
            i = j + 1
          
    # There may be a number
    # left in the end
    if (i < n) :
        total = (total + sign * float(equation[i: len(equation)]))
  
    
    ans = -total / coeff
    return Constant(ans) 
  


