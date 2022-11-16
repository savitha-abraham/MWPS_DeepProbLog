from __future__ import print_function

from problog.logic import unquote, make_safe, list2term, Term, term2list, Constant
from problog.extern import problog_export, problog_export_raw



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
  


