from __future__ import print_function

import sys

sys.path.append('/home/savitha/Documents/MWPS/DPL_New/deepproblog-dev-master/src/deepproblog/examples/ParaphraseGeneration/WAP/next-word-prediction-master')


from problog.logic import unquote, make_safe, list2term, Term, term2list, Constant
from problog.extern import problog_export, problog_export_raw

from next_word_prediction import GPT2



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
    
    gpt2 = GPT2()

    # Encode a text input
    prediction = gpt2.predict_next(text, 1)

    print("{} {}".format(text, prediction))

    return Constant(prediction)     





