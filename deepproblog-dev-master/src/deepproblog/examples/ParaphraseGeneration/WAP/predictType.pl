%:- use_module('next-word-prediction-master/main.py').
:- use_module('expression.py').

nn(nn_n0, [Text, N], Y, [0, 1, 2, 3, 4, 5, 6]):: net1([Text, N] , Y).


nn(nn_query, Text, Y, [0, 1, 2, 3, 4, 5, 6]):: net2(Text, Y).

%0-initial, 1-inc, 2-dec, 3-final, 4-part, 5-num_parts, 6-total



%-----------------------------------------------------------

%CF_inc
getSchema(0, 1, 3, 1, 'n0+n1'). 

getSchema(1, 0, 3, 1, 'n0+n1'). 

%CF_dec
getSchema(0, 2, 3, 2, 'n0-n1'). 
getSchema(2, 0, 3, 2, 'n1-n0'). 

%inc
getSchema(0, 3, 1, 3, 'n1-n0'). 
getSchema(3, 0, 1, 3, 'n0-n1'). 

%dec
getSchema(0, 3, 2, 4, 'n0-n1').
getSchema(3, 0, 2, 4, 'n1-n0').
 
%CI_inc
getSchema(1, 3, 0, 5, 'n1-n0'). 
getSchema(3, 1, 0, 5, 'n0-n1'). 

%CI_dec
getSchema(2, 3, 0, 6, 'n0+n1').
getSchema(3, 2, 0, 6, 'n0+n1'). 

%part
getSchema(5, 6, 4, 7, 'n1/n0'). 
getSchema(6, 5, 4, 7, 'n0/n1').
 
%num_part
getSchema(4, 6, 5, 8, 'n1/n0'). 
getSchema(6, 4, 5, 8, 'n0/n1').


%total
getSchema(4, 5, 6, 9, 'n0*n1').
getSchema(5, 4, 6, 9, 'n0*n1').


%------------------------------------------------------------

checkequal(Equation1, Equation1_p):- ground(Equation1), equal([Equation1,Equation1_p],1).
checkequal(Equation1, Equation1_p):- var(Equation1), Equation1=Equation1_p.


%-----------------------------------------------------------
predictType(WP1, Equation1,  Type) :-  
    net1([WP1, 'n0'], Cat1_n0 ),
    net1([WP1, 'n1'], Cat1_n1),
    net2([WP1], Cat1_query),
    Cat1_query \= Cat1_n0, Cat1_query \= Cat1_n1, Cat1_n1 \= Cat1_n0,
    getSchema(Cat1_n0, Cat1_n1, Cat1_query, Type, Equation1_p),
    checkequal(Equation1, Equation1_p),
    checkequal(Equation2, Equation2_p). 
    
