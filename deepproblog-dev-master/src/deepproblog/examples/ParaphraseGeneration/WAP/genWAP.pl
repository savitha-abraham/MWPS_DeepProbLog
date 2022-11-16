%:- use_module('next-word-prediction-master/main.py').
:- use_module('expression.py').

nn(nn_n0, [Text,'n0'], Y, [0, 1, 2, 3, 4, 5, 6]):: nn_n0(Text, 'n0', Y).

nn(nn_n1, [Text,'n1'], Y, [0, 1, 2, 3, 4, 5, 6]):: nn_n1(Text, 'n1', Y).

nn(nn_n1, [Text, 'query'], Y, [0, 1, 2, 3, 4, 5, 6]):: nn_query(Text, 'query', Y).

%0-initial, 1-inc, 2-dec, 3-final, 4-part, 5-num_parts, 6-total

generate(Cat, Gen_Wp) :- s11(Container, EN1, EN2, S1),  s12(S1, S2), sbar_11(S2, EN1, EN2, Gen_Wp).

%----------------s1------------------------------%
s11(Container, EN1, EN2, VP11) :- writeln("At s11:"), np_11(Container), vp_11(Container, VP11, EN1, EN2).

np_11(Container):- writeln("At np_11:"), nnp_11(Container).
%get_next predicts next word given the predicted sequence so far, the pos tag of the next word.
nnp_11(Container) :- writeln("At nnp_11:"), get_next(["<|endoftext|>", "NN"], Container), writeln("After get_next:").


vp_11(Gen, NP, EN1, EN2) :- vbz_11(Gen, VBZ), np_12(VBZ, NP, EN1, EN2).
vbz_11(Gen, VBZ) :- get_next([Gen,"VB"], Verb), concat([Gen,Verb], VBZ).
np_12(Gen, NP, EN1, EN2) :- sbar_12(Gen, NP, EN1, EN2).

sbar_12(Gen, NP, EN1, EN2) :-  advp_11(Gen, AD), np_13(AD, NP, EN1, EN2).
advp_11(Gen, AD) :- dt_11(Gen, AD).
dt_11(Gen, AD) :- get_next([Gen,"DT"], Dt), concat([Gen,Dt], AD).

np_13(Gen, NN, EN1, EN2) :- nn_11(Gen, NP, EN1), nns_11(NP, NN, EN2) .

nn_11(Gen, NP, N) :- get_next([Gen,"NN"], N), concat([Gen,N], NP).
nns_11(Gen, NNS, N) :- get_next([Gen,"NN"], N), concat([Gen,N,"."], NNS).


%--------------------S2-------------------------------
s12(Gen, VP) :- np_121(Gen, NP ), vp_121(NP, VP).
np_121(Gen, NP) :- prp_121(Gen, NP).
prp_121(Gen, PP) :- get_next([Gen,"PRP"], P), concat([Gen,P], PP).

vp_121(Gen, Ad) :- vbz_121(Gen, Vbz), adjp_121(Vbz, Ad).
vbz_121(Gen, Vbz) :-  get_next([Gen,"VB"], V), concat([Gen,V], Vbz).
adjp_121(Gen, Ad) :- rb_121(Gen, Rb), rbr_121(Rb, Ad).
rb_121(Gen, Rb) :- concat([Gen,'n0'], Rb).

rbr_121(Gen, Rbr) :- get_next([Gen,"RB"], R), concat([Gen,R, "."], Rbr).


%---------------S3---------------------------------------    

sbar_11(Gen, EN1, EN2, S3) :-  whadjp_131(Gen, Wh), s_131(Wh, EN1,EN2, S3).
whadjp_131(Gen, Wh) :- wrb_131(Gen, W), jj_131(W, Wh).

wrb_131(Gen, W) :- get_next([Gen,"WRB"], Wh), concat([Gen,Wh], W).
jj_131(Gen, Wh) :- get_next([Gen,"JJ"], JJ), concat([Gen,JJ], Wh).

s_131(Gen, EN1, EN2, S) :- np_131(Gen, EN1, EN2, NP), vp_131(NP, EN2, S).
np_131(Gen, EN1, EN2, NP) :- nn_131(Gen, EN1, NN), nns_131(NN, EN2, NP).
nn_131(Gen, EN1, NN) :- concat([Gen,EN1], NN).
nns_131(Gen, EN2, NP) :- concat([Gen,EN2], NP).

vp_131(Gen, EN2, Vp) :- vbd_131(Gen, V), np_132(V, N), advp_131(N, A), sbar_131(A, EN2, Vp).

vbd_131(Gen, Vb) :- get_next([Gen,"VB"], V), concat([Gen,V], Vb).

np_132(Gen, N) :- prp_131(Gen, N).
prp_131(Gen, Pr) :- get_next([Gen,"PRP"], P), concat([Gen,P], Pr).

advp_131(Gen, A) :- rb_131(Gen, A).
rb_131(Gen, Rb) :- get_next([Gen,"RB"], R), concat([Gen,R], Rb).

sbar_131(Gen, EN2, Sb) :- in_131(Gen, In), s_132(In, EN2, Sb).
in_131(Gen, In) :- get_next([Gen,"IN"], I), concat([Gen,I], In).
s_132(Gen, EN2, Sb) :- np_133(Gen, Np), vp_132(Np, EN2, Sb).
np_133(Gen, Np) :- prp_132(Gen, Np).
prp_132(Gen, Pr) :- get_next([Gen,"PRP"], P), concat([Gen,P], Pr).
vp_132(Gen, EN2, Sb) :- vbz_131(Gen, Vp), np_134(Vp, EN2, Np), advp_132(Np, Sb).

vbz_131(Gen, Vp) :- get_next([Gen,"verb_h"], V), concat([Gen,V], Vp).
np_134(Gen, EN2, Np) :- nn_132(Gen, NN), nns_132(NN, EN2, Np).
nn_132(Gen, NN) :- concat([Gen,'n1'], NN).
nns_132(Gen, EN2, Np) :- concat([Gen,EN2], Np).

advp_132(Gen, Sb) :- rb_132(Gen, Sb).

rb_132(Gen, Sb) :-  get_next([Gen,"RB"], R), concat([Gen,R, "?"], Sb).  

%-----------------------------------------------------------

%CF_inc
getSchema(0, 1, Cat_query, 1, 'n0+n1'). 
getSchema(1, 0, Cat_query, 1, 'n0+n1'). 

%CF_dec
getSchema(0, 2, Cat_query, 2, 'n0-n1'). 
getSchema(2, 0, Cat_query, 2, 'n1-n0'). 

%inc
getSchema(0, 3, 1, 3, 'n1-n0'). 
getSchema(3, 0, 1, 3, 'n0-n1'). 

%dec
getSchema(0, 3, 2, 4, 'n0-n1').
getSchema(3, 0, 2, 4, 'n1-n0').
 
%CI_inc
getSchema(1, 3, Cat_query, 5, 'n1-n0'). 
getSchema(3, 1, Cat_query, 5, 'n0-n1'). 

%CI_dec
getSchema(2, 3, Cat_query, 6, 'n0+n1'). 
getSchema(3, 2, Cat_query, 6, 'n0+n1'). 


%-----------------------------------------------------------
detectSim(WP1, WP2, Schema_Type1, Schema_Type2, Equation1, Equation2, 1) :-  

    nn_n0(WP1, 'n0', Cat1_n0 ), 
    nn_n0(WP1, 'n1', Cat1_n1),
    nn_query(WP1, 'query', Cat1_query),
    nn_n0(WP2, 'n0', Cat2_n0 ), 
    nn_n0(WP2, 'n1', Cat2_n1),
    nn_query(WP2, 'query', Cat2_query) 
    getSchema_Template(Cat1_n0, Cat1_n1, Cat1_query, Schema_Type1, Equation1),
    getSchema_Template(Cat2_n0, Cat2_n1, Cat2_query, Schema_Type2, Equation2),
    Schema_Type1 == Schema_Type2, 
    writeln("Schema_Type1:"), 
    writeln(Schema_Type1),
    writeln("Schema_Type2:"), 
    writeln(Schema_Type2).
%-----------------------------------------------------------
detectSim(WP1, WP2, Schema_Type1, Schema_Type2, Equation1, Equation2, 0) :-  

    nn_n0(WP1, 'n0', Cat1_n0 ), 
    nn_n0(WP1, 'n1', Cat1_n1),
    nn_query(WP1, 'query', Cat1_query),
    nn_n0(WP2, 'n0', Cat2_n0 ), 
    nn_n0(WP2, 'n1', Cat2_n1),
    nn_query(WP2, 'query', Cat2_query) 
    getSchema_Template(Cat1_n0, Cat1_n1, Cat1_query, Schema_Type1, Equation1),
    getSchema_Template(Cat2_n0, Cat2_n1, Cat2_query, Schema_Type2, Equation2),
    Schema_Type1 =\= Schema_Type2, 
    writeln("Schema_Type1:"), 
    writeln(Schema_Type1),
    writeln("Schema_Type2:"), 
    writeln(Schema_Type2).