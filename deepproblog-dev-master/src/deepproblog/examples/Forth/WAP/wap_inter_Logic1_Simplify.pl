% NLP file - given to me by Savitha Sam Abraham for a question
:- use_module('expression.py').

%word tagging
nn(nn_wc, [Text,CS,I], Y, [0, 1, 2]):: net1(Text,CS,I,Y).
%verb classifier
nn(nn_vc, [Text,CS], Y, [0, 1, 2]):: net2(Text,CS,Y).

vc(0, holds).
vc(1, increase).
vc(2, decrease).

wc(0, container_source).
wc(1, entity).
wc(2, none).


almost_equal(X,Y) :- ground(Y), T1 is (X-Y), abs(T1) < 0.0001.
almost_equal(X,Y) :- var(Y), Y is float(X).


match([H|_],0,H).
match([_|T],N,H) :-N>0, N1 is N-1,match(T,N1,H).

memberCheckSimple([], H, -1, -1, 0).
memberCheckSimple([[H, I1, N1]|T], H, I1, N1, 1).
memberCheckSimple([[H|T1]|T], Elem, I1, N1, Flag) :- Elem \= H,memberCheckSimple(T, Elem, I1, N1, Flag).  
checkQuantity(Ind, I1,N1, [H|T], Flag):- memberCheckSimple([H|T], Ind, I1, N1, Flag).

cq(Ind,I, X, Quantities):- checkQuantity(Ind,I, X, Quantities, 1).
cq(Ind,I, "X", Quantities):- checkQuantity(Ind,I, X, Quantities, 0).

findFirstQ([[H,I1,N1]|Tail], H).

getEnd([], [], I, Prev).
getEnd([H|TS], [E|T], I, Prev):- N is (I+1), length(H, L1), E is (Prev+L1), getEnd(TS, T, N, E).

checkAllowed(container_source, np, 1).
checkAllowed(container_destn, np, 1).
checkAllowed(entity, np, 1).

getWord(Ends,TS, P, Type, W,CS, CP):- 
    between(0,Ends, I), match(CP, I, PO),checkAllowed(Type, PO, 1), 
    net1(TS,CS,I, T), wc(T, Type), match(CS, I, W).

prV(TS, CS, V):- net2(TS, CS, T),  vc(T, V).  

%sentence maps to repr holds(Ends,CS,CP,Wc, X,  We) 
reprH(Ends,Ind,TS,  P, Quantities,  holds, Wc, X,  We,0,CS, CP):-   
    prV(TS, CS, holds),  getWord(Ends,TS, P, container_source, Wc,CS, CP),  
    getWord(Ends,TS, P, entity, We,CS, CP),   Wc\=We,  cq(Ind,I, X, Quantities). 

%Increase 
reprI(Ends,Ind,TS,  P,Quantities,  increase, Wc, X, We,0,CS, CP):-  
    prV(TS, CS, increase), getWord(Ends,TS, P, container_source, Wc,CS, CP),  
    getWord(Ends,TS, P, entity, We,CS, CP), Wc\=We,  cq(Ind,I, X, Quantities).


%Decrease 
reprD(Ends,Ind,TS,  P,Quantities, decrease, Wc, X,  We,0,CS, CP):- 
    prV(TS, CS, decrease), getWord(Ends,TS, P, container_source, Wc,CS, CP),  
    getWord(Ends,TS, P,  entity, We,CS, CP), Wc\=We,  cq(Ind,I, X, Quantities).



getRepr(Ends,CS,CP,Ind,TS,P, Quantities, holds, Wc, X, We):- 
    reprH(Ends,Ind,TS,  P, Quantities,  holds, Wc, X,  We, 0,CS, CP). 
getRepr(Ends,CS,CP,Ind,TS, P,Quantities, increase, Wc, X, We):- 
    reprI(Ends,Ind,TS,  P,Quantities,  increase, Wc, X,  We, 0,CS, CP). 
getRepr(Ends,CS,CP,Ind,TS, P,Quantities, decrease, Wc, X, We):- 
    reprD(Ends,Ind,TS,  P,Quantities, decrease, Wc, X,  We, 0,CS, CP). 



%query representation
holds(Ends,CS,CP,Tok_Sentences,POS,Quantities,  Wc, X,  We):- 
    length(Tok_Sentences, L), Sq_i is (L-1),  match(Tok_Sentences, Sq_i, Tsq), match(POS, Sq_i, P),  match(Ends, Sq_i, En), 
    getRepr(En,CS,CP,Sq_i, Tsq, P, Quantities, holds, Wc, X,  We).
    
increase(Ends,CS,CP,Tok_Sentences,POS,Quantities, Wc, X,  We):- 
    length(Tok_Sentences, L), Sq_i is (L-1), 
    match(Tok_Sentences, Sq_i, Tsq), match(POS, Sq_i, P), match(Ends, Sq_i, En),
    getRepr(En,CS,CP,Sq_i,Tsq, P, Quantities, increase, Wc, X,  We).

decrease(Ends,CS,CP,Tok_Sentences,POS, Quantities, Wc, X,  We):- 
    length(Tok_Sentences, L), Sq_i is (L-1), 
    match(Tok_Sentences, Sq_i, Tsq), match(POS, Sq_i, P),  match(Ends, Sq_i, En), 
    getRepr(En,CS,CP,Sq_i, Tsq, P, Quantities, decrease, Wc, X,  We).

%Process each sentence - starting from first to the last but one sentence
process(Tok_Sentences, POS, Ends, [[Ind, WI, N1]], CS, CP,Wc, We, [N1], [V]):-
    match(Tok_Sentences, Ind, TS), match(POS, Ind, P),  match(Ends, Ind, E),
    getRepr(E,CS,CP,Ind,TS,P, Quantities, V, Wc, N1, We).

process(Tok_Sentences, POS, Ends, [[Ind, WI, N1]|QTail], CS, CP,Wc, We, [N1|OutN],[V|OutV]):- 
    match(Tok_Sentences, Ind, TS), match(POS, Ind, P),  match(Ends, Ind, E),
    getRepr(E,CS,CP,Ind,TS,P, Quantities, V, Wc, N1, We), 
    process(Tok_Sentences, POS, Ends, QTail, CS, CP,Wc, We, OutN,OutV). 
    

%Depending on whether problem queries for value of a variable at start, final or intermediate state.

wap(Tok_Sentences, POS,  Quantities,CS, CP, R):-  
    
    getEnd(Tok_Sentences, Ends, 0, 0),  
    increase(Ends,CS,CP,Tok_Sentences,POS,Quantities,Wc, "X",  We),
    process(Tok_Sentences, POS, Ends, Quantities, CS, CP,Wc, We, [X1,X2],[holds,holds]),
    R1 is (X2-X1),	
    almost_equal(R1,R), 
    writeln("Found_Sol:"),writeln("Container:"), writeln(Wc), writeln("Entity:"), writeln(We), writeln("n2-n1"), wwriteln(R) .

wap(Tok_Sentences, POS,  Quantities,CS, CP, R):-  
    getEnd(Tok_Sentences, Ends, 0, 0),  
    decrease(Ends,CS,CP,Tok_Sentences,POS,Quantities,Wc, "X",  We),
    process(Tok_Sentences, POS,  Ends, Quantities, CS, CP,Wc, We,[X1,X2],[holds,holds]),
    R1 is (X1-X2),	
    almost_equal(R1,R), 
    writeln("Found_Sol:"),writeln("Container:"), writeln(Wc), writeln("Entity:"), writeln(We),writeln("n1-n2"),writeln(R).

wap(Tok_Sentences, POS,  Quantities,  CS, CP, R):-  
    getEnd(Tok_Sentences, Ends, 0, 0),  
    holds(Ends,CS,CP,Tok_Sentences,POS,Quantities,Wc, "X",  We),
    process(Tok_Sentences, POS, Ends, Quantities, CS, CP,Wc, We,[X1,X2],[increase,holds]),
    R1 is (X2-X1),	
    almost_equal(R1,R), 
    writeln("Found_Sol:"),writeln("Container:"), writeln(Wc), writeln("Entity:"), writeln(We),writeln("n2-n1"),writeln(R).

wap(Tok_Sentences, POS,  Quantities, CS, CP, R):-  
    getEnd(Tok_Sentences, Ends, 0, 0),  
    holds(Ends,CS,CP,Tok_Sentences,POS,Quantities,Wc, "X",  We),
    process(Tok_Sentences, POS, Ends,  Quantities, CS, CP,Wc, We,[X1,X2],[decrease,holds]),
    R1 is (X2+X1),	
    almost_equal(R1,R), 
    writeln("Found_Sol:"),writeln("Container:"), writeln(Wc), writeln("Entity:"), writeln(We),writeln("n2+n1"),writeln(R).

wap(Tok_Sentences, POS,  Quantities,  CS, CP, R):-  
    getEnd(Tok_Sentences, Ends, 0, 0),  
    holds(Ends,CS,CP,Tok_Sentences,POS,Quantities,Wc, "X",  We),
    process(Tok_Sentences, POS, Ends, Quantities, CS, CP,Wc, We,[X1,X2],[holds,increase]),
    R1 is (X2+X1),	
    almost_equal(R1,R), 
    writeln("Found_Sol:"),writeln("Container:"), writeln(Wc), writeln("Entity:"), writeln(We), writeln("n2+n1"),writeln(R).

wap(Tok_Sentences, POS,  Quantities,  CS, CP, R):-  
    getEnd(Tok_Sentences, Ends, 0, 0),  
    holds(Ends,CS,CP,Tok_Sentences,POS,Quantities,Wc, "X",  We),
    process(Tok_Sentences, POS, Ends, Quantities,CS, CP,Wc, We,[X1,X2],[holds,decrease]),
    R1 is (X1-X2),	
    almost_equal(R1,R), 
    writeln("Found_Sol:"),writeln("Container:"), writeln(Wc), writeln("Entity:"), writeln(We),writeln("n1-n2"),writeln(R).

wap(Tok_Sentences, POS,  Quantities,  CS, CP, R):-  
    getEnd(Tok_Sentences, Ends, 0, 0),  
    holds(Ends,CS,CP,Tok_Sentences,POS,Quantities,Wc, "X",  We),
    process(Tok_Sentences, POS, Ends, Quantities,CS, CP,Wc, We,[X1,X2,X3],[decrease, increase, holds]),
    R1 is (X3-X2), R2 is (R1+X1),	
    almost_equal(R2,R), 
    writeln("Found_Sol:"),writeln("Container:"), writeln(Wc), writeln("Entity:"), writeln(We),writeln("n3-n2+n1"),writeln(R).

wap(Tok_Sentences, POS,  Quantities,  CS, CP, R):-  
    getEnd(Tok_Sentences, Ends, 0, 0),  
    holds(Ends,CS,CP,Tok_Sentences,POS,Quantities,Wc, "X",  We),
    process(Tok_Sentences, POS, Ends, Quantities,CS, CP,Wc, We,[X1,X2,X3],[increase, decrease, holds]),
    R1 is (X3+X2), R2 is (R1-X1),	
    almost_equal(R2,R), 
    writeln("Found_Sol:"),writeln("Container:"), writeln(Wc), writeln("Entity:"), writeln(We),writeln("n3+n2-n1"),writeln(R).

wap(Tok_Sentences, POS,  Quantities,  CS, CP, R):-  
    getEnd(Tok_Sentences, Ends, 0, 0),  
    holds(Ends,CS,CP,Tok_Sentences,POS,Quantities,Wc, "X",  We),
    process(Tok_Sentences, POS, Ends, Quantities,CS, CP,Wc, We,[X1,X2,X3],[increase, increase, holds]),
    R1 is (X3-X2), R2 is (R1-X1),	
    almost_equal(R2,R), 
    writeln("Found_Sol:"),writeln("Container:"), writeln(Wc), writeln("Entity:"), writeln(We),writeln("n3-n2-n1"),writeln(R).

wap(Tok_Sentences, POS,  Quantities,  CS, CP, R):-  
    getEnd(Tok_Sentences, Ends, 0, 0),  
    holds(Ends,CS,CP,Tok_Sentences,POS,Quantities,Wc, "X",  We),
    process(Tok_Sentences, POS, Ends, Quantities,CS, CP,Wc, We,[X1,X2,X3],[decrease, decrease, holds]),
    R1 is (X3+X2), R2 is (R1+X1),	
    almost_equal(R2,R), 
    writeln("Found_Sol:"),writeln("Container:"), writeln(Wc), writeln("Entity:"), writeln(We),writeln("n3+n1+n2"),writeln(R).


wap(Tok_Sentences, POS,  Quantities,  CS, CP, R):-  
    getEnd(Tok_Sentences, Ends, 0, 0),  
    holds(Ends,CS,CP,Tok_Sentences,POS,Quantities,Wc, "X",  We),
    process(Tok_Sentences, POS, Ends, Quantities,CS, CP,Wc, We,[X1,X2,X3],[holds,decrease, increase]),
    R1 is (X1-X2), R2 is (R1+X3),	
    almost_equal(R2,R), 
    writeln("Found_Sol:"),writeln("Container:"), writeln(Wc), writeln("Entity:"), writeln(We),writeln("n1-n2+n3"),writeln(R).

wap(Tok_Sentences, POS,  Quantities,  CS, CP, R):-  
    getEnd(Tok_Sentences, Ends, 0, 0),  
    holds(Ends,CS,CP,Tok_Sentences,POS,Quantities,Wc, "X",  We),
    process(Tok_Sentences, POS, Ends, Quantities,CS, CP,Wc, We,[X1,X2,X3],[holds,increase, decrease]),
    R1 is (X1+X2), R2 is (R1-X3),	
    almost_equal(R2,R), 
    writeln("Found_Sol:"),writeln("Container:"), writeln(Wc), writeln("Entity:"), writeln(We),writeln("n1+n2-n3"),writeln(R).

wap(Tok_Sentences, POS,  Quantities,  CS, CP, R):-  
    getEnd(Tok_Sentences, Ends, 0, 0),  
    holds(Ends,CS,CP,Tok_Sentences,POS,Quantities,Wc, "X",  We),
    process(Tok_Sentences, POS, Ends, Quantities,CS, CP,Wc, We,[X1,X2,X3],[holds,increase, increase]),
    R1 is (X1+X2), R2 is (R1+X3),	
    almost_equal(R2,R), 
    writeln("Found_Sol:"),writeln("Container:"), writeln(Wc), writeln("Entity:"), writeln(We),writeln("n1+n2+n3"),writeln(R).

wap(Tok_Sentences, POS,  Quantities,  CS, CP, R):-  
    getEnd(Tok_Sentences, Ends, 0, 0),  
    holds(Ends,CS,CP,Tok_Sentences,POS,Quantities,Wc, "X",  We),
    process(Tok_Sentences, POS, Ends, Quantities,CS, CP,Wc, We,[X1,X2,X3],[holds,decrease, decrease]),
    R1 is (X1-X2), R2 is (R1-X3),	
    almost_equal(R2,R), 
    writeln("Found_Sol:"),writeln("Container:"), writeln(Wc), writeln("Entity:"), writeln(We),writeln("n1-n2-n3"),writeln(R).
