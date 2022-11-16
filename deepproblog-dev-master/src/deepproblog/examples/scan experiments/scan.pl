nn(nn_action, [X],Y,[action,direction,property,repeat]) :: net_category(X,Y).
nn(nn_hasProp, [S, X1, X2], Y, [0,1]) :: net_prop([S, X1, X2],Y).
nn(nn_hasDir, [S, X1, X2], Y, [0,1]) :: net_dir([S, X1, X2],Y).
nn(nn_hasRepeat, [S, X1, X2], Y, [0,1]) :: net_repeat([S, X1, X2],Y).
nn(nn_hasFollows, [S, X1, X2], Y, [0,1]) :: net_follows([S, X1, X2],Y).



operator(plus,X,Y,Z) :- Z is X+Y.
operator(minus,X,Y,Z) :- Z is X-Y.
operator(times,X,Y,Z) :- Z is X*Y.
operator(div,X,Y,Z) :- Y > 0, 0 =:= X mod Y,Z is X//Y.

scan([H|T],C) :-
    rnn(Text,Embed),
    net1(Embed,Perm),
    net2(Embed,Op1),
    net3(Embed,Swap),
    net4(Embed,Op2),
    permute(Perm,X1,X2,X3,N1,N2,N3),
    operator(Op1,N1,N2,Res1),
    swap(Swap,Res1,N3,X,Y),
    operator(Op2,X,Y,Out).
