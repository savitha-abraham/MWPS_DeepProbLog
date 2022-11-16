concat([], "").
concat([L|Ls], Str) :-concat(Ls, Str0), append("", Str0, Str1), append(L, Str1, Str).
query(concat(["X", "=", 11]), Str).
