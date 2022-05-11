function l = subdistance(A,B)

[U,S,V]=svd(A'*B);
O = U*V';
l = norm(A*O-B, 'fro');