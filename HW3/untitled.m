m = 1000;
n = 100000;
X = orth(randn(m,m));
Y = orth(randn(n,m));
d1 = zeros(r,1);
for i = 1:r
    d1(i) = r - i + 1;
end
d2 = diag(4 * 10^(-3) * ones(m-r));
d = [d1;d2];
D = diag(d);
A = X * D * Y';
[U_A,S_A,V_A] = svd(A,'econ');

