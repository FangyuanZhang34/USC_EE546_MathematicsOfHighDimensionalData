function r = spectralClustering(S,gamma,k,numCluster)
[D,N] = size(S);
A = buildKernel(S,gamma,k);
d = diag(sum(A,2)');
L = eye(size(d,1))-(d^(-1/2))*A*(d^(-1/2));
[V,E] = eig(L);
X = V(:,1:numCluster);
Y = zeros(size(X));
for i = 1:N
    Y(i,:) = X(i,:)./norm(X(i,:),2);
end
r = kmeans(Y,numCluster);