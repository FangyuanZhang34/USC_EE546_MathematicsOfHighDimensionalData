function K = buildKernel(S,gamma,k)
[D,N] = size(S);
for i = 1:N
    S(:,i) = S(:,i)/norm(S(:,i));
end
K = zeros(N,N);
for i = 1:N
    for j = i+1:N
        K(i,j) = exp(-gamma*(norm(S(:,i)-S(:,j))^2));
    end
end
K = K + K';
for j = 1:N
    [val,ind] = sort(K(:,j),'descend');
    K(ind(k+1:N),j) = 0;
end
K = (K+K')/2  + eye(N);