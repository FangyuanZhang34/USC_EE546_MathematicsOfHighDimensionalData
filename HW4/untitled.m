% (1) preperation
clear
% load('aca2.mat');
load('aca5.mat');

% 
k = 2;

gammaVals = [0.1:0.1:0.9 1:1:100];
rec = zeros(size(gammaVals, 2), 1);


Xdat = X(:,1:2:end);
sdat = s(1:2:end);
k_ = max(sdat);

N = size(Xdat, 2);

for i = 1: N
    Xdat(:, i) = Xdat(:, i)/ norm(Xdat(:, i)); 
end

S = zeros(N, N);
for i = 1: N - 1
    for j = i+1: N
        S(i , j) = (norm(Xdat(:, i) - Xdat(:, j)))^2;
    end
end
S = S + S';

W = zeros(N, N);
[temp, srtind] = sort(S,1);
    
idx = 1;
for gamma = gammaVals
    for i = 1: N
        W(srtind(2: k+1, i), i) = exp(-gamma * S(srtind(2: k+1), i));
    end

    W = (W + W')/2; 

    D = zeros(N, N);
    for i = 1: N
        D(i, i) = sum(W(i, :));
    end

    L = D^(-1/2) * W * D^(-1/2);
    % X_ is the stacking of k_ largest e-vectors in columns
    [X_, D_] = eigs(L, k_);
    Y = zeros(N, k_);

    for i = 1: N
        for j = 1: k_
            Y(i, j) = X_(i, j) / norm(X_(i, :));
        end
    end

    pred = kmeans(Y, k_);

    % evaluate
    missrate = Misclassification(pred, sdat);
    fprintf('k = %d, gamma = %d, missrate = %f\n', k, gamma, missrate);
    rec(idx, 1) = missrate;
    idx = idx + 1;
end

%
min(rec)