clear
load('aca2.mat');
% load('aca5.mat');
X1 = X(:,1:2:size(X,2));
s1 = s(1:2:size(s,2));
numCluster = max(s1);
minMissrate = 1;
minGamma = 0;
minK = 0;
missRate = zeros(100,49);
for gamma=10:10
    for k=5:5
        r = spectralClustering(X1,gamma,k,numCluster);
        groups = r;
        Missrate = Misclassification(groups,s1);
        missRate(gamma,k) = Missrate;
        if (Missrate < minMissrate)
            minMissrate = Missrate;
            minGamma = gamma;
            minK = k;
        end 
    end
end
csvwrite('myFile2.txt',missRate);
% csvwrite('myFile2.txt',missRate);