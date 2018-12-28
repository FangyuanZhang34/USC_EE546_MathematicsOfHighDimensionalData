import pandas as pd
import numpy as np
import itertools
import kmeans 


def buildKernel(S,gamma,k):
    D,N = S.shape
    S = S/np.linalg.norm(S,axis=0).reshape((1,N))
    K = np.zeros((N,N))
    for i in range(N):
        for j in range(i+1,N):
            K[i,j] = np.exp(-gamma*(np.linalg.norm(S[:,i]-S[:,j])**2))
    K += K.T
    max_k = np.partition(K,k,axis = 0)
    print(max_k)
    K[K<max_k] = 0
    K = (K+K.T)/2
    return K

def spectralClustering(S,gamma,k):
    D,N = S.shape
    A = buildKernel(S,gamma,k)
    d = np.diag(A.sum(axis=1)**(-1/2))
    L = d.dot(A).dot(d)
    W, V = np.linalg.eig(L)
    X = V[:,:k]
    Y = X/np.linalg.norm(X,axis=1).reshape((N,1))
    kMeans = kmeans.KMeans(n_cluster=k)
    cen, r, i = kMeans.fit(Y)
    return r

def missClassGroups(Segmentation,RefSegmentation,ngroups):
    
    MyArray = np.arange(ngroups+1)
    permut = itertools.permutations(MyArray)
    permut_array = np.empty((0,ngroups+1))
    for p in permut:
        permut_array = np.append(permut_array,np.atleast_2d(p),axis=0)
    if Segmentation.shape[1]==1:
        Segmentation=Segmentation.T
    miss = np.zeros((permut_array.shape[0],Segmentation.shape[0]));
    for k in Segmentation.shape[0]:
        for j in permut_array.shape[0]:
            miss[j,k] = sum(Segmentation[k,:] != permut_array[j,RefSegmentation.tolist()])
    
    temp = np.argmin(miss,axis = 0);
    miss = miss[temp]
    index = permut_array[temp,:]
    
    return miss, index
    
#function [miss,index] = missclassGroups(Segmentation,RefSegmentation,ngroups)
#
#Permutations = perms(1:ngroups);
#if(size(Segmentation,2)==1)
#    Segmentation=Segmentation';
#end
#miss = zeros(size(Permutations,1),size(Segmentation,1));
#for k=1:size(Segmentation,1)
#    for j=1:size(Permutations,1)
#        miss(j,k) = sum(Segmentation(k,:)~=Permutations(j,RefSegmentation));
#    end
#end
#
#[miss,temp] = min(miss,[],1);
#index = Permutations(temp,:);

def missClassification(groups,s):
    print(groups.shape)
    N,G= groups.shape
    Missrate = np.zeros((G,1))
    n = np.max(s)
    print(n)
    for i in range(G):
        Missrate[i,1] = missClassGroups(groups[:,i].reshape(s.shape),s,n) / N
    return Missrate

def main():
    X2 = pd.read_csv("aca2X.txt",header=None)
    X5 = pd.read_csv("aca5X.txt",header=None)
    s2 = pd.read_csv("aca2s.txt",header=None)
    s5 = pd.read_csv("aca5s.txt",header=None)
    
    X2 = X2[X2.columns[::2]]
    X5 = X5[X5.columns[::2]]
    s2 = s2[s2.columns[::2]]
    s5 = s5[s5.columns[::2]]
    
    X2 = np.array(X2)
    X5 = np.array(X5)
    s2 = np.array(s2).T
    s5 = np.array(s5).T    
    
    s2_pred = spectralClustering(X2,100,10).reshape(s2.shape)
    s5_pred = spectralClustering(X5,100,10).reshape(s5.shape)
    miss_s2 = missClassification(np.hstack((s2_pred,s2_pred)),s2)
    
    pd.DataFrame(s2_pred).to_csv('s2_pred.csv')
    pd.DataFrame(s5_pred).to_csv('s5_pred.csv')
    
    
    return s2, s5, s2_pred, s5_pred, miss_s2
    
    
if __name__ == "__main__":
    s2, s5, s2_pred, s5_pred = main()
    
    
    
    
    

            
    
    
    