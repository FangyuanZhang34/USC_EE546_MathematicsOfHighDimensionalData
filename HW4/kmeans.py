import numpy as np


class KMeans():

	'''
	    Class KMeans:
	    Attr:
	        n_cluster - Number of cluster for kmeans clustering (Int)
	        max_iter - maximum updates for kmeans clustering (Int) 
	        e - error tolerance (Float)
	'''

	def __init__(self, n_cluster, max_iter=100, e=0.0001):
		self.n_cluster = n_cluster
		self.max_iter = max_iter
		self.e = e

	def fit(self, x):
		'''
		    Finds n_cluster in the data x
		    params:
		        x - N X D numpy array
		    returns:
		        A tuple
		        (centroids a n_cluster X D numpy array, y a size (N,) numpy array where cell i is the ith sample's assigned cluster, number_of_updates an Int)
		    Note: Number of iterations is the number of time you update the assignment
		''' 
		assert len(x.shape) == 2, "fit function takes 2-D numpy arrays as input"
		np.random.seed(42)
		N, D = x.shape

		# TODO:
		# - comment/remove the exception.
		# - Initialize means by picking self.n_cluster from N data points
		# - Update means and membership until convergence or until you have made self.max_iter updates.
		# - return (means, membership, number_of_updates)

		#Step 3: Initialize
		cen_init = np.random.choice(N, self.n_cluster) # n_cluster X D
		cen = x[cen_init,:]
		J = 1e10
		# J_new = J
		dist = np.full((N,self.n_cluster), np.inf) # distance between x and centroids
		r = np.full((N,), self.n_cluster) # cluster x belongs to
		r_ik = np.full((N, self.n_cluster), 0) # cluster x belongs to, one hot encode

		#Step 4: Repeat
		for i in range(self.max_iter):
			#Step 5: compute membership
			for j in range(self.n_cluster):
				dist[:,j] = np.linalg.norm(x - cen[j,:],axis = 1) ** 2
			r = np.argmin(dist, axis = 1)
			r_ik[np.arange(N), r] = 1

			#Step 6: compute distortion objective
			J_new = np.sum(dist.min(axis = 1))/N

			#Step 7,8: justify error tolerance - e
			if (abs(J - J_new) <= self.e):
				break

			#Step 9: set J = J_new
			J = J_new

			#Step 10: compute centroids
			nonz = np.where(r_ik.max(axis=0) == 1)
			# r_ik_nonz = r_ik[:,nonz] # nonzero cluster
			for k in nonz[0]:
				cen[k,:] = (r_ik[:,k].reshape((N,1)) * x).sum(axis = 0)	/ np.sum(r_ik[:,k])		

		return cen, r, i



		# DONOT CHANGE CODE ABOVE THIS LINE
		# raise Exception(
		# 	'Implement fit function in KMeans class (filename: kmeans.py)')
		# DONOT CHANGE CODE BELOW THIS LINE

class KMeansClassifier():

	'''
	Class KMeansClassifier:
	Attr:
	    n_cluster - Number of cluster for kmeans clustering (Int)
	    max_iter - maximum updates for kmeans clustering (Int) 
	    e - error tolerance (Float) 
	'''

	def __init__(self, n_cluster, max_iter=100, e=1e-6):
		self.n_cluster = n_cluster
		self.max_iter = max_iter
		self.e = e

	def fit(self, x, y):
		'''
			Train the classifier
			params:
			    x - N X D size  numpy array
			    y - (N,) size numpy array of labels
			returns:
			    None
			Stores following attributes:
			    self.centroids : centroids obtained by kmeans clustering (n_cluster X D numpy array)
			    self.centroid_labels : labels of each centroid obtained by 
			        majority voting ((N,) numpy array) 
		'''

		assert len(x.shape) == 2, "x should be a 2-D numpy array"
		assert len(y.shape) == 1, "y should be a 1-D numpy array"
		assert y.shape[0] == x.shape[0], "y and x should have same rows"

		np.random.seed(42)
		N, D = x.shape
		# TODO:
		# - comment/remove the exception.
		# - Implement the classifier
		# - assign means to centroids
		# - assign labels to centroid_labels

		kmean = KMeans(self.n_cluster, self.max_iter, self.e)
		cen, r, i = kmean.fit(x)
		u, c = np.unique(y, return_counts=True)
		


		#Step 11: label centroids with majority class of the cluster
		r_ik = np.eye(self.n_cluster)[r] # 1-hot
		# nonz = np.where(r_ik.max(axis=0) == 1)
		# r_ik_nonz = r_ik[:,nonz[0]]
		r_label = r_ik * y.reshape((N,1))
		n_clu = r_label.shape[1]

		centroids = cen
		centroid_labels = np.zeros(n_clu)

		for k in range(n_clu):
			# print(k)
			# nonzero = np.nonzero(r_label[:,k])
			# print(np.bincount(r_label[:,k].astype(int))[0])
			if (r_label[:,k].max() == 0):
				centroid_labels[k] = 0
			else:
				unique, counts = np.unique(r_label[:,k], return_counts=True)
				label = counts[1:].argmax()
				centroid_labels[k] = unique[label + 1]



		

		# DONOT CHANGE CODE ABOVE THIS LINE
		# raise Exception(
		# 	'Implement fit function in KMeansClassifier class (filename: kmeans.py)')

	    # DONOT CHANGE CODE BELOW THIS LINE

		self.centroid_labels = centroid_labels
		self.centroids = centroids

		assert self.centroid_labels.shape == (self.n_cluster,), 'centroid_labels should be a numpy array of shape ({},)'.format(
			self.n_cluster)

		assert self.centroids.shape == (self.n_cluster, D), 'centroid should be a numpy array of shape {} X {}'.format(
			self.n_cluster, D)

	def predict(self, x):
		'''
		    Predict function

		    params:
		        x - N X D size  numpy array
		    returns:
		        predicted labels - numpy array of size (N,)
		'''

		assert len(x.shape) == 2, "x should be a 2-D numpy array"

		np.random.seed(42)
		N, D = x.shape
		# TODO:
		# - comment/remove the exception.
		# - Implement the prediction algorithm
		# - return labels

		dist = np.full((N,self.n_cluster), np.inf)
		r_ik = np.full((N, self.n_cluster), 0) # cluster x belongs to, one hot encode


		for k in range(self.n_cluster):
			dist[:,k] = np.linalg.norm(x - self.centroids[k,:],axis = 1) ** 2
		r_pred = np.argmin(dist, axis = 1)
		r_ik[np.arange(N), r_pred] = 1
		labels = r_ik.dot(self.centroid_labels)


		# DONOT CHANGE CODE ABOVE THIS LINE
		# raise Exception(
		#     'Implement predict function in KMeansClassifier class (filename: kmeans.py)')
		# DONOT CHANGE CODE BELOW THIS LINE
		return labels

