
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats
import sklearn
from sklearn import datasets
from sklearn.utils import shuffle
from sklearn.metrics.pairwise import euclidean_distances


iris = datasets.load_iris()
irisdf = pd.DataFrame(data=np.c_[iris['data']], columns=iris['feature_names'])
#irisdf = pd.DataFrame(data=iris['data'], columns=iris['feature_names'] )
irisdf.head()

class K_means:
	def _init_(self,k=3,tol=0.001,max_iter=300):
		self.k=k
		self.tol = tol
		self.max_iter = max_iter
	
	def fit(self,data):
	
		self.centroids = {}
		
		for i in range(self.k)
			self.centroids[i] = data[i]
			
		for i in range(self.max_iter):
			self.classifications= ()
			
			for i in range(self.k):
				self.classifications[i] = []
			
			for featureset in irisdf:
				distances = [np.linalg.norm(featureset-self.centroids[centroid]) for centroid in self.centroids]
                classification = distances.index(min(distances))
                self.classifications[classification].append(featureset)
            
            prev_centroids =dict(self.centroids)
            
            for classification in self.classifications:
                    pass
                    self.centroids[classification] = np.average(self.classfications[classification],axis=0)
            


	def predict(self,data) :
		pass

	
	












'''
irisdf=shuffle(irisdf)

cl1=irisdf[irisdf.target==0]
cl2=irisdf[irisdf.target==1]
cl3=irisdf[irisdf.target==2]

cl1mean=cl1.iloc[:,0:4].mean()
cl2mean=cl2.iloc[:,0:4].mean()
cl3mean=cl3.iloc[:,0:4].mean()

d1=euclidean_distances(cl1.iloc[:,0:4] , cl1mean).sum()
d2=euclidean_distances(cl2.iloc[:,0:4] , cl2mean).sum()
d3=euclidean_distances(cl3.iloc[:,0:4] , cl3mean).sum()

D=[]
D.append(sum([d1,d2,d3]))

loop=1
Dmin=[]

for i in range(0,99):
	while True:
		for j in range(0, len(irisdf)):
			c1=[]
			c2=[]
			c3=[]				
			ed1=euclidean_distances(irisdf.iloc[j,0:4] , cl1mean).sum()
			ed2=euclidean_distances(irisdf.iloc[j,0:4] , cl2mean).sum()
			ed3=euclidean_distances(irisdf.iloc[j,0:4] , cl3mean).sum()
			if ed1==min(ed1,ed2,ed3) :
                        	c1.append[j]=irisdf.iloc[j,0:4]                    
                        elif ed2==min(ed1,ed2,ed3) :
                        	c2.append[j]=irisdf.iloc[j,0:4]
                  	elif ed3==min(ed1,ed2,ed3) :
                        	c3.append[j]=irisdf.iloc[j,0:4]
			
				

		
		cl1mean=c1.iloc[:,0:4].mean()
		cl2mean=c2.iloc[:,0:4].mean()
		cl3mean=c3.iloc[:,0:4].mean()

		d1=euclidean_distances(c1.iloc[:,0:4] , cl1mean).sum()
		d2=euclidean_distances(c2.iloc[:,0:4] , cl2mean).sum()
		d3=euclidean_distances(c3.iloc[:,0:4] , cl3mean).sum()
		D.append(sum([d1,d2,d3]))

     		if (D[loop] - D[loop-1] < epsilon):
        		break
		else:
			loop=loop+1

	Dmin[iter]=D[loop]



besti=np.where(Dmin == Dmin.min())


'''

'''
def dist2(f1, f2):
    a = np.array
    d = a(f1)-a(f2)
return np.sqrt(np.dot(d, d))

def mean(feats):
return np.mean(feats, axis=0)


def assign(centers):
    new_centers = defaultdict(list)
    for cx in centers:
        for x in centers[cx]:
            best = min(centers, key=lambda c: dist2(x,c))
            new_centers[best] += [x]
return new_centers

def update(centers):
    new_centers = {}
    for c in centers:
        new_centers[mean(centers[c])] = centers[c]
    return new_centers

def kmeans(features, k, maxiter=100):
    centers = dict((c,[c]) for c in features[:k])
    centers[features[k-1]] += features[k:]
    for i in xrange(maxiter):
        new_centers = assign(centers)
        new_centers = update(new_centers)
        if centers == new_centers:
            break
        else:
            centers = new_centers
    return centers
'''





















