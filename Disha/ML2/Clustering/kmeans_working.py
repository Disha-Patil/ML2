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

'''
class K_Means:
    def _init_(self,k=3,tol=0.001,max_iter=300):
        self.k=k
        self.tol = tol
        self.max_iter = max_iter

    def fit(self,data):
        self.centroids = {}

        for i in range(self.k):
            self.centroids[i] = data[i]

        for i in range(self.max_iter):
            self.classifications= {}

            for i in range(self.k):
                self.classifications[i] = [] 

            for featureset in data:
                distances = [np.linalg.norm(featureset-self.centroids[centroid]) for centroid in self.centroids]
                classification = distances.index(min(distances))
                self.classifications[classification].append(featureset)
            
            prev_centroids =dict(self.centroids)
            
            for classification in self.classifications:
                    pass
                    #self.centroids[classification] = np.average(self.classfications[classification],axis=0)
            
            optimized = True        
            
            for c in self.centroids:
                original_centroid = prev_centroids[c]
                current_Centroid = self.centroids[c]
                if np.sum((current_centroid-original_centroid)/original_centroid*100.0)>self.tol:
                    optimized = False
            
            if optimized:
                break

    def predict(self,data) :
        distances = [np.linalg.norm(data-self.centroids[centroid]) for centroid in self.centroids]
        classification = distances.index(min(distances))
        return classification

    
clf = K_Means()
clf.fit(irisdf)
    
for centroid in clf.centroids:
    plt.scatter(clf.centroids[centroid][0],clf.centroids[centroid][1],marker="o",color="k",s=150,linewidths=5)
    
for classification in clf.classifications:
    color = colors[classification]
    for featureset in clf.classifications[classification]:
        plt.scatter(featureset[0],featureset[1],marker="x",color=color,s=150,linewidths=5)
            
plt.show()
'''


#new np style
# necessary imports
%matplotlib inline
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns; sns.set()

def initialize_centroids(points, k):
    """returns k centroids from the initial points"""
    centroids = points.copy()
    np.random.shuffle(centroids)
    return centroids[:k]

iri=np.array(irisdf)
c=initialize_centroids(iri, 3)

def closest_centroid(points, centroids):
    """returns an array containing the index to the nearest centroid for each point"""
    distances = np.sqrt(((points - centroids[:, np.newaxis])**2).sum(axis=2))
    return np.argmin(distances, axis=0)
'''
c = initialize_centroids(iri, 3)
closest_centroid(iri, c)

c_extended = c[: , np.newaxis, :]
c_extended

c_extended.shape

iri[:4] - c_extended

np.sqrt(((iri[:4] - c_extended)**2).sum(axis=2))

np.argmin(np.sqrt(((iri[:4] - c_extended)**2).sum(axis=2)), axis=0)

'''

def move_centroids(points, closest, centroids):
    """returns the new centroids assigned from the points closest to them"""
    return np.array([points[closest==k].mean(axis=0) for k in range(centroids.shape[0])])


move_centroids(iri, closest_centroid(iri, c), c)


for i in range(0,100):
    closest = closest_centroid(iri, c)
    centroids = move_centroids(iri, closest, c)
    







'''
plt.subplot(121)
plt.scatter(points[:, 0], points[:, 1])
centroids = initialize_centroids(points, 3)
plt.scatter(centroids[:, 0], centroids[:, 1], c='r', s=100)

plt.subplot(122)
plt.scatter(points[:, 0], points[:, 1])
closest = closest_centroid(points, centroids)
centroids = move_centroids(points, closest, centroids)
plt.scatter(centroids[:, 0], centroids[:, 1], c='r', s=100)
'''



