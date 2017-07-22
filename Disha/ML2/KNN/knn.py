
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats
import sklearn
from sklearn import datasets
from sklearn.utils import shuffle
from sklearn.metrics.pairwise import euclidean_distances


iris = datasets.load_iris()
irisdf = pd.DataFrame(data=np.c_[iris['data'], iris['target']], columns=iris['feature_names'] + ['target'])
#irisdf = pd.DataFrame(data=iris['data'], columns=iris['feature_names'] )
irisdf.head()

k=3
epsilon=0.01

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
			else :
				break

		
		cl1mean=c1.iloc[:,0:4].mean()
		cl2mean=c2.iloc[:,0:4].mean()
		cl3mean=c3.iloc[:,0:4].mean()

		d1=euclidean_distances(c1.iloc[:,0:4] , cl1mean).sum()
		d2=euclidean_distances(c2.iloc[:,0:4] , cl2mean).sum()
		d3=euclidean_distances(c3.iloc[:,0:4] , cl3mean).sum()

     		if (D[loop] - D[loop-1] < epsilon):
        		break
		else:
			loop=loop+1

	Dmin[iter]=D[loop]




besti=np.where(Dmin == Dmin.min())







