import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats
import sklearn
from sklearn.datasets import load_boston

x=pd.read_csv("ex4x.dat",sep="   ",header=None)
y=pd.read_csv("ex4y.dat",header=None)

x_df=pd.DataFrame(x)
y_df=pd.DataFrame(y)
x_df['x0']=1

x_df=x_df.as_matrix()
y_df=y_df.as_matrix()

x=x_df
y=y_df

#logit
def logit (z):
    
   return 1/(1+np.exp(-z));

# hypothesis 

def hypo (x,th):
    
   return logit(np.dot(x,th));

#cost J
def J (x,y,th,m):
    
   return ((1/m) * sum(-np.transpose(y) * np.log(hypo(x,th)) - np.transpose(1-y) * np.log(1- hypo(x,th))) );


#gradient
def grad (x,y,th,m):
    
   return ( 1/m * np.dot(np.transpose(x),(hypo(x,th) - y)) );

#hessian
def hess (x,yx,thx,m):
    
   return (1/m * np.dot(np.transpose(x),x) * np.diag(hypo(x,th)) * np.diag(1-hypo(x,th)) );


Jj=np.ones((10, 1))
m=len(x)
th=np.ones((3, 1)) * 3.5
th=np.asmatrix(th)

for i in range(0,9):
	Jj[i]=J(x,y,th,m)
	th=th - np.dot(np.linalg.inv(hess(x,y,th,m)),grad(x,y,th,m))




 x1 = [min(x[1]), max(x[1])]
 x2 = (-1/th[2,]) * ((th[1,] * x1) + th[0,])
 plot(x1,x2, type='l',  xlab="test1", ylab="test2")
 points(x[,2],x[,3],col=as.factor(y))




