#!/usr/bin/env python
# coding: utf-8

# # Homework 2
# ## by:
# ### Bern Lior 206263071
# ### Lissitsa Denis 314880477

# # Question 1

# In[1]:


import scipy.io as sio
import scipy.stats as st
import matplotlib.pyplot as plt
import random as rnd
import numpy as np


# In[6]:


def MSE(x,y,rng):
    '''
    Function for calculating the beta vector with the minimal Mean Standard Error

    Parameters
    ----------
    x : Array-like
        Vector for the X axes.
    y : Array-like
        Vector for the Y axes.
    rng : Range object
        The relevant range object.

    Returns
    -------
    minMSEbeta : Touple
        Touple of the beta vector with the minimal MSE.

    '''
    rng=[i/10 for i in list(rng)] #turning the range from whole numbers to a decimals
    bestBeta=np.nan
    minMSE=10000000
    for a in rng: #loop for iterating over all possible beta pairs
        for b in rng:
            SSE=0
            for i in range(len(x)):
                prediction=a*x[i]+b #calculating model prediction
                SSE+=(y[i]-prediction)**2 #calculating SSE
            MSE=(1/len(x))*SSE #calculating MSE
            if MSE<minMSE:
                minMSE=MSE
                bestBeta=(a,b)
    return bestBeta


# ## Section a

# In[3]:


mat_data1a=sio.loadmat('HW2_1a_data.mat') #loading data

X=mat_data1a['x'][0].tolist() #X vector
Y=mat_data1a['y'][0].tolist() #Y vector
rnga=range(-150,151,1) #creating range x10


# In[7]:


a,b=MSE(X,Y,rnga) #calculating best fit beta vector
newY=a*mat_data1a['x'][0]+b #calculating predicted Y vector
print('The slope and intercept value with the lowest MSE: a='+ str(a)+ ', b='+str(b))
print('The best fit linear regression is: y='+str(a)+'*x+'+str(b))


# In[18]:


#plotting
plt.plot(X,Y,'ro',X,newY,'k')
plt.ylabel('Y (AU)')
plt.xlabel('X (AU)')
plt.title('Scatter plot of the measured data (vectors X, Y) and the best fit linear regression')


# In[19]:


#Calculating STD of noise distribution
noise_dist=[] 
for i in range(len(X)):
    noise_dist.append((Y[i]-(a*X[i]+b))) #creating noise distribution from unexplained variance
noiseSTD=np.std(noise_dist,ddof=1)
print("Estimated Noise STD: "+ str(noiseSTD))


# ## Section b

# In[24]:


#bootstrapping
mat_data1b=sio.loadmat('HW2_1b_data.mat')
samplelst=[]
Xvec=mat_data1b['x'][0,:]
Ymat=mat_data1b['y']
sampnum=1000
betalst=[np.nan]*sampnum #creating empty list for faster runspeed
for i in range(sampnum):
    ysamp=[]
    for j in range(41):
        ysamp.append(rnd.choice(list(Ymat[:,j]))) #creating the sample's y-vector by choosing one y at random for each x
    samplelst.append(ysamp)
    betalst[i]=MSE(Xvec,samplelst[i],rnga)


# In[25]:


slopelst,interceptlst=zip(*betalst) #unpack betas into a list of slopes and a list of intercepts
print(np.mean(interceptlst),np.mean(slopelst))


# In[27]:


#plotting
lCI,hCI=st.norm.interval(confidence=0.975, loc=np.mean(interceptlst), scale=np.std(interceptlst))#calculate confidence intervals
print("The boundries of the CI for the intercepts are: ",lCI,hCI)
fig, axs = plt.subplots(1, 2)
axs[0].hist(interceptlst, bins='auto', facecolor='g')
axs[0].axvline(lCI, color='k', linestyle='dashed', linewidth=1)
axs[0].axvline(hCI, color='k', linestyle='dashed', linewidth=1)
axs[0].axvline(np.mean(interceptlst), color='r', linewidth=1)
axs[0].set_xlabel('Intercept Value (AU)')
axs[0].set_ylabel('Occurances')
axs[0].set_title('Histogram of Intercept values')
lCI,hCI=st.norm.interval(confidence=0.975, loc=np.mean(slopelst), scale=np.std(slopelst))#calculate confidence intervals
print("The boundries of the CI for the slopes are: ",lCI,hCI)
axs[1].hist(slopelst, bins='auto', facecolor='r')
axs[1].axvline(lCI, color='k', linestyle='dashed', linewidth=1)
axs[1].axvline(hCI, color='k', linestyle='dashed', linewidth=1)
axs[1].axvline(np.mean(slopelst), color='g', linewidth=1)
axs[1].set_xlabel('Slope Value (AU)')
axs[1].set_ylabel('Occurances')
axs[1].set_title('Histogram of Slope values')


# # Question 2

# # #Section a

# In[6]:


def LogLikelihood(y,theta,X):
    '''
    Function for calculating the log-likelihood of a set of parameters

    Parameters
    ----------
    y : Numpy Array
        Vector of Y values.
    theta : Array-like
        Vector of parameters.
    X : Numpy Array
        Matrix of X values.

    Returns
    -------
    LL : Float
        The calculated log-likelihood value.

    '''
    LL=0
    for i in range(y.shape[1]):
        thetaX=theta[0]
        for j in range(len(X[:,i])): #loop for calucalting the theta*x value (the predicted y)
            thetaX+=theta[j+1]*X[j,i]
        LL+=y[0,i]*thetaX-np.exp(thetaX) #LL function
    return LL


# In[22]:


mat_data2=sio.loadmat('HW2_2_data.mat')
thetas=[(17,6,-19,48),(1,4,-7,1),(3,0.02,0.1,-1)]
X=mat_data2['X']
y=mat_data2['y'].transpose()


# In[8]:


LLlistA=[]
for theta in thetas:
    LLlistA.append(LogLikelihood(y, theta, X))
print('The calculated Log-Likelihood values are: '+ str(LLlistA))
print('The theta vector with the max LL is: '+ str(thetas[LLlistA.index(max(LLlistA))]))


# ## Section b

# In[9]:


LLlistB=[]
for theta in thetas:
    regfactor=0
    for i in theta:
        regfactor+=i*i
    LLlistB.append(LogLikelihood(y, theta, X)-0.5*regfactor)
print('The calculated new Log-Likelihood values are: '+ str(LLlistB))
print('The theta vector with the max LL is: '+ str(thetas[LLlistB.index(max(LLlistB))]))


# ## Section c

# In[23]:


import statsmodels.api as sm
endog=y.transpose() #creating endogenous matrix
exog = sm.add_constant(X.transpose()) #creating exogenous matrix and adding a row of 1's for intercept calculation
poisson_model=sm.GLM(endog,exog,family=sm.families.Poisson()) #creating model
poisson_results=poisson_model.fit() #fitting model
print(poisson_results.summary())
print('The Theta vector is:', poisson_results.params)

