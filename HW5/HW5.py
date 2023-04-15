import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import pandas as pd
import statsmodels.api as sm
colors=['#006BA6','#7EB77F','#FFBC42','#D81159','#8F2D56']
xticks=np.arange(-16,17,2)
#sample prep
distA=stats.norm(loc=35, scale=7)
distB=stats.poisson(mu=20)
distC=stats.uniform(loc=20, scale=20)
distributions={'A':distA,'B':distB,'C':distC}
samples={}
N=35
stimuli=np.arange(-16,17)
np.random.seed(69)
for key,value in distributions.items():
    samples[key]=np.empty((N,0))
    for i in stimuli:
        trials=value.rvs(size=N).reshape(N,1)
        samples[key]=np.concatenate((samples[key], trials), axis=1)
#a
fig,axs=plt.subplots(1,3)
for i,(key,value) in enumerate(samples.items()):
    means=np.mean(value, axis=0)
    stds=np.std(value, axis=0)
    axs[i].plot(stimuli,means,c=colors[i],label='Mean',linewidth=3)
    axs[i].fill_between(stimuli, means - stds, means + stds, alpha=0.420, label='STD',color=colors[i])
    axs[i].set_xticks(xticks)
    axs[i].set_title('Neuron '+key)
    axs[i].set_xlabel('Stimuli [A.U]')
    axs[i].set_ylabel('Firing Rate [Hz]')
    axs[i].legend()
plt.suptitle('Tuning Curve of 3 Neurons',fontweight="bold",size=16)
plt.show()
#b
meanFR={'Neuron '+key:np.mean(value) for key,value in samples.items()}
#c
def poisson(x,lamda):
    return (np.exp(-lamda)*np.power(lamda,x))/(np.math.factorial(x))
    
trialFR=[32,24,27]
likelihoods={}
for i,(key,value) in enumerate(samples.items()):
    likelihood=[]
    x=trialFR[i]
    for j in range(value.shape[1]):
        lamda=np.mean(value[:,j])
        likelihood.append(poisson(x, lamda))
    likelihoods[key]=np.array(likelihood)
    
plt.figure()    
for i,(key,value) in enumerate(likelihoods.items()):
    plt.plot(stimuli,value,label='Neuron '+key,c=colors[i])
plt.legend()
plt.title('Likelihood of Stimuli Value Given a FR')
plt.xticks(xticks)
plt.xlabel('Stimuli [A.U]')
plt.ylabel('Likelihood [A.U]')
plt.show()  

#d
MLEs={key:(max(value),stimuli[np.where(value == max(value))]) for key,value in likelihoods.items()}
print('The MLEs for the 3 neurons are as follows:')
for key,value in MLEs.items():
    print('Neuron '+key+':', 'The MLE is: ', round(value[0],4), 'and the most likely stimulus value is:' ,value[1])

#e
PopLogLikelihood=[]
for i in range(33):
    product=1    
    for j,value in enumerate(samples.values()):
        x=trialFR[j]
        lamda=np.mean(value[:,i])
        product*=poisson(x, lamda)
    PopLogLikelihood.append(np.log(product))
print("According to the population's maximum log likelihood, the most likely stimulus value is:",stimuli[np.argmax(PopLogLikelihood)] )
##2

#simulate
#a
odd=stats.norm(loc=15, scale=5)
even=stats.uniform(loc=5, scale=20)
recordings=np.empty((0,64))
for i in range(100):
    odds=odd.rvs(size=32)
    evens=even.rvs(size=32)
    electrodes=np.array([x for t in zip(odds, evens) for x in t]).reshape(1,64)
    recordings=np.concatenate((recordings,electrodes))
#b
pca=PCA(n_components=2)
pca.fit(recordings)
projected=pca.transform(recordings)
explained_variance = pca.explained_variance_ratio_*100

fig,axs=plt.subplots(1,2)
labels=['PC1','PC2']
axs[0].bar(labels,explained_variance,color=colors[3])
axs[0].set_title('Scree Plot',size=16)
axs[0].set_ylabel('Variance Explained [%]',size=14)
explained_variance=[x.round(2) for x in explained_variance]
axs[1].scatter(projected[:,0],projected[:,1],c=colors[0])
axs[1].set_title('Data Projected to Two PCs ('+str(sum(explained_variance))+'% total var explained)',size=16)
axs[1].set_xlabel('PC1('+str(explained_variance[0])+'% var explained)',size=14)
axs[1].set_ylabel('PC2('+str(explained_variance[1])+'% var explained)',size=14)
#c
SSElst=[] #list to contain SSE values
Ks=range(5,16)
for k in Ks:
    kmeans = KMeans(n_clusters=k) # Create an instance of the KMeans class
    kmeans.fit(projected) # Fit the model to the dataset   
    clusters = kmeans.predict(projected) # Predict the cluster for each data point
    SSElst.append(kmeans.inertia_) #Access the SSE of the clustering
    
plt.figure()
plt.plot(list(Ks),SSElst)
plt.title('SSE per Number of Clusters')
plt.xlabel('Clusters [K]')
plt.ylabel('SSE [A.U]')

optimal_k=np.argmax(np.abs(np.diff(np.diff(np.array(SSElst)))))+6
kmeans=KMeans(n_clusters=optimal_k)
kmeans.fit(projected)
klabels = kmeans.labels_
centroids=kmeans.cluster_centers_

plt.figure()
plt.scatter(projected[:,0],projected[:,1],c=klabels,cmap='viridis')
plt.scatter(centroids[:,0],centroids[:,1],label='Centroids',c='#FA0F36',edgecolor='#640212',s=77.7,marker='X')
plt.legend()
plt.title('Data Clustered by KMeans (k='+str(optimal_k)+' clusters) in PC Space')
plt.xlabel('PC1('+str(explained_variance[0])+'% var explained)')
plt.ylabel('PC2('+str(explained_variance[1])+'% var explained)')

#d
PC1weights=pca.components_[0]
electrodes=np.argsort(PC1weights)[-5:]
print('The 5 electrodes with the heighest PC1 loading scores (from lowest to heighest) are:', electrodes+1)

electroDF=pd.DataFrame({'E'+str(i+1):recordings[1:,i] for i in electrodes})
Arthur=pd.DataFrame({'E'+str(i+1):recordings[:1,i] for i in electrodes[:-1]})
names=list(electroDF.columns)
model_cnfg = f"{names[4]}~{names[0]}+{names[1]}+{names[2]}+{names[3]}"

Poisson_model = sm.GLM.from_formula(model_cnfg, data=electroDF, family=sm.families.Poisson())
Poisson_results=Poisson_model.fit()
Poisson_predict=Poisson_results.predict(Arthur)
print(Poisson_results.summary())
print('The Theta vector is:\n', Poisson_results.params)

Linear_model = sm.GLM.from_formula(model_cnfg, data=electroDF, family=sm.families.Gaussian())
Linear_results=Linear_model.fit()
Linear_predict=Linear_results.predict(Arthur)
print(Linear_results.summary())
print('The Theta vector is:\n', Linear_results.params)

def BIC(k,n,loglikelihood):
    return k*np.log(n)-2*loglikelihood
def AIC(k,loglikelihood):
    return 2*k-2*loglikelihood

Poisson_likelihood=Poisson_results.llf
P_AIC=AIC(4,Poisson_likelihood)
P_BIC=BIC(4,99,Poisson_likelihood)

Linear_likelihood=Linear_results.llf
L_AIC=AIC(4,Linear_likelihood)
L_BIC=BIC(4,99,Linear_likelihood)

criteria=np.array([[P_AIC,P_BIC],[L_AIC,L_BIC]])
width=0.3
r1 = np.arange(2)
r2 = [x + width for x in r1]
plt.figure()
plt.bar(r1,criteria[:,0],width=width,color=colors[1],label='Poisson Regression')
plt.bar(r2,criteria[:,1],width=width,color=colors[2],label='Linear Regression')
plt.xticks([r + width/2 for r in range(2)], ['AIC','BIC'])
plt.ylabel('AIC/BIC [A.U]')
plt.title('Model Comparison')
plt.legend()
    