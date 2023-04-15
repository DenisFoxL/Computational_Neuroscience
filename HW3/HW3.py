#!/usr/bin/env python
# coding: utf-8

# # Homework 3
# ### by Bern Lior 260263071 & Lissitsa Denis 314880477

# In[1]:


import scipy.io as sio
import numpy as np 
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn import mixture
from matplotlib.patches import Ellipse


# In[2]:


colors=['#900C3F','#FF5733','#C1DA95','#C70039','#FFC300','#581845','#669933','#99CC99','#99CCFF','#000033']


# ## Question 1

# In[3]:


LFP=sio.loadmat('HW3_1.mat')['lfp'] #loading HW data


# ### Section a

# In[4]:


means_vec=np.mean(LFP,axis=0) #calculating the mean of each dimension (column)

centered_mat=LFP-means_vec #creating a centered matrix by substracting the mean of each column from each value in that column

cov_mat=np.cov(LFP,rowvar=False) #calculating covariance matrix

eigenvalues, eigenvectors = np.linalg.eig(cov_mat) #calculating the eigenvectors and eigenvalues of the matrix

# sorting the eigenvectors by the eigenvalues (from highest to lowest) #!!!!!
sorted_i=eigenvalues.argsort()[::-1] 
eigenvalues,eigenvectors=eigenvalues[sorted_i],eigenvectors[:,sorted_i]

percentage_var = np.round((eigenvalues/sum(eigenvalues))*100,2) #turning raw values to precentage

dims=2 #number of wanted dimensions

feature_vector = eigenvectors[:, :dims] #selecting k(dimensions) eigenvectors with the highest eigenvalues

print('The Principal Components are:\nPC1: '+str(feature_vector[:,0])+'\nPC2: '+str(feature_vector[:,1]))


# ### Section b 

# In[5]:


projected_mat=np.dot(centered_mat,feature_vector) #projecting the data to the calculated PC's

#plottting scatterplot
fig, ax = plt.subplots()
ax.scatter(projected_mat[:, 0], projected_mat[:, 1],color='purple',alpha=0.77)
plt.xlabel('PC1 ('+str(percentage_var[0])+'% var explained) [A.U]')
plt.ylabel('PC2 ('+str(percentage_var[1])+'% var explained) [A.U]')
plt.title('PCA result for '+str(dims)+' dimensions (total variance explained='+str(percentage_var[0]+percentage_var[1])+'%)')
plt.show()


# ### Section c

# In[15]:


#plotting histograms
fig, ax = plt.subplots(1,2)
ax[0].hist(projected_mat[:, 0],bins='auto',color=colors[4])
ax[0].set_title('PC1('+str(percentage_var[0])+'% var explained)')
ax[0].set_ylabel('Count [n]')
ax[0].set_xlabel('Value [A.U]')
ax[1].hist(projected_mat[:, 1],bins='auto',color=colors[2])
ax[1].set_title('PC2('+str(percentage_var[1])+'% var explained)')
ax[1].set_ylabel('Count [n]')
ax[1].set_xlabel('Value [A.U]')
fig.suptitle("Histograms of 2 highest variance-explained PC's")
plt.show()


# ### Section f

# In[60]:


#performing
pca=PCA(n_components=20)
pca.fit(centered_mat)
prj_mat=pca.transform(centered_mat)
var_explained=pca.explained_variance_ratio_*100 #turning probability to precentage


# #### i

# In[61]:


#scree plot
fig, ax = plt.subplots()
ax.plot(var_explained,'o-',color=colors[2])
ax.bar(range(len(var_explained)),var_explained,color=colors[4])
ax.set_title('Scree Plot')
ax.set_xlabel('Principal Component [n]')
ax.set_ylabel('Explained Variance Precentage [%]')
plt.show()


# #### ii

# In[64]:


#plottting scatterplot
fig, ax = plt.subplots()
ax.scatter(prj_mat[:, 0], prj_mat[:, 1],color='purple',alpha=0.77)
plt.xlabel('PC1 ('+str(round(pca.explained_variance_ratio_[0]*100,3))+'% var explained) [A.U]')
plt.ylabel('PC2 ('+str(round(pca.explained_variance_ratio_[1]*100,3))+'% var explained) [A.U]')
plt.title('PCA result using sklearn (total variance explained='+str(round((pca.explained_variance_ratio_[0]+pca.explained_variance_ratio_[1])*100,3))+'%)')
plt.show()


# ### Section g (bonus)

# In[8]:


variances=LFP.var(axis=0) #array of variation within feature
top5vars=np.argsort(variances)[-5:] #top 5 features by variation
weights=np.abs(feature_vector[:,0]) #array of PC1 weights of each feature
top5weights=np.argsort(weights)[-5:] #top 5 features by PC1 weights
print('The top 5 features(in order) with the highest variance are: ',top5vars)
print('The top 5 features(in order) by PC1 weights are: ',top5weights)


# ## Question 2

# In[5]:


#accessory functions
def Centroid_Distance(data,centroids):
    '''
    Function for finding the distance of each data point to each centroid

    Parameters
    ----------
    data : Numpy Array
        Given Data Array.
    centroids : Numpy Array
        Array of centroid coordinates.

    Returns
    -------
    dist_dic : Dictionary
        Dictionary where keys are the index of a data point and values are a list of distances from each centroid.

    '''
    dist_dic={} #creating the dict
    for i in range(len(data)): #loop for iterrating over all the data points
        len_lst=[] #list for storing distances
        for j in range(len(centroids)): #loop for iterrating over centroids
            len_lst.append(np.linalg.norm(data[i]-centroids[j])) #calculating eucalidean distance from the point to the centroid and appending it
        dist_dic[i]=len_lst
    return dist_dic

def Clusters(data,dist_dic,centroids):
    '''
    Function for allocating each data point to a cluster by the minimal distance

    Parameters
    ----------
    data : Numpy Array
        Given Data Array.
    dist_dic : Dictionary
        Dictionary where keys are the index of a data point and values are a list of distances from each centroid.
    centroids : Numpy Array
        Array of centroid coordinates.

    Returns
    -------
    cluster_dict : Dictionary
        Dictionary where keys are indices of clusters and values are list of data points allocated to that cluster.

    '''
    cluster_dict={i: [] for i in range(len(centroids))} #creating empty dict
    for i in range(len(data)): #loop for iterrating over data points
        MINi=dist_dic[i].index(min(dist_dic[i])) #finding the index of the cluster with the minimal distance
        cluster_dict[MINi].append(i) #adding the index of the point to the proximal cluster
    return cluster_dict

def centroidim(data,cluster_dict):
    '''
    Function for calculating the mean of each cluster and making it the new centroid

    Parameters
    ----------
    data : Numpy Array
        Given Data Array.
    cluster_dict : Dictionary
        Dictionary where keys are indices of clusters and values are list of data points allocated to that cluster.

    Returns
    -------
    new_centroids : Numpy Array
        Array of the coordinates of the new centroids.

    '''
    new_centroids=np.array([[],[]]).T #creating an empty array
    for value in cluster_dict.values(): #iterrating over the dictionary
        points=[data[i] for i in value] #translating the points' indices to their coordinates
        if points==[]: #condition for if the centroid is not nearest to any point
            centroid=np.array([np.nan,np.nan])
        else:        
            centroid=np.mean(points,axis=0) #calculating the cluster's mean
        new_centroids=np.vstack([new_centroids,centroid]) 
    return new_centroids


# In[6]:


def K_means(data,centroids):
    '''
    Function for performing the K-Means algorithm on a given data set
    
    Parameters
    ----------
    data : Numpy Array
        Given Data Array.
    centroids : List
        List of lists of centroid coordinates.

    Returns
    -------
    new_centroids : Array
        Array of the coordinates of the final centroids.
    clustered_dict : Dictionary
        Dictionary where keys are the indices of the clusters and values are arrays of the coordinates of the points belonging to those clusters.

    '''
    centroids=np.array(centroids) #converting list to array
    dist_dic=Centroid_Distance(data, centroids) #calculating distances from points to centroids
    cluster_dict=Clusters(data, dist_dic, centroids) #clustering points
    new_centroids=centroidim(data, cluster_dict) #calculating cluster means
    same=np.array_equal(centroids,new_centroids) #stop condition
    while not same: #while loop that runs as long as there is change in the centroids
        dist_dic=Centroid_Distance(data,new_centroids) #calculating distances from points to centroids
        cluster_dict=Clusters(data, dist_dic, new_centroids) #clustering points
        centroidss=centroidim(data, cluster_dict) #calculating cluster means
        same= np.array_equal(new_centroids,centroidss) #stop condition
        new_centroids=centroidss #resetting centroid variable
    #Converting point index to coordinate
    clustered_dict={}
    for key,value in cluster_dict.items(): 
        clustered_dict[key]=np.array([data[i] for i in value])
    return new_centroids,clustered_dict


# In[7]:


data=sio.loadmat('HW3_2.mat')['data'] #loading HW data


# ### Section a

# In[48]:


#plotting
fig,ax=plt.subplots()
centroids,clustered_dict=K_means(data,centroids=[[0,0],[1,1],[-1,-1],[-1,1]])
for key,value in clustered_dict.items():
    ax.scatter(value[:,0],value[:,1],c=colors[key],alpha=0.2)
    ax.scatter(centroids[key][0],centroids[key][1], s=150, c=colors[key],edgecolor='black',label='Centroid'+str(key+1))
    plt.xlabel('X [A.U]')
    plt.ylabel('Y [A.U]')
    plt.title('K-Means Clustering for k=4')
    plt.legend(loc='upper right')
plt.show()


# ### Section c

# In[55]:


#plotting original data
fig,ax=plt.subplots(2,2,figsize=(16, 12))
ax[0,0].scatter(data[:,0],data[:,1],c=colors[0])
ax[0,0].set_title('(1)Original Data')
ax[0,0].set_xlabel('X [A.U]')
ax[0,0].set_ylabel('Y [A.U]')

#plotting k=4
centroids,clustered_dict=K_means(data,centroids=[[0,0],[1,1],[-1,-1],[-1,1]]) #K-means algorithm with 4 centroids
for key,value in clustered_dict.items(): #loop for plotting and coloring
    ax[0,1].scatter(value[:,0],value[:,1],c=colors[key],alpha=0.2)
    ax[0,1].scatter(centroids[key][0],centroids[key][1], s=150, c=colors[key],edgecolor='black')
    ax[0,1].set_xlabel('X [A.U]')
    ax[0,1].set_ylabel('Y [A.U]')
    ax[0,1].set_title('(2) K-means Clustering with k=4')

#plotting k=5
centroids,clustered_dict=K_means(data,centroids=[[0,-10],[7,7],[-6,-1],[-5,7],[10,-10]]) #K-means algorithm with 5 centroids
for key,value in clustered_dict.items(): #loop for plotting and coloring
    ax[1,0].scatter(value[:,0],value[:,1],c=colors[key],alpha=0.2)
    ax[1,0].scatter(centroids[key][0],centroids[key][1], s=150, c=colors[key],edgecolor='black')
    ax[1,0].set_xlabel('X [A.U]')
    ax[1,0].set_ylabel('Y [A.U]')
    ax[1,0].set_title('(3) K-means Clustering with k=5')

#plotting k=6
centroids,clustered_dict=K_means(data,centroids=[[0,-10],[7,7],[-6,-1],[-5,7],[10,-10],[6.9,-4.20]]) #K-means algorithm with 6 centroids
for key,value in clustered_dict.items(): #loop for plotting and coloring
    ax[1,1].scatter(value[:,0],value[:,1],c=colors[key],alpha=0.2)
    ax[1,1].scatter(centroids[key][0],centroids[key][1], s=150, c=colors[key],edgecolor='black')
    ax[1,1].set_xlabel('X [A.U]')
    ax[1,1].set_ylabel('Y [A.U]')
    ax[1,1].set_title('(4) K-means Clustering with k=6')

fig.suptitle('Section c Figure')
plt.show()


# ### Section d

# In[34]:


SSElst=[] #list to contain SSE values
CentroidList=[] #list to contain centroid coordinates
Ks=range(2,16)
for k in Ks:
    kmeans = KMeans(n_clusters=k) # Create an instance of the KMeans class
    kmeans.fit(data) # Fit the model to the dataset   
    clusters = kmeans.predict(data) # Predict the cluster for each data point
    CentroidList.append(kmeans.cluster_centers_) # Access the cluster means
    SSElst.append(kmeans.inertia_) #Access the SSE of the clustering


# In[41]:


#elbow plot
fig,ax=plt.subplots()
plt.grid()
ax.plot(Ks,SSElst,'o-',c=colors[0])
plt.xlabel('Number of clusters [k]')
plt.ylabel('SSE [A.U]')
plt.title('K-means clustering SSE vs. number of clusters')
plt.show()


# ### Section e

# In[11]:


#find point of conflict by similar distances to two clusters
centroids,clustered_dict=K_means(data,centroids=[[0,-10],[7,7],[-6,-1],[-5,7],[10,-10]])
centroidsdist=Centroid_Distance(data, centroids)
last2centroidsdist={key: [value[0],value[-1]] for key, value in centroidsdist.items()} #remove irrelevant clusters
smallest_difference = float("inf")
index = None
for k,dists in last2centroidsdist.items(): #loop for finding most similar distance 
    difference = abs(dists[0] - dists[1]) # Calculate the difference between the two distances  
    if difference < smallest_difference: 
        smallest_difference = difference
        index=k
print('The point of conflict is: ',index)
print('The distances from the two clusters are: ',last2centroidsdist[167])


# In[12]:


#plotting the clusters
fig,ax=plt.subplots()
centroids,clustered_dict=K_means(data,centroids=[[0,-10],[7,7],[-6,-1],[-5,7],[10,-10]])
plt.scatter(data[index,0],data[index,1],c='black',s=300,marker='x',linewidth=3,label='Conflict Point') #adding marking for section d
for key,value in clustered_dict.items():
    ax.scatter(value[:,0],value[:,1],c=colors[key],alpha=0.7)
    ax.scatter(centroids[key][0],centroids[key][1], s=150, c=colors[key],edgecolor='black',label='Centroid'+str(key+1))
    plt.xlabel('X [A.U]')
    plt.ylabel('Y [A.U]')
    plt.title('K-Means Clustering for k=5')
    plt.legend(loc='upper right')
plt.show()


# ## Question 3

# In[13]:


data=sio.loadmat('HW3_3.mat')['data']


# ### Section a

# In[50]:


#Scatterplot
fig,ax=plt.subplots()
ax.scatter(data[:,0],data[:,1],c=colors[5],s=10)
plt.xlabel('X [A.U]')
plt.ylabel('Y [A.U]')
plt.title('Question 3 Data Scatter Plot')


# ### Section b

# In[14]:


fig,ax=plt.subplots(2,3,figsize=(16, 12))
plt.suptitle('Implementation of Mixture of Gaussian for k clusters')
indices={5: (0, 0), 6: (0, 1), 7: (0, 2), 8: (1, 0), 9: (1, 1), 10: (1, 2)}
for k in range(5,11):   
    gmm = mixture.GaussianMixture(n_components=k)
    gmm.fit(data)
    cluster_labels=gmm.predict(data)
    clustered_points=np.column_stack((cluster_labels,data))
    for i in range(k):
        mask=np.isin(clustered_points[:, 0], [i])
        ax[indices[k]].scatter(clustered_points[mask,1],clustered_points[mask,2],c=colors[i],alpha=0.8,s =5)
        ax[indices[k]].set_xlabel('X [A.U]')
        ax[indices[k]].set_ylabel('Y [A.U]')
        ax[indices[k]].set_title('k='+str(k))
plt.show()


# ### Section c

# In[16]:


#fitting
gmm = mixture.GaussianMixture(n_components=3)
gmm.fit(data)
cluster_labels=gmm.predict(data)
clustered_points=np.column_stack((cluster_labels,data))
cov3=gmm.covariances_
mean3=gmm.means_
print('The means of the Gaussians:\n',mean3)
print('The covariances of the Gaussians:\n',cov3)


# In[17]:


#plotting
fig,ax=plt.subplots()
plt.scatter(data[2887,0],data[2887,1],c='black',s=420,marker='x',linewidth=3) #adding marking for section d
for i in range(3):
    mask=np.isin(clustered_points[:, 0], [i])
    ax.scatter(clustered_points[mask,1],clustered_points[mask,2],c=colors[i],alpha=0.8,s =10)
    ellipse = Ellipse(xy=(mean3[i,0],mean3[i,1]), width=4*np.sqrt(cov3[i][0, 0]), height=4*np.sqrt(cov3[i][1, 1]),angle=np.rad2deg(np.arctan2(2*cov3[i][0, 1], cov3[i][0, 0]-cov3[i][1, 1])),facecolor='none',edgecolor='blue',linestyle='--')
    ax.add_artist(ellipse)
plt.xlabel('X [A.U]')
plt.ylabel('Y [A.U]')
plt.title('Mixture of Gaussian Clustering for k=3')
plt.show()


# ### Section d

# In[49]:


posterior=gmm.predict_proba(data)[2887] 
posterior=posterior[posterior!=min(posterior)]
print('The point belongs to one cluster with a posterior probability of: '+str(posterior[0])+'\n and to another cluster with a probability of: '+str(posterior[1]))

