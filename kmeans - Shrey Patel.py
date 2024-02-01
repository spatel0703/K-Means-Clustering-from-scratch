#!/usr/bin/env python
# coding: utf-8

# # One Hot Encoding 

# In[1]:


from sklearn.preprocessing import OneHotEncoder
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#get folder and file path
folder_path = r'C:\\Users\\shrey\\CSC 4850 - Machine Learning\\Homework 1 - KNN and K-Means'

#load all 70 data files
patients_data = []
max_length = 0

for i in range(1, 71):
    filename = f'{folder_path}\\data-{i:02d}' 
    with open(filename, 'r') as file:
        patient_sequence = file.read().strip()
        patients_data.append(patient_sequence)
        max_length = max(max_length, len(patient_sequence))

#one hot encoding
encoder = OneHotEncoder(sparse=False, handle_unknown='ignore')

#transforms each patients sequence to one hot encoding
one_hot_data = []
for patient_sequence in patients_data:
    padded_sequence = patient_sequence.ljust(max_length, '0')
    one_hot_sequence = encoder.fit_transform(np.array(list(padded_sequence)).reshape(-1, 1))
    one_hot_data.append(one_hot_sequence.flatten())

#combines all vectors to get a final matrix
final_data_matrix = np.array(one_hot_data)


# In[2]:


final_data_matrix


# # The actual kmeans function

# In[3]:


from tqdm import tqdm  

def initialize_clusters(data, k):
    #randomly initilzie k clusters
    indices = np.random.choice(len(data), k, replace=False)
    return np.vstack(data[indices])

def pad_sequences(data):
    #pad sequences with 0s to make equal lengths
    max_length = max(len(seq) for seq in data)
    padded_data = [np.pad(seq, (0, max_length - len(seq))) for seq in data]
    return np.vstack(padded_data)

def assign_to_clusters(data, centroids):
    #assigns data point to nearest cluster
    distances = np.linalg.norm(data[:, np.newaxis, :] - centroids, axis=2)
    return np.argmin(distances, axis=1)

def update_centroids(data, cluster_assignments, k):
    centroids = np.zeros((k, data.shape[1]))
    for i in range(k):
        cluster_points = data[cluster_assignments == i]
        if len(cluster_points) > 0:
            centroids[i] = np.mean(cluster_points, axis=0)
    return centroids

def kmeans(data, k, max_iterations=100, tolerance=1e-4):
    #this performs kmeans clustering
    centroids = initialize_clusters(data, k)
    
    for iteration in tqdm(range(max_iterations), desc="K-Means Iterations", unit="iteration"):
        cluster_assignments = assign_to_clusters(data, centroids)
        new_centroids = update_centroids(data, cluster_assignments, k)
        
        if np.linalg.norm(centroids - new_centroids) < tolerance:
            print(f"Converged after {iteration + 1} iterations.")
            break
        
        centroids = new_centroids
    
    return cluster_assignments

k = 4  #k can be changed to find different clusters

#padding matrix
padded_data_matrix = pad_sequences(final_data_matrix)

cluster_assignments = kmeans(padded_data_matrix, k)


# In[6]:


print(cluster_assignments)


# In[7]:


from sklearn.metrics import silhouette_score

#use padded matrix since that was used to find the clusters
silhouette_avg = silhouette_score(padded_data_matrix, cluster_assignments)

print(f"Silhouette Score: {silhouette_avg}")


# In[ ]:




