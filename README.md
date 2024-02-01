#K-Means Clustering from Scratch for One-Hot Encoded Data

This repository contains a K-Means clustering implementation from scratch for clustering patient sequences based on their one-hot encoded representations.

Overview:

One-Hot Encoding:
Patient sequences are loaded from 70 data files, representing a series of events or observations. Each sequence is then one-hot encoded, converting each character into a binary vector. The resulting one-hot encoded data is organized into a matrix.

K-Means Clustering:
The K-Means clustering algorithm is applied to the one-hot encoded data. The process involves initializing clusters, padding sequences to equal lengths, iteratively assigning data points to clusters, and updating centroids. The number of clusters (k) can be adjusted to explore different clusterings.

Silhouette Score Evaluation:
The quality of the clusters is evaluated using the silhouette score, measuring how similar each data point is to its own cluster compared to other clusters. The silhouette score provides insights into the effectiveness of the clustering.

Academic Integrity Warning:

This implementation is provided for educational purposes to understand the one-hot encoding and K-Means clustering processes. Users are advised not to use this code for academic assignments without proper attribution. Plagiarism in educational contexts can lead to severe consequences, and students are encouraged to understand the code and use it as a reference, seeking permission from their educational institution before incorporating it into their work. It is crucial to adhere to academic integrity policies and guidelines when utilizing external resources for academic purposes
