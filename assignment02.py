#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
from sklearn.datasets import fetch_olivetti_faces

X,y = fetch_olivetti_faces(return_X_y=True)


# In[2]:


import numpy as np
labels = pd.DataFrame(y).value_counts()
labels = np.vectorize(lambda x: x[0])(labels.keys().values)


# In[3]:


labels


# In[4]:


from sklearn.model_selection import train_test_split
# Given the scarcity of the dataset, an 80-20 ratio for train/test will be considered.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=True,random_state=42,stratify=y)


# In[5]:


from sklearn.model_selection import StratifiedKFold
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# With 8 samples available for each person for training, We set the n_splits=8 , so for each split, we'll
# have 7 records per person to train and 1 record to validate with.
skf = StratifiedKFold(n_splits=8,shuffle=True,random_state=42)
clf = Pipeline([
    ("scaler", StandardScaler()),
    ("nn", MLPClassifier(solver='adam', activation='tanh',hidden_layer_sizes=(80,60,40,40), random_state=42,max_iter=2000))
])

scores = []
for train_index, test_index in skf.split(X_train,y_train):
    X_train_batch, X_test_batch = X_train[train_index], X_train[test_index]
    y_train_batch, y_test_batch = y_train[train_index], y_train[test_index]
    clf.fit(X_train_batch,y_train_batch)
    pred = clf.predict(X_test_batch)
    score = accuracy_score(y_test_batch, pred)
    scores.append(score)
    print(score)
    
print(f'Average score: {np.mean(scores)}')


# In[6]:


pred = clf.predict(X_test)
score = accuracy_score(y_test, pred)


# In[7]:


score


# In[8]:


from sklearn.metrics import silhouette_samples, silhouette_score
from sklearn.cluster import KMeans

# For the first iteration the range(2,400,20) where considered,
# For the second iteration, the values from the highest scores
# in the range(100,130,5) were considered.
range_n_clusters = range(100,130,5)

silhouette_scores = {}
for n_clusters in range_n_clusters:
    clusterer = KMeans(n_clusters=n_clusters, random_state=42)
    cluster_labels = clusterer.fit_predict(X)
    silhouette_avg = silhouette_score(X, cluster_labels)
    print("For n_clusters =", n_clusters, "The average silhouette_score is :", silhouette_avg)
    silhouette_scores[n_clusters] = silhouette_avg

best_num_clusters = max(silhouette_scores, key=silhouette_scores.get)    


# In[9]:


best_num_clusters


# In[10]:


clusterer = KMeans(n_clusters=best_num_clusters, random_state=42)
cluster_labels = clusterer.fit_predict(X)


# In[12]:


X_train_new = clusterer.transform(X_train)
X_test_new = clusterer.transform(X_test)


# In[22]:


clf_new = Pipeline([
    ("scaler", StandardScaler()),
    ("nn", MLPClassifier(solver='adam', activation='tanh',hidden_layer_sizes=(80,60,40,40), random_state=42,max_iter=1000))
])

scores = []
for train_index, test_index in skf.split(X_train_new,y_train):
    X_train_batch, X_test_batch = X_train_new[train_index], X_train_new[test_index]
    y_train_batch, y_test_batch = y_train[train_index], y_train[test_index]
    clf_new.fit(X_train_batch,y_train_batch)
    pred = clf_new.predict(X_test_batch)
    score = accuracy_score(y_test_batch, pred)
    scores.append(score)
    print(score)
    
print(f'Average score: {np.mean(scores)}')


# In[23]:


pred = clf_new.predict(X_test_new)
score = accuracy_score(y_test, pred)


# In[24]:


score


# In[ ]:




