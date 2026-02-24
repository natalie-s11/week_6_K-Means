# %%
# load libraries
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

<<<<<<< HEAD

# %%
# load data
help(pd.read_csv)
df = pd.read_csv("house_votes_Dem.csv", encoding = 'Latin-1')


# %%
# take a look at the data
df.head()
df.info()


# %%
# separate out the numeric features
numeric_features = df[['aye','nay','other']]

=======
# %%
# load data
df = pd.read_csv("house_votes_Dem.csv",encoding='latin-1')

# %%
# take a look at the data
df.info()

# %%
# separate out the numeric features
c_num = df[["aye", "nay", "other"]]
>>>>>>> 7684c978d953cedec96bebe0e1f674d45c9288e5

# %%
# documentation for kmeans in sklearn
help(KMeans)
<<<<<<< HEAD


# %% build a kmeans model
kmeans = KMeans(n_clusters=3, random_state=42, verbose=1)
kmeans.fit(numeric_features)


# %% look at the information in the model
print(kmeans.cluster_centers_)
print(kmeans.labels_)


# %%
# add the cluster labels to the original data frame
df['clusters'] = kmeans.labels_
print(df.head())


#%% 
# Use a for loop to check different cluster numbers and see how the intertia changes
=======

# %% 
# build a kmeans model
kmeans = KMeans(n_clusters=3, random_state=42, verbose=1)
kmeans.fit(c_num)

# %% 
# look at the information in the model
print(kmeans.cluster_centers_)
print(kmeans.labels_)

# %%
# add the cluster labels to the original data frame
df['cluster'] = kmeans.labels_

# %% 
# simple plot of the clusters
help(plt.scatter)

# %%
# use a foor loop to check different cluster
# numbers and see how the inertia changes
>>>>>>> 7684c978d953cedec96bebe0e1f674d45c9288e5
inertias = []
k_values = range(1,10)
for k in k_values:
    kmeans = KMeans(n_clusters=k, random_state=42)
<<<<<<< HEAD
    kmeans.fit(numeric_features)
    inertias.append(kmeans.inertia_)





# %% 
# plot the inertia values to see if there is an elbow in the plot 
=======
    kmeans.fit(c_num)
    inertias.append(kmeans.inertia_)
  
# %%
# plot the inertia values to see if there is an elbow in the plot
>>>>>>> 7684c978d953cedec96bebe0e1f674d45c9288e5
plt.figure(figsize=(10,5))
plt.plot(k_values, inertias, marker='o')
plt.xlabel('Number of Clusters (k)')
plt.ylabel('Inertia')
<<<<<<< HEAD
plt.show()


=======
plt.show
# %%
>>>>>>> 7684c978d953cedec96bebe0e1f674d45c9288e5
