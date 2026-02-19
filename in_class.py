# %%
# load libraries
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
# %%
# load data
df = pd.read_csv("house_votes_Dem.csv" , encoding = "latin-1") 
# take a look at the data
df.info()
# %%
# separate out the numeric features
c_num = df[['aye' , 'nay' , 'other']]
# %%
# documentation for kmeans in sklearn
help(KMeans)
# %% build a kmeans model
kmeans = KMeans(n_clusters = 3, random_state = 42 , verbose = 1)
kmeans.fit(c_num)
# %% look at the information in the model
print(kmeans.cluster_centers_)
print(kmeans.labels_)
# %%
# add the cluster labels to the original data frame
df["cluster"] = kmeans.labels_
df
# %%
  
# %% simple plot of the clusters
help(plt.scatter)
 
# %%


