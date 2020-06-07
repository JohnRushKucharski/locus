
#%%[markdown]
# # Storm Centering
# The file analyzes spatial patterns found in annual maximum daily precipitation records taken from the Livneh dataset. 
# This analysis is performed for the North Branch of the Potomac River Watershed located in Western Maryland.

#%%[markdown] 
#<b>First</b>, an input directory containing the watershed annual maximum precipitation data is specified.
#The files in this directory following the 'prec.<year>.csv' naming convention, from a specified range of <year>s are imported as pandas dataframes.
#The files are reduced to contain only the precipitation depths from each of the watershed grid cells, and appended to create a single annual maximum file. 

#%%
import pandas as pd

input_directory: str = 'C:/Users/q0hecjrk/Documents/_Projects/northbranch/data/livneh/daily/max/1day/'
for yr in range(1915, 2012):
    if yr == 1915:
        df = pd.read_csv(input_directory + 'prec.' + str(yr) + '.csv').sort_values(by = ['id']).filter(['id', 'prec']).rename(columns = {'prec' : str(yr)}).set_index('id').T
    else:
        df2 = pd.read_csv(input_directory + 'prec.' + str(yr) + '.csv').sort_values(by = ['id']).filter(['id', 'prec']).rename(columns = {'prec' : str(yr)}).set_index('id').T
        df = df.append(df2)
#should be 97 rows (1 per year) x 130 columns (1 per grid cell)
df.count

#%%[markdown]
# <b>Next</b>, a principial component decomposition is performed to reduce the dimensionality of the task. 
# It shows that the first 4 components explain about 90% of the variance in the data. 
# The results of the PCA analysis are saved to a dataframe.

# %%
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

pca = PCA(n_components=20)
#df_std = StandardScaler().fit_transform(df)
components = pca.fit_transform(StandardScaler().fit_transform(df))
df_components = pd.DataFrame(components)
print(pca.explained_variance_ratio_)

#%%
import matplotlib.pyplot as plt

features = range(pca.n_components_)
plt.bar(features, pca.explained_variance_ratio_)
plt.xlabel('PCA features')
plt.ylabel('variance %')
plt.xticks(features)

#%%[markdown]
# The next block experiments with different k Means cluster sizes (by changing k). 
# The plot shows that much of the variance in the data is explained by the first 4 cluster centers. 

# %%
from sklearn.cluster import KMeans

ks = range(1, 10)
cluster_centers = []
for k in ks:
    model = KMeans(n_clusters=k) #create kMeans model with k clusters
    model.fit(df_components.iloc[:,:4]) #fit model to first 4 principal components   
    cluster_centers.append(model.inertia_) #append variance explained by clusters centers
  
plt.plot(ks, cluster_centers, '-o')
plt.xlabel('number of clusters, k')
plt.ylabel('inertia')
plt.xticks(ks)
plt.show()

#%%[markdown]
# The first 4 clusters are retained.
# The labels show the mapping of the 97 years of data into each of the 4 clusters.
# %%
model = KMeans(n_clusters = 4)
clusters = model.fit(df_components.iloc[:,:4])
print(clusters.labels_)
print(len(clusters.labels_))

#%%[markdown]
# Next step, average each clusters data and create heat maps for each...
# It may be that 3 clusters was sufficient. Some visual inspection of average maps would be helpful.