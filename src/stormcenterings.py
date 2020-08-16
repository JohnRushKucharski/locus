#%%[markdown]
# # Storm Centering
# The notebook analyzes the spatial patterns of annaul daily maximum precipitation. It performs this analysis on the North Branch of the Potomac Watershed, using a dataset constructed from the Livneh data $^{1}$. This dataset is constructed using by the <b>imports.py</b> contained in this module.</br>
#
# The analysis in this notebook: 1. generates a single datafile for analysis, 
# 2. explores two different normalization routines, 
# 3. projects the normalized data across its first n prinicpal components, 
# 4. clusters the data (projected onto its first n principal components) around k-Means, 
# <em>5...N subsquent steps will help us visualize, explore the results of the normalization, pca and clustering... </em></br>   
#
# References: <br>
# $^{1}$ Livneh B., E.A. Rosenberg, C. Lin, B. Nijssen, V. Mishra, K.M. Andreadis, E.P. Maurer, and D.P. Lettenmaier, 2013: A Long-Term Hydrologically Based Dataset of Land Surface Fluxes and States for the Conterminous United States: Update and Extensions, Journal of Climate, 26, 9384–9392. <br>

#%%[markdown]
# ## Global Variables
#%%
years = list(range(1915, 2012))
input_directory: str = 'C:/Users/q0hecjrk/Documents/_Projects/northbranch/data/livneh/daily/max/1day/'

#%%[markdown]
# ## Data Imports
# This section import data and generates files for analysis </b>
# 
# The data being analyzed includes the annual maximum day of precipitation from 1915 through 2011 for the North Branch of the Potomac Watershed in Western Maryland, USA.
# The data for each of these 97 days (between 1915 and 2011) contains the preciptation depth for all 130 livneh grid cells located within or intersected by the North Branch of the Potomac 8 digit Hydrologic Unit Code (HUC08) boundary.
#%%
import pandas as pd
import matplotlib.pyplot as plt
import dataimports.livneh as livneh

#%%[markdown]
# ### Event Dates
# The 'events' dataframe below contains the dates of the 97 events being analyzed.
# The histogram below displays the distribution of those events across months. 
# It appears about 62 events occured during the official North Atlantic hurricane season (June through November), thus 35 of the events occured outside the North Atlantic season (December through May).
#
# TODO: Add additonal analysis here... percentages, peak months, find storms from historic dates.
#%%
events = livneh.eventdates(input_directory, years)
ax = events['month'].plot.hist(bins = 12)
ax.set_xlabel("Month")

#%%[markdown]
# ### Dataset for Analysis
# The code below generates the primary file being analyzed in this notebook, labeled: 'df' in the code below. 
# It contains 97 rows, one for each year in the analysis period and 130 columns, one for each Livneh gridcell in the watershed. 
# Each cell in the dataset records the depth of precipitation in the Livneh grid cell (which cooresponds with its column) on the day (which cooresponds with its row).
# Therefore, the sum of a row's columns gives the total precipitation depth for the watershed on the day (row) being summed,
# and the sum of a columns row's gives the total precipitation recieved in that grid cell over the 97 days covered in the 97 year period of analysis.
#%%
df = livneh.aggregateprocessedfiles(input_directory, years)
df.head()

#%%[markdown]
# ## Normalization Routines
# <p>The data must be normalized, otherwise outliers will dominate the principal component analysis and clustering.
# The data can reasonably be expected to contain outliers for several reasons:
#
#  1. Event Magnitudes - the events being analyzed represent annaul maximum days of precipitation. 
# Therefore, to one degree or another all of the events being analyzed are 'outliers' from the perspective of the underlying precipitation distribution.
# Maximum annual precipitation day values such as these are typically fit to an extreme values distribution (EVD), used to model events taken from the tail of some other distribution.
# The EVDs model the asympotic behavior or the under distributions tail (or tails), therefore we should expect our 97 year sample to exhibit some of this asymptotic behanvior.
# 
# 2. Spatial Outliers - one would expect rainfall totals to be higher at higher elevations, as adiabatic cooling forces more moisture to rain out of the air.
# This orographic effect is likely to lead to some grid cells (or columns) with substaintially larger means and variability (perhaps). 
# Secondly, large rain event over an areas the size of the one being analyzed are typically dominated by advective synoptical scale events,
# which are driven by specific patterns of atmopheric flow (cite Schlef). We seek to test if this mode of spatial variability drives different clusterings in the data.</p>

#%%[markdown]
# <b>Two</b> normalization routines are explored below. For simplicity they are refered to as: 
# (1) the "hypothesis-based" routine, and (2) the "nieve" routine.
# Both normalization routines normalize the data using the equation: 
# 
#       (x - u) / s
# 
# where x is the observed rainfall total for the cell in the dataset, u is the mean of the data being normalized, and s is the standard deviation (of the data being normalized).
# The methods differ primarily in the data used to measure u and s. 
# 
# The "hypothesis-based" normalization routine is a two step process. 
# <b>First</b> events (or rows) of data are normalized. 
# During this step, u represents an average amount of rainfall recieved in a livneh grid cell <em>during that event</em>.
# After this step the dataset values describe the size of each livneh grid cell's deviation from this mean, expressed in standard deviations units.
# For example, the value 2 would describe a livneh grid cell with a rainfall total that was 2 standard deviations above the mean gridcell total for that particular row's event.
# <b>TODO: I think it could be valuable to also consider clustering pcs generated from this dataset, since this should capture the combined orographic + atmospheric flow patterns of precipitation.</b>
# <b>Next</b> the columns (or livneh grid cell) values are normalized.
# In the example above, I hypothesize that the value of 2 (or whatever value is found) may <em>not</em> be as anomolious as it would seem on face value.
# Perhaps, the grid cell is located in a zone of extreme orographic lift, and as a result it tends to recieve much more than an average grid cell amount of rain - across all 97 days in the analysis. 
# In this case, the value of 2 may be an average value <em>for that grid cell</em> to help disentangle the orographic and storm centering patterns impact on rainfall totals we normalize this column of data.
# If in fact the value 2 in the first step was a local (i.e. grid cell average) we wil be left with a data set that describes the deviations from this localized average in standard deviation unit.
# For example, now a value of 2 would represent an anomolously high rainfall total <em>for that grid cell</em> based on its average across all event in the period of analysis.
#
# The "nieve" normalization routine. Simply applies the normalization equation to all rows and columns simultaneously.
# Thus, the mean: u represents the average livneh grid cell total across all grid cells and events. 
# A value of 2 after this normalization routine would indicate a large rainfall total (two standard deviation above the mean) for that grid cell,
# relative to all grid cells and all events. This 2 could be product of an anomolously large event - in which case a dispropotionate share of the grid cells in that row would have postive values.
# On the other hand the value 2 could be representative of a typically wet grid cells (due to orgographics or other factors) - in which case a disproportionate share of the cells in that column would have positive values.
# Or it could be due to both, this a more emperical view of the data.

#%%[markdown]
# ### Hypothesis-based routine
# This routine normalizes the data with the apriori assumption that: 
# 
# (1) rows must be normalized, to remove the impact of outlier events; and
# 
# (2) if columns are normalized to remove the impact of orographic impacts 
# 
# <b>then</b> the remaining variability that will be captured by the PCA and patterns that will be identified by the k-Means clustering will be driven by patterns atmospheric flow (and perhaps other unknown phenomenon).

#%%
import statistics

def standardize_events(df: pd.DataFrame) -> pd.DataFrame:
    '''
    Normalizes the row data using the formula: (x - u) / s, 
    where x is a value in one of the row's columns, u is the row mean and s is the row standard deviation.
    Assumes each row contains an list of the grid cell precipitaiton values for a particular event or year.

    The resulting dataframe reports precipitation values for each grid cell in terms of unit variance for that event's grid cell values.
    The idea is that this normalization capture both normal orographically influenced spatial patterns as well as spatial characteristics of the storm.
    
    If these values are averaged across all events or years it should provide information about the normal (orographically influenced) spatial patterns in the rainfall.
    '''
    new_df = pd.DataFrame(columns = df.columns)
    for index, row in df.iterrows():
        data = list(row)
        u = statistics.mean(data)
        s = statistics.stdev(data) 
        new_row: list = []
        for x in data:
            new_row.append((x - u) / s)
        new_df.loc[index] = new_row
    return new_df
normevents = standardize_events(df)

def standardize_grids(df: pd.DataFrame) -> pd.DataFrame:
    '''
    Normalizes the column data using the formula: (x - u) / s, 
    where x is a value in a row of one of the columns, u is the column mean and s is the column standard deviation.
    Assumes each column contains an list of a grid cell precipitaiton values for all the events or years of interest.

    If the events have been standardized then this will report precipitation values for each grid cell as deviations (of unit variance) of that specific grid cell's normalized portion of the event total.
    The idea is that this process of standardizing by event and then standardizing by grid cell should provide deviations from the normal (oragraphically influenced) spatial characteristics of rainfall patterns in the watershed.
    
    If the events have NOT been standarized first then the standarized results will be heavily influenced by the size of the event, rather than the spatial characteristics fo the storm.
    '''
    new_df = pd.DataFrame(index = df.index)
    for name, col in df.iteritems():
        data = list(col)
        u = statistics.mean(data)
        s = statistics.stdev(data)
        new_col: list = []
        for x in data:
            new_col.append((x - u) / s)
        new_df[name] = new_col
    return new_df
hypothesis_df = standardize_grids(normevents)
hypothesis_df.head()

#%%[markdown]
# # TODO: PCA and kMeans clustering on hypothesis-based normalization
# --------------------------------

#%%[markdown]
# ### Nieve routine
# This routine normalizes the entire dataset in one pass. 
# It is an emperical look at the dataset, not prefaced with any pre-concieved structure
#%%
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

pca = PCA(n_components=20)
df_std = StandardScaler().fit_transform(df)
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