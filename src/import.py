# %% [markdown]
# # North Branch Extreme Precipitation
# The analysis in this notebook: 1. loads gridded precipitation data, 2. clips it to the spaital domain of the waterboundary data, 3. identifies the maximum precipitaton day for each year of data, and 4. writes this data to .shp and .csv files. <br>
# <br>
# <b> note:</b> running the this file can take up to 5 minutes per year of data being processed.
# <br>
# #### This analysis is based on two data sets: </br>
# 1. <b> Livneh Daily Precipitation Data </b> this provides daily gridded metrological observations from 1915 to 2011 with 1/16 degree (latitude or longitude) resolution.$^{1}$ <br>
# 2. <b> USGS Spatial Waterboundary Data </b> this provides nested watershed polygons. The analysis in this notebook focuses on the 'North Branch Potomac' 8 digit hydrologic unit code (HUC8) watershed.$^{2}$ <br>
# <br>
# References: <br>
# $^{1}$ Livneh B., E.A. Rosenberg, C. Lin, B. Nijssen, V. Mishra, K.M. Andreadis, E.P. Maurer, and D.P. Lettenmaier, 2013: A Long-Term Hydrologically Based Dataset of Land Surface Fluxes and States for the Conterminous United States: Update and Extensions, Journal of Climate, 26, 9384–9392. <br>
# $^{2}$ U.S. Geological Survey, 2020, National Waterboundary Dataset for 2 digit hydrologic Unit (HU) 02 (mid-atlantic), accessed April 11, 2020 at URL http://prd-tnm.s3-website-us-west-2.amazonaws.com/?prefix=StagedProducts/Hydrography/WBD/HU2/Shape/ <br>

# %% [markdown]
# ## Global Variables
# 1. Input-Output Directories
# %%
importpath: str = 'C:/Users/q0hecjrk/Documents/_Data/'
outputpath: str = 'C:/Users/q0hecjrk/Documents/_Projects/northbranch/data/'

# %% [markdown]
# 2. Livneh Data
# %%
years: list = [*range(1915, 2012, 1)]
variables: list = ['prec']
livnehdirectory: str = importpath +'Livneh/Daily/'
outputdirectory: str = outputpath +'livneh/daily/'
maxdaydirectory: str = outputdirectory + 'max/1day/'
coordinatesystem: str = '4326'
measurementcoordinatesystem: str = '3857'

# %% [markdown]
# 3. Watershed Boundaries
# %%
variable: str = 'HUC8'
code: str = '02070002'
name: str = 'North Branch Potomac'
watershedfile: str = importpath + 'Geospatial/Waterboundary/WBD_02_HU2_Shape/Shape/WBDHU8.shp'
coordinatesystem: str = '4326'

# %% [markdown]
# ## Import and Clip Data 
# %%
import dataimports.livneh as livneh
import dataimports.waterboundary as wbd 

# %% [markdown]
# 4. Watershed Geometries
# %%
nbshape = wbd.importwaterboundary(watershedfile, variable, code)
nbarea_km2 = nbshape.iloc[0]['AreaSqKm']
nbbox = wbd.boundaryboxfromshape(nbshape)
nbshape.head()

# %% [markdown]
# 5. Import Livneh Data, write out max day files and return max dates
# %%
infiles = livneh.inputfilepaths(livnehdirectory, variables, years)
outfiles = livneh.outputfilepaths(outputdirectory, infiles)
maxdays = livneh.processfiles(infiles, outfiles, nbbox, nbshape, nbarea_km2, maxdaydirectory)
print(maxdays)


