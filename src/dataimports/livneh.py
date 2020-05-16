import geopandas
import pandas as pd
import xarray as xr 
import dataimports.waterboundary as wbd 

from shapely.geometry import Polygon

def inputfilepaths(directory: str, vars: list, yrs: list) -> list:
    '''
    Generates a list of file paths in the specified directory with the form <var>.<year>.nc

    Inputs: 
    (1) directory: string directory path. 
    (2) vars: list of variable identification strings (i.e. prec, tmin, tmax, etc.).
    (3) yrs: list of year strings in the form yyyy.
    Output: a list of string file paths.
    '''
    files: list = []
    print('Input file list: \n\t[')
    for var in vars:
        for yr in yrs:
            filename: str = str(var) + '.' + str(yr) + '.nc'
            files.append(directory + filename)
            print('\t' + '  ' + filename)
    print('\t]')
    return files

def outputfilepaths(directory: str, filelist: str, extension: str = '.csv') -> list:
    '''
    Generates a list of file paths in the form <var>.<year>.<extension>

    Inputs:
    (1) directory: string directory path.
    (2) filelist: list of paths to files that will be processed.
    (3) extension: string output file extension, .csv by default.
    Output: a slist of string file paths.
    '''
    print('Output file list: \n\t[')
    files: list = []
    for file in filelist:
        outpath = outputfilepath(directory, file, extension)
        files.append(outpath)
    print('\t]')
    return files

def outputfilepath(directory: str, file: str, extension: str = '.csv'):
    split: list = file.rsplit('/', 1)
    file: str = split[1]
    split: str = file.rsplit('.', 1)
    name: str = split[0]
    filename = name + extension
    print('\t' + '  ' + filename)
    return directory + filename

def processfiles(files: list, outputs: list, box: list, shape: geopandas.GeoDataFrame, shapearea: float, maxdaydirectory: str):
    #dfs = []
    maxdays = {}
    for i, file in enumerate(files):
        print(i)
        df = processfile(file, box, shape)
        if i == 0:
            print('Generating IDs dataframe...')
            ids = km2areasandIDs(df, shapearea)
            print(ids.head())
        print('\t Merging areas and IDs...')
        df = mergebylatlon(df, ids)
        #dfs.append(df)
        df.to_csv(outputs[i])
        day = maxday(df, outputfilepath(maxdaydirectory, file))
        maxdays.update({day[0] : day[1]})
    return maxdays

def maxday(df, outpath: str):
    print('Computing max day:')
    df['weightedprec'] = df['prec'] * df['areaweight']
    day = df.groupby(['time', 'year', 'month', 'day'])['weightedprec'].sum().reset_index()
    maxprec = day['weightedprec'].max()
    maxday = day[day['weightedprec'] == maxprec]
    maxdate = maxday.iloc[0]['time']
    maximum = df[df['time'] == maxdate]
    maximum.to_csv(outpath)
    entry = (maxdate, maxprec)
    print(entry)
    return  entry

def processfile(filepath: str, boundingbox: list, mask: geopandas.GeoDataFrame, coordinatesystem: str = '4326', shift: int = 1/32):
    df = importfile(filepath, boundingbox)
    print('\t Creating file geometry...')
    gdf = convert2geodataframe(df, coordinatesystem)
    print('\t Converting livneh points to grid geometry...')
    grids = points2grids(gdf, shift)
    print('\t Clipping data to watershed polygon (this may take a while)...')
    wbgrids = clipdataframe(gdf, mask)
    return wbgrids

def importfile(filepath: str, boundingbox: list)-> pd.DataFrame:
    '''
    Imports a Livneh NetCDF file and returns a pandas DataFrame

    Inputs:
    (1) filepath: Livneh NetCDF string file path
    (2) boundingbox: list of latitude and longitude coordinates in form [E, S, W, N].
    Output: a pandas DataFrame
    '''
    print('Importing ' + str(filepath))
    #Import NetCDF file as XArray data
    netCDF = xr.open_dataset(filepath)
    if boundingbox is not None:
        # bounding box is based on 180 longitudes
        # livneh data is based on 360 degree longitude   
        print('\t Clipping data to boundary box...')
        netCDF = netCDF \
        .where(netCDF.lon > boundingbox[0] + 360, drop=True) \
        .where(netCDF.lat > boundingbox[1], drop=True) \
        .where(netCDF.lon < boundingbox[2] + 360, drop=True) \
        .where(netCDF.lat < boundingbox[3], drop=True)
    df: pd.DataFrame = netCDF.to_dataframe().reset_index()
    df: pd.DataFrame = __180longitude(df)
    df: pd.DataFrame = __processdates(df)
    return df

def convert2geodataframe(df: pd.DataFrame, coordinatesystem = '4326'):
    gdf = geopandas.GeoDataFrame(df, geometry = geopandas.points_from_xy(df['lon'], df['lat']), crs='EPSG:' + str(coordinatesystem))
    return gdf

def points2grids(gdf: geopandas.GeoDataFrame, shift: int = 1/32):
    '''
    Turns Livneh gridcell point geometry into a Livneh grid polygon.

    Inputs:
    (1) gdf: geopandas.GeoDataFrame contining the Livneh grid cell centroids.
    (2) shift: a radius shift from gridcell centroid to the grid polygon edges. 1/32nd degree by default. 
    Output: a geopandas.GeoDataFrame containing the Livneh grids polygon geometry. 
    '''
    geometry: list = []
    for i, row in gdf.iterrows():
        w: float = float(row['lon']) - shift
        e: float = float(row['lon']) + shift
        n: float = float(row['lat']) + shift
        s: float = float(row['lat']) - shift
        lons, lats = [w, e, e, w], [n, n, s, s]
        geometry.append(Polygon(zip(lons, lats)))
    gdf['geometry'] = geometry
    return gdf

def clipdataframe(gdf: geopandas.GeoDataFrame, mask: geopandas.GeoDataFrame) -> geopandas.GeoDataFrame:
    df = gdf.to_crs(mask.crs)
    return geopandas.clip(df, mask, keep_geom_type=False)

def __180longitude(df: pd.DataFrame) -> pd.DataFrame:
    df['lon'] = df['lon'].apply(lambda x: x - 360)
    return df
def __processdates(df: pd.DataFrame) -> pd.DataFrame:
    dt: list = df['time'].apply(lambda x: str(x).split("-", 4))
    df['year'] = dt.apply(lambda y: int(y[0]))
    df['month'] = dt.apply(lambda m: int(m[1]))
    df['day'] = dt.apply(lambda d: int(d[2].split()[0])) 
    return df

def km2areasandIDs(gdf: geopandas.GeoDataFrame, totalarea: float):
    t = gdf.iloc[0]['time']
    df = gdf[gdf['time'] == t]
    df = df.sort_values(by=['lat', 'lon'], ascending=[False, True])
    df = df.reset_index()
    df['id'] = df.index
    df = df.to_crs('EPSG:3857')
    df['area_km2'] = df['geometry'].area
    df['area_km2'] = df['area_km2'].apply(lambda x: x / 1000000)
    df['wsarea_km2'] = totalarea
    df['areaweight'] = df['area_km2'] / totalarea
    df = df.filter(['lat', 'lon', 'area_km2', 'wsarea_km2', 'areaweight', 'id'])
    return df

def mergebylatlon(left, right):
    df = left.merge(right, on=['lat', 'lon'], how='inner')
    return df



def importlivnehfiles(filepaths: list, maskingdata = geopandas.GeoDataFrame, grids: bool = True, crs: str = '4326', max: bool = True, outputfilepaths: list = []):
    '''
    Imports a set of livneh files returning a list of the geopandas GeoDataFrames

    Inputs:
    (1) a string list of livneh netCDF filepaths
    (2) a geopandas GeoDataFrame containing the watershed of interest, used to clip the data
    (3) a boolean indicating if grids or points should be returned, the default value True indicates grids
    (4) a string identifier for the coordinate reference system, '4356' by default.
    Output: a list of geopandas GeoDataFrames
    '''
    i: int = 0
    dfs: list = []
    export: bool = False
    if len(outputfilepaths) == len(filepaths):
        export = True
    # create a bounding box for importing the netCDF files
    boundingbox = wbd.boundaryboxfromshape(maskingdata, buffer=True, buffersize=1/32)
    for file in filepaths:
        print('Processing ' + str(file) + '...')
        netCDF = xr.open_dataset(file)
        # edit longitudes: livneh longitudes are in 360 degree format, but the bounding boxes are in 180 (e.g. negative west longitude) format
        netCDF = netCDF \
        .where(netCDF.lon > boundingbox[0] + 360, drop=True) \
        .where(netCDF.lat > boundingbox[1], drop=True) \
        .where(netCDF.lon < boundingbox[2] + 360, drop=True) \
        .where(netCDF.lat < boundingbox[3], drop=True) 
        # flattening the mult-index dataframe that comes fromfrom xarray
        df: pd.DataFrame = __cleaner(netCDF.to_dataframe().reset_index())
        # convert to geopandas with lat lon points
        print('\t...creating file geometry')
        df = geopandas.GeoDataFrame(df, geometry=geopandas.points_from_xy(df['lon'], df['lat']), crs='EPSG:' + str(crs))
        if grids:
            # convert to point to grids
            df = __points2grids(df, crs=crs)
        print('\t...clipping file to masking layer')
        df = geopandas.clip(df, maskingdata, keep_geom_type=False)
        
        # max precipitation day - make it a function that is passed in.
        if max:
            a = df.to_crs(epsg='3857')
            a['area_m2'] = a['geometry'].area
            wsarea = a[(a['month'] == 1) & (a['day'] == 1)]
            wsarea['basin_area_m2'] = wsarea['area_m2'].sum()
            wsarea = wsarea.filter(['time', 'basin_area_m2'])
            merge = a.merge(wsarea, on='time')
            merge['weights'] = merge['area_m2'] / merge['basin_area_m2']
            merge['weightedprec'] = merge['prec'] * merge['weights']
            day = merge.groupby(['time', 'year', 'month', 'day'])['weightedprec'].sum().reset_index()
            print(str(day['weightedprec']) + ' on ' + str(day['time']))
            time = day.iloc[0]['time']
            maxday = merge[merge['time'] == str(time)]
            #check 2
            maxday.to_csv('C:/Users/q0hecjrk/Documents/_Projects/northbranch/data/livneh/daily/max/maxprec.1997.csv', index = False)
        dfs.append(maxday)
        if export:
            merge.to_csv(outputfilepaths[i], index=False)
            i = i + 1
    #dfs = addIDs(dfs)      
    return dfs

def __cleaner(df: geopandas.GeoDataFrame) -> geopandas.GeoDataFrame:
    df['lon'] = df['lon'].apply(lambda x: x - 360)
    dt: list = df['time'].apply(lambda x: str(x).split("-", 4))
    df['year'] = dt.apply(lambda y: int(y[0]))
    df['month'] = dt.apply(lambda m: int(m[1]))
    df['day'] = dt.apply(lambda d: int(d[2].split()[0])) 
    return df

def __points2grids(gdf: geopandas.GeoDataFrame, shift: float = 1/32, crs: str = '4326') -> geopandas.GeoDataFrame:
    '''
    Converts livneh centroid 'lat' 'lon' coordinates to polygon grids.
        
    Inputs:
    (1) a geopandas GeoDataFrame containing 'lat' and 'lon' series
    (2) a float resenting the shortest distance between the centroid and the grid edge
    (3) the coordinate system identified to be used, '4326' by default
    Output:
    (1) a geopandas GeoDataFrame containing the grid cells polygon geometry

    Notes:
    The 'shift' parameter is used to identify the north, east, south and west most extents of the polygon, as so...
        ne---nw
        |     |
        se---sw
    '''
    grids_geometry: list = []
    for i, row in gdf.iterrows():
        w: float = float(row['lon']) - shift
        e: float = float(row['lon']) + shift
        n: float = float(row['lat']) + shift
        s: float = float(row['lat']) - shift
        lons, lats = [w, e, e, w], [n, n, s, s]
        grids_geometry.append(Polygon(zip(lons, lats)))
    gdf['geometry'] = grids_geometry
    return gdf.to_crs(epsg = crs)

def addIDs(dfs: list, crs: str = '4326') -> list:
    coordinates = []
    ids = dfs[0]
    for i, row in ids.iterrows():
        coordinates.append((row['lat'], row['lon']))
    ids['coordinates'] = coordinates
    ids.drop_duplicates(['lat', 'lon'], inplace=True)
    ids.sort_values(['lat', 'lon'], ascending=[False, True], axis = 0, inplace=True)
    ids.reset_index(drop = True, inplace=True)
    ids['id'] = ids.index
    ids = ids.to_crs(epsg = crs)
    ids['total_area_m2'] = ids['geometry'].area
    ids = ids.filter(items=['coordinates', 'total_area_m2' 'id'])
    merged = []
    for df in dfs:
        merged.append(pd.merge(df, ids, on='coordinates'))      
    return merged

def livnehIDsAndAreas(df: geopandas.GeoDataFrame, crs: str = '4326') -> dict:
    # clipped data
    df.drop_duplicates(['id'], inplace=True)
    df.sort_values(['id'], axis = 0, inplace=True)
    df = df.to_crs(epsg = crs)
    df['area_m2'] = df['geometry'].area
    df = df.filter(items=['coordinates', 'lat', 'lon', 'id', 'area_m2'])
    df = __points2grids(df, crs=crs)
    df = df.to_crs(epsg = crs)
    df['total_area_m2'] = df['geometry'].area
    return df





            
