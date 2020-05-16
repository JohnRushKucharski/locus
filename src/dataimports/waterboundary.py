import geopandas


def importwaterboundariesfile(filepath: str) -> geopandas.GeoDataFrame:
    '''
    Imports a waterboundary shape file as a geopandas GeoDataFrame

    Inputs: a string file path
    Output: a geopandas GeoDataFrame
    '''
    return geopandas.read_file(filepath).to_crs("EPSG:4326")

def importwaterboundary(filepath: str, column: str, value: str) -> geopandas.GeoDataFrame:
    '''
    Imports a single waterboundary as a geopandas GeoDataFrame from a waterboundary shape file

    Inputs: 
    (1) filepath: a string file path
    (2) column: the column name in the shape file attribute table to use for the watershed selection
    (3) value: a string value to select on in the column
    Outputs: a geopandas GeoDataFrame
    '''
    geodf: geopandas.GeoDataFrame = importwaterboundariesfile(filepath)
    return geodf[geodf[column] == value]

def boundaryboxfromshape(geodf: geopandas.GeoDataFrame, buffer: bool = True, buffersize: float = 1/32) -> list:
    '''
    Returns a list of latitude longitude coordinates from a waterboundary shape stored as geopandad GeoDataFrame

    Inputs:
    (1) geodf: a geopandas GeoDataFrame
    (2) buffer: True if the box should include a buffer area, False by default
    (3) buffersize: a float buffer area, 0 by default
    Output: a boundary box with latitude and longitude coordinates in the form: [west, south, east, north]
    '''
    coordinates = list(geodf.iloc[0]['geometry'].bounds)
    if buffer:
        coordinates[0] = coordinates[0] - buffersize
        coordinates[1] = coordinates[1] - buffersize
        coordinates[2] = coordinates[2] + buffersize
        coordinates[3] = coordinates[3] + buffersize
    return coordinates

def importboundarybox(filepath: str, column: str, value: str, buffer: bool = False, buffersize: float = 0) -> list:
    '''
    Generates a list of bounding latitude and longitude coordinates for a specific watershed from a waterboundary shapefile.

    Import:
    (1) filepath: a string file path
    (2) column: a shape file attribute table column name for watershed selection
    (3) value: the value to select on from the attribute table colum
    (4) buffer: True if the bounding box should be buffered, False otherwise
    (5) buffersize:  a float buffer size 
    Output: a list of boundary coordiantes in the form: [west, south, east, north]
    '''
    return boundaryboxfromshape(importwaterboundary(filepath, column, value), buffer, buffersize)

