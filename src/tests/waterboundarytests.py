import unittest
import dataimports.waterboundary as wbd

class waterboundarytests(unittest.TestCase):
    
    def test_importwaterboundary(self, filepath: str = 'C:/Users/q0hecjrk/Documents/_Data/Geospatial/Waterboundary/WBD_02_HU2_Shape/Shape/WBDHU8.shp', column: str = 'HUC8', code: str = '02070002'):
        gdf = wbd.importwaterboundary(filepath, column, code)
        actual: int = len(gdf.index) if gdf is not None else -1
        expected: int = 1
        self.assertEqual(actual, expected)
    
    #def test_boundaryboxfromshape(self, shape, buffer: float = 1/32):

