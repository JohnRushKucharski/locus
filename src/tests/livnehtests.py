import unittest
import dataimports.livneh as livneh

class livnehtests(unittest.TestCase):

    def test_livnehfilepaths(self, dir: str = 'C:/testexample/livneh/daily/', yr: list = ['1997'], var: list = ['prec']):
        actual: list = livneh.inputfilepaths(dir, var, yr)
        expected: list = [ 'C:/testexample/livneh/daily/prec.1997.nc']
        self.assertListEqual(actual, expected)

    def test_outputfilepaths(self, dir: str = 'C:/testexample/livneh/daily/', files: list = ['C:/testexample/livneh/daily/prec.1997.nc'], extension: str = '.csv'):
        actual: list = livneh.outputfilepaths(dir, files, extension)
        expected: list = ['C:/testexample/livneh/daily/prec.1997.csv']
        self.assertListEqual(actual, expected)

    def test_importfile(self, filepath: str = 'C:/Users/q0hecjrk/Documents/_Data/Livneh/Daily/prec.1915.nc', box: list = [-79, 39, -78, 40]):
        df = livneh.importfile(filepath, box)
        actual: int = df.count + 1
        print(actual)
        print(actual / 365.0)
        expected: int = 365
        self.assertIsNotNone(df)
    
    