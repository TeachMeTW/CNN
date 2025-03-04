import unittest
from unittest.mock import patch, MagicMock
import pandas as pd
from data import db


# FILE: tests/db_tests/test_testCsvUploader.py


class TestDB(unittest.TestCase):

    def setUp(self):
        self.data = ["a", "b", "c"]
    

    def test_upload():
        collection =  get_collection()

    def tearDown(self):
        
   
if __name__ == '__main__':
    unittest.main()