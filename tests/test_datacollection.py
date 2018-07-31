# -*- coding: utf-8 -*-

import os
import unittest

from datacomp.datacollection import DataCollection
from datacomp.data_functions import get_data


dir_path = os.path.dirname(os.path.realpath(__file__))

test_data_path = os.path.join(dir_path, 'test_data.csv')

df_names = ["Test1", "Test2"]

group_col = "Site"


class TestDataCollection(unittest.TestCase):
    """ """

    def setUp(self):
        self.datacol = get_data(test_data_path, df_names, group_col)

    def test_get_data(self):
        self.assertEqual(self.datacol[0].shape[0], 29)
