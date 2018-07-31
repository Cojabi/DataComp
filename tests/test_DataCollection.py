# -*- coding: utf-8 -*-

import os
import unittest

from datacomp.DataCollection import DataCollection
from datacomp.data_functions import get_data


dir_path = os.path.dirname(os.path.realpath(__file__))

text_xml_path = os.path.join(dir_path, 'test_data.csv')


class DatabaseMixin(unittest.TestCase):
    """ """

    def setUp(self):
        datacol = get_data