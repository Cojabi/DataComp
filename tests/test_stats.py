# -*- coding: utf-8 -*-

import os
import unittest
import warnings

from scipy.stats import mannwhitneyu, chisquare

from datacomp.datacollection import get_data
from datacomp.stats import test_num_feats as num_test, p_correction
from datacomp.stats import test_cat_feats as cat_test
from datacomp.utils import _categorical_table

dir_path = os.path.dirname(os.path.realpath(__file__))
test_data_path = os.path.join(dir_path, 'test_data.csv')

# Constants
df_names = ["Test1", "Test2"]
group_col = "site"
cat_feats = ["cat1", "cat2", "cat3"]
feat_subset = ["feat1", "feat2", "cat1"]
exclude_feats = ["site"]

class TestDataCollection(unittest.TestCase):
    """ """

    def setUp(self):
        self.datacol = get_data(test_data_path, df_names, cat_feats, group_col)
        self.zipper = self.datacol.create_zipper()


    def test_test_num_feats(self):
        p_vals = num_test(self.zipper, feat_subset=["feat1", "feat2"])

        # check if p-value for feat1 is correct
        feat1_p_val = mannwhitneyu(self.datacol[0]["feat1"].dropna(), self.datacol[1]["feat1"].dropna(),
                                   alternative="two-sided")
        self.assertEqual(p_vals["feat1"][(1, 2)], feat1_p_val.pvalue)


    def test_test_cat_feats(self):
        # test if warnings are risen due to few data points
        with warnings.catch_warnings(record=True) as w:
            # Cause all warnings to always be triggered.
            warnings.simplefilter("always")
            # Trigger a warning.
            p_vals = cat_test(self.zipper, feat_subset=self.datacol.categorical_feats)

            warns = sorted([str(m.message) for m in w])
            self.assertEqual(len(w), 4) # 4 because one warining will be a divide by zero assertion error
            self.assertTrue("cat1" in warns[0])

        # check categorical table creation
        test_data = _categorical_table(["A", "A", "C", "A", "A"])
        self.assertTrue((test_data == [4, 1]).all())

        # check categorical p-value
        cat1_p_val = chisquare([1, 3, 1],[2, 1, 3])
        self.assertEqual(p_vals["cat3"][(1, 2)], cat1_p_val.pvalue)

    def test_p_correction(self):
        pass
