# -*- coding: utf-8 -*-

import os
import unittest

from datacomp.datacollection import get_data

dir_path = os.path.dirname(os.path.realpath(__file__))

test_data_path = os.path.join(dir_path, 'test_data.csv')

df_names = ["Test1", "Test2"]

group_col = "Site"

feat_subset = ["ADAS", "Age", "EmptyFull", "EmptyS1"]

features = [{"ADAS", "MMSE", "Site", "Age", "Diagnosis",
             "Class", "EmptyS1", "EmptyFull", "Gender"},
            {"ADAS", "MMSE", "Site", "Diagnosis", "Age",
             "Class", "EmptyS1", "EmptyFull", "Gender"}]

exclude_feats = ["ADAS", "MMSE", "Site", "Diagnosis",
                 "Age", "Class", "EmptyS1"]


class TestDataCollection(unittest.TestCase):
    """ """

    def setUp(self):
        self.datacol = get_data(test_data_path, df_names, group_col)

    def test_get_data(self):
        """Test get_data function."""
        self.assertEqual(self.datacol[0].shape[0], 5)
        self.assertEqual(self.datacol[1].shape[0], 6)

    def test_create_zipper(self):
        """Test the create_zipper function."""
        zipper = self.datacol.create_zipper(feat_subset)

        self.assertEqual(zipper["ADAS"],
                         ([10, 11, 10, 10], [10, 10, 10, 9]))

        self.assertEqual(zipper["EmptyS1"],
                         ([], [11.0, 13.0, 15.0, 17.0, 19.0, 21.0]))

        self.assertEqual(set(zipper.keys()), set(feat_subset))

    def test_get_feature_sets(self):
        feature_sets = self.datacol.get_feature_sets()
        self.assertEqual(feature_sets, features)

    def test_get_common_features(self):
        common_feats = self.datacol.get_common_features(exclude=exclude_feats)
        self.assertEqual(set(common_feats), {"EmptyFull", "Gender"})

    def test_reduce_dfs_to_feat_subset(self):
        reduced_datacol = self.datacol.reduce_dfs_to_feat_subset(feat_subset)
        self.assertEqual(list(reduced_datacol[0]), feat_subset)

        # test that changes on reduced df do not affect original df
        changed_feat = feat_subset[0]
        # before change, drop NaNs because in pandas/numpy NaN != NaN
        self.assertTrue(
            (reduced_datacol[0][changed_feat].dropna() == self.datacol[0][changed_feat].dropna()).all()
        )
        # change
        reduced_datacol[0][changed_feat] = "changed_value"
        # after change
        self.assertFalse(
            (reduced_datacol[0][changed_feat] == self.datacol[0][changed_feat]).all()
        )

    def test_reduce_dfs_to_value(self):
        reduced_datacol = self.datacol.reduce_dfs_to_value("Gender", 1)
        self.assertEqual(len(reduced_datacol[0]), 3)
        self.assertEqual(list(reduced_datacol[1].index), ["x23", "x25", "x27"])

        # test that changes on reduced df do not affect original df
        changed_feat = feat_subset[0]
        # before change, drop NaNs because in pandas/numpy NaN != NaN
        self.assertTrue(
            (reduced_datacol[1][changed_feat].dropna() == self.datacol[1].loc[
                ["x23", "x25", "x27"], changed_feat].dropna()).all()
        )
        # change
        reduced_datacol[1][changed_feat] = "changed_value"
        # after change
        self.assertFalse(
            (reduced_datacol[1][changed_feat] == self.datacol[1].loc[["x23", "x25", "x27"], changed_feat]).all()
        )
