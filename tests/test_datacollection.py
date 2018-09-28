# -*- coding: utf-8 -*-

import os
import unittest

from datacomp.datacollection import get_data

dir_path = os.path.dirname(os.path.realpath(__file__))

test_data_path = os.path.join(dir_path, 'test_data.csv')

df_names = ["Test1", "Test2"]

group_col = "site"

cat_feats = ["cat1", "cat2", "cat3"]

feat_subset = ["feat1", "feat2", "cat1"]

features = [{"feat1", "feat2", "site", "feat3", "cat1",
             "cat2", "cat3"},
            {"feat1", "feat2", "site", "cat1", "feat3",
             "cat2", "empty1", "cat3"}]

exclude_feats = ["feat1", "feat2", "site", "cat1",
                 "feat3", "cat2"]


class TestDataCollection(unittest.TestCase):
    """ """

    def setUp(self):
        self.datacol = get_data(test_data_path, df_names, cat_feats, group_col)

    def test_get_data(self):
        """Test get_data function."""
        self.assertEqual(self.datacol[0].shape, (5, 7))
        self.assertEqual(self.datacol[1].shape, (6, 8))

    def test_create_zipper(self):
        """Test the create_zipper function."""
        zipper = self.datacol.create_zipper(feat_subset)

        self.assertEqual(zipper["feat2"],
                         ([10, 11, 10, 10], [10, 10, 10, 9]))

        self.assertEqual(zipper["cat1"],
                         (["C", "M", "M", "C"], ["A", "A", "C", "A", "A"]))

        self.assertEqual(set(zipper.keys()), set(feat_subset))

    def test_get_feature_sets(self):
        feature_sets = self.datacol.get_feature_sets()
        self.assertEqual(feature_sets, features)

    def test_get_common_features(self):
        common_feats = self.datacol.get_common_features(exclude=exclude_feats)
        self.assertEqual(set(common_feats), {"cat3"})

    def test_reduce_dfs_to_feat_subset(self):
        reduced_datacol = self.datacol.reduce_dfs_to_feat_subset(feat_subset)
        self.assertEqual(set.union(*reduced_datacol.get_feature_sets()), set(feat_subset))

        # test that changes on reduced df do not affect original df
        changed_feat = feat_subset[0]
        # before change, drop NaNs because in pandas/numpy NaN != NaN
        self.assertFalse(
            (reduced_datacol[0][changed_feat].dropna(inplace=True) == self.datacol[0][changed_feat]).all()
        )
        # change
        reduced_datacol[0][changed_feat] = "changed_value"
        # after change
        self.assertFalse(
            (reduced_datacol[0][changed_feat] == self.datacol[0][changed_feat]).all()
        )

    def test_reduce_dfs_to_value(self):
        reduced_datacol = self.datacol.reduce_dfs_to_value("cat2", 1)
        self.assertEqual(reduced_datacol[0].shape, (3, 7))
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
