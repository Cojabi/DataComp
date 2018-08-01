# -*- coding: utf-8 -*-

import pandas as pd
import matplotlib_venn as mv
import matplotlib.pylab as plt

from pymatch.Matcher import Matcher
from collections import UserList
from .stats import test_cat_feats, test_num_feats, p_correction
from .prop_matching import construct_formula


class DataCollection(UserList):
    """
    A class representing the Collection of datasets, that shall be compared. Datasets are stored list like in self.data
    and the DataCollection class can be used as if it is a list. Provides functions to reduce the dataframes to specific
    subsets of features or to compare the feature ranges of the two dataframes to each other.
    """

    def __init__(self, df_list, df_names):
        """
        Initialize DataCollection object.

        :param df_list: List containing pandas.DataFrames
        :param df_names: List of strings containing the names of the datasets in df_list
        """
        super().__init__(df_list)
        self.df_names = df_names

    def create_zipper(self, feats=None):
        """
        Create zipper containing the values of the same features per df in one list.
        (df1_feat1, df2_feat1, df3_feat1), (df1_feat2, df2_feat2, df3_feat2)

        :param feats:
        :return:
        """
        if feats is None:
            feats = list(self[0])

        df_feats = []

        for df in self:
            df_feats.append([list(df[feat].dropna()) for feat in feats])

        zip_values = zip(*df_feats)
        zipper = dict(zip(feats, zip_values))
        return zipper

    def reduce_dfs_to_value(self, col, val):
        """
        Keep only the rows in the dataframes where a specific column holds a specific value.

        :param col: Specific column in which the value should be found.
        :param val: Specific value that should be present in the column.
        :return:
        """
        # create list with reduced dataframes
        reduced_dfs = [df[df[col] == val] for df in self]
        return DataCollection(reduced_dfs, self.df_names)

    def reduce_dfs_to_feat_subset(self, feat_subset):
        """
        Manipulate the DataCollection that the dataframes inside only contain a subset of the original features.

        :return: List of dataframes where the features are identical
        """
        reduced_dfs = [df[feat_subset] for df in self]
        return DataCollection(reduced_dfs, self.df_names)

    def get_feature_sets(self):
        """
        Creates a list of sets, where in each set the a ll the feature names of one dataframe are stored respectively.

        :return: List of sets. Each set contains the feature names of one of the dataframes
        """
        # Create list containing features per dataframe as sets
        return [set(df) for df in self]

    def get_common_features(self, exclude=None):
        """
        Creates a set of the common features shared between dataframes.

        :param exclude: List of features which shall be exluded from the common features.
        :return: set of common features across the dataframes
        """
        feats = self.get_feature_sets()

        common_feats = set.intersection(*feats)

        if exclude:
            for feat in exclude:
                common_feats.remove(feat)

        return list(common_feats)

    def get_feature_differences(self):
        """
        Assess differences in features across the dataframes. A dictionary will be created where the dataframe
        combination which is compared serves as key and the value is a set with the differences across them.

        :return: Dictionary where dataframe combination is the key and the value is a corresponding set of
        non overlapping features.
        """
        feats = self.get_feature_sets()

        # create a dictionary storing the features which are distinct
        diff_dict = dict()

        # compare each dataset against each and collect differences
        for i in range(len(feats)):
            for j in range(i + 1, len(feats)):
                # take union from differences
                diff_dict[i, j] = feats[i].difference(feats[j]).union(feats[j].difference(feats[i]))

        return diff_dict

    def create_value_set(self, col):
        """
        Creates a set of the combined dataframe values present in a specific column.

        :param col:
        :return:
        """

        value_set = set()

        for df in self:
            value_set.update(df[col])
        return value_set

    ## Stats

    def analyze_feature_ranges(self, cat_feats, num_feats, include=None, exclude=None, verbose=True):
        """
        This function can be used to compare all features easily. It works as a wrapper for the categorical and numerical
        feature comparison functions.

        :param cat_feats: List of the categorical features found in the dataframes.
        :param num_feats: List of numerical features found in the dataframes.
        :param include: List of features that should be considered for the comparison solely.
        :param exclude: List of features that should be excluded from the comparison.
        :return: pandas.Dataframe showing the p-values and corrected p-values of the comparison
        """

        # create zipper
        zipper = self.create_zipper()
        # create dictionary that will store the results for feature comparison
        p_values = dict()

        # delete label if given
        if exclude:
            for feat in exclude:
                del zipper[feat]
            # update feature lists
            cat_feats = set(cat_feats).difference(exclude)
            num_feats = set(num_feats).difference(exclude)

        if include:
            cat_feats = set(cat_feats).intersection(include)
            num_feats = set(num_feats).intersection(include)

        # test features:
        p_values.update(test_cat_feats(zipper, cat_feats))
        p_values.update(test_num_feats(zipper, num_feats))

        # test numerical features
        results = p_correction(p_values)

        if verbose:
            print("Fraction of significantly deviating features:",
                  str(results["signf"].sum()) + "/" + str(len(results["signf"])))

        return results.sort_values("signf")

    ## Propensity score matching

    def create_df_for_matching(self, label_name, prop_feats, save_path=None, labels=None):
        """
        Will create a combined dataframe in which labels are assigned depending on the dataset membership.
        The resulting dataframe will be saved under save_path and can be used for propensity_score_matching.

        :param label_name:
        :param save_path:
        :param labels:
        :return:
        """
        if labels is None:
            labels = [1, 0]

        # reduce dfs to prop_feats only.
        prop_reduced_datcol = self.reduce_dfs_to_feat_subset(prop_feats)

        # add labels to dataframes
        for i in range(len(labels)):
            prop_reduced_datcol[i][label_name] = labels[i]
            prop_reduced_datcol[i].dropna(inplace=True)

        # combine datasets and save them under save_path if save_path is given
        if save_path:
            prop_df = pd.concat(prop_reduced_datcol)
            prop_df.to_csv(save_path)

        return prop_reduced_datcol

    def qc_prop_matching(dfs, rel_cols, label):
        """
        Evaluates the need for a propensity score matching and can be used to quality control a propensity score matched
        population. Will train classifiers and create a plot.
        :param dfs: List of dataframes
        :param rel_cols: relevant columns
        :param label: Label or class which should be regressed. (cohort1/cohort2, case/control, treatment/untreated etc.)
        """

        cols = rel_cols[::]

        # create reduced copies of the dataframes for propensity score quality control
        qc_dfs = []
        for df in dfs:
            qc_dfs.append(df[cols])

        # construct formula
        cols.remove(label)
        formula = construct_formula(label, cols)

        # create Matcher
        m = Matcher(*qc_dfs, yvar=label, formula=formula)
        # train classifier to asses predictability
        m.fit_scores(balance=True, nmodels=10)
        # calculate and visualize propensity scores
        m.predict_scores()
        m.plot_scores()

    ## Visualization

    def feat_venn_diagram(self):
        """
        Plots a venn diagram illustrating the overlap in features between the datasets.

        :return:
        """
        feat_set = self.get_feature_sets()

        # Plotting when two datasets are compared
        if len(self) == 2:
            # set variables needed to assign new color scheme
            colors = ["blue", "green"]
            ids = ["A", "B"]
            # create circles
            v = mv.venn2(feat_set, set_labels=self.df_names)
            # create lines around circles
            circles = mv.venn2_circles(feat_set)

        # Plotting when three datasets are compared
        elif len(self) == 3:
            # set variables needed to assign new color scheme
            colors = ["blue", "green", "purple"]
            ids = ["A", "B", "001"]
            # create cirlces
            v = mv.venn3_unweighted(feat_set, set_labels=self.df_names)
            # create lines around circles
            circles = mv.venn3_circles(subsets=(1, 1, 1, 1, 1, 1, 1))

        else:
            raise ValueError("Too many datasets in DataCollection. Venn diagram only supported up to 3 datasets.")

        # set colors for the circles in venn diagram
        for df_name, color in zip(ids, colors):
            v.get_patch_by_id(df_name).set_color(color)

        # reduce line width around circles
        for c in circles:
            c.set_lw(1.0)

        plt.title("Feature Overlap")
