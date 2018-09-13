# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import warnings

from scipy.stats import mannwhitneyu, ttest_ind, chisquare
from statsmodels.sandbox.stats.multicomp import multipletests
from statsmodels.multivariate.manova import MANOVA
from collections import Counter

from .utils import construct_formula


def test_num_feats(zipper, feat_subset=None, method=None):
    """
    Performs a hypothesis test to check if the value distributions of numerical features deviate signifcantly between
    the datasets. Currently t-test as a parametric and U-test as a non-parametric test are supported.

    :param zipper: Dictionary storing the feature values of the datasets in a list. Feature name is used as the key.
    :param feat_subset: A list containing feature names. If given, analysis will only be performed for the contained
    features. If not given all features will be considered.
    :param method: Specify which statistical test should be used. "u" for Mann-Whitney-U-test and "t" for t-test.
    :return: dictionary storing the p_values of the analysis. Feature names are used as keys.
    """

    def _test_if_all_vals_equal(vals1, vals2):
        """Checks if the union of two iterables is 1 and returns True if that is the case. Is used in test_num_feats to
        check if two lists of values have only the same value and nothing else.

        :param vals1:
        :param vals2:
        :return:
        """
        # build union between values
        uni = set.union(set(vals1), set(vals2))
        # if only one value is in the union, they are equal
        return len(uni) == 1

    # if no method is specified used Mann-Whitney-U-test as standard
    if method is None:
        method = "u"

    # initialize dictionary which stores the p_values
    p_values = dict()

    if feat_subset is None:
        feat_subset = zipper.keys()

    for feat in feat_subset:  # run through all variables

        # initiate dict in dict for d1 vs d2, d2 vs d3 etc. per feature
        p_values[feat] = dict()

        for i in range(len(zipper[feat]) - 1):  # select dataset1
            for j in range(i + 1, len(zipper[feat])):  # select dataset2

                # handle the case that all values of current feature are equal across current datasets
                if _test_if_all_vals_equal(zipper[feat][i], zipper[feat][j]):
                    warnings.warn(
                        "Values of \"{}\" are the identical across the two datasets. It will be skipped.".format(feat),
                        UserWarning)
                    # delete already created dict for i, j in p_values and continue with next feature
                    del p_values[feat]
                    continue

                # only calculate score if there are values in each dataset
                if zipper[feat][i] and zipper[feat][j]:
                    # calculate u-test and return p-value
                    if method == "u":
                        stat_test_result = mannwhitneyu(zipper[feat][i], zipper[feat][j], alternative="two-sided")
                    # calculate t-test and return p-value
                    elif method == "t":
                        stat_test_result = ttest_ind(zipper[feat][i], zipper[feat][j])

                    p_values[feat][i + 1, j + 1] = stat_test_result.pvalue

                # if one or both sets are empty
                else:
                    p_values[feat][i + 1, j + 1] = np.nan

    return p_values


def test_cat_feats(zipper, feat_subset=None, method="chi"):
    """
    Performs hypothesis testing to identify significantly deviating categorical features. A chi-squared test is used.

    :param zipper: Dictionary storing the feature values of the datasets in a list. Feature name is used as the key.
    :param feat_subset: A list containing feature names. If given, analysis will only be performed for the contained
    features. If not given all features will be considered.
    :return:
    """

    def _categorical_table(data):
        """
        Returns the counts of occurences for the categories. Is used to build the observation table
        for a chi square test.

        :param data:
        :return:
        """
        # count occurences
        c = Counter(data)
        # get rid of NaNs
        c = {key: c[key] for key in c if not pd.isnull(key)}
        return pd.Series(c)

    def _non_present_values_to_zero(test_data):
        """
        Fills keys in the test data of one dataframe if it is not present but present in one of the other datasets.

        :param test_data:
        :return:
        """
        for dataset1 in test_data:
            for dataset2 in test_data:

                for key in dataset1.keys():

                    if key not in dataset2:
                        dataset2[key] = 0
        return test_data

    p_values = dict()

    # consider all features is no feature subset was specified
    if feat_subset is None:
        feat_subset = zipper.keys()

    for feat in feat_subset:
        # initiate dict in dict for d1 vs d2, d2 vs d3 etc. per feature
        p_values[feat] = dict()

        for i in range(len(zipper[feat]) - 1):  # select dataset1
            for j in range(i + 1, len(zipper[feat])):  # select dataset2
                # count occurences of categorical features like in a confusion matrix for Chi2 tests
                test_data = [_categorical_table(zipper[feat][i]), _categorical_table(zipper[feat][j])]

                # fill missing keys in test data:
                test_data = _non_present_values_to_zero(test_data)

                # sort testing data by index(categories) to align the counts for the categories
                test_data = [data.sort_index() for data in test_data]

                if method == "chi":
                    # skip feature if number of events per group is smaller than 5
                    if (test_data[0] < 5).any() or (test_data[1] < 5).any():
                        warnings.warn(UserWarning, feat+"is skipped due to few observations in one or more groups.")
                        # calculate u statistic and return p-value
                        test_result = chisquare(*test_data)

                elif method == "fisher":
                    raise NotImplementedError

                p_values[feat][i + 1, j + 1] = test_result.pvalue

    return p_values


def p_correction(p_values):
    """
    Corrects p_values for multiple testing.

    :param p_values: Dictionary storing p_values with corresponding feature names as keys.
    :return: DataFrame which shows the results of the analysis; p-value, corrected p-value and boolean indicating
    significance.
    """

    def _transform_p_dict(p_value_dict):
        """
        Utility function that transforms a dictionary of dicts into a dataframe representing the dicts as rows
        (like tuples). Is needed to keep track of the feature names and corresponding values.
        The underlying datastructures are confusing.

        :param p_value_dict: dictionary of dictionaries storing the p_values
        :return: dataframe where the keys are added to the p_values as columns
        """
        # Turn dictionary of dictionaries into a collection of the key-value pairs represented as nested tuples
        item_dict = dict()

        for feat in p_value_dict:
            item_dict[feat] = list(p_value_dict[feat].items())

        # building a matrix (nested lists) by extracting and sorting data from nested tuples
        # (items[0], (nested_items[0], nested_items[1]))
        df_matrix = []

        for items in item_dict.items():
            for nested_items in items[1]:
                df_matrix.append([nested_items[1], nested_items[0], items[0]])
        return pd.DataFrame(df_matrix)

    def _create_result_table(result, p_val_col, p_trans):
        """
        Builds the dataframe showing the results

        :param result:
        :param p_val_col:
        :param p_trans:
        :return:
        """
        # store test results
        result_table = pd.DataFrame(p_val_col)

        # rename columns and set values in result dataframe
        result_table.rename(columns={0: "pv"}, inplace=True)
        result_table["cor_pv"] = result[1]
        result_table["signf"] = result[0]
        # combine p_value information with dataset and feature information stored in p_trans
        result_table = pd.concat([result_table, p_trans], axis=1, join="outer")

        # create multi index brom feature name result_table[2] and datasets result_table[1]
        result_table.index = result_table[[2, 1]]
        result_table.index = pd.MultiIndex.from_tuples(result_table.index)
        result_table.drop([0, 1, 2], axis=1, inplace=True)
        return result_table

    p_trans = _transform_p_dict(p_values)

    # get and drop features which are NaN to skip them in multitest correction
    nan_features = p_trans[pd.isnull(p_trans[0])]
    p_trans = p_trans.dropna(axis=0, subset=[0])

    # extract p_value column to pass into multiple testing correction
    p_val_col = p_trans[0].sort_values()

    # add NaN features back to p_trans to include them into result table later on
    p_trans = pd.concat([p_trans, nan_features])

    # raise Error if no p_values where calculated that can be passed into multipletest correction
    if p_val_col.values.size == 0:
        raise ValueError("Empty list of p_values have been submitted into multiple test correction.")

    # correct p-values
    result = multipletests(p_val_col.values)
    # build a table storing the p_values and corrected p_values for all features
    result_table = _create_result_table(result, p_val_col, p_trans)

    return result_table.sort_index()


def manova(datacol, label, variable_cols):
    """
    Performs a MANOVA to assess for example batch effects: Check if a significant proportion of the data variance is
    explained by the dataset membership.
    For more documentation see: https://www.statsmodels.org/stable/generated/statsmodels.multivariate.manova.MANOVA.html

    :param datacol: A DataCollection object storing the datasets
    :param label: The name of the label column that will be created and represents the factor in the MANOVA
    :param variable_cols: A subset of features which shall be used as variables in the MANOVA
    :return: A multiindex dataframe listing important outcome statistics of the MANOVA.
    """

    # create combined dataframe with dataframe membership as label
    df_manova = datacol.combine_dfs(label, variable_cols)

    # construct formula
    formula = construct_formula(label, variable_cols, label_side="r")

    return MANOVA.from_formula(formula, df_manova).mv_test().summary()
