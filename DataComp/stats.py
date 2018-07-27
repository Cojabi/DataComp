# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import warnings

# stats
from scipy.stats import mannwhitneyu
from statsmodels.sandbox.stats.multicomp import multipletests
from scipy.stats import chisquare
from collections import Counter

def test_single_cat(dfs, col_name, printer=False):
    """
    DEPRECATED
    Uses a chi square test to check whether the distribution of categorical features accross the datasets differ
    significantly.
    :param dfs: list containing the two dataframes
    :param col_name: Name of the column which should be tested, compared
    :param printer: Flag parameter which results in printing the bumer of appearances of the discrete values
    :return:
    """

    def _categorical_table(series):
        """
        Returns the counts of occurences for the categories. Is used to build the observation table for a chi square test.
        :param series:
        :return:
        """
        c = Counter(series)
        # get rid of NaNs
        c = {key: c[key] for key in c if not pd.isnull(key)}
        return pd.Series(c)

    test_data = []

    # arrange data
    for df in dfs:
        series = df[col_name]
        test_data.append(_categorical_table(series))

    # print data for visual inspection
    if printer:
        print(test_data)

    # compute chi-square
    return chisquare(*test_data)

def test_num_feats(zipper, feats=None):
    """Perform a hypothesis test to check if the distributions vary signifcantly from each other"""

    def _test_if_all_vals_equal(vals1, vals2):
        """
        Checks if the union of two iterables is 1 and returns True if that is the case. Is used in test_num_feats to
        check if two lists of values have only the same value and nothing else.
        :param vals1:
        :param vals2:
        :return:
        """
        # build union between values
        uni = set.union(set(vals1), set(vals2))
        # if only one value is in the union, they are equal
        if len(uni) == 1:
            return True
        else:
            return False

    p_values = dict()

    if feats is None:
        feats = zipper.keys()

    for feat in feats:  # run through all variables

        # initiate dict in dict for d1 vs d2, d2 vs d3 etc. per feature
        p_values[feat] = dict()

        for i in range(len(zipper[feat]) - 1):  # select dataset1
            for j in range(i + 1, len(zipper[feat])):  # select dataset2

                #handle the case that all values are equal across datasets
                if _test_if_all_vals_equal(zipper[feat][i], zipper[feat][j]):
                    # delete already created dict for i, j in p_values and continue with next feature
                    warnings.warn("Values of \"{}\" are the identical across the two datasets. It will be skipped.".format(feat), UserWarning)
                    del p_values[feat]
                    continue

                # only calculate score if there are values in each dataset
                if zipper[feat][i] and zipper[feat][j]:
                    # calculate u statistic and return p-value
                    z = mannwhitneyu(zipper[feat][i], zipper[feat][j], alternative="two-sided")
                    p_values[feat][i+1, j+1] = z.pvalue

                # if one or both sets are empty
                else:
                    #del p_values[feat]
                    p_values[feat][i+1, j+1] = np.nan

    return p_values


def test_cat_feats(zipper, feats=None):
    """Perform a hypothesis test to check if the distributions vary signifcantly from each other"""

    def _categorical_table(data):
        """
        Returns the counts of occurences for the categories. Is used to build the observation table
        for a chi square test.
        :param series:
        :return:
        """

        c = Counter(data)
        # get rid of NaNs
        c = {key: c[key] for key in c if not pd.isnull(key)}
        return pd.Series(c)

    p_values = dict()

    if feats is None:
        feats = zipper.keys()

    for feat in feats:  # run through all variables
        # initiate dict in dict for d1 vs d2, d2 vs d3 etc. per feature
        p_values[feat] = dict()

        for i in range(len(zipper[feat]) - 1):  # select dataset1
            for j in range(i + 1, len(zipper[feat])):  # select dataset2
                # count occurences of categorical features like in a confusion matrix for Chi2 tests
                test_data = [_categorical_table(zipper[feat][i]), _categorical_table(zipper[feat][j])]
                # calculate u statistic and return p-value
                z = chisquare(*test_data)
                p_values[feat][i+1, j+1] = z.pvalue

    return p_values


def p_correction(p_values):
    """Apply p value correction for multiple testing"""

    def _transform_p_dict(p_value_dict):
        """
        Transforms a dictionary of dicts into a dataframe representing the dicts as rows (like tuples).
        The resulting DataFrame can then be used to sort the p_values such that
        :param p_value_dict: dictionary of dictionaries storing the p_values
        :return: dataframe where the keys are added to the p_values as columns
        """
        temp_dict = dict()

        for feat in p_value_dict:
            temp_dict[feat] = list(p_value_dict[feat].items())
        # building a matrix ordered. extract and sort data from a tuple (x[0], (i[0], i[1]))
        list_repr = [[i[1], i[0], x[0]] for x in list(temp_dict.items()) for i in x[1]]

        return pd.DataFrame(list_repr)

    p_trans = _transform_p_dict(p_values)
    p = p_trans[0].sort_values()

    # correct p-values
    result = multipletests(p.values)

    # store test results
    p = pd.DataFrame(p)
    p.rename(columns={0: "pv"}, inplace=True)
    p["cor_pv"] = result[1]
    p["signf"] = result[0]

    p = pd.concat([p, p_trans], axis=1, join="inner")
    # create multi index
    p.index = p[[2, 1]]
    p.index = pd.MultiIndex.from_tuples(p.index)
    p.drop([0, 1, 2], axis=1, inplace=True)

    return p.sort_index()


def analyze_feature_ranges(zipper, cat_feats, num_feats, include=None, exclude=None, verbose=True):
    """
    This function can be used to compare all features easily. It works as a wrapper for the categorical and numerical
    feature comparison functions.
    :param zipper:
    :param cat_feats: List of categorical features
    :param num_feats: List of numerical features
    :param include: List of features that should be included into the comparison
    :param exclude: List of features that should be excluded from the comparison
    :return: dataframe showing the results of the comparison
    """

    # create coppy of zipper to avoid changing original zipper
    zipl = zipper.copy()
    # create dictionary that will store the results for feature comparison
    p_values = dict()

    # delete label if given
    if exclude:
        for feat in exclude:
            del zipl[feat]
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
              str(results["signf"].sum())+"/"+str(len(results["signf"])))

    return results.sort_values("signf")