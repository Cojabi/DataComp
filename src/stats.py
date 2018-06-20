# -*- coding: utf-8 -*-

import pandas as pd

# stats
from scipy.stats import mannwhitneyu
from statsmodels.sandbox.stats.multicomp import multipletests
from scipy.stats import chisquare
from collections import Counter

# propensity score matching
from pymatch.Matcher import Matcher

def test_categorical(dfs, col_name, printer=False):
    """
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


def test_num_dist(zipper, feats=None):
    """Perform a hypothesis test to check if the distributions vary signifcantly from each other"""
    p_values = dict()

    if feats is None:
        feats = zipper.keys()

    for feat in feats:  # run through all variables
        for i in range(len(zipper[feat]) - 1):  # select dataset1
            for j in range(i + 1, len(zipper[feat])):  # select dataset2
                # calculate u statistic and return p-value
                z = mannwhitneyu(zipper[feat][i], zipper[feat][j], alternative="two-sided")
                p_values[feat] = z.pvalue

    return p_values


def p_bonfer_cor(p_values):
    """Apply p value correction for multiple testing"""

    p = pd.Series(p_values).sort_values()

    # correct p-values
    result = multipletests(p.values)

    # store test results
    p = pd.DataFrame(p)
    p.rename(columns={0: "pv"}, inplace=True)
    p["cor_pv"] = result[1]
    p["signf"] = result[0]

    return p