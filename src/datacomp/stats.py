# -*- coding: utf-8 -*-

import warnings

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu, ttest_ind, chisquare, wilcoxon, fisher_exact
from statsmodels.multivariate.manova import MANOVA
from statsmodels.sandbox.stats.multicomp import multipletests

from .utils import construct_formula, _categorical_table, _non_present_values_to_zero, \
    _test_if_all_vals_equal, _create_result_table, _transform_p_dict


def test_num_feats(zipper, feat_subset=None, method=None):
    """
    Performs a hypothesis test to check if the value distributions of numerical features deviate signifcantly between
    the datasets. Currently t-test as a parametric and U-test as a non-parametric test are supported.

    :param zipper: Dictionary storing the feature values of the datasets in a list. Feature name is used as the key.
    :param feat_subset: A list containing feature names. If given, analysis will only be performed for the contained \
    features. If not given all features will be considered.
    :param method: Specify which statistical test should be used. "u" for Mann-Whitney-U-test, "t" for t-test and \
    "wilcoxon" for a Wilcoxon signed rank test.
    :return: dictionary storing the p_values of the analysis. Feature names are used as keys.
    """

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
                    elif method == "wilcoxon":
                        stat_test_result = wilcoxon(zipper[feat][i], zipper[feat][j])

                    p_values[feat][i + 1, j + 1] = stat_test_result.pvalue

                # if one or both sets are empty
                else:
                    p_values[feat][i + 1, j + 1] = np.nan

    return p_values


def test_cat_feats(zipper, feat_subset=None, method=None, print_data=False):
    """
    Performs hypothesis testing to identify significantly deviating categorical features. A chi-squared test is used.

    :param zipper: Dictionary storing the feature values of the datasets in a list. Feature name is used as the key.
    :param feat_subset: A list containing feature names. If given, analysis will only be performed for the contained \
    features. If not given all features will be considered.
    :return:
    """

    p_values = dict()

    # consider all features if no feature subset was specified
    if feat_subset is None:
        feat_subset = zipper.keys()

    # set default method to chi-square test
    if method is None:
        method = "chi"

    for feat in feat_subset:
        # initiate dict in dict for dataset1 vs dataset2, d1 vs d3 etc. per feature
        p_values[feat] = dict()

        for i in range(len(zipper[feat]) - 1):  # select dataset1
            for j in range(i + 1, len(zipper[feat])):  # select dataset2
                # count occurences of categorical features like in a confusion matrix for Chi2 tests
                test_data = [_categorical_table(zipper[feat][i]), _categorical_table(zipper[feat][j])]

                # fill missing keys in test data:
                test_data = _non_present_values_to_zero(test_data)

                # sort testing data by index(categories) to align the counts for the categories
                test_data = [data.sort_index() for data in test_data]

                # print testing data if specified
                if print_data:
                    print(feat)
                    print(pd.DataFrame(test_data))
                    print()

                if method == "chi":
                    # skip feature if number of events per group is smaller than 5
                    if (test_data[0] < 5).any() or (test_data[1] < 5).any():
                        warnings.warn(feat + " has under 5 observations in one or more groups.", UserWarning)
                    # calculate u statistic and return p-value
                    p_val = chisquare(*test_data).pvalue

                elif method == "fisher":
                    p_val = fisher_exact(test_data)[1]

                p_values[feat][i + 1, j + 1] = p_val

    return p_values


def p_correction(p_values, counts):
    """
    Corrects p_values for multiple testing.

    :param p_values: Dictionary storing p_values with corresponding feature names as keys.
    :param counts: DataFrame storing the number of observations per dataset, per feature.
    :return: DataFrame which shows the results of the analysis; p-value, corrected p-value and boolean indicating \
    significance.
    """

    p_trans = _transform_p_dict(p_values)

    # get and drop features which are NaN to skip them in multitest correction
    nan_features = p_trans[pd.isnull(p_trans[0])]
    p_trans = p_trans.dropna(axis=0, subset=[0])

    # extract p_value column to pass into multiple testing correction
    p_val_col = p_trans[0].sort_values()

    # add NaN features back to p_trans to include them into result table later on
    p_trans = pd.concat([p_trans, nan_features])

    # raise Error if no p_values where calculated that can be passed into multiple test correction
    if p_val_col.values.size == 0:
        # unpack the p_values which are stored in 2 layer nested dicts.
        nested_values = []
        for value in p_values.values():
            nested_values.append(*value.values())

        # if all p_values are nan, return an all nan result table
        if pd.isnull(nested_values).all():
            result_table = _create_result_table(None, p_val_col, p_trans, counts)
            return result_table.sort_index()

        raise ValueError("No p_values have been submitted into multiple test correction.")

    # correct p-values
    result = multipletests(p_val_col.values)

    # build a table storing the p_values and corrected p_values for all features
    result_table = _create_result_table(result, p_val_col, p_trans, counts)

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
