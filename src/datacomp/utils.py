# -*- coding: utf-8 -*-

from collections import Counter
from operator import itemgetter

import numpy as np
import pandas as pd


def get_sig_feats(sig_df):
    """
    Get's the feature names of significantly deviating features from a result table.

    :param sig_df: Dataframe storing the p_values and the corrected p_values like returned by stats.p_correction()
    :return:
    """
    # grab significant deviances
    bool_series = sig_df["signf"].map(bool)
    sig_entries = sig_df[bool_series.replace(np.nan, False)]
    index_labels = sig_entries.index.labels[0]
    return set(itemgetter(index_labels)(sig_entries.index.levels[0]))


def construct_formula(label, rel_cols, label_side="l"):
    """
    Constructs a generic formula string from column names and a label name.
    label ~ Col1+Col2+...+ColN

    :param label: Label or class which should be regressed for. (case/control, treatment/untreated etc.)
    :param rel_cols: Relevant columns for the formula
    :return: formula string
    """
    cols = rel_cols[::]

    # exclude label from rel_cols if contained
    if label in rel_cols:
        cols.remove(label)

    if label_side == "left" or label_side == "l":
        formula = label + " ~ " + "+".join(cols)
    elif label_side == "right" or label_side == "r":
        formula = "+".join(cols) + " ~ " + label

    return formula


def calc_prog_scores(time_series, bl_index, method):
    """
    Calculates the progression scores. Can be done using either a z-score normalization to baseline or expressing the \
    score as log-ratio of baseline value.

    :param time_series: pandas.Series storing the values at the different points in time which shall be transformed \
    into progression scores.
    :param bl_index: Value representing the baseline measurement in the time column.
    :param method: Specifies which progression score should be calculated. z-score ("z-score") or ratio of baseline \
    ("robl")
    :return: Calculated progression scores
    """

    def _z_score_formula(x, bl, sd):
        """
        Calculates a z-score.

        :param x: Feature value
        :param bl: Baseline feature value
        :param sd: Standard deviation
        :return: z-score
        """
        return (x - bl) / sd

    def _robl_formula(x, bl):
        """
        Calculates the log-ratio between the current feature value and the baseline feature value.

        :param x: Feature Value
        :param bl: Baseline feature Value
        :return: Baseline feature value ratio
        """
        return np.log(x / bl)

    # get baseline value
    try:
        bl_value = time_series.loc[bl_index]
    # raise error if no baseline value is present
    except KeyError:
        raise KeyError("No Baseline value found for entity.")

    # when there is no baseline measurement present return NaN for all values
    if type(bl_value) == pd.Series:
        raise ValueError("Multiple baseline entries have been found have been found for one entity.")

    if not pd.isnull(bl_value):
        if method == "z-score":
            # calculate standard deviation
            sd = np.std(time_series)
            return time_series.apply(_z_score_formula, args=(bl_value, sd))

        elif method == "robl":
            return time_series.apply(_robl_formula, args=(bl_value,))

    else:
        return np.nan


def _non_present_values_to_zero(test_data):
    """
    Adds keys to the test data of one dataframe, if key was not present in that one but in one of the other datasets.

    :param test_data:
    :return:
    """
    for dataset1 in test_data:
        for dataset2 in test_data:

            for key in dataset1.keys():

                if key not in dataset2:
                    dataset2[key] = 0
    return test_data

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

def _create_result_table(result, p_val_col, p_trans, counts):
    """
    Builds the dataframe showing the results.

    :param result:
    :param p_val_col:
    :param p_trans:
    :param counts:
    :return:
    """
    # store test results
    result_table = pd.DataFrame(p_val_col)

    # rename columns and set values in result dataframe
    result_table.rename(columns={0: "p-value"}, inplace=True)

    # insert corrected p_values
    if result:
        result_table["cor_p-value"] = result[1]
        result_table["signf"] = result[0]
    elif result is None:
        result_table["cor_p-value"] = np.nan
        result_table["signf"] = np.nan

    # combine p_value information with dataset and feature information stored in p_trans
    result_table = pd.concat([result_table, p_trans], axis=1, join="outer")

    # create multi index brom feature name result_table[2] and datasets result_table[1]
    result_table.index = result_table[[2, 1]]
    result_table.index = pd.MultiIndex.from_tuples(result_table.index)
    result_table.drop([0, 1, 2], axis=1, inplace=True)

    # name index levels
    result_table.index.levels[0].name = "features"
    result_table.index.levels[1].name = "datasets"

    # join with counts dataframe to display number of datapoint for each comparison
    result_table = result_table.join(counts, how="outer")

    return result_table


def create_contin_mat(data, dataset_labels, observation_col):
    """
    Creates a contingency table from a dictionary of observations.

    :param data: Dataframe containing observations.
    :param dataset_labels: Labels of the datasets used as keys in 'data' dict.
    :param observation_col: Name of the column in which the values of interest are stored. e.g. "Gender".
    :return: contingency matrix
    """
    contingency_matrix = dict()

    # count for each label
    for dataset_nr in data[dataset_labels].unique():
        # select subset of the dataframe, that belongs to one of the original datasets
        dataset = data[data[dataset_labels] == dataset_nr][::]
        # drop data points with missing values in value column
        dataset.dropna(subset=[observation_col], inplace=True)

        # count occurences
        counts = Counter(dataset[observation_col])

        # add to confusion matrix
        contingency_matrix[dataset_nr] = counts

    return pd.DataFrame(contingency_matrix).transpose()


def _categorical_table(data):
    """
    Returns the number of occurrences for the categories. Is used to build the observation table
    for a chi square test.

    :param data:
    :return:
    """
    # count occurences
    c = Counter(data)
    # delete NaNs
    c = {key: c[key] for key in c if not pd.isnull(key)}

    return pd.Series(c)


def calculate_cluster_purity(contingency_mat):
    """
    Will calculate the cluster purity given a contingency matrix.

    :param contingency_mat: Contingency matrix containing the observations.
    :return: Cluster purity value
    """
    return contingency_mat.max().sum() / contingency_mat.values.sum()

def make_ticks_int(tick_list):
    """
    Converts axis ticks to integers.

    :param tick_list: Iterable of the axis ticks to be converted. Should be sortend in the order they shall be put on \
    the axis.
    :return:
    """
    return [int(tick) for tick in tick_list]