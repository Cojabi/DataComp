# -*- coding: utf-8 -*-

import warnings

from collections import Counter
from operator import itemgetter

import numpy as np
import pandas as pd

from scipy.stats import sem, t


def get_sig_feats(sig_df):
    """
    Get's the feature names of significantly deviating features from a result table.

    :param sig_df: Dataframe storing the p_values and the corrected p_values like returned by stats.p_correction()
    :return:
    """
    # grab significant deviances
    series = sig_df["signf"]
    series = series.fillna("False")
    index_labels = series[series == True].index.labels[0]
    return set(itemgetter(index_labels)(series.index.levels[0]))


def get_diff_feats(sig_df):
    """
    Get's the feature names of features from a result table who's confidence interval for difference in means does not \
    include 0.

    :param sig_df: Dataframe storing the mean difference confidence intervals like returned by stats.p_correction()
    :return:
    """
    # grab significant deviances
    series = sig_df["diff_flag"]
    series = series.fillna("False")
    index_labels = series[series == True].index.labels[0]
    return set(itemgetter(index_labels)(series.index.levels[0]))


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


def _convert_multindex(val_dict, rename_dict):
    """
    This function will create a multiindex dataframe which lists features as outer and dataset combinations \
    as inner level for row indices. It is for example used to convert the output of stats.calc_diff_invs into a \
    dataframe that can be joined with the result_table.

    :param val_dict: Nested dictionary with features as outer keys and dataset combinations as inner keys.
    :return: result_table like table with multiindex
    """

    # create a multiindex dict using tuples as keys from a nested dict. Pandas expects multiindeces to come as tuples
    reform = {(outerKey, innerKey): values for outerKey, innerDict in val_dict.items() for innerKey, values in
              innerDict.items()}

    result_table = pd.DataFrame(reform).transpose()

    # create multiindex to fit result table
    result_table.index.levels[0].name = "features"
    result_table.index.levels[1].name = "datasets"
    # rename columns
    result_table.rename(columns=rename_dict, inplace=True)

    return result_table


def _create_result_table(result, p_val_col, p_trans, diff_confs, conf_invs, counts):
    """
    Builds the dataframe displaying the results.
    The first three arguments are used to handle the p-values and match them back to the corresponding dataset and \
    features, because they will be ordered during multiple testing correction.

    :param result:
    :param p_val_col: Stores the uncorrected p_values
    :param p_trans: Stores uncorrected p_values, the dataset combination which was tested and the corresponding \
    feature name
    :param conf_invs: DataFrame storing the 95% confidence interval per dataset, per feature.
    :param counts: DataFrame storing the number of observations per dataset, per feature.
    :return: Result table listing all important outcomes of the statistical comparison
    """
    # store test results
    result_table = pd.DataFrame(p_val_col)

    # rename columns and set values in result dataframe
    result_table.rename(columns={"0": "p-value"}, inplace=True)

    # insert corrected p_values
    if result:
        result_table["cor_p-value"] = result[1]
        result_table["signf"] = result[0]
    elif result is None:
        result_table["cor_p-value"] = np.nan
        result_table["signf"] = np.nan

    # combine p_value information with dataset and feature information stored in p_trans
    result_table = pd.concat([result_table, p_trans], axis=1, join="outer")

    # create multi index from feature name (result_table[2]) and datasets (result_table[1])
    result_table.index = result_table[[2, 1]]
    result_table.index = pd.MultiIndex.from_tuples(result_table.index)
    result_table.drop([0, 1, 2], axis=1, inplace=True)

    # name index levels
    result_table.index.levels[0].name = "features"
    result_table.index.levels[1].name = "datasets"

    # prepare confidence interval for mean difference
    diff_confs = _convert_multindex(diff_confs, {0: "mean_diff", 1: "diff_flag"})

    # join with mean difference confidence intervals dataframe
    result_table = result_table.join(diff_confs, how="outer")
    # join with actual mean confidence intervals dataframe
    result_table = result_table.join(conf_invs, how="outer")
    # join with counts dataframe to display number of datapoint for each comparison
    result_table = result_table.join(counts, how="outer")

    return result_table.sort_index()


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


def conf_interval(data_series):
    """
    Calculate the confidence interval for the data distribution under the assumptions that it can be calculated using \
    a student-t distribution.

    :return start: Starting value of the interval
    :return end: Ending value of the interval
    """

    mean = np.mean(data_series)

    conf_int = sem(data_series) * t.ppf((1 + 0.95) / 2, len(data_series) - 1)

    start = mean - conf_int
    end = mean + conf_int

    return start, end


def get_cat_frequencies(series):
    """
    Counts the occurrences for each factor of a categorical variable and calculates the relative frequencies.

    :param series: Iterable storing the realisations of a categorical random variable / feature.
    :return freqs: Pandas Series storing the relative frequencies using the corresponding factor as index
    :return counts.sum(): Total number of realisations of the categorical variable
    :return counts: Pandas Series storing the counts using the corresponding factor as index
    """

    # count occurrences and store in Series
    counts = pd.Series(Counter(series))
    # calculate frequencies
    freqs = counts / counts.sum()

    return freqs, counts.sum(), counts


def means_vars_numerical(series1, series2):
    """
    Calculates means and variances for two iterable stroing data. The outputs are then futher used in the calculation \
    of confidence intervals for the difference in means.

    :param series1: Iterable storing the values of variable 1.
    :param series2: Iterable storing the values of variable 2.
    :return mean1: Mean for values in series1
    :return mean2: Mean for values in series2
    :return var1: Variance for values in series1
    :return var2: Variance for values in series2
    """
    mean1 = np.mean(series1)
    mean2 = np.mean(series2)
    var1 = np.var(series1, ddof=1)
    var2 = np.var(series2, ddof=1)

    return mean1, mean2, var1, var2


def means_vars_categorical(series1, series2):
    """
    A generator which will yield means and variances as well as number of occurences for each factor encountered in \
    the two provided data series.

    :param series1: Iterable storing the values of variable 1.
    :param series2: Iterable storing the values of variable 2.
    :return mean1: Mean for values in series1
    :return mean2: Mean for values in series2
    :return var1: Variance for values in series1
    :return var2: Variance for values in series2
    :return  counts1[factor]: Number of occurrences of the current factor in data series 1.
    :return  counts2[factor]: Number of occurrences of the current factor in data series 2.
    :return factor: The currently processed factor of the categorical variable
    """

    # calculate relative frequencies and n for each categorical value
    freqs1, n1, counts1 = get_cat_frequencies(series1)
    freqs2, n2, counts2 = get_cat_frequencies(series2)

    # calculate means and variances for each factor for each dataset (series)
    for factor in freqs1.index.union(freqs2.index):
        mean1 = n1 * freqs1[factor]
        mean2 = n2 * freqs2[factor]
        var1 = n1 * freqs1[factor] * (1 - freqs1[factor])
        var2 = n2 * freqs2[factor] * (1 - freqs2[factor])

        yield mean1, mean2, var1, var2, counts1[factor], counts2[factor], factor


def diff_mean_conf_formula(n1, n2, mean1, mean2, var1, var2, rnd=2):
    """
    Calculates the confidence interval for the difference of means between two features.

    :param n1: Sample size of sample 1
    :param n2: Sample size of sample 2
    :param mean1: Mean of sample 1
    :param mean2: Mean of sample 2
    :param var1: Variance of sample 1
    :param var2: Variance of sample 2
    :param rnd: Number of decimal places the result shall be round to. Default is 2.
    :return: Confidence interval given as a list: [intervat start, interval end]
    """
    # estimate common variance
    s2 = ((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 - 1 + n2 - 1)

    # estimate standard deviation
    sd = np.sqrt(s2 * (1 / n1 + 1 / n2))

    # calculate difference in means
    diff = mean1 - mean2

    # set z value. 1.96 is standard for a 95% significance level
    z = 1.96  # t.ppf((1+0.95) / 2, len(series1)-1+len(series2)-1)

    start = diff - z * sd
    end = diff + z * sd

    return [np.round(start, rnd), np.round(end, rnd)]


def diff_prop_conf_formula(n1, n2, prop1, prop2, rnd=2):
    """
    Calculates the confidence interval for the difference of proportions between two features.

    :param n1: Sample size of sample 1
    :param n2: Sample size of sample 2
    :param prop1: Mean of sample 1
    :param prop2: Mean of sample 2
    :param rnd: Number of decimal places the result shall be round to. Default is 2.
    :return: Confidence interval given as a list: [intervat start, interval end]
    """

    # calculate combined standard error
    se = np.sqrt(prop1 * (1 - prop1) / n1 ** 2 + prop2 * (1 - prop2) / n2 ** 2)

    # calculate difference in means
    diff = prop1 - prop2

    # set z value. 1.96 is standard for a 95% significance level
    z = 1.96  # t.ppf((1+0.95) / 2, len(series1)-1+len(series2)-1)

    start = diff - z * se
    end = diff + z * se

    return [np.round(start, rnd), np.round(end, rnd)]


def calc_mean_diff(series1, series2, var_type="n", rnd=2):
    """
    Calculates the confidence interval of the difference in means between two iterables.

    :param series1: Iterable storing the values of variable 1.
    :param series2: Iterable storing the values of variable 2.
    :param var_type: Indicator if numerical or categorical. "n" for numerical "c" for categorical.
    :param rnd: Number of decimal positions on which result shall be rounded.
    :return: List representing the interval
    """
    # Handle numerical features
    if var_type == "n":
        mean1, mean2, var1, var2 = means_vars_numerical(series1, series2)
        return diff_mean_conf_formula(len(series1), len(series2),
                                      mean1, mean2, var1, var2, rnd)

    elif var_type == "c":

        # calculate relative frequencies and n for each categorical value
        freqs1, n1, counts1 = get_cat_frequencies(series1)
        freqs2, n2, counts2 = get_cat_frequencies(series2)

        # collect all intervals for a variable in dictionary using the factors as keys
        invs = dict()

        # for each factor that the categorigal variable can take on, perform the confidence interval calculation
        for factor in freqs1.index.union(freqs2.index):

            prop1, prop2, n1, n2 = freqs1[factor], freqs2[factor], counts1[factor], counts2[factor]

            # check statistical assumptions
            if (n1 * prop1) < 10 or (n1 * (1-prop1)) < 10 or (n2 * prop2) < 10 or (n2 * (1-prop2)) < 10:
                warnings.warn(
                    "Sample size is smaller than 10 for the realisation", factor,"of the feature.",
                    UserWarning)

            invs[factor] = diff_prop_conf_formula(n1, n2, prop1, prop2, rnd)

    return invs
