# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np

from operator import itemgetter


def get_sig_feats(sig_df):
    """
    Get's the feature names of significantly deviating features from a result table.

    :param sig_df: Dataframe storing the p_values and the corrected p_values like returned by stats.p_correction()
    :return:
    """
    # grab significant deviances
    sig_entries = sig_df[sig_df["signf"]]
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
    Calculates the progression scores. Can be done using either a z-score normalization to baseline or expressing the
    score as ratio of baseline value.

    :param time_series: pandas.Series storing the values at the different points in time which shall be transformed into
    progression scores.
    :param bl_index: Value representing the baseline measurement in the time column.
    :param method: Specifies which progression score should be calculated. z-score ("z-score") or
    ratio of baseline ("robl")
    :return:
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
        Calculates the ratio between the current feature value and the baseline feature value.

        :param x: Feature Value
        :param bl: Baseline feature Value
        :return: Baseline feature value ratio
        """
        return x / bl

    # get baseline value
    try:
        bl_value = time_series.loc[bl_index]
    # raise error if no baseline value is present
    except KeyError:
        raise KeyError("No Baseline value found for entity.")

    # when there is no baseline measurement present return NaN for all values
    if type(bl_value) == pd.Series:
        # raise ValueError("Multiple baseline entries have been found have been found for one entity.")
        return np.nan  # TODO FIX THIS ITS just added
    if not pd.isnull(bl_value):
        if method == "z-score":
            # calculate standard deviation
            sd = np.std(time_series)
            return time_series.apply(_z_score_formula, args=(bl_value, sd))

        elif method == "robl":
            return time_series.apply(_robl_formula, args=(bl_value,))

    else:
        return np.nan
