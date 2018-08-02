# -*- coding: utf-8 -*-

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
    Constructs a formula string from column names and label
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
