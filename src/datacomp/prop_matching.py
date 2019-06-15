# -*- coding: utf-8 -*-

import pandas as pd

from .datacollection import DataCollection


def create_prop_matched_dfs(matched, datacol):
    """
    Creates a new DataCollection containing only the matched cases. A table listing the matched data points is required.

    :param matched: Either a path to a csv which contains the matched data or a dataframe containing the matches. \
    2 Columns: one lists the subjects of df1 and the other lists the matching sample from df2.
    :param datacol: DataCollection object
    :return: DataCollection object containing only the matches samples
    """

    # load matches and drop non matched ids
    if type(matched) == str:
        matched = pd.read_csv(matched)

    matched.dropna(inplace=True)

    # create dfs containing only matched data. Try to get oder of dataframes and matching columns correct
    # check if mached_labels are in the first or second position of the DataCollection

    if set(datacol[0].index.intersection(matched.iloc[:, 0])) == set(matched.iloc[:, 0]):
        prop_dfs = [datacol[0].loc[datacol[0].index.intersection(matched.iloc[:, 0])],
                    datacol[1].loc[datacol[1].index.intersection(matched.iloc[:, 1])]]

    elif set(datacol[1].index.intersection(matched.iloc[:, 0])) == set(matched.iloc[:, 0]):
        prop_dfs = [datacol[1].loc[datacol[1].index.intersection(matched.iloc[:, 0])],
                    datacol[0].loc[datacol[0].index.intersection(matched.iloc[:, 1])]]
    else:
        raise ValueError("Matched labels do not fit to either of the dataframes in the datacollection!")

    return DataCollection(prop_dfs, datacol.df_names, datacol.categorical_feats)


def create_prop_matched_dfs_longitudinal(matches_path, datacol, pat_col):
    """
    Creates a new Collection containing only the matched cases. A table listing the matched data points is required.

    :param matches_path: Path to a csv which contains the matched data. 2 Columns: one lists the subjects of df1 and \
    the other lists the matching sample from df2.
    :param datacol: DataCollection object
    :return: DataCollection object containing only the matches samples
    """

    # load matches and drop non matched ids
    matched = pd.read_csv(matches_path)
    matched.dropna(inplace=True)

    majority_inds = matched.iloc[:, 1]
    minority_inds = matched.iloc[:, 0]

    # create dfs containing only matched data. Try to get oder of dataframes and matching columns correct
    try:
        majority_pats = datacol[1].loc[majority_inds, pat_col]
        majority_df = datacol[1][datacol[1][pat_col].isin(majority_pats)]

        minority_pats = datacol[0].loc[minority_inds, pat_col]
        minority_df = datacol[0][datacol[0][pat_col].isin(minority_pats)]

        prop_dfs = [minority_df, majority_df]


    except KeyError:

        majority_pats = datacol[0].loc[majority_inds, pat_col]
        majority_df = datacol[0][datacol[0][pat_col].isin(majority_pats)]

        minority_pats = datacol[1].loc[minority_inds, pat_col]
        minority_df = datacol[1][datacol[1][pat_col].isin(minority_pats)]

        prop_dfs = [majority_df, minority_df]

    return DataCollection(prop_dfs, datacol.df_names, datacol.categorical_feats)
