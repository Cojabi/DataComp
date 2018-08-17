# -*- coding: utf-8 -*-

import pandas as pd

from .datacollection import DataCollection


def create_prop_matched_dfs(matches_path, datacol):
    """
    Creats a new list of dataframes but now only containing the matched cases. Propensity Score Matching must be performed
    previously.
    :param matches_path: Path to a csv which contains the matched data. 2 Columns: one lists the subjects of df1 and
    the other lists the matching sample from df2.
    :param datacol: list of dataframes
    :return: list of dataframes containing only the matches samples
    """

    # load matches and drop non matched ids
    matched = pd.read_csv(matches_path, index_col=0)
    matched.dropna(inplace=True)

    # create dfs containing only matched data. Try to get oder of dataframes and matching columns correct
    try:
        prop_dfs = [datacol[1].loc[matched.index], datacol[0].loc[matched["Match"]]]
    except KeyError:
        prop_dfs = [datacol[0].loc[matched.index], datacol[1].loc[matched["Match"]]]

    return DataCollection(prop_dfs, datacol.df_names)


def create_prop_matched_dfs_longitudinal(matches_path, datacol, pat_col):
    """
    Creats a new list of dataframes but now only containing the matched cases. Propensity Score Matching must be performed
    previously.
    :param matches_path: Path to a csv which contains the matched data. 2 Columns: one lists the subjects of df1 and
    the other lists the matching sample from df2.
    :param datacol: list of dataframes
    :return: list of dataframes containing only the matches samples
    """

    # load matches and drop non matched ids
    matched = pd.read_csv(matches_path, index_col=0)
    matched.dropna(inplace=True)

    majority_inds = matched["Match"]
    minority_inds = matched.index

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

    return DataCollection(prop_dfs, datacol.df_names)
