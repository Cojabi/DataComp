# -*- coding: utf-8 -*-

import pandas as pd
import os

from operator import itemgetter

from .datacollection import DataCollection

def get_data(paths, df_names, groupby=None, exclude_classes=[], rel_cols=None, sep=","):
    """Will load the data and return a list of two dataframes
    that can then be used for later comparism.
    :param path1: Path to dataframe1
    :param path2: Path to dataframe2. Optional if all data for comparison is in df1.
                  Then use groupby argument
    :param rel_cols: List of relevant columns to consider. When given only those columns will be used. Otherwise all
    :param groupby: name of the column which specifies classes to compare to each other. (e.g. sampling site)
    """
    def _load_data(path, sep=sep):
        """small function to load according to the dataformat. (excel or csv)"""
        filename, file_extension = os.path.splitext(path)

        if file_extension in [".csv", ".tsv"]:
            df = pd.read_csv(path, index_col=0, sep=sep)
        else:
            df = pd.read_excel(path, index_col=0)

        return df

    # initialize list to store dataframes in
    dfs = []

    # Handle single path input
    if groupby and (len(paths)==1 or isinstance(paths, str)):

        # load data depending on if the single path is given in a list of as string
        if isinstance(paths, str):
            data = _load_data(paths, sep)
        elif isinstance(paths, list):
            data = _load_data(*paths, sep)
        else:
            raise ValueError("Seems that the input was a single path. Please input path as string or inside a list.")

        grouping = data.groupby(groupby)

        # split dataframe groups and create a list with all dataframes
        for name, grp in grouping:
            # skip class if it should be excluded
            if name in exclude_classes:
                continue

            df = grouping.get_group(name)[::]

            # consider all columns as relevant is no rel_cols given.
            if rel_cols is None:
                rel_cols = list(df)

            # consider the relevant columns
            dfs.append(df[rel_cols])

    # Handle multiple paths input
    elif len(paths) > 1:
        for path in paths:
            df = _load_data(path)
            dfs.append(df)

    return DataCollection(dfs, df_names)

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
