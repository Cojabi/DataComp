# -*- coding: utf-8 -*-

import pandas as pd


def get_data(path1, path2=None, groupby=None, class1=None, class2=None):
    """Will load the data and return a list of two dataframes
    that can then be used for later comparism.
    :param path1: Path to dataframe1
    :param path2: Path to dataframe2. Optional if all data for comparison is in df1.
                  Then use groupby argument
    :param groupby: name of the column which specifies classes to compare to each other. (e.g. sampling site)
    """
    data = pd.read_csv(path1, index_col=0)

    if groupby:
        grouping = data.groupby(groupby)
        df1 = grouping.get_group(class1)[::]
        df2 = grouping.get_group(class2)[::]

    if path2:
        df1 = data
        df2 = pd.read_csv(path2, index_col=0)

    dfs = [df1, df2]
    df_names = [class1, class2]

    return dfs, df_names


def create_zipper(dfs, feats=None):
    """create zipper containing the values of the same features per df in one list.
    (df1_feat1, df2_feat1, df3_feat1), (df1_feat2, df2_feat2, df3_feat2),"""
    if feats is None:
        feats = list(dfs[0])

    df_feats = []

    for df in dfs:
        df_feats.append([list(df[feat].dropna()) for feat in feats])

    zip_values = zip(*df_feats)
    zipper = dict(zip(feats, zip_values))
    return zipper

