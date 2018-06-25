# -*- coding: utf-8 -*-

import pandas as pd


def get_data(paths, groupby=None, classes=None, sep=","):
    """Will load the data and return a list of two dataframes
    that can then be used for later comparism.
    :param path1: Path to dataframe1
    :param path2: Path to dataframe2. Optional if all data for comparison is in df1.
                  Then use groupby argument
    :param groupby: name of the column which specifies classes to compare to each other. (e.g. sampling site)
    """

    dfs = []

    if groupby:
        data = pd.read_csv(*paths, index_col=0, sep=sep)
        grouping = data.groupby(groupby)

        for name, grp in grouping: # split dataframe groups and create a list with all dataframes
            df = grouping.get_group(name)[::]
            dfs.append(df)

    if len(paths) > 1:
        for path in paths:
            df = pd.read_csv(path, index_col=0)
            dfs.append(df)

    if classes:
        df_names = classes
    else:
        df_names = ["df" + str(x) for x in range(1, len(dfs)+1)]

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


if __name__ == '__main__':
    get_data("/home/colin/git/DataComp/niklas_test.csv", groupby="DD01")