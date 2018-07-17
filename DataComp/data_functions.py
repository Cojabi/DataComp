# -*- coding: utf-8 -*-

import pandas as pd

from operator import itemgetter

def get_data(paths, groupby=None, classes=None, exclude_classes=[], rel_cols=None, sep=","):
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
        try:
            df = pd.read_csv(path, index_col=0, sep=sep)
        except  : #TODO get fitting error
            df = pd.read_excel(path, index_col=0)
        return df

    # initialize list to store dataframes in
    dfs = []

    # Handle single path input
    if groupby and len(paths)==1:
        data = _load_data(*paths)
        grouping = data.groupby(groupby)

        for name, grp in grouping:  # split dataframe groups and create a list with all dataframes
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
    if len(paths) > 1:
        for path in paths:
            df = _load_data(path)
            dfs.append(df)

    if classes:
        df_names = classes
    else:
        df_names = ["df" + str(x) for x in range(1, len(dfs) + 1)]

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

def get_common_features(dfs, exclude=None):
    """
    Creates a set of the common features shared between dataframes.
    :param dfs: List of dataframes
    :param exclude: List of features which shall be taken out of consideration
    :return: set of common features across the dataframes
    """
    feats = get_feature_sets(dfs)

    common_feats = set.intersection(*feats)

    if exclude:
        for feat in exclude:
            common_feats.remove(feat)

    return list(common_feats)

def reduce_to_feat_subset(dfs, feat_subset=None):
    """
    Manipulate the dataframe to only contain the overlapping features.
    :param dfs: List of Dataframes
    :return: List of dataframes where the features are identical
    """
    if feat_subset is None:
        feat_subset = get_common_features(dfs)

    return [df[feat_subset] for df in dfs]

def reduce_dfs(dfs, col, val):
    """ """
    # create list with reduced dataframes
    reduced_dfs = [df[df[col] == val] for df in dfs]
    return reduced_dfs

def get_feature_sets(dfs):
    """
    Creats a list of sets, where each set stores the variable names of one dataframe
    :param dfs: list of dataframes
    :return: List of sets. Each set contains the feature names of one of the dataframes
    """
    # Create list containing features as sets
    return [set(df) for df in dfs]

def get_feature_differences(dfs):
    """ """
    feats = get_feature_sets(dfs)

    # create a dictionary storing the features which are distinct
    diff_dict = dict()
    # compare each dataset against each and collect differences
    for i in range(len(feats)):
        for j in range(i + 1, len(feats)):
            # take union from differences
            diff_dict[i, j] = feats[i].difference(feats[j]).union(feats[j].difference(feats[i]))

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