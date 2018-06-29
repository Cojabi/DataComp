# -*- coding: utf-8 -*-

import pandas as pd
from pymatch.Matcher import Matcher


def get_data(paths, groupby=None, classes=None, rel_cols=None, sep=","):
    """Will load the data and return a list of two dataframes
    that can then be used for later comparism.
    :param path1: Path to dataframe1
    :param path2: Path to dataframe2. Optional if all data for comparison is in df1.
                  Then use groupby argument
    :param rel_cols: List of relevant columns to consider. When given only those columns will be used. Otherwise all
    :param groupby: name of the column which specifies classes to compare to each other. (e.g. sampling site)
    """

    dfs = []

    if groupby:
        data = pd.read_csv(*paths, index_col=0, sep=sep)
        grouping = data.groupby(groupby)

        for name, grp in grouping:  # split dataframe groups and create a list with all dataframes
            df = grouping.get_group(name)[::]

            # consider all columns as relevant is no rel_cols given.
            if rel_cols is None:
                rel_cols = list(df)

            # consider the relevant columns
            dfs.append(df[rel_cols])

    if len(paths) > 1:
        for path in paths:
            df = pd.read_csv(path, index_col=0)
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


def creat_prop_matched_df(matches_path, dfs):
    """ """

    matches_path = "/home/colin/SCAI/git/Dataset_comparison/compare_sites_data/matches.csv"

    # load matches and drop non matched ids
    matches = pd.read_csv(matches_path)
    matches.dropna(inplace=True)

    # prepare matched ids
    adni_ids = matches["1"]

    # create dfs containing only matched data
    prop_dfs = [dfs[1].loc[matches["Unnamed: 0"]], dfs[0].loc[adni_ids]]

    return prop_dfs

def qc_prop_matching(dfs, rel_cols, label):
    """
    Evaluates the need for a propensity score matching and
    :param dfs:
    :param rel_cols:
    :param label:
    :return:
    """

    cols = rel_cols[::]

    # create reduced copies of the dataframes for propensity score quality control
    qc_dfs = []
    for df in dfs:
        qc_dfs.append(df[cols])

    # construct formula
    cols.remove(label)
    formula = construct_formula(label, cols)

    # create Matcher
    m = Matcher(*qc_dfs, yvar=label, formula=formula)
    # train classifier to asses predictability
    m.fit_scores(balance=True, nmodels=10)
    # calculate and visualize propensity scores
    m.predict_scores()
    m.plot_scores()

def construct_formula(label, rel_cols):
    """
    Constructs a formula string from column names and label
    :param label: Label or class which should be regressed for. (case/control, treatment/untreated etc.)
    :param rel_cols: Relevant columns for the formula
    :return: formula string
    """

    formula = label + " ~ " + "+".join(rel_cols)
    return formula
