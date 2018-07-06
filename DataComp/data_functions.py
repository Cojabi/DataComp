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
    def _load_data(path):
        """small function to load according to the dataformat. (excel or csv)"""
        try:
            df = pd.read_csv(path, index_col=0)
        except  :
            df = pd.read_excel(path, index_col=0)
        return df

    # initialize list to store dataframes in
    dfs = []

    # Handle single path input
    if groupby and len(paths)==1:
        data = _load_data(*paths)
        grouping = data.groupby(groupby)

        for name, grp in grouping:  # split dataframe groups and create a list with all dataframes
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


def creat_prop_matched_df(matches_path, dfs):
    """
    Creats a new list of dataframes but now only containing the matched cases. Propensity Score Matching must be performed
    previously.
    :param matches_path: Path to a csv which contains the matched data. 2 Columns: one lists the subjects of df1 and
    the other lists the matching sample from df2.
    :param dfs: list of dataframes
    :return: list of dataframes containing only the matches samples
    """

    # load matches and drop non matched ids
    matched = pd.read_csv(matches_path, index_col=0)
    matched.dropna(inplace=True)

    # create dfs containing only matched data. Try to get oder of dataframes and matching columns correct
    try:
        prop_dfs = [dfs[1].loc[matched.index], dfs[0].loc[matched["Match"]]]
    except KeyError:
        prop_dfs = [dfs[0].loc[matched.index], dfs[1].loc[matched["Match"]]]

    return prop_dfs

def qc_prop_matching(dfs, rel_cols, label):
    """
    Evaluates the need for a propensity score matching and can be used to quality control a propensity score matched
    population. Will train classifiers and create a plot.
    :param dfs: List of dataframes
    :param rel_cols: relevant columns
    :param label: Label or class which should be regressed. (cohort1/cohort2, case/control, treatment/untreated etc.)
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
    cols = rel_cols[::]

    if label in rel_cols:
        cols.remove(label)

    formula = label + " ~ " + "+".join(cols)
    return formula

def get_common_features(dfs, exclude=None):
    """
    Creates a set of the common features shared between dataframes.
    :param dfs: List of dataframes
    :param exclude: List of features which shall be taken out of consideration
    :return: set of common features across the dataframes
    """
    feats = get_feature_sets()

    common_feats = set.intersection(*feats)

    if exclude:
        for feat in exclude:
            common_feats.remove(feat)

    return list(common_feats)

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
