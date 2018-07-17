import pandas as pd
from pymatch.Matcher import Matcher

def create_prop_match_labels(dfs, label):
    """creates dataset labels for each data set to use in propensity scoring."""

    for i, df in zip(range(len(dfs)), dfs):
        df[label] = i

    return dfs

def create_dfs_for_matching(dfs, label_name, save_path, labels=[1,0]):
    """
    Will create a combined dataframe in which labels are assigned for propensity score matching. The resulting dataframe
    will be saved under save_path and can be used for propensity_score_matching.
    :param dfs:
    :param label_name:
    :param save_path:
    :param labels:
    :return:
    """
    for i in range(len(labels)):
        dfs[i][label_name] = labels[i]

    pd.concat(dfs).to_csv(save_path)

def create_prop_matched_dfs(matches_path, dfs):
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