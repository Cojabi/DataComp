import pandas as pd

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
