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

