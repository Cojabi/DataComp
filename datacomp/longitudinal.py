import pandas as pd
import os

##### ONGOING WORK######
"""
def analyze_longitudinal_feats(dfs, time_col, bl_index, cat_feats=None, num_feats=None, include=None, exclude=None):
    """ """
    # dict to collect p_values in
    p_values = dict()
    # dict to collect dataframes reduced to only one time point. time point will be the key to the dataframe
    red_df_store = dict()

    # if no list of features is given, take all
    if not num_feats:
        num_feats = list(dfs[0])
    # if no categorical features are given take empty list
    if not cat_feats:
        cat_feats = []

    # create a set of all time_points present in the dataframes
    time_points = create_value_set(dfs, time_col)
    time_points.remove(bl_index)

    # for each timepoint collect the data and compare the data
    for time in time_points:

        reduced_dfs = reduce_dfs(dfs, time_col, time)
        red_df_store[time] = reduced_dfs
        time_zipper = create_zipper(reduced_dfs)

        p_values[time] = analyze_all_features(time_zipper, cat_feats=cat_feats, num_feats=num_feats,
                                                      exclude=exclude, include=include, verbose=False)
    return p_values, red_df_store
"""

# !!! STILL VERY TIME INEFFICIENT. WORKS FOR NOW BUT NEEDS REWORK LATER ON !!!
def transform_to_longitudinal(df, feats, pat_col, time_col, save_folder):
    """
    Transforms a long format (each visit of patient stored in one row) dataframe feature into a longitudinal format
    dataframe. The values in time column will give the column names while one row will store all the consecutive
    visits of a patient.

    :param df: The pandas dataframe storing the features in long format
    :param feats: A list of features for which longitudinal dataframes shall be constructed
    :param pat_col: The column name listing the patient IDs
    :param time_col: The column name listing the time events (visits, months...)
    :param save_folder: A folder in which the longitudinal dataframes shall be saved.
    :return:
    """
    # create dataframe in which longitudinal data is stored
    long_df = pd.DataFrame()
    patients = df[pat_col]

    for feat in feats:

        for patient in patients.unique():
            # collect data relevant for patient
            pat_df = df[df[pat_col] == patient][::]
            # include patient data as row into dataframe
            pat_df.index = pat_df[time_col]
            long_df = long_df.append(pat_df[feat])

        # set labels for longitudinal dataframe
        long_df.index = patients.unique()
        # try to convert column names into integers for appropriate sorting
        try:
            long_df.columns = long_df.columns.astype(int)
        except TypeError:
            pass
        # save longitudinal version of the current feature
        save_path = os.path.join(save_folder, feat + "_longitudinal.csv")
        long_df.to_csv(save_path)
