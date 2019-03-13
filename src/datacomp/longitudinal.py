# -*- coding: utf-8 -*-

import os

import pandas as pd


##### DEPRECATED? ######
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
