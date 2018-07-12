import pandas as pd
import os
import numpy as np


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

##### ONGOING WORK######

def calc_prog_scores(values, bl_index, method):
    """ """

    def _z_score_formula(x, bl, sd):
        """ """
        return (x - bl) / sd

    def _pobl_formula(x, bl):
        """ """
        return x / bl

    # get baseline value
    bl = values.loc[bl_index]

    # when there is no baseline measurement present return NaN for all values
    if not pd.isnull(bl):
        if method == "z-score":
            # calculate standard deviation
            sd = np.std(values)
            return values.apply(_z_score_formula, args=(bl, sd))
        elif method == "pobl":
            return values.apply(_pobl_formula, args=(bl,))
    else:
        return np.nan


def create_progression_tables(dfs, feats, time_col, patient_col, method, bl_index):
    """ """

    for df in dfs:
        patients = df[patient_col]
        # create dataframe copy to keep from alternating original dataframe
        prog_df = df[feats][::]

        for feat in feats:

            for patient in patients.unique():
                # collect values for sinlge patient
                pat_inds = df[df[patient_col] == patient].index
                # create value series storing the values of a patient
                values = df.loc[pat_inds, feat]
                values.index = df.loc[pat_inds, time_col]

                # calculate scores for patient and reindex to merge back into dataframe copy
                try:  # DEBUGGING MESS
                    try:
                        scores = calc_prog_scores(values, bl_index, method)
                        try:
                            scores.index = pat_inds
                            prog_df.loc[pat_inds, feat] = scores
                        except AttributeError:
                            prog_df.loc[pat_inds, feat] = scores
                    except ValueError:
                        print(pat_inds)
                except KeyError:
                    print(pat_inds)

    return prog_df