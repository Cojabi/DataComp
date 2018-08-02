# -*- coding: utf-8 -*-

import matplotlib.pylab as plt
import numpy as np
import os
import seaborn as sns

from .utils import get_sig_feats

# load plot style
plt.style.use('ggplot')


def bp_all_sig_feats(sig_df, zipper, df_names, feat_subset=None, save_folder=None):
    """
    Plots boxplots for each significant feature to allow for visual comparison.

    :param sig_df: Dataframe storing the p_values, corrected p_values and a boolean if significant or not.
    Is provided as outcome of the p_correction or analyze_feature_ranges function
    :param zipper: zipper dict, that contains variable values. For each key the value is a list containing x
    lists (the values of the features in the x dataframes)
    :param df_names: List storing the names of the dataframes. Used for the x-axis label
    :param feat_subset: list a subset of the features. Only for the mentioned features, a plot will be created.
    :param save_folder: Path to a folder in which the plots shall be saved
    :return:
    """
    # get significant features
    sig_feats = get_sig_feats(sig_df)

    # create zipper containing only the significantly deviating features
    sig_zipper = {x: zipper[x] for x in sig_feats}
    bp_single_features(sig_zipper, df_names, feat_subset=feat_subset, save_folder=save_folder)

def bp_single_features(zipper, df_names, feat_subset=None, save_folder=None):
    """
    Creates one boxplot figure per feature

    :param zipper: zipper dict, that contains numerical variables. For each key the value is a list containing x
    lists (the values of the features in the x dataframes)
    :param df_names: names of the datasets to label figures accordingly
    :param feat_subset: a list of features for which a plot shall be made. One plot per feature. If None all features
    will be considered.
    :param save_folder: a path to a directory where to store the figures.
    :return:
    """
    # calculate positions for boxplots
    positions = range(1, len(df_names) + 1)

    if feat_subset is None:
        feat_subset = zipper.keys()

    for feat in feat_subset:
        # create new figure
        fig = plt.figure()
        ax = plt.axes()

        bp = plt.boxplot(zipper[feat], positions=positions, widths=0.6)
        # colorbps(bp)

        # set axes limits and labels
        plt.xlim(0, np.max(positions) + 1)
        ax.set_xticks(positions)
        ax.set_xticklabels(df_names)
        # set title
        plt.title(feat)

        if save_folder:
            save_file = os.path.join(save_folder, feat + ".png")
            fig.savefig(save_file)
        else:
            plt.show()


def feature_distplots(zipper, feat_subset=None, save_folder=None):
    # set colors
    colors = ["b", "c", "r"]

    # if no feature subset is provided, consider all features
    if feat_subset is None:
        feat_subset = zipper.keys()

    for feat in feat_subset:
        # include feature distributions of all dataframes into plot
        for dataset_feature, color in zip(zipper[feat], colors):
            sns.distplot(dataset_feature, hist=False, color=color, kde_kws={"shade": True})

            # set title
            plt.title(feat)

        if save_folder:
            save_file = os.path.join(save_folder, feat + ".png")
            plt.savefig(save_file)
        else:
            plt.show()

#### NOT NEEDED???
"""Muss noch nen colorschema bekommen plus legende, damit man die verschiedenen dfs unterscheiden kann."""
def bp_all_features(num_zipper, df_names, save=None):
    """
    Plots boxplots for all features and all dfs into one figure
    :param num_zipper: zipper dict, that contains numerical variables. For each key the value is a list containing x
    lists (the values of the features in the x dataframes)
    :param save: a path where to save the figure to
    :return:
    """
    fig = plt.figure()
    ax = plt.axes()

    add_value = len(df_names) + 1  # is used to define the positons of the boxplots
    positions = range(1, add_value)
    xticks = []  # stores the positions where axis ticks shall be

    for feat_data in num_zipper:
        bp = plt.boxplot(num_zipper[feat_data], positions=positions, widths=0.6)
        # colorbp(bp)
        xticks.append(np.mean(positions))
        positions = [x + add_value for x in positions]

    # set axes limits and labels
    plt.xlim(0, np.max(positions))
    ax.set_xticklabels(num_zipper.keys())
    ax.set_xticks(xticks)

    if save:
        fig.savefig(save)
    else:
        plt.show()