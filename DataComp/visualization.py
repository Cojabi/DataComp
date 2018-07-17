# -*- coding: utf-8 -*-

import matplotlib.pylab as plt
import numpy as np
import os
import matplotlib_venn as mv

from operator import itemgetter
from .data_functions import get_feature_sets

plt.style.use('ggplot')


def bp_all_sig_feats(sig_df, zipper, df_names, subset_feats=None, save_folder=None):
    """
    Plots boxplots for each significant feature to allow for visual comparison.
    :param sig_df: Dataframe storing the p_values, corrected p_values and a boolean if significant or not.
    Is provided as outcome of the p_correction or analyze_feature_ranges function
    :param zipper: zipper dict, that contains variable values. For each key the value is a list containing x
    lists (the values of the features in the x dataframes)
    :param df_names: List storing the names of the dataframes. Used for the x-axis label
    :param subset_feats: list a subset of the features. Only for the mentioned features, a plot will be created.
    :param save_folder: Path to a folder in which the plots shall be saved
    :return:
    """

    # grab significant deviances
    sig_entries = sig_df[sig_df["signf"]]
    index_labels = sig_entries.index.labels[0]
    sig_feats = set(itemgetter(index_labels)(sig_entries.index.levels[0]))

    # create zipper containing only the significantly deviating features
    sig_zipper = {x: zipper[x] for x in sig_feats}
    bp_single_features(sig_zipper, df_names, feats=subset_feats, save_folder=save_folder)


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


def bp_single_features(zipper, df_names, feats=None, save_folder=None):
    """
    Creates one boxplot figure per feature
    :param zipper: zipper dict, that contains numerical variables. For each key the value is a list containing x
    lists (the values of the features in the x dataframes)
    :param df_names: names of the datasets to label figures accordingly
    :param feats: a list of features for which a plot shall be made. One plot per feature
    :param save_folder: a path to a directory where to store the figures.
    :return:
    """

    positions = range(1, len(df_names) + 1)
    xticks = []  # stores the positions where axis ticks shall be
    i = 0  # counter to keep track of the feature names

    if feats is None:
        feats = zipper.keys()

    for feat in feats:
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
            save_file = os.path.join(save_folder, zipper.keys()[i] + ".png")
            fig.savefig(save_file)
        else:
            plt.show()
        # increase i to process next feature
        i += 1


def feat_venn_diagram(dfs, df_names):
    """
    Plots a venn diagram illustrating the overlap in features between the datasets.
    :param dfs: List of dataframes
    :return:
    """
    feat_set = get_feature_sets(dfs)


    # plot overlap as venn diagram
    if len(dfs) == 2:
        # set variables needed to assign new color scheme
        colors = ["blue", "green"]
        ids = ["A", "B"]
        v = mv.venn2(feat_set, set_labels=df_names)

        for df_name, color in zip(ids, colors):
            v.get_patch_by_id(df_name).set_color(color)

        # create lines around circles
        circles = mv.venn2_circles(feat_set)
        # reduce line width
        for c in circles:
            c.set_lw(1.0)

        plt.title("Feature Overlap")

    if len(dfs) == 3:
        # set variables needed to assign new color scheme
        colors = ["blue", "green", "purple"]
        ids = ["A", "B", "001"]

        v = mv.venn3_unweighted(feat_set, set_labels=df_names)

        # set colors
        for df_name, color in zip(ids, colors):
            v.get_patch_by_id(df_name).set_color(color)

        # create lines around circles
        circles = mv.venn3_circles(subsets=(1, 1, 1, 1, 1, 1, 1))
        # reduce line width
        for c in circles:
            c.set_lw(1.0)

        plt.title("Feature Overlap")