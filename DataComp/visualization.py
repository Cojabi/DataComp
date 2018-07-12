# -*- coding: utf-8 -*-

import matplotlib.pylab as plt
import numpy as np
import os
import matplotlib_venn as mv

from operator import itemgetter
from .data_functions import get_feature_sets

plt.style.use('ggplot')


def bp_all_sig_feats(df, zipper, df_names, feats=None, save_folder=None):
    """ """

    # grab significant deviances
    sig_entries = df[df["signf"]]
    index_labels = sig_entries.index.labels[0]
    sig_feats = set(itemgetter(index_labels)(sig_entries.index.levels[0]))

    sig_zipper = {x: zipper[x] for x in sig_feats}
    bp_single_feature(sig_zipper, df_names, feats=None, save_folder=None)


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


def bp_single_feature(zipper, df_names, feats=None, save_folder=None):
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
    feats = get_feature_sets(dfs)
    # plot overlap as venn diagram
    if len(dfs) == 2:
        mv.venn2(feats, set_labels=df_names)

    if len(dfs) == 3:
        mv.venn3(feats, set_labels=df_names)
