# -*- coding: utf-8 -*-

import matplotlib.pylab as plt
import numpy as np
import os

plt.style.use('ggplot')


"""Muss noch nen colorschema bekommen plus legende, damit man die verschiedenen dfs unterscheiden kann."""
def all_features_bp(num_zipper, save=None):
    """
    Plots boxplots for all features and all dfs into one figure
    :param num_zipper: zipper dict, that contains numerical variables. For each key the value is a list containing x
    lists (the values of the features in the x dataframes)
    :param save: a path where to save the figure to
    :return:
    """
    fig = plt.figure()
    ax = plt.axes()

    add_value = len(dfs) + 1  # is used to define the positons of the boxplots
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


def single_feature_bp(zipper, df_names, feats=None, save_folder=None):
    """
    Creates one boxplot figure per feature
    :param zipper: zipper dict, that contains numerical variables. For each key the value is a list containing x
    lists (the values of the features in the x dataframes)
    :param df_names: names of the datasets to label figures accordingly
    :param feats: a list of features for which a plot shall be made. One plot per feature
    :param save_folder: a path to a directory where to store the figures.
    :return:
    """

    positions = range(1, len(dfs) + 1)
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