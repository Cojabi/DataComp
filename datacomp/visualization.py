# -*- coding: utf-8 -*-

import matplotlib.pylab as plt
import numpy as np
import os
import seaborn as sns

from .utils import get_sig_feats
from itertools import compress

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
    :param feat_subset: List of a subset of the features. Only for the mentioned features, a plot will be created.
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
    """
    Plots distribution plots for each dataframe in one figure.

    :param zipper:
    :param feat_subset: List of a subset of the features. Only for the mentioned features, a plot will be created.
    :param save_folder: Path to a folder in which the plots shall be saved
    :return:
    """
    # set colors
    colors = ["b", "c", "r"]  # TODO adjust color pallette

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


## longitudinal plotting

def plot_prog_scores(time_dfs, feat_subset, plot_bp=True, plot_means=True, show_sig=False, p_values=None,
                     save_folder=None):
    """
    Creates one plot per feature displaying the progession scores calculated for the datasets in comparison to each
    other. Plots the distribution of the progression scores as boxplots and the means as a line connecting the different
    time points.

    :param time_dfs: Dictionary storing the calculated progression scores per time point.
    :param feat_subset: List containing feature names for which plots shall be created.
    :param plot_bp: Flag if boxplots shall be plotted.
    :param plot_means: Flag is line connecting the means shall be plotted.
    :param show_sig: Flag if significant deviations shall be marked in the plot.
    :param p_values: Result table from significance testing on the progression scores.
    :param save_folder: Folder in which plots will be saved.
    :return:
    """

    def _calculate_means_per_timepoint(time_dfs, feat):
        """
        Calculates the means for each time point for each dataset.

        :param time_dfs: Dictionary storing the calculated progression scores per time point.
        :param feat: Feature name for which the means shall be calculated.
        :return: Dictionary storing lists with the means at the time points for each dataset
        """
        means = dict()

        for time_datcol in time_dfs.values():
            for i in range(len(time_datcol)):

                if i in means.keys():
                    means[i].append(time_datcol[i][feat].replace(np.inf, np.nan).mean())
                else:
                    means[i] = []
                    means[i].append(time_datcol[i][feat].replace(np.inf, np.nan).mean())

        return means

    def _calc_positions(num_dfs, num_time):
        """
        Calculates the x-axis positions of the boxplots and lines.

        :param num_dfs: Number of datasets
        :param num_time: Number of time points
        :return: List storing the boxplot positions; List storing the x-axis-tick positions.
        """
        add_value = num_dfs + 1  # is used to define the positons of the boxplots
        bp_positions = [range(1, add_value)]
        xticks_positions = []  # stores the positions where x axis ticks shall be

        # calculate and store positions
        for i in range(num_time):
            xticks_positions.append(np.mean(bp_positions[i]))
            bp_positions.append([x + add_value for x in bp_positions[i]])

        return bp_positions, xticks_positions

    def _plot_prog_score_means(means, xticks_positions):
        """
        Plots the means of the progression scores over time.

        :param means: Dictionary storing lists with the means at the time points for each dataset
        :param xticks_positions: List storing the x-axis-tick positions.
        :return:
        """
        LN_COLORS = ["#1799B5", "#00FFFF"]  # TODO change color palette

        # plot lines
        for dataset_means, color in zip(means.values(), LN_COLORS):
            plt.plot(xticks_positions, dataset_means, "-", color=color)

    def _bp_all_timepoints(time_dfs, bp_positions, feat):
        """
        Plot progression score distributions per time point as boxplots.

        :param time_dfs: Dictionary storing the calculated progression scores per time point.
        :param bp_positions: List storing the boxplot positions
        :param feat: Feature name for which plot shall be created.
        :return:
        """

        colors = ["#1f77b4", "#17becf", "#d62728"]  # TODO change color palette

        for time, bp_time_pos in zip(time_dfs, bp_positions):

            # prepare data for plotting: extract feature data and exclude NaN's and inf's
            time_data = [timepoint[feat].replace(np.inf, np.nan).dropna() for timepoint in time_dfs[time]]

            # create boxplots
            for i in range(len(time_data)):

                # create boxplot at specific position
                bp = plt.boxplot(time_data[i], positions=[bp_time_pos[i]], patch_artist=True, widths=0.6)

                # change boxplot outline colors
                for bp_part in ['boxes', 'whiskers', 'fliers', 'caps']:
                    for element in bp[bp_part]:
                        plt.setp(element, color=colors[i])

    def plot_significances(xticks_positions, p_values):
        """
        Plot significance marker.

        :param xticks_positions: List storing the x-axis-tick positions.
        :param p_values: Result table from significance testing on the progression scores.
        :return:
        """
        significances = p_values.loc[feat, "signf"]
        sig_ticks = list(compress(xticks_positions, significances))
        y_axis_values = [0 for i in range(len(sig_ticks))]
        plt.plot(sig_ticks, y_axis_values, "*")

    # get the number of dataframes and the dataframe names
    df_names = list(time_dfs.values())[0].df_names
    num_dfs = len(df_names)
    num_timepoints = len(time_dfs.keys())

    # plot one figure for each feature
    for feat in feat_subset:
        # calculate positions on x axis
        means = _calculate_means_per_timepoint(time_dfs, feat)
        bp_positions, xticks_positions = _calc_positions(num_dfs, num_timepoints)

        # plot mean progression
        if plot_means:
            _plot_prog_score_means(means, xticks_positions)

        # plot progression scores at each time point as boxplots
        if plot_bp:
            _bp_all_timepoints(time_dfs, bp_positions, feat)

        if show_sig:
            plot_significances(xticks_positions, p_values)

        # set axes limits, labels and plot title
        ax = plt.axes()
        plt.xlim(0, np.max(bp_positions))
        ax.set_xticklabels(time_dfs.keys())
        ax.set_xticks(xticks_positions)
        plt.title(feat)

        if save_folder:
            save_file = os.path.join(save_folder, feat + "prog_score.png")
            plt.savefig(save_file)
        else:
            plt.show()


def plot_signf_progs(time_dfs, p_values, plot_bp=True, plot_means=True, save_folder=None):
    """
    Plots progression score plots for each feature that shows significant deviations at some time point.

    :param time_dfs: Dictionary storing the calculated progression scores per time point.
    :param p_values: Result table from significance testing on the progression scores.
    :param plot_bp: Flag if boxplots shall be plotted.
    :param plot_means: Flag is line connecting the means shall be plotted.
    :param save_folder: Folder in which plots will be saved.
    :return:
    """

    sig_feats = get_sig_feats(p_values)
    plot_prog_scores(time_dfs, sig_feats, plot_bp=plot_bp, plot_means=plot_means, show_sig=True, p_values=p_values,
                     save_folder=save_folder)


def plot_entities_per_timepoint(datacol, time_col, label_name):
    """
    Plots a bar plot which shows the number of entities at each point in time for each dataset.

    :param datacol: DataCollection storing the data
    :param time_col: Column name of the column storing the time information.
    :param label_name: Name of the label which should be used to organize the x-axis.
    :return:
    """
    combined = datacol.combine_dfs(label_name)
    sns.countplot(x=time_col, hue=label_name, data=combined)
