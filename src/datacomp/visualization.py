# -*- coding: utf-8 -*-

import os
from itertools import compress

import matplotlib.patches as mpatches
import matplotlib.pylab as plt
import numpy as np
import seaborn as sns

from .utils import get_sig_feats, make_ticks_int

# load plot style
plt.style.use('ggplot')


def plot_sig_num_feats(datacol, sig_df, feat_subset=None, boxplot=True, kdeplot=True, save_folder=None):
    """
    Plots boxplots for each significant feature to allow for visual comparison.

    :param sig_df: Dataframe storing the p_values, corrected p_values and a boolean if significant or not. \
    Is provided as outcome of the p_correction or analyze_feature_ranges function
    :param feat_subset: List of a subset of the features. Only for the mentioned features, a plot will be created.
    :param boxplot: Flag that indicates if boxplot figures shall be created
    :param kdeplot: Flag that indicates if kde plot figures shall be created
    :param save_folder: Path to a folder in which all the plots shall be saved
    :return:
    """
    # get significant features
    sig_feats = get_sig_feats(sig_df)

    # create zipper
    sig_zipper = datacol.create_zipper(sig_feats)

    # work with numerical features
    if feat_subset:
        num_feats_to_plot = set(datacol.numerical_feats).intersection(sig_feats).intersection(feat_subset)
    else:
        num_feats_to_plot = set(datacol.numerical_feats).intersection(sig_feats)

    # plot boxplots
    if boxplot:
        all_feature_boxplots(sig_zipper, datacol.df_names, feat_subset=num_feats_to_plot, save_folder=save_folder)
    # plot kde plots
    if kdeplot:
        all_feature_kdeplots(sig_zipper, datacol.df_names, feat_subset=num_feats_to_plot, save_folder=save_folder)


def all_feature_boxplots(zipper, df_names, feat_subset=None, colors=None, legend=True, fig_size=None, save_folder=None):
    """
    Creates one boxplot figure per feature.

    :param zipper: Zipper dict that contains numerical variables. For each key the value is a list containing one list \
    for each dataset covering the respective values.
    :param df_names: names of the datasets to label figures accordingly
    :param feat_subset: a list of features for which a plot shall be made. One plot per feature. If None all features \
    will be considered.
    :param colors: Iterable storing the colors that shall be used for visualizing. One color per dataset in the \
    DataCollection.
    :param legend: Boolean if legend shall be created. True by default.
    :param fig_size: List or tuple containing the figure size. Default is [8, 6]
    :param save_folder: a path to a directory where to store the figures.
    :return:
    """
    # set colors
    if colors is None:
        colors = ["#1f77b4", "#17becf", "#e8a145", "#71ea20"]

    # set figure size
    if not fig_size:
        fig_size = [8, 6]

    # calculate positions for boxplots
    positions = range(1, len(df_names) + 1)

    if feat_subset is None:
        feat_subset = zipper.keys()

    for feat in feat_subset:
        # create new figure
        plt.figure(figsize=fig_size)
        ax = plt.axes()

        for df_feature, color, position in zip(zipper[feat], colors, positions):
            # set color and linewidth of boxplots
            box_properties = dict(linewidth=1.6, color=color)
            plt.boxplot(df_feature, positions=[position], widths=0.6,
                        boxprops=box_properties, whiskerprops=box_properties,
                        capprops=box_properties)


        # configure axes ticks, limits and labels
        plt.xlim(0, np.max(positions) + 1)
        ax.set_xticks(positions)
        ax.set_xticklabels(df_names)
        ax.tick_params(labelsize="large")
        plt.ylabel("Feature value", fontsize=12.5)
        # set title
        plt.title(feat)


        # create legend
        if legend:
            create_legend(df_names, colors)

        if save_folder:
            save_file = os.path.join(save_folder, feat + "_boxplot.png")
            plt.savefig(save_file, dpi=500, bbox_inches="tight")
            plt.clf()
        else:
            plt.show()


def all_feature_kdeplots(zipper, df_names, feat_subset=None, colors=None, save_folder=None):
    """
    Plots distribution plots for each dataframe in one figure.

    :param zipper: Zipper dict that contains numerical variables. For each key the value is a list containing one list \
    for each dataset covering the respective values.
    :param df_names: Iterable containing the names of the datasets.
    :param feat_subset: List of a subset of the features. Only for the mentioned features, a plot will be created.
    :param colors: Iterable storing the colors that shall be used for visualizing. One color per dataset in the \
    DataCollection.
    :param save_folder: Path to a folder in which the plots shall be saved
    :return:
    """
    # set colors
    if colors is None:
        colors = ["#1f77b4", "#17becf", "#e8a145", "#71ea20"]

    # if no feature subset is provided, consider all features
    if feat_subset is None:
        feat_subset = zipper.keys()

    for feat in feat_subset:
        # include feature distributions of all dataframes into plot
        for dataset_feature, color in zip(zipper[feat], colors):
            sns.distplot(dataset_feature, hist=False, color=color, kde_kws={"shade": True})

            # set title
            plt.title(feat)
            plt.xlabel("Feature value")
            create_legend(df_names, colors)

        if save_folder:
            save_file = os.path.join(save_folder, feat + "_kdeplot.png")
            plt.savefig(save_file, dpi=300)
            plt.clf()
        else:
            plt.show()


def plot_sig_cat_feats(datacol, sig_df, feat_subset=None, save_folder=None):
    """
    Plots boxplots for each significant feature to allow for visual comparison.

    :param sig_df: Dataframe storing the p_values, corrected p_values and a boolean if significant or not. \
    Is provided as outcome of the p_correction or analyze_feature_ranges function
    :param feat_subset: List of a subset of the features. Only for the mentioned features, a plot will be created.
    :param save_folder: Path to a folder in which the plots shall be saved
    :return:
    """
    # get significant features
    sig_feats = get_sig_feats(sig_df)

    # create zipper
    sig_zipper = datacol.create_zipper(sig_feats)

    # work with numerical features
    if feat_subset:
        cat_feats_to_plot = set(datacol.categorical_feats).intersection(sig_feats).intersection(feat_subset)
    else:
        cat_feats_to_plot = set(datacol.categorical_feats).intersection(sig_feats)

    # plot countplots
    countplot_single_features(datacol, feat_subset=cat_feats_to_plot, save_folder=save_folder)


def countplot_single_features(datacol, feat_subset=None, normalize=False, save_folder=None):
    """
    Creates countplots with discrete feature split over x axis and number of occurences on y axis.

    :param datacol: DataCollection object
    :param feat_subset: Iterable storing features of interest. Only those feature will be plotted. Default is None.
    :param save_folder: Path to a folder where to save the generated plots.
    :return:
    """

    def normalize_count(df):  # TODO add normalization to the total number of entities in that group.
        """ """
        raise NotImplementedError

    combined = datacol.combine_dfs("Dataset", labels=datacol.df_names)

    if feat_subset is None:
        feat_subset = datacol.categorical_feats

    for feat in feat_subset:

        if normalize:
            normalize_count(combined)  # TODO add normalization to the total number of entities in that group.

        sns.countplot(x=feat, hue="Dataset", data=combined)

        if save_folder:
            save_file = os.path.join(save_folder, feat + ".png")
            plt.savefig(save_file, bpi=300)
            plt.clf()
        else:
            plt.show()


def create_legend(labels, colors):
    """
    Creates a legend for plots using the given colors and labels.

    :param labels: Iterable of labels that shall be used in the legend.
    :param colors: Iterable of colors that shall be used in the legend.
    :return:
    """
    patches = []

    for color, label in zip(colors, labels):
        patches.append(mpatches.Patch(color=color, label=label))

    plt.legend(handles=patches, prop={'size': 12})

## longitudinal plotting

def plot_prog_scores(time_dfs, feat_subset, plot_bp=True, plot_means=True, bp_colors=None, mean_colors=None,
                     mean_sign=None, show_sig=False, p_values=None, x_label=None, save_folder=None):
    """
    Creates one plot per feature displaying the progession scores calculated for the datasets in comparison to each
    other. Plots the distribution of the progression scores as boxplots and the means as a line connecting the different
    time points.

    :param time_dfs: Dictionary storing the calculated progression scores per time point.
    :param feat_subset: List containing feature names for which plots shall be created.
    :param plot_bp: Flag if boxplots shall be plotted.
    :param plot_means: Flag is line connecting the means shall be plotted.
    :param mean_sign: Matplotlib option how data points shall be represented in the plot. Default is "-". This \
    might lead to bad plots if too many missing values are present.
    :param show_sig: Flag if significant deviations shall be marked in the plot.
    :param p_values: Result table from significance testing on the progression scores.
    :param bp_colors: Iterable storing the colors that shall be used for visualizing the boxplots. \
    One color per dataset in the DataCollection.
    :param mean_colors: Iterable storing the colors that shall be used for visualizing the means. \
    One color per dataset in the DataCollection.
    :param x_label: Label for the x axis.
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

        # build dataframe holding only the data for a single time point.
        time_data = [time_dfs[timepoint] for timepoint in sorted(time_dfs.keys())]

        # iterate over the time points
        for time_datcol in time_data:

            # iterate over the number of datasets
            for i in range(len(time_datcol)):

                # if list for dataset exists append to it, if not create list for dataset and append then
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

    def _plot_prog_score_means(means, xticks_positions, mean_sign, colors):
        """
        Plots the means of the progression scores over time.

        :param means: Dictionary storing lists with the means at the time points for each dataset
        :param xticks_positions: List storing the x-axis-tick positions.
        :param mean_sign: Matplotlib option how data points shall be represented in the plot. Default is "-". This \
        might lead to bad plots if too many missing values are present.
        :param colors: Iterable storing the colors that shall be used for visualizing. One color per dataset in the \
        DataCollection.
        :return:
        """

        # set mean sign if none:
        if mean_sign is None:
            mean_sign = "-"

        # plot lines
        for dataset_means, color in zip(means.values(), colors):
            plt.plot(xticks_positions, dataset_means, mean_sign, color=color)

    def _bp_all_timepoints(time_dfs, bp_positions, feat, colors):
        """
        Plot progression score distributions per time point as boxplots.

        :param time_dfs: Dictionary storing the calculated progression scores per time point.
        :param bp_positions: List storing the boxplot positions
        :param feat: Feature name for which plot shall be created.
        :param colors: Iterable storing the colors that shall be used for visualizing. One color per dataset in the \
        DataCollection.
        :return:
        """

        timepoints = list(time_dfs.keys())
        timepoints.sort()

        for time, bp_time_pos in zip(timepoints, bp_positions):

            # prepare data for plotting: extract feature data and exclude NaN's and inf's
            time_data = [timepoint[feat].replace(np.inf, np.nan).dropna() for timepoint in time_dfs[time]]

            # create boxplots, i iterates over the number of different datasets
            for i in range(len(time_data)):

                # check and skip dataset for current time point if no data is available
                if len(time_data[i]) > 1:
                    # create boxplot at specific position
                    bp = plt.boxplot(time_data[i], positions=[bp_time_pos[i]], patch_artist=True, widths=0.6)

                    # change boxplot outline colors
                    for bp_part in ['boxes', 'whiskers', 'fliers', 'caps']:
                        for element in bp[bp_part]:
                            plt.setp(element, color=colors[i])
                else:
                    continue

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

    if bp_colors is None:
        bp_colors = ["#1f77b4", "#17becf", "#e8a145", "#71ea20"]  # TODO change color palette

    if mean_colors is None:
        mean_colors = ["#1799B5", "#00FFFF", "#f7b42e", "#8ff74a"]  # TODO change color palette

    # get the number of dataframes and the dataframe names
    df_names = list(time_dfs.values())[0].df_names
    num_dfs = len(df_names)
    num_timepoints = len(time_dfs.keys())

    # plot one figure for each feature
    for feat in feat_subset:
        # calculate positions on x axis
        bp_positions, xticks_positions = _calc_positions(num_dfs, num_timepoints)

        # plot mean progression
        if plot_means:
            means = _calculate_means_per_timepoint(time_dfs, feat)
            _plot_prog_score_means(means, xticks_positions, mean_sign, mean_colors)

        # plot progression scores at each time point as boxplots
        if plot_bp:
            _bp_all_timepoints(time_dfs, bp_positions, feat, bp_colors)

        if show_sig:
            plot_significances(xticks_positions, p_values)

        # set axis ticks, limits, labels
        ax = plt.axes()
        plt.xlim(0, np.max(bp_positions))

        # try converting x tick labels into integers
        xticks = make_ticks_int(sorted(time_dfs.keys()))

        ax.set_xticklabels(xticks)
        ax.set_xticks(xticks_positions)

        # set label for x-axis and plot title
        if x_label is None:
            x_label = "Longitudinal Value"
        plt.xlabel(x_label)
        plt.ylabel("Feature value")
        plt.title(feat)

        create_legend(df_names, bp_colors)

        if save_folder:
            save_file = os.path.join(save_folder, feat + "prog_score.png")
            plt.savefig(save_file, dpi=300)
            plt.clf()
        else:
            plt.show()


def plot_signf_progs(time_dfs, p_values, plot_bp=True, plot_means=True, mean_sign=None, show_sig=False, save_folder=None):
    """
    Plots progression score plots for each feature that shows significant deviations at some time point.

    :param time_dfs: Dictionary storing the calculated progression scores per time point.
    :param p_values: Result table from significance testing on the progression scores.
    :param plot_bp: Flag if boxplots shall be plotted.
    :param plot_means: Flag is line connecting the means shall be plotted.
    :param mean_sign: Matplotlib option how data points shall be represented in the plot. Default is "-". This \
    might lead to bad plots if too many missing values are present.
    :param show_sig: Flag if significant deviations shall be marked in the plot.
    :param save_folder: Folder in which plots will be saved.
    :return:
    """

    sig_feats = get_sig_feats(p_values)
    plot_prog_scores(time_dfs, sig_feats, plot_bp=plot_bp, plot_means=plot_means, mean_sign=mean_sign,
                     show_sig=show_sig, p_values=p_values, save_folder=save_folder)


def plot_entities_per_timepoint(datacol, time_col, label_name, labels=None, rotate_labels=False, save_path=None):
    """
    Plots a bar plot which shows the number of entities at each point in time for each dataset.

    :param datacol: DataCollection storing the data
    :param time_col: Column name of the column storing the time information.
    :param label_name: Name of the label which should be used to organize the x-axis.
    :param labels: Labels that shall be shown on the legend. The order needs to match the order of the datasets in the\
    DataCollection.
    :param rotate_labels: If True x axis labels will be slightly rotated to prevent overlapping.
    :param save_path: Path to which the plot shall be saved.
    :return:
    """
    if labels is None:
        labels = range(1, len(datacol) + 1)

    # combined data for counting
    combined = datacol.combine_dfs(label_name, labels=labels)
    # create plot
    ax = sns.countplot(x=time_col, hue=label_name, data=combined)
    plt.title("Number of entities per time point")

    if rotate_labels:
        ax.set_xticklabels(ax.get_xticklabels(), rotation=40, ha="right")
        plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300)
    else:
        plt.show()