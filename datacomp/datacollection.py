# -*- coding: utf-8 -*-

import pandas as pd
import matplotlib_venn as mv
import matplotlib.pylab as plt
import os
import warnings

from pymatch.Matcher import Matcher
from collections import UserList, Counter
from sklearn.cluster import AgglomerativeClustering
from .stats import test_cat_feats, test_num_feats, p_correction
from .utils import construct_formula, calc_prog_scores


def create_datacol(df, categorical_feats, groupby, df_names=None, exclude_classes=[], rel_cols=None):
    """
    Creates a DataCollection from a DataFrame by grouping and splitting it according to values stated in the groupby
    column.

    :param df: Pandas DataFrame object
     :param df_names: List of the dataframe names
    :param categorical_feats: List of feautres which are categorical
    :param groupby: Column name of the column by which the dataset shall be splitted
    :param exclude_classes: A value present in the groupby column can be specified here. All entries containing that
    value will not be included into the DataCollection.
    :param rel_cols: A list of feature names can be given to consider only those features of the datasets. Other columns
    will be excluded from the DataCollection.
    :return: DataCollection object
    """

    dfs = []

    grouping = df.groupby(groupby)

    # split dataframe groups and create a list with all dataframes
    for name, grp in grouping:
        # skip class if it should be excluded
        if name in exclude_classes:
            continue

        df = grouping.get_group(name)[::]

        # consider all columns as relevant is no rel_cols given.
        if rel_cols is None:
            rel_cols = list(df)

        # consider the relevant columns
        dfs.append(df[rel_cols])

    # extract dataset names from groupby column if none are given
    if df_names is None:
        df_names = list(sorted(grouping.groups.keys()))

    return DataCollection(dfs, df_names, categorical_feats)

def get_data(paths, df_names, categorical_feats, groupby=None, exclude_classes=[], rel_cols=None, sep=","):
    """
    This function will load the data and create a DataCollection object. It takes either a list of paths to the datasets
    of a single path as string. If only a single path is given, the given dataset will be split into different datasets
    based on the value in the groupby column.

    :param paths: List of paths to the different datasets or just a single path to one dataset. If only single pa
    :param df_names: List of the dataframe names
    :param categorical_feats: List of feautres which are categorical
    :param groupby: Column name of the column by which the dataset shall be splitted
    :param exclude_classes: A value present in the groupby column can be specified here. All entries containing that
    value will not be included into the DataCollection.
    :param rel_cols: A list of feature names can be given to consider only those features of the datasets. Other columns
    will be excluded from the DataCollection.
    :param sep: Separator for .csv files. Can be changed e.g. to "\t" is file is tab separated.
    :return: DataCollection object
    """

    def _load_data(path, sep=sep):
        """small function to load according to the dataformat. (excel or csv)"""
        filename, file_extension = os.path.splitext(path)

        if file_extension in [".csv", ".tsv"]:
            df = pd.read_csv(path, index_col=0, sep=sep)
        else:
            df = pd.read_excel(path, index_col=0)

        return df

    # initialize list to store dataframes in
    dfs = []

    # Handle single path input
    if groupby and (len(paths) == 1 or isinstance(paths, str)):

        # load data depending on if the single path is given in a list of as string
        if isinstance(paths, str):
            data = _load_data(paths, sep)
        elif isinstance(paths, list):
            data = _load_data(*paths, sep)
        else:
            raise ValueError("It seems like the input was a single path. Please input path as string or inside a list.")

        grouping = data.groupby(groupby)

        # split dataframe groups and create a list with all dataframes
        for name, grp in grouping:
            # skip class if it should be excluded
            if name in exclude_classes:
                continue

            df = grouping.get_group(name)[::]

            # consider all columns as relevant is no rel_cols given.
            if rel_cols is None:
                rel_cols = list(df)

            # consider the relevant columns
            dfs.append(df[rel_cols])

    # Handle multiple paths input
    elif len(paths) > 1:
        for path in paths:
            df = _load_data(path)
            dfs.append(df)

    return DataCollection(dfs, df_names, categorical_feats)


class DataCollection(UserList):
    """
    A class representing the Collection of datasets, that shall be compared. Datasets are stored list like in self.data
    and the DataCollection class can be used as if it is a list. Provides functions to reduce the dataframes to specific
    subsets of features or to compare the feature ranges of the two dataframes to each other.
    """

    def __init__(self, df_list, df_names, categorical_feats, numerical_feats=None):
        """
        Initialize DataCollection object.

        :param df_list: List containing pandas.DataFrames
        :param df_names: List of strings containing the names of the datasets in df_list
        :param categorical_feats: List of categorical features
        """
        super().__init__(df_list)

        # check if there is an empty dataframe in the datacollection
        if True in {df.empty for df in self}:
            warnings.warn("One of the dataframes in the DataCollection is empty!", UserWarning)

        self.df_names = df_names
        self.categorical_feats = categorical_feats
        # get numerical features if not given
        if numerical_feats is None:
            self.numerical_feats = self.get_numerical_features(self.categorical_feats)
        else:
            self.numerical_feats = numerical_feats


    def get_numerical_features(self, categorical_feats):
        """
        Given the categorical features, this function will return the features being non-categorical.

        :param categorical_feats:
        :return: Set of numerical features
        """
        common_feats = self.get_common_features()
        return list(set(common_feats).difference(categorical_feats))


    def create_zipper(self, feats=None):
        """
        Create a Dictionary containing the values of the same features per dataset in one list.
        featname : (df1_feat1, df2_feat1, df3_feat1)

        :param feats:
        :return:
        """
        if feats is None:
            feats = list(self[0])

        df_feats = []

        for df in self:
            df_feats.append([list(df[feat].dropna()) for feat in feats])

        zip_values = zip(*df_feats)
        zipper = dict(zip(feats, zip_values))
        return zipper

    def print_number_of_entities(self, entity_col):
        """
        Prints the number of entities per dataset.

        :param entity_col: Column containing the entity identifiers
        :return:
        """
        for df in self:
            print("# of entities: ", len(df[entity_col].unique()))

    def reduce_dfs_to_value(self, col, val):
        """
        Keep only the rows in the dataframes where a specific column holds a specific value.

        :param col: Specific column in which the value should be found.
        :param val: Specific value that should be present in the column.
        :return:
        """
        # create list with reduced dataframes
        reduced_dfs = []

        for df in self:
            indices = df[df[col] == val].index
            reduced_dfs.append(df.loc[indices])

        return DataCollection(reduced_dfs, self.df_names, self.categorical_feats, self.numerical_feats)

    def reduce_dfs_to_feat_subset(self, feat_subset):
        """
        Manipulate the DataCollection that the dataframes inside only contain a subset of the original features.

        :return: List of dataframes where the features are identical
        """
        reduced_dfs = []

        for df in self:
            reduced_dfs.append(df.loc[:, feat_subset])

        # keep only categorical feats that are present in the feat_subset
        categorical_feats = list(set(self.categorical_feats).intersection(feat_subset))

        return DataCollection(reduced_dfs, self.df_names, categorical_feats)

    def get_feature_sets(self):
        """
        Creates a list of sets, where in each set the a ll the feature names of one dataframe are stored respectively.

        :return: List of sets. Each set contains the feature names of one of the dataframes
        """
        # Create list containing features per dataframe as sets
        return [set(df) for df in self]

    def get_common_features(self, exclude=None):
        """
        Creates a set of the common features shared between dataframes.

        :param exclude: List of features which shall be exluded from the common features.
        :return: set of common features across the dataframes
        """
        feats = self.get_feature_sets()

        common_feats = set.intersection(*feats)

        if exclude:
            for feat in exclude:
                common_feats.remove(feat)

        return list(common_feats)

    def get_feature_differences(self):
        """
        Assess differences in features across the dataframes. A dictionary will be created where the dataframe
        combination which is compared serves as key and the value is a set with the differences across them.

        :return: Dictionary where dataframe combination is the key and the value is a corresponding set of
        non overlapping features.
        """
        feats = self.get_feature_sets()

        # create a dictionary storing the features which are distinct
        diff_dict = dict()

        # compare each dataset against each and collect differences
        for i in range(len(feats)):
            for j in range(i + 1, len(feats)):
                # take union from differences
                diff_dict[i, j] = feats[i].difference(feats[j]).union(feats[j].difference(feats[i]))

        return diff_dict

    def create_value_set(self, col):
        """
        Creates a set of the combined dataframe values present in a specific column.

        :param col: Column which contains the values that should be put into a set.
        :return: Set containing all values represented in column.
        """

        value_set = set()

        for df in self:
            value_set.update(df[col])
        return value_set

    def combine_dfs(self, label_name, feat_subset=None, cca=False, save_path=None, labels=None):
        """
        Will create a combined dataframe in which labels are assigned depending on the dataset membership.
        The resulting dataframe will be saved under save_path and can be used for propensity_score_matching.

        :param label_name: Name of the label column, that will be created.
        :param feat_subset: List of feature names. All features not in the list will be excluded.
        :param cca: If true only complete cases are kept (cases where all features are non missing values)
        :param save_path: Path where to save the combined dataset.
        :param labels: List of labels that shall be assigned to the datasets in the label column. If none are given the
        first dataset will get a 1 and all the others 0.
        :return: pandas.DataFrame object storing a concatinated version of the DataCollection datasets.
        """

        # reduce dfs to feat_subset
        if feat_subset:
            reduced_datcol = self.reduce_dfs_to_feat_subset(feat_subset)
        else:
            reduced_datcol = self[::]

        # create labels; set label for first df to 1 all others to 0
        if labels is None:
            labels = [1] + [0 for i in range(len(self) - 1)]

        # add labels to dataframes
        for i in range(len(labels)):
            reduced_datcol[i][label_name] = labels[i]

        # drop all non complete cases
        if cca:
            for i in range(len(labels)):
                reduced_datcol[i].dropna(inplace=True)

        # combine datasets
        combined_df = pd.concat(reduced_datcol)

        # save them under save_path if save_path is given
        if save_path:
            combined_df.to_csv(save_path)

        return combined_df

    ## Stats

    def analyze_feature_ranges(self, include=None, exclude=None, verbose=True):
        """
        This function can be used to compare all features easily. It works as a wrapper for the categorical and
        numerical feature comparison functions.

        :param include: List of features that should be considered for the comparison solely.
        :param exclude: List or set of features that should be excluded from the comparison.
        :return: pandas.Dataframe showing the p-values and corrected p-values of the comparison
        """

        # create zipper
        zipper = self.create_zipper()
        # create dictionary that will store the results for feature comparison
        p_values = dict()

        cat_feats = self.categorical_feats[::]
        num_feats = self.numerical_feats[::]

        # delete label if given
        if exclude:
            for feat in exclude:
                del zipper[feat]
            # update feature lists
            cat_feats = set(cat_feats).difference(exclude)
            num_feats = set(num_feats).difference(exclude)

        if include:
            cat_feats = set(cat_feats).intersection(include)
            num_feats = set(num_feats).intersection(include)

        # test features:
        p_values.update(test_cat_feats(zipper, cat_feats))
        p_values.update(test_num_feats(zipper, num_feats))

        # test numerical features
        results = p_correction(p_values)

        if verbose:
            print("Fraction of significantly deviating features:",
                  str(results["signf"].sum()) + "/" + str(len(results["signf"])))

        return results.sort_values("signf")

    ## Clustering

    def hierarchical_clustering(self, label=None, str_cols=None, return_data=False):
        """
        Performs an agglomerative clustering to assign entites in the datasets to clusters and evaluate the distribution
        of dataset memberships across the clusters. Outcome will be the cluster purity w.r.t. the dataset membership
        labels and the confusion matrix, listing how many entities out of which dataset (rows) are assigned to which
        cluster (columns).

        :param label: Column name of the column that should store the dataset membership labels. If None is given, a
        column named "Dataset" will be created and labels from 1 to number of datasets will be assigned as labels.
        :param str_cols: List of features where the values are non numeric. Must be excluded for clustering.
        :return: Cluster purity, Confusion matrix
        """

        def _confusion_matrix(data, label):
            """ """
            confusion_matrix = dict()

            # count for each label
            for dataset_nr in data[label].unique():
                # select subset out of dataframe
                dataset = data[data[label] == dataset_nr]

                # count occurences
                c = Counter(dataset["Cluster"])

                # get rid of NaNs
                c = {key: c[key] for key in c if not pd.isnull(key)}

                # add to confusion matrix
                confusion_matrix[dataset_nr] = c

            return pd.DataFrame(confusion_matrix).transpose()

        def calculate_cluster_purity(confusion_matrix):
            """ """
            return confusion_matrix.max().sum() / confusion_matrix.values.sum()

        if label is None:
            label = "Dataset"

        # create labels for the datasets
        labels = range(1, len(self) + 1)

        # pre-process data to allow for clustering
        cl_data = self.combine_dfs(label, labels=labels)
        num_datasets = len(cl_data[label].unique())

        # exclude string columns if given
        if str_cols:
            cl_data.drop(str_cols, axis=1, inplace=True)

        # make CCA
        cl_data.dropna(inplace=True)

        # create model for clustering and fit it to the data
        model = AgglomerativeClustering(num_datasets)
        cl_labels = model.fit_predict(cl_data)

        # set label column
        cl_data["Cluster"] = cl_labels
        confusion_m = _confusion_matrix(cl_data, label)

        if return_data:
            return calculate_cluster_purity(confusion_m), confusion_m, cl_data

        # calculate datasets distributions across clusters
        return calculate_cluster_purity(confusion_m), confusion_m

    ## longitudinal

    def create_progression_tables(self, feat_subset, time_col, patient_col, method, bl_index):
        """
        Creates a new datacollection object which now stores the feature values not as absolute numbers but "progression
        scores". The feature values of patients are normalized to their baseline values using either the "visit to
        baseline" ratio or a z-score normalized to baseline.

        :param feat_subset: List containing feature names, which shall be included into the new progression score
        datacollection.
        :param time_col: Name of the column in which the time information is stored. e.g. Months, Days, Visit number
        :param patient_col: Name of the column in which the patient identifiers connecting the separate visits are
        stored.
        :param method: String indicating which progression score shall be calculated. z-score ("z-score") or
        ratio of baseline ("robl")
        :param bl_index: Value representing the baseline measurement in the time column.
        :return: DataCollection storing the progression scores for the features
        """

        prog_dfs = []

        for df in self:
            patients = df[patient_col]

            # create dataframe copy to keep from alternating original dataframe
            prog_df = df[feat_subset][::]

            for feat in feat_subset:

                for patient in patients.unique():
                    # collect values for sinlge patient
                    pat_inds = df[df[patient_col] == patient].index
                    # create value series storing the values of a patient
                    values = df.loc[pat_inds, feat]
                    values.index = df.loc[pat_inds, time_col]

                    # calculate scores for patient and reindex to merge back into dataframe copy
                    scores = calc_prog_scores(values, bl_index, method)

                    # if only NaN has been returned as score set patients progression to nan at all visits
                    if type(scores) != pd.Series:
                        prog_df.loc[pat_inds, feat] = scores

                    else:  # input normal progression scores for visits
                        scores.index = pat_inds
                        prog_df.loc[pat_inds, feat] = scores

            # get columns from original dataframe to concatinate them to resulting DF
            concat_columns = df[[patient_col, time_col]]
            prog_df = pd.concat([concat_columns, prog_df], join="outer", axis=1)

            # add prog_df to list
            prog_dfs.append(prog_df)

        # keep track of which categorical features are still in the collection
        categorical_feats = list(set(self.categorical_feats).intersection(feat_subset))

        return DataCollection(prog_dfs, self.df_names, categorical_feats)

    def analyze_longitudinal_feats(self, time_col, bl_index, include=None, exclude=None):
        """
        Performs the longitudinal comparison. For each timepoint the progression scores of all variables will be
        compared.

        :param time_col: Column name of the column storing the time dimension
        :param bl_index: Value of the time column which refers to the baseline
        :param include: List of feature names which shall be considered in the comparison
        :param exclude: List of feature names which shall be excluded in the comparison
        :return: pandas.DataFrame storing the results of the comprasion, List of dataframes that have the progression
        scores.
        """

        # dict to collect p_values
        p_values = dict()
        # dict to collect dataframes reduced to only one time point. time point will be the key to the dataframe
        time_dfs = dict()

        # create a set of all time_points present in the dataframes
        time_points = self.create_value_set(time_col)
        time_points.remove(bl_index)

        # add time to exclude features if not in there yet
        if exclude:
            exclude = set(exclude)
            exclude.add(time_col)
        else:
            exclude = {time_col}

        # for each timepoint collect the data and compare the data
        for time in time_points:

            # set filterwarnings to error to catch warning, that one dataframe is empty and jump to next timepoint
            with warnings.catch_warnings():
                warnings.filterwarnings("error", category=UserWarning)
                # catch Warning that one dataframe is empty
                try:
                    time_point_datacol = self.reduce_dfs_to_value(time_col, time)
                    time_dfs[time] = time_point_datacol

                    p_values[time] = time_point_datacol.analyze_feature_ranges(exclude=exclude, include=include,
                                                                               verbose=False)
                # skip time point if only values in one dataset are available
                except UserWarning:
                    continue

        # concatinate dataframes of single time points and sort by feature name
        long_result_table = pd.concat(p_values).swaplevel(0, 1).sort_index()

        return long_result_table, time_dfs

    ## propensity score matching

    def qc_prop_matching(self, rel_cols, label):
        """
        Evaluates the need for a propensity score matching and can be used to quality control a propensity score matched
        population. Will train classifiers and create a plot.

        :param rel_cols: relevant columns
        :param label: Label or class which should be regressed. (cohort1/cohort2, case/control, treatment/untreated etc.)
        """

        cols = rel_cols[::]

        # create reduced copies of the dataframes for propensity score quality control
        qc_dfs = []
        for df in self:
            qc_dfs.append(df[cols])

        # exclude label if included into columns
        if label in cols:
            cols.remove(label)

        # construct formula
        formula = construct_formula(label, cols)
        print(formula)

        # create Matcher
        m = Matcher(*qc_dfs, yvar=label, formula=formula)
        # train classifier to asses predictability
        m.fit_scores(balance=True, nmodels=10)
        # calculate and visualize propensity scores
        m.predict_scores()
        m.plot_scores()

    ## Visualization

    def feat_venn_diagram(self, save_path=None):
        """
        Plots a venn diagram illustrating the overlap in features between the datasets.

        :param save_path: Path where to save the venn diagram
        :return:
        """
        feat_set = self.get_feature_sets()

        # Plotting when two datasets are compared
        if len(self) == 2:
            # set variables needed to assign new color scheme
            colors = ["blue", "green"]
            ids = ["A", "B"]
            # create circles
            v = mv.venn2(feat_set, set_labels=self.df_names)
            # create lines around circles
            circles = mv.venn2_circles(feat_set)

        # Plotting when three datasets are compared
        elif len(self) == 3:
            # set variables needed to assign new color scheme
            colors = ["blue", "green", "purple"]
            ids = ["A", "B", "001"]
            # create cirlces
            v = mv.venn3_unweighted(feat_set, set_labels=self.df_names)
            # create lines around circles
            circles = mv.venn3_circles(subsets=(1, 1, 1, 1, 1, 1, 1))

        else:
            raise ValueError("Too many datasets in DataCollection. Venn diagram only supported up to 3 datasets.")

        # set colors for the circles in venn diagram
        for df_name, color in zip(ids, colors):
            v.get_patch_by_id(df_name).set_color(color)

        # reduce line width around circles
        for c in circles:
            c.set_lw(1.0)

        plt.title("Feature Overlap")

        if save_path:
            plt.savefig(save_path)
        else:
            plt.show()
