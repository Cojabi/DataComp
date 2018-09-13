# -*- coding: utf-8 -*-

from .datacollection import DataCollection, get_data, create_datacol

from .stats import p_correction, test_num_feats, test_cat_feats, manova

from .visualization import bp_single_features, plot_sig_num_feats, plot_prog_scores, feature_kdeplots, \
    plot_entities_per_timepoint, plot_signf_progs, plot_sig_cat_feats

from .utils import get_sig_feats, construct_formula

from .prop_matching import create_prop_matched_dfs, create_prop_matched_dfs_longitudinal

name = "datacomp"
