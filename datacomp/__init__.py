from .datacollection import DataCollection, get_data

from .stats import p_correction, test_num_feats, test_cat_feats, manova

from .visualization import bp_all_features, bp_single_features, bp_all_sig_feats

from .utils import get_sig_feats, construct_formula

from .prop_matching import create_prop_matched_dfs

name = "datacomp"
