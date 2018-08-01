import pandas as pd

from collections import Counter

# ANM CONSTANTS
rename_anm = {"ADAS_COG": "ADAS13", "CDR_SOB": "CDRSB",
              "Sex": "PTGENDER", "Age": "AGE", "APOE": "APOE4",
              "MMSE": "MMSE", "Fulltime_Education_Years": "PTEDUCAT"}

## ADNI CONSTANTS
rename_adni = {"DX.bl": "Diagnosis"}

adni_unwanted_cols = ["DX", "RAVLT.learning", "RAVLT.forgetting", "RAVLT.perc.forgetting", "EcogPtMem", "EcogPtLang",
                      "EcogPtVisspat", "EcogPtPlan", "EcogPtOrgan", "EcogPtDivatt", "EcogPtTotal", "EcogSPMem",
                      "EcogSPLang", "EcogSPVisspat", "EcogSPPlan", "EcogSPOrgan", "EcogSPDivatt", "FLDSTRENG",
                      "FSVERSION", "Ventricles", "Hippocampus", "WholeBrain", "Entorhinal", "Fusiform", "MidTemp",
                      "ICV"]


def format_diagnosis(entry):
    if "MCI" in entry:
        return "MCI"
    elif entry == "CN":
        return "CTL"
    else:
        return entry


def format_apoe(entry):
    if pd.isnull(entry):
        return entry
    else:
        return Counter(entry)["4"]


def make_visit_id(entry):
    return entry + "_1"


def make_anm_comparable(anm, cohort_label=True, bl_only=True):

    anm.rename(columns=rename_anm, inplace=True)

    # get rid of patients with strange or non Diagnosis
    anm = anm[~(anm["Diagnosis"] == "Other")]
    anm.dropna(subset=["Diagnosis"], inplace=True)

    anm["APOE4"] = anm["APOE4"].apply(format_apoe)

    # add cohort label
    if cohort_label:
        anm["Cohort"] = 1

    # select only baseline
    if bl_only:
        anm = anm[anm["Visit"] == 1]


def make_adni_comparable(adni, cohort_label=True, bl_only=True):
    # rename columns
    adni.rename(columns=rename_adni, inplace=True)

    # get rid of columns which would lead to a wrong impression in comparison results due to duplication

    adni_unwanted_cols += [x for x in adni.columns if ".bl" in x]
    adni.drop(adni_unwanted_cols, axis=1, inplace=True)

    adni["Diagnosis"] = adni["Diagnosis"].apply(format_diagnosis)

    # drop SMC patients due to lack of SMC diagnosis in ANM
    adni = adni[~(adni["Diagnosis"] == "SMC")]

    if cohort_label:
        adni["Cohort"] = 0

    # select only baseline
    if bl_only:
        adni = adni[adni["VISCODE"] == "bl"][::]
