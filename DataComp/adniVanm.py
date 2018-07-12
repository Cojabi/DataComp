import pandas as pd

rename_anm = {"ADAS_COG":"ADAS13", "CDR_SOB":"CDRSB",
               "Sex":"PTGENDER", "Age":"AGE", "APOE":"APOE4",
               "MMSE":"MMSE", "Fulltime_Education_Years":"PTEDUCAT"}

rename_adni = {"DX.bl":"Diagnosis"}

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

def make_anm_adni_comparable(anm, adni):

    ## ANM

    anm.rename(columns=rename_anm, inplace=True)

    # select only baseline
    anm = anm[anm["Visit"] == 1]
    # get rid of patients with strange or non Diagnosis
    anm = anm[~(anm["Diagnosis"] == "Other")]
    anm.dropna(subset=["Diagnosis"], inplace=True)

    anm["APOE4"] = anm["APOE4"].apply(format_apoe)
    anm["Cohort"] = 1

    ## ADNI

    # rename columns
    adni.rename(columns=rename_adni, inplace=True)

    # get rid of columns which would lead to a wrong impression in comparison results due to duplication
    unwanted_cols = ["DX", "RAVLT.learning", "RAVLT.forgetting", "RAVLT.perc.forgetting", "EcogPtMem", "EcogPtLang",
    "EcogPtVisspat", "EcogPtPlan", "EcogPtOrgan", "EcogPtDivatt", "EcogPtTotal", "EcogSPMem", "EcogSPLang",
    "EcogSPVisspat", "EcogSPPlan", "EcogSPOrgan", "EcogSPDivatt", "FLDSTRENG", "FSVERSION", "Ventricles",
    "Hippocampus", "WholeBrain", "Entorhinal", "Fusiform", "MidTemp", "ICV"]

    unwanted_cols += [x for x in adni.columns if ".bl" in x]
    adni.drop(unwanted_cols, axis=1, inplace=True)

    # select only baseline
    adni= adni[adni["VISCODE"] == "bl"][::]

    adni["Cohort"] = 0
    adni["Diagnosis"] = adni["Diagnosis"].apply(format_diagnosis)

    # drop SMC patients due to lack of SMC diagnosis in ANM
    adni = adni[~(adni["Diagnosis"] == "SMC")]