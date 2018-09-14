Pre-processing The Datasets
===========================

For DataComp to work properly, some standard pre-processing needs to be performed on the datasets. Most commonly these
steps are taken anyway, since they allow for interoperability across, and the comparison of, the datasets:
- Entities should be represented in the rows
- Feature/variables should be stored in columns
- Feature names should be equivalent between the datasets - semantically equal features should bear the same name.
- Feature values should be represented in the same way (e.g. same dummy variable coding, same strings for categories)
- Any normalization/imputation or similar approaches should be carried out in the same manner on the same features
(except if you want to compare the if different methods lead to different results)

Features that are named differently will be treated as non-common features.