Prerequisites
=============

To ensure proper DataComp functionality some basic assumptions have to be met by the datasets:

- Tabular data formats need to be used (.csv, .tsv, excel etc.)
- Data points / entities are represented as rows
- Features / variables should be columns
- Feature names should be equivalent among the datasets - semantically equal features should bear the same name.
  Features that are named differently will be treated as features that are not in common.
- Feature values should be represented in the same way (e.g. same variable coding, same categories for discrete variables)
- Any data alternations (e.g. normalization) should be carried out the same way on the features to be compared.
  Otherwise they will influence comparison results


Common Errors
-------------

Make sure that numeric feature columns hold solely numeric data and/or missing values (nan's).
String values like for example ">90" must be converted into numerical values other wise errors will occur.