# DataComp: A python framework to statistically compare multiple datasets

## Description

DataComp can be used to systematically compare and statistically assess differences across multipe datasets. This could for example help to identify suitable training and validation datasets for machine learning protocols, to quality control if and how/where simulated data deviates from real world data or if different sites of a multi site study make similar observations.
It literally compares everything, the only condition is, that the datasets share the same feature names for common features, otherwise those will be handled as not in common.


## Main Features

DataComp supports:
- Evaluating and visualizing the overlap in features across the datasets
- Parametric and non-parametric statistical hypothesis testing to compare the feature value distributions
- (Automatically) visualizing feature distributions of (significantly deviating) features for visual comparison in form of boxplots or overlapping kde plots
- Performing MANOVA to assess how much of an influence features show onto the dataset membership
- Hierarchical clustering of the entities in the datasets to evaluate if dataset membership labels are evenly distributed across clusters or separated
- Normalizing time series data to baseline and statistically comparing the progression of features over time
- Visualization of feature progression over time


## Installation
```
pip install datacomp
```

## Documentation

Still in the works