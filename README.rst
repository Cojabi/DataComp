DataComp: A python framework to statistically compare multiple data sets
========================================================================

Description
-----------
DataComp can be used to systematically compare and statistically assess differences across multiple datasets. This
could for example help to identify suitable training and validation datasets for machine learning approaches, to
quality control if, how and where simulated data differs from real world data or if different sites of a multi-site
study make similar observations or show significant differences.

It literally compares everything, the only condition is, that the features that shall be compared share the same column
names, otherwise those will be handled as distinct features.

Main Features
-------------
DataComp supports:

- Evaluating and visualizing the overlap in features across the data sets
- Parametric and non-parametric statistical hypothesis testing to compare feature value distributions
- Visualizing feature distributions of (significantly deviating) features in form of boxplots or overlapping kde plots
  for visual comparison
- Performing MANOVA to assess how much of an influence features show onto the dataset membership
- Hierarchical clustering of the entities in the data sets to evaluate if dataset membership labels are evenly
  distributed across clusters or assigned to distinct clusters
- Normalizing time series data to baseline and statistically comparing the progression of features over time
- Visualization of feature progression over time


Installation
------------
.. code-block:: sh

   pip install datacomp


Documentation
-------------
Full package documentation can be found here_.

.. _here: https://datacomp.readthedocs.io/en/latest/


Application examples
~~~~~~~~~~~~~~~~~~~~
Example notebooks showcasing Datacomp workflows and results on simulated data can be found at DataComp_Examples_:

.. _DataComp_Examples:

- `Cross-sectional Comparison Example`_

.. _Cross-sectional Comparison Example: https://github.com/Cojabi/DataComp_Examples/blob/master/cross-sectional_example.ipynb

- `Longitudinal Comparison Example`_

.. _Longitudinal Comparison Example: https://github.com/Cojabi/DataComp_Examples/blob/master/longitudinal_example.ipynb

