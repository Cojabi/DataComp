.. DataComp documentation master file, created by
   sphinx-quickstart on Thu Sep  6 15:45:33 2018.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

DataComp Documentation
======================

DataComp can be used to systematically compare and statistically assess differences across multiple datasets. This
could for example help to identify suitable training and validation datasets for machine learning approaches, to
quality control if, how and where simulated data differs from real world data or if different sites of a multi-site
study make similar observations or show significant differences.

It literally compares everything, the only condition is, that the features that shall be compared share the same column
names, otherwise those will be handled as distinct features.

Main Features
=============
DataComp supports:

- Evaluating and visualizing the overlap in features across the data sets
- Parametric and non-parametric statistical hypothesis testing to compare feature value distributions
- Visualizing feature distributions of (significantly deviating) features in form of boxplots or overlapping kde plots
  for visual comparison
- Performing a MANOVA to assess how much of an influence features show onto the dataset membership
- Hierarchical clustering of the entities in the data sets to evaluate if dataset membership labels are evenly
  distributed across clusters or assigned to distinct clusters
- Normalizing time series data to baseline and statistically comparing the progression of features over time
- Visualization of feature progression over time

Links
=====
- Versioning on GitHub_
- Documentation on `Read the docs`_
- Distribution via PyPi

.. _GitHub: https://github.com/Cojabi/DataComp
.. _Read the docs: https://datacomp.readthedocs.io/en/latest/)

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   install
   examples
   preprocessing


.. toctree::
   :maxdepth: 2
   :caption: Code Documentation:

   datacollection
   stats
   visualization
   utils
   longitudinal
   prop_matching

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
