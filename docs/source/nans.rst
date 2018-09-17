Missing Data
============

DataComp will handle missing data in several ways if encountered. The amount of data for each variable in each dataset
will be displayed next to the significance test results to evaluate reliability.

Dataset Composition
-------------------
If a single dataframe is turned into a DataCollection based on the values in a single column, the original dataframe is
split into many. If in one of these many dataframes one feature is all missing values (nan's) it will be excluded from
this dataframe.

Statistical Comparison
----------------------

Numerical Features
~~~~~~~~~~~~~~~~~~
In the direct comparison of numerical feature distributions missing values (nan's) will not be taken into account.
(e.g. distribution [1, 2, nan, 3, 3] will be handled as [1, 2, 3, 4])

Categorical Features
~~~~~~~~~~~~~~~~~~~~
If categories are not present at all in one of the samples, they will be added and their observation count will be set
to 0. This can be a problem, since a chi²-test assumes that there should be at least 5 observations per field in the
contingency table. If only one cell is below 5 the assumptions can still be somewhat met but results should be treated
carefully and be checked upon. DataComp will through a warning, when this is encountered.

Longitudinal Entity Drop-out
----------------------------
For the longitudinal comparison it makes sense to visualize the data availability for each dataset per time point.
This can be done using:


.. code-block:: python

    plot_entities_per_timepoint(datacol, time_col, label_name)


An example can be seen here_.

.. _here: https://github.com/Cojabi/DataComp_Examples/blob/master/longitudinal_example.ipynb

For the comparisons at each single time point, missing data is handled as described in the section 'Statistical Comparison'.
