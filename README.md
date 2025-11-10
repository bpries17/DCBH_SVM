# DCBH_SVM
This repository contains SVMs and the associated code for classifying candidate halos hosting DCBHs in the Renaissance simulations. The results of this work are presented HERE.

## `models` directory
This directory contains the twelve models discussed in the paper. The first set of six are the models developed for different simulation types (`dm`, `gas`, and `star`) and for different feature availability (`main` and `full`). The second set of six are the models in two-dimensional subspaces of the full feature space, considering all combinations of the following four features: metallicity (`metallicity`), Lyman-Werner flux (`LW`), magnitude of the radial gas mass flux (`radial_mass_flux`), and stellar mass (`stellar_mass`).

## `grid_search` directory
This directory contains scripts for running the hyperparameter tuning grid searches. `full` runs on the full feature space, `subset` runs on the feature subsets identified with particular codes, and `subspace` runs on a specific two-dimensional subspace.
