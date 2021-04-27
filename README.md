# STA-663-project

Authors: Yanqin Shen, Gaojia Xu, Nancy Fu

This project implements, discusses, and applied data on the paper Biclustering via Sparse Singular Value Decomposition written by Mihee Lee , Haipeng Shen, Jianhua Z Huang, and J S Marron.

The major algorithm of Sparse Singular Value Decomposition (SSVD) returns rank 1 approximation of the original data matrix, with high-dimensional features, and identifies possible clusters.

It could further provide subsequent layers, which can be obtained by the same method from the residual matrix after the previous layer. 

You can install the package by this command line: `pip install git+https://github.com/yanqinshen/STA-663-project.git@main`.

The functions include: `SSVD`, `SSVD_numba`, and `get_plot`

You can load the function by the following commands:

`from SSVD.SSVD_numba import SSVD_numba`

`from SSVD.SSVD import SSVD`

`from SSVD.get_plot import get_plot`

# SSVD

SSVD obtains rank 1 approximation of the original data matrix X
Inputs are:

1. X = matrix to be decomposed
2. tol = stopping criteria for convergence
3. gamma = weight parameters
4. max_it = maximum # of iterations
Output: U, S, V, number of iteration

# get_plot

get_plot function plots the rank 1 approximation result from SSVD_numba function.

Inputs are: 
1. u, s, v = return values from SSVD_numba function
2. clusters= the vector of subgroups assigned originally. 
3. dismiss = index determined to be dicarded.
Output: The heatmap of rank 1 approximation.
Details can be found in section 5: applications to simulated data sets


### Reference

Lee, M., Shen, H., Huang, J., and Marron, J. (2010). Biclustering via sparse singular value decomposition. Biometrics 66, 1087-1095.
