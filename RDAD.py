#!/usr/bin/env python3
# -*- coding: utf-8 -*-

############################
# 
# This file is a Python script that implements the computation of the Robust Density-Aware Distance 
# (RDAD) function and its topology.
# The theory of the paper is explained in [1].
# A Jypter notebook demonstration is available at
# These codes are written by Chunyin Siu, corresponding author of [1], who may be reached at
# cs2323 [at] cornell.edu
# 
############################


import gudhi as gd
import numpy as np

from scipy.spatial.distance import cdist
from scipy.special import rgamma
from sklearn.neighbors import NearestNeighbors

_has_tqdm = True
try: import tqdm
except ImportError: _has_tqdm = False

_has_numba = True
try: from numba import jit
except ImportError: _has_numba = False


def generate_grid(**kwargs):
    # construct a 2D grid from defining quantities
    # input: either (xlim, ylim, dx, dy) or (Xgrid, Ygrid)
    # xlim[0] and xlim[1] are the min and max x-coordinates of points in the grid
    # dx is the grid size of the grid in x-direction
    # ylim and dy are similar
    # Xgrid is a matrix of x-coordinates of a grid points, so each column is a constant column
    # Ygrid is similar.
    #
    # output: a dictionary containing all necessary information about the grid
    # the keys are xlim, ylim, dx, dy, Xgrid, Ygrid, vert, nV
    # We only explain vert and nV, because the rest have been explained in the input part.
    # nV: number of vertices in the grid
    # vert: nV x 2 numpy array, each row is a grid point
    
    lim_flag = "xlim" in kwargs.keys()
    if lim_flag == 0 and "Xgrid" not in kwargs.keys():
        raise NameError("Neither xlim nor Xgrid is supplied to generate the grid.")

    keys = [
        ["Xgrid", "Ygrid"],
        ["xlim", "ylim", "dx", "dy"]
        ]
    grid = {key : kwargs[key] for key in keys[lim_flag]}
    grid = _build_grid_from_input_(grid, lim_flag)
    grid = _complete_grid_info_(grid)
    
    return grid

def _build_grid_from_input_(grid, lim_flag):
    if lim_flag:
        grid["xgrid"] = np.arange(grid["xlim"][0], grid["xlim"][1] + grid["dx"], grid["dx"])
        grid["ygrid"] = np.arange(grid["ylim"][0], grid["ylim"][1] + grid["dx"], grid["dx"])        
        grid["Xgrid"], grid["Ygrid"] = np.meshgrid(grid["xgrid"], grid["ygrid"])
    else:
        grid["xgrid"] = grid["Xgrid"][0, :]
        grid["ygrid"] = grid["Ygrid"][:, 0]
        grid["xlim"] = [min(grid["xgrid"]), max(grid["xgrid"])]
        grid["ylim"] = [min(grid["ygrid"]), max(grid["ygrid"])]
        grid["dx"] = grid["xgrid"][1] - grid["xgrid"][0]
        grid["dy"] = grid["ygrid"][1] - grid["ygrid"][0]
    return grid

def _complete_grid_info_(grid):
    grid["vert"] = np.vstack((grid["Xgrid"].flatten(), grid["Ygrid"].flatten())).T
    grid["nV"] = np.shape(grid["vert"])[0]
    
    return grid

def __weighted_dist_to_measure__local_(grid, grid_idx_range, pts, weight, k):
    weighted_d = cdist(grid["vert"][grid_idx_range], pts)*weight #each pt is a col, each grid_pt is a row
    return np.array([np.partition(row, k)[:k] for row in weighted_d])

if _has_numba:
    @jit()
    def _first_k_loop_by_row_(mat, k):
        return [np.partition(row, k)[:k] for row in mat]
else:
    def _first_k_loop_by_row_(mat, k):
        return [np.partition(row, k)[:k] for row in mat]

def _weighted_dist_to_measure_(pts,
                             weight,
                             k,
                             grid,
                             is_divisive = 1,
                             offset = 0,
                             batch_size = np.inf):
    
    if is_divisive: weight = 1/weight
    weight = offset + weight
    
    batch_size = min(grid["nV"], batch_size)
    

    grid_idx = 0
    rwd = np.zeros((grid["nV"], k))
    while grid_idx < grid["nV"]:
        grid_idx_range = np.arange(grid_idx, min(grid["nV"], grid_idx+batch_size))
        weighted_d = cdist(grid["vert"][grid_idx_range], pts)*weight #each pt is a col, each grid_pt is a row
        rwd[grid_idx_range,:] = np.array(_first_k_loop_by_row_(weighted_d, k))
        # rwd[grid_idx_range,:] = np.array([np.partition(row, k)[:k] for row in weighted_d])
        grid_idx += batch_size

    rwd = rwd**2
    rwd = rwd.mean(axis = 1)
    return np.sqrt(rwd)

def _filtration_constant_from_dim_(dim, k_denest, num_points):
    
    if dim <= 3:
        ball_vol = {1: 2,
                    2: np.pi,
                    3: 4 / 3 * np.pi
                    }
        V = ball_vol[dim]
    else:
        V = np.pi ** (dim/2) * rgamma(dim/2 + 1)

    C = V * num_points / k_denest
    C = C ** (1/dim)
    
    return C

def RDAD(points, grid, m_DTM, k_denest = None, scaling_dim = None, weighted_flag = 1, **kwargs):
    
    # compute the RDAD function of a dataset on a grid
    #
    # INPUTS
    # points: N x 2 array, each row is a data point
    # grid: dictionary, output of generate_grid, the grid on which the function is computed
    # m_DTM: double, a real number between 0 and 1 (inclusively), denoising threshold, more smoothing 
    #        when large
    # k_denest: integer at most the number of data points, density estimate computed from the distance 
    #           from the k_denest-nearest neighbor
    #           if not supplied, k_denest is log10(N)^2, where N is the number of data points
    # scaling_dim: dimension of manifold (or CW complex) from which the data points are drawn
    #              if not supplied, scaling_dim is the dimension of the data points (which has to be 2
    #              in this implementation, but modification to 3-dimensional datasets should not be too 
    #              complicated). This should not be problematic even if the data is supported on a lower
    #              dimensional space. See the paper for details.
    # weighted_flag: 1 (default) or 0; RDAD is computed if 1, DTM is computed otherwise
    # kwargs: batch_size: integer; if the grid dataset is too big, they can be processed in batches, and
    #                     batch_size is the maximum size of each batch (default: np.inf)
    #
    # OUTPUT

    # compute parameters
    num_points = points.shape[0]
    k_DTM = int(max(1, np.ceil(num_points * m_DTM)))
    if k_denest is None and weighted_flag == 1:
        k_denest = int(np.ceil(np.log10(num_points)**2))
    
    # compute weight
    if not weighted_flag:
        weight = np.ones(num_points)
    else:
        
        # compute kNN
        nbrs = NearestNeighbors(n_neighbors = k_denest).fit(points)
        kdist, indices = nbrs.kneighbors(points)
        weight = kdist[:, -1]
        
        #scale weight
        if scaling_dim is None:
            scaling_dim = points.shape[1]
        
        C = _filtration_constant_from_dim_(
            dim = scaling_dim,
            k_denest = k_denest,
            num_points = len(weight))
        weight *= C
    
    # compute filtration function on grid
    fil = _weighted_dist_to_measure_(
        pts = points,
        weight = weight,
        k = k_DTM,
        grid = grid,
        **kwargs)
        
    return fil

def compute_persistence(grid, fil):
    #
    # Given a function on a grid, compute its persistent homology
    # grid: output of generate_grid
    # fil: 1D array, ith entry is the filtration function evaluated at
    #      the ith point in the grid in some lexigraphical order
    #      (i.e. grid["Xgrid"].flatten(), grid["Ygrid"].flatten(), 
    #       equivalently, in the list vert)
    #
    # ouput: persistent homology summarized in different ways
    # persistences: a list of pairs, each pair is a homological feature
    #               of the form (dim, (birth, death))
    # pairs: a pair of two lists (A, B)
    #        The first list, A is a list of features that die in finite time;
    #        the second list, B, at infinity
    #        The d-th entry of each of A and B is a numpy array of features of 
    #        dimension d, whose i-th row is the index of the creating (and
    #        killing) cell.
    # vals: filtration value of the creating and killing cells in pairs
    
    st = gd.CubicalComplex(
        dimensions = grid["Xgrid"].T.shape,
        top_dimensional_cells = fil)
    persistences = st.persistence()
    # persistences is a list of pairs of the form (dim, (birth, death))
    
    pairs = st.cofaces_of_persistence_pairs()
    # pairs is a pair of two lists (A, B)
    # The first list, A is a list of features that die in finite time;
    # the second list, B, at infinity
    # The d-th entry of each of A and B is a numpy array of features of 
    # dimension d, whose i-th row is the index of the creating (and
    # killing) cell.
    
    vals = [
                [fil[pairs_of_fixed_dim] for pairs_of_fixed_dim in pairs_of_fixed_type]
                for pairs_of_fixed_type in pairs
                ]
    # vals correspond to the filtration value of the creating and killing
    # cells in pairs
    
    return persistences, pairs, vals

def _subsample_from_sample(points, seed):
    
    rng = np.random.default_rng(seed = seed)
    subsample_points = rng.choice(
        points, len(points), replace=True
    )
    return subsample_points

def compute_confidence_band(points, persistences, grid, m_DTM, k_denest = None, scaling_dim = None, weighted_flag = 1, bootstrap_dim = 1, B = 100, alpha = 0.05, seeds = None, **kwargs):
    #
    # output the confidence band constructed by subsample bootstrapping
    # We draw subsamples from the data points, treat them as new datasets and compute their persistence diagrams
    # We construct the band as 2 times of a large percentile of the bottleneck distances of the empirical persistence diagram and the bootstrap diagrams
    #
    # INPUTS
    # points: N x 2 numpy array, each row is a data point
    # persistences: output of compute_persistence
    # grid, m_DTM, k_denest, scaling_dim, weighted_flag, kwargs: inputs of RDAD
    # bootstrap_dim: dimension of features of question, in particular, bottleneck distance is computed for the persistent diagram of this dimension
    #                 default: 1
    # B: number of bootstrap samples to generate, default: 100
    # alpha: (1 - alpha) is the percentile of the bottleneck distance used to generate the band, default: 0.05
    # seeds: a length-B array of seeds to be used for drawing bootstrap samples
    #        MUST BE DISTINCT
    #        default: randomly generated
    #
    # OUTPUT
    # double, the confidence band width

    if seeds is None:
        rng = np.random.default_rng()
        seeds = rng.uniform(size = B + 10) * min(10000, 100*B)
        seeds = seeds.astype(int)
        seeds = np.unique(seeds)
        seeds = seeds[:B]
    
    persistences = [bd for (dim, bd) in persistences if dim == bootstrap_dim]
    
    bottleneck_distances = []
    #superPD = []
    print("Computing persistence diagrams of bootstrap subsamples.")
    if _has_tqdm: iterator = tqdm.tqdm(range(B))
    else: iterator = range(B)
    for i in iterator:
        subsample_points = _subsample_from_sample(points, seeds[i])
        subsample_fil = RDAD(subsample_points, grid, m_DTM, k_denest = k_denest, scaling_dim = scaling_dim, weighted_flag = weighted_flag, **kwargs)
        subsample_persistences, subsample_pairs, subsample_vals = compute_persistence(grid, subsample_fil)
        subsample_persistences = [bd for (dim, bd) in subsample_persistences if dim == bootstrap_dim]
        #superPD.append(subsample_persistences)
        bottleneck_distance = gd.bottleneck_distance(persistences, subsample_persistences)
        bottleneck_distances.append(bottleneck_distance)
    
    quan = np.quantile(
        bottleneck_distances,
        1 - alpha
        )
    
    print("Done computing persistence diagrams of bootstrap subsamples and the confidence band.")
    
    return 2*quan
    
def get_significant_features(pairs, vals, grid, dim, band):
    #
    # return the coordinates of the killing pixel of features whose persistence exceeds a threshold
    #
    # INPUT
    #
    # pairs, vals: output of compute_persistence
    # grid: output of generate_grid
    # dim: double, dimension of feature of question
    # band: double, persistence threshold
    # 
    # OUTPUT
    # coordinates of killing pixels with large enough persistence 

    pairs = pairs[0]
    bds = vals[0]
    try:

        pairs = pairs[dim]
        bds = bds[dim]
        persistent = bds[:, 1] - bds[:, 0] > band
        pairs = pairs[persistent, :]
        death_cells = pairs[:, 1]

    except IndexError:
        # no feature at all in the given dimension
        pass
    
    x = grid["Xgrid"].flatten()[death_cells]
    y = grid["Ygrid"].flatten()[death_cells]
    
    return np.vstack([x, y]).T