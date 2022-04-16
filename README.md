# RDAD

## What is this?

The Robust Density-Aware Distance (RDAD) function is a mathematical function that highlights small topological features in a dataset. This package contains algorithms that comptue the RDAD function and the persistent homology of its sublevel sets.

## How to cite?



## What's in here?

This repo contains three files other than this `README.md`.

1. `RDAD.py` contains all the functions needed to compute and visualize the RDAD function and its topology.
2. `tutorial.ipynb` is a Jupyter notebook that demonstrate the use of functions in `RDAD.py`.
3. `requirements.txt` is a list of all packages in the virtual environment in which the codes were developed.

## What is needed?

This package utilizes Gudhi 3.4.1, Numpy 1.21.2, Scipy 1.8.0, Sklearn 1.0.2, Matplotlib 3.5.0 and Python 3.9.7.

Tqdm 4.62.3 and Numba 0.55.0 are used to generate progress reports and to speed up the codes respectively. They are optional.

## How to use?

Just copy `RDAD.py` to your directory and import it to your codes. Details on usage of specific functions may be found in the demo `tutorial.ipynb`.

## Authors and Contact

* [corresponding author] Chunyin Siu (Alex), Center of Applied Mathematics, Cornell University, NY, USA (cs2323 [at] cornell.edu)

* Gennady Samorodnitsky (Alex), School of Operations Research and Information Engineering, Cornell University, NY, USA (gs18 [at] cornell.edu)

* Christina Lee Yu, School of Operations Research and Information Engineering, Cornell University, NY, USA (cleeyu [at] cornell.edu)

* Andrey Yao, Department of Mathematics, Cornell University, NY, USA (awy32 [at] cornell.edu)

## History

220409 Repo made public.