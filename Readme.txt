##### Readme File #####


## Folders ##

Dataset_Matrices -> Contains the pairwise distance matrices of the case study: lower bound "D_I_low_...", upper bound "D_I_up_...", sup norm between functions to check stability "Norm_...", and bottleneck distances between persistence diagrams "PD_..."

Trees -> Contains the files needed to work with merge trees/dendrograms. 
		 "Trees_OPT.py" contains the python class used to represent merge trees/dendrograms.
		 "Interleaving_distance.py" contains the function <interleaving> which computes d_I between two such objects. 

		 "Utils_dendrograms_OPT.py" contains the functions needed to obtain dendrograms from point clouds or (possibly multivariate) functions.
		 "Utils_OPT.py" contains other auxiliary functions.

		 
## Files ##
		 
The following jupyter notebooks are contained in the main folder: 

1) "Example_Notebook.ipynb" contains an easy example in which the process of obtaining merge trees from functions and computing upper and lower bounds is presented;

2) "Comparison_with_Curry_et_al.ipynb" runs the comparison between the upper bound contained in the paper "A Graph Matching Formulation ..." And the upper bound presented by the paper "Decorated Merge Trees ...", by Curry et al.;

3) "Runtimes_Simulation.ipynb" investigates the runtimes for computing our upper bound as the number of leaves in the trees grows;

4) "Compute_Distances_Datasets.pynb" computes the matrices which are stored in the folder "Dataset_Matrices", related to several functional data analysis benchmark data sets;

5) "Analize_Datasets.ipynb" builds on the results of the previous notebook, considering some classification and regression case studies the the aforementioned functional data sets; the notebook "Analize_Beef_Dataset.ipynb" contains the analysis of the last data sets, which was implemented differently due to the way in which the data set is loaded.


The file "requirements.txt" cointains the environment requirements needed to run the code.


## Actions Required ##

A linear integer solver is needed to run the code. The solver is specified at the lines 566-570 of "Interleaving_distance.py.py".

There are two ways to specify the solver:
1) choose a solver from the pyomo library e.g. "pyo.SolverFactory('glpk')"
2) insert the address of the binaries of a solver available on the machine e.g. "pyo.SolverFactory('cplex', executable='address')"

