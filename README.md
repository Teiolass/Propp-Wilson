This repo is for the code written for the Computational Lab at University of
Pisa.
This is a work in progress, the main topic will be the implementation of the
Propp-Wilson algorithm for the study of the Ising model.

Keywords: Propp-Wilson, Markov chain, MCMC, Coupling from the past, Ising model


A short description of the scripts:

- `configuration_cluster.py`: Uses Propp-Wilson with clusters to plot a
  configuration of the Ising model at given N and beta
- `cluster_times.py`: plot on a log scale the number of step needed for
  coalescence using cluster.
- `cluster.py`: Uses clusters for the Ising model on a torus grid.
- `graph_cluster.py`: Failed attempt to use the hdtgraph package to speedup the
  computation with the connected components of the graph.
- `ising_mag.py`: uses propp-wilson without clusters, plots the magnetization as
  a function of beta.
- `grid_clustered.py`: same as `ising_mag`, but uses clusters.
- `pw_times.py`: as as `cluster_times`, but without clusters.
- `ssh_download.sh`: small utily script for downloading results from a server.
- `plotter.py`: plots data from csv file
- `multi_ising_mag.py`: same as `ising_mag`, but can run on multiple processors
  at the same time.
  
**Warning:**: I haven't reverified the next ones! They may not work.

- `vanilla_mcmc.py`: here I use a simple Monte Carlo method with Markov Chains
  for generating a target distribution over a finite set of states.
- `simple_pw.py`: as the name says, it is a Propp Wilson that runs all the N
  chains, with the same distributions of `vanilla_mcmc`.
- `false_pw.py`: same as `simple_pw`, but this one regenerates all the random
  values. It shouldn't work.
- `monotone_pw.py`: implements the sandwiching method with a linear chain.
- `monotone_legth_pw.py`:  same as above, but plots the numbers of steps needed
  for convergence.
- `simple_ising.py`: it is a simple ising model, with the pw algorithm. Prints
  the log of the number of steps needed for coalescence and the ratio between
  the number of +1 and the total number of nodes.
- `ising_depth.py`: this one runs multiple ising simulations and plot the log of
  the steps needed for coalescence as a function of beta. It works more or less
  for N<40 and beta<0.3.
- `ising_frac.py`: same as above, but plots the fraction of +1 as a function of
  beta.
